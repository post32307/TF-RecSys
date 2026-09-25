import world
import math
import torch
from dataloader import BasicDataset
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigs
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import svds
import torch.nn.functional as F

import metrics
import helpers
import multiprocessing

import torch.nn as nn




class LinearScoreMatrix:
    """Combine stored score matrices only for the users being evaluated."""

    def __init__(self, first, second, weight):
        if first.shape != second.shape:
            raise ValueError("Ensemble score matrices must have the same shape.")
        self.first, self.second = first, second
        self.weight = float(weight)
        self.shape = first.shape

    def get_rows(self, users, device):
        # Index on each source's device before transferring the small batch.
        first = self.first[torch.as_tensor(users, device=self.first.device)].to(device)
        second = self.second[torch.as_tensor(users, device=self.second.device)].to(device)
        return first * self.weight + second * (1.0 - self.weight)


class LowRankEASEUScoreMatrix:
    """Compute User-based EASE low-rank approximate scores on the fly for small batches."""

    def __init__(self, X, k=500, lambda_=50.0):
        self.X = X
        self.shape = X.shape
        import numpy as np
        from scipy.sparse.linalg import svds
        
        k = min(k, min(X.shape) - 1)
        
        U, S, Vt = svds(X.astype(np.float32), k=k)
        
        Lambda = S ** 2
        self.D = Lambda / (Lambda + float(lambda_))
        self.U = U
        self.lambda_ = float(lambda_)
        
        self.z = np.sum((U ** 2) * self.D, axis=1)
        self.Ut_X = self.U.T @ self.X

    def get_rows(self, users, device):
        import torch
        U_batch = self.U[users]
        z_batch = self.z[users]
        
        U_D_batch = U_batch * self.D
        pred_batch = U_D_batch @ self.Ut_X
        
        X_batch = self.X[users].toarray()
        
        denom = 1.0 - z_batch[:, None]
        denom[denom == 0] = 1e-9
        pred_batch = (pred_batch - z_batch[:, None] * X_batch) / denom
        
        return torch.as_tensor(pred_batch, dtype=torch.float32, device=device)


class Ensemble3WayScoreMatrix:
    """Combine S_SGFCF, S_EASE_I, S_EASE_U."""

    def __init__(self, sgfcf, ease_i, ease_u, gamma_i, gamma_u):
        self.sgfcf = sgfcf
        self.ease_i = ease_i
        self.ease_u = ease_u
        self.gamma_i = float(gamma_i)
        self.gamma_u = float(gamma_u)
        self.gamma_sgfcf = 1.0 - self.gamma_i - self.gamma_u
        self.shape = sgfcf.shape

    def get_rows(self, users, device):
        import torch
        def _get(mat):
            if hasattr(mat, "get_rows"):
                return mat.get_rows(users, device).to(device)
            else:
                indices = torch.as_tensor(users, dtype=torch.long, device=mat.device)
                return mat[indices].to(device)
                
        s = _get(self.sgfcf)
        ei = _get(self.ease_i)
        eu = _get(self.ease_u)
        
        return s * self.gamma_sgfcf + ei * self.gamma_i + eu * self.gamma_u


class TFmodel:

    def __init__(self, config: dict, dataset: BasicDataset):
        self.config = config
        self.dataset = dataset
        self.n_users = dataset.n_users
        self.m_items = dataset.m_items
        self.device = torch.device(config['device'])   
        self.k = config['k']
        self.Sigmoid = F.sigmoid

        self.R = self.dataset.UserItemNet.tocsr() #CSR
        self.A = sp.bmat([[None, self.R], [self.R.T, None]], format='csr') #CSR
        rowsum = np.array(self.A.sum(axis=1)).flatten()
        rowsum[rowsum == 0] = 1e-9
        self.D = sp.diags(rowsum) #CSR
        self.D_inv = sp.diags(1. / rowsum) #CSR
    


    # Model Functions

    def S_EASE_I(self, lambda_=50.0, cache=None):
        X = self.R
        G_I = X.T.dot(X).toarray()
        diagIndices = np.diag_indices(G_I.shape[0])
        G_I[diagIndices] += lambda_
        P_I = np.linalg.inv(G_I)
        B_I = P_I / (-np.diag(P_I))
        B_I[diagIndices] = 0

        pred = X.dot(B_I)
        return torch.as_tensor(pred, dtype=torch.float32, device=self.device)

    def S_EASE_U(self, lambda_=50.0, cache=None):
        return LowRankEASEUScoreMatrix(self.R, k=self.k, lambda_=lambda_)

    def S_SGFCF(self, beta_1=1.0, beta_2=1.0, alpha=0.0,
                eps=0.5, gamma=1.0, k=None, cache=None):
        """Return the SGFCF user-item scoring matrix."""
        if k is not None:
            self.k = int(k)
        degrees_u = np.asarray(self.R.sum(axis=1)).ravel()
        degrees_i = np.asarray(self.R.sum(axis=0)).ravel()
        # Zero-degree users/items can occur after the train/validation split.
        # In particular, alpha=0 with eps>0 would otherwise evaluate
        # 0**(-eps) and inject inf values into the normalized matrix.
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            user_scale = np.power(degrees_u + float(alpha), -float(eps))
            item_scale = np.power(degrees_i + float(alpha), -float(eps))
        user_scale[~np.isfinite(user_scale)] = 0.0
        item_scale[~np.isfinite(item_scale)] = 0.0
        norm_cache = None if cache is None else cache.setdefault("normalized", {})
        # Validation and test loaders may infer slightly different dimensions
        # when an ID appears only in one split.  Namespace cache entries by
        # model instance so matrices with different shapes are never reused.
        model_key = id(self)
        norm_key = (model_key, float(alpha), float(eps))
        if norm_cache is not None and norm_key in norm_cache:
            normalized, adjacency = norm_cache[norm_key]
        else:
            normalized = sp.diags(user_scale).dot(self.R).dot(sp.diags(item_scale)).tocsr()
            adjacency = sp.bmat([[None, normalized], [normalized.T, None]], format="csr")
            if norm_cache is not None:
                limit = int(cache.get("max_normalized_entries", 1))
                while len(norm_cache) >= limit:
                    norm_cache.pop(next(iter(norm_cache)))
                norm_cache[norm_key] = (normalized, adjacency)
        eigen_cache = None if cache is None else cache.setdefault("eigen", {})
        key = (model_key, int(self.k), float(alpha), float(eps))
        if eigen_cache is not None and key in eigen_cache:
            vectors_cpu, values_cpu = eigen_cache[key]
            vectors = vectors_cpu.to(self.device)
            values = values_cpu.to(self.device)
        else:
            vectors, values = self.Eigendecomposition(adjacency)
            if eigen_cache is not None:
                limit = int(cache.get("max_eigen_entries", 1))
                while len(eigen_cache) >= limit:
                    eigen_cache.pop(next(iter(eigen_cache)))
                # Keep the reusable eigenspace in host memory.  Only this
                # candidate's copy is moved to the accelerator below.
                eigen_cache[key] = (vectors.detach().cpu(), values.detach().cpu())
        hu_cpu, hi_cpu = self.compute_homophily()
        hu = hu_cpu.to(self.device)
        hi = hi_cpu.to(self.device)
        response = self.individual_weight(torch.abs(values).clamp_min(1e-12),
                                           torch.cat((hu, hi)), beta_1, beta_2)
        weighted = vectors * response
        scores = weighted[:self.n_users] @ weighted[self.n_users:].t()
        # The eigenspace may remain in the cache, but these intermediate GPU
        # tensors are no longer needed after the graph score is formed.
        del weighted, response
        denominator = scores.sum(dim=1, keepdim=True)
        denominator[denominator == 0] = 1.0
        scores = scores / denominator
        if float(gamma) != 0.0:
            conv_cache = None if cache is None else cache.setdefault("convolution", {})
            if conv_cache is not None and norm_key in conv_cache:
                conv = conv_cache[norm_key].to(self.device)
            else:
                conv = normalized.dot(normalized.T).dot(normalized).toarray()
                conv = torch.as_tensor(conv, dtype=torch.float32, device=self.device)
                conv_denominator = conv.sum(dim=1, keepdim=True)
                conv_denominator[conv_denominator == 0] = 1.0
                conv = conv / conv_denominator
                if conv_cache is not None:
                    limit = int(cache.get("max_convolution_entries", 1))
                    while len(conv_cache) >= limit:
                        conv_cache.pop(next(iter(conv_cache)))
                    conv_cache[norm_key] = conv.detach().cpu()
            scores = scores + float(gamma) * conv
            # Do not delete the cached tensor; only release the local alias.
            if conv_cache is None:
                del conv
        if norm_cache is None:
            del normalized, adjacency
        return scores


    def _threshold_offdiag_graph(self, matrix):
        graph = matrix.tocsr(copy=True)
        graph.setdiag(0)
        graph.eliminate_zeros()
        graph.data[graph.data <= 1] = 0
        graph.eliminate_zeros()
        graph.data = np.ones(graph.nnz, dtype=np.int8)
        return graph

    def _induced_edges(self, graph, nodes, marker, stamp):
        if len(nodes) <= 1:
            return 0, stamp
        stamp += 1
        if stamp == np.iinfo(marker.dtype).max:
            marker.fill(0)
            stamp = 1
        marker[nodes] = stamp
        count = 0
        for node in nodes:
            neighbors = graph.indices[graph.indptr[node]:graph.indptr[node + 1]]
            count += np.count_nonzero(marker[neighbors] == stamp)
        return count, stamp

    def compute_homophily(self):
        """Compute SGFCF user/item homophily ratios."""
        cached = getattr(self, "_homophily_cache", None)
        if cached is not None:
            return cached
        R = self.R.astype(np.float32).tocsr()
        R_csc = R.tocsc()
        item_graph = self._threshold_offdiag_graph(R.T.dot(R))
        user_graph = self._threshold_offdiag_graph(R.dot(R.T))
        user_ratio = np.zeros(R.shape[0], dtype=np.float32)
        marker = np.zeros(R.shape[1], dtype=np.int32)
        stamp = 0
        for user in range(R.shape[0]):
            nodes = R[user].indices
            edges, stamp = self._induced_edges(item_graph, nodes, marker, stamp)
            if len(nodes) > 1:
                user_ratio[user] = edges / (len(nodes) * (len(nodes) - 1))
        item_ratio = np.zeros(R.shape[1], dtype=np.float32)
        marker = np.zeros(R.shape[0], dtype=np.int32)
        stamp = 0
        for item in range(R.shape[1]):
            nodes = R_csc[:, item].indices
            edges, stamp = self._induced_edges(user_graph, nodes, marker, stamp)
            if len(nodes) > 1:
                item_ratio[item] = edges / (len(nodes) * (len(nodes) - 1))
        # Homophily is reused across many filter candidates, so retain it on
        # the CPU and transfer only the current copy used by SGFCF.
        self._homophily_cache = (
            torch.as_tensor(user_ratio, dtype=torch.float32),
            torch.as_tensor(item_ratio, dtype=torch.float32),
        )
        return self._homophily_cache

    def individual_weight(self, value, homophily, beta_1=1.0, beta_2=1.0):
        """Compute the SGFCF adaptive eigenvalue response."""
        value = value if torch.is_tensor(value) else torch.as_tensor(value, dtype=torch.float32, device=self.device)
        homophily = homophily if torch.is_tensor(homophily) else torch.as_tensor(homophily, dtype=torch.float32, device=self.device)
        minimum, maximum = homophily.min(), homophily.max()
        if torch.isclose(minimum, maximum).item():
            exponent = torch.ones_like(homophily) * float(beta_1)
        else:
            exponent = ((float(beta_2) - float(beta_1)) * homophily / (maximum - minimum)
                        + (maximum * float(beta_1) - float(beta_2) * minimum)
                        / (maximum - minimum))
        return value.unsqueeze(0).pow(exponent.unsqueeze(1))
        

    def _normalize_scores(self, scores):
        """Convert a NumPy/SciPy score result to a safe row-normalized tensor."""
        if not torch.is_tensor(scores):
            scores = torch.as_tensor(np.asarray(scores), dtype=torch.float32,
                                     device=self.device)
        scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        denominator = scores.sum(dim=1, keepdim=True)
        denominator[denominator == 0] = 1.0
        return scores / denominator







    # Dataset Operation Functions

    def getNormAdj(self,A,D_inv,alpha1=0.2,alpha2=0.4):
        adj_norm = D_inv.power(alpha1) @ A @ D_inv.power(alpha2)
        return adj_norm



    # Embedding Functions

    def Eigendecomposition(self,X):
        if torch.is_tensor(X):
            X = X.cpu().numpy()
        Eigs,U = eigsh(X, k=self.k,which='LA')
        U = U.real
        Eigs = Eigs.real
        U = torch.tensor(U, dtype=torch.float32).to(self.device)
        Eigs = torch.tensor(Eigs, dtype=torch.float32).to(self.device)
        U = torch.nn.functional.normalize(U, p=2, dim=0)
        return U,Eigs
    
    def Eigendecomposition_Asym(self,X):
        if torch.is_tensor(X):
            X = X.cpu().numpy()
        
        Eigs_R,U_R = eigs(X, k=self.k,which='LR')
        U_R = np.asarray(U_R.real, dtype=np.float32)
        Eigs_R = np.asarray(Eigs_R.real, dtype=np.float32)
        U_R = torch.from_numpy(U_R).to(self.device)
        Eigs_R = torch.from_numpy(Eigs_R).to(self.device)
        U_R = torch.nn.functional.normalize(U_R, p=2, dim=0)

        _,U_L = eigs(X.T, k=self.k,which='LR')
        U_L = np.asarray(U_L.real, dtype=np.float32)
        U_L = torch.from_numpy(U_L).to(self.device)
        U_L = torch.nn.functional.normalize(U_L, p=2, dim=0)

        Eigs = Eigs_R
        
        return U_L,U_R,Eigs
    


    # Evaluation Functions

    def getUsersRating(self,X,users,apply_sigmoid=True):
        X = X.to(self.device)
        users = users.long()
        users_emb = X[:self.n_users,:]
        items_emb = X[self.n_users:,:]
        users_emb = users_emb[users,:]
        items_emb = items_emb
        scores = torch.matmul(users_emb, items_emb.t())
        return self.Sigmoid(scores) if apply_sigmoid else scores

    def test_one_batch(self,X_groundTrue):
        sorted_items = X_groundTrue[0].numpy()
        groundTrue = X_groundTrue[1]
        r = metrics.getLabel(groundTrue, sorted_items)
        pre, recall, ndcg = [], [], []
        for k in world.topks:
            ret = metrics.RecallPrecision_ATk(groundTrue, r, k)
            pre.append(ret['precision'])
            recall.append(ret['recall'])
            ndcg.append(metrics.NDCGatK_r(groundTrue,r,k))
        return {'recall':np.array(recall), 
                'precision':np.array(pre), 
                'ndcg':np.array(ndcg)}

    def Test(self, X, dataset, users=None, items=None, multicore=0):
        if X.shape == (self.n_users, self.m_items):
            return self.Test_RatingMatrix(X, dataset, users, items, multicore)
        u_batch_size = world.config['test_u_batch_size']
        testDict: dict = dataset.testDict
        max_K = max(world.topks)
        results = {'precision': np.zeros(len(world.topks)),
                'recall': np.zeros(len(world.topks)),
                'ndcg': np.zeros(len(world.topks))}
        # print(multicore)
        if multicore == 1:
            pool = multiprocessing.Pool(world.CORES)
        with torch.no_grad():
            if users is None:
                users = list(testDict.keys())
                
            if items is not None:
                items_set = set(items)
                valid_users = []
                for u in users:
                    if len(items_set.intersection(testDict[u])) > 0:
                        valid_users.append(u)
                users = valid_users
                
            if len(users) == 0:
                if multicore == 1:
                    pool.close()
                return results
                
            u_batch_size = min(u_batch_size, max(1, len(users) // 10))
            try:
                assert u_batch_size <= len(users) / 10
            except AssertionError:
                print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
            users_list = []
            rating_list = []
            groundTrue_list = []
            total_batch = (len(users) + u_batch_size - 1) // u_batch_size if len(users) > 0 else 0
            for batch_users in helpers.minibatch(users, batch_size=u_batch_size):
                allPos = dataset.getUserPosItems(batch_users)
                if items is not None:
                    groundTrue = [list(items_set.intersection(testDict[u])) for u in batch_users]
                else:
                    groundTrue = [testDict[u] for u in batch_users]
                batch_users_gpu = torch.Tensor(batch_users).long()
                batch_users_gpu = batch_users_gpu.to(world.device)
                rating = self.getUsersRating(X,batch_users_gpu)
                exclude_index = []
                exclude_items = []
                for range_i, items_pos in enumerate(allPos):
                    exclude_index.extend([range_i] * len(items_pos))
                    exclude_items.extend(items_pos)
                rating[exclude_index, exclude_items] = -(1<<10)
                
                
                _, rating_K = torch.topk(rating, k=max_K)
                rating = rating.cpu().numpy()
                del rating
                users_list.append(batch_users)
                rating_list.append(rating_K.cpu())
                groundTrue_list.append(groundTrue)
            assert total_batch == len(users_list)
            X_zip = zip(rating_list, groundTrue_list)
            if multicore == 1:
                pre_results = pool.map(self.test_one_batch, X_zip)
            else:
                pre_results = []
                for x in X_zip:
                    pre_results.append(self.test_one_batch(x))
            scale = float(u_batch_size/len(users))
            for result in pre_results:
                results['recall'] += result['recall']
                results['precision'] += result['precision']
                results['ndcg'] += result['ndcg']
            results['recall'] /= float(len(users))
            results['precision'] /= float(len(users))
            results['ndcg'] /= float(len(users))
            
            if multicore == 1:
                pool.close()
            return results



    def Test_RatingMatrix(self, X, dataset, users=None, items=None, multicore=0):
        u_batch_size = world.config['test_u_batch_size']
        if hasattr(X, "get_rows") or (torch.is_tensor(X) and X.device.type == "cpu"):
            u_batch_size = min(u_batch_size, 256)
        testDict: dict = dataset.testDict
        max_K = max(world.topks)
        results = {'precision': np.zeros(len(world.topks)),
                'recall': np.zeros(len(world.topks)),
                'ndcg': np.zeros(len(world.topks))}
        if multicore == 1:
            pool = multiprocessing.Pool(world.CORES)
        with torch.no_grad():
            if users is None:
                users = list(testDict.keys())
                
            if items is not None:
                items_set = set(items)
                valid_users = []
                for u in users:
                    if len(items_set.intersection(testDict[u])) > 0:
                        valid_users.append(u)
                users = valid_users
                
            if len(users) == 0:
                if multicore == 1:
                    pool.close()
                return results

            u_batch_size = min(u_batch_size, max(1, len(users) // 10))
            try:
                assert u_batch_size <= len(users) / 10
            except AssertionError:
                print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
            users_list = []
            rating_list = []
            groundTrue_list = []
            total_batch = (len(users) + u_batch_size - 1) // u_batch_size if len(users) > 0 else 0
            for batch_users in helpers.minibatch(users, batch_size=u_batch_size):
                allPos = dataset.getUserPosItems(batch_users)
                if items is not None:
                    groundTrue = [list(items_set.intersection(testDict[u])) for u in batch_users]
                else:
                    groundTrue = [testDict[u] for u in batch_users]
                if hasattr(X, "get_rows"):
                    rating = X.get_rows(batch_users, self.device)
                else:
                    indices = torch.as_tensor(batch_users, dtype=torch.long, device=X.device)
                    rating = X[indices].to(self.device)
                exclude_index = []
                exclude_items = []
                for range_i, items_pos in enumerate(allPos):
                    exclude_index.extend([range_i] * len(items_pos))
                    exclude_items.extend(items_pos)
                rating[exclude_index, exclude_items] = -(1<<10)
                
                
                _, rating_K = torch.topk(rating, k=max_K)
                del rating
                users_list.append(batch_users)
                rating_list.append(rating_K.cpu())
                groundTrue_list.append(groundTrue)
            assert total_batch == len(users_list)
            X_zip = zip(rating_list, groundTrue_list)
            if multicore == 1:
                pre_results = pool.map(self.test_one_batch, X_zip)
            else:
                pre_results = []
                for x in X_zip:
                    pre_results.append(self.test_one_batch(x))
            for result in pre_results:
                results['recall'] += result['recall']
                results['precision'] += result['precision']
                results['ndcg'] += result['ndcg']
            results['recall'] /= float(len(users))
            results['precision'] /= float(len(users))
            results['ndcg'] /= float(len(users))
            
            if multicore == 1:
                pool.close()
            return results
