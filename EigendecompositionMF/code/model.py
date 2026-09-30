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
        self.response = np.broadcast_to(self.D, self.U.shape).copy()
        self.z = np.sum((U ** 2) * self.response, axis=1)
        self.Ut_X = self.U.T @ self.X

    def set_individual_response(self, response):
        """Set a user-by-component IGF response after the SVD is fitted."""
        response = np.asarray(response, dtype=self.U.dtype)
        if response.shape != self.U.shape:
            raise ValueError(
                f"EASE-U response must have shape {self.U.shape}, got {response.shape}."
            )
        self.response = response
        self.z = np.sum((self.U ** 2) * self.response, axis=1)

    def get_rows(self, users, device):
        import torch
        U_batch = self.U[users]
        z_batch = self.z[users]
        
        U_D_batch = U_batch * self.response[users]
        pred_batch = U_D_batch @ self.Ut_X
        
        X_batch = self.X[users].toarray()
        
        denom = 1.0 - z_batch[:, None]
        denom[denom == 0] = 1e-9
        pred_batch = (pred_batch - z_batch[:, None] * X_batch) / denom
        
        return torch.sigmoid(torch.as_tensor(pred_batch, dtype=torch.float32, device=device))


class OriginalSGFCFScoreMatrix:
    """Original SGFCF scorer evaluated in user batches.

    This is the same rectangular-SVD construction as
    ``TrainingFreeRecSys/SGFCF/sgfcf.py``. Keeping factors and the normalized
    interaction matrix on CPU prevents a full user-item score matrix from
    occupying accelerator memory while preserving the original score formula.
    """

    observed_mask_value = -1000.0

    def __init__(self, user_factor, item_factor, normalized, gamma):
        self.user_factor = user_factor.detach().cpu()
        self.item_factor = item_factor.detach().cpu()
        self.normalized_csr = normalized.tocsr()
        self.normalized_csc = normalized.tocsc()
        self.gamma = float(gamma)
        self.shape = (self.user_factor.shape[0], self.item_factor.shape[0])

    def get_rows(self, users, device):
        index = torch.as_tensor(users, dtype=torch.long)
        user_factor = self.user_factor[index].to(device)
        item_factor = self.item_factor.to(device)
        rate = user_factor.matmul(item_factor.t())
        rate_sum = rate.sum(dim=1, keepdim=True)
        rate_sum[rate_sum == 0] = 1.0
        rate = rate / rate_sum
        del user_factor, item_factor, rate_sum

        if self.gamma != 0.0:
            # Reference sgfcf.py: R_tilde R_tilde^T R_tilde, normalized per
            # user, before it is multiplied by gamma and added to the score.
            batch = self.normalized_csr[users]
            conv = batch.dot(self.normalized_csc.T).dot(self.normalized_csr).toarray()
            conv = torch.as_tensor(conv, dtype=torch.float32, device=device)
            conv_sum = conv.sum(dim=1, keepdim=True)
            conv_sum[conv_sum == 0] = 1.0
            rate = rate + self.gamma * (conv / conv_sum)
            del conv, conv_sum
        return torch.sigmoid(rate)


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

    def S_EASE_I(self, lambda_=50.0, beta_1=0.9, beta_2=1.1, cache=None):
        """Return item-EASE scores using an SVD ridge solve, not ``inv``.

        For ``X = U Σ Vᵀ``, the EASE precision matrix is
        ``P = (XᵀX + λI)⁻¹``.  This implementation evaluates ``XP`` and
        ``diag(P)`` directly from the singular values, then applies EASE's
        zero-diagonal coefficient constraint without materialising ``P`` or
        calling a dense matrix inverse.
        """
        regularization = float(lambda_)
        if regularization <= 0.0:
            raise ValueError("S_EASE_I requires lambda_ > 0 for the ridge SVD solve.")

        model_key = id(self)
        svd_cache = None if cache is None else cache.setdefault("ease_i_svd", {})
        if svd_cache is not None and model_key in svd_cache:
            singular_values, right_vectors = svd_cache[model_key]
        else:
            # Economy SVD is sufficient: the omitted right-singular subspace
            # contributes the 1/lambda identity term below.
            dense_interactions = self.R.toarray().astype(np.float64, copy=False)
            _, singular_values, right_transpose = np.linalg.svd(
                dense_interactions, full_matrices=False
            )
            right_vectors = right_transpose.T
            del dense_interactions, right_transpose
            if svd_cache is not None:
                limit = int(cache.get("max_ease_i_svd_entries", 1))
                while len(svd_cache) >= limit:
                    svd_cache.pop(next(iter(svd_cache)))
                svd_cache[model_key] = (singular_values, right_vectors)

        inverse_lambda = 1.0 / regularization
        spectral_correction = (
            1.0 / (singular_values ** 2 + regularization) - inverse_lambda
        )
        # ``individual_weight`` is defined for non-negative filter responses.
        # Apply it to the correction magnitude, restore the original (negative)
        # ridge-correction sign, and obtain an item-by-component response.
        _, item_homophily = self.compute_homophily()
        correction_magnitude = torch.as_tensor(
            np.abs(spectral_correction), dtype=torch.float64
        )
        adaptive_correction = self.individual_weight(
            correction_magnitude, item_homophily, beta_1, beta_2
        ).cpu().numpy()
        adaptive_correction *= np.sign(spectral_correction)[None, :]

        # The response is item-specific, so both diag(P) and XP use the same
        # weighted V rows.  beta_1=beta_2=1 reproduces ordinary EASE-I.
        precision_diagonal = inverse_lambda + np.sum(
            (right_vectors ** 2) * adaptive_correction, axis=1
        )
        precision_diagonal = np.maximum(precision_diagonal, np.finfo(np.float64).eps)

        # XP = X P = X/lambda + (X V) [V * adaptive_correction]^T.
        XP = self.R.dot(right_vectors)
        XP = XP.dot((right_vectors * adaptive_correction).T)
        XP += self.R.toarray().astype(np.float64, copy=False) * inverse_lambda

        # -XP / diag(P) corresponds to X[-P/diag(P)].  The latter has a -1
        # diagonal, so adding X is exactly EASE's B_ii = 0 correction.
        pred = self.R.toarray().astype(np.float64, copy=False) - XP / precision_diagonal[None, :]
        return torch.sigmoid(torch.as_tensor(pred, dtype=torch.float32, device=self.device))

    def S_EASE_U(self, lambda_=50.0, beta_1=1.0, beta_2=1.0, cache=None):
        # Keep EASE-U independent of SGFCF's searched/mutable rank.  scipy
        # svds requires k < min(R.shape), hence the minus one cap.
        ease_u_k = max(1, min(128, min(self.R.shape) - 1))
        scorer = LowRankEASEUScoreMatrix(self.R, k=ease_u_k, lambda_=lambda_)

        # EASE-U filters user-space singular components, so use the user
        # homophily response (the EASE-I analogue uses item homophily).
        user_homophily, _ = self.compute_homophily()
        response = self.individual_weight(
            torch.as_tensor(scorer.D, dtype=torch.float64),
            user_homophily,
            beta_1,
            beta_2,
        ).cpu().numpy()
        scorer.set_individual_response(response)
        return scorer

    def S_SGFCF(self, beta_1=1.0, beta_2=1.0, alpha=0.0,
                eps=0.5, gamma=1.0, k=None, cache=None,
                implementation_version=2):
        """Return the original rectangular-SVD SGFCF scoring component."""
        if int(implementation_version) != 2:
            raise ValueError("SGFCF parameters must be revalidated for implementation 2.")
        if k is not None:
            self.k = int(k)
        degrees_u = np.asarray(self.R.sum(axis=1)).ravel()
        degrees_i = np.asarray(self.R.sum(axis=0)).ravel()
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            user_scale = np.power(degrees_u + float(alpha), -float(eps))
            item_scale = np.power(degrees_i + float(alpha), -float(eps))
        user_scale[~np.isfinite(user_scale)] = 0.0
        item_scale[~np.isfinite(item_scale)] = 0.0

        model_key = id(self)
        norm_key = (model_key, float(alpha), float(eps))
        norm_cache = None if cache is None else cache.setdefault("sgfcf_normalized", {})
        if norm_cache is not None and norm_key in norm_cache:
            normalized = norm_cache[norm_key]
        else:
            normalized = sp.diags(user_scale).dot(self.R).dot(sp.diags(item_scale)).tocsr()
            if norm_cache is not None:
                limit = int(cache.get("max_normalized_entries", 1))
                while len(norm_cache) >= limit:
                    norm_cache.pop(next(iter(norm_cache)))
                norm_cache[norm_key] = normalized

        # Original sgfcf.py computes k+200 singular vectors, sorts them, and
        # uses the leading k components in its graph-filter score.
        svd_rank = min(int(self.k) + 200, min(normalized.shape) - 1)
        if svd_rank < 1:
            raise ValueError("SGFCF requires at least two users and two items.")
        svd_key = (model_key, svd_rank, float(alpha), float(eps))
        svd_cache = None if cache is None else cache.setdefault("sgfcf_svd", {})
        if svd_cache is not None and svd_key in svd_cache:
            u_cpu, s_cpu, v_cpu = svd_cache[svd_key]
        else:
            if self.device.type == "cuda":
                cells = normalized.shape[0] * normalized.shape[1]
                if cells <= 60_000_000:
                    dense = torch.as_tensor(normalized.toarray(), dtype=torch.float64,
                                            device=self.device)
                    u, s, vh = torch.linalg.svd(dense, full_matrices=False)
                    u, s, v = u[:, :svd_rank], s[:svd_rank], vh[:svd_rank].t()
                    del dense, vh
                else:
                    coo = normalized.tocoo()
                    sparse = torch.sparse_coo_tensor(
                        torch.as_tensor(np.vstack((coo.row, coo.col)), dtype=torch.long,
                                        device=self.device),
                        torch.as_tensor(coo.data, dtype=torch.float32, device=self.device),
                        coo.shape, device=self.device).coalesce()
                    u, s, v = torch.svd_lowrank(sparse, q=svd_rank, niter=30)
                    del sparse
                order = torch.argsort(s, descending=True)
                u_cpu = u[:, order].float().cpu()
                s_cpu = s[order].float().cpu()
                v_cpu = v[:, order].float().cpu()
                del u, s, v, order
            else:
                u_np, s_np, vt_np = svds(normalized, k=svd_rank)
                order = np.argsort(s_np)[::-1]
                u_cpu = torch.as_tensor(u_np[:, order], dtype=torch.float32)
                s_cpu = torch.as_tensor(s_np[order], dtype=torch.float32)
                v_cpu = torch.as_tensor(vt_np[order, :].T, dtype=torch.float32)
            if svd_cache is not None:
                limit = int(cache.get("max_sgfcf_svd_entries", 1))
                while len(svd_cache) >= limit:
                    svd_cache.pop(next(iter(svd_cache)))
                svd_cache[svd_key] = (u_cpu, s_cpu, v_cpu)

        effective_k = min(int(self.k), s_cpu.numel())
        if effective_k < 1:
            raise ValueError("SGFCF did not obtain a usable singular component.")
        values = (s_cpu[:effective_k] / s_cpu[0].clamp_min(1e-12)).to(self.device)
        hu_cpu, hi_cpu = self.compute_homophily()
        user_response = self.individual_weight(values, hu_cpu.to(self.device), beta_1, beta_2)
        item_response = self.individual_weight(values, hi_cpu.to(self.device), beta_1, beta_2)
        user_factor = u_cpu[:, :effective_k].to(self.device) * user_response
        item_factor = v_cpu[:, :effective_k].to(self.device) * item_response
        result = OriginalSGFCFScoreMatrix(user_factor, item_factor, normalized, gamma)
        del values, user_response, item_response, user_factor, item_factor
        return result


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
                rating[exclude_index, exclude_items] = getattr(X, "observed_mask_value", -(1<<10))
                
                
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
                rating[exclude_index, exclude_items] = getattr(X, "observed_mask_value", -(1<<10))
                
                
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
