import world
import torch
from dataloader import BasicDataset
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigs
from scipy.sparse.linalg import eigsh
import torch.nn.functional as F

import metrics
import helpers
import multiprocessing

import torch.nn as nn




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
        return (torch.as_tensor(user_ratio, dtype=torch.float32, device=self.device),
                torch.as_tensor(item_ratio, dtype=torch.float32, device=self.device))

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

    def fit_dan(self, interactions=None, model="RLAE_DAN", reg_p=20.0,
                alpha=0.0, beta=0.0, drop_p=0.6, xi=0.1,
                alpha_i=None, lambda_i=None,
                alpha_u=None, lambda_u=None):
        """Fit DAN using the SGFCF-style user/item normalization.

        When all four normalization arguments are supplied, the input is
        transformed as ``D_U @ R @ D_I`` where
        ``D_U=(d_U+lambda_u)^(-alpha_u)`` and
        ``D_I=(d_I+lambda_i)^(-alpha_i)``.  Leaving them unset preserves the
        previous behavior and fits DAN on the supplied interaction matrix.
        """
        if model not in {"LAE_DAN", "EASE_DAN", "RLAE_DAN"}:
            raise ValueError(f"Unknown DAN model: {model}")
        X = (self.R if interactions is None else interactions).astype(np.float64).tocsr()
        if all(value is not None for value in
               (alpha_i, lambda_i, alpha_u, lambda_u)):
            user_degree = np.asarray(X.sum(axis=1)).ravel()
            item_degree = np.asarray(X.sum(axis=0)).ravel()
            with np.errstate(divide="ignore", invalid="ignore"):
                user_scale = np.power(
                    user_degree + float(lambda_u), -float(alpha_u)
                )
                item_scale = np.power(
                    item_degree + float(lambda_i), -float(alpha_i)
                )
            user_scale[~np.isfinite(user_scale)] = 0.0
            item_scale[~np.isfinite(item_scale)] = 0.0
            X = sp.diags(user_scale).dot(X).dot(sp.diags(item_scale)).tocsr()
        item_counts = np.asarray(X.sum(axis=0)).ravel()
        user_counts = np.asarray(X.sum(axis=1)).ravel()
        safe_users = np.maximum(user_counts, 1.0)
        regularization = float(reg_p) + float(drop_p) / (1.0 - float(drop_p)) * item_counts
        gram = X.multiply(np.power(safe_users, -float(beta))[:, None]).T.dot(X).toarray()
        diagonal = np.diag_indices(gram.shape[0])
        gram[diagonal] += regularization
        precision = np.linalg.inv(gram)
        diag_precision = np.diag(precision)
        if model == "EASE_DAN":
            base = np.eye(len(diag_precision)) - precision / diag_precision
        elif model == "LAE_DAN":
            base = np.eye(len(diag_precision)) - precision * regularization
        else:
            condition = (1.0 - regularization * diag_precision) > float(xi)
            lagrangian = ((1.0 - float(xi)) / diag_precision - regularization) * condition
            base = np.eye(len(diag_precision)) - precision * (regularization + lagrangian)
        safe_items = np.maximum(item_counts, 1.0)
        weights = base * np.power(safe_items, float(alpha)).reshape(-1, 1)
        weights *= np.power(safe_items, -float(alpha))
        weights[diagonal] = 0.0
        return {"model": model, "weights": weights, "interactions": X,
                "cold_items": item_counts == 0,
                "params": {"reg_p": float(reg_p), "alpha": float(alpha),
                           "beta": float(beta), "drop_p": float(drop_p), "xi": float(xi)}}



    def build_combined_X(self, beta_1=1.0, beta_2=1.0, gamma=1.0,
                         dan_coefficient=1.0, dan_params=None,
                         alpha_i=0.2, lambda_i=1.0,
                         alpha_u=0.2, lambda_u=1.0):
        """Build a dense user-item score matrix for SGFCF+DAN testing.

        The SGFCF graph-filter scores and DAN autoencoder scores are computed
        with this class's component methods, normalized row-wise, combined,
        and passed through a sigmoid.  The returned ``X`` has shape
        ``(n_users, m_items)`` and can be used directly with
        :meth:`Test_RatingMatrix`.
        """
        # SGFCF graph filtering on the normalized bipartite adjacency.
        user_degree = np.asarray(self.R.sum(axis=1)).ravel()
        item_degree = np.asarray(self.R.sum(axis=0)).ravel()
        with np.errstate(divide="ignore", invalid="ignore"):
            user_scale = np.power(user_degree + float(lambda_u), -float(alpha_u))
            item_scale = np.power(item_degree + float(lambda_i), -float(alpha_i))
        user_scale[~np.isfinite(user_scale)] = 0.0
        item_scale[~np.isfinite(item_scale)] = 0.0
        normalized_interaction = sp.diags(user_scale).dot(
            self.R
        ).dot(sp.diags(item_scale)).tocsr()
        normalized_adj = sp.bmat([
            [None, normalized_interaction],
            [normalized_interaction.T, None],
        ], format="csr")
        eigenvectors, eigenvalues = self.Eigendecomposition(normalized_adj)
        user_homophily, item_homophily = self.compute_homophily()
        homophily = torch.cat((user_homophily, item_homophily))
        response = self.individual_weight(
            torch.clamp(torch.abs(eigenvalues), min=1e-12),
            homophily,
            beta_1=beta_1,
            beta_2=beta_2,
        )
        filtered = eigenvectors * response
        graph_scores = filtered[:self.n_users] @ filtered[self.n_users:].t()
        graph_sum = graph_scores.sum(dim=1, keepdim=True)
        graph_sum[graph_sum == 0] = 1.0
        graph_scores = graph_scores / graph_sum
        # The convolution term is controlled by ``gamma``.  Avoid building
        # the large dense user-item convolution when its coefficient is zero.
        gamma_value = float(gamma)
        convolution = None
        if gamma_value != 0.0:
            normalized_interactions = normalized_adj[:self.n_users, self.n_users:]
            convolution = normalized_interactions.dot(
                normalized_interactions.T
            ).dot(normalized_interactions).toarray()
            convolution = torch.as_tensor(convolution, dtype=torch.float32,
                                           device=self.device)
            convolution_sum = convolution.sum(dim=1, keepdim=True)
            convolution_sum[convolution_sum == 0] = 1.0
            convolution = convolution / convolution_sum

        # DAN autoencoder scores from the same training interaction matrix.
        dan_options = dict(dan_params or {})
        dan_options.pop("alpha", None)
        dan_options.pop("beta", None)
        dan_state = self.fit_dan(
            interactions=self.R,
            alpha=0.0,
            beta=0.0,
            alpha_i=alpha_i,
            lambda_i=lambda_i,
            alpha_u=alpha_u,
            lambda_u=lambda_u,
            **dan_options,
        )
        # Score DAN with the same normalized interaction matrix used to fit
        # the model, rather than reverting to the raw interaction matrix.
        dan_scores = np.asarray(
            normalized_interaction.dot(dan_state["weights"]), dtype=np.float32
        )
        dan_scores[:, dan_state["cold_items"]] = -np.inf
        dan_scores = torch.as_tensor(dan_scores, dtype=torch.float32,
                                     device=self.device)
        finite_dan = torch.isfinite(dan_scores)
        dan_for_sum = torch.where(finite_dan, dan_scores, torch.zeros_like(dan_scores))
        dan_sum = dan_for_sum.sum(dim=1, keepdim=True)
        dan_sum[dan_sum == 0] = 1.0
        dan_for_sum = dan_for_sum / dan_sum

        if convolution is None:
            X = torch.sigmoid(graph_scores
                              + float(dan_coefficient) * dan_for_sum)
        else:
            X = torch.sigmoid(graph_scores + gamma_value * convolution
                              + float(dan_coefficient) * dan_for_sum)
        cold_items = np.asarray(self.R.sum(axis=0)).ravel() == 0
        X[:, torch.as_tensor(cold_items, device=self.device)] = -(1 << 10)
        return X



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
                batch_users_gpu = torch.Tensor(batch_users).long()
                batch_users_gpu = batch_users_gpu.to(world.device)
                rating = X[batch_users_gpu]
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
