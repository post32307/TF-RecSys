import os
import torch
import metric
import numpy as np
import scipy.sparse as sp
from sklearn.mixture import GaussianMixture
from scipy.sparse.linalg import svds


class HiGSP(object):
    def __init__(self, init_adj_mat,
                 alpha1=0.5, alpha2=0.5, order1=2, order2=2, n_clusters=20, pri_factor=256,
                 user_interacted_items_dict=None, topks=[5, 10, 20]):
        self.adj_mat = init_adj_mat

        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.order1 = order1
        self.order2 = order2
        self.n_clusters = n_clusters
        self.pri_factor = pri_factor

        self.user_interacted_items_dict = user_interacted_items_dict
        self.topks = topks

    def normalize_adj_mat(self, adj_mat):
        adj_mat = sp.csr_matrix(adj_mat)
        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_inv[np.isnan(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        norm_adj = d_mat.dot(adj_mat)
        colsum = np.array(adj_mat.sum(axis=0))
        d_inv = np.power(colsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_inv[np.isnan(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        d_inv = 1.0 / d_inv
        d_inv[np.isinf(d_inv)] = 0.
        d_inv[np.isnan(d_inv)] = 0.
        norm_adj = norm_adj.dot(d_mat)
        return norm_adj

    def normalize_adj_mat_sp(self, adj_mat):
        adj_mat = sp.csr_matrix(adj_mat)
        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_inv[np.isnan(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        norm_adj = d_mat.dot(adj_mat)

        colsum = np.array(adj_mat.sum(axis=0))
        d_inv = np.power(colsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_inv[np.isnan(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        d_mat_i = d_mat
        d_inv = 1.0 / d_inv
        d_inv[np.isinf(d_inv)] = 0.
        d_inv[np.isnan(d_inv)] = 0.
        d_mat_i_inv = sp.diags(d_inv)
        norm_adj = norm_adj.dot(d_mat)
        return norm_adj, d_mat_i, d_mat_i_inv

    def bmatpow(self, mat, order):
        with torch.no_grad():
            R = torch.FloatTensor(mat)
            mat = torch.FloatTensor(mat)
            I = torch.FloatTensor(np.expand_dims(np.identity(mat.shape[1]), axis=0).repeat(self.n_clusters, axis=0))
            if order == 1:
                return R.detach().numpy()
            for ord in range(2, order+1):
                R = torch.bmm(R.transpose(1, 2), mat)
            R = R.detach().numpy()
        return R

    def matpow(self, mat, order):
        with torch.no_grad():
            R = torch.FloatTensor(mat)
            mat = torch.FloatTensor(mat)
            if order == 1:
                return R.detach().numpy()
            for ord in range(2, order+1):
                R = torch.matmul(R.transpose(0, 1), mat)
            R = R.detach().numpy()
        return R

    def construct_cluster_wise_filter(self, adj_mat):
        from sklearn.cluster import KMeans
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clustering = KMeans(n_clusters=self.n_clusters, random_state=42)
            cluster_labels = clustering.fit_predict(adj_mat)
        n_clusters = len(set(cluster_labels))

        local_A_tildes = []
        for i in range(n_clusters):
            mask = (cluster_labels == i)
            C_tilde = adj_mat.copy()
            if sp.issparse(C_tilde):
                C_tilde.data[np.in1d(C_tilde.nonzero()[0], np.where(~mask)[0])] = 0
                C_tilde.eliminate_zeros()
            else:
                C_tilde[~mask, :] = 0
                C_tilde = sp.csr_matrix(C_tilde)
                
            A_tilde = C_tilde.T @ C_tilde
            A_tilde = self.normalize_adj_mat(A_tilde)
            local_A_tildes.append(A_tilde)
            
        return local_A_tildes, cluster_labels

    def construct_global_aware_filter(self, adj_mat):
        norm_adj, self.d_mat_i, self.d_mat_i_inv = self.normalize_adj_mat_sp(adj_mat)
        norm_adj = norm_adj.tocsc()
        try:
            u, s, vt = svds(norm_adj, k=self.pri_factor)
        except Exception:
            from sklearn.utils.extmath import randomized_svd
            u, s, vt = randomized_svd(norm_adj, n_components=self.pri_factor, random_state=42)
        self.vt = vt

        R_tilde = self.normalize_adj_mat(adj_mat)
        P_tilde = R_tilde.T @ R_tilde
        return P_tilde

    def train(self):
        self.local_A_tildes, self.item_cluster_labels = self.construct_cluster_wise_filter(self.adj_mat)
        self.P_tilde = self.construct_global_aware_filter(self.adj_mat)
        return self

    def eval_test(self, user_items_dict, dummy_ratings=None):
        users = list(user_items_dict.keys())
        user_prediction_items_list = []
        user_truth_items_list = []
        
        batch_size = 500
        n_clusters = len(self.local_A_tildes)
        
        for start_idx in range(0, len(users), batch_size):
            end_idx = min(start_idx + batch_size, len(users))
            batch_users = users[start_idx:end_idx]
            
            batch_adj = self.adj_mat[batch_users, :]
            if sp.issparse(batch_adj):
                batch_adj = batch_adj.toarray()
            
            ratings = np.zeros_like(batch_adj)
            
            for i in range(n_clusters):
                batch_mask = (self.item_cluster_labels[batch_users] == i)
                C_i = batch_adj.copy()
                C_i[~batch_mask, :] = 0
                
                Y = C_i.copy()
                for _ in range(self.order1):
                    Y = Y - Y @ self.local_A_tildes[i]
                ratings += C_i - Y
                
            temp1 = batch_adj @ self.d_mat_i @ self.vt.T
            ratings += self.alpha1 * (temp1 @ (self.vt @ self.d_mat_i_inv))
            
            Y_glob = batch_adj.copy()
            for _ in range(self.order2):
                Y_glob = Y_glob - Y_glob @ self.P_tilde
            ratings += self.alpha2 * (batch_adj - Y_glob)
            
            for idx, user in enumerate(batch_users):
                rating = ratings[idx, :]
                rating = torch.from_numpy(rating).view(1, -1)
                user_interacted_items = list(self.user_interacted_items_dict[user])
                rating[0, user_interacted_items] = -999999.0
                ranked_items = torch.topk(rating, k=max(self.topks))[1].numpy()[0]
                user_prediction_items_list.append(ranked_items)
                user_truth_items_list.append(user_items_dict[user])
                
        precisions, recalls, f1_scores, mrrs, ndcgs = metric.calculate_all(user_truth_items_list,
                                                                                      user_prediction_items_list,
                                                                                      self.topks)
        return recalls, f1_scores, mrrs, ndcgs
