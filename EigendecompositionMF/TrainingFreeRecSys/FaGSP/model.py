import torch
import metric
import numpy as np
import scipy.sparse as sp
from sklearn.utils.extmath import randomized_svd


class FaGSP(object):
    def __init__(self, init_adj_mat, init_rat_mat,
                 pri_factor1=256, pri_factor2=256, alpha1=0.3, alpha2=0.3, order1=2, order2=2,
                 user_interacted_items_dict=None):
        self.adj_mat = init_adj_mat
        self.rat_mat = init_rat_mat
        self.pri_factor1 = pri_factor1
        self.pri_factor2 = pri_factor2
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.order1 = order1
        self.order2 = order2

        self.user_interacted_items_dict = user_interacted_items_dict
        self.topks = [10, 20]
        self.random_state = 42

    def _truncated_svd(self, mat, k, label):
        k = min(k, min(mat.shape) - 1)
        if k <= 0:
            raise ValueError(f"Cannot compute SVD with k={k} for {label} shape={mat.shape}")
        print(f"FaGSP: sparse SVD for {label}, shape={mat.shape}, k={k}", flush=True)
        from scipy.sparse.linalg import svds
        try:
            _, _, vt = svds(mat, k=k)
        except Exception:
            print(f"FaGSP: sparse SVD failed for {label}; falling back to randomized SVD", flush=True)
            _, _, vt = randomized_svd(mat, n_components=k, random_state=self.random_state)
        return vt

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
        d_mat_i = d_mat
        d_inv = 1.0 / d_inv
        d_inv[np.isinf(d_inv)] = 0.
        d_inv[np.isnan(d_inv)] = 0.
        d_mat_i_inv = sp.diags(d_inv)
        norm_adj = norm_adj.dot(d_mat)
        norm_adj = norm_adj.tocsc()
        return norm_adj, d_mat_i, d_mat_i_inv

    def matpow(self, mat, order):
        R = mat
        if order == 1:
            return R
        for ord in range(2, order+1):
            R = R.T @ mat
        return R

    def get_user_rating(self, quan):
        rat_mat = self.rat_mat
        batch_test = sp.csr_matrix(rat_mat)

        Y = batch_test.copy()
        for _ in range(self.order1):
            Y = Y - Y @ self.RTR1
        P11 = batch_test - Y

        Y = batch_test.copy()
        for _ in range(self.order2):
            Y = Y - self.RTR2 @ Y
        P12 = batch_test - Y

        P = P11 + P12
        if sp.issparse(P):
            P = P.toarray()
        else:
            P = np.array(P)

        vt2 = self.vt2[-self.pri_factor2:]
        temp = batch_test @ self.d_mat_i @ vt2.T
        P30 = temp @ (vt2 @ self.d_mat_i_inv)

        quan_val = np.quantile(P30, q=quan, axis=0, keepdims=True)
        P30[P30 > quan_val] = 1.0
        P30[P30 <= quan_val] = 0.0

        dense_batch_test = batch_test.toarray()
        P30[dense_batch_test < 1] = 0.0

        P3 = dense_batch_test + self.alpha2 * P30
        P3 = sp.csr_matrix(P3)

        print("FaGSP: normalizing P3", flush=True)
        norm_P3, d_mat_i_P3, d_mat_i_inv_P3 = self.normalize_adj_mat(P3)
        vt1 = self._truncated_svd(norm_P3, self.pri_factor1, "P3")

        print("FaGSP: projecting P3", flush=True)
        temp2 = P3 @ d_mat_i_P3 @ vt1.T
        P2 = temp2 @ (vt1 @ d_mat_i_inv_P3)
        P += self.alpha1 * P2

        return P

    def train(self, quan):
        print("FaGSP: normalizing adjacency", flush=True)
        self.norm_adj, self.d_mat_i, self.d_mat_i_inv = self.normalize_adj_mat(self.adj_mat)
        self.vt2 = self._truncated_svd(self.norm_adj, self.pri_factor2, "adjacency")

        print("FaGSP: building propagation operators", flush=True)
        self.RTR1 = self.norm_adj.T @ self.norm_adj
        self.RTR2 = self.norm_adj @ self.norm_adj.T

        print("FaGSP: computing user ratings", flush=True)
        ratings = self.get_user_rating(quan)
        print("FaGSP: finished user ratings", flush=True)
        return ratings

    def eval_test(self, user_items_dict, ratings):
        users = list(user_items_dict.keys())
        user_prediction_items_list = []
        user_truth_items_list = []
        for user in users:
            rating = ratings[user, :]
            rating = torch.from_numpy(rating).view(1, -1)
            user_interacted_items = list(self.user_interacted_items_dict[user])
            rating[0, user_interacted_items] = -999999.0
            ranked_items = torch.topk(rating, k=max(self.topks))[1].numpy()[0]
            user_prediction_items_list.append(ranked_items)
            user_truth_items_list.append(user_items_dict[user])
        _, recalls, f1_scores, mrrs, ndcgs = metric.calculate_all(user_truth_items_list, user_prediction_items_list, self.topks)
        return recalls, f1_scores, mrrs, ndcgs
