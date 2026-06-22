import torch
from torch.autograd import Variable
import argparse
import numpy as np
import pandas as pd
import time
import gc


parser = argparse.ArgumentParser(description='Argument parser for the algorithm.')

parser.add_argument('--dataset', type=str, default='yelp', help='Dataset name')
parser.add_argument('--density', type=str, default='dense', help='Density setting: dense or sparse')
parser.add_argument('--k', type=int, default=100, help='The number of required features')
parser.add_argument('--beta_1', type=float, default=1.0, help='coef for the filter')
parser.add_argument('--beta_2', type=float, default=1.0, help='coef for the filter')
parser.add_argument('--alpha', type=float, default=0.0, help='param for G^2N')
parser.add_argument('--eps', type=float, default=0.5, help='param for G^2N')
parser.add_argument('--gamma', type=float, default=1.0, help='weight for non-low frequency')
parser.add_argument('--seed', type=int, default=2020)

args = parser.parse_args()
 
dataset = args.dataset
setting = args.density
k = args.k
beta_1 = args.beta_1
beta_2 = args.beta_2
alpha = args.alpha
eps = args.eps
gamma=args.gamma


import os

def load_txt(filename):
    rows, cols = [], []
    mu, mi = 0, 0
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 0: continue
            u = int(parts[0])
            if dataset == "lastfm":
                i_idx = int(parts[1])
                rows.append(u)
                cols.append(i_idx)
                if u > mu: mu = u
                if i_idx > mi: mi = i_idx
            else:
                for i in parts[1:]:
                    i_idx = int(i)
                    rows.append(u)
                    cols.append(i_idx)
                    if u > mu: mu = u
                    if i_idx > mi: mi = i_idx
    return pd.DataFrame({'user': rows, 'item': cols}), mu, mi

data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", dataset))
train_file = "data1.txt" if dataset == "lastfm" else "train.txt"
test_file = "test1.txt" if dataset == "lastfm" else "test.txt"

df_train, mu1, mi1 = load_txt(os.path.join(data_dir, train_file))
df_test, mu2, mi2 = load_txt(os.path.join(data_dir, test_file))

user = max(mu1, mu2) + 1
item = max(mi1, mi2) + 1


#hyperparamter setting for [beta_1, beta_2]
#Citeulike
#80%: [0.9,1.1]
#20%: [4.3,5.1]
#Pinterest
#80%: almost does not bring improvemnt
#20%: [2.0,2.4]
#Yelp
#80%: [1.0,1.3]
#20: [2.0,5.0]
#Gowalla
#80%: [0.3,2.5]
#20%: [2.5,6.5]




import scipy.sparse as sp

rows = df_train['user'].values
cols = df_train['item'].values
data = np.ones(len(rows), dtype=np.float32)
freq_matrix_csr = sp.csr_matrix((data, (rows, cols)), shape=(user, item))

test_data = [[] for i in range(user)]
for row in df_test.itertuples():
    test_data[row[1]].append(row[2])

def individual_weight(value, homo_ratio):
    y_min, y_max = beta_1, beta_2
    x_min, x_max = homo_ratio.min(), homo_ratio.max()
    if x_max == x_min:
        homo_weight = torch.ones_like(homo_ratio) * y_min
    else:
        homo_weight = (y_max - y_min) / (x_max - x_min) * homo_ratio + (x_max * y_min - y_max * x_min) / (x_max - x_min)
    return value.unsqueeze(0).pow(homo_weight.unsqueeze(1))

train_data = [[] for i in range(user)]
train_data_item = [[] for i in range(item)]
for row in df_train.itertuples():
    train_data[row[1]].append(row[2])
    train_data_item[row[2]].append(row[1])

print("Computing homo ratios...")
ItemCo = freq_matrix_csr.T.dot(freq_matrix_csr)
UserCo = freq_matrix_csr.dot(freq_matrix_csr.T)

homo_ratio_user = np.zeros(user)
for u in range(user):
    items_u = train_data[u]
    size = len(items_u)
    if size > 1:
        sub_mat = ItemCo[items_u, :][:, items_u].toarray()
        sub_mat -= 1
        np.fill_diagonal(sub_mat, 0)
        ratio = np.count_nonzero(sub_mat) / (size * (size - 1))
        homo_ratio_user[u] = ratio

homo_ratio_item = np.zeros(item)
for i in range(item):
    users_i = train_data_item[i]
    size = len(users_i)
    if size > 1:
        sub_mat = UserCo[users_i, :][:, users_i].toarray()
        sub_mat -= 1
        np.fill_diagonal(sub_mat, 0)
        ratio = np.count_nonzero(sub_mat) / (size * (size - 1))
        homo_ratio_item[i] = ratio

homo_ratio_user = torch.FloatTensor(homo_ratio_user).cuda()
homo_ratio_item = torch.FloatTensor(homo_ratio_item).cuda()

D_u = np.array(freq_matrix_csr.sum(axis=1)).flatten()
D_u = 1.0 / np.power(D_u + alpha, eps)
D_u[np.isinf(D_u)] = 0

D_i = np.array(freq_matrix_csr.sum(axis=0)).flatten()
D_i = 1.0 / np.power(D_i + alpha, eps)
D_i[np.isinf(D_i)] = 0

print("Computing SVD...")
norm_freq_matrix = sp.diags(D_u).dot(freq_matrix_csr).dot(sp.diags(D_i))

from scipy.sparse.linalg import svds
try:
    u_mat, s_mat, vt_mat = svds(norm_freq_matrix, k=min(k+200, min(user, item)-1))
except Exception:
    from sklearn.utils.extmath import randomized_svd
    u_mat, s_mat, vt_mat = randomized_svd(norm_freq_matrix, n_components=min(k+200, min(user, item)-1), random_state=42)

idx = np.argsort(s_mat)[::-1]
s_mat = s_mat[idx]
u_mat = u_mat[:, idx]
vt_mat = vt_mat[idx, :]
V_mat = vt_mat.T

U = torch.FloatTensor(u_mat).cuda()
value = torch.FloatTensor(s_mat).cuda()
V = torch.FloatTensor(V_mat).cuda()

value = value / value.max()
U_weighted = U[:, :k] * individual_weight(value[:k], homo_ratio_user)
V_weighted = V[:, :k] * individual_weight(value[:k], homo_ratio_item)

norm_freq_csc = norm_freq_matrix.tocsc()
norm_freq_csr = norm_freq_matrix.tocsr()

def test():
    def cal_idcg(k=20):
        idcg_set = [0]
        scores = 0.0
        for i in range(1, k + 1):
            scores += 1 / np.log2(1 + i)
            idcg_set.append(scores)
        return idcg_set

    ndcg10, ndcg20, recall10, recall20 = 0.0, 0.0, 0.0, 0.0
    idcg_set = cal_idcg()
    valid_users = 0
    batch_size = 500
    
    print("Evaluating in batches...")
    for start_idx in range(0, user, batch_size):
        end_idx = min(start_idx + batch_size, user)
        batch_users = end_idx - start_idx
        
        rate_batch = U_weighted[start_idx:end_idx].mm(V_weighted.t())
        rate_sum = rate_batch.sum(dim=1, keepdim=True)
        # Avoid division by zero
        rate_sum[rate_sum == 0] = 1.0
        rate_batch = rate_batch / rate_sum
        
        batch_norm_csr = norm_freq_csr[start_idx:end_idx]
        temp = batch_norm_csr.dot(norm_freq_csc.T)
        norm_conv_batch = temp.dot(norm_freq_csr).toarray()
        
        norm_conv_sum = norm_conv_batch.sum(axis=1, keepdims=True)
        norm_conv_sum[norm_conv_sum == 0] = 1.0
        norm_conv_batch = norm_conv_batch / norm_conv_sum
        norm_conv_batch_t = torch.FloatTensor(norm_conv_batch).cuda()
        
        rate_batch = torch.sigmoid(rate_batch + gamma * norm_conv_batch_t)
        
        freq_batch = freq_matrix_csr[start_idx:end_idx].toarray()
        freq_batch_t = torch.FloatTensor(freq_batch).cuda()
        rate_batch = rate_batch - freq_batch_t * 1000
        
        topn_batch = rate_batch.topk(20)[1].cpu().numpy()
        
        for idx in range(batch_users):
            now_user = start_idx + idx
            test_lens = len(test_data[now_user])
            if test_lens == 0: continue
            valid_users += 1
            
            all10 = 10 if test_lens > 10 else test_lens
            all20 = 20 if test_lens > 20 else test_lens
            
            topn = topn_batch[idx]
            dcg10, dcg20, hit10, hit20 = 0.0, 0.0, 0.0, 0.0
            for kk in range(20):
                max_item = topn[kk]
                if test_data[now_user].count(max_item) != 0:
                    if kk < 10:
                        dcg10 += 1 / np.log2(2 + kk)
                        hit10 += 1
                    dcg20 += 1 / np.log2(2 + kk)
                    hit20 += 1
                    
            ndcg10 += (dcg10 / idcg_set[all10])
            ndcg20 += (dcg20 / idcg_set[all20])
            recall10 += (hit10 / all10)
            recall20 += (hit20 / all20)

    if valid_users > 0:
        ndcg10, ndcg20 = round(ndcg10 / valid_users, 4), round(ndcg20 / valid_users, 4)
        recall10, recall20 = round(recall10 / valid_users, 4), round(recall20 / valid_users, 4)
    print(f"Recall@20: {recall20}")
    print(f"NDCG@20: {ndcg20}")
    print(ndcg10, ndcg20, recall10, recall20)

test()