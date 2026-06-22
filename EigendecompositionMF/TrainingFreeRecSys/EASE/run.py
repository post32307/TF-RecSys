import argparse
import os
import sys
import numpy as np
import pandas as pd
import scipy.sparse as sp

# Add Turbo_CF to path to use its load_txt and metrics
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Turbo_CF')))
from utils import recall_at_k, ndcg_at_k

from model import EASE

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="lastfm")
parser.add_argument("--lambda_", type=float, default=500.0)
parser.add_argument("--seed", type=int, default=2020)
args = parser.parse_args()

def load_txt(filename, dataset):
    rows, cols, data = [], [], []
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
                data.append(1.0)
                if u > mu: mu = u
                if i_idx > mi: mi = i_idx
            else:
                for i in parts[1:]:
                    i_idx = int(i)
                    rows.append(u)
                    cols.append(i_idx)
                    data.append(1.0)
                    if u > mu: mu = u
                    if i_idx > mi: mi = i_idx
    return rows, cols, data, mu, mi

data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", args.dataset))
train_file = "data1.txt" if args.dataset == "lastfm" else "train.txt"
test_file = "test1.txt" if args.dataset == "lastfm" else "test.txt"

tr_r, tr_c, tr_d, mu1, mi1 = load_txt(os.path.join(data_dir, train_file), args.dataset)
ts_r, ts_c, ts_d, mu2, mi2 = load_txt(os.path.join(data_dir, test_file), args.dataset)

n_u = max(mu1, mu2) + 1
n_i = max(mi1, mi2) + 1

# Create DataFrames for EASE model
df_train = pd.DataFrame({'user_id': tr_r, 'item_id': tr_c})

model = EASE()
model.user_enc.fit(np.arange(n_u))
model.item_enc.fit(np.arange(n_i))

model.fit(df_train, lambda_=args.lambda_, implicit=True)

# model.pred is the full prediction matrix (users x items).
# But wait, self.pred will have shape (n_users_in_train x n_items_in_train).
# The encoding maps to train indices, which might not be all items!
# To align perfectly with test indices, we should directly build matrices.

# Let's bypass model.fit and do EASE directly here to avoid encoding issues with test set!
X = sp.csr_matrix((tr_d, (tr_r, tr_c)), shape=(n_u, n_i))
G = X.T.dot(X).toarray()
diagIndices = np.diag_indices(G.shape[0])
G[diagIndices] += args.lambda_
P = np.linalg.inv(G)
B = P / (-np.diag(P))
B[diagIndices] = 0

pred = X.dot(B)

# Mask training interactions
pred = pred + (-99999) * X.toarray()

# Build test ground truth matrix
gt_mat = sp.csr_matrix((ts_d, (ts_r, ts_c)), shape=(n_u, n_i)).toarray()

recall = recall_at_k(gt_mat, pred, k=20)
ndcg = ndcg_at_k(gt_mat, pred, k=20)

print(f"recall@[20]: [{recall:.4f}]")
print(f"ndcg@[20]: [{ndcg:.4f}]")
