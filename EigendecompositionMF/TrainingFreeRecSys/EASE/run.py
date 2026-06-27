import argparse
import os
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import splu

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Turbo_CF')))

from model import EASE

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="lastfm")
parser.add_argument("--lambda_", type=float, default=500.0)
parser.add_argument("--seed", type=int, default=2020)
parser.add_argument("--solver", type=str, default="auto", choices=["auto", "primal", "dual"], help="Exact EASE solver. auto uses dual when items greatly exceed users.")
parser.add_argument("--block_size", type=int, default=256, help="Block size for exact dual diagonal/scoring computation.")
args = parser.parse_args()


def load_txt(filename, dataset):
    rows, cols, data = [], [], []
    mu, mi = 0, 0
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 0:
                continue
            u = int(parts[0])
            if dataset == "lastfm":
                i_idx = int(parts[1])
                rows.append(u)
                cols.append(i_idx)
                data.append(1.0)
                if u > mu:
                    mu = u
                if i_idx > mi:
                    mi = i_idx
            else:
                for i in parts[1:]:
                    i_idx = int(i)
                    rows.append(u)
                    cols.append(i_idx)
                    data.append(1.0)
                    if u > mu:
                        mu = u
                    if i_idx > mi:
                        mi = i_idx
    return rows, cols, data, mu, mi


def make_user_items(rows, cols, n_u):
    user_items = [set() for _ in range(n_u)]
    for u, i in zip(rows, cols):
        user_items[u].add(i)
    return user_items


def evaluate_batched(n_u, train_items, test_items, score_batch_fn, k=20, batch_size=256):
    recall_sum = 0.0
    ndcg_sum = 0.0
    valid_users = 0
    discounts = 1.0 / np.log2(np.arange(2, k + 2))

    for start in range(0, n_u, batch_size):
        end = min(start + batch_size, n_u)
        scores = np.asarray(score_batch_fn(start, end))
        for local_idx, u in enumerate(range(start, end)):
            relevant_items = test_items[u]
            if not relevant_items:
                continue
            if train_items[u]:
                scores[local_idx, list(train_items[u])] = -99999.0
            ranked = np.argsort(-scores[local_idx])[:k]
            hits = np.array([item in relevant_items for item in ranked], dtype=np.float32)
            recall_sum += hits.sum() / len(relevant_items)
            ideal_len = min(len(relevant_items), k)
            idcg = discounts[:ideal_len].sum()
            ndcg_sum += (hits * discounts[:len(hits)]).sum() / idcg if idcg > 0 else 0.0
            valid_users += 1

    if valid_users == 0:
        return 0.0, 0.0
    return recall_sum / valid_users, ndcg_sum / valid_users


def choose_solver(solver, n_u, n_i):
    if solver != "auto":
        return solver
    return "dual" if n_i > n_u and n_i >= 50000 else "primal"


def build_primal_score_fn(X, lambda_):
    n_i = X.shape[1]
    print(f"EASE: exact primal inverse solve for {n_i} items", flush=True)
    G = X.T.dot(X).toarray().astype(np.float64, copy=False)
    diag_indices = np.diag_indices(G.shape[0])
    G[diag_indices] += lambda_
    P = np.linalg.inv(G)
    B = P / (-np.diag(P))
    B[diag_indices] = 0.0

    def score_batch(start, end):
        return X[start:end].dot(B)

    return score_batch


def build_dual_score_fn(X, lambda_, block_size):
    n_u, n_i = X.shape
    print(f"EASE: exact dual Woodbury sparse solve for {n_u} users, {n_i} items", flush=True)
    a = 1.0 / lambda_
    M = sp.eye(n_u, format="csc", dtype=np.float64) + (X.dot(X.T).tocsc().astype(np.float64) * a)
    lu = splu(M)
    diag_p = np.empty(n_i, dtype=np.float64)

    print("EASE: computing exact inverse diagonal in item blocks", flush=True)
    for start in range(0, n_i, block_size):
        end = min(start + block_size, n_i)
        C = X[:, start:end].toarray().astype(np.float64, copy=False)
        Z = lu.solve(C)
        diag_p[start:end] = a - (a * a) * np.einsum("ij,ij->j", C, Z)
    diag_p[diag_p == 0] = np.finfo(np.float64).eps

    def score_batch(start, end):
        Xb = X[start:end]
        overlap = Xb.dot(X.T).toarray().T.astype(np.float64, copy=False)
        Z = lu.solve(overlap)
        low_rank = X.T.dot(Z).T
        low_rank = np.asarray(low_rank, dtype=np.float64)
        XP = a * Xb.toarray().astype(np.float64, copy=False) - (a * a) * low_rank
        return -XP / diag_p

    return score_batch


data_root = os.environ.get(
    "TFRECSYS_DATA_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data")),
)
data_dir = os.path.abspath(os.path.join(data_root, args.dataset))
train_file = "data1.txt" if args.dataset == "lastfm" else "train.txt"
test_file = "test1.txt" if args.dataset == "lastfm" else "test.txt"

tr_r, tr_c, tr_d, mu1, mi1 = load_txt(os.path.join(data_dir, train_file), args.dataset)
ts_r, ts_c, ts_d, mu2, mi2 = load_txt(os.path.join(data_dir, test_file), args.dataset)

n_u = max(mu1, mu2) + 1
n_i = max(mi1, mi2) + 1

# Keep the encoder setup for compatibility with the original EASE implementation.
df_train = pd.DataFrame({'user_id': tr_r, 'item_id': tr_c})
model = EASE()
model.user_enc.fit(np.arange(n_u))
model.item_enc.fit(np.arange(n_i))

X = sp.csr_matrix((tr_d, (tr_r, tr_c)), shape=(n_u, n_i), dtype=np.float64)
train_items = make_user_items(tr_r, tr_c, n_u)
test_items = make_user_items(ts_r, ts_c, n_u)

solver = choose_solver(args.solver, n_u, n_i)
print(f"EASE solver: {solver}", flush=True)
if solver == "dual":
    exact_scores = build_dual_score_fn(X, args.lambda_, args.block_size)
else:
    exact_scores = build_primal_score_fn(X, args.lambda_)

recall, ndcg = evaluate_batched(n_u, train_items, test_items, exact_scores, k=20, batch_size=256)

print(f"recall@[20]: [{recall:.4f}]")
print(f"ndcg@[20]: [{ndcg:.4f}]")
