import random
import numpy as np
import torch
import scipy.sparse as sp
import os
import argparse
from utils import csr2torch, recall_at_k, ndcg_at_k, normalize_sparse_adjacency_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
current_directory = os.getcwd()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    type=str,
    default="gowalla",
    help="Either gowalla, yelp, amazon, or ml-1m",
)
parser.add_argument(
    "--verbose",
    type=int,
    default=1,
    help="Whether to print the results or not. 1 prints the results, 0 does not.",
)
parser.add_argument("--alpha", type=float, default=0.5, help="For normalization of R")
parser.add_argument("--power", type=float, default=1, help="For normalization of P")
parser.add_argument("--filter", type=float, default=1, help="1:linear, 2: 2nd-order, 3: Poly.approx of ideal LPF")
parser.add_argument("--seed", type=int, default=2020)


random.seed(2022)
np.random.seed(2022)


if __name__ == "__main__":
    args = parser.parse_args()
    if args.verbose:
        print(f"Device: {device}")
    dataset = args.dataset
    data_root_override = os.environ.get("TFRECSYS_DATA_ROOT")
    path_tr = f"{current_directory}/dataset/{dataset}_train.npz"
    path_ts = f"{current_directory}/dataset/{dataset}_test.npz"
    
    if not data_root_override and os.path.exists(path_tr) and os.path.exists(path_ts):
        R_tr = csr2torch(sp.load_npz(path_tr)).to(device)
        R_ts = csr2torch(sp.load_npz(path_ts)).to(device)
    else:
        def load_txt(filename):
            rows, cols, data = [], [], []
            mu, mi = 0, 0
            with open(filename, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 0: continue
                    u = int(parts[0])
                    if dataset == "lastfm":
                        i_idx = int(parts[1]) - 1
                        u = u - 1
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

        data_root = data_root_override or os.path.abspath(os.path.join(current_directory, "..", "..", "data"))
        data_dir = os.path.abspath(os.path.join(data_root, dataset))
        train_file = "data1.txt" if dataset == "lastfm" else "train.txt"
        test_file = "test1.txt" if dataset == "lastfm" else "test.txt"
        
        tr_r, tr_c, tr_d, mu1, mi1 = load_txt(os.path.join(data_dir, train_file))
        ts_r, ts_c, ts_d, mu2, mi2 = load_txt(os.path.join(data_dir, test_file))
        
        n_u = max(mu1, mu2) + 1
        n_i = max(mi1, mi2) + 1
        
        R_tr_csr = sp.csr_matrix((tr_d, (tr_r, tr_c)), shape=(n_u, n_i))
        R_ts_csr = sp.csr_matrix((ts_d, (ts_r, ts_c)), shape=(n_u, n_i))
        
        R_tr = csr2torch(R_tr_csr).to(device)
        R_ts = csr2torch(R_ts_csr).to(device)

    n_users = R_tr.shape[0]
    n_items = R_tr.shape[1]
    if args.verbose:
        print(f"number of users: {n_users}")
        print(f"number of items: {n_items}")

    n_inters = torch.nonzero(R_tr._values()).cpu().size(0) + torch.nonzero(
        R_ts[0]._values()
    ).cpu().size(0)

    if args.verbose:
        print(f"number of overall ratings: {n_inters}")

    # Use scipy sparse to do the calculations efficiently without OOM
    alpha = args.alpha
    rowsum = np.array(R_tr_csr.sum(axis=1)).flatten()
    rowsum = np.power(rowsum + 1e-12, -alpha)
    d_mat_rows = sp.diags(rowsum)
    
    colsum = np.array(R_tr_csr.sum(axis=0)).flatten()
    colsum = np.power(colsum + 1e-12, alpha - 1)
    d_mat_cols = sp.diags(colsum)
    
    R_norm_sp = d_mat_rows.dot(R_tr_csr).dot(d_mat_cols)
    
    P_sp = R_norm_sp.T.dot(R_norm_sp)
    
    # Apply power
    P_sp.data = np.where(P_sp.data > 1e-6, P_sp.data ** args.power, 0)
    P_sp.eliminate_zeros()
    
    # Filter
    if args.filter == 1:
        results_sp = R_tr_csr.dot(P_sp)
    elif args.filter == 2:
        P2 = P_sp.dot(P_sp)
        results_sp = R_tr_csr.dot(2*P_sp - P2)
    elif args.filter == 3:
        P2 = P_sp.dot(P_sp)
        P3 = P2.dot(P_sp)
        results_sp = R_tr_csr.dot(P_sp + 0.01*(-P3 + 10*P2 - 29*P_sp))
        
    # Evaluate in batches
    batch_size = 1000
    n_users = R_tr_csr.shape[0]
    recall_sum = 0.0
    ndcg_sum = 0.0
    valid_users = 0
    
    for start_idx in range(0, n_users, batch_size):
        end_idx = min(start_idx + batch_size, n_users)
        
        batch_results = results_sp[start_idx:end_idx].toarray()
        batch_tr = R_tr_csr[start_idx:end_idx].toarray()
        batch_ts = R_ts_csr[start_idx:end_idx].toarray()
        
        batch_results -= 99999 * batch_tr
        
        for i in range(end_idx - start_idx):
            relevant_items = set(np.where(batch_ts[i, :] > 0)[0])
            if len(relevant_items) == 0:
                continue
            valid_users += 1
            top_predicted_items = np.argsort(-batch_results[i, :])[:20]
            
            num_relevant_items = len(relevant_items.intersection(top_predicted_items))
            recall_sum += num_relevant_items / len(relevant_items)
            
            dcg = 0
            idcg = 0
            for j in range(20):
                if top_predicted_items[j] in relevant_items:
                    dcg += 1 / np.log2(j + 2)
                if j < len(relevant_items):
                    idcg += 1 / np.log2(j + 2)
            ndcg_sum += dcg / idcg if idcg > 0 else 0
            
    recall = recall_sum / valid_users if valid_users > 0 else 0
    ndcg = ndcg_sum / valid_users if valid_users > 0 else 0
    print(f"Recall@20: {recall:.4f}")
    print(f"NDCG@20: {ndcg:.4f}")
