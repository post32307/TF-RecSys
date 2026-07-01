import argparse
import hashlib
import os
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import coo_matrix, diags, hstack, identity, vstack
from scipy.sparse.linalg import eigsh


parser = argparse.ArgumentParser(description="Runnable PGSP implementation converted from PGSP.ipynb.")
parser.add_argument("--dataset", type=str, default="gowalla")
parser.add_argument("--k", type=int, default=512, help="Number of eigenvectors, called DIM in the notebook.")
parser.add_argument("--phi", type=float, default=0.7, help="Mixture weight for P0 versus P1.")
parser.add_argument("--top_k", type=int, default=20)
parser.add_argument("--seed", type=int, default=2020)
parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
parser.add_argument("--no_p0_norm", action="store_true", help="Use the notebook's raw P0 branch.")
parser.add_argument("--no_p1_norm", action="store_true", help="Use the notebook's raw P1 branch.")
parser.add_argument("--cache_dir", type=str, default=None)
parser.add_argument("--no_cache", action="store_true")
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
D_TYPE = np.float32 if args.dtype == "float32" else np.float64
P0_NORM = not args.no_p0_norm
P1_NORM = not args.no_p1_norm

BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("TFRECSYS_DATA_ROOT", BASE_DIR.parent.parent / "data" / f"00_dataset_seed_{args.seed}"))
data_dir = DATA_ROOT / args.dataset
train_file = data_dir / "train.txt"
test_file = data_dir / os.environ.get("TFRECSYS_EVAL_FILE", "test.txt")
cache_root = Path(args.cache_dir) if args.cache_dir else BASE_DIR / "temp"


def parse_items(parts, dataset):
    return [int(i) for i in parts[1:]]


def load_train(path, dataset):
    rows, cols = [], []
    max_user = -1
    max_item = -1
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            uid = int(parts[0])
            items = parse_items(parts, dataset)
            max_user = max(max_user, uid)
            for item in items:
                rows.append(uid)
                cols.append(item)
                max_item = max(max_item, item)
    user_nums = max_user + 1
    item_nums = max_item + 1
    values = np.ones(len(rows), dtype=D_TYPE)
    R = coo_matrix((values, (rows, cols)), shape=(user_nums, item_nums), dtype=D_TYPE).tocsr()
    return R, np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)


def load_test(path, dataset, item_nums):
    ground_truth = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            uid = int(parts[0])
            items = [item for item in parse_items(parts, dataset) if item < item_nums]
            if items:
                ground_truth[uid] = items
    return ground_truth


def inv_sqrt_diag(values):
    values = np.asarray(values).reshape(-1)
    out = np.zeros_like(values, dtype=D_TYPE)
    np.power(values, -0.5, out=out, where=values > 0)
    return diags(out, offsets=0, dtype=D_TYPE)


def sqrt_diag(values):
    values = np.asarray(values).reshape(-1)
    out = np.zeros_like(values, dtype=D_TYPE)
    np.sqrt(values, out=out, where=values > 0)
    return diags(out, offsets=0, dtype=D_TYPE)


def file_digest(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def cache_path(kind, suffix):
    path = cache_root / kind
    path.mkdir(parents=True, exist_ok=True)
    return path / suffix


def process_is_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@contextmanager
def cache_lock(lock_file):
    lock_file = Path(lock_file)
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            fd = os.open(lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w") as f:
                f.write(str(os.getpid()))
            break
        except FileExistsError:
            try:
                lock_pid = int(lock_file.read_text().strip())
            except (OSError, ValueError):
                lock_pid = None
            if lock_pid is not None and not process_is_alive(lock_pid):
                try:
                    lock_file.unlink()
                    continue
                except FileNotFoundError:
                    continue
            time.sleep(1)
    try:
        yield
    finally:
        try:
            lock_file.unlink()
        except FileNotFoundError:
            pass


def atomic_save_npy(path, array):
    path = Path(path)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp_path, "wb") as f:
            np.save(f, array)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass


def load_npy_or_recompute(path, description, compute_fn):
    if args.no_cache:
        return compute_fn()

    lock_file = Path(str(path) + ".lock")
    with cache_lock(lock_file):
        if path.exists():
            try:
                print(f"Loading cached {description}...", flush=True)
                return np.load(path)
            except (OSError, ValueError) as exc:
                print(f"Discarding unreadable cached {description}: {exc}", flush=True)
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass

        print(f"Computing {description}...", flush=True)
        array = compute_fn()
        atomic_save_npy(path, array)
        return array


def load_eig_or_recompute(vec_file, val_file, dim):
    if args.no_cache:
        print(f"Computing eigsh dim={dim}...", flush=True)
        return eigsh(L_norm, k=dim, which="SA")

    lock_file = cache_path("eig", f"eig_{cache_tag}_{dim}.lock")
    with cache_lock(lock_file):
        if vec_file.exists() and val_file.exists():
            try:
                print(f"Loading cached eigsh dim={dim}...", flush=True)
                return np.load(val_file), np.load(vec_file)
            except (OSError, ValueError) as exc:
                print(f"Discarding unreadable cached eigsh dim={dim}: {exc}", flush=True)
                for path in (vec_file, val_file):
                    try:
                        path.unlink()
                    except FileNotFoundError:
                        pass

        print(f"Computing eigsh dim={dim}...", flush=True)
        val, vec = eigsh(L_norm, k=dim, which="SA")
        atomic_save_npy(vec_file, vec)
        atomic_save_npy(val_file, val)
        return val, vec


def compute_metrics(pred_indices, ground_truth, user_nums, top_k):
    recall = 0.0
    precise = 0.0
    ndcg = 0.0
    test_user_nums = len(ground_truth)
    if test_user_nums == 0:
        return 0.0, 0.0, 0.0
    for test_uid in range(user_nums):
        if test_uid not in ground_truth:
            continue
        truth = ground_truth[test_uid]
        truth_set = set(truth)
        pred = pred_indices[test_uid].tolist()
        dcg = 0.0
        idcg = 0.0
        for i in range(top_k):
            if pred[i] in truth_set:
                dcg += 1 / np.log(2 + i)
        for i in range(min(top_k, len(truth))):
            idcg += 1 / np.log(2 + i)
        ndcg += dcg / idcg if idcg > 0 else 0.0
        hits = len(set(pred) & truth_set)
        recall += hits / len(truth)
        precise += hits / top_k
    recall /= test_user_nums
    precise /= test_user_nums
    f1 = 2 * recall * precise / (recall + precise) if recall + precise > 0 else 0.0
    ndcg /= test_user_nums
    return recall, ndcg, f1


print(f"PGSP dataset={args.dataset} k={args.k} phi={args.phi}", flush=True)
R, train_users, train_items = load_train(train_file, args.dataset)
ground_truth = load_test(test_file, args.dataset, R.shape[1])
user_nums, item_nums = R.shape
print(f"Loaded train matrix users={user_nums} items={item_nums} nnz={R.nnz}", flush=True)

user_degree = np.asarray(R.sum(axis=1)).reshape(-1)
item_degree = np.asarray(R.sum(axis=0)).reshape(-1)
Du_ = inv_sqrt_diag(user_degree)
Di_ = inv_sqrt_diag(item_degree)
Du = sqrt_diag(user_degree)
Di = sqrt_diag(item_degree)

Ru = Du_ * R
Ri = R * Di_
Cu = Ri * Ri.T
Ci = Ru.T * Ru
R_post = Ru * Di_
Ci0 = R_post.T * R_post
Cu0 = R_post * R_post.T

fingerprint = file_digest(train_file) if not args.no_cache else "nocache"
cache_tag = f"{args.dataset}_{fingerprint}_{args.dtype}"

A = vstack([hstack([Cu, R]), hstack([R.T, Ci])], dtype=D_TYPE).tocsr()
I = identity(user_nums + item_nums, dtype=D_TYPE, format="csr")
D_ = inv_sqrt_diag(np.asarray(A.sum(axis=0)).reshape(-1))
L_norm = I - D_ * A * D_

dim = min(args.k, user_nums + item_nums - 2)
if dim <= 0:
    raise ValueError("PGSP requires at least three total user/item nodes")
vec_file = cache_path("eig", f"vec_{cache_tag}_{dim}.npy")
val_file = cache_path("eig", f"val_{cache_tag}_{dim}.npy")
val, vec = load_eig_or_recompute(vec_file, val_file, dim)

R_b = hstack([Cu0, R], dtype=D_TYPE).tocsr()
rb_degree = np.asarray(R_b.sum(axis=0)).reshape(-1)
D_Rb_i_ = inv_sqrt_diag(rb_degree)
D_Rb_i = sqrt_diag(rb_degree).toarray()

p0_suffix = "norm" if P0_NORM else "raw"
p0_file = cache_path("P0", f"P0_{cache_tag}_{p0_suffix}.npy")
def compute_p0():
    if P0_NORM:
        P0 = R * Di_
        P0 = P0 * Ci0
        P0 = P0 * Di
    else:
        P0 = R * Ci0
    return P0.toarray().astype(D_TYPE, copy=False)

P0 = load_npy_or_recompute(p0_file, "P0", compute_p0)

p1_suffix = "norm" if P1_NORM else "raw"
p1_file = cache_path("P1", f"P1_{cache_tag}_{dim}_{p1_suffix}.npy")
def compute_p1():
    if P1_NORM:
        P1 = R_b * D_Rb_i_
        P1 = P1 * vec
        P11 = np.matmul(vec.T, D_Rb_i)
        P1 = np.matmul(P1, P11)
        P1 = P1[:, user_nums:]
    else:
        P1 = R_b * vec
        P1 = np.matmul(P1, vec.T)
        P1 = P1[:, user_nums:]
    return P1.astype(D_TYPE, copy=False)

P1 = load_npy_or_recompute(p1_file, "P1", compute_p1)

print("Evaluating...", flush=True)
P = args.phi * P0 + (1 - args.phi) * P1
P = torch.from_numpy(np.asarray(P))
P[train_users, train_items] = 1e-9
_, pred_indices = P.topk(args.top_k)
recall, ndcg, f1 = compute_metrics(pred_indices, ground_truth, user_nums, args.top_k)
print(f"Recall@20: {recall:.4f}")
print(f"NDCG@20: {ndcg:.4f}")
print(np.round(recall, 4), np.round(ndcg, 4))
