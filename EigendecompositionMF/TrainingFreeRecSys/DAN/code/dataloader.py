import os
from pathlib import Path

import numpy as np
import scipy.sparse as sp


def read_interactions(path):
    interactions = {}
    max_uid = -1
    max_iid = -1
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            parts = raw_line.split()
            if not parts:
                continue
            uid = int(parts[0])
            items = {int(item) for item in parts[1:]}
            max_uid = max(max_uid, uid)
            if items:
                interactions[uid] = items
                max_iid = max(max_iid, max(items))
    return interactions, max_uid, max_iid


def interactions_to_csr(interactions, shape):
    rows = []
    cols = []
    for uid, items in interactions.items():
        rows.extend([uid] * len(items))
        cols.extend(items)
    values = np.ones(len(rows), dtype=np.float64)
    return sp.csr_matrix((values, (rows, cols)), shape=shape, dtype=np.float64)


def default_data_root(seed):
    return (
        Path(__file__).resolve().parents[3]
        / "data"
        / "split_dataset"
        / f"00_dataset_seed_{seed}"
    )


def load_dataset(dataset, seed):
    data_root = Path(
        os.environ.get("TFRECSYS_DATA_ROOT", default_data_root(seed))
    ).expanduser()
    data_dir = data_root / dataset
    eval_filename = os.environ.get("TFRECSYS_EVAL_FILE", "test.txt")
    train_path = data_dir / "train.txt"
    eval_path = data_dir / eval_filename

    train_inter, train_max_uid, train_max_iid = read_interactions(train_path)
    eval_inter, eval_max_uid, eval_max_iid = read_interactions(eval_path)
    num_users = max(train_max_uid, eval_max_uid) + 1
    num_items = max(train_max_iid, eval_max_iid) + 1
    if num_users <= 0 or num_items <= 0:
        raise ValueError(f"No interactions found in {train_path} and {eval_path}")
    if not eval_inter:
        raise ValueError(f"No evaluable users found in {eval_path}")

    train_matrix = interactions_to_csr(
        train_inter, shape=(num_users, num_items)
    )
    return train_matrix, eval_inter, data_root, eval_filename
