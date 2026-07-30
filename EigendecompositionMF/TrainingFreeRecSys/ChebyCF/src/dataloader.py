import logging
import os
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader, Dataset


def _read_interactions(path):
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


def _build_interaction_matrix(interactions, shape):
    rows = []
    cols = []
    for uid, items in interactions.items():
        rows.extend([uid] * len(items))
        cols.extend(items)
    values = np.ones(len(rows), dtype=np.float32)
    return sp.coo_matrix((values, (rows, cols)), shape=shape, dtype=np.float32)


class AllRankData(Dataset):
    def __init__(self, observed_inter, test_inter):
        self.observed_inter = observed_inter.tocsr()
        self.test_inter = test_inter
        self.test_users = sorted(test_inter)

    def __len__(self):
        return len(self.test_users)

    def __getitem__(self, idx):
        uid = self.test_users[idx]
        observed = self.observed_inter[uid].toarray().ravel()
        return observed, self.test_inter[uid]


def collate_fn(batch, device):
    observed_inters, label_inters = zip(*batch)
    observed = torch.from_numpy(np.stack(observed_inters)).to(device)
    return observed, label_inters


def _default_data_root(seed):
    return (
        Path(__file__).resolve().parents[3]
        / "data"
        / "split_dataset"
        / f"00_dataset_seed_{seed}"
    )


def load_data(dataset, batch_size, device, seed):
    data_root = Path(
        os.environ.get("TFRECSYS_DATA_ROOT", _default_data_root(seed))
    ).expanduser()
    data_dir = data_root / dataset
    eval_filename = os.environ.get("TFRECSYS_EVAL_FILE", "test.txt")
    train_path = data_dir / "train.txt"
    eval_path = data_dir / eval_filename

    train_inter, train_max_uid, train_max_iid = _read_interactions(train_path)
    eval_inter, eval_max_uid, eval_max_iid = _read_interactions(eval_path)
    num_users = max(train_max_uid, eval_max_uid) + 1
    num_items = max(train_max_iid, eval_max_iid) + 1
    if num_users <= 0 or num_items <= 0:
        raise ValueError(f"No interactions found in {train_path} and {eval_path}")

    train_matrix = _build_interaction_matrix(
        train_inter, shape=(num_users, num_items)
    )
    test_data = AllRankData(train_matrix, eval_inter)
    if len(test_data) == 0:
        raise ValueError(f"No evaluable users found in {eval_path}")

    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, device),
    )

    logging.info("[DATASET] %s", dataset)
    logging.info("Data root: %s", data_root)
    logging.info("Evaluation file: %s", eval_filename)
    logging.info("# of Users: %d", num_users)
    logging.info("# of Items: %d", num_items)
    logging.info("# of Training Interactions: %d", train_matrix.count_nonzero())
    logging.info(
        "# of Evaluation Interactions: %d",
        sum(len(items) for items in eval_inter.values()),
    )
    logging.info("")
    return train_matrix, test_dataloader
