#!/usr/bin/env python3
"""Split combined datasets into seed-specific train/val/test folders."""

from __future__ import annotations

import argparse
import random
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent
COMBINED_DIR = DATA_DIR / "combined_dataset"


def read_combined(path: Path) -> dict[int, list[int]]:
    interactions: dict[int, list[int]] = {}
    with path.open() as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            user = int(parts[0])
            items = sorted({int(item) for item in parts[1:]})
            if items:
                interactions[user] = items
    return interactions


def split_counts(n_items: int, train_ratio: float, val_ratio: float) -> tuple[int, int, int]:
    if n_items <= 0:
        return 0, 0, 0
    if n_items == 1:
        return 1, 0, 0
    if n_items == 2:
        return 1, 0, 1

    test_ratio = 1.0 - train_ratio - val_ratio
    n_val = max(1, int(round(n_items * val_ratio)))
    n_test = max(1, int(round(n_items * test_ratio)))
    if n_val + n_test >= n_items:
        overflow = n_val + n_test - (n_items - 1)
        reduce_test = min(overflow, max(0, n_test - 1))
        n_test -= reduce_test
        overflow -= reduce_test
        n_val -= min(overflow, max(0, n_val - 1))
    n_train = n_items - n_val - n_test
    return n_train, n_val, n_test


def write_split(path: Path, rows: dict[int, list[int]]) -> int:
    interactions = 0
    with path.open("w") as f:
        for user in sorted(rows):
            items = rows[user]
            if not items:
                continue
            interactions += len(items)
            f.write(" ".join([str(user)] + [str(item) for item in sorted(items)]) + "\n")
    return interactions


def split_file(
    combined_file: Path,
    output_root: Path,
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> tuple[int, int, int]:
    dataset = combined_file.name.removesuffix("_combined.txt")
    rng = random.Random(f"{seed}:{dataset}")
    interactions = read_combined(combined_file)

    train_rows: dict[int, list[int]] = {}
    val_rows: dict[int, list[int]] = {}
    test_rows: dict[int, list[int]] = {}

    for user, items in interactions.items():
        shuffled = list(items)
        rng.shuffle(shuffled)
        n_train, n_val, _ = split_counts(len(shuffled), train_ratio, val_ratio)
        train_rows[user] = shuffled[:n_train]
        val_rows[user] = shuffled[n_train:n_train + n_val]
        test_rows[user] = shuffled[n_train + n_val:]

    dataset_dir = output_root / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    train_count = write_split(dataset_dir / "train.txt", train_rows)
    val_count = write_split(dataset_dir / "val.txt", val_rows)
    test_count = write_split(dataset_dir / "test.txt", test_rows)
    return train_count, val_count, test_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Create seed train/validation/test splits.")
    parser.add_argument("seed", type=int, help="Random seed used for per-user splits.")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--datasets", nargs="+", help="Optional dataset names to split.")
    args = parser.parse_args()

    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-8:
        raise ValueError("Split ratios must sum to 1.0")
    if not COMBINED_DIR.exists():
        raise FileNotFoundError(f"Combined dataset directory not found: {COMBINED_DIR}")

    selected = set(args.datasets or [])
    files = sorted(COMBINED_DIR.glob("*_combined.txt"))
    if selected:
        files = [path for path in files if path.name.removesuffix("_combined.txt") in selected]

    output_root = DATA_DIR / f"00_dataset_seed_{args.seed}"
    output_root.mkdir(parents=True, exist_ok=True)

    for combined_file in files:
        train_count, val_count, test_count = split_file(
            combined_file,
            output_root,
            args.seed,
            args.train_ratio,
            args.val_ratio,
        )
        dataset = combined_file.name.removesuffix("_combined.txt")
        print(
            f"{dataset:<20} train={train_count:<9} "
            f"val={val_count:<9} test={test_count:<9}"
        )
    print(f"Wrote split dataset(s) to {output_root}")


if __name__ == "__main__":
    main()
