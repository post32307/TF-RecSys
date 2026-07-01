#!/usr/bin/env python3
"""Combine original train/test interactions into one file per dataset.

Outputs standard zero-based user-line files:
    combined_dataset/<dataset>_combined.txt

Each output line is:
    user item1 item2 ...
"""

from __future__ import annotations

import argparse
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent
OG_DIR = DATA_DIR / "og_dataset"
COMBINED_DIR = DATA_DIR / "combined_dataset"


def dataset_dirs(selected: list[str] | None = None) -> list[Path]:
    if not OG_DIR.exists():
        raise FileNotFoundError(f"Original dataset directory not found: {OG_DIR}")
    wanted = set(selected or [])
    dirs = [
        path for path in OG_DIR.iterdir()
        if path.is_dir() and not path.name.startswith(".") and (not wanted or path.name in wanted)
    ]
    return sorted(dirs, key=lambda path: path.name)


def add_interaction(interactions: dict[int, set[int]], user: int, item: int) -> None:
    if user < 0 or item < 0:
        raise ValueError(f"Negative id after normalization: user={user}, item={item}")
    interactions.setdefault(user, set()).add(item)


def read_standard_file(path: Path, interactions: dict[int, set[int]]) -> int:
    count = 0
    if not path.exists():
        return count
    with path.open() as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            user = int(parts[0])
            for raw_item in parts[1:]:
                add_interaction(interactions, user, int(raw_item))
                count += 1
    return count


def read_lastfm_file(path: Path, interactions: dict[int, set[int]]) -> int:
    count = 0
    if not path.exists():
        return count
    with path.open() as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            add_interaction(interactions, int(parts[0]) - 1, int(parts[1]) - 1)
            count += 1
    return count


def combine_dataset(dataset_dir: Path, output_dir: Path) -> tuple[int, int, int]:
    interactions: dict[int, set[int]] = {}
    dataset = dataset_dir.name

    if dataset == "lastfm":
        train_count = read_lastfm_file(dataset_dir / "data1.txt", interactions)
        test_count = read_lastfm_file(dataset_dir / "test1.txt", interactions)
    else:
        train_count = read_standard_file(dataset_dir / "train.txt", interactions)
        test_count = read_standard_file(dataset_dir / "test.txt", interactions)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{dataset}_combined.txt"
    with output_path.open("w") as f:
        for user in sorted(interactions):
            items = sorted(interactions[user])
            if items:
                f.write(" ".join([str(user)] + [str(item) for item in items]) + "\n")

    combined_count = sum(len(items) for items in interactions.values())
    return train_count, test_count, combined_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine original train/test datasets.")
    parser.add_argument("--datasets", nargs="+", help="Optional dataset names to combine.")
    args = parser.parse_args()

    total = 0
    for dataset_dir in dataset_dirs(args.datasets):
        train_count, test_count, combined_count = combine_dataset(dataset_dir, COMBINED_DIR)
        total += 1
        print(
            f"{dataset_dir.name:<20} train={train_count:<9} "
            f"test={test_count:<9} combined_unique={combined_count:<9}"
        )
    print(f"Combined {total} dataset(s) into {COMBINED_DIR}")


if __name__ == "__main__":
    main()
