#!/usr/bin/env python3
"""Report statistics for original or seed-split datasets."""

from __future__ import annotations

import argparse
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent
SPLIT_ROOT = DATA_DIR / "split_dataset"


def add_standard(path: Path, users: set[int], items: set[int]) -> int:
    count = 0
    if not path.exists():
        return count
    with path.open() as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            user = int(parts[0])
            users.add(user)
            for raw_item in parts[1:]:
                items.add(int(raw_item))
                count += 1
    return count


def add_lastfm_original(path: Path, users: set[int], items: set[int]) -> int:
    count = 0
    if not path.exists():
        return count
    with path.open() as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            users.add(int(parts[0]) - 1)
            items.add(int(parts[1]) - 1)
            count += 1
    return count


def stats_for_dataset(dataset_dir: Path, original: bool) -> dict[str, int]:
    users: set[int] = set()
    items: set[int] = set()
    dataset = dataset_dir.name

    if original and dataset == "lastfm":
        train = add_lastfm_original(dataset_dir / "data1.txt", users, items)
        val = 0
        test = add_lastfm_original(dataset_dir / "test1.txt", users, items)
    else:
        train = add_standard(dataset_dir / "train.txt", users, items)
        val = add_standard(dataset_dir / "val.txt", users, items)
        test = add_standard(dataset_dir / "test.txt", users, items)

    return {
        "users": len(users),
        "items": len(items),
        "max_user": max(users) if users else -1,
        "max_item": max(items) if items else -1,
        "train": train,
        "val": val,
        "test": test,
        "total": train + val + test,
    }


def split_ratio_text(stats: dict[str, int]) -> str:
    total = stats["total"]
    if total == 0:
        return "0.0:0.0:0.0"
    return (
        f"{stats['train'] / total:.3f}:"
        f"{stats['val'] / total:.3f}:"
        f"{stats['test'] / total:.3f}"
    )


def print_stats(root: Path, original: bool, datasets: list[str] | None) -> None:
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    selected = set(datasets or [])
    dirs = [
        path for path in root.iterdir()
        if path.is_dir() and not path.name.startswith(".") and (not selected or path.name in selected)
    ]
    dirs = sorted(dirs, key=lambda path: path.name)

    print(f"Dataset root: {root}")
    print("-" * 130)
    print(
        f"{'Dataset':<20} | {'Users':>8} | {'Items':>8} | {'MaxU':>8} | "
        f"{'MaxI':>8} | {'Train':>10} | {'Val':>10} | {'Test':>10} | "
        f"{'Total':>10} | {'Train:Val:Test':>16}"
    )
    print("-" * 130)
    for dataset_dir in dirs:
        stats = stats_for_dataset(dataset_dir, original)
        print(
            f"{dataset_dir.name:<20} | {stats['users']:>8} | {stats['items']:>8} | "
            f"{stats['max_user']:>8} | {stats['max_item']:>8} | "
            f"{stats['train']:>10} | {stats['val']:>10} | "
            f"{stats['test']:>10} | {stats['total']:>10} | {split_ratio_text(stats):>16}"
        )
    print("-" * 130)


def main() -> None:
    parser = argparse.ArgumentParser(description="Report dataset statistics.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--seed", type=int, help="Inspect data/split_dataset/00_dataset_seed_<seed>.")
    group.add_argument("--root", type=Path, help="Inspect an explicit dataset root.")
    parser.add_argument("--datasets", nargs="+", help="Optional dataset names to report.")
    args = parser.parse_args()

    if args.seed is not None:
        root = SPLIT_ROOT / f"00_dataset_seed_{args.seed}"
        original = False
    else:
        root = args.root if args.root else DATA_DIR / "og_dataset"
        original = root.resolve() == (DATA_DIR / "og_dataset").resolve()

    print_stats(root, original, args.datasets)


if __name__ == "__main__":
    main()
