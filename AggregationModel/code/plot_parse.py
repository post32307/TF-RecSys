"""Command-line options used only by plot_signal.py."""

import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare baseline scoring matrices by user/item groups"
    )
    parser.add_argument(
        "--models", nargs="+", default=["EASE", "SGFCF"],
        choices=["EASE", "SGFCF", "BSPM", "ChebyCF", "DAN",
                 "GF-CF", "PGSP", "TurboCF"],
        help="Baseline models to score and plot (default: EASE SGFCF)",
    )
    parser.add_argument("--k", type=int, default=64,
                        help="Spectral dimension used by SGFCF")
    parser.add_argument("--dataset", type=str, default="amazon-electronics",
                        help="Dataset name")
    parser.add_argument("--seed", type=int, default=2020,
                        help="Random seed")
    parser.add_argument("--gpu_ID", type=int, default=0,
                        help="GPU device index when CUDA is available")
    parser.add_argument("--item_index", type=int, default=0,
                        help="Item group ID to plot (zero-based; default: 0)")
    parser.add_argument("--n_user_groups", type=int, default=10,
                        help="Number of cosine-similarity user groups")
    parser.add_argument("--n_item_groups", type=int, default=8,
                        help="Number of cosine-similarity item groups")
    return parser.parse_args()
