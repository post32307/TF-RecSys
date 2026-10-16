import argparse
import random
import time

import numpy as np

from dataloader import load_dataset
from model import DAN
from Procedure import evaluate


def parse_args():
    parser = argparse.ArgumentParser(description="Data-Adaptive Normalization")
    parser.add_argument("--dataset", type=str, default="gowalla")
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument(
        "--model",
        choices=sorted(DAN.SUPPORTED_MODELS),
        default="EASE_DAN",
    )
    parser.add_argument("--reg_p", type=float, default=20.0)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--drop_p", type=float, default=0.6)
    parser.add_argument("--xi", type=float, default=0.1)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument(
        "--solver",
        choices=["auto", "primal", "dual"],
        default="auto",
    )
    parser.add_argument("--block_size", type=int, default=256)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    train_matrix, eval_interactions, data_root, eval_filename = load_dataset(
        args.dataset, args.seed
    )
    print(f"Dataset: {args.dataset}")
    print(f"Data root: {data_root}")
    print(f"Evaluation file: {eval_filename}")
    print(
        f"Users: {train_matrix.shape[0]}, Items: {train_matrix.shape[1]}, "
        f"Train interactions: {train_matrix.nnz}"
    )
    print(f"Current model: {args.model}")

    started = time.perf_counter()
    model = DAN(
        model=args.model,
        reg_p=args.reg_p,
        alpha=args.alpha,
        beta=args.beta,
        drop_p=args.drop_p,
        xi=args.xi,
        solver=args.solver,
        block_size=args.block_size,
    ).fit(train_matrix)
    fit_time = time.perf_counter() - started

    started = time.perf_counter()
    recall, ndcg = evaluate(
        model,
        train_matrix,
        eval_interactions,
        k=args.top_k,
        batch_size=args.batch_size,
    )
    eval_time = time.perf_counter() - started
    print(f"Closed-form fit time: {fit_time:.4f}s")
    print(f"Solver: {model.selected_solver}")
    print(f"Evaluation time: {eval_time:.4f}s")
    print(f"Recall@{args.top_k}: {recall:.10f}")
    print(f"NDCG@{args.top_k}: {ndcg:.10f}")


if __name__ == "__main__":
    main()
