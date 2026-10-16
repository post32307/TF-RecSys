import argparse
import logging
import os
import random
import time

import numpy as np
import torch
import yaml


def get_args():
    parser = argparse.ArgumentParser(description="ChebyCF")
    parser.add_argument("--config", type=str)
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--comment", type=str)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--dataset", type=str, default="gowalla")
    parser.add_argument("--metrics", type=str, default="recall,ndcg")
    parser.add_argument("--top_ks", type=str, default="20")
    parser.add_argument("--model", type=str, default="chebycf")
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--phi", type=float, default=1.5)
    parser.add_argument("--eta", type=int, default=256)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--beta", type=float, default=0.4)
    args = parser.parse_args()

    if args.config:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "config",
            f"{args.config}.yml",
        )
        with open(config_path, mode="r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle.read())
        parser_defaults = {
            action.dest: action.default for action in parser._actions
        }
        for name, value in config.items():
            if hasattr(args, name) and getattr(args, name) == parser_defaults.get(name):
                setattr(args, name, value)

    def str_to_list(value, elem_type):
        if isinstance(value, str):
            return [elem_type(x.strip()) for x in value.split(",")]
        return list(value)

    args.metrics = str_to_list(args.metrics, str)
    args.top_ks = str_to_list(args.top_ks, int)
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        logging.warning("CUDA is unavailable; falling back to CPU.")
        args.device = "cpu"
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def set_logger(comment):
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "log")
    os.makedirs(log_dir, exist_ok=True)
    suffix = f"-{comment}" if comment else ""
    log_file = os.path.join(log_dir, f"{time.strftime('%y%m%d-%H%M%S')}{suffix}.log")
    file_handler = logging.FileHandler(log_file, "a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def log_args(args):
    logging.info("[SETTING]")
    for arg, value in vars(args).items():
        if value is not None:
            logging.info("%s: %s", arg, value)
    logging.info("")
