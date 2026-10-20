"""Minimal runtime configuration for plot_signal.py."""

import os
import sys
from os.path import join
import multiprocessing

import torch

from plot_parse import parse_args


os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
args = parse_args()

ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
CODE_PATH = join(ROOT_PATH, "code")
DATA_PATH = join(ROOT_PATH, "data")
FILE_PATH = join(CODE_PATH, "checkpoints")
sys.path.append(join(CODE_PATH, "sources"))

config = {
    "k": args.k,
    "test_u_batch_size": 100,
    "device": torch.device(args.gpu_ID if torch.cuda.is_available() else "cpu"),
}
seed = args.seed
dataset = args.dataset
item_index = args.item_index
n_user_groups = args.n_user_groups
n_item_groups = args.n_item_groups
models = args.models
device = config["device"]
CORES = max(1, multiprocessing.cpu_count() // 2)
topks = [20]
model_name = "pse-model"


def cprint(message):
    print(f"\033[0;30;43m{message}\033[0m")

