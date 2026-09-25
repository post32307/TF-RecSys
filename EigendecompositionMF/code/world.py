import os
from os.path import join
import torch
from enum import Enum
from parse import parse_args
import multiprocessing

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
args = parse_args()

ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
CODE_PATH = join(ROOT_PATH, 'code')
DATA_PATH = join(ROOT_PATH, 'data')
BOARD_PATH = join(CODE_PATH, 'runs')
FILE_PATH = join(CODE_PATH, 'checkpoints')
import sys
sys.path.append(join(CODE_PATH, 'sources'))


if not os.path.exists(FILE_PATH):
    os.makedirs(FILE_PATH, exist_ok=True)


config = {}
all_dataset = ['lastfm', 'gowalla', 'yelp2018', 'amazon-book', 'dummy_small', 'dummy_large', 'amazon-CD','amazon-electronics','citeulike','movielens', 'news', 'douban-book','epinions','yelp']
all_models  = ['pse-model']
config['k'] = args.k
config['test_u_batch_size'] = 100#128
GPU = torch.cuda.is_available()
device = torch.device(args.gpu_ID if GPU else "cpu")
config['device'] = device
CORES = multiprocessing.cpu_count() // 2
seed = args.seed

dataset = args.dataset
model_name = args.model
input_model = args.input_model

params_filename = args.params_filename
if dataset not in all_dataset:
    raise NotImplementedError(f"Haven't supported {dataset} yet!, try {all_dataset}")
if model_name not in all_models:
    raise NotImplementedError(f"Haven't supported {model_name} yet!, try {all_models}")

LOAD = args.load
topks = eval(args.topks)
tensorboard = args.tensorboard
comment = args.comment

from warnings import simplefilter
simplefilter(action="ignore", category=FutureWarning)



def cprint(words : str):
    print(f"\033[0;30;43m{words}\033[0m")

logo = r"""
██╗      ██████╗ ███╗   ██╗
██║     ██╔════╝ ████╗  ██║
██║     ██║  ███╗██╔██╗ ██║
██║     ██║   ██║██║╚██╗██║
███████╗╚██████╔╝██║ ╚████║
╚══════╝ ╚═════╝ ╚═╝  ╚═══╝
"""
PSE = r"""
██████╗ ███████╗███████╗
██╔══██╗██╔════╝██╔════╝
██████╔╝███████╗█████╗  
██╔═══╝ ╚════██║██╔══╝  
██║     ███████║███████╗
╚═╝     ╚══════╝╚══════╝
"""
TF = r"""
████████╗██████╗  █████╗ ██╗███╗   ██╗██╗███╗   ██╗ ██████╗        ███████╗██████╗ ███████╗███████╗
╚══██╔══╝██╔══██╗██╔══██╗██║████╗  ██║██║████╗  ██║██╔════╝        ██╔════╝██╔══██╗██╔════╝██╔════╝
   ██║   ██████╔╝███████║██║██╔██╗ ██║██║██╔██╗ ██║██║  ███╗█████╗ █████╗  ██████╔╝█████╗  █████╗  
   ██║   ██╔══██╗██╔══██║██║██║╚██╗██║██║██║╚██╗██║██║   ██║╚════╝ ██╔══╝  ██╔══██╗██╔══╝  ██╔══╝  
   ██║   ██║  ██║██║  ██║██║██║ ╚████║██║██║ ╚████║╚██████╔╝       ██║     ██║  ██║███████╗███████╗
   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝╚═╝╚═╝  ╚═══╝ ╚═════╝        ╚═╝     ╚═╝  ╚═╝╚══════╝╚══════╝
"""
# font: ANSI Shadow
# refer to http://patorjk.com/software/taag/#p=display&f=ANSI%20Shadow&t=Sampling
# print(logo)
print(TF)
# print(PSE)
