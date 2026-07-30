import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    # Default hparams
    parser.add_argument('--dataset', type=str,default='msd')
    parser.add_argument('--model', type=str, default='EASE')
    parser.add_argument('--seed', type=int, default=2020, help='random seed')
    parser.add_argument('--gpu', type=int, default=-1, help="Please give a value for gpu id")
    parser.add_argument('--testbatch', type=int, default=4096, help="the batch size of users for testing")
    parser.add_argument('--topks', nargs='?',default="[20, 100]", help="@k test list, e.g. [10,20,30,40,50]")
    parser.add_argument('--multicore', type=int, default=0, help='whether we use multiprocessing or not in test')
    parser.add_argument('--parallel', type=eval, default=False, choices=[True, False])
   
    parser.add_argument('--reg_p', type=float, default=100)
    parser.add_argument('--diag_const', type=eval, default=True, choices=[True, False])
    parser.add_argument('--drop_p', type=float, default=0)
    parser.add_argument('--xi', type=float, default=0.0)
    parser.add_argument('--relax_type', type=str,default='relative')
    parser.add_argument('--alpha', type=float, default=0.0)
    parser.add_argument('--beta', type=float, default=0.0)
    parser.add_argument('--relax', type=eval, default=True, choices=[True, False])
    # return parser.parse_args(args=[])
    return parser.parse_args()