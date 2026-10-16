'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
'''
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Go lightGCN")
    parser.add_argument('--bpr_batch', type=int,default=2048,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--recdim', type=int,default=64,
                        help="the embedding size of lightGCN")
    parser.add_argument('--layer', type=int,default=3,
                        help="the layer num of lightGCN")
    parser.add_argument('--lr', type=float,default=0.001,
                        help="the learning rate")
    parser.add_argument('--decay', type=float,default=1e-4,
                        help="the weight decay for l2 normalizaton")
    parser.add_argument('--dropout', type=int,default=0,
                        help="using the dropout or not")
    parser.add_argument('--keepprob', type=float,default=0.6,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--a_fold', type=int,default=100,
                        help="the fold num used to split large adj matrix, like gowalla")
    parser.add_argument('--testbatch', type=int,default=100,
                        help="the batch size of users for testing")
    parser.add_argument('--dataset', type=str,default='lastfm',
                        help="available datasets: [lastfm, gowalla, yelp2018, amazon-book]")
    parser.add_argument('--path', type=str,default="./checkpoints",
                        help="path to save weights")
    parser.add_argument('--topks', nargs='?',default="[20]",
                        help="@k test list")
    parser.add_argument('--tensorboard', type=int,default=1,
                        help="enable tensorboard")
    parser.add_argument('--comment', type=str,default="lgn")
    parser.add_argument('--load', type=int,default=0)
    parser.add_argument('--epochs', type=int,default=1000)
    parser.add_argument('--multicore', type=int, default=0, help='whether we use multiprocessing or not in test')
    parser.add_argument('--pretrain', type=int, default=0, help='whether we use pretrained weight or not')
    parser.add_argument('--seed', type=int, default=2020, help='random seed')
    parser.add_argument('--model', type=str, default='lgn', help='rec-model, support [mf, lgn]')

    parser.add_argument('--search', type=int, default=0)
    parser.add_argument('--search_recdims', nargs='?', default="[32, 64, 128]",
                        help="embedding sizes to try when --search=1")
    parser.add_argument('--search_lrs', nargs='?', default="[0.001, 0.0001]",
                        help="learning rates to try when --search=1")
    parser.add_argument('--search_decays', nargs='?', default="[1e-4, 1e-5]",
                        help="weight decays to try when --search=1")
    parser.add_argument('--search_layers', nargs='?', default="[1, 2, 3, 4]",
                        help="LightGCN layers to try when --search=1")
    parser.add_argument('--early_stop', type=int, default=1,
                        help="enable early stopping on validation recall")
    parser.add_argument('--early_stop_patience', type=int, default=10,
                        help="validation checks without improvement before stopping")
    parser.add_argument('--early_stop_delta', type=float, default=1e-5,
                        help="minimum validation recall improvement")
    parser.add_argument('--early_stop_interval', type=int, default=10,
                        help="validate every N epochs")
    parser.add_argument('--early_stop_min_epochs', type=int, default=0,
                        help="minimum epoch before early stopping can trigger")
    parser.add_argument('--datasets', nargs='+', default=None,
                        help="datasets to run; use this for multi-dataset runs")
    parser.add_argument('--seeds', nargs='+', type=int, default=None,
                        help="seeds to run; use this for multi-seed runs")
    parser.add_argument('--concurrent_runs', type=int, default=1,
                        help="maximum number of concurrent single-run processes")
    parser.add_argument('--gpu_ids', nargs='+', default=None,
                        help="GPU IDs assigned round-robin to child runs; e.g. --gpu_ids 0 1 2 3")
    parser.add_argument('--single_run', type=int, default=0,
                        help="internal flag used by the multi-run scheduler")
    parser.add_argument('--child_result_file', type=str, default=None,
                        help="internal file used by child runs to report ordered results")
    parser.add_argument('--task_mode', type=str, default="full",
                        choices=["full", "val", "test"],
                        help="internal task type used by the multi-run scheduler")
    parser.add_argument('--task_recdim', type=int, default=None,
                        help="internal search-task embedding size")
    parser.add_argument('--task_lr', type=float, default=None,
                        help="internal search-task learning rate")
    parser.add_argument('--task_decay', type=float, default=None,
                        help="internal search-task weight decay")
    parser.add_argument('--task_layer', type=int, default=None,
                        help="internal search-task LightGCN layer count")

    return parser.parse_args()
