import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument('--ensemble_models', nargs='+',
                        default=['EASE', 'SGFCF'],
                        choices=['BSPM', 'EASE', 'GF-CF', 'PGSP',
                                 'TurboCF', 'SGFCF', 'ChebyCF', 'DAN'],
                        help='Models to include in the ensemble')

    parser.add_argument('--M', type=int, default=3,
                        help='Number of configurations retained per model')
    parser.add_argument('--dataset', type=str,default='lastfm',
                        help="available datasets: ['lastfm', 'gowalla', 'yelp2018', 'amazon-CD','amazon-electronics','citeulike','movielens', 'news', 'douban-book','epinions']")
    parser.add_argument('--seed', type=int, default=2020, help='random seed')
    parser.add_argument('--gpu_ID', type=int, default=0)    
    parser.add_argument('--k', type=int,default=64,
                        help="the embedding size")
    parser.add_argument('--topks', nargs='?',default="[20]",
                        help="@k test list")
    
    parser.add_argument('--params_filename', type=str, default='default_filename',
                        help='Parameter log filename (without .txt)')

    parser.add_argument('--load', type=str,default=False)
    parser.add_argument('--path', type=str,default="./checkpoints",
                        help="path to save weights")
    parser.add_argument('--tensorboard', type=int,default=1,
                        help="enable tensorboard")
    parser.add_argument('--comment', type=str, default="")
    # parser.add_argument('--model', type=str, default='pse-model')

    return parser.parse_args()
