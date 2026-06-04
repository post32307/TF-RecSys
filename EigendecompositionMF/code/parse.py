import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="")
    
    parser.add_argument('--k', type=int,default=128,
                        help="the embedding size")
    
    parser.add_argument('--dataset', type=str,default='yelp2018',
                        help="available datasets: [lastfm, gowalla, yelp2018, amazon-book, dummy_small, dummy_large, 'amazon-CD','amazon-electronics','citeulike','movielens', 'news', 'douban-book','epinions','yelp']")
    
    parser.add_argument('--load', type=str,default=False)

    parser.add_argument('--path', type=str,default="./checkpoints",
                        help="path to save weights")
    
    parser.add_argument('--topks', nargs='?',default="[20]",
                        help="@k test list")
    
    parser.add_argument('--tensorboard', type=int,default=1,
                        help="enable tensorboard")
    
    parser.add_argument('--comment', type=str, default="")
    
    parser.add_argument('--seed', type=int, default=2020, help='random seed')
    
    parser.add_argument('--model', type=str, default='pse-model')
    
    parser.add_argument('--gpu_ID', type=int, default=7)
    
    return parser.parse_args()
