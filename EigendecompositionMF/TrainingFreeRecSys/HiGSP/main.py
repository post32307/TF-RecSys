import random
from dataloader import Dataset
from model import *
import warnings
import argparse
import os
from warnings import simplefilter
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser(description='HiGSP')
    parser.add_argument('--dataset', default='ml100k', type=str)
    parser.add_argument('--alpha1', default=0.08, type=float)
    parser.add_argument('--alpha2', default=0.73, type=float)
    parser.add_argument('--order1', default=2, type=int)
    parser.add_argument('--order2', default=12, type=int)
    parser.add_argument('--pri_factor', default=80, type=int)
    parser.add_argument('--n_clusters', default=25, type=int)
    parser.add_argument('--seed', type=int, default=2020, help='Random seed')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ.setdefault(
        "TFRECSYS_DATA_ROOT",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "split_dataset", f"00_dataset_seed_{args.seed}")),
    )

    dataloader = Dataset(args.dataset)
    init_adj_mat = dataloader.get_init_mats()
    user_interacted_items_dict, valid_user_items_dict, test_user_items_dict = dataloader.get_train_test_data()

    lm = HiGSP(init_adj_mat,
               alpha1=args.alpha1, alpha2=args.alpha2, order1=args.order1, order2=args.order2, n_clusters=args.n_clusters, pri_factor=args.pri_factor,
               user_interacted_items_dict=user_interacted_items_dict)
    print('Train model...')
    ratings = lm.train()

    print('Evaluate model...')
    valid_recalls, valid_f1s, valid_mrrs, valid_ndcgs = lm.eval_test(valid_user_items_dict, ratings)  # for hyper-parameter tuning
    test_metrics = lm.eval_test(test_user_items_dict, ratings)

    (best_recalls, best_f1s, best_mrrs, best_ndcgs) = test_metrics
    test_metric_info = '\tRecall@10:{:.4f}\tF1@10:{:.4f}\tMRR@10:{:.4f}\tNDCG@10:{:.4f}\n' \
                       '\tRecall@20:{:.4f}\tF1@20:{:.4f}\tMRR@20:{:.4f}\tNDCG@20:{:.4f}'.format(
        best_recalls[0], best_f1s[0], best_mrrs[0], best_ndcgs[0], best_recalls[1], best_f1s[1], best_mrrs[1], best_ndcgs[1])
    print('[Test]\n{}'.format(test_metric_info))


if __name__ == '__main__':
    main()
