import numpy as np
import pandas as pd
from collections import defaultdict


class Dataset:
    def __init__(self, dataset):
        self.dataset=dataset
        self.data_path = '../../data/{}'.format(dataset)
        train_file = 'data1.txt' if dataset == 'lastfm' else 'train.txt'
        test_file = 'test1.txt' if dataset == 'lastfm' else 'test.txt'

        def load_standard(filepath, is_lastfm):
            u_list, i_list = [], []
            try:
                with open(filepath, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 0: continue
                        u = int(parts[0])
                        if is_lastfm:
                            if len(parts) >= 2:
                                u_list.append(u - 1)
                                i_list.append(int(parts[1]) - 1)
                        else:
                            for i in parts[1:]:
                                u_list.append(u)
                                i_list.append(int(i))
            except FileNotFoundError:
                pass
            df = pd.DataFrame({'u': u_list, 'i': i_list, 'r': [1.0]*len(u_list), 't': [0]*len(u_list)})
            return df

        self.train_df = load_standard(self.data_path + '/' + train_file, dataset == 'lastfm')
        self.valid_df = pd.DataFrame({'u': [], 'i': [], 'r': [], 't': []})
        self.test_df = load_standard(self.data_path + '/' + test_file, dataset == 'lastfm')

        global_uid = set(set(self.train_df['u'].unique()).union(set(self.valid_df['u'].unique()))).union(set(self.test_df['u'].unique()))
        global_iid = set(set(self.train_df['i'].unique()).union(set(self.valid_df['i'].unique()))).union(set(self.test_df['i'].unique()))
        uid_mapping = dict(zip(global_uid, [i for i in range(len(global_uid))]))
        iid_mapping = dict(zip(global_iid, [i for i in range(len(global_iid))]))
        self.num_user, self.num_item = len(uid_mapping), len(iid_mapping)
        self.train_df['u'] = self.train_df['u'].map(uid_mapping)
        self.valid_df['u'] = self.valid_df['u'].map(uid_mapping)
        self.test_df['u'] = self.test_df['u'].map(uid_mapping)
        self.train_df['i'] = self.train_df['i'].map(iid_mapping)
        self.valid_df['i'] = self.valid_df['i'].map(iid_mapping)
        self.test_df['i'] = self.test_df['i'].map(iid_mapping)

    def get_init_mats(self):
        init_adj_mat = np.zeros((self.num_user, self.num_item))
        for ind in range(self.train_df.shape[0]):
            u, i, r = self.train_df['u'][ind], self.train_df['i'][ind], self.train_df['r'][ind]
            init_adj_mat[u][i] = 1.0
        return init_adj_mat

    def get_train_test_data(self):
        user_interacted_items_dict = defaultdict(list)
        valid_user_items_dict = defaultdict(list)
        test_user_items_dict = defaultdict(list)
        for ind_train in range(self.train_df.shape[0]):
            u, i = self.train_df['u'][ind_train], self.train_df['i'][ind_train]
            user_interacted_items_dict[u].append(i)
        for ind_valid in range(self.valid_df.shape[0]):
            u, i = self.valid_df['u'][ind_valid], self.valid_df['i'][ind_valid]
            valid_user_items_dict[u].append(i)
        for ind_test in range(self.test_df.shape[0]):
            u, i = self.test_df['u'][ind_test], self.test_df['i'][ind_test]
            test_user_items_dict[u].append(i)
        return user_interacted_items_dict, valid_user_items_dict, test_user_items_dict

