import world
import torch
from dataloader import BasicDataset
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigs
import torch.nn.functional as F

import metrics
import helpers
import multiprocessing

class PSEmodel:
    def __init__(self, config: dict, dataset: BasicDataset):
        self.config = config
        self.dataset = dataset
        self.n_users = dataset.n_users
        self.m_items = dataset.m_items
        self.num_users = self.n_users
        self.num_items = self.m_items
        self.device = torch.device(config['device'])   
        self.k = config['k']
        self.Sigmoid = F.sigmoid

        self.R = self.dataset.UserItemNet.tocsr() #CSR
        self.A = sp.bmat([[None, self.R], [self.R.T, None]], format='csr') #CSR
        rowsum = np.array(self.A.sum(axis=1)).flatten()
        rowsum[rowsum == 0] = 1e-9
        self.D = sp.diags(rowsum) #CSR
        self.D_inv = sp.diags(1. / rowsum) #CSR

    # PSE Functions

    # 1.

    # 2.

    # 3.



    # Dataset Operation Functions

    def getNormAdj(self,A,D_inv,alpha1=0.2,alpha2=0.4):
        adj_norm = D_inv.power(alpha1) @ A @ D_inv.power(alpha2)
        return adj_norm


    # Embedding Functions

    def Eigendecomposition(self,X):
        if torch.is_tensor(X):
            X = X.cpu().numpy()
        Eigs,U = eigs(X, k=self.k,which='LR')
        U = U.real
        Eigs = Eigs.real
        U = torch.tensor(U, dtype=torch.float32).to(self.device)
        Eigs = torch.tensor(Eigs, dtype=torch.float32).to(self.device)
        U = torch.nn.functional.normalize(U, p=2, dim=0)
        return U,Eigs
    

    # Evaluation Functions

    def getUsersRating(self,X,users):
        X = X.to(self.device)
        users = users.long()
        users_emb = X[:self.num_users,:]
        items_emb = X[self.num_users:,:]
        users_emb = users_emb[users,:]
        items_emb = items_emb
        scores = torch.matmul(users_emb, items_emb.t())
        return self.Sigmoid(scores)

    def test_one_batch(self,X_groundTrue):
        sorted_items = X_groundTrue[0].numpy()
        groundTrue = X_groundTrue[1]
        r = metrics.getLabel(groundTrue, sorted_items)
        pre, recall, ndcg = [], [], []
        for k in world.topks:
            ret = metrics.RecallPrecision_ATk(groundTrue, r, k)
            pre.append(ret['precision'])
            recall.append(ret['recall'])
            ndcg.append(metrics.NDCGatK_r(groundTrue,r,k))
        return {'recall':np.array(recall), 
                'precision':np.array(pre), 
                'ndcg':np.array(ndcg)}

    def Test(self, X, dataset, users=None, items=None, multicore=0):
        if X.shape == (self.num_users, self.num_items):
            return self.Test_RatingMatrix(X, dataset, users, items, multicore)
        u_batch_size = world.config['test_u_batch_size']
        testDict: dict = dataset.testDict
        max_K = max(world.topks)
        results = {'precision': np.zeros(len(world.topks)),
                'recall': np.zeros(len(world.topks)),
                'ndcg': np.zeros(len(world.topks))}
        # print(multicore)
        if multicore == 1:
            pool = multiprocessing.Pool(world.CORES)
        with torch.no_grad():
            if users is None:
                users = list(testDict.keys())
                
            if items is not None:
                items_set = set(items)
                valid_users = []
                for u in users:
                    if len(items_set.intersection(testDict[u])) > 0:
                        valid_users.append(u)
                users = valid_users
                
            if len(users) == 0:
                if multicore == 1:
                    pool.close()
                return results
                
            u_batch_size = min(u_batch_size, max(1, len(users) // 10))
            try:
                assert u_batch_size <= len(users) / 10
            except AssertionError:
                print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
            users_list = []
            rating_list = []
            groundTrue_list = []
            total_batch = (len(users) + u_batch_size - 1) // u_batch_size if len(users) > 0 else 0
            for batch_users in helpers.minibatch(users, batch_size=u_batch_size):
                allPos = dataset.getUserPosItems(batch_users)
                if items is not None:
                    groundTrue = [list(items_set.intersection(testDict[u])) for u in batch_users]
                else:
                    groundTrue = [testDict[u] for u in batch_users]
                batch_users_gpu = torch.Tensor(batch_users).long()
                batch_users_gpu = batch_users_gpu.to(world.device)
                rating = self.getUsersRating(X,batch_users_gpu)
                exclude_index = []
                exclude_items = []
                for range_i, items_pos in enumerate(allPos):
                    exclude_index.extend([range_i] * len(items_pos))
                    exclude_items.extend(items_pos)
                rating[exclude_index, exclude_items] = -(1<<10)
                
                
                _, rating_K = torch.topk(rating, k=max_K)
                rating = rating.cpu().numpy()
                del rating
                users_list.append(batch_users)
                rating_list.append(rating_K.cpu())
                groundTrue_list.append(groundTrue)
            assert total_batch == len(users_list)
            X_zip = zip(rating_list, groundTrue_list)
            if multicore == 1:
                pre_results = pool.map(self.test_one_batch, X_zip)
            else:
                pre_results = []
                for x in X_zip:
                    pre_results.append(self.test_one_batch(x))
            scale = float(u_batch_size/len(users))
            for result in pre_results:
                results['recall'] += result['recall']
                results['precision'] += result['precision']
                results['ndcg'] += result['ndcg']
            results['recall'] /= float(len(users))
            results['precision'] /= float(len(users))
            results['ndcg'] /= float(len(users))
            
            if multicore == 1:
                pool.close()
            return results
    


    def Test_RatingMatrix(self, X, dataset, users=None, items=None, multicore=0):
        u_batch_size = world.config['test_u_batch_size']
        testDict: dict = dataset.testDict
        max_K = max(world.topks)
        results = {'precision': np.zeros(len(world.topks)),
                'recall': np.zeros(len(world.topks)),
                'ndcg': np.zeros(len(world.topks))}
        if multicore == 1:
            pool = multiprocessing.Pool(world.CORES)
        with torch.no_grad():
            if users is None:
                users = list(testDict.keys())
                
            if items is not None:
                items_set = set(items)
                valid_users = []
                for u in users:
                    if len(items_set.intersection(testDict[u])) > 0:
                        valid_users.append(u)
                users = valid_users
                
            if len(users) == 0:
                if multicore == 1:
                    pool.close()
                return results

            u_batch_size = min(u_batch_size, max(1, len(users) // 10))
            try:
                assert u_batch_size <= len(users) / 10
            except AssertionError:
                print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
            users_list = []
            rating_list = []
            groundTrue_list = []
            total_batch = (len(users) + u_batch_size - 1) // u_batch_size if len(users) > 0 else 0
            for batch_users in helpers.minibatch(users, batch_size=u_batch_size):
                allPos = dataset.getUserPosItems(batch_users)
                if items is not None:
                    groundTrue = [list(items_set.intersection(testDict[u])) for u in batch_users]
                else:
                    groundTrue = [testDict[u] for u in batch_users]
                batch_users_gpu = torch.Tensor(batch_users).long()
                batch_users_gpu = batch_users_gpu.to(world.device)
                rating = X[batch_users_gpu]
                exclude_index = []
                exclude_items = []
                for range_i, items_pos in enumerate(allPos):
                    exclude_index.extend([range_i] * len(items_pos))
                    exclude_items.extend(items_pos)
                rating[exclude_index, exclude_items] = -(1<<10)
                
                
                _, rating_K = torch.topk(rating, k=max_K)
                rating = rating.cpu().numpy()
                del rating
                users_list.append(batch_users)
                rating_list.append(rating_K.cpu())
                groundTrue_list.append(groundTrue)
            assert total_batch == len(users_list)
            X_zip = zip(rating_list, groundTrue_list)
            if multicore == 1:
                pre_results = pool.map(self.test_one_batch, X_zip)
            else:
                pre_results = []
                for x in X_zip:
                    pre_results.append(self.test_one_batch(x))
            for result in pre_results:
                results['recall'] += result['recall']
                results['precision'] += result['precision']
                results['ndcg'] += result['ndcg']
            results['recall'] /= float(len(users))
            results['precision'] /= float(len(users))
            results['ndcg'] /= float(len(users))
            
            if multicore == 1:
                pool.close()
            return results