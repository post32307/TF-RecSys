"""
Define models here
"""
import torch
from dataloader import BasicDataset
import dataloader
from torch import nn
import numpy as np
from scipy import sparse
from time import time
from Procedure import get_valid_score


class BasicModel(nn.Module):    
    def __init__(self):
        super(BasicModel, self).__init__()
    
    def getUsersRating(self, users):
        raise NotImplementedError


class EASE(BasicModel):
    def __init__(self, config:dict, dataset:BasicDataset):
        super(EASE, self).__init__()

        self.dataset : dataloader.BasicDataset = dataset
        self.config = config
        self.num_users = dataset.n_users
        self.num_items = dataset.m_items
        
        self.reg_p = config['reg_p']
        self.diag_const = config['diag_const']
        
        self.__init_weight()
    
    def __init_weight(self):
        X = self.dataset.UserItemNet
        self.valid_matrix = self.dataset.validUserItemNet.tocsr()
        self.test_matrix = self.dataset.testUserItemNet.tocsr()
        self.best_epoch = 0

        train_start = time()
        G = np.array(X.T.dot(X).toarray())
        G[np.diag_indices(self.num_items)] += self.reg_p
        P = np.linalg.inv(G)

        if self.diag_const:
            self.W = P / (-np.diag(P))
        else:
            self.W = P * -self.reg_p
        self.W[np.diag_indices(self.num_items)] = 0

        train_end = time()
        self.train_time = train_end - train_start
        print(f"costing {self.train_time}s for training")
        
        self.valid_ndcg, self.valid_undcg = get_valid_score(self, self.dataset)

    def getUsersRating(self, users):
        users = users.detach().cpu().numpy()

        input_matrix = np.array(self.test_matrix[users].toarray())
        eval_output = input_matrix @ self.W

        return torch.FloatTensor(eval_output)

    def getvalidUsersRating(self, users):
        users = users.detach().cpu().numpy()

        input_matrix = np.array(self.valid_matrix[users].toarray())
        eval_output = input_matrix @ self.W

        return torch.FloatTensor(eval_output)



class LAE_DAN(BasicModel):
    def __init__(self, config:dict, dataset:BasicDataset):
        super(LAE_DAN, self).__init__()

        self.dataset : dataloader.BasicDataset = dataset
        self.config = config
        self.num_users = dataset.n_users
        self.num_items = dataset.m_items
        
        self.reg_p = config['reg_p']
        self.alpha = 1 - config['alpha']
        self.beta = config['beta']
        self.drop_p = config['drop_p']
        self.gamma = 1
        self.__init_weight()
    
    def __init_weight(self):
        X = self.dataset.UserItemNet
        self.train_matrix = self.dataset.UserItemNet.astype(np.float32)

        self.valid_matrix = self.dataset.validUserItemNet.tocsr()
        self.test_matrix = self.dataset.testUserItemNet.tocsr()
        self.best_epoch = 0        
        train_start = time()

        item_counts = np.array(self.train_matrix.sum(axis=0))
        user_counts = np.array(self.train_matrix.sum(axis=1))

        X_T = X.multiply(np.power(user_counts, -self.beta)).T
        G = X_T.dot(X).toarray()
        lmbda = self.reg_p + self.drop_p / (1 - self.drop_p) * np.power(item_counts, 1)
        G[np.diag_indices(self.num_items)] += lmbda.reshape(-1)
        
        P = np.linalg.inv(G)
        B_DLAE = np.eye(self.num_items) - P * lmbda
        item_power_term = np.power(item_counts, -(1 - self.alpha))
    
        self.W = B_DLAE * (1/item_power_term).reshape(-1, 1) * item_power_term
        self.W[np.diag_indices(self.num_items)] = 0
        self.valid_ndcg, self.valid_undcg = get_valid_score(self, self.dataset)

        train_end = time()
        self.train_time = train_end - train_start

    def getUsersRating(self, users):
        users = users.detach().cpu().numpy()

        input_matrix = np.array(self.test_matrix[users].toarray())
        eval_output = input_matrix @ self.W

        return torch.FloatTensor(eval_output)

    def getvalidUsersRating(self, users):
        users = users.detach().cpu().numpy()

        input_matrix = np.array(self.valid_matrix[users].toarray())
        eval_output = input_matrix @ self.W

        return torch.FloatTensor(eval_output)


class EASE_DAN(BasicModel):
    def __init__(self, config:dict, dataset:BasicDataset):
        super(EASE_DAN, self).__init__()

        self.dataset : dataloader.BasicDataset = dataset
        self.config = config
        self.num_users = dataset.n_users
        self.num_items = dataset.m_items
        
        self.reg_p = config['reg_p']
        self.relax = config['relax']
        self.xi = config['xi']
        self.alpha = 1 - config['alpha']
        self.beta = config['beta']
        self.drop_p = config['drop_p']
        self.__init_weight()
    
    def __init_weight(self):
        X = self.dataset.UserItemNet
        self.train_matrix = self.dataset.UserItemNet.astype(np.float32)

        self.valid_matrix = self.dataset.validUserItemNet.tocsr()
        self.test_matrix = self.dataset.testUserItemNet.tocsr()
        self.best_epoch = 0        
        train_start = time()

        item_counts = np.array(self.train_matrix.sum(axis=0))
        user_counts = np.array(self.train_matrix.sum(axis=1))

        X_T = X.multiply(np.power(user_counts, -self.beta)).T
        G = X_T.dot(X).toarray()
        lmbda = self.reg_p + self.drop_p / (1 - self.drop_p) * np.power(item_counts, 1)
        G[np.diag_indices(self.num_items)] += lmbda.reshape(-1)
        
        P = np.linalg.inv(G)
        B_DLAE = np.eye(self.num_items) - P / np.diag(P)
        item_power_term = np.power(item_counts, -(1 - self.alpha))
    
        self.W = B_DLAE * (1/item_power_term).reshape(-1, 1) * item_power_term
        self.W[np.diag_indices(self.num_items)] = 0
        self.valid_ndcg, self.valid_undcg = get_valid_score(self, self.dataset)

        train_end = time()
        self.train_time = train_end - train_start

    def getUsersRating(self, users):
        users = users.detach().cpu().numpy()

        input_matrix = np.array(self.test_matrix[users].toarray())
        eval_output = input_matrix @ self.W

        return torch.FloatTensor(eval_output)

    def getvalidUsersRating(self, users):
        users = users.detach().cpu().numpy()

        input_matrix = np.array(self.valid_matrix[users].toarray())
        eval_output = input_matrix @ self.W

        return torch.FloatTensor(eval_output)



class RLAE_DAN(BasicModel):
    def __init__(self, config:dict, dataset:BasicDataset):
        super(RLAE_DAN, self).__init__()

        self.dataset : dataloader.BasicDataset = dataset
        self.config = config
        self.num_users = dataset.n_users
        self.num_items = dataset.m_items
        
        self.reg_p = config['reg_p']
        self.relax = config['relax']
        self.xi = config['xi']
        self.alpha = 1 - config['alpha']
        self.beta = config['beta']
        self.drop_p = config['drop_p']
        self.__init_weight()
    
    def __init_weight(self):
        X = self.dataset.UserItemNet
        self.train_matrix = self.dataset.UserItemNet.astype(np.float32)

        self.valid_matrix = self.dataset.validUserItemNet.tocsr()
        self.test_matrix = self.dataset.testUserItemNet.tocsr()
        self.best_epoch = 0        
        train_start = time()

        item_counts = np.array(self.train_matrix.sum(axis=0))
        user_counts = np.array(self.train_matrix.sum(axis=1))

        X_T = X.multiply(np.power(user_counts, -self.beta)).T
        G = X_T.dot(X).toarray()
        lmbda = self.reg_p + self.drop_p / (1 - self.drop_p) * np.power(item_counts, 1)
        G[np.diag_indices(self.num_items)] += lmbda.reshape(-1)
        
        P = np.linalg.inv(G)

        diag_P = np.diag(P)
        condition = (1 - lmbda * diag_P) > self.xi
        lagrangian = ((1 - self.xi) / diag_P - lmbda) * condition.astype(float)
        B_DLAE = np.eye(self.num_items) - P * (lmbda.reshape(-1) + lagrangian)
        item_power_term = np.power(item_counts, -(1 - self.alpha))
    
        self.W = B_DLAE * (1/item_power_term).reshape(-1, 1) * item_power_term
        self.W[np.diag_indices(self.num_items)] = 0
        self.valid_ndcg, self.valid_undcg = get_valid_score(self, self.dataset)

        train_end = time()
        self.train_time = train_end - train_start

    def getUsersRating(self, users):
        users = users.detach().cpu().numpy()

        input_matrix = np.array(self.test_matrix[users].toarray())
        eval_output = input_matrix @ self.W

        return torch.FloatTensor(eval_output)

    def getvalidUsersRating(self, users):
        users = users.detach().cpu().numpy()

        input_matrix = np.array(self.valid_matrix[users].toarray())
        eval_output = input_matrix @ self.W

        return torch.FloatTensor(eval_output)
