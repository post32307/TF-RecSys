import world
from world import cprint
import torch
torch.set_num_threads(4)
import numpy as np
import os
from time import time
from tensorboardX import SummaryWriter
import time
from os.path import join
from math import log
# ==============================
import helpers
helpers.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset

#1/2#Library################################################################
import scipy.sparse as sp
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch.nn.functional as F
from scipy.sparse.linalg import svds, LinearOperator
from scipy import sparse

import sys

import warnings
from scipy.sparse.linalg import MatrixRankWarning
warnings.filterwarnings("ignore", category=UserWarning, module="scipy.sparse.linalg")

# try:
#     from cppimport import imp_from_filepath
#     from os.path import join, dirname
#     path = join(dirname(__file__), "sources/sampling.cpp")
#     #print(path)
#     sampling = imp_from_filepath(path)
# except:
#     world.cprint("Cpp extension not loaded")
#     sample_ext = False
#################NEW CODE#################
import os
import sys
# 1. Add the 'sources' directory to the Python path so it can see the .so file
sources_path = os.path.join(os.path.dirname(__file__), "sources")
if sources_path not in sys.path:
    sys.path.append(sources_path)
sample_ext = False
try:
    # 2. Try to load the PRE-COMPILED .so file directly
    # import sampling
    from cppimport import imp_from_filepath
    from os.path import join, dirname
    path = join(dirname(__file__), "sources/sampling.cpp")
    sampling = imp_from_filepath(path)

    sampling.seed(world.seed)
    sample_ext = True
    world.cprint("✅ Success: Pre-compiled C++ extension loaded (.so)")
except ImportError:
    # 3. Only if the .so is missing, try to compile from source (for servers with g++)
    try:
        from cppimport import imp_from_filepath
        path = os.path.join(sources_path, "sampling.cpp")
        sampling = imp_from_filepath(path)
        sampling.seed(world.seed)
        sample_ext = True
        world.cprint("🔨 Loaded via cppimport (compiled on the fly)")
    except Exception as e:
        world.cprint(f"❌ Cpp extension not loaded: {e}")
        sample_ext = False
#################NEW CODE#################


from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import eigs

from sklearn.preprocessing import MaxAbsScaler

import itertools
############################################################################

#2/2#Tools##################################################################
# import analysis_tools
# import PSEs
import copy

class WrappedDataset:
    def __init__(self, original_dataset, new_test_dict, R_train):
        self.dataset = original_dataset
        self.testDict = new_test_dict
        self.R_train = R_train
    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            # R_train is CSR, so we get the col indices for the given row
            posItems.append(self.R_train[user].nonzero()[1])
        return posItems
    def __getattr__(self, name):
        return getattr(self.dataset, name)

def split_matrix(matrix, proportion):
    num_users = matrix.shape[0]
    train_rows, train_cols = [], []
    val_rows, val_cols = [], []
    
    rng = np.random.default_rng(world.seed)
    
    for u in range(num_users):
        start = matrix.indptr[u]
        end = matrix.indptr[u+1]
        items = matrix.indices[start:end]
        if len(items) == 0:
            continue
        
        indices = np.arange(len(items))
        rng.shuffle(indices)
        split_idx = int(len(items) * proportion)
        
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        train_rows.extend([u] * len(train_indices))
        train_cols.extend(items[train_indices])
        val_rows.extend([u] * len(val_indices))
        val_cols.extend(items[val_indices])
        
    R_train = sp.csr_matrix((np.ones(len(train_rows)), (train_rows, train_cols)), shape=matrix.shape)
    R_val = sp.csr_matrix((np.ones(len(val_rows)), (val_rows, val_cols)), shape=matrix.shape)
    
    val_dict = {}
    for r, c in zip(val_rows, val_cols):
        if r not in val_dict: val_dict[r] = []
        val_dict[r].append(c)
        
    return R_train, R_val, val_dict
############################################################################



import model
Model = model.PSEmodel(world.config, dataset)
weight_file = helpers.getFileName()
print(f"load and save to {weight_file}")

if world.LOAD:
    try:
        Model.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
        world.cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")
Neg_k = 1

# init tensorboard
if world.tensorboard:
    w : SummaryWriter = SummaryWriter(
                                    join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
                                    )
else:
    w = None
    world.cprint("not enable tensorflowboard")

try:
    print("k:",Model.k)

    A = torch.sparse_csr_tensor(
        torch.from_numpy(Model.A.indptr),
        torch.from_numpy(Model.A.indices),
        torch.from_numpy(Model.A.data).to(torch.float32),
        size=Model.A.shape).to(Model.device)

    print()
    
    X = Model.A
    U, Eigs = Model.Eigendecomposition(X)
    print("U_A:",Model.Test(U,dataset))

    X = Model.getNormAdj(Model.A,Model.D_inv,0.5,0.5)
    U, Eigs = Model.Eigendecomposition(X)
    print("U_Atilde:",Model.Test(U,dataset))

    X = Model.getNormAdj(Model.A,Model.D_inv,0.2,0.4)
    U, Eigs = Model.Eigendecomposition(X)
    print("U_Anorm:",Model.Test(U,dataset))
    print("U_Anorm_sqrtEig:",Model.Test(U@torch.diag(Eigs**0.5),dataset))
    world.cprint(" ")

    # from CodesToRemove import model as model
    # Recmodel = model.GeneralEmbModel(world.config, dataset)
    # Recmodel = Recmodel.to(world.device)
    # from CodesToRemove import PSEs
    # k=Model.k
    # print(k)
    # v0 = np.random.default_rng(seed=world.seed).standard_normal(Recmodel.A.shape[0])
    # U_low, Eigs_low = PSEs.get_U_eigs_full(Recmodel.A, k, v0, 'LR')
    # E = U_low[:,:k]
    # results = analysis_tools.Test_PSE(E, dataset, Recmodel, epoch=1)# Bash used Recmodel.dataset
    # print (results)
    # E = U_low[:,:k] @ torch.diag(torch.from_numpy(Eigs_low[:k]))
    # results = analysis_tools.Test_PSE(E, dataset, Recmodel, epoch=1)# Bash used Recmodel.dataset
    # print (results)

    # world.cprint(" ")

    # Base Data for PSEs
    # A = Recmodel.A
    # R = Recmodel.R
    # D = Recmodel.D
    # D_inv = Recmodel.D_inv
    # k = Recmodel.latent_dim
    # num_users = Recmodel.num_users
    # num_items = Recmodel.num_items

    # # ----------------------------------------
    # # Hyperparameter Search (Tuning)
    # # ----------------------------------------
    # # world.cprint(f"Starting Hyperparameter Search Stage ...")
    # proportion = 0.8 # Use 20% of data for hyperparameter search
    # R_train, R_val, val_dict = split_matrix(Recmodel.R, proportion)
    # A_train = sp.bmat([[None, R_train], [R_train.T, None]], format='csr')
    # val_dataset = WrappedDataset(Recmodel.dataset, val_dict, R_train)
    # # print(R_train[0])
    # # A_train = A
    
    # # # Degree Matrix for Train
    # # rowsum_train = np.array(A_train.sum(axis=1)).flatten()
    # # D_train = sp.diags(rowsum_train)
    # # rowsum_train[rowsum_train == 0] = 1 
    # # D_inv_train = sp.diags(1.0/rowsum_train)

    # # # NOTE : Search may result in lower final results
    # # best_recall = -1
    # # best_param = [0.2,0.4,128,1.0]
    # # for a1 in [0.1,0.2,0.3,0.4,0.5]:
    # #     for a2 in [0.1,0.2,0.3,0.4,0.5]:
    # #         X_train = sp.diags(D_inv_train.diagonal(), format='csr').power(a1) @ A_train @ sp.diags(D_inv_train.diagonal(), format='csr').power(a2)
    # #         v0 = np.random.default_rng(seed=world.seed).standard_normal(X_train.shape[0])
    # #         if world.dataset=="amazon-book":
    # #             U_low, Eigs_low = PSEs.get_U_eigs_full(X_train, 512, v0, 'LR')
    # #             for k in [32,64,128,256,512]:
    # #                 for b in [0.25,0.5,0.75,1.0]:
    # #                     E = U_low[:,:k] @ torch.diag(torch.from_numpy(Eigs_low[:k]))**b
    # #                     results = analysis_tools.Test_PSE(E, val_dataset, Recmodel, epoch=1)# Bash used Recmodel.dataset
    # #                     if results["recall"] >= best_recall:
    # #                         best_recall = results["recall"][0]
    # #                         best_param = [a1,a2,k,b]
    # #         else:
    # #             # U_low, Eigs_low = PSEs.get_U_eigs_full(X_train, 1024, v0, 'LR') # For full computation

    # #             # ####Test####
    # #             U_low, Eigs_low = PSEs.get_U_eigs_full(X_train, 128, v0, 'LR')
    # #             # ############

    # #             for k in [32,64,128,256,512,1024]:
    # #                 for b in [0.25,0.5,0.75,1.0]:

    # #                     # ####Test####
    # #                     if a1!=0.2 or a2!=0.4 or b!=1.0 or k>128: continue # Test
    # #                     # ############

    # #                     E = U_low[:,:k] @ torch.diag(torch.from_numpy(Eigs_low[:k]))**b
    # #                     results = analysis_tools.Test_PSE(E, val_dataset, Recmodel, epoch=1)# Bash used Recmodel.dataset

    # #                     # ####Test####
    # #                     print(a1,a2,k,b,results)
    # #                     # ############

    # #                     if results["recall"] > best_recall:
    # #                         # print(a1,a2,k,b)
    # #                         best_recall = results["recall"][0]
    # #                         best_param = [a1,a2,k,b]
    # # print("Best Params:",best_param,"Best Recall:",best_recall,"Best Results:",results)
    

    # # X = sp.diags(D_inv.diagonal(), format='csr').power(best_param[0]) @ A @ sp.diags(D_inv.diagonal(), format='csr').power(best_param[1])
    # # v0 = np.random.default_rng(seed=world.seed).standard_normal(X.shape[0])
    # # U_low, Eigs_low = PSEs.get_U_eigs_full(X, best_param[2], v0, 'LR') # Lowpass

    # # E = U_low @ torch.diag(torch.from_numpy(Eigs_low))**best_param[3]
    # # results_full = analysis_tools.Test_PSE(E, Recmodel.dataset, Recmodel, epoch=1)
    # # print("Best Params:",best_param,"Best :",results_full)

    # # res = f"{world.dataset}_(seed {world.seed}) : {best_param} / Results using split train: {results} Results using full train: {results_full}\n"
    # # with open("results.txt", "a") as f:
    # #     f.write(res)

finally:
    if world.tensorboard:
        w.close()