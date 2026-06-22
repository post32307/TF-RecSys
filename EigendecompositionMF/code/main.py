import world
from world import cprint
import torch
torch.set_num_threads(4)
import numpy as np
import os
import sys
from time import time
import time
from tensorboardX import SummaryWriter
from os.path import join
from math import log
# ==============================
import helpers
helpers.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset

sources_path = os.path.join(os.path.dirname(__file__), "sources")
if sources_path not in sys.path:
    sys.path.append(sources_path)
sample_ext = False
try:
    from cppimport import imp_from_filepath
    from os.path import join, dirname
    path = join(dirname(__file__), "sources/sampling.cpp")
    sampling = imp_from_filepath(path)

    sampling.seed(world.seed)
    sample_ext = True
    world.cprint("Success: Pre-compiled C++ extension loaded (.so)")
except ImportError:
    try:
        from cppimport import imp_from_filepath
        path = os.path.join(sources_path, "sampling.cpp")
        sampling = imp_from_filepath(path)
        sampling.seed(world.seed)
        sample_ext = True
        world.cprint("Loaded via cppimport (compiled on the fly)")
    except Exception as e:
        world.cprint(f"Cpp extension not loaded: {e}")
        sample_ext = False



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
    print()
    world.cprint("Results")

    print()
    X = Model.A
    U, Eigs = Model.Eigendecomposition(X)
    print("U_A:",Model.Test(U,dataset)['recall'])
    print("U_A_Eig:",Model.Test(U@torch.diag(Eigs**1.0),dataset)['recall'])
    print("U_A_sqrtEig:",Model.Test(U@torch.diag(Eigs**0.5),dataset)['recall'])
    print("U_A_sqrtEig:",Model.Test(U@torch.diag(abs(Eigs)**0.5*Eigs/abs(Eigs)),dataset)['recall'])
    
    print()
    X = Model.getNormAdj(Model.A,Model.D_inv,0.5,0.5)
    U, Eigs = Model.Eigendecomposition(X)
    print("U_Atilde:",Model.Test(U,dataset)['recall'])
    print("U_Atilde_Eig:",Model.Test(U@torch.diag(Eigs**1.0),dataset)['recall'])
    print("U_Atilde_sqrtEig:",Model.Test(U@torch.diag(Eigs**0.5),dataset)['recall'])
    print("U_A_sqrtEig:",Model.Test(U@torch.diag(abs(Eigs)**0.5*Eigs/abs(Eigs)),dataset)['recall'])

    print()
    X = Model.getNormAdj(Model.A,Model.D_inv,0.2,0.4)
    U, Eigs = Model.Eigendecomposition(X)
    print("U_Anorm:",Model.Test(U,dataset)['recall'])
    print("U_Anorm_Eig:",Model.Test(U@torch.diag(Eigs**1.0),dataset)['recall'])
    print("U_Anorm_sqrtEig:",Model.Test(U@torch.diag(Eigs**0.5),dataset)['recall'])
    print("U_A_sqrtEig:",Model.Test(U@torch.diag(abs(Eigs)**0.5*Eigs/abs(Eigs)),dataset)['recall'])

    print()
    world.cprint("Degree-based Comparison (Items)")
    item_degrees = np.array(dataset.UserItemNet.sum(axis=0)).squeeze()
    test_items = list(set([i for items in dataset.testDict.values() for i in items]))
    test_item_degrees = item_degrees[test_items]
    
    q25, q50, q75 = np.percentile(test_item_degrees, [25, 50, 75])
    
    high_degree_items = [i for i in test_items if item_degrees[i] >= q50]
    low_degree_items = [i for i in test_items if item_degrees[i] < q50]
    
    q1_items = [i for i in test_items if item_degrees[i] <= q25]
    q2_items = [i for i in test_items if q25 < item_degrees[i] <= q50]
    q3_items = [i for i in test_items if q50 < item_degrees[i] <= q75]
    q4_items = [i for i in test_items if item_degrees[i] > q75]
    
    print(f"Item degrees percentiles - 25th: {q25}, 50th: {q50}, 75th: {q75}")
    print(f"Low degree items count: {len(low_degree_items)}, High degree items count: {len(high_degree_items)}")
    print(f"Quantiles count: Q1: {len(q1_items)}, Q2: {len(q2_items)}, Q3: {len(q3_items)}, Q4: {len(q4_items)}")
    
    matrices = {
        'A': Model.A,
        'A_tilde': Model.getNormAdj(Model.A, Model.D_inv, 0.5, 0.5),
        'A_norm': Model.getNormAdj(Model.A, Model.D_inv, 0.2, 0.4)
    }
    
    for name, X_mat in matrices.items():
        print(f"\nEvaluating {name}...")
        U, Eigs = Model.Eigendecomposition(X_mat)
        
        configs = {
            'U': U,
            'U_Eig': U @ torch.diag(Eigs**1.0),
            'U_sqrtEig': U @ torch.diag(Eigs**0.5)
        }
        
        for cfg_name, emb in configs.items():
            recall_all = Model.Test(emb, dataset)['recall']
            recall_high = Model.Test(emb, dataset, items=high_degree_items)['recall']
            recall_low = Model.Test(emb, dataset, items=low_degree_items)['recall']
            recall_q1 = Model.Test(emb, dataset, items=q1_items)['recall']
            recall_q2 = Model.Test(emb, dataset, items=q2_items)['recall']
            recall_q3 = Model.Test(emb, dataset, items=q3_items)['recall']
            recall_q4 = Model.Test(emb, dataset, items=q4_items)['recall']
            print(f"  {cfg_name:10s} | All: {recall_all}")
            print(f"             | Low: {recall_low} | High: {recall_high}")
            print(f"             | Q1: {recall_q1} | Q2: {recall_q2} | Q3: {recall_q3} | Q4: {recall_q4}")

    sys.exit()
    
    print()
    world.cprint("Search")
    test_file_path = os.path.join(os.path.dirname(__file__), "test.txt")
    with open(test_file_path, "w") as f:
        f.write("Search\n")
        # for a in np.arange(0.1, 1.1, 0.1):
        #     for b in np.arange(0.1, 1.1, 0.1):
        for t in np.arange(0.1,2.1,0.1):
            for di in np.arange(-1.0,1.1,0.1):

                a= -di+t/2
                b = di+t/2

                if a==0 or b==0: continue

                X = Model.getNormAdj(Model.A,Model.D_inv,a,b)
                U, Eigs = Model.Eigendecomposition(X)
                
                recall = Model.Test(U,dataset)['recall']
                recall_eig = Model.Test(U@torch.diag(Eigs**1.0),dataset)['recall']
                recall_sqrtEig = Model.Test(U@torch.diag(Eigs**0.5),dataset)['recall']
                
                out_recall = f"a,b: {a:.2f} {b:.2f} / tau,del: {t:.2f} {di:.2f} : recall         = {recall}"
                out_eig = f"a,b: {a:.2f} {b:.2f} / tau,del: {t:.2f} {di:.2f} : recall_eig     = {recall_eig}"
                out_sqrtEig = f"a,b: {a:.2f} {b:.2f} / tau,del: {t:.2f} {di:.2f} : recall_sqrtEig = {recall_sqrtEig}"
                
                # print(out_recall)
                # print(out_eig)
                print(out_sqrtEig)
                
                f.write(out_recall + "\n")
                f.write(out_eig + "\n")
                f.write(out_sqrtEig + "\n")
                f.flush()

finally:
    if world.tensorboard:
        w.close()