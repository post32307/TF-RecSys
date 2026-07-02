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
import dataloader

from tqdm import tqdm  # Standard progress bar library

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
    import scipy.sparse as sp
    import itertools
    # ---------------------------------------------------------
    # 1. Define Paths & Load Target Main Datasets
    # ---------------------------------------------------------
    data_root = os.environ.get(
        "TFRECSYS_DATA_ROOT",
        os.path.join(world.ROOT_PATH, "data", "split_dataset", f"00_dataset_seed_{world.seed}"),
    )
    dataset_name = world.dataset
    dataset_path = os.path.join(data_root, dataset_name)
    split_dir = dataset_path

    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")
    world.cprint(f"Loading Full Evaluation Dataset: {dataset_name}")
    previous_eval_file = os.environ.get("TFRECSYS_EVAL_FILE")
    os.environ["TFRECSYS_EVAL_FILE"] = "test.txt"
    try:
        test_dataset = dataloader.Loader(path=dataset_path)
    finally:
        if previous_eval_file is None:
            os.environ.pop("TFRECSYS_EVAL_FILE", None)
        else:
            os.environ["TFRECSYS_EVAL_FILE"] = previous_eval_file

    # ---------------------------------------------------------
    # DYNAMIC MULTI-SCALE RESOLUTION LAYOUT GENERATION
    # ---------------------------------------------------------
    val_true = 1
    #   0: skip parameter search and test with default values
    #   1: search on validation dataset, then test on test dataset
    #   2: search on test dataset, then test on test dataset
    max_k = world.config['k']  # e.g., 128
    k_interval = 64
    n = 1
    b = 8
    if b < 1:
        raise ValueError("b must be at least 1.")
    if val_true not in (0, 1, 2):
        raise ValueError("val_true must be 0, 1, or 2.")
    if val_true == 1 and not os.path.isfile(os.path.join(split_dir, "val.txt")):
        raise FileNotFoundError(f"Validation file not found: {os.path.join(split_dir, 'val.txt')}")
    print(f"n : {n} / b : {b} / val_true : {val_true}")
    
    default_tau = 0.6
    default_eps1 = 0.0
    default_eps2 = 0.0
    default_delta = 0.1
    default_beta = [1.0 for _ in range(b)]
    # Example for setting interval-specific default betas:
    # default_beta[0]=-0.5
    # default_beta[1]=-0.5
    # default_beta[2]=-0.5
    # default_beta[3]=-0.5
    # default_beta[4]=0.5
    # default_beta[5]=1
    if len(default_beta) != b:
        raise ValueError("default_beta must have exactly b values.")
    if any(default_beta[idx] > default_beta[idx + 1] for idx in range(b - 1)):
        raise ValueError("default_beta values must be nondecreasing across intervals.")

    k_dimensions = []
    current_dim = max_k
    current_n = n
    while current_dim >= 32 and current_n > 0:
        k_dimensions.append(current_dim)
        current_dim -= k_interval
        current_n -= 1
        
    num_scales = len(k_dimensions)
    world.cprint(f"Dynamic Scale Subspaces Enabled: {k_dimensions} (N={num_scales} distinct delta steps)")

    # ---------------------------------------------------------
    # 2. Multi-Scale Search Loop with Sliced Subspaces & Separate Deltas
    # ---------------------------------------------------------
    eps_space = [0.0] # [0.0, 0.1, 0.3] # [0.0, 0.1, 0.3, 0.5] # Space for epsilon_1, epsilon_2
    alpha_space_1 = [0.6] # [0.4, 0.6, 0.8]  # Space for tau
    alpha_space_2 = [0.1] # [0.1, 0.2, 0.3]  # Space for delta
    beta_space = [-1.0, -0.5, 0.0, 0.5, 1.0] # [-1.0, -0.5, 0.0, 0.5, 1.0] # Space for betas (ustom structural eigenvalue exponent filter space)
    #     tau and delta control the degrees in asymmetric normalization
    #     epsilon controls the smoothness of degree operations
    #     beta controls the eigenvalues
    
    best_recall = -1.0
    # Store dynamic keys dynamically based on scale size index
    best_params = {"tau": default_tau, "eps1": default_eps1, "eps2": default_eps2}
    for idx in range(num_scales):
        best_params[f"delta{idx+1}"] = default_delta
        best_params[f"mask{idx+1}"] = 1
        for interval_idx in range(b):
            best_params[f"beta{idx+1}_{interval_idx+1}"] = default_beta[interval_idx]
    best_val_results = None

    def signed_eigen_power(eigs, powers):
        return torch.sign(eigs) * torch.pow(torch.abs(eigs), powers)

    def prepare_eval_batches(current_dataset):
        u_batch_size = world.config['test_u_batch_size']
        testDict = current_dataset.testDict
        users = list(testDict.keys())
        if len(users) == 0:
            return [], 0

        u_batch_size = min(u_batch_size, max(1, len(users) // 10))
        eval_batches = []
        for batch_users in helpers.minibatch(users, batch_size=u_batch_size):
            eval_batches.append((
                batch_users,
                current_dataset.getUserPosItems(batch_users),
                [testDict[u] for u in batch_users],
                torch.Tensor(batch_users).long().to(world.device)
            ))
        return eval_batches, len(users)

    def test_factorized_rating(current_model, current_dataset, user_factors, item_factors, eval_batches=None, user_count=None):
        max_K = max(world.topks)
        results = {
            'precision': np.zeros(len(world.topks)),
            'recall': np.zeros(len(world.topks)),
            'ndcg': np.zeros(len(world.topks))
        }
        if eval_batches is None or user_count is None:
            eval_batches, user_count = prepare_eval_batches(current_dataset)
        if user_count == 0:
            return results

        with torch.no_grad():
            for batch_users, allPos, groundTrue, batch_users_gpu in eval_batches:
                rating = user_factors[batch_users_gpu, :] @ item_factors.t()

                exclude_index = []
                exclude_items = []
                for range_i, items_pos in enumerate(allPos):
                    exclude_index.extend([range_i] * len(items_pos))
                    exclude_items.extend(items_pos)
                rating[exclude_index, exclude_items] = -(1 << 10)

                _, rating_K = torch.topk(rating, k=max_K)
                batch_result = current_model.test_one_batch((rating_K.cpu(), groundTrue))
                results['recall'] += batch_result['recall']
                results['precision'] += batch_result['precision']
                results['ndcg'] += batch_result['ndcg']
                del rating, rating_K

        results['recall'] /= float(user_count)
        results['precision'] /= float(user_count)
        results['ndcg'] /= float(user_count)
        return results

    if val_true == 0:
        world.cprint("Skipping validation search and using default hyperparameters.")
    else:
        validation_start_time = time.time()
        val_dataset = None
        if val_true == 1:
            world.cprint(f"Using Validation File for Parameter Search: {dataset_name}")
            search_target_name = "Validation"
        else:
            world.cprint(f"Using Full Test Dataset for Parameter Search: {dataset_name}")
            search_target_name = "Test"

        previous_eval_file = os.environ.get("TFRECSYS_EVAL_FILE")
        try:
            os.environ["TFRECSYS_EVAL_FILE"] = "val.txt" if val_true == 1 else "test.txt"
            val_dataset = dataloader.Loader(path=dataset_path)
            val_eval_batches, val_user_count = prepare_eval_batches(val_dataset)

            val_model = model.PSEmodel(world.config, val_dataset)
            val_model.A = sp.bmat([[None, val_model.R], [val_model.R.T, None]], format='csr')
            base_rowsum = np.array(val_model.A.sum(axis=1)).flatten()
            base_rowsum[base_rowsum == 0] = 1e-9

            print(f"\nStarting Optimized Dynamic N-Scale Grid Search on Seed {world.seed} {search_target_name} Data...")
            print("-" * 80)
            beta_interval_choices = list(itertools.combinations_with_replacement(sorted(beta_space), b))
            total_iterations = (len(eps_space) ** 2) * len(alpha_space_1) * (len(alpha_space_2) ** num_scales) * (len(beta_interval_choices) ** num_scales) * (2 ** num_scales)
            
            with tqdm(total=total_iterations, desc="Validation Search Progress", unit="it") as pbar:
                for eps2 in eps_space:
                    rowsum = base_rowsum + eps2
                    val_model.D_inv = sp.diags(1. / rowsum)

                    for tau in alpha_space_1:
                        val_model.k = max_k
                        X_norm = val_model.getNormAdj(val_model.A, val_model.D_inv, alpha1=tau/2, alpha2=tau/2)
                        U_full, Eigs_full = val_model.Eigendecomposition(X_norm)

                        eig_cache = {}
                        for idx, current_k in enumerate(k_dimensions):
                            Eigs_slice = Eigs_full[:current_k]
                            interval_edges = np.linspace(0, current_k, b + 1, dtype=int)
                            for beta_tuple in beta_interval_choices:
                                individual_betas = torch.empty(current_k, dtype=torch.float32, device=val_model.device)
                                for interval_idx, beta_value in enumerate(beta_tuple):
                                    start_idx = interval_edges[interval_idx]
                                    end_idx = interval_edges[interval_idx + 1]
                                    individual_betas[start_idx:end_idx] = beta_value
                                eig_cache[(idx, beta_tuple)] = signed_eigen_power(Eigs_slice, individual_betas)

                        for eps1 in eps_space:
                            d_diagonal = base_rowsum + eps1
                            d_power_cache = {
                                delta: torch.tensor(d_diagonal ** delta, dtype=torch.float32, device=val_model.device)
                                for delta in alpha_space_2
                            }
                            scaled_slice_cache = {}
                            for idx, current_k in enumerate(k_dimensions):
                                U_base_slice = U_full[:, :current_k]
                                for delta, d_power in d_power_cache.items():
                                    U_scaled = U_base_slice * d_power.unsqueeze(1)
                                    scaled_slice_cache[(idx, delta)] = (
                                        U_scaled[:val_model.n_users, :],
                                        U_scaled[val_model.n_users:, :]
                                    )

                            for delta_tuple, beta_tuple_by_scale, mask_tuple in itertools.product(
                                itertools.product(alpha_space_2, repeat=num_scales),
                                itertools.product(beta_interval_choices, repeat=num_scales),
                                itertools.product([0, 1], repeat=num_scales)
                            ):
                                if sum(mask_tuple) == 0:
                                    pbar.update(1)
                                    continue

                                transformed_u_user_list = []
                                transformed_u_item_list = []
                                for idx, current_k in enumerate(k_dimensions):
                                    if mask_tuple[idx] == 0:
                                        continue

                                    current_delta = delta_tuple[idx]
                                    beta_tuple = beta_tuple_by_scale[idx]
                                    u_slice, i_slice = scaled_slice_cache[(idx, current_delta)]
                                    filtered_eigs = eig_cache[(idx, beta_tuple)]
                                    transformed_u_user_list.append(u_slice * filtered_eigs.unsqueeze(0))
                                    transformed_u_item_list.append(i_slice * filtered_eigs.unsqueeze(0))

                                U_user_final = torch.cat(transformed_u_user_list, dim=1)
                                U_item_final = torch.cat(transformed_u_item_list, dim=1)

                                current_results = test_factorized_rating(
                                    val_model, val_dataset, U_user_final, U_item_final,
                                    val_eval_batches, val_user_count
                                )
                                current_recall = current_results['recall'][0]

                                if current_recall > best_recall:
                                    best_recall = current_recall
                                    best_params["tau"] = tau
                                    best_params["eps1"] = eps1
                                    best_params["eps2"] = eps2
                                    for idx in range(num_scales):
                                        best_params[f"delta{idx+1}"] = delta_tuple[idx]
                                        best_params[f"mask{idx+1}"] = mask_tuple[idx]
                                        for interval_idx, beta_value in enumerate(beta_tuple_by_scale[idx]):
                                            best_params[f"beta{idx+1}_{interval_idx+1}"] = beta_value
                                    best_val_results = current_results

                                pbar.set_postfix({"Best Recall": f"{best_recall:.4f}", "Cur Recall": f"{current_recall:.4f}"})
                                pbar.update(1)

                                del U_user_final, U_item_final
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()

                        del U_full, Eigs_full, eig_cache
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

            print("-" * 80)
            print(f"GRID SEARCH ON {search_target_name.upper()} DATA COMPLETE")
            print("Best Configuration Discovered (with Piecewise Vectorized Eigenvalue Betas):")
            for k, v in best_params.items():
                print(f"  {k}: {v}")
            print(f"Validation Metrics -> {best_val_results}")
            print("-" * 80)
        finally:
            if previous_eval_file is None:
                os.environ.pop("TFRECSYS_EVAL_FILE", None)
            else:
                os.environ["TFRECSYS_EVAL_FILE"] = previous_eval_file

    if val_true != 0 and best_val_results is None:
        raise RuntimeError("Validation search did not find a valid configuration.")

    if val_true != 0:
        validation_total_runtime = time.time() - validation_start_time
        validation_hours = int(validation_total_runtime // 3600)
        validation_minutes = int((validation_total_runtime % 3600) // 60)
        validation_seconds = validation_total_runtime % 60
        print(
            f"Validation Total Runtime: {validation_total_runtime:.2f} seconds "
            f"({validation_hours:02d}:{validation_minutes:02d}:{validation_seconds:05.2f})"
        )

    # ---------------------------------------------------------
    # 3. Out-Of-Sample Evaluation Using the Best Split Configurations
    # ---------------------------------------------------------
    world.cprint(f"\nEvaluating Best Hyperparameters on the Main Dataset Layer...")
    
    eval_model = model.PSEmodel(world.config, test_dataset)
    eval_model.A = sp.bmat([[None, eval_model.R], [eval_model.R.T, None]], format='csr')
    eval_base_rowsum = np.array(eval_model.A.sum(axis=1)).flatten()
    eval_base_rowsum[eval_base_rowsum == 0] = 1e-9

    tau_opt = best_params['tau']
    eps1_opt = best_params['eps1']
    eps2_opt = best_params['eps2']

    eval_rowsum = eval_base_rowsum + eps2_opt
    eval_model.D_inv = sp.diags(1. / eval_rowsum)
    d_diagonal_eval = eval_base_rowsum + eps1_opt

    eval_model.k = max_k
    X_norm_eval = eval_model.getNormAdj(eval_model.A, eval_model.D_inv, alpha1=tau_opt/2, alpha2=tau_opt/2)
    U_full_eval, Eigs_full_eval = eval_model.Eigendecomposition(X_norm_eval)

    eval_u_user_list = []
    eval_u_item_list = []

    for idx, current_k in enumerate(k_dimensions):
        if best_params[f"mask{idx+1}"] == 0:
            continue
            
        opt_delta_scale = best_params[f"delta{idx+1}"]
        opt_beta_tuple = tuple(best_params[f"beta{idx+1}_{interval_idx+1}"] for interval_idx in range(b))
        
        U_eval_slice = U_full_eval[:, :current_k]
        Eigs_eval_slice = Eigs_full_eval[:current_k]
        
        d_power_eval = torch.tensor(d_diagonal_eval ** opt_delta_scale, dtype=torch.float32, device=eval_model.device)
        U_trans_scaled = U_eval_slice * d_power_eval.unsqueeze(1)
        
        u_eval_slice = U_trans_scaled[:eval_model.n_users, :]
        i_eval_slice = U_trans_scaled[eval_model.n_users:, :]
        
        eval_interval_edges = np.linspace(0, current_k, b + 1, dtype=int)
        eval_individual_betas = torch.empty(current_k, dtype=torch.float32, device=eval_model.device)
        for interval_idx, beta_value in enumerate(opt_beta_tuple):
            start_idx = eval_interval_edges[interval_idx]
            end_idx = eval_interval_edges[interval_idx + 1]
            eval_individual_betas[start_idx:end_idx] = beta_value
        
        filtered_eval_eigs = signed_eigen_power(Eigs_eval_slice, eval_individual_betas)
        
        u_eval_filtered = u_eval_slice * filtered_eval_eigs.unsqueeze(0)
        i_eval_filtered = i_eval_slice * filtered_eval_eigs.unsqueeze(0)
        
        eval_u_user_list.append(u_eval_filtered)
        eval_u_item_list.append(i_eval_filtered)

    U_user_eval_final = torch.cat(eval_u_user_list, dim=1)
    U_item_eval_final = torch.cat(eval_u_item_list, dim=1)

    with torch.no_grad():
        final_test_results = test_factorized_rating(eval_model, test_dataset, U_user_eval_final, U_item_eval_final)
        
        print("=" * 80)
        print(f"FINAL OUT-OF-SAMPLE TEST METRICS (MAIN DATASET : {world.dataset}):")
        print(f"Global Tuned Configuration (seed={world.seed}): {best_params}")
        print(f"Recall@20: {final_test_results['recall'][0]:.6f}")
        print(f"Precision: {final_test_results['precision'].tolist()}")
        print(f"NDCG:      {final_test_results['ndcg'].tolist()}")
        print("=" * 80)

    X_norm_eval = eval_model.getNormAdj(eval_model.A, eval_model.D_inv, alpha1=0.2, alpha2=0.4)
    U_full_eval_L, U_full_eval_R, Eigs_full_eval = eval_model.Eigendecomposition_Asym(X_norm_eval)
    print("Eigenvalues (first/last):",Eigs_full_eval[0].item(),Eigs_full_eval[-1].item())
    print("Eigenvalues (min/max):",torch.min(Eigs_full_eval).item(),torch.max(Eigs_full_eval).item())

    print()
    results = eval_model.Test_RatingMatrix(U_full_eval_R[:eval_model.n_users,:] @ U_full_eval_R[eval_model.n_users:,:].T, eval_model.dataset)
    print(f"E=U_R")
    print(f"Recall@20: {results['recall'][0]:.6f}")
    print(f"Precision: {results['precision'].tolist()}")
    print(f"NDCG:      {results['ndcg'].tolist()}")

    results = eval_model.Test_RatingMatrix(U_full_eval_R[:eval_model.n_users,:] @ torch.diag(Eigs_full_eval) @ U_full_eval_R[eval_model.n_users:,:].T, eval_model.dataset)
    print(f"E=U_R Lambda^0.5")
    print(f"Recall@20: {results['recall'][0]:.6f}")
    print(f"Precision: {results['precision'].tolist()}")
    print(f"NDCG:      {results['ndcg'].tolist()}")

    results = eval_model.Test_RatingMatrix(U_full_eval_R[:eval_model.n_users,:] @ torch.diag(Eigs_full_eval**2) @ U_full_eval_R[eval_model.n_users:,:].T, eval_model.dataset)
    print(f"E=U_R Lambda^1.0")
    print(f"Recall@20: {results['recall'][0]:.6f}")
    print(f"Precision: {results['precision'].tolist()}")
    print(f"NDCG:      {results['ndcg'].tolist()}")

finally:
    if world.tensorboard:
        w.close()
