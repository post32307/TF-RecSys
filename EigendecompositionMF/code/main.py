import world
from world import cprint
import torch
torch.set_num_threads(4)
import numpy as np
import os
import shutil
import sys
import tempfile
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
    import itertools

    def arbitrary_fifth_order_filter(lambda_values, w0=1.0, w1=1.0, w2=0.0, w3=0.0, w4=0.0, w5=0.0):
        """
        Applies an arbitrary 5th-order polynomial filter.
        f(lam) = w0 + w1*lam + w2*lam^2 + w3*lam^3 + w4*lam^4 + w5*lam^5
        """
        lam = np.asarray(lambda_values)
        
        # Scale to [-1, 1] to keep polynomial powers numerically stable
        max_mag = np.max(np.abs(lam))
        if max_mag > 0:
            lam = lam / max_mag
            
        return (w0 + 
                w1 * lam + 
                w2 * (lam ** 2) + 
                w3 * (lam ** 3) + 
                w4 * (lam ** 4) + 
                w5 * (lam ** 5))

    data_root = os.path.join(world.ROOT_PATH, "data")
    dataset_name = world.dataset
    dataset_path = os.path.join(data_root, dataset_name)
    world.cprint("Exact Arbitrary 5th-Order Polynomial Search")
    
    if dataset_name == "lastfm":
        dataset = dataloader.LastFM(path=dataset_path)
    else:
        dataset = dataloader.Loader(path=dataset_path)
    
    model = model.PSEmodel(world.config, dataset)
    X_cheb = model.getNormAdj(model.A, model.D_inv, 0.2, 0.4)
    U_L, U_R, Eigs = model.Eigendecomposition_Asym(X_cheb)

    # Pre-extract the numpy array to avoid redundant operations inside the loop
    eigs_np = Eigs.detach().cpu().numpy()
    
    # Fast search space to avoid 500k+ loops (5^6 = 15,625 iterations)
    coeff_space = [-1.0, -0.5, 0.0, 0.5, 1.0]
    
    # UNCOMMENT BELOW IF YOU PREFER YOUR ORIGINAL 9-ELEMENT ARRAY (9^6 = 531,441 iterations):
    # coeff_space = [-5.0, -3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0, 5.0]
    
    best_param_1 = [0, 0, 0, 0, 0, 0]
    best_result_1 = {"recall": -10000.0}
    best_param_2 = [0, 0, 0, 0, 0, 0]
    best_result_2 = {"recall": -10000.0}
    
    print("\nStarting Grid Search for Arbitrary 5th-Order Polynomial...")
    
    with torch.no_grad():
        # itertools.product creates every combination of w0, w1, w2, w3, w4, w5
        for w0, w1, w2, w3, w4, w5 in itertools.product(coeff_space, repeat=6):
            # Skip the all-zero filter to save computation
            if w0 == 0.0 and w1 == 0.0 and w2 == 0.0 and w3 == 0.0 and w4 == 0.0 and w5 == 0.0:
                continue            
                
            exact_f_Eigs_Arb = torch.from_numpy(
                arbitrary_fifth_order_filter(eigs_np, w0=w0, w1=w1, w2=w2, w3=w3, w4=w4, w5=w5)
            ).to(world.device)        
            
            Exact_Arb_Matrix = U_R @ torch.diag(exact_f_Eigs_Arb)
            S = U_R @ torch.diag(exact_f_Eigs_Arb) @ U_R.T
            
            # Evaluate the filter
            result_1 = model.Test(Exact_Arb_Matrix, model.dataset)
            result_2 = model.Test_RatingMatrix(S, model.dataset)
            
            # Track best parameters
            if best_result_1["recall"] < result_1["recall"]:
                best_result_1 = result_1
                best_param_1 = [w0, w1, w2, w3, w4, w5]
            if best_result_2["recall"] < result_2["recall"]:
                best_result_2 = result_2
                best_param_2 = [w0, w1, w2, w3, w4, w5]
            del Exact_Arb_Matrix, S
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            print(f"Coeff {(w0, w1, w2, w3, w4, w5)} : {result_1['recall']}")
            print(f"Coeff {(w0, w1, w2, w3, w4, w5)} : {result_2['recall']}")
            # break
        
    print("-" * 100)
    print("BEST RESULT (embedding):", best_result_1)
    print("BEST PARAM (embedding):", best_param_1)
    print("-" * 70)
    print("BEST RESULT (rating):", best_result_2)
    print("BEST PARAM (rating):", best_param_2)
    print("-" * 100)

    sys.exit()
    
    

    print()
    world.cprint("Validation Search and Final Test")

    data_root = os.path.join(world.ROOT_PATH, "data")
    search_root = os.path.join(data_root, "00_search_sets_seed_2026")
    val_results_file = os.path.join(world.CODE_PATH, "validation_search_results_seed_2026.txt")
    test_results_file = os.path.join(world.CODE_PATH, "test_results_seed_2026.txt")
    k_values = [16, 32, 64, 128, 256, 512]   # [32,64,128,256,512,1024]
    search_step = 0.1

    def link_or_copy(src, dst):
        try:
            os.symlink(src, dst)
        except OSError:
            shutil.copyfile(src, dst)

    def make_dataset(dataset_name, split):
        world.dataset = dataset_name
        if split == "validation":
            split_dir = os.path.join(search_root, dataset_name)
            with tempfile.TemporaryDirectory(prefix=f"pse-{dataset_name}-val-") as tmpdir:
                if dataset_name == "lastfm":
                    link_or_copy(os.path.join(split_dir, "train_val.txt"), os.path.join(tmpdir, "data1.txt"))
                    link_or_copy(os.path.join(split_dir, "validation_val.txt"), os.path.join(tmpdir, "test1.txt"))
                    link_or_copy(
                        os.path.join(data_root, "lastfm", "trustnetwork.txt"),
                        os.path.join(tmpdir, "trustnetwork.txt")
                    )
                    return dataloader.LastFM(path=tmpdir)
                link_or_copy(os.path.join(split_dir, "train_val.txt"), os.path.join(tmpdir, "train.txt"))
                link_or_copy(os.path.join(split_dir, "validation_val.txt"), os.path.join(tmpdir, "test.txt"))
                return dataloader.Loader(path=tmpdir)

        dataset_dir = os.path.join(data_root, dataset_name)
        if dataset_name == "lastfm":
            return dataloader.LastFM(path=dataset_dir)
        return dataloader.Loader(path=dataset_dir)

    def selected_dataset():
        dataset_name = world.dataset
        dataset_dir = os.path.join(data_root, dataset_name)
        split_dir = os.path.join(search_root, dataset_name)
        if not os.path.isdir(dataset_dir):
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"Validation split directory not found: {split_dir}")
        return dataset_name

    def build_components(current_model, component_k_values):
        original_k = current_model.k
        valid_k_values = [
            k for k in component_k_values
            if k < current_model.num_users + current_model.num_items - 1
        ]
        if not valid_k_values:
            raise ValueError("No valid k values for this dataset")

        max_k = max(valid_k_values)
        current_model.k = max_k
        X = current_model.getNormAdj(current_model.A, current_model.D_inv, 0.2, 0.4)
        U, Eigs = current_model.Eigendecomposition(X)
        eig_order = torch.argsort(Eigs, descending=True)
        U = U[:, eig_order]
        Eigs = Eigs[eig_order]

        components = []
        for k_value in valid_k_values:
            U_k = U[:, :k_value]
            Eigs_k = Eigs[:k_value]
            user_U = U_k[:current_model.num_users, :]
            item_U = U_k[current_model.num_users:, :]
            components.append((user_U, Eigs_k, item_U))
            print(f"  S{len(components)} constructed with k={k_value}")

        current_model.k = original_k
        return components, valid_k_values

    def evaluate_weights(current_model, current_dataset, components, weights):
        u_batch_size = world.config['test_u_batch_size']
        testDict = current_dataset.testDict
        max_K = max(world.topks)
        results = {
            'precision': np.zeros(len(world.topks)),
            'recall': np.zeros(len(world.topks)),
            'ndcg': np.zeros(len(world.topks))
        }
        users = list(testDict.keys())
        if len(users) == 0:
            return results
        u_batch_size = min(u_batch_size, max(1, len(users) // 10))
        rating_list = []
        groundTrue_list = []

        with torch.no_grad():
            for batch_users in helpers.minibatch(users, batch_size=u_batch_size):
                allPos = current_dataset.getUserPosItems(batch_users)
                groundTrue = [testDict[u] for u in batch_users]
                batch_users_gpu = torch.Tensor(batch_users).long().to(world.device)
                rating = None

                for weight, (user_U, eigs, item_U) in zip(weights, components):
                    if weight == 0:
                        continue
                    component_rating = (user_U[batch_users_gpu, :] * eigs) @ item_U.t()
                    if rating is None:
                        rating = weight * component_rating
                    else:
                        rating = rating + weight * component_rating

                if rating is None:
                    rating = torch.zeros((len(batch_users), current_model.num_items), device=world.device)

                exclude_index = []
                exclude_items = []
                for range_i, items_pos in enumerate(allPos):
                    exclude_index.extend([range_i] * len(items_pos))
                    exclude_items.extend(items_pos)
                rating[exclude_index, exclude_items] = -(1 << 10)

                _, rating_K = torch.topk(rating, k=max_K)
                rating_list.append(rating_K.cpu())
                groundTrue_list.append(groundTrue)
                del rating

        for x in zip(rating_list, groundTrue_list):
            result = current_model.test_one_batch(x)
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']

        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))
        return results

    def generate_weight_indices(total, parts):
        if parts == 1:
            yield [total]
            return
        for value in range(total + 1):
            for rest in generate_weight_indices(total - value, parts - 1):
                yield [value] + rest

    def format_weights(weights):
        return ",".join(f"a{idx + 1}={weight:.2f}" for idx, weight in enumerate(weights))

    def format_metrics(metrics):
        return (
            f"precision={metrics['precision'].tolist()} "
            f"recall={metrics['recall'].tolist()} "
            f"ndcg={metrics['ndcg'].tolist()}"
        )

    def search_best_weights(current_model, current_dataset, components):
        grid_size = int(round(1.0 / search_step))
        best_recall = None
        best_weights = None
        best_metrics = None
        for weight_indices in generate_weight_indices(grid_size, len(components)):
            weights = [idx * search_step for idx in weight_indices]
            metrics = evaluate_weights(current_model, current_dataset, components, weights)
            recall_score = metrics['recall'][0]
            if best_recall is None or recall_score > best_recall:
                best_recall = recall_score
                best_weights = tuple(weights)
                best_metrics = metrics
            print(f"    {format_weights(weights)} | recall={metrics['recall']}")
        return best_weights, best_metrics

    dataset_name = selected_dataset()
    with open(val_results_file, "w") as val_file, open(test_results_file, "w") as test_file:
        val_file.write("dataset | k_values | best_weights | validation_metrics\n")
        test_file.write("dataset | k_values | best_weights | test_metrics\n")

        print()
        world.cprint(f"[VALIDATION] {dataset_name}")
        val_dataset = make_dataset(dataset_name, "validation")
        val_model = model.PSEmodel(world.config, val_dataset)
        val_components, valid_k_values = build_components(val_model, k_values)
        best_weights, best_val_metrics = search_best_weights(val_model, val_dataset, val_components)
        val_line = (
            f"{dataset_name} | {valid_k_values} | {format_weights(best_weights)} | "
            f"{format_metrics(best_val_metrics)}"
        )
        print(val_line)
        val_file.write(val_line + "\n")
        val_file.flush()

        del val_components, val_model, val_dataset
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        world.cprint(f"[TEST] {dataset_name}")
        test_dataset = make_dataset(dataset_name, "test")
        test_model = model.PSEmodel(world.config, test_dataset)
        test_components, test_k_values = build_components(test_model, valid_k_values)
        test_metrics = evaluate_weights(test_model, test_dataset, test_components, best_weights)
        test_line = (
            f"{dataset_name} | {test_k_values} | {format_weights(best_weights)} | "
            f"{format_metrics(test_metrics)}"
        )
        print(test_line)
        test_file.write(test_line + "\n")
        test_file.flush()

        del test_components, test_model, test_dataset
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"Validation results saved to {val_results_file}")
    print(f"Test results saved to {test_results_file}")

finally:
    if world.tensorboard:
        w.close()
