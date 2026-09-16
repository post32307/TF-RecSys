import world
import torch
torch.set_num_threads(4)
import os
import sys
import time
import gc
import fcntl
import ast
from itertools import product
import numpy as np
from tensorboardX import SummaryWriter
from os.path import join
# ==============================
import helpers
helpers.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================

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

Neg_k = 1


def load_dataset_split(loader_class, dataset_path, eval_file):
    """Load train.txt while selecting the validation or test target file."""
    previous_eval_file = os.environ.get("TFRECSYS_EVAL_FILE")
    os.environ["TFRECSYS_EVAL_FILE"] = eval_file
    try:
        return loader_class(path=dataset_path)
    finally:
        if previous_eval_file is None:
            os.environ.pop("TFRECSYS_EVAL_FILE", None)
        else:
            os.environ["TFRECSYS_EVAL_FILE"] = previous_eval_file

# init tensorboard
if world.tensorboard:
    w: SummaryWriter = SummaryWriter(
        join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
    )
else:
    w = None
    world.cprint("not enable tensorflowboard")

try:
    
    val_true = 1
    #   0: no hyperparameter search; test default params on test.txt
    #   1: train on train.txt, search on val.txt, test on test.txt
    if val_true not in (0, 1):
        raise ValueError("val_true must be 0 or 1.")

    fixed_k = 0
    #   0: search over values of k
    #   1: use the --k argument
    if fixed_k not in (0, 1):
        raise ValueError("fixed_k must be 0 or 1.")



    # ---------------------------------------------------------
    # 1. Seeded dataset paths and run controls
    # ---------------------------------------------------------
    data_root = os.environ.get(
        "TFRECSYS_DATA_ROOT",
        os.path.join(world.ROOT_PATH, "data", "split_dataset", f"00_dataset_seed_{world.seed}"),
    )
    dataset_name = world.dataset
    dataset_path = os.path.join(data_root, dataset_name)

    params_file = os.path.join(os.path.dirname(__file__), "etc", "params.txt")

    def save_best_parameters(model_name, parameters):
        """Atomically update one dataset/seed/model entry in the parameter log."""
        os.makedirs(os.path.dirname(params_file), exist_ok=True)
        seed_name = str(world.seed)
        key = (dataset_name, seed_name, model_name)
        entries = {}
        with open(params_file, "a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.seek(0)
            for line in handle:
                parts = line.rstrip("\n").split(" : ", 3)
                if len(parts) == 4:
                    entries[(parts[0].strip(), parts[1].strip(), parts[2].strip())] = parts[3].strip()
            entries[key] = repr(parameters)
            model_order = {"SGFCF": 0, "DAN": 1, "ensemble": 2}
            seed_order = {"2020": 0, "2026": 1, "42": 2}
            ordered = sorted(
                entries.items(),
                key=lambda item: (item[0][0], seed_order.get(item[0][1], 99),
                                  item[0][1],
                                  model_order.get(item[0][2], 99), item[0][2]),
            )
            handle.seek(0)
            handle.truncate()
            for (dataset_key, seed_key, model_key), value in ordered:
                handle.write(f"{dataset_key} : {seed_key} : {model_key} : {value}\n")
            handle.flush()
            os.fsync(handle.fileno())
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def load_saved_parameters():
        """Load saved parameters for the current dataset and seed."""
        saved = {}
        if not os.path.isfile(params_file):
            return saved
        seed_name = str(world.seed)
        with open(params_file, "r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.rstrip("\n").split(" : ", 3)
                if len(parts) != 4:
                    continue
                dataset_key, seed_key, model_key, value = (part.strip() for part in parts)
                if dataset_key != dataset_name or seed_key != seed_name:
                    continue
                try:
                    saved[model_key] = ast.literal_eval(value)
                except (SyntaxError, ValueError):
                    world.cprint(f"Ignoring malformed saved parameters for {model_key}")
        return saved
    
    if val_true == 1 and not os.path.isfile(os.path.join(dataset_path, "val.txt")):
        raise FileNotFoundError(f"Validation file not found: {os.path.join(dataset_path, 'val.txt')}")

    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

    input_k = world.config["k"]

    world.cprint(f"Dataset root: {data_root}")
    world.cprint(f"Selected dataset: {dataset_name}")
    world.cprint(f"Dataset path: {dataset_path}")
    world.cprint(f"val_true: {val_true}")
    world.cprint(f"input k: {input_k}")

    print()
    import dataloader
    import model
    dataset = load_dataset_split(dataloader.Loader, dataset_path, "test.txt")
    Recmodel = model.TFmodel(world.config, dataset)
    spectral_dimension = min(Recmodel.R.shape)
    if fixed_k == 1:
        search_k_values = [int(input_k)]
    else:
        search_k_values = [
            value for value in (16, 32, 64, 128, 256, 512, 1024)
            if value < spectral_dimension
        ]
        # search_k_values = [
        #     value for value in (32, 128, 512)
        #     if value < spectral_dimension
        # ]
        if not search_k_values:
            search_k_values = [max(1, spectral_dimension - 1)]
    world.cprint(f"fixed_k: {fixed_k}")
    world.cprint(f"k candidates: {search_k_values}")



    # ---------------------------------------------------------
    # 2. Independent validation searches and linear ensemble
    # ---------------------------------------------------------
    search_dataset = load_dataset_split(
        dataloader.Loader, dataset_path, "val.txt"
    ) if val_true == 1 else None
    if val_true == 0:
        search_dataset = dataset
    search_model = model.TFmodel(world.config, search_dataset)
    cache = {
        "eigen": {}, "max_eigen_entries": 1,
        "normalized": {}, "max_normalized_entries": 1,
        "convolution": {}, "max_convolution_entries": 1,
        "dan": {}, "max_dan_entries": 1,
    }
    saved_parameters = load_saved_parameters()
    saved_sg = saved_parameters.get("SGFCF")
    saved_dan = saved_parameters.get("DAN")
    saved_ensemble = saved_parameters.get("ensemble")
    if saved_sg is not None:
        world.cprint("Found saved SGFCF parameters; skipping SGFCF search.")
    if saved_dan is not None:
        world.cprint("Found saved DAN parameters; skipping DAN search.")
    if saved_ensemble is not None:
        world.cprint("Found saved ensemble parameters; skipping ensemble search.")

    sg_space = product((0.8, 1.0), (1.1, 1.3),
                       (0.0, 1.0, 20.0), (0.1, 0.3, 0.5),
                       (0.1, 0.5, 1.0))
    dan_space = product((20.0, 50.0), (0.1, 0.3, 0.5),
                        (0.1, 0.3, 0.5), (0.2, 0.6),
                        (0.1, 0.3, 0.5))
    # Ensemble scores are SGFCF + a_sum * DAN.
    a_sum_space = (0.1, 0.3, 0.5, 0.7, 0.9)

    def metric(scores, evaluator, target):
        return evaluator.Test_RatingMatrix(scores, target)

    def keep_top(entries, candidate, limit=10):
        """Insert a validation candidate and retain only the top scores."""
        entries.append(candidate)
        entries.sort(key=lambda value: value[0], reverse=True)
        del entries[limit:]

    top_sg = []
    top_dan = []
    if val_true:
        # Keep normalization and k outermost.  This makes the cached
        # normalized adjacency/eigenspace reusable for every filter setting.
        if saved_sg is None:
            sg_candidates = list(sg_space)
            sg_index = 0
            for alpha in (0.0, 1.0, 20.0):
                for eps in (0.1, 0.3, 0.5):
                    for k in search_k_values:
                        for b1, b2, gamma in product((0.8, 1.0), (1.1, 1.3),
                                                       (0.1, 0.5, 1.0)):
                            sg_index += 1
                            scores = search_model.S_SGFCF(b1, b2, alpha, eps, gamma, k, cache)
                            result = metric(scores, search_model, search_dataset)
                            score = float(result["recall"][0])
                            sg_params = {
                                "beta_1": b1, "beta_2": b2, "alpha": alpha,
                                "eps": eps, "gamma": gamma, "k": k}
                            keep_top(top_sg, (score, sg_params, result))
                            del scores
                        if sg_index % 25 == 0 or sg_index == len(sg_candidates) * len(search_k_values):
                            world.cprint(f"SGFCF search {sg_index}/{len(sg_candidates) * len(search_k_values)}: best Recall@20={top_sg[0][0]:.6f}")
            save_best_parameters("SGFCF", top_sg[0][1])
        else:
            top_sg = [(0.0, dict(saved_sg), None)]
        if saved_dan is None:
            dan_candidates = list(dan_space)
            for index, (reg_p, alpha, beta, drop_p, xi) in enumerate(dan_candidates, 1):
                scores = search_model.S_DAN("RLAE_DAN", reg_p, alpha, beta, drop_p, xi, cache)
                result = metric(scores, search_model, search_dataset)
                score = float(result["recall"][0])
                dan_params = {
                        "model": "RLAE_DAN", "reg_p": reg_p, "alpha": alpha,
                        "beta": beta, "drop_p": drop_p, "xi": xi}
                keep_top(top_dan, (score, dan_params, result))
                del scores
                if index % 10 == 0 or index == len(dan_candidates):
                    world.cprint(f"DAN search {index}/{len(dan_candidates)}: best Recall@20={top_dan[0][0]:.6f}")
            save_best_parameters("DAN", top_dan[0][1])
        else:
            top_dan = [(0.0, dict(saved_dan), None)]
        # Search the second-stage ensemble over all pairs of the ten best
        # configurations from each model and every a_sum value.
        best_a_sum, best_result, best_score = None, None, -np.inf
        best_sg, best_dan = None, None
        best_sg_result, best_dan_result = top_sg[0][2], top_dan[0][2]
        if saved_ensemble is not None:
            best_a_sum = float(saved_ensemble["a_sum"])
            best_sg = dict(saved_ensemble["SGFCF"])
            best_dan = dict(saved_ensemble["DAN"])
            best_result = None
        else:
            # Keep DAN outermost so each DAN score matrix is built once and
            # reused across all retained SGFCF candidates.
            for _, dan_params, _ in top_dan:
                dan_scores = search_model.S_DAN(**dan_params, cache=cache)
                for _, sg_params, _ in top_sg:
                    sg_scores = search_model.S_SGFCF(**sg_params, cache=cache)
                    for a_sum in a_sum_space:
                        ensemble = (float(a_sum) * sg_scores +
                                    (1.0 - float(a_sum)) * dan_scores)
                        result = metric(ensemble, search_model, search_dataset)
                        score = float(result["recall"][0])
                        if score > best_score:
                            best_score = score
                            best_a_sum = a_sum
                            best_result = result
                            best_sg = dict(sg_params)
                            best_dan = dict(dan_params)
                        del ensemble
                    del sg_scores
                del dan_scores
            save_best_parameters(
                "ensemble",
                {"a_sum": best_a_sum, "SGFCF": best_sg, "DAN": best_dan},
            )
    else:
        best_sg = saved_sg or {
            "beta_1": 1.0, "beta_2": 1.0, "alpha": 0.0,
            "eps": 0.5, "gamma": 1.0,
            "k": int(input_k) if fixed_k else int(search_k_values[0])}
        best_dan = saved_dan or {
            "model": "RLAE_DAN", "reg_p": 20.0, "alpha": 0.2,
            "beta": 0.3, "drop_p": 0.6, "xi": 0.1}
        if saved_ensemble is not None:
            best_a_sum = float(saved_ensemble["a_sum"])
            best_sg = dict(saved_ensemble["SGFCF"])
            best_dan = dict(saved_ensemble["DAN"])
        else:
            best_a_sum = 0.5
        best_result = None




    # ---------------------------------------------------------
    # 3. Final test of the combined model
    # ---------------------------------------------------------
    sg_test = Recmodel.S_SGFCF(**best_sg, cache=cache)
    dan_test = Recmodel.S_DAN(**best_dan, cache=cache)
    test_scores = float(best_a_sum) * sg_test + (1.0 - float(best_a_sum)) * dan_test
    test_result = Recmodel.Test_RatingMatrix(test_scores, dataset)
    world.cprint("=== Validation / search results ===")
    if val_true:
        world.cprint(f"SGFCF validation: {best_sg_result}")
        world.cprint(f"DAN validation: {best_dan_result}")
        world.cprint(f"a_sum={best_a_sum}: {best_result}")
    world.cprint("=== Final test results ===")
    world.cprint(f"SGFCF parameters: {best_sg}")
    world.cprint(f"DAN parameters: {best_dan}")
    world.cprint(f"a_sum (SGFCF coefficient): {best_a_sum}")
    world.cprint(f"DAN coefficient: {1.0 - float(best_a_sum)}")
    world.cprint(f"ensemble test: {test_result}")
    del sg_test, dan_test, test_scores
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    



finally:
    if world.tensorboard:
        w.close()
