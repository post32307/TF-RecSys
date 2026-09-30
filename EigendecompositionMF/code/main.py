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

    set_I = None
    set_U = None
    #   None None : search gamma_I and/or gamma_U
    #   0.x None : set gamma_I = 0.x and search gamma_U
    #   None 0.y : search gamma_I and set gamma_U=0.y
    #   0.x 0.y : set gamma_I = 0.x and gamma_U = 0.y


    # ---------------------------------------------------------
    # 1. Seeded dataset paths and run controls
    # ---------------------------------------------------------
    data_root = os.environ.get(
        "TFRECSYS_DATA_ROOT",
        os.path.join(world.ROOT_PATH, "data", "split_dataset", f"00_dataset_seed_{world.seed}"),
    )
    dataset_name = world.dataset
    dataset_path = os.path.join(data_root, dataset_name)

    params_filename = str(world.params_filename)
    if not params_filename.endswith(".txt"):
        params_filename += ".txt"
    params_file = os.path.join(os.path.dirname(__file__), "etc", params_filename)

    def save_best_parameters(model_name, parameters=None, test_results=None):
        """Update parameters, optionally appending final test metrics."""
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
                elif len(parts) == 3:
                    entries[(parts[0].strip(), parts[1].strip(), parts[2].strip())] = None
            # Migrate the old unqualified ensemble entry once a pair-specific
            # key is written; otherwise it would remain as a stale duplicate.
            if model_name.startswith("ensemble:"):
                entries.pop((dataset_name, seed_name, "ensemble"), None)
            entries[key] = None if parameters is None else repr(parameters)
            if test_results is not None:
                def format_metric(name):
                    values = np.asarray(test_results[name]).reshape(-1)
                    if len(values) == 1:
                        return f"{float(values[0]):.8f}"
                    # Preserve every cutoff, in world.topks order.
                    return repr([float(value) for value in values])

                entries[key] += (
                    f" : Precision {format_metric('precision')}"
                    f" / Recall {format_metric('recall')}"
                    f" / NDCG {format_metric('ndcg')}"
                )
            model_order = {"SGFCF": 1, "DAN": 2, "ensemble": 3}
            seed_order = {"2020": 0, "2026": 1, "42": 2}
            ordered = sorted(
                entries.items(),
                key=lambda item: (item[0][0], seed_order.get(item[0][1], 99),
                                  item[0][1],
                                  (0 if item[0][2].startswith("Ensembling ")
                                   else model_order.get(item[0][2], 99)),
                                  item[0][2]),
            )
            model_entries = [
                item for item in ordered
                if (item[0][2] != "ensemble" and
                    not item[0][2].startswith("ensemble:") and
                    not item[0][2].startswith("Ensembling "))
            ]
            ensemble_entries = [
                item for item in ordered
                if (item[0][2] == "ensemble" or
                    item[0][2].startswith("ensemble:"))
            ]
            handle.seek(0)
            handle.truncate()
            for (dataset_key, seed_key, model_key), value in model_entries:
                if value is None:
                    handle.write(f"{dataset_key} : {seed_key} : {model_key}\n")
                else:
                    handle.write(f"{dataset_key} : {seed_key} : {model_key} : {value}\n")
            if model_entries and ensemble_entries:
                handle.write("\n")
            for (dataset_key, seed_key, model_key), value in ensemble_entries:
                if value is None:
                    handle.write(f"{dataset_key} : {seed_key} : {model_key}\n")
                else:
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
                    # Metric annotations are not part of the parameter dict.
                    parameters = ast.literal_eval(value.split(" : Precision ", 1)[0])
                    if model_key == "ChebyCF":
                        candidates = parameters if isinstance(parameters, list) else [parameters]
                        if not all(p.get("implementation_version") == 2 for p in candidates):
                            world.cprint("Ignoring old ChebyCF parameters; the corrected model requires validation.")
                            continue
                    elif model_key == "S_SGFCF":
                        candidates = parameters if isinstance(parameters, list) else [parameters]
                        if not all(p.get("implementation_version") == 2 for p in candidates):
                            world.cprint("Ignoring old SGFCF parameters; the original SVD implementation requires validation.")
                            continue
                    elif isinstance(parameters, dict) and "ChebyCF" in parameters:
                        if parameters["ChebyCF"].get("implementation_version") != 2:
                            world.cprint("Ignoring ensemble parameters fitted with the old ChebyCF implementation.")
                            continue
                    elif isinstance(parameters, dict) and "sgfcf_params" in parameters:
                        if parameters["sgfcf_params"].get("implementation_version") != 2:
                            world.cprint("Ignoring ensemble parameters fitted with the old SGFCF implementation.")
                            continue
                    saved[model_key] = parameters
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
        # Original SGFCF rank grid from TrainingFreeRecSys/run_sweep.py.
        search_k_values = [
            value for value in (32, 64, 128, 256, 512, 1024)
            if value < spectral_dimension
        ]
        if not search_k_values:
            search_k_values = [max(1, spectral_dimension - 1)]
    world.cprint(f"fixed_k: {fixed_k}")
    world.cprint(f"k candidates: {search_k_values}")



    # ---------------------------------------------------------
    # 2. Independent validation search for the chosen model
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
    saved_model = saved_parameters.get(world.input_model)
    
    def metric(scores, evaluator, target):
        return evaluator.Test_RatingMatrix(scores, target)

    def score_model(recmodel, model_name, params, model_cache):
        if model_name == "S_EASE_I":
            return recmodel.S_EASE_I(cache=model_cache, **params)
        elif model_name == "S_EASE_U":
            return recmodel.S_EASE_U(cache=model_cache, **params)
        elif model_name == "S_SGFCF":
            return recmodel.S_SGFCF(cache=model_cache, **params)
        elif model_name == "Ensemble_3Way":
            sgfcf_key = f"sgfcf_{params['sgfcf_params']}"
            ease_i_key = f"ease_i_{params['ease_i_params']}"
            ease_u_key = f"ease_u_{params['ease_u_params']}"
            
            if sgfcf_key not in model_cache:
                for k in list(model_cache.keys()):
                    if k.startswith("sgfcf_"):
                        del model_cache[k]
                model_cache[sgfcf_key] = recmodel.S_SGFCF(cache=model_cache, **params['sgfcf_params'])
                
            if ease_i_key not in model_cache:
                model_cache[ease_i_key] = recmodel.S_EASE_I(cache=model_cache, **params['ease_i_params'])
                
            if ease_u_key not in model_cache:
                model_cache[ease_u_key] = recmodel.S_EASE_U(cache=model_cache, **params['ease_u_params'])
                
            return model.Ensemble3WayScoreMatrix(
                model_cache[sgfcf_key],
                model_cache[ease_i_key],
                model_cache[ease_u_key],
                params['gamma_i'],
                params['gamma_u']
            )
        raise ValueError(f"Unknown model: {model_name}")

    # Exact SGFCF sweep grid from TrainingFreeRecSys/run_sweep.py. These
    # curated configurations include alpha=19.0, which the old local grid
    # omitted and which produced the published amazon-electronics result.
    ORIGINAL_SGFCF_BASE_COMBOS = (
        {"beta_1": 1.0, "beta_2": 1.0, "alpha": 0.0, "eps": 0.5, "gamma": 1.0},
        {"beta_1": 0.7, "beta_2": 1.1, "alpha": 0.0, "eps": 0.28, "gamma": 1.5},
        {"beta_1": 4.3, "beta_2": 5.1, "alpha": 2.3, "eps": 0.5, "gamma": 0.05},
        {"beta_1": 0.9, "beta_2": 1.0, "alpha": 0.0, "eps": 0.37, "gamma": 0.3},
        {"beta_1": 2.0, "beta_2": 2.4, "alpha": 0.0, "eps": 0.37, "gamma": 0.07},
        {"beta_1": 1.0, "beta_2": 1.3, "alpha": 19.0, "eps": 0.5, "gamma": 0.5},
        {"beta_1": 2.0, "beta_2": 5.0, "alpha": 5.3, "eps": 0.5, "gamma": 0.025},
        {"beta_1": 0.3, "beta_2": 2.5, "alpha": 0.0, "eps": 0.34, "gamma": 1.6},
        {"beta_1": 2.5, "beta_2": 6.5, "alpha": 4.8, "eps": 0.5, "gamma": 0.025},
    )

    def original_sgfcf_parameter_space():
        return [
            {**base, "k": int(k), "implementation_version": 2}
            for base in ORIGINAL_SGFCF_BASE_COMBOS
            for k in search_k_values
        ]

    # Shared EASE search spaces.  Both direct-model and ensemble validation
    # paths use these definitions to prevent parameter-grid drift.  EASE-I
    # and EASE-U both use these beta values for their IGF responses.
    EASE_LAMBDA_SPACE = (50.0, 100.0, 200.0, 500.0, 1000.0)
    EASE_I_BETA_1_SPACE = (0.8, 1.0)
    EASE_I_BETA_2_SPACE = (1.0, 1.2)

    def ease_parameter_space(model_name):
        if model_name in ("S_EASE_I", "S_EASE_U"):
            return [
                {"lambda_": lambda_, "beta_1": beta_1, "beta_2": beta_2}
                for lambda_ in EASE_LAMBDA_SPACE
                for beta_1 in EASE_I_BETA_1_SPACE
                for beta_2 in EASE_I_BETA_2_SPACE
            ]
        raise ValueError(f"EASE parameter space is not defined for {model_name}.")

    def default_model_parameters(model_name):
        if model_name == "S_EASE_I":
            return {"lambda_": 500.0, "beta_1": 1.0, "beta_2": 1.0}
        elif model_name == "S_EASE_U":
            return {"lambda_": 500.0, "beta_1": 1.0, "beta_2": 1.0}
        elif model_name == "S_SGFCF":
            return dict(original_sgfcf_parameter_space()[0])
        elif model_name == "IGF_EASE":
            return {"beta_1": 1.0, "beta_2": 1.1, "alpha": 1.0,
                    "eps": 0.5, "lambda_": 50.0, "gamma": 1.0, "k": int(search_k_values[0])}
        elif model_name == "Ensemble_3Way":
            def_sgfcf = dict(original_sgfcf_parameter_space()[0])
            def_ease = {"lambda_": 500.0}
            sgfcf_p = saved_parameters.get("S_SGFCF", def_sgfcf)
            if isinstance(sgfcf_p, list): sgfcf_p = sgfcf_p[0]
            ease_i_p = saved_parameters.get("S_EASE_I", def_ease)
            if isinstance(ease_i_p, list): ease_i_p = ease_i_p[0]
            ease_u_p = saved_parameters.get("S_EASE_U", def_ease)
            if isinstance(ease_u_p, list): ease_u_p = ease_u_p[0]
            return {
                "sgfcf_params": sgfcf_p,
                "ease_i_params": ease_i_p,
                "ease_u_params": ease_u_p,
                "gamma_i": 0.3,
                "gamma_u": 0.3
            }
        raise ValueError(f"Unknown model: {model_name}")

    def model_parameter_space(model_name):
        if model_name in ("S_EASE_I", "S_EASE_U"):
            return ease_parameter_space(model_name)
        elif model_name == "S_SGFCF":
            return original_sgfcf_parameter_space()
        elif model_name == "IGF_EASE":
            return [dict(beta_1=b1, beta_2=b2, alpha=alpha, eps=eps,
                         lambda_=lambda_, gamma=gamma, k=k)
                    for alpha in (0.0, 1.0, 20.0)
                    for eps in (0.1, 0.3, 0.5)
                    for k in search_k_values
                    for lambda_ in (50, 500, 1000)
                    for b1, b2, gamma in product((0.8, 1.0), (1.1, 1.3),
                                                   (0.1, 0.5, 1.0))]
        elif model_name == "Ensemble_3Way":
            sgfcf_space = original_sgfcf_parameter_space()
            ease_i_space = ease_parameter_space("S_EASE_I")
            ease_u_space = ease_parameter_space("S_EASE_U")
            
            # gammas_i = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0] if set_I is None else [float(set_I)]
            # gammas_u = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0] if set_U is None else [float(set_U)]

            # gammas_i = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] if set_I is None else [float(set_I)]
            gammas_i = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0] if set_I is None else [float(set_I)]
            gammas_u = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0] if set_U is None else [float(set_U)]

            return [
                {
                    "sgfcf_params": sp,
                    "ease_i_params": eip,
                    "ease_u_params": eup,
                    "gamma_i": round(gi, 2),
                    "gamma_u": round(gu, 2)
                }
                for sp in sgfcf_space
                for eip in ease_i_space
                for eup in ease_u_space
                for gi in gammas_i for gu in gammas_u if round(gi + gu, 2) <= 1.0
            ]
        raise ValueError(f"Unknown model: {model_name}")

    # ---------------------------------------------------------
    # 3. Three-stage validation and ensemble search
    # ---------------------------------------------------------
    # The three-way model is deliberately selected entirely on val.txt:
    # 1) retain the top 10 SGFCF and top 3 EASE-I/EASE-U settings, then 2)
    # choose the best component triplet and valid gamma pair from them.
    # test.txt is reached only once, after all validation decisions are fixed.
    TOP_COMPONENT_PARAMS = 10
    EASE_TOP_COMPONENT_PARAMS = 3
    def gamma_parameter_pairs():
        gamma_values = (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0)
        gamma_i_values = gamma_values if set_I is None else (float(set_I),)
        gamma_u_values = gamma_values if set_U is None else (float(set_U),)
        return [
            (round(gamma_i, 2), round(gamma_u, 2))
            for gamma_i in gamma_i_values
            for gamma_u in gamma_u_values
            if round(gamma_i + gamma_u, 2) <= 1.0
        ]

    def clear_cuda_cache():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def search_top_parameters(model_name, candidates, keep_top=TOP_COMPONENT_PARAMS):
        """Evaluate every candidate on val.txt and return its top settings."""
        ranked = []
        best_score = -np.inf
        best_params = None
        best_result = None
        for index, params in enumerate(candidates, 1):
            scores = score_model(search_model, model_name, params, cache)
            result = metric(scores, search_model, search_dataset)
            score = float(result["recall"][0])
            # The index makes ties deterministic: earlier grid entries win.
            ranked.append((score, index, dict(params), result))
            if score > best_score:
                best_score, best_params, best_result = score, dict(params), result
            del scores
            clear_cuda_cache()
            if index % 10 == 0 or index == len(candidates):
                print(f"\033[0;30;43m{model_name} validation {index}/{len(candidates)}: "
                      f"best Recall@20={best_score:.6f}\033[0m", end=" ")
                print(f"| best_params={best_params}")
        ranked.sort(key=lambda entry: (-entry[0], entry[1]))
        return [entry[2] for entry in ranked[:min(keep_top, len(ranked))]], best_result

    def cpu_component_scores(model_name, parameters):
        """Construct a validation component once and keep GPU memory free."""
        scores = score_model(search_model, model_name, parameters, cache)
        if torch.is_tensor(scores):
            scores = scores.detach().cpu()
        clear_cuda_cache()
        return scores

    if world.input_model == "Ensemble_3Way" and val_true == 1:
        # Stage 1: each independent component is searched exclusively on val.txt.
        top_sgfcf, sgfcf_validation = search_top_parameters(
            "S_SGFCF", original_sgfcf_parameter_space())
        top_ease_i, ease_i_validation = search_top_parameters(
            "S_EASE_I", ease_parameter_space("S_EASE_I"), EASE_TOP_COMPONENT_PARAMS)
        top_ease_u, ease_u_validation = search_top_parameters(
            "S_EASE_U", ease_parameter_space("S_EASE_U"), EASE_TOP_COMPONENT_PARAMS)
        save_best_parameters("S_SGFCF", top_sgfcf)
        save_best_parameters("S_EASE_I", top_ease_i)
        save_best_parameters("S_EASE_U", top_ease_u)

        # Stage 2: evaluate every retained component triplet and gamma pair on
        # val.txt.  SGFCF and EASE-U are CPU-resident reusable score objects;
        # EASE-I is retained one setting at a time to bound peak memory.
        validation_sgfcf = {
            repr(params): cpu_component_scores("S_SGFCF", params)
            for params in top_sgfcf
        }
        validation_ease_u = {
            repr(params): cpu_component_scores("S_EASE_U", params)
            for params in top_ease_u
        }
        gamma_pairs = gamma_parameter_pairs()
        total = len(top_sgfcf) * len(top_ease_i) * len(top_ease_u) * len(gamma_pairs)
        best_score = -np.inf
        best_params = None
        best_result = None
        search_index = 0

        for ease_i_params in top_ease_i:
            ease_i_scores = cpu_component_scores("S_EASE_I", ease_i_params)
            for ease_u_params in top_ease_u:
                ease_u_scores = validation_ease_u[repr(ease_u_params)]
                for sgfcf_params in top_sgfcf:
                    sgfcf_scores = validation_sgfcf[repr(sgfcf_params)]
                    for gamma_i, gamma_u in gamma_pairs:
                        search_index += 1
                        scores = model.Ensemble3WayScoreMatrix(
                            sgfcf_scores, ease_i_scores, ease_u_scores,
                            gamma_i, gamma_u)
                        result = metric(scores, search_model, search_dataset)
                        score = float(result["recall"][0])
                        if score > best_score:
                            best_score = score
                            best_params = {
                                "sgfcf_params": dict(sgfcf_params),
                                "ease_i_params": dict(ease_i_params),
                                "ease_u_params": dict(ease_u_params),
                                "gamma_i": gamma_i,
                                "gamma_u": gamma_u,
                            }
                            best_result = result
                        del scores
                        if search_index % 100 == 0 or search_index == total:
                            print(f"\033[0;30;43mEnsemble_3Way validation "
                                  f"{search_index}/{total}: best Recall@20="
                                  f"{best_score:.6f}\033[0m", end=" ")
                            print(f"| best_params={best_params}")
            del ease_i_scores
            clear_cuda_cache()

        # Stage 3: construct the selected scoring matrices on the test loader
        # and evaluate them once against test.txt.
        test_scores = score_model(Recmodel, "Ensemble_3Way", best_params, cache)
        test_result = Recmodel.Test_RatingMatrix(test_scores, dataset)
        save_best_parameters("Ensemble_3Way", best_params, test_results=test_result)

        world.cprint(f"=== Dataset: {dataset_name} ===")
        world.cprint("=== Validation / search results ===")
        world.cprint(f"SGFCF top-{TOP_COMPONENT_PARAMS} validation best: {sgfcf_validation}")
        world.cprint(f"EASE_I top-{EASE_TOP_COMPONENT_PARAMS} validation best: {ease_i_validation}")
        world.cprint(f"EASE_U top-{EASE_TOP_COMPONENT_PARAMS} validation best: {ease_u_validation}")
        world.cprint(f"Ensemble_3Way validation: {best_result}")
        world.cprint("=== Final test results ===")
        print(f"\033[0;30;43mEnsemble_3Way test: {test_result}\033[0m", end=" ")
        print(f"| params: {best_params}")

        del test_scores, validation_sgfcf, validation_ease_u
        clear_cuda_cache()
    else:
        # Preserve direct single-model validation/test behavior.
        best_score = -np.inf
        best_params = default_model_parameters(world.input_model)
        best_result = None

        if val_true:
            candidates = model_parameter_space(world.input_model)
            for index, params in enumerate(candidates, 1):
                scores = score_model(search_model, world.input_model, params, cache)
                result = metric(scores, search_model, search_dataset)
                score = float(result["recall"][0])
                if score > best_score:
                    best_score = score
                    best_params = params
                    best_result = result
                del scores
                if index % 10 == 0 or index == len(candidates):
                    print(f"\033[0;30;43m{world.input_model} search {index}/{len(candidates)}: best Recall@20={best_score:.6f}\033[0m", end=" ")
                    print(f"| best_params={best_params}")
            save_best_parameters(world.input_model, best_params)
        elif saved_model:
            best_params = saved_model[0] if isinstance(saved_model, list) else saved_model

        test_scores = score_model(Recmodel, world.input_model, best_params, cache)
        test_result = Recmodel.Test_RatingMatrix(test_scores, dataset)
        if val_true or saved_model is not None:
            save_best_parameters(world.input_model, best_params, test_results=test_result)

        world.cprint(f"=== Dataset: {dataset_name} ===")
        world.cprint("=== Validation / search results ===")
        world.cprint(f"{world.input_model} validation: {best_result}")
        world.cprint("=== Final test results ===")
        print(f"\033[0;30;43m{world.input_model} test: {test_result}\033[0m", end=" ")
        print(f"| params: {best_params}")

        del test_scores
        clear_cuda_cache()
    



finally:
    if world.tensorboard:
        w.close()
