import world
import torch
torch.set_num_threads(2)
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

    # # # Important : The model now performs averaging on the stage1 ensemble models. It then looks for the best a_sum only in the second stage.
    
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
    
    TOP_CANDIDATES = world.ensemble_k
    # Number of per-model candidates retained from the individual validation searches before the ensemble search.

    # a_sum_space = (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0)
    a_sum_space = (0.7,) # Removing secondary model
    # a_sum_space = (1.0,) # Removing primary model
    # a_sum_space = (0.5,) # Simple Average
    # Ensemble scores are (1-a_sum) * Model_1 + a_sum * Model_2


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
                    elif isinstance(parameters, dict) and "ChebyCF" in parameters:
                        candidates = parameters["ChebyCF"]
                        if isinstance(candidates, dict):
                            candidates = [candidates]
                        if not candidates or any(
                            not isinstance(p, dict) or p.get("implementation_version") != 2
                            for p in candidates
                        ):
                            world.cprint("Ignoring ensemble parameters fitted with the old ChebyCF implementation.")
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
    ensemble_models = sorted(set(world.ensemble_models))
    # Ensemble results are keyed by the participating model pair so multiple
    # ensembles for the same dataset/seed do not overwrite one another.
    ensemble_key = "ensemble:" + "+".join(ensemble_models)
    saved_ensemble = saved_parameters.get(ensemble_key)
    if saved_ensemble is None:
        # Backward compatibility with older logs that used the single key
        # ``ensemble``.  It is accepted only when it contains this pair.
        legacy_ensemble = saved_parameters.get("ensemble")
        if (isinstance(legacy_ensemble, dict) and
                set(ensemble_models).issubset(legacy_ensemble)):
            saved_ensemble = legacy_ensemble
    world.cprint(f"Ensemble models: {', '.join(ensemble_models)}")
    # Only parameters for models requested in this run are eligible for
    # skipping.  A saved DAN entry must not affect an SGFCF+TurboCF run.
    if "SGFCF" not in ensemble_models:
        saved_sg = None
    if "DAN" not in ensemble_models:
        saved_dan = None
    # Ensemble parameters are logged only on the pair-specific parameter
    # line below; no separate ``Ensembling ...`` marker is emitted.
    if saved_ensemble is not None:
        if not set(ensemble_models).issubset(set(saved_ensemble)):
            saved_ensemble = None
    if isinstance(saved_sg, list) and len(saved_sg) >= TOP_CANDIDATES:
        world.cprint(f"Found saved SGFCF top-{TOP_CANDIDATES} parameters; skipping SGFCF search.")
    if isinstance(saved_dan, list) and len(saved_dan) >= TOP_CANDIDATES:
        world.cprint(f"Found saved DAN top-{TOP_CANDIDATES} parameters; skipping DAN search.")
    if saved_ensemble is not None:
        world.cprint("Found saved ensemble parameters; skipping ensemble search.")

    sg_space = product((0.8, 1.0), (1.1, 1.3),
                       (0.0, 1.0, 20.0), (0.1, 0.3, 0.5),
                       (0.1, 0.5, 1.0))
    dan_space = product((20.0, 50.0), (0.1, 0.3, 0.5),
                        (0.1, 0.3, 0.5), (0.2, 0.6),
                        (0.1, 0.3, 0.5))

    def metric(scores, evaluator, target):
        return evaluator.Test_RatingMatrix(scores, target)

    def score_model(recmodel, model_name, params, model_cache):
        scores = dispatch_score_model(recmodel, model_name, params, model_cache)
        if "ChebyCF" in ensemble_models:
            # Release the other model's GPU scores too before ChebyCF fitting
            # or ensemble evaluation. Only user batches are transferred back.
            scores = scores.detach().cpu()
        return scores

    def dispatch_score_model(recmodel, model_name, params, model_cache):
        """Dispatch every in-process baseline to a common score interface."""
        if model_name == "SGFCF":
            return recmodel.S_SGFCF(cache=model_cache, **params)
        if model_name == "DAN":
            return recmodel.S_DAN(cache=model_cache, **params)
        if model_name == "BSPM":
            return recmodel.S_BSPM(cache=model_cache, **params)
        if model_name == "ChebyCF":
            return recmodel.S_ChebyCF(cache=model_cache, **params)
        if model_name == "EASE":
            return recmodel.S_EASE(cache=model_cache, **params)
        if model_name == "GF-CF":
            return recmodel.S_GF_CF(cache=model_cache, **params)
        if model_name == "PGSP":
            return recmodel.S_PGSP(cache=model_cache, **params)
        if model_name == "TurboCF":
            return recmodel.S_TurboCF(cache=model_cache, **params)
        raise ValueError(f"Unknown ensemble model: {model_name}")

    def default_model_parameters(model_name):
        if model_name == "SGFCF":
            return {"beta_1": 1.0, "beta_2": 1.1, "alpha": 1.0,
                    "eps": 0.5, "gamma": 0.0, "k": int(search_k_values[0])}
        if model_name == "DAN":
            return {"model": "RLAE_DAN", "reg_p": 20.0, "alpha": 0.2,
                    "beta": 0.3, "drop_p": 0.6, "xi": 0.1}
        if model_name == "BSPM":
            return {"k": int(search_k_values[0]), "alpha": 0.5,
                    "beta": 0.5, "gamma": 1.0}
        if model_name == "ChebyCF":
            return {"K": 8, "phi": 1.5, "eta": 256,
                    "alpha": 0.1, "beta": 0.2, "implementation_version": 2}
        if model_name == "EASE":
            return {"lambda_": 50.0}
        if model_name == "GF-CF":
            return {"k": int(search_k_values[0])}
        if model_name == "PGSP":
            return {"k": int(search_k_values[0]), "phi": 0.7}
        if model_name == "TurboCF":
            return {"alpha": 0.5, "power": 1.0, "filter": 1}
        raise ValueError(f"Unknown ensemble model: {model_name}")

    def model_parameter_space(model_name):
        """Return the validation grid used by the top-10 model searches."""
        if model_name == "BSPM":
            return [
                {"k": k, "alpha": alpha, "beta": beta, "gamma": gamma}
                for k in search_k_values
                for alpha in (0.3, 0.5, 0.7)
                for beta in (0.3, 0.5, 0.7)
                for gamma in (0.5, 1.0)
            ]
        if model_name == "ChebyCF":
            return [
                {"K": K, "phi": phi, "eta": eta, "alpha": alpha, "beta": beta,
                 "implementation_version": 2}
                # Keep eta outermost to reuse the bounded exact-rank SVD cache.
                for eta in (256, 512)
                if eta < min(search_model.R.shape)
                for K in (8, 32)
                for phi in (1.5, 10.0, 15.0)
                for alpha in (0.1, 0.2)
                for beta in (0.2, 0.4)
            ]
        if model_name == "EASE":
            return [{"lambda_": value} for value in (50, 100, 200, 500, 1000, 2000)]
        if model_name == "GF-CF":
            return [{"k": k} for k in search_k_values]
        if model_name == "PGSP":
            return [{"k": k, "phi": phi}
                    for k in search_k_values
                    for phi in (0.0, 0.25, 0.5, 0.75, 1.0)]
        if model_name == "TurboCF":
            return [{"alpha": alpha, "power": power, "filter": filter_id}
                    for alpha in (0.1, 0.3, 0.5, 0.7, 0.9)
                    for power in (0.5, 1.0, 1.5, 2.0)
                    for filter_id in (1, 2, 3)]
        if model_name == "SGFCF":
            return [dict(beta_1=b1, beta_2=b2, alpha=alpha, eps=eps,
                         gamma=gamma, k=k)
                    for alpha in (0.0, 1.0, 20.0)
                    for eps in (0.1, 0.3, 0.5)
                    for k in search_k_values
                    for b1, b2, gamma in product((0.8, 1.0), (1.1, 1.3),
                                                   (0.1, 0.5, 1.0))]
        if model_name == "DAN":
            return [dict(model="RLAE_DAN", reg_p=reg_p, alpha=alpha,
                         beta=beta, drop_p=drop_p, xi=xi)
                    for reg_p in (20.0, 50.0)
                    for alpha in (0.1, 0.3, 0.5)
                    for beta in (0.1, 0.3, 0.5)
                    for drop_p in (0.2, 0.6)
                    for xi in (0.1, 0.3, 0.5)]
        raise ValueError(f"Unknown ensemble model: {model_name}")

    def keep_top(entries, candidate, limit=TOP_CANDIDATES):
        """Insert a validation candidate and retain only the top scores."""
        entries.append(candidate)
        entries.sort(key=lambda value: value[0], reverse=True)
        del entries[limit:]

    top_sg = []
    top_dan = []
    generic_best = None
    if val_true:
        # Keep normalization and k outermost.  This makes the cached
        # normalized adjacency/eigenspace reusable for every filter setting.
        sg_saved_top10 = isinstance(saved_sg, list) and len(saved_sg) >= TOP_CANDIDATES
        dan_saved_top10 = isinstance(saved_dan, list) and len(saved_dan) >= TOP_CANDIDATES
        if "SGFCF" in ensemble_models and not sg_saved_top10:
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
            save_best_parameters("SGFCF", [entry[1] for entry in top_sg])
        elif "SGFCF" in ensemble_models:
            saved_sg_candidates = (saved_sg[:TOP_CANDIDATES]
                                   if isinstance(saved_sg, list) else [saved_sg])
            top_sg = [(0.0, dict(params), None) for params in saved_sg_candidates]
        else:
            top_sg = []
        if "DAN" in ensemble_models and not dan_saved_top10:
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
            save_best_parameters("DAN", [entry[1] for entry in top_dan])
        elif "DAN" in ensemble_models:
            saved_dan_candidates = (saved_dan[:TOP_CANDIDATES]
                                    if isinstance(saved_dan, list) else [saved_dan])
            top_dan = [(0.0, dict(params), None) for params in saved_dan_candidates]
        else:
            top_dan = []
        if ensemble_models != ["DAN", "SGFCF"]:
            # The generic branch below handles other pairs.  Keep any
            # SGFCF/DAN top-10 results already computed above, and only seed
            # placeholders for models that are not part of this run.
            if "SGFCF" not in ensemble_models:
                top_sg = [(0.0, default_model_parameters("SGFCF"), None)]
            if "DAN" not in ensemble_models:
                top_dan = [(0.0, default_model_parameters("DAN"), None)]
        # Search the legacy SGFCF+DAN second stage only for the default pair.
        # Other pairs are handled exclusively by the generic top-10 branch
        # below; running this block for them would evaluate irrelevant
        # placeholder SGFCF/DAN scores and overwrite their ensemble log.
        best_a_sum, best_result, best_score = None, None, -np.inf
        best_sg, best_dan = None, None
        best_sg_result, best_dan_result = None, None
        if ensemble_models == ["DAN", "SGFCF"]:
            best_sg_result, best_dan_result = top_sg[0][2], top_dan[0][2]
            if saved_ensemble is not None:
                best_a_sum = float(saved_ensemble["a_sum"])
                best_sg = dict(saved_ensemble["SGFCF"])
                best_dan = dict(saved_ensemble["DAN"])
                best_result = None
            else:
                # Keep DAN outermost so each DAN score matrix is built once
                # and reused across all retained SGFCF candidates.
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
                    ensemble_key,
                    {"a_sum": best_a_sum, "SGFCF": best_sg, "DAN": best_dan},
                )
        if ensemble_models != ["DAN", "SGFCF"]:
            # For the additional baselines, run the same validation/resume
            # protocol through their scoring adapters.  Their default grids
            # can be expanded without changing the ensemble machinery.
            model_candidates = {}
            model_results = {}
            # Every model participates in the same top-10 validation protocol.
            # Previously only SGFCF and DAN were searched; the other adapters
            # silently fell back to one default configuration.  A saved list
            # with ten entries can be reused, while an old single-entry log is
            # deliberately refreshed so it is upgraded to top-10 format.
            for model_name in ensemble_models:
                # SGFCF/DAN may have just been searched by the dedicated
                # blocks above.  Reuse those in-memory results instead of
                # repeating the full grid because the on-disk log was loaded
                # before the current search started.
                if model_name == "SGFCF" and top_sg and top_sg[0][2] is not None:
                    model_candidates[model_name] = [entry[1] for entry in top_sg]
                    continue
                if model_name == "DAN" and top_dan and top_dan[0][2] is not None:
                    model_candidates[model_name] = [entry[1] for entry in top_dan]
                    continue
                saved = saved_parameters.get(model_name)
                saved_candidates = (saved if isinstance(saved, list) else
                                    ([saved] if saved is not None else []))
                if len(saved_candidates) >= TOP_CANDIDATES:
                    model_candidates[model_name] = [dict(p) for p in saved_candidates[:TOP_CANDIDATES]]
                    world.cprint(
                        f"Found saved {model_name} top-{len(model_candidates[model_name])} parameters; skipping {model_name} search."
                    )
                    continue

                candidates = model_parameter_space(model_name)
                entries = []
                for index, params in enumerate(candidates, 1):
                    scores = score_model(search_model, model_name, params, cache)
                    result = metric(scores, search_model, search_dataset)
                    score = float(result["recall"][0])
                    keep_top(entries, (score, dict(params), result))
                    del scores
                    if index % 25 == 0 or index == len(candidates):
                        world.cprint(
                            f"{model_name} search {index}/{len(candidates)}: "
                            f"best Recall@20={entries[0][0]:.6f}"
                        )
                # A non-empty grid is guaranteed by model_parameter_space; the
                # fallback keeps the code robust if a future grid is empty.
                if not entries:
                    params = default_model_parameters(model_name)
                    scores = score_model(search_model, model_name, params, cache)
                    result = metric(scores, search_model, search_dataset)
                    entries = [(float(result["recall"][0]), params, result)]
                    del scores
                model_candidates[model_name] = [entry[1] for entry in entries]
                model_results[model_name] = entries[0][2]
                save_best_parameters(model_name, model_candidates[model_name])
            first_model, second_model = ensemble_models
            saved_generic = saved_ensemble
            saved_first = (saved_generic.get(first_model)
                           if saved_generic is not None else None)
            saved_second = (saved_generic.get(second_model)
                            if saved_generic is not None else None)
            if (isinstance(saved_first, list) and len(saved_first) >= TOP_CANDIDATES
                    and isinstance(saved_second, list)
                    and len(saved_second) >= TOP_CANDIDATES):
                generic_best = saved_generic
            else:
                best_score_generic = -np.inf
                mean_scores = {}
                for model_name in (first_model, second_model):
                    candidate_scores = [
                        score_model(search_model, model_name, params, cache)
                        for params in model_candidates[model_name]
                    ]
                    mean_scores[model_name] = sum(candidate_scores) / len(candidate_scores)
                    del candidate_scores
                for a_sum in a_sum_space:
                    combined = model.LinearScoreMatrix(
                        mean_scores[first_model], mean_scores[second_model], a_sum
                    )
                    result = metric(combined, search_model, search_dataset)
                    score = float(result["recall"][0])
                    if score > best_score_generic:
                        best_score_generic = score
                        generic_best = {
                            "a_sum": a_sum,
                            first_model: [dict(p) for p in model_candidates[first_model]],
                            second_model: [dict(p) for p in model_candidates[second_model]],
                        }
                    del combined
                del mean_scores
                save_best_parameters(ensemble_key, generic_best)
    else:
        saved_sg_default = saved_sg[0] if isinstance(saved_sg, list) else saved_sg
        saved_dan_default = saved_dan[0] if isinstance(saved_dan, list) else saved_dan
        best_sg = saved_sg_default or {
            "beta_1": 1.0, "beta_2": 1.0, "alpha": 0.0,
            "eps": 0.5, "gamma": 1.0,
            "k": int(input_k) if fixed_k else int(search_k_values[0])}
        best_dan = saved_dan_default or {
            "model": "RLAE_DAN", "reg_p": 20.0, "alpha": 0.2,
            "beta": 0.3, "drop_p": 0.6, "xi": 0.1}
        if ensemble_models != ["DAN", "SGFCF"]:
            generic_best = saved_ensemble or {
                "a_sum": 0.5,
                **{name: default_model_parameters(name) for name in ensemble_models},
            }
            best_a_sum = generic_best["a_sum"]
        elif saved_ensemble is not None:
            best_a_sum = float(saved_ensemble["a_sum"])
            best_sg = dict(saved_ensemble["SGFCF"])
            best_dan = dict(saved_ensemble["DAN"])
        else:
            best_a_sum = 0.5
        best_result = None




    # ---------------------------------------------------------
    # 3. Final test of the combined model
    # ---------------------------------------------------------
    if generic_best is not None:
        first_model, second_model = ensemble_models
        first_candidates = generic_best[first_model]
        second_candidates = generic_best[second_model]
        if isinstance(first_candidates, dict):
            first_candidates = [first_candidates]
        if isinstance(second_candidates, dict):
            second_candidates = [second_candidates]
        first_test_scores = [score_model(Recmodel, first_model, params, cache)
                             for params in first_candidates]
        second_test_scores = [score_model(Recmodel, second_model, params, cache)
                              for params in second_candidates]
        first_test = sum(first_test_scores) / len(first_test_scores)
        second_test = sum(second_test_scores) / len(second_test_scores)
        del first_test_scores, second_test_scores
        test_scores = model.LinearScoreMatrix(first_test, second_test, generic_best["a_sum"])
        del first_test, second_test
    else:
        sg_test = Recmodel.S_SGFCF(**best_sg, cache=cache)
        dan_test = Recmodel.S_DAN(**best_dan, cache=cache)
        test_scores = float(best_a_sum) * sg_test + (1.0 - float(best_a_sum)) * dan_test
    test_result = Recmodel.Test_RatingMatrix(test_scores, dataset)
    # Update both freshly searched and resumed ensembles after testing.
    # Do not mark unvalidated defaults from val_true=0 as a completed search.
    if val_true or saved_ensemble is not None:
        ensemble_parameters = generic_best if generic_best is not None else {
            "a_sum": best_a_sum, "SGFCF": best_sg, "DAN": best_dan,
        }
        save_best_parameters(ensemble_key, ensemble_parameters, test_results=test_result)
    world.cprint("=== Validation / search results ===")
    if val_true and generic_best is None:
        world.cprint(f"SGFCF validation: {best_sg_result}")
        world.cprint(f"DAN validation: {best_dan_result}")
        world.cprint(f"a_sum={best_a_sum}: {best_result}")
    world.cprint("=== Final test results ===")
    if generic_best is None:
        world.cprint(f"SGFCF parameters: {best_sg}")
        world.cprint(f"DAN parameters: {best_dan}")
        world.cprint(f"a_sum (SGFCF coefficient): {best_a_sum}")
        world.cprint(f"DAN coefficient: {1.0 - float(best_a_sum)}")
    else:
        world.cprint(f"Ensemble parameters: {generic_best}")
    world.cprint(f"ensemble test: {test_result}")
    del test_scores
    if generic_best is None:
        del sg_test, dan_test
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    



finally:
    if world.tensorboard:
        w.close()
