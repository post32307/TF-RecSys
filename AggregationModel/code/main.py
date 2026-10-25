import world
import torch
torch.set_num_threads(8)
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
    tensorWrite: SummaryWriter = SummaryWriter(
        join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
    )
else:
    tensorWrite = None
    world.cprint("not enable tensorflowboard")

try:

    # Average the selected configurations for each model, then search the
    # weight applied to the first model in the sorted ensemble pair.
    
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
    
    TOP_CANDIDATES = world.M
    # Number of per-model candidates retained from the individual validation searches before the ensemble search.

    w_space = (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0)
    # Ensemble scores are w * Model_1 + (1-w) * Model_2.


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
                parts = line.rstrip("\n").split(" : ", 5)
                if len(parts) >= 4:
                    row_key = (parts[0].strip(), parts[1].strip(), parts[2].strip())
                    if row_key[2] == "test_time_seconds":
                        continue
                    tail = [part.strip() for part in parts[3:]]
                    # Preserve the time, parameters, and results fields when
                    # this writer rewrites the file for another model.
                    if len(tail) >= 3 and tail[0].replace(".", "", 1).isdigit():
                        entries[row_key] = {
                            "time": tail[0], "parameters": tail[1],
                            "results": " : ".join(tail[2:]),
                        }
                    elif len(tail) == 1 and tail[0].replace(".", "", 1).isdigit():
                        entries[row_key] = {"time": tail[0], "parameters": None, "results": None}
                    elif len(tail) == 2 and tail[1].startswith("Precision "):
                        entries[row_key] = {
                            "time": None, "parameters": tail[0], "results": tail[1],
                        }
                    else:
                        entries[row_key] = {"time": None, "parameters": " : ".join(tail), "results": None}
                elif len(parts) == 3:
                    entries[(parts[0].strip(), parts[1].strip(), parts[2].strip())] = {
                        "time": None, "parameters": None, "results": None,
                    }
            # Migrate the old unqualified ensemble entry once a pair-specific
            # key is written; otherwise it would remain as a stale duplicate.
            if model_name.startswith("ensemble:"):
                entries.pop((dataset_name, seed_name, "ensemble"), None)
            entry = entries.get(key, {"time": None, "parameters": None, "results": None})
            entry["parameters"] = None if parameters is None else repr(parameters)
            if test_results is not None:
                def format_metric(name):
                    values = np.asarray(test_results[name]).reshape(-1)
                    if len(values) == 1:
                        return f"{float(values[0]):.8f}"
                    # Preserve every cutoff, in world.topks order.
                    return repr([float(value) for value in values])

                entry["results"] = (
                    f"Precision {format_metric('precision')}"
                    f" / Recall {format_metric('recall')}"
                    f" / NDCG {format_metric('ndcg')}"
                )
            entries[key] = entry
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
                handle.write(format_parameter_entry(dataset_key, seed_key, model_key, value))
            if model_entries and ensemble_entries:
                handle.write("\n")
            for (dataset_key, seed_key, model_key), value in ensemble_entries:
                handle.write(format_parameter_entry(dataset_key, seed_key, model_key, value))
            handle.flush()
            os.fsync(handle.fileno())
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def format_parameter_entry(dataset_key, seed_key, model_key, value):
        fields = [dataset_key, seed_key, model_key]
        if value["time"] is not None:
            fields.append(value["time"])
        if value["parameters"] is not None:
            fields.append(value["parameters"])
        if value["results"] is not None:
            fields.append(value["results"])
        return " : ".join(fields) + "\n"

    def save_test_time(seconds):
        """Store runtime in the ensemble's existing parameter/result row."""
        seed_name = str(world.seed)
        with open(params_file, "a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                handle.seek(0)
                updated = []
                found = False
                for line in handle.readlines():
                    parts = line.rstrip("\n").split(" : ", 5)
                    if len(parts) >= 3 and tuple(p.strip() for p in parts[:3]) == (
                        dataset_name, seed_name, ensemble_key
                    ):
                        found = True
                        tail = [p.strip() for p in parts[3:]]
                        if tail and tail[0].replace(".", "", 1).isdigit():
                            tail = tail[1:]
                        line = " : ".join(parts[:3] + [f"{float(seconds):.9f}"] + tail) + "\n"
                    elif len(parts) >= 4 and parts[2].strip() == "test_time_seconds":
                        continue
                    updated.append(line)
                if not found:
                    updated.append(
                        f"{dataset_name} : {seed_name} : {ensemble_key} : {float(seconds):.9f}\n"
                    )
                handle.seek(0)
                handle.truncate()
                handle.writelines(updated)
                handle.flush()
                os.fsync(handle.fileno())
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def load_saved_parameters():
        """Load saved parameters for the current dataset and seed."""
        saved = {}
        if not os.path.isfile(params_file):
            return saved
        seed_name = str(world.seed)
        with open(params_file, "r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.rstrip("\n").split(" : ", 5)
                if len(parts) < 4:
                    continue
                dataset_key, seed_key, model_key = (part.strip() for part in parts[:3])
                tail = [part.strip() for part in parts[3:]]
                if tail and tail[0].replace(".", "", 1).isdigit() and len(tail) > 1:
                    tail = tail[1:]
                value = " : ".join(tail)
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
    ensemble_models = sorted(set(world.ensemble_models))
    if len(ensemble_models) != 2:
        raise ValueError("Select exactly two distinct ensemble models.")
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
    # Ensemble parameters are logged only on the pair-specific parameter
    # line below; no separate ``Ensembling ...`` marker is emitted.
    if saved_ensemble is not None:
        if (not isinstance(saved_ensemble, dict) or
                not set(ensemble_models).issubset(saved_ensemble) or
                saved_ensemble.get("weight_model") != ensemble_models[0]):
            # Older entries do not specify which model receives w.
            # Recompute their weight with the shared scoring convention.
            saved_ensemble = None

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

    if val_true:
        model_candidates = {}
        for model_name in ensemble_models:
            saved = saved_parameters.get(model_name)
            saved_candidates = (saved if isinstance(saved, list) else
                                ([saved] if saved is not None else []))
            if len(saved_candidates) >= TOP_CANDIDATES:
                model_candidates[model_name] = [
                    dict(params) for params in saved_candidates[:TOP_CANDIDATES]
                ]
                world.cprint(
                    f"Found saved {model_name} top-{len(model_candidates[model_name])} "
                    f"parameters; skipping {model_name} search."
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
            if not entries:
                params = default_model_parameters(model_name)
                scores = score_model(search_model, model_name, params, cache)
                result = metric(scores, search_model, search_dataset)
                entries = [(float(result["recall"][0]), params, result)]
                del scores
            model_candidates[model_name] = [entry[1] for entry in entries]
            save_best_parameters(model_name, model_candidates[model_name])

        first_model, second_model = ensemble_models
        if (saved_ensemble is not None and
                all(isinstance(saved_ensemble.get(name), list) and
                    len(saved_ensemble[name]) >= TOP_CANDIDATES
                    for name in ensemble_models)):
            generic_best = saved_ensemble
            world.cprint("Found saved ensemble parameters; skipping ensemble search.")
        else:
            best_score_generic = -np.inf
            mean_scores = {}
            for model_name in ensemble_models:
                candidate_scores = [
                    score_model(search_model, model_name, params, cache)
                    for params in model_candidates[model_name]
                ]
                mean_scores[model_name] = sum(candidate_scores) / len(candidate_scores)
                del candidate_scores
            for w in w_space:
                combined = model.LinearScoreMatrix(
                    mean_scores[first_model], mean_scores[second_model], w
                )
                result = metric(combined, search_model, search_dataset)
                score = float(result["recall"][0])
                if score > best_score_generic:
                    best_score_generic = score
                    generic_best = {
                        "w": w,
                        "weight_model": first_model,
                        first_model: [dict(p) for p in model_candidates[first_model]],
                        second_model: [dict(p) for p in model_candidates[second_model]],
                    }
                del combined
            del mean_scores
            save_best_parameters(ensemble_key, generic_best)
    else:
        generic_best = saved_ensemble or {
            "w": 0.5,
            "weight_model": ensemble_models[0],
            **{name: default_model_parameters(name) for name in ensemble_models},
        }


    # ---------------------------------------------------------
    # 3. Final test of the combined model
    # ---------------------------------------------------------
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    test_start = time.perf_counter()

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
    test_scores = model.LinearScoreMatrix(first_test, second_test, generic_best["w"])
    del first_test, second_test
    test_result = Recmodel.Test_RatingMatrix(test_scores, dataset)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    test_seconds = time.perf_counter() - test_start

    # Update both freshly searched and resumed ensembles after testing.
    # Do not mark unvalidated defaults from val_true=0 as a completed search.
    if val_true or saved_ensemble is not None:
        save_best_parameters(ensemble_key, generic_best, test_results=test_result)
    world.cprint("=== Final test results ===")
    world.cprint(f"Ensemble parameters: {generic_best}")
    world.cprint(f"ensemble test: {test_result}")
    del test_scores
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    save_test_time(test_seconds)
    world.cprint(f"Final test phase time: {test_seconds:.6f} seconds")



finally:
    if world.tensorboard:
        tensorWrite.close()
