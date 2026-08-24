import world
import torch
torch.set_num_threads(4)
import os
import sys
import time
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
    import tmp
    dataset = tmp.load_dataset_split(dataloader.Loader, dataset_path, "test.txt")
    Recmodel = model.TFmodel(world.config, dataset)
    spectral_dimension = min(Recmodel.R.shape)
    if fixed_k == 1:
        search_k_values = [int(input_k)]
    else:
        search_k_values = [
            value for value in (16, 32, 64, 128, 256)
            if value < spectral_dimension
        ]
        if not search_k_values:
            search_k_values = [max(1, spectral_dimension - 1)]
    world.cprint(f"fixed_k: {fixed_k}")
    world.cprint(f"k candidates: {search_k_values}")

    # ---------------------------------------------------------
    # 2. Validation / hyperparameter selection
    # ---------------------------------------------------------
    dan_dataset_params = {
        **tmp.DAN_DEFAULT_PARAMS,
        **tmp.DAN_DATASET_PARAMS.get(dataset_name, {}),
    }
    default_params = dict(tmp.SGFCF_SEARCH_SPACE[0])
    default_params["k"] = int(input_k)
    best_configs = {
        "SGFCF": {"params": dict(default_params)},
        "DAN": {"params": dict(default_params), "dan_params": {
            **dan_dataset_params, "model": "RLAE_DAN"
        }},
        "SGFCF+DAN": {"params": dict(default_params), "dan_coefficient": 0.0,
                       "dan_params": {**dan_dataset_params, "model": "RLAE_DAN"}},
        "UnifiedNorm-SGFCF+DAN": {"params": dict(default_params), "dan_coefficient": 0.0,
                                  "dan_params": {**dan_dataset_params, "model": "RLAE_DAN"}},
    }
    shared_homophily = tmp.compute_homophily(Recmodel)

    if val_true == 1:
        search_file = "val.txt"
        search_dataset = tmp.load_dataset_split(
            dataloader.Loader, dataset_path, search_file
        )
        search_model = model.TFmodel(world.config, search_dataset)
        search_homophily = shared_homophily
        if search_model.R.shape != Recmodel.R.shape:
            search_homophily = tmp.compute_homophily(search_model)
        print()
        best_configs, validation_results = tmp.search(
            search_model,
            search_dataset,
            homophily=search_homophily,
            k_values=search_k_values,
            dan_params=dan_dataset_params,
        )
        print()
        world.cprint("=== Validation / search results ===")
        world.cprint(f"Hyperparameter search split: {search_file}")
        for model_name, result in validation_results.items():
            world.cprint(f"{model_name} validation: {result}")
        print()

    # ---------------------------------------------------------
    # 3. Final test of models
    # ---------------------------------------------------------
    test_results = {}
    world.cprint(
        f"=== Hyperparameter search concluded for dataset: {dataset_name} ==="
    )
    world.cprint("=== Final test results ===")
    for model_name in ("SGFCF", "DAN", "SGFCF+DAN", "UnifiedNorm-SGFCF+DAN"):
        config = best_configs[model_name]
        params = config["params"]
        Recmodel.k = int(params.get("k", input_k))
        if model_name == "DAN":
            dan_model = tmp.DANScoringModel(Recmodel.R, **config["dan_params"])
            test_results[model_name] = tmp._evaluate_dan(dan_model, Recmodel, dataset)
        else:
            components = tmp.compute_components(Recmodel, params, spectrum="largest")
            if model_name == "SGFCF":
                test_results[model_name] = tmp.evaluate(
                    tmp.SGFCFModel(Recmodel, params), dataset, components, shared_homophily
                )
            else:
                unified = model_name.startswith("UnifiedNorm")
                dan_input = components[3] if unified else Recmodel.R
                dan_model = tmp.DANScoringModel(dan_input, **config["dan_params"])
                combined = tmp.SGFCFPlusDANModel(
                    Recmodel, params, dan_model, config["dan_coefficient"], dan_interactions=dan_input
                )
                test_results[model_name] = tmp.evaluate_sgfcf_dan(
                    combined, dataset, components, shared_homophily
                )
        world.cprint(f"{model_name} best parameters: {config}")
        world.cprint(f"{model_name} test: {test_results[model_name]}")
        print()


finally:
    if world.tensorboard:
        w.close()
