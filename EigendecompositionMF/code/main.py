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
    plot_true = 0
    #   0: skip eigenvalue plotting and continue to the dataset-root setup
    #   1: plot real eigenvalues, save to plots/, and terminate with sys.exit()
    if plot_true not in (0, 1):
        raise ValueError("plot_true must be 0 or 1.")
    
    val_true = 1
    #   0: no hyperparameter search; test default params on test.txt
    #   1: train on train.txt, search on val.txt, test on test.txt
    #   2: train on train.txt, search on test.txt, test on test.txt
    if val_true not in (0, 1, 2):
        raise ValueError("val_true must be 0, 1, or 2.")

    if plot_true == 1:
        import temp_plot
        temp_plot.plot_eigenvalues()
        sys.exit()



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

    max_k = world.config["k"]

    world.cprint(f"Dataset root: {data_root}")
    world.cprint(f"Selected dataset: {dataset_name}")
    world.cprint(f"Dataset path: {dataset_path}")
    world.cprint(f"val_true: {val_true}")
    world.cprint(f"max_k: {max_k}")



    print()
    import dataloader
    import model
    import tmp
    dataset = tmp.load_dataset_split(dataloader.Loader, dataset_path, "test.txt")
    Recmodel = model.PSEmodel(world.config, dataset)

    # ---------------------------------------------------------
    # 2. Validation / hyperparameter selection
    # ---------------------------------------------------------
    base_params = tmp.default_params(dataset_name, Recmodel.k)
    best_configs = {
        "original": {"params": dict(base_params)},
        "three_component": {
            "params": dict(base_params),
            "mid_coefficient": 0.1,
            "high_coefficient": 0.1,
        },
        "two_component": {
            "params": dict(base_params),
            "low_coefficient": 1.0,
            "high_coefficient": 1.0,
        },
        "generalized_normalization": {
            "params": dict(base_params),
            "normalization": dict(
                tmp.ASYMMETRIC_NORMALIZATION_SPACE[0]
            ),
        },
    }
    validation_results = None
    shared_homophily = tmp.compute_homophily(Recmodel)

    if val_true in (1, 2):
        search_file = "val.txt" if val_true == 1 else "test.txt"
        if search_file == "test.txt":
            search_dataset, search_model = dataset, Recmodel
        else:
            search_dataset = tmp.load_dataset_split(
                dataloader.Loader, dataset_path, search_file
            )
            search_model = model.PSEmodel(world.config, search_dataset)
        search_homophily = shared_homophily
        if search_model.R.shape != Recmodel.R.shape:
            search_homophily = tmp.compute_homophily(search_model)
        best_configs, validation_results = tmp.search(
            search_model, search_dataset, homophily=search_homophily
        )

        world.cprint(f"Hyperparameter search split: {search_file}")
        for model_name, result in validation_results.items():
            world.cprint(f"{model_name} validation: {result}")

    # ---------------------------------------------------------
    # 3. Final test of all four SGFCF models
    # ---------------------------------------------------------
    test_homophily = shared_homophily
    component_cache = {}
    test_results = {}
    for model_name, config in best_configs.items():
        params = config["params"]
        normalization = config.get("normalization")
        if model_name == "three_component":
            spectrum = "three_bands"
        elif model_name == "two_component":
            spectrum = "two_bands"
        elif model_name == "original":
            spectrum = "largest_edge"
        else:
            spectrum = "largest"
        if normalization is None:
            component_key = (
                "symmetric",
                spectrum,
                params["alpha"],
                params["eps"],
                Recmodel.k,
            )
        else:
            component_key = (
                "generalized",
                normalization["alpha_1"],
                normalization["eps_1"],
                normalization["alpha_2"],
                normalization["eps_2"],
                Recmodel.k,
            )
        if component_key not in component_cache:
            three_component_key = (
                "symmetric", "three_bands",
                params["alpha"], params["eps"], Recmodel.k,
            )
            can_reuse_three_bands = (
                spectrum == "two_bands"
                and three_component_key in component_cache
                and 3 * int(Recmodel.k) <= min(Recmodel.R.shape)
            )
            if can_reuse_three_bands:
                component_cache[component_key] = tmp.select_spectral_bands(
                    component_cache[three_component_key],
                    (tmp.BAND_LOW, tmp.BAND_HIGH),
                )
            else:
                component_cache[component_key] = tmp.compute_components(
                    Recmodel,
                    params,
                    normalization=normalization,
                    spectrum=spectrum,
                )

        if model_name == "three_component":
            sgfcf_model = tmp.SGFCFModel(
                Recmodel,
                params,
                mid_coefficient=config["mid_coefficient"],
                high_coefficient=config["high_coefficient"],
                spectrum="three_bands",
            )
        elif model_name == "two_component":
            sgfcf_model = tmp.TwoComponentSGFCFModel(
                Recmodel,
                params,
                low_coefficient=config["low_coefficient"],
                high_coefficient=config["high_coefficient"],
            )
        else:
            sgfcf_model = tmp.SGFCFModel(Recmodel, params)
        test_results[model_name] = tmp.evaluate(
            sgfcf_model,
            dataset,
            components=component_cache[component_key],
            homophily=test_homophily,
        )
        world.cprint(f"{model_name} best parameters: {config}")
        world.cprint(f"{model_name} test: {test_results[model_name]}")
finally:
    if world.tensorboard:
        w.close()
