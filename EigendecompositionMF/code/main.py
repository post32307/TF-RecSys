import world
import torch
torch.set_num_threads(4)
import os
import sys
import time
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
        # search_k_values = [
        #     value for value in (16, 32, 64, 128, 256, 512, 1024)
        #     if value < spectral_dimension
        # ]
        search_k_values = [
            value for value in (32, 128, 512)
            if value < spectral_dimension
        ]
        if not search_k_values:
            search_k_values = [max(1, spectral_dimension - 1)]
    world.cprint(f"fixed_k: {fixed_k}")
    world.cprint(f"k candidates: {search_k_values}")



    # ---------------------------------------------------------
    # 2. Validation search for the SGFCF+DAN combined model
    # ---------------------------------------------------------
    search_dataset = load_dataset_split(
        dataloader.Loader, dataset_path, "val.txt"
    ) if val_true == 1 else None
    if val_true == 0:
        search_dataset = dataset
    search_model = model.TFmodel(world.config, search_dataset)

    # SGFCF search variables: rho and gamma, with
    # beta_1=0.9*rho and beta_2=1.1*rho.
    # Shared normalization is searched separately as
    # (alpha_i, alpha_u, lambda_degree), where lambda_degree is shared.
    sgfcf_space = list(product(
        (0.7, 1.3), (0.1, 0.5, 1.0),
    )) if val_true == 1 else []
    normalization_space = list(product(
        (0.1, 0.3, 0.5), (0.1, 0.3, 0.5),
        (1.0, 5.0, 20.0),
    )) if val_true == 1 else []
    # DAN search tuple order: (reg_p, drop_p, xi).  Shared normalization
    # replaces DAN's former independent alpha/beta normalization parameters.
    dan_space = list(product(
        (20.0, 50.0), (0.2, 0.6), (0.1, 0.5),
    )) if val_true == 1 else []
    # Linear SGFCF+DAN mixing coefficient search values.
    dan_coefficients = (0.1, 0.5, 1.0) if val_true == 1 else ()

    def evaluate_configuration(recmodel, eval_dataset, sgfcf_params,
                               dan_params, coefficient):
        recmodel.k = int(sgfcf_params["k"])
        X = recmodel.build_combined_X(
            beta_1=sgfcf_params["beta_1"],
            beta_2=sgfcf_params["beta_2"],
            gamma=sgfcf_params["gamma"],
            dan_coefficient=coefficient,
            dan_params=dan_params,
            alpha_i=sgfcf_params["alpha_i"],
            lambda_i=sgfcf_params["lambda_degree"],
            alpha_u=sgfcf_params["alpha_u"],
            lambda_u=sgfcf_params["lambda_degree"],
        )
        return recmodel.Test_RatingMatrix(X, eval_dataset)

    # Joint validation search: every SGFCF configuration is evaluated with
    # every DAN configuration and every linear-mixing coefficient.
    best_sgfcf = None
    best_dan = None
    best_coefficient = None
    best_validation = None
    best_score = -np.inf
    total_candidates = (len(sgfcf_space) * len(normalization_space)
                        * len(search_k_values) * len(dan_space)
                        * len(dan_coefficients))
    candidate_index = 0
    for candidate_k in search_k_values:
        for rho, gamma in sgfcf_space:
            for alpha_i, alpha_u, lambda_degree in normalization_space:
                sgfcf_params = {
                    "rho": rho,
                    "beta_1": 0.9 * rho, "beta_2": 1.1 * rho,
                    "alpha_i": alpha_i, "alpha_u": alpha_u,
                    "lambda_degree": lambda_degree,
                    "gamma": gamma, "k": candidate_k,
                }
                for reg_p, drop_p, xi in dan_space:
                    dan_params = {"model": "RLAE_DAN", "reg_p": reg_p,
                                  "drop_p": drop_p, "xi": xi}
                    for coefficient in dan_coefficients:
                        result = evaluate_configuration(
                            search_model, search_dataset, sgfcf_params,
                            dan_params, coefficient
                        )
                        score = float(result["recall"][0])
                        if score > best_score:
                            best_score = score
                            best_sgfcf = dict(sgfcf_params)
                            best_dan = dict(dan_params)
                            best_coefficient = coefficient
                            best_validation = result
                        candidate_index += 1
                        if candidate_index % 25 == 0 or candidate_index == total_candidates:
                            world.cprint(
                                f"Joint search {candidate_index}/{total_candidates}: "
                                f"best Recall@20={best_score:.6f}"
                            )

    if val_true == 0:
        # No-search mode: use the documented default configuration and test
        # it directly on test.txt.
        best_sgfcf = {
            "rho": 1.0, "beta_1": 0.9, "beta_2": 1.1,
            "alpha_i": 0.5, "alpha_u": 0.5,
            "lambda_degree": 5.0,
            "gamma": 1.0,
            "k": int(input_k) if fixed_k == 1 else int(search_k_values[0]),
        }
        best_dan = {"model": "RLAE_DAN", "reg_p": 50.0,
                    "drop_p": 0.6, "xi": 0.1}
        best_coefficient = 1.0
        best_validation = None

    if val_true == 1:
        world.cprint("=== Validation / search results ===")
        world.cprint(f"SGFCF parameters: {best_sgfcf}")
        world.cprint(f"DAN parameters: {best_dan}")
        world.cprint(f"dan_coefficient: {best_coefficient}")
        world.cprint(f"combined validation: {best_validation}")
    else:
        world.cprint("=== No hyperparameter search (val_true=0) ===")



    # ---------------------------------------------------------
    # 3. Final test of the combined model
    # ---------------------------------------------------------
    test_X = Recmodel.build_combined_X(
        beta_1=best_sgfcf["beta_1"],
        beta_2=best_sgfcf["beta_2"],
        gamma=best_sgfcf["gamma"],
        dan_coefficient=best_coefficient,
        dan_params=best_dan,
        alpha_i=best_sgfcf["alpha_i"],
        lambda_i=best_sgfcf["lambda_degree"],
        alpha_u=best_sgfcf["alpha_u"],
        lambda_u=best_sgfcf["lambda_degree"],
    )
    world.cprint(f"=== Hyperparameter search concluded for dataset: {dataset_name} ===")
    world.cprint("=== Final test results ===")
    world.cprint(f"SGFCF parameters: {best_sgfcf}")
    world.cprint(f"DAN parameters: {best_dan}")
    world.cprint(f"dan_coefficient: {best_coefficient}")
    world.cprint(f"combined test: {Recmodel.Test_RatingMatrix(test_X, dataset)}")



finally:
    if world.tensorboard:
        w.close()
