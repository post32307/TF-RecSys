"""Compare score matrices across one model's main.py parameter space."""

import os
import sys
import itertools
import json

import numpy as np
import torch

import plot_world

sys.modules["world"] = plot_world

CORRELATION_MIN = -1.0
CORRELATION_MAX = 1.0
HEATMAP_FIGSIZE = (12, 12)


PARAMETERS = {
    "EASE": {"lambda_": 50.0},
    "SGFCF": {"beta_1": 1.0, "beta_2": 1.1, "alpha": 1.0,
              "eps": 0.5, "gamma": 0.0, "k": None},
    "BSPM": {"k": None, "alpha": 0.5, "beta": 0.5, "gamma": 1.0},
    "ChebyCF": {"K": 8, "phi": 1.5, "eta": 256, "alpha": 0.1,
                "beta": 0.2, "implementation_version": 2},
    "DAN": {"model": "RLAE_DAN", "reg_p": 20.0, "alpha": 0.2,
            "beta": 0.3, "drop_p": 0.6, "xi": 0.1},
    "GF-CF": {"k": None},
    "PGSP": {"k": None, "phi": 0.7},
    "TurboCF": {"alpha": 0.5, "power": 1.0, "filter": 1},
}
SCORER_NAMES = {
    "EASE": "S_EASE", "SGFCF": "S_SGFCF", "BSPM": "S_BSPM",
    "ChebyCF": "S_ChebyCF", "DAN": "S_DAN", "GF-CF": "S_GF_CF",
    "PGSP": "S_PGSP", "TurboCF": "S_TurboCF",
}
RANK_CANDIDATES = (16, 32, 64, 128, 256, 512, 1024)


def product_configs(grid):
    """Expand a parameter grid into dictionaries of parameter settings."""
    names = list(grid)
    return [dict(zip(names, values)) for values in itertools.product(
        *(grid[name] for name in names)
    )]


def save_heatmap(correlations, labels, output_path, title):
    """Save a Pearson heatmap with a shared, fixed [-1, 1] scale."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, Normalize

    figure, axis = plt.subplots(figsize=HEATMAP_FIGSIZE)
    cmap = ListedColormap(np.vstack([
        plt.get_cmap("magma")(np.linspace(0, 1, 128)),
        plt.get_cmap("viridis_r")(np.linspace(0, 1, 128)),
    ]))
    axis.imshow(
        np.ma.masked_invalid(correlations), cmap=cmap,
        vmin=CORRELATION_MIN, vmax=CORRELATION_MAX,
    )
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title(title, fontsize=32, fontweight="semibold", pad=14)
    positive_scale = plt.cm.ScalarMappable(
        norm=Normalize(vmin=0, vmax=CORRELATION_MAX),
        cmap="viridis_r",
    )
    colorbar = figure.colorbar(
        positive_scale, ax=axis, fraction=0.046, pad=0.04,
    )
    colorbar.set_label("Pearson correlation", fontsize=24)
    colorbar.ax.tick_params(labelsize=21.6)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved heatmap: {output_path}")


SWEEP_CONFIGS = {
    "EASE": product_configs({"lambda_": [50, 100, 200, 500, 1000, 2000]}),
    "TurboCF": product_configs({
        "alpha": [0.1, 0.3, 0.5, 0.7, 0.9],
        "power": [0.5, 1, 1.5, 2], "filter": [1, 2, 3],
    }),
}

# FaGSP and HiGSP have no scorer in this local TFmodel implementation.
UNSUPPORTED_SWEEP_MODELS = {"FaGSP", "HiGSP"}


def load_test_dataset(loader_class, dataset_path):
    """Load the dataset with test.txt selected as the evaluation target."""
    previous_eval_file = os.environ.get("TFRECSYS_EVAL_FILE")
    os.environ["TFRECSYS_EVAL_FILE"] = "test.txt"
    try:
        return loader_class(path=dataset_path)
    finally:
        if previous_eval_file is None:
            os.environ.pop("TFRECSYS_EVAL_FILE", None)
        else:
            os.environ["TFRECSYS_EVAL_FILE"] = previous_eval_file


def score_model(recmodel, model_name, parameters, cache):
    """Compute a score matrix for one model and parameter setting."""
    return getattr(recmodel, SCORER_NAMES[model_name])(
        cache=cache, **parameters
    )


def average_rowwise_correlations(score_matrices):
    """Average per-user Pearson correlations for every score-matrix pair."""
    matrices = [np.asarray(scores, dtype=np.float64) for scores in score_matrices]
    if not matrices or any(matrix.shape != matrices[0].shape for matrix in matrices):
        raise ValueError("All score matrices must have the same shape.")

    correlations = np.full((len(matrices), len(matrices)), np.nan)
    valid_counts = np.zeros((len(matrices), len(matrices)), dtype=int)
    for i, first in enumerate(matrices):
        for j in range(i, len(matrices)):
            second = matrices[j]
            valid = np.isfinite(first).all(axis=1) & np.isfinite(second).all(axis=1)
            first_rows = first[valid]
            second_rows = second[valid]
            if not len(first_rows):
                continue

            first_centered = first_rows - first_rows.mean(axis=1, keepdims=True)
            second_centered = second_rows - second_rows.mean(axis=1, keepdims=True)
            first_norm = np.linalg.norm(first_centered, axis=1)
            second_norm = np.linalg.norm(second_centered, axis=1)
            nonconstant = (first_norm > 0) & (second_norm > 0)
            if not np.any(nonconstant):
                continue

            row_correlations = np.sum(
                first_centered[nonconstant] * second_centered[nonconstant], axis=1
            ) / (first_norm[nonconstant] * second_norm[nonconstant])
            correlations[i, j] = correlations[j, i] = row_correlations.mean()
            valid_counts[i, j] = valid_counts[j, i] = int(nonconstant.sum())
    return correlations, valid_counts


def model_configurations(model_name, parameter_name, values, max_rank):
    """Return this model's sweep configurations and their display labels."""
    if model_name in UNSUPPORTED_SWEEP_MODELS:
        raise ValueError(
            f"{model_name} has no scorer in this script's local TFmodel implementation."
        )
    ranks = [rank for rank in RANK_CANDIDATES if rank <= max_rank]
    if not ranks:
        ranks = [max_rank]
    if model_name == "BSPM":
        configurations = product_configs({
            "k": ranks, "alpha": [0.3, 0.5, 0.7],
            "beta": [0.3, 0.5, 0.7], "gamma": [0.5, 1.0],
        })
    elif model_name == "GF-CF":
        configurations = [{"k": rank} for rank in ranks]
    elif model_name == "PGSP":
        configurations = product_configs({
            "k": ranks, "phi": [0.0, 0.25, 0.5, 0.75, 1.0],
        })
    elif model_name == "SGFCF":
        configurations = product_configs({
            "alpha": [0.0, 1.0, 20.0], "eps": [0.1, 0.3, 0.5],
            "k": ranks, "beta_1": [0.8, 1.0],
            "beta_2": [1.1, 1.3], "gamma": [0.1, 0.5, 1.0],
        })
    elif model_name == "ChebyCF":
        dimension = max_rank + 1
        configurations = product_configs({
            "eta": [eta for eta in (256, 512) if eta < dimension],
            "K": [8, 32], "phi": [1.5, 10.0, 15.0],
            "alpha": [0.1, 0.2], "beta": [0.2, 0.4],
            "implementation_version": [2],
        })
    elif model_name == "DAN":
        configurations = product_configs({
            "model": ["RLAE_DAN"], "reg_p": [20.0, 50.0],
            "alpha": [0.1, 0.3, 0.5], "beta": [0.1, 0.3, 0.5],
            "drop_p": [0.2, 0.6], "xi": [0.1, 0.3, 0.5],
        })
    else:
        configurations = SWEEP_CONFIGS[model_name]
    if parameter_name:
        if parameter_name not in PARAMETERS[model_name]:
            valid_parameters = ", ".join(PARAMETERS[model_name])
            raise ValueError(
                f"{parameter_name!r} is not a parameter for {model_name}; "
                f"choose from: {valid_parameters}"
            )
        if not values or len(values) < 2:
            raise ValueError("Provide at least two values with --values.")
        configurations = [{parameter_name: value} for value in values]
    elif values:
        raise ValueError("Use --values together with --parameter.")
    if not configurations:
        raise ValueError(
            f"No {model_name} parameter configurations are valid for this dataset."
        )
    return configurations


def print_model_correlations(recmodel, model_name, max_rank, parameter_name,
                             values, cache):
    """Score one model's configurations and print their correlation table."""
    configurations = model_configurations(
        model_name, parameter_name, values, max_rank
    )
    base_parameters = PARAMETERS[model_name].copy()
    for name in ("k", "eta"):
        if name in base_parameters and base_parameters[name] is not None:
            base_parameters[name] = min(base_parameters[name], max_rank)

    labels, score_matrices = [], []
    for index, config in enumerate(configurations, start=1):
        parameters = {**base_parameters, **config}
        for name in ("k", "eta"):
            if name in parameters and parameters[name] is not None:
                parameters[name] = min(int(parameters[name]), max_rank)
        label = f"ID{index:04d}"
        print(f"Calculating {model_name} scores ({index}/{len(configurations)}): {label}")
        scores = score_model(recmodel, model_name, parameters, cache)
        labels.append(label)
        score_matrices.append(scores.detach().cpu().numpy())

    correlations, valid_counts = average_rowwise_correlations(score_matrices)
    print(f"\nAverage row-wise Pearson correlation for {model_name}:")
    print("              " + "  ".join(f"{label:>16}" for label in labels))
    for label, row in zip(labels, correlations):
        print(f"{label:>14}  " + "  ".join(f"{value:16.6f}" for value in row))
    print("\nValid user-row counts for each configuration pair:")
    print("              " + "  ".join(f"{label:>16}" for label in labels))
    for label, row in zip(labels, valid_counts):
        print(f"{label:>14}  " + "  ".join(f"{count:16d}" for count in row))

    output_dir = os.path.join(plot_world.CODE_PATH, "pearson_within")
    os.makedirs(output_dir, exist_ok=True)
    suffix = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in (parameter_name or "main_parameter_space")
    )
    output_path = os.path.join(
        output_dir,
        f"{plot_world.dataset}_seed_{plot_world.seed}_{model_name}_{suffix}.png",
    )
    mapping_path = os.path.splitext(output_path)[0] + "_parameters.json"
    with open(mapping_path, "w", encoding="utf-8") as mapping_file:
        json.dump(
            {label: config for label, config in zip(labels, configurations)},
            mapping_file, indent=2,
        )
    dataset_title = "-".join(
        part[:1].upper() + part[1:] for part in plot_world.dataset.split("-")
    )
    if plot_world.dataset.lower() == "lastfm":
        dataset_title = "LastFM"
    save_heatmap(
        correlations, labels, output_path,
        f"Within-Model Corr.({model_name} / {dataset_title})",
    )
    print(f"Saved parameter ID mapping: {mapping_path}")


def main():
    import dataloader
    import model

    model_names = list(dict.fromkeys(plot_world.args.models))
    if not model_names:
        raise ValueError("Specify at least one model with --models.")

    data_root = os.environ.get(
        "TFRECSYS_DATA_ROOT",
        os.path.join(plot_world.ROOT_PATH, "data", "split_dataset",
                     f"00_dataset_seed_{plot_world.seed}"),
    )
    dataset_path = os.path.join(data_root, plot_world.dataset)
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

    torch.set_num_threads(4)
    dataset = load_test_dataset(dataloader.Loader, dataset_path)
    recmodel = model.TFmodel(plot_world.config, dataset)
    max_rank = max(1, min(recmodel.R.shape) - 1)
    for model_name in model_names:
        cache = {
            "eigen": {}, "max_eigen_entries": 1,
            "normalized": {}, "max_normalized_entries": 1,
            "convolution": {}, "max_convolution_entries": 1,
            "dan": {}, "max_dan_entries": 1,
        }
        try:
            print_model_correlations(
                recmodel, model_name, max_rank,
                plot_world.args.parameter, plot_world.args.values, cache,
            )
        except (KeyError, ValueError) as error:
            print(f"\nSkipping {model_name}: {error}")


if __name__ == "__main__":
    main()
