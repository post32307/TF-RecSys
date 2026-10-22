"""Compute pairwise Pearson correlations between selected model score matrices."""

import os
import sys

import numpy as np
import torch

import plot_world

sys.modules["world"] = plot_world

CORRELATION_CMAP = [(0.0, "green"), (0.5, "red"), (1.0, "blue")]
CORRELATION_MIN = -1.0
CORRELATION_MAX = 1.0
HEATMAP_FIGSIZE = (12, 12)


def save_heatmap(correlations, labels, output_path, title):
    """Save a Pearson heatmap with a shared, fixed [-1, 1] scale."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    figure, axis = plt.subplots(figsize=HEATMAP_FIGSIZE)
    cmap = LinearSegmentedColormap.from_list(
        "pearson_green_red_blue", CORRELATION_CMAP
    )
    image = axis.imshow(
        np.ma.masked_invalid(correlations), cmap=cmap,
        vmin=CORRELATION_MIN, vmax=CORRELATION_MAX,
    )
    axis.set_xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
    axis.set_yticks(np.arange(len(labels)), labels)
    axis.set_title(title, fontsize=16, fontweight="semibold", pad=14)
    figure.colorbar(image, ax=axis, label="Pearson correlation",
                    fraction=0.046, pad=0.04)
    if len(labels) <= 15:
        for row in range(len(labels)):
            for column in range(len(labels)):
                value = correlations[row, column]
                text = "nan" if not np.isfinite(value) else f"{value:.2f}"
                axis.text(column, row, text, ha="center", va="center", fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved heatmap: {output_path}")


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


def model_parameters(model_name, k):
    """Return the parameter set used to score one selected model."""
    parameter_sets = {
        "EASE": {"lambda_": 50.0},
        "SGFCF": {"beta_1": 1.0, "beta_2": 1.1, "alpha": 1.0,
                  "eps": 0.5, "gamma": 0.0, "k": k},
        "BSPM": {"k": k, "alpha": 0.5, "beta": 0.5, "gamma": 1.0},
        "ChebyCF": {"K": 8, "phi": 1.5, "eta": min(256, k),
                    "alpha": 0.1, "beta": 0.2,
                    "implementation_version": 2},
        "DAN": {"model": "RLAE_DAN", "reg_p": 20.0, "alpha": 0.2,
                "beta": 0.3, "drop_p": 0.6, "xi": 0.1},
        "GF-CF": {"k": k},
        "PGSP": {"k": k, "phi": 0.7},
        "TurboCF": {"alpha": 0.5, "power": 1.0, "filter": 1},
    }
    try:
        return parameter_sets[model_name]
    except KeyError as error:
        raise ValueError(f"Unsupported baseline model: {model_name}") from error


def score_model(recmodel, model_name, k, cache):
    """Compute one selected baseline's user-item score matrix."""
    parameters = model_parameters(model_name, k)
    scorers = {
        "EASE": recmodel.S_EASE,
        "SGFCF": recmodel.S_SGFCF,
        "BSPM": recmodel.S_BSPM,
        "ChebyCF": recmodel.S_ChebyCF,
        "DAN": recmodel.S_DAN,
        "GF-CF": recmodel.S_GF_CF,
        "PGSP": recmodel.S_PGSP,
        "TurboCF": recmodel.S_TurboCF,
    }
    try:
        return scorers[model_name](cache=cache, **parameters)
    except KeyError as error:
        raise ValueError(f"Unsupported baseline model: {model_name}") from error


def average_rowwise_correlations(score_matrices):
    """Average per-user Pearson correlations for every score-matrix pair."""
    matrices = [np.asarray(scores, dtype=np.float64) for scores in score_matrices]
    if not matrices or any(matrix.shape != matrices[0].shape for matrix in matrices):
        raise ValueError("All models must produce score matrices of the same shape.")

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


def main():
    import dataloader
    import model

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
    k = min(plot_world.config["k"], max(1, min(recmodel.R.shape) - 1))
    cache = {
        "eigen": {}, "max_eigen_entries": 1,
        "normalized": {}, "max_normalized_entries": 1,
        "convolution": {}, "max_convolution_entries": 1,
        "dan": {}, "max_dan_entries": 1,
    }

    model_names = list(dict.fromkeys(plot_world.models))
    score_matrices = []
    for name in model_names:
        print(f"Calculating {name} scores...")
        scores = score_model(recmodel, name, k, cache)
        score_matrices.append(scores.detach().cpu().numpy())

    correlations, valid_counts = average_rowwise_correlations(score_matrices)
    print("\nPairwise average row-wise Pearson correlation across users:")
    print("              " + "  ".join(f"{name:>10}" for name in model_names))
    for name, row in zip(model_names, correlations):
        values = "  ".join(f"{value:10.6f}" for value in row)
        print(f"{name:>12}  {values}")
    print("\nValid user-row counts for each model pair:")
    print("              " + "  ".join(f"{name:>10}" for name in model_names))
    for name, row in zip(model_names, valid_counts):
        print(f"{name:>12}  " + "  ".join(f"{count:10d}" for count in row))

    output_dir = os.path.join(plot_world.CODE_PATH, "pearson_inter")
    os.makedirs(output_dir, exist_ok=True)
    model_tag = "_".join(model_names).replace("/", "_")
    output_path = os.path.join(
        output_dir,
        f"{plot_world.dataset}_seed_{plot_world.seed}_{model_tag}.png",
    )
    save_heatmap(
        correlations, model_names, output_path,
        f"Inter-model average row-wise Pearson: {plot_world.dataset}",
    )


if __name__ == "__main__":
    main()
