"""Compare selected baseline score matrices by user and item groups."""

import os
import sys

import numpy as np
import torch

import plot_world as plot_world
sys.modules["world"] = plot_world
import helpers


torch.set_num_threads(4)
helpers.set_seed(plot_world.seed)


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


def default_parameters(model_name, k):
    """Return the scoring parameters used for the selected baseline."""
    parameters = {
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
        return parameters[model_name]
    except KeyError as error:
        raise ValueError(f"Unsupported baseline model: {model_name}") from error


def score_baseline(recmodel, model_name, parameters, cache):
    """Compute a baseline user-item score matrix."""
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
        scorer = scorers[model_name]
    except KeyError as error:
        raise ValueError(f"Unsupported baseline model: {model_name}") from error
    return scorer(cache=cache, **parameters)


def _cluster_items(interaction_matrix, n_groups, seed):
    """Cluster items by cosine similarity of their original interaction columns."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import normalize

    item_vectors = interaction_matrix.T.tocsr().astype(np.float32)
    normalized_item_vectors = normalize(item_vectors, norm="l2", axis=1)
    n_item_groups = min(max(1, int(n_groups)), item_vectors.shape[0])
    return KMeans(
        n_clusters=n_item_groups, random_state=seed, n_init=10
    ).fit_predict(normalized_item_vectors)


def _cluster_users(interaction_matrix, n_groups, seed):
    """Cluster users by cosine similarity of their original interaction rows."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import normalize

    user_vectors = interaction_matrix.tocsr().astype(np.float32)
    normalized_user_vectors = normalize(user_vectors, norm="l2", axis=1)
    n_user_groups = min(max(1, int(n_groups)), user_vectors.shape[0])
    return KMeans(
        n_clusters=n_user_groups, random_state=seed, n_init=10
    ).fit_predict(normalized_user_vectors)


def _group_id_and_item_score(scores, item_labels, user_labels,
                             item_group_index=0):
    """Summarize row-normalized scores using shared user and item groups."""

    scores = np.nan_to_num(
        np.asarray(scores, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )
    row_min = scores.min(axis=1, keepdims=True)
    row_range = scores.max(axis=1, keepdims=True) - row_min
    row_normalized_scores = np.divide(
        scores - row_min,
        row_range,
        out=np.zeros_like(scores),
        where=row_range != 0,
    )

    if scores.shape[1] != len(item_labels):
        raise ValueError("Score matrix item count does not match the interaction matrix.")
    if scores.shape[0] != len(user_labels):
        raise ValueError("Score matrix user count does not match the interaction matrix.")

    if item_group_index not in item_labels:
        available_groups = np.unique(item_labels).tolist()
        raise IndexError(
            f"Item group {item_group_index} is unavailable. "
            f"Available item group IDs: {available_groups}."
        )

    reference_group_scores = row_normalized_scores[
        :, item_labels == item_group_index
    ].mean(axis=1)

    user_group_ids = np.unique(user_labels)
    average_item_scores = np.asarray([
        reference_group_scores[user_labels == group_id].mean()
        for group_id in user_group_ids
    ])
    return user_group_ids, average_item_scores


def plot_degree_score(model_scores, interaction_matrix,
                      n_user_groups=10, n_item_groups=5, seed=2020,
                      item_group_index=0, output_dir=None):
    """Plot group scores and save them for the selected item group(s)."""
    import matplotlib.pyplot as plt

    item_labels = _cluster_items(interaction_matrix, n_item_groups, seed)
    user_labels = _cluster_users(interaction_matrix, n_user_groups, seed)
    available_groups = np.unique(item_labels)

    def plot_group(ax, group_id):
        palette = [
            "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b",
            "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        ]
        grouped_scores = {}
        group_ids = None
        for model_name, scores in model_scores.items():
            model_group_ids, model_group_scores = _group_id_and_item_score(
                scores, item_labels, user_labels,
                item_group_index=int(group_id),
            )
            if group_ids is None:
                group_ids = model_group_ids
            elif not np.array_equal(group_ids, model_group_ids):
                raise ValueError("Models produced inconsistent user group IDs.")
            grouped_scores[model_name] = model_group_scores

        # Rank user groups by their combined average row-normalized score
        # across every model, highest first, and use the same order per series.
        total_group_scores = np.sum(list(grouped_scores.values()), axis=0)
        order = np.argsort(-total_group_scores)
        ordered_group_ids = group_ids[order]
        x_positions = np.arange(len(ordered_group_ids))

        other_color_index = 0
        for model_name, group_scores in grouped_scores.items():
            if model_name == "EASE":
                color = "red"
            elif model_name == "SGFCF":
                color = "blue"
            else:
                color = palette[other_color_index % len(palette)]
                other_color_index += 1
            ax.plot(
                x_positions, group_scores[order], color=color, marker="o",
                linestyle="-", label=model_name,
            )
        ax.set_xticks(x_positions, ordered_group_ids)
        ax.tick_params(axis="both", labelsize=12)
        ax.set_title(f"Item Group {group_id}", fontsize=18)
        ax.set_xlabel(
            "User Group ID (Ordered by combined model score)", fontsize=14
        )
        ax.set_ylabel("Average Row-normalized Score", fontsize=14)
        ax.legend(
            loc="center left", bbox_to_anchor=(1.02, 0.5),
            fontsize=14, frameon=False, borderaxespad=0,
        )

    def save_figure(figure, user_group_count, group_id, item_group_count):
        if output_dir is None:
            return
        os.makedirs(output_dir, exist_ok=True)
        dataset_name = getattr(plot_world, "dataset", "dataset")
        model_suffix = "".join(
            "".join(part.capitalize() for part in model_name.replace("-", " ").split())
            for model_name in model_scores
        )
        output_name = (
            f"{dataset_name}_seed_{seed}_{user_group_count}_usergroups_"
            f"item_group_{group_id}_of_{item_group_count}_itemgroups_"
            f"{model_suffix}.png"
        )
        output_path = os.path.join(output_dir, output_name)
        figure.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"Saved plot: {output_path}")

    if item_group_index == "all":
        for group_id in available_groups:
            figure, ax = plt.subplots(figsize=(10, 4))
            plot_group(ax, group_id)
            figure.tight_layout(rect=(0, 0, 0.78, 1))
            save_figure(figure, n_user_groups, group_id, n_item_groups)
    else:
        figure, ax = plt.subplots(figsize=(12, 4.5))
        plot_group(ax, item_group_index)
        figure.tight_layout(rect=(0, 0, 0.78, 1))
        save_figure(figure, n_user_groups, item_group_index, n_item_groups)
    plt.show()


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

    dataset = load_test_dataset(dataloader.Loader, dataset_path)
    recmodel = model.TFmodel(plot_world.config, dataset)
    k = min(plot_world.config["k"], max(1, min(recmodel.R.shape) - 1))
    cache = {
        "eigen": {}, "max_eigen_entries": 1,
        "normalized": {}, "max_normalized_entries": 1,
        "convolution": {}, "max_convolution_entries": 1,
        "dan": {}, "max_dan_entries": 1,
    }
    model_scores = {}
    for model_name in dict.fromkeys(plot_world.models):
        print(f"Calculating {model_name} scores...")
        model_scores[model_name] = score_baseline(
            recmodel, model_name, default_parameters(model_name, k), cache
        ).detach().cpu().numpy()
    print("Scores calculated")

    item_group_index = "all"
    while True:
        try:
            plot_degree_score(
                model_scores, recmodel.R,
                n_user_groups=plot_world.n_user_groups,
                n_item_groups=plot_world.n_item_groups,
                seed=plot_world.seed,
                item_group_index=item_group_index,
                output_dir=os.path.join(plot_world.CODE_PATH, "plots"),
            )
        except IndexError as error:
            print(error)

        response = input(
            "Enter an item group ID, 'all', or any other input to exit: "
        ).strip()
        if response.lower() == "all":
            item_group_index = "all"
            continue
        if not response.isdigit():
            break
        item_group_index = int(response)


if __name__ == "__main__":
    main()
