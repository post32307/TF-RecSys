"""Function-grouped smoke tests for the six ``TFmodel`` Model Functions."""

import numpy as np
import scipy.sparse as sp
import torch

import model
import world


class CustomDataset:
    """Minimal dataset constructed from a user-provided interaction matrix."""

    def __init__(self, interactions):
        matrix = np.asarray(interactions, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError("interactions must be a two-dimensional matrix")
        if matrix.size >= 40:
            raise ValueError("custom datasets must contain fewer than 40 elements")
        self.UserItemNet = sp.csr_matrix(matrix)
        self.n_users, self.m_items = self.UserItemNet.shape


def compact(value):
    """Convert tensors/sparse matrices/containers to readable output text."""
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    if sp.issparse(value):
        matrix = np.array2string(value.toarray(), precision=5, separator=", ")
        return f"shape={value.shape}, nnz={value.nnz}, data=\n{matrix}"
    if isinstance(value, np.ndarray):
        if value.dtype == bool:
            return repr(value.tolist())
        return np.array2string(value, precision=5, separator=", ")
    if isinstance(value, dict):
        return "{" + ", ".join(f"{key}={compact(item)}" for key, item in value.items()) + "}"
    if isinstance(value, (tuple, list)):
        return "(" + ", ".join(compact(item) for item in value) + ")"
    return repr(value)


def print_pair(label, value):
    """Print a labelled value with matrix rows on separate lines."""
    indent = " " * 8
    label_width = 42
    value_indent = indent + " " * (label_width + 4)
    lines = compact(value).splitlines()
    print(f"{indent}{label:<{label_width}}:    {lines[0]}")
    for line in lines[1:]:
        print(f"{value_indent}{line}")


def run_dataset(index, interactions):
    dataset = CustomDataset(interactions)
    config = dict(world.config)
    config.update({"device": "cpu", "k": min(2, dataset.n_users + dataset.m_items - 1)})
    recmodel = model.TFmodel(config, dataset)
    params = {"beta_1": 0.9, "beta_2": 1.1, "alpha_i": 0.5,
              "lambda_i": 1.0, "alpha_u": 0.5, "lambda_u": 1.0}

    outputs = {}

    item_cooccurrence = recmodel.R.T.dot(recmodel.R)
    item_graph = recmodel._threshold_offdiag_graph(item_cooccurrence)
    outputs["_threshold_offdiag_graph"] = (
        {"matrix": item_cooccurrence,
         "diagonal_removed": True, "entries_thresholded_at": 1},
        {"graph": item_graph}
    )

    nodes = recmodel.R[0].indices
    marker = np.zeros(dataset.m_items, dtype=np.int32)
    induced_edges, stamp = recmodel._induced_edges(item_graph, nodes, marker, 0)
    outputs["_induced_edges"] = (
        {"graph": item_graph, "nodes": nodes.tolist(),
         "marker": marker, "stamp": 0},
        {"edges": int(induced_edges), "stamp": int(stamp)},
    )

    user_h, item_h = recmodel.compute_homophily()
    outputs["compute_homophily"] = (
        {"R": recmodel.R, "R_csc": recmodel.R.tocsc(),
         "item_graph": recmodel._threshold_offdiag_graph(item_cooccurrence),
         "user_graph": recmodel._threshold_offdiag_graph(recmodel.R.dot(recmodel.R.T))},
        {"user_homophily": user_h, "item_homophily": item_h},
    )

    eigenvalues = torch.linspace(0.25, 1.0, config["k"])
    response = recmodel.individual_weight(
        eigenvalues, torch.cat((user_h, item_h)),
        beta_1=params["beta_1"], beta_2=params["beta_2"],
    )
    homophily = torch.cat((user_h, item_h))
    minimum, maximum = homophily.min(), homophily.max()
    outputs["individual_weight"] = (
        {"value": eigenvalues, "homophily": homophily,
         "beta_1": params["beta_1"], "beta_2": params["beta_2"],
         "minimum_homophily": minimum, "maximum_homophily": maximum},
        {"response": response},
    )

    dan_params = {"model": "RLAE_DAN", "reg_p": 5.0,
                  "drop_p": 0.2, "xi": 0.1,
                  "alpha_i": params["alpha_i"], "lambda_i": params["lambda_i"],
                  "alpha_u": params["alpha_u"], "lambda_u": params["lambda_u"]}
    dan_state = recmodel.fit_dan(interactions=recmodel.R, **dan_params)
    normalized_dan_input = dan_state["interactions"]
    item_counts = np.asarray(normalized_dan_input.sum(axis=0)).ravel()
    regularization = dan_params["reg_p"] + (
        dan_params["drop_p"] / (1.0 - dan_params["drop_p"])
    ) * item_counts
    outputs["fit_dan"] = (
        {**dan_params, "normalized_interactions": normalized_dan_input,
         "item_counts": item_counts, "regularization": regularization},
        {"weights": dan_state["weights"], "cold_items": dan_state["cold_items"]}
    )

    combined_params = {**params, "gamma": 0.0, "dan_coefficient": 0.5,
                       "dan_params": {key: value for key, value in dan_params.items()
                                      if key not in {"alpha_i", "lambda_i", "alpha_u", "lambda_u"}}}
    combined = recmodel.build_combined_X(**combined_params)
    outputs["build_combined_X"] = (
        {**combined_params, "normalized_interaction": normalized_dan_input,
         "eigenvalues": eigenvalues, "homophily": homophily},
        {"X": combined}
    )
    return dataset, recmodel.R, outputs


def main():
    custom_inputs = [
        np.array([
            [1, 1, 1, 0],
            [1, 1, 0, 1],
            [1, 1, 1, 1],
        ], dtype=np.float32),                                      # 3 x 4 = 12
        np.array([
            [1, 1, 1, 0, 0],
            [1, 1, 0, 1, 0],
            [1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1],
        ], dtype=np.float32),                                      # 4 x 5 = 20
        np.array([
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 0, 1, 0],
            [1, 1, 0, 1, 1, 0],
            [0, 1, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1],
        ], dtype=np.float32),                                      # 5 x 6 = 30
    ]
    grouped = {}
    for index, interactions in enumerate(custom_inputs, start=1):
        dataset, original_matrix, outputs = run_dataset(index, interactions)
        for function_name, pair in outputs.items():
            grouped.setdefault(function_name, []).append(
                (index, dataset, original_matrix, pair[0], pair[1])
            )

    # Functions are grouped first; each dataset then shows input and output on
    # one line. Two tab stops are used on both sides of the separating colon.
    for function_name, records in grouped.items():
        print(f"\n{'=' * 10}{function_name}{'=' * 40}")
        for index, dataset, original_matrix, function_input, function_output in records:
            print(f"Dataset {index} ({dataset.n_users}x{dataset.m_items})")
            print("Original dataset:")
            print_pair("original_dataset", original_matrix)
            print("Function input/output:")
            print_pair(f"{function_name}.input", function_input)
            print_pair(f"{function_name}.output", function_output)


if __name__ == "__main__":
    main()
