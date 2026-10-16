"""SGFCF models with frequency-selective and generalized normalization variants."""

from collections import OrderedDict
import os
import gc
from itertools import product

import numpy as np
import scipy.sparse as sp
import torch
from scipy.linalg import eigh
from scipy.sparse.linalg import LinearOperator, eigsh, svds

import world


COEFFICIENT_VALUES = (0.0, 0.1, 0.5, 1.0)
BAND_LOW = 0
BAND_MIDDLE = 1
BAND_HIGH = 2
DENSE_GRAM_HARD_MAX_DIM = 8192
DENSE_GRAM_MAX_BYTES = 2 * 1024 ** 3
_REPORTED_SOLVER_CHOICES = set()

# Directed user/item normalization pairs for generalized-normalization SGFCF.
# Every candidate uses alpha_1 != alpha_2 and eps_1 != eps_2.
ASYMMETRIC_NORMALIZATION_SPACE = (
    {"alpha_1": 0.0, "eps_1": 0.28, "alpha_2": 19.0, "eps_2": 0.5},
    {"alpha_1": 19.0, "eps_1": 0.5, "alpha_2": 0.0, "eps_2": 0.28},
    {"alpha_1": 0.0, "eps_1": 0.37, "alpha_2": 19.0, "eps_2": 0.5},
    {"alpha_1": 19.0, "eps_1": 0.5, "alpha_2": 0.0, "eps_2": 0.37},
    {"alpha_1": 2.3, "eps_1": 0.5, "alpha_2": 0.0, "eps_2": 0.28},
    {"alpha_1": 0.0, "eps_1": 0.28, "alpha_2": 2.3, "eps_2": 0.5},
    {"alpha_1": 5.3, "eps_1": 0.5, "alpha_2": 0.0, "eps_2": 0.37},
    {"alpha_1": 0.0, "eps_1": 0.37, "alpha_2": 5.3, "eps_2": 0.5},
)

# Parameter families used by the repository's original SGFCF sweep.
SGFCF_SEARCH_SPACE = (
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

DAN_DEFAULT_PARAMS = {"reg_p": 20.0, "alpha": 0.2, "beta": 0.3, "drop_p": 0.6, "xi": 0.1}
DAN_DATASET_PARAMS = {
    "amazon-electronics": {"reg_p": 50.0, "alpha": 0.2, "beta": 0.3, "drop_p": 0.6},
    "lastfm": {"reg_p": 50.0, "alpha": 0.3, "beta": 0.3, "drop_p": 0.7},
    "citeulike": {"reg_p": 10.0, "alpha": 0.2, "beta": 0.2, "drop_p": 0.6},
    "epinions": {"reg_p": 50.0, "alpha": 0.2, "beta": 0.3, "drop_p": 0.7},
    "yelp": {"reg_p": 50.0, "alpha": 0.3, "beta": 0.2, "drop_p": 0.7},
    "douban-book": {"reg_p": 10.0, "alpha": 0.2, "beta": 0.2, "drop_p": 0.4},
    "amazon-CD": {"reg_p": 50.0, "alpha": 0.2, "beta": 0.3, "drop_p": 0.7},
    "news": {"reg_p": 50.0, "alpha": 0.2, "beta": 0.2, "drop_p": 0.4},
    "movielens": {"reg_p": 20.0, "alpha": 0.2, "beta": 0.2, "drop_p": 0.4},
    "gowalla": {"reg_p": 20.0, "alpha": 0.2, "beta": 0.3, "drop_p": 0.6},
    "yelp2018": {"reg_p": 50.0, "alpha": 0.3, "beta": 0.3, "drop_p": 0.7},
}

# Explicit search spaces used by the four-model pipeline.
SGFCF_SEARCH_SPACE = tuple(
    {"beta_1": b1, "beta_2": b2, "alpha": a, "eps": e, "gamma": g}
    for b1, b2, a, e, g in product(
        (0.8, 1.0), (1.1, 1.3), (0.0, 1.0, 20.0),
        (0.1, 0.3, 0.5), (0.1, 0.3, 0.5)
    )
)
DAN_SEARCH_SPACE = tuple(
    {"model": "RLAE_DAN", "reg_p": 50.0, "alpha": a,
     "beta": b, "drop_p": d, "xi": x}
    for a, b, d, x in product(
        (0.1, 0.3, 0.5), (0.1, 0.3, 0.5),
        (0.2, 0.6), (0.1, 0.3, 0.5)
    )
)
DAN_COEFFICIENT_VALUES = (0.1, 0.5, 1.0)

SGFCF_DEFAULTS = {
    "amazon-electronics": {"beta_1": 1.0, "beta_2": 1.3, "alpha": 19.0, "eps": 0.5, "gamma": 0.5},
    "lastfm": {"beta_1": 0.7, "beta_2": 1.1, "alpha": 0.0, "eps": 0.28, "gamma": 1.5},
    "citeulike": {"beta_1": 0.7, "beta_2": 1.1, "alpha": 0.0, "eps": 0.28, "gamma": 1.5},
    "epinions": {"beta_1": 1.0, "beta_2": 1.3, "alpha": 19.0, "eps": 0.5, "gamma": 0.5},
    "yelp": {"beta_1": 1.0, "beta_2": 1.3, "alpha": 19.0, "eps": 0.5, "gamma": 0.5},
    "douban-book": {"beta_1": 1.0, "beta_2": 1.3, "alpha": 19.0, "eps": 0.5, "gamma": 0.5},
    "amazon-CD": {"beta_1": 1.0, "beta_2": 1.3, "alpha": 19.0, "eps": 0.5, "gamma": 0.5},
    "news": {"beta_1": 0.9, "beta_2": 1.0, "alpha": 0.0, "eps": 0.37, "gamma": 0.3},
    "movielens": {"beta_1": 0.9, "beta_2": 1.0, "alpha": 0.0, "eps": 0.37, "gamma": 0.3},
    "gowalla": {"beta_1": 0.7, "beta_2": 1.1, "alpha": 0.0, "eps": 0.28, "gamma": 1.5},
    "yelp2018": {"beta_1": 0.9, "beta_2": 1.0, "alpha": 0.0, "eps": 0.37, "gamma": 0.3},
}


def load_dataset_split(loader_class, dataset_path, eval_file):
    """Load train.txt with an explicit validation or test target file."""
    previous = os.environ.get("TFRECSYS_EVAL_FILE")
    os.environ["TFRECSYS_EVAL_FILE"] = eval_file
    try:
        return loader_class(path=dataset_path)
    finally:
        if previous is None:
            os.environ.pop("TFRECSYS_EVAL_FILE", None)
        else:
            os.environ["TFRECSYS_EVAL_FILE"] = previous


def default_params(dataset_name, model_k):
    fallback = {"beta_1": 1.0, "beta_2": 1.0, "alpha": 0.0, "eps": 0.5, "gamma": 1.0}
    params = dict(SGFCF_DEFAULTS.get(dataset_name, fallback))
    params["k"] = int(model_k)
    return params


def _threshold_graph(matrix):
    graph = matrix.tocsr(copy=True)
    graph.setdiag(0)
    graph.eliminate_zeros()
    graph.data[graph.data <= 1] = 0
    graph.eliminate_zeros()
    graph.data = np.ones(graph.nnz, dtype=np.int8)
    return graph


def _induced_edge_count(graph, nodes, marker, stamp):
    if len(nodes) <= 1:
        return 0, stamp
    stamp += 1
    if stamp == np.iinfo(marker.dtype).max:
        marker.fill(0)
        stamp = 1
    marker[nodes] = stamp
    count = 0
    for node in nodes:
        neighbors = graph.indices[graph.indptr[node]:graph.indptr[node + 1]]
        count += np.count_nonzero(marker[neighbors] == stamp)
    return count, stamp


def compute_homophily(base_model):
    """Reproduce SGFCF's user and item local-homophily ratios."""
    interactions = base_model.R.astype(np.float32).tocsr()
    interactions_csc = interactions.tocsc()
    n_users, n_items = interactions.shape
    item_graph = _threshold_graph(interactions.T.dot(interactions))
    user_graph = _threshold_graph(interactions.dot(interactions.T))

    user_ratio = np.zeros(n_users, dtype=np.float32)
    marker = np.zeros(n_items, dtype=np.int32)
    stamp = 0
    for user in range(n_users):
        nodes = interactions[user].indices
        edges, stamp = _induced_edge_count(item_graph, nodes, marker, stamp)
        if len(nodes) > 1:
            user_ratio[user] = edges / (len(nodes) * (len(nodes) - 1))

    item_ratio = np.zeros(n_items, dtype=np.float32)
    marker = np.zeros(n_users, dtype=np.int32)
    stamp = 0
    for item in range(n_items):
        nodes = interactions_csc[:, item].indices
        edges, stamp = _induced_edge_count(user_graph, nodes, marker, stamp)
        if len(nodes) > 1:
            item_ratio[item] = edges / (len(nodes) * (len(nodes) - 1))

    return (
        torch.as_tensor(user_ratio, dtype=torch.float32, device=base_model.device),
        torch.as_tensor(item_ratio, dtype=torch.float32, device=base_model.device),
    )


def _svd_edge_band(matrix, rank, which):
    """Return singular triplets from one edge of the spectrum."""
    user_vectors, values, item_vectors_t = svds(
        matrix, k=rank, which=which, solver="propack"
    )
    order = np.argsort(values)[::-1]
    return user_vectors[:, order], values[order], item_vectors_t[order, :].T


def _middle_singular_triplets(matrix, rank, largest_value):
    """Find singular triplets nearest 0.5 * sigma_max without a full SVD."""
    use_user_gram = matrix.shape[0] <= matrix.shape[1]
    dimension = min(matrix.shape)

    def gram_matvec(vector):
        if use_user_gram:
            return matrix.dot(matrix.T.dot(vector))
        return matrix.T.dot(matrix.dot(vector))

    target = float(0.5 * largest_value) ** 2

    def proximity_matvec(vector):
        shifted = gram_matvec(vector) - target * vector
        return -(gram_matvec(shifted) - target * shifted)

    proximity = LinearOperator(
        (dimension, dimension), matvec=proximity_matvec, dtype=np.float32
    )
    _, gram_vectors = eigsh(proximity, k=rank, which="LA")
    gram_products = np.column_stack(
        [gram_matvec(gram_vectors[:, index]) for index in range(rank)]
    )
    squared_values = np.sum(gram_vectors * gram_products, axis=0)
    values = np.sqrt(np.clip(squared_values, 0.0, None))
    safe_values = np.maximum(values, 1e-12)

    if use_user_gram:
        user_vectors = gram_vectors
        item_vectors = matrix.T.dot(user_vectors) / safe_values
    else:
        item_vectors = gram_vectors
        user_vectors = matrix.dot(item_vectors) / safe_values

    order = np.argsort(np.abs(values - 0.5 * largest_value))
    return user_vectors[:, order], values[order], item_vectors[:, order]


def _dense_gram_components(matrix, rank, spectrum):
    """Use a fast dense Gram eigensolve when the smaller side is modest."""
    use_user_gram = matrix.shape[0] <= matrix.shape[1]
    if use_user_gram:
        gram = matrix.dot(matrix.T).toarray()
    else:
        gram = matrix.T.dot(matrix).toarray()
    squared_values, gram_vectors = eigh(
        gram, overwrite_a=True, check_finite=False, driver="evd"
    )
    values = np.sqrt(np.clip(squared_values, 0.0, None))
    dimension = len(values)

    if spectrum in ("largest", "largest_edge"):
        selected = np.arange(dimension - rank, dimension)[::-1]
        bands = np.full(rank, BAND_LOW, dtype=np.int8)
    else:
        divisor = 2 if spectrum == "two_bands" else 3
        band_rank = min(rank, max(1, dimension // divisor))
        low_indices = np.arange(dimension - band_rank, dimension)[::-1]
        high_indices = np.arange(band_rank)
        selected_parts = [low_indices]
        band_parts = [np.full(band_rank, BAND_LOW, dtype=np.int8)]
        if spectrum == "three_bands":
            available = np.ones(dimension, dtype=bool)
            available[low_indices] = False
            available[high_indices] = False
            candidates = np.flatnonzero(available)
            target = 0.5 * values[-1]
            middle_indices = candidates[
                np.argsort(np.abs(values[candidates] - target))[:band_rank]
            ]
            selected_parts.append(middle_indices)
            band_parts.append(
                np.full(len(middle_indices), BAND_MIDDLE, dtype=np.int8)
            )
        selected_parts.append(high_indices)
        band_parts.append(np.full(band_rank, BAND_HIGH, dtype=np.int8))
        selected = np.concatenate(selected_parts)
        bands = np.concatenate(band_parts)

    selected_values = values[selected]
    selected_gram_vectors = gram_vectors[:, selected]
    safe_values = np.maximum(selected_values, 1e-12)
    if use_user_gram:
        user_vectors = selected_gram_vectors
        item_vectors = matrix.T.dot(user_vectors) / safe_values
    else:
        item_vectors = selected_gram_vectors
        user_vectors = matrix.dot(item_vectors) / safe_values
    return user_vectors, selected_values, item_vectors, bands


def _available_memory_bytes():
    """Return available physical memory without adding a new dependency."""
    try:
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):
        return DENSE_GRAM_MAX_BYTES * 4


def choose_eigendecomposition(matrix, requested_rank, spectrum):
    """Choose dense-complete or sparse-partial decomposition from cost estimates."""
    dimension = min(matrix.shape)
    rank = min(max(1, int(requested_rank)), max(1, dimension - 1))
    divisor = 3 if spectrum == "three_bands" else 2
    if spectrum in ("two_bands", "three_bands"):
        band_rank = min(rank, max(1, dimension // divisor))
        selected_count = divisor * band_rank
        solver_count = divisor
        # Interior and smallest eigenpairs require many more Krylov iterations.
        iteration_factor = 1200.0
    else:
        selected_count = rank
        solver_count = 1
        iteration_factor = 100.0

    # Gram matrix, eigenvectors, LAPACK workspace, and temporary arrays.
    estimated_dense_bytes = 24 * dimension * dimension
    memory_budget = min(
        DENSE_GRAM_MAX_BYTES,
        int(0.30 * _available_memory_bytes()),
    )
    memory_safe = (
        dimension <= DENSE_GRAM_HARD_MAX_DIM
        and estimated_dense_bytes <= memory_budget
    )
    if not memory_safe:
        return "sparse_partial"

    dense_work = float(dimension) ** 3
    sparse_matvec_work = (
        solver_count * matrix.nnz * band_rank * iteration_factor
        if spectrum in ("two_bands", "three_bands")
        else matrix.nnz * rank * iteration_factor
    )
    sparse_orthogonalization_work = (
        solver_count * dimension * band_rank ** 2 * 10.0
        if spectrum in ("two_bands", "three_bands")
        else dimension * rank ** 2 * 10.0
    )
    sparse_work = sparse_matvec_work + sparse_orthogonalization_work
    coverage = selected_count / float(dimension)

    # Complete decomposition has very low overhead on modest Gram matrices.
    small_dense_override = (
        dimension <= 2048
        and (
            spectrum in ("two_bands", "three_bands")
            or coverage >= 0.05
        )
    )
    if small_dense_override or dense_work <= sparse_work:
        return "dense_complete"
    return "sparse_partial"


def compute_components(base_model, params, normalization=None,
                       spectrum="largest"):
    """Compute rank-k largest, two-band, or three-band spectral components."""
    valid_spectra = {"largest", "largest_edge", "two_bands", "three_bands"}
    if spectrum not in valid_spectra:
        raise ValueError(f"Unknown SGFCF spectrum selection: {spectrum}")
    interactions = base_model.R.astype(np.float32).tocsr()
    full_rank = min(interactions.shape)
    if full_rank < 2:
        raise ValueError("SGFCF requires at least two users and two items.")
    truncated_rank = min(max(1, int(base_model.k)), full_rank - 1)

    user_degree = np.asarray(interactions.sum(axis=1)).ravel()
    item_degree = np.asarray(interactions.sum(axis=0)).ravel()
    if normalization is None:
        alpha_1 = alpha_2 = params["alpha"]
        eps_1 = eps_2 = params["eps"]
    else:
        alpha_1 = normalization["alpha_1"]
        eps_1 = normalization["eps_1"]
        alpha_2 = normalization["alpha_2"]
        eps_2 = normalization["eps_2"]
    with np.errstate(divide="ignore"):
        d_user = 1.0 / np.power(user_degree + alpha_1, eps_1)
        d_item = 1.0 / np.power(item_degree + alpha_2, eps_2)
    d_user[~np.isfinite(d_user)] = 0.0
    d_item[~np.isfinite(d_item)] = 0.0
    normalized = sp.diags(d_user).dot(interactions).dot(sp.diags(d_item))
    normalized = normalized.astype(np.float32).tocsr()

    solver = choose_eigendecomposition(
        normalized, truncated_rank, spectrum
    )
    report_key = (normalized.shape, truncated_rank, spectrum, solver)
    if report_key not in _REPORTED_SOLVER_CHOICES:
        estimated_rank = truncated_rank
        if spectrum in ("two_bands", "three_bands"):
            divisor = 2 if spectrum == "two_bands" else 3
            estimated_rank = min(truncated_rank, max(1, full_rank // divisor))
        world.cprint(
            f"Eigendecomposition strategy: {solver}, "
            f"shape={normalized.shape}, spectrum={spectrum}, "
            f"rank_per_band={estimated_rank}"
        )
        _REPORTED_SOLVER_CHOICES.add(report_key)

    if solver == "dense_complete":
        user_vectors, values, item_vectors, bands = _dense_gram_components(
            normalized, truncated_rank, spectrum
        )
    elif spectrum in ("largest", "largest_edge"):
        if spectrum == "largest_edge":
            user_vectors, values, item_vectors = _svd_edge_band(
                normalized, truncated_rank, "LM"
            )
        else:
            svd_rank = min(truncated_rank + 200, full_rank - 1)
            user_vectors, values, item_vectors_t = svds(
                normalized, k=svd_rank
            )
            order = np.argsort(values)[::-1][:truncated_rank]
            user_vectors = user_vectors[:, order]
            values = values[order]
            item_vectors = item_vectors_t[order, :].T
        bands = np.full(len(values), BAND_LOW, dtype=np.int8)
    elif spectrum in ("two_bands", "three_bands"):
        divisor = 2 if spectrum == "two_bands" else 3
        band_rank = min(max(1, int(base_model.k)), max(1, full_rank // divisor))
        low_u, low_values, low_v = _svd_edge_band(
            normalized, band_rank, "LM"
        )
        high_u, high_values, high_v = _svd_edge_band(
            normalized, band_rank, "SM"
        )
        user_parts = [low_u]
        value_parts = [low_values]
        item_parts = [low_v]
        band_parts = [np.full(band_rank, BAND_LOW, dtype=np.int8)]
        if spectrum == "three_bands":
            middle_u, middle_values, middle_v = _middle_singular_triplets(
                normalized, band_rank, low_values.max()
            )
            user_parts.append(middle_u)
            value_parts.append(middle_values)
            item_parts.append(middle_v)
            band_parts.append(
                np.full(band_rank, BAND_MIDDLE, dtype=np.int8)
            )
        user_parts.append(high_u)
        value_parts.append(high_values)
        item_parts.append(high_v)
        band_parts.append(np.full(band_rank, BAND_HIGH, dtype=np.int8))
        user_vectors = np.concatenate(user_parts, axis=1)
        values = np.concatenate(value_parts)
        item_vectors = np.concatenate(item_parts, axis=1)
        bands = np.concatenate(band_parts)
    else:
        raise ValueError(f"Unknown SGFCF spectrum selection: {spectrum}")

    device = base_model.device
    user_vectors = torch.as_tensor(user_vectors, dtype=torch.float32, device=device)
    item_vectors = torch.as_tensor(item_vectors, dtype=torch.float32, device=device)
    values = torch.as_tensor(values, dtype=torch.float32, device=device)
    values /= torch.clamp(values.max(), min=1e-12)
    bands = torch.as_tensor(bands, dtype=torch.int8, device=device)
    return user_vectors, values, item_vectors, normalized, bands


def select_spectral_bands(components, selected_bands):
    """Reuse an existing decomposition while retaining selected band labels."""
    user_vectors, values, item_vectors, normalized, bands = components
    keep = torch.zeros_like(bands, dtype=torch.bool)
    for band in selected_bands:
        keep |= bands == band
    return (
        user_vectors[:, keep],
        values[keep],
        item_vectors[:, keep],
        normalized,
        bands[keep],
    )


def _node_exponents(homophily, beta_1, beta_2):
    minimum, maximum = homophily.min(), homophily.max()
    if torch.isclose(minimum, maximum).item():
        return torch.full_like(homophily, float(beta_1))
    return (
        (beta_2 - beta_1) * homophily / (maximum - minimum)
        + (maximum * beta_1 - beta_2 * minimum) / (maximum - minimum)
    )


def _frequency_weights(values, bands, homophily, params, mid_coefficient,
                       high_coefficient):
    """Combine original low-pass SGFCF with gated mid/high residuals."""
    exponents = _node_exponents(homophily, params["beta_1"], params["beta_2"])
    exponent_matrix = exponents.unsqueeze(1)
    values = torch.clamp(values, min=1e-12, max=1.0)
    low_band = (bands == BAND_LOW).float().unsqueeze(0)
    middle_band = (bands == BAND_MIDDLE).float().unsqueeze(0)
    high_band = (bands == BAND_HIGH).float().unsqueeze(0)
    low = values.unsqueeze(0).pow(exponent_matrix) * low_band

    minimum, maximum = homophily.min(), homophily.max()
    if torch.isclose(minimum, maximum).item():
        normalized_homophily = torch.full_like(homophily, 0.5)
    else:
        normalized_homophily = (homophily - minimum) / (maximum - minimum)
    mid_gate = ((normalized_homophily >= 1.0 / 3.0)
                & (normalized_homophily < 2.0 / 3.0)).float().unsqueeze(1)
    high_gate = (normalized_homophily < 1.0 / 3.0).float().unsqueeze(1)

    mid_response = torch.clamp(4.0 * values * (1.0 - values), min=1e-12)
    high_response = torch.clamp(1.0 - values, min=1e-12)
    mid = mid_response.unsqueeze(0).pow(exponent_matrix) * middle_band
    high = high_response.unsqueeze(0).pow(exponent_matrix) * high_band
    return low + mid_coefficient * mid_gate * mid + high_coefficient * high_gate * high


def _two_component_weights(values, bands, homophily, params, low_coefficient,
                           high_coefficient):
    """Gate low-pass to high-homophily nodes and high-pass to low ones."""
    exponents = _node_exponents(homophily, params["beta_1"], params["beta_2"])
    exponent_matrix = exponents.unsqueeze(1)
    values = torch.clamp(values, min=1e-12, max=1.0)

    minimum, maximum = homophily.min(), homophily.max()
    if torch.isclose(minimum, maximum).item():
        normalized_homophily = torch.full_like(homophily, 0.5)
    else:
        normalized_homophily = (homophily - minimum) / (maximum - minimum)
    low_frequency_gate = (normalized_homophily >= 2.0 / 3.0).float().unsqueeze(1)
    high_frequency_gate = (normalized_homophily < 1.0 / 3.0).float().unsqueeze(1)

    low_band = (bands == BAND_LOW).float().unsqueeze(0)
    high_band = (bands == BAND_HIGH).float().unsqueeze(0)
    low = values.unsqueeze(0).pow(exponent_matrix) * low_band
    high_response = torch.clamp(1.0 - values, min=1e-12)
    high = high_response.unsqueeze(0).pow(exponent_matrix) * high_band
    return (
        low_coefficient * low_frequency_gate * low
        + high_coefficient * high_frequency_gate * high
    )


class SGFCFModel:
    """Original SGFCF with optional mid/high-frequency terms."""

    def __init__(self, base_model, params, mid_coefficient=0.0,
                 high_coefficient=0.0, spectrum="largest"):
        self.base_model = base_model
        self.params = dict(params)
        self.params["k"] = int(base_model.k)
        self.mid_coefficient = float(mid_coefficient)
        self.high_coefficient = float(high_coefficient)
        self.spectrum = spectrum

    @property
    def is_original(self):
        return (
            self.mid_coefficient == 0.0
            and self.high_coefficient == 0.0
        )

    def factors(self, components, homophily):
        user_vectors, values, item_vectors, _, bands = components
        user_weight = _frequency_weights(
            values, bands, homophily[0], self.params,
            self.mid_coefficient, self.high_coefficient,
        )
        item_weight = _frequency_weights(
            values, bands, homophily[1], self.params,
            self.mid_coefficient, self.high_coefficient,
        )
        return user_vectors * user_weight, item_vectors * item_weight


class TwoComponentSGFCFModel:
    """Low-pass for high homophily and high-pass for low homophily."""

    def __init__(self, base_model, params, low_coefficient=1.0,
                 high_coefficient=1.0):
        self.base_model = base_model
        self.params = dict(params)
        self.params["k"] = int(base_model.k)
        self.low_coefficient = float(low_coefficient)
        self.high_coefficient = float(high_coefficient)
        self.spectrum = "two_bands"

    def factors(self, components, homophily):
        user_vectors, values, item_vectors, _, bands = components
        user_weight = _two_component_weights(
            values, bands, homophily[0], self.params,
            self.low_coefficient, self.high_coefficient,
        )
        item_weight = _two_component_weights(
            values, bands, homophily[1], self.params,
            self.low_coefficient, self.high_coefficient,
        )
        return user_vectors * user_weight, item_vectors * item_weight


def _component_factors(components, homophily, params, model_type):
    """Build each gated frequency factor once for a coefficient-grid search."""
    user_vectors, values, item_vectors, _, bands = components
    values = torch.clamp(values, min=1e-12, max=1.0)
    responses = {
        "low": values,
        "middle": torch.clamp(4.0 * values * (1.0 - values), min=1e-12),
        "high": torch.clamp(1.0 - values, min=1e-12),
    }
    band_ids = {
        "low": BAND_LOW,
        "middle": BAND_MIDDLE,
        "high": BAND_HIGH,
    }
    factors = {}

    for name in ("low", "middle", "high"):
        band_mask = bands == band_ids[name]
        if not torch.any(band_mask).item():
            continue
        band_response = responses[name][band_mask]
        side_factors = []
        for vectors, side_homophily in zip(
            (user_vectors, item_vectors), homophily
        ):
            exponents = _node_exponents(
                side_homophily, params["beta_1"], params["beta_2"]
            )
            weights = band_response.unsqueeze(0).pow(exponents.unsqueeze(1))
            minimum, maximum = side_homophily.min(), side_homophily.max()
            if torch.isclose(minimum, maximum).item():
                normalized_homophily = torch.full_like(side_homophily, 0.5)
            else:
                normalized_homophily = (
                    (side_homophily - minimum) / (maximum - minimum)
                )

            if model_type == "three_component" and name == "middle":
                gate = (
                    (normalized_homophily >= 1.0 / 3.0)
                    & (normalized_homophily < 2.0 / 3.0)
                )
                weights = weights * gate.float().unsqueeze(1)
            elif name == "high":
                gate = normalized_homophily < 1.0 / 3.0
                weights = weights * gate.float().unsqueeze(1)
            elif model_type == "two_component" and name == "low":
                gate = normalized_homophily >= 2.0 / 3.0
                weights = weights * gate.float().unsqueeze(1)

            side_factors.append(vectors[:, band_mask] * weights)
        factors[name] = tuple(side_factors)
    return factors


def _empty_results():
    return {
        "precision": np.zeros(len(world.topks)),
        "recall": np.zeros(len(world.topks)),
        "ndcg": np.zeros(len(world.topks)),
    }


def evaluate_frequency_grids(base_model, dataset, three_components,
                             two_components, homophily, params,
                             coefficient_values=COEFFICIENT_VALUES,
                             batch_size=500):
    """Evaluate original and frequency grids with one convolution per batch."""
    three_factors = _component_factors(
        three_components, homophily, params, "three_component"
    )
    two_factors = _component_factors(
        two_components, homophily, params, "two_component"
    )
    original_results = _empty_results()
    three_results = OrderedDict(
        ((float(mid), float(high)), _empty_results())
        for mid in coefficient_values
        for high in coefficient_values
    )
    two_results = OrderedDict(
        ((float(low), float(high)), _empty_results())
        for low in coefficient_values
        for high in coefficient_values
    )
    candidates = [(original_results, three_factors, {"low": 1.0})]
    candidates.extend(
        (result, three_factors, {"low": 1.0, "middle": mid, "high": high})
        for (mid, high), result in three_results.items()
    )
    candidates.extend(
        (result, two_factors, {"low": low, "high": high})
        for (low, high), result in two_results.items()
    )

    normalized = three_components[3]
    normalized_csc = normalized.tocsc()
    test_dict = dataset.testDict
    users = list(test_dict.keys())
    if not users:
        return original_results, three_results, two_results
    max_k = min(max(world.topks), base_model.m_items)

    with torch.no_grad():
        for start in range(0, len(users), batch_size):
            batch_users = users[start:start + batch_size]
            batch_index = torch.as_tensor(
                batch_users, dtype=torch.long, device=base_model.device
            )
            convolution = normalized[batch_users].dot(
                normalized_csc.T
            ).dot(normalized).toarray()
            convolution_sum = convolution.sum(axis=1, keepdims=True)
            convolution_sum[convolution_sum == 0] = 1.0
            convolution = torch.as_tensor(
                convolution / convolution_sum,
                dtype=torch.float32,
                device=base_model.device,
            )

            positive_items = dataset.getUserPosItems(batch_users)
            exclude_rows, exclude_items = [], []
            for row, items in enumerate(positive_items):
                exclude_rows.extend([row] * len(items))
                exclude_items.extend(items)
            ground_truth = [test_dict[user] for user in batch_users]

            for results, factor_parts, coefficients in candidates:
                user_parts = []
                item_parts = []
                for name, coefficient in coefficients.items():
                    if coefficient == 0.0:
                        continue
                    part_user, part_item = factor_parts[name]
                    user_parts.append(coefficient * part_user[batch_index])
                    item_parts.append(coefficient * part_item)
                if not user_parts:
                    rating = torch.zeros(
                        (len(batch_users), base_model.m_items),
                        dtype=torch.float32,
                        device=base_model.device,
                    )
                else:
                    user_factor = torch.cat(user_parts, dim=1)
                    item_factor = torch.cat(item_parts, dim=1)
                    rating = user_factor @ item_factor.t()
                    rating_sum = rating.sum(dim=1, keepdim=True)
                    rating_sum[rating_sum == 0] = 1.0
                    rating = rating / rating_sum
                rating = torch.sigmoid(
                    rating + params["gamma"] * convolution
                )
                if exclude_rows:
                    rating[exclude_rows, exclude_items] = -(1 << 10)
                top_items = torch.topk(rating, k=max_k).indices.cpu()
                batch_result = base_model.test_one_batch(
                    (top_items, ground_truth)
                )
                for metric in results:
                    results[metric] += batch_result[metric]

    for results in [original_results, *three_results.values(), *two_results.values()]:
        for metric in results:
            results[metric] /= float(len(users))
    return original_results, three_results, two_results


def evaluate_parameter_grid(base_model, dataset, components, homophily,
                            parameter_grid, batch_size=500):
    """Evaluate parameter variants while sharing convolution and batch data."""
    models = [SGFCFModel(base_model, params) for params in parameter_grid]
    model_factors = [model.factors(components, homophily) for model in models]
    all_results = [_empty_results() for _ in models]
    normalized = components[3]
    normalized_csc = normalized.tocsc()
    test_dict = dataset.testDict
    users = list(test_dict.keys())
    if not users:
        return all_results
    max_k = min(max(world.topks), base_model.m_items)

    with torch.no_grad():
        for start in range(0, len(users), batch_size):
            batch_users = users[start:start + batch_size]
            batch_index = torch.as_tensor(
                batch_users, dtype=torch.long, device=base_model.device
            )
            convolution = normalized[batch_users].dot(
                normalized_csc.T
            ).dot(normalized).toarray()
            convolution_sum = convolution.sum(axis=1, keepdims=True)
            convolution_sum[convolution_sum == 0] = 1.0
            convolution = torch.as_tensor(
                convolution / convolution_sum,
                dtype=torch.float32,
                device=base_model.device,
            )
            positive_items = dataset.getUserPosItems(batch_users)
            exclude_rows, exclude_items = [], []
            for row, items in enumerate(positive_items):
                exclude_rows.extend([row] * len(items))
                exclude_items.extend(items)
            ground_truth = [test_dict[user] for user in batch_users]

            for model, factors, results in zip(
                models, model_factors, all_results
            ):
                user_factors, item_factors = factors
                rating = user_factors[batch_index] @ item_factors.t()
                rating_sum = rating.sum(dim=1, keepdim=True)
                rating_sum[rating_sum == 0] = 1.0
                rating = torch.sigmoid(
                    rating / rating_sum
                    + model.params["gamma"] * convolution
                )
                if exclude_rows:
                    rating[exclude_rows, exclude_items] = -(1 << 10)
                top_items = torch.topk(rating, k=max_k).indices.cpu()
                batch_result = base_model.test_one_batch(
                    (top_items, ground_truth)
                )
                for metric in results:
                    results[metric] += batch_result[metric]

    for results in all_results:
        for metric in results:
            results[metric] /= float(len(users))
    return all_results


def evaluate(model, dataset, components=None, homophily=None, batch_size=500):
    """Evaluate SGFCF in batches without materializing all user-item scores."""
    base_model = model.base_model
    if components is None:
        components = compute_components(
            base_model,
            model.params,
            spectrum=getattr(model, "spectrum", "largest"),
        )
    if homophily is None:
        homophily = compute_homophily(base_model)
    user_factors, item_factors = model.factors(components, homophily)
    normalized = components[3]
    normalized_csc = normalized.tocsc()
    test_dict = dataset.testDict
    users = list(test_dict.keys())
    results = {
        "precision": np.zeros(len(world.topks)),
        "recall": np.zeros(len(world.topks)),
        "ndcg": np.zeros(len(world.topks)),
    }
    if not users:
        return results

    max_k = min(max(world.topks), base_model.m_items)
    with torch.no_grad():
        for start in range(0, len(users), batch_size):
            batch_users = users[start:start + batch_size]
            batch_index = torch.as_tensor(
                batch_users, dtype=torch.long, device=base_model.device
            )
            rating = user_factors[batch_index] @ item_factors.t()
            rating_sum = rating.sum(dim=1, keepdim=True)
            rating_sum[rating_sum == 0] = 1.0
            rating = rating / rating_sum

            convolution = normalized[batch_users].dot(normalized_csc.T).dot(normalized)
            convolution = convolution.toarray()
            convolution_sum = convolution.sum(axis=1, keepdims=True)
            convolution_sum[convolution_sum == 0] = 1.0
            convolution = torch.as_tensor(
                convolution / convolution_sum,
                dtype=torch.float32,
                device=base_model.device,
            )
            rating = torch.sigmoid(
                rating + model.params["gamma"] * convolution
            )

            positive_items = dataset.getUserPosItems(batch_users)
            exclude_rows, exclude_items = [], []
            for row, items in enumerate(positive_items):
                exclude_rows.extend([row] * len(items))
                exclude_items.extend(items)
            if exclude_rows:
                rating[exclude_rows, exclude_items] = -(1 << 10)

            top_items = torch.topk(rating, k=max_k).indices.cpu()
            ground_truth = [test_dict[user] for user in batch_users]
            batch_result = base_model.test_one_batch((top_items, ground_truth))
            for metric in results:
                results[metric] += batch_result[metric]

    for metric in results:
        results[metric] /= float(len(users))
    return results


def _score(results):
    return float(results["recall"][0])


# ---------------------------------------------------------------------------
# DAN references and the two SGFCF+DAN combinations
# ---------------------------------------------------------------------------
class DANScoringModel:
    """Repository DAN (primal closed-form implementation)."""

    def __init__(self, interactions, model="RLAE_DAN", reg_p=20.0,
                 alpha=0.2, beta=0.3, drop_p=0.6, xi=0.1):
        if model not in {"LAE_DAN", "EASE_DAN", "RLAE_DAN"}:
            raise ValueError(f"Unknown DAN model: {model}")
        self.model = model
        self.interactions = interactions.astype(np.float64).tocsr()
        counts_i = np.asarray(self.interactions.sum(axis=0)).ravel()
        counts_u = np.asarray(self.interactions.sum(axis=1)).ravel()
        safe_u = np.maximum(counts_u, 1.0)
        regularization = float(reg_p) + float(drop_p) / (1.0 - float(drop_p)) * counts_i
        gram = self.interactions.multiply(np.power(safe_u, -float(beta))[:, None]).T.dot(self.interactions).toarray()
        diagonal = np.diag_indices(gram.shape[0])
        gram[diagonal] += regularization
        precision = np.linalg.inv(gram)
        diag_precision = np.diag(precision)
        if model == "EASE_DAN":
            base = np.eye(len(diag_precision)) - precision / diag_precision
        elif model == "LAE_DAN":
            base = np.eye(len(diag_precision)) - precision * regularization
        else:
            condition = (1.0 - regularization * diag_precision) > float(xi)
            lagrangian = ((1.0 - float(xi)) / diag_precision - regularization) * condition
            base = np.eye(len(diag_precision)) - precision * (regularization + lagrangian)
        self.weights = base * np.power(np.maximum(counts_i, 1.0), float(alpha)).reshape(-1, 1)
        self.weights *= np.power(np.maximum(counts_i, 1.0), -float(alpha))
        self.weights[diagonal] = 0.0
        self.cold_items = counts_i == 0
        self.params = {"model": model, "reg_p": float(reg_p), "alpha": float(alpha),
                       "beta": float(beta), "drop_p": float(drop_p), "xi": float(xi)}

    def score_batch(self, interactions):
        scores = np.asarray(interactions.dot(self.weights))
        scores[:, self.cold_items] = -np.inf
        return scores


class SGFCFPlusDANModel(SGFCFModel):
    """Linear score combination of SGFCF and DAN."""

    def __init__(self, base_model, params, dan_model, dan_coefficient,
                 dan_interactions=None):
        super().__init__(base_model, params)
        self.dan_model = dan_model
        self.dan_coefficient = float(dan_coefficient)
        self.dan_interactions = base_model.R if dan_interactions is None else dan_interactions


class UnifiedNormalizationSGFCFDANModel(SGFCFPlusDANModel):
    """SGFCF+DAN where DAN is fitted to SGFCF's normalized interaction matrix."""


def _evaluate_dan(model, base_model, dataset, batch_size=500):
    users = list(dataset.testDict.keys())
    results = _empty_results()
    if not users:
        return results
    max_k = min(max(world.topks), base_model.m_items)
    for start in range(0, len(users), batch_size):
        batch_users = users[start:start + batch_size]
        scores = torch.as_tensor(model.score_batch(base_model.R[batch_users]), device=base_model.device)
        positive = dataset.getUserPosItems(batch_users)
        rows, cols = [], []
        for row, items in enumerate(positive):
            rows.extend([row] * len(items)); cols.extend(items)
        if rows:
            scores[rows, cols] = -(1 << 10)
        top_items = torch.topk(scores, k=max_k).indices.cpu()
        batch_result = base_model.test_one_batch((top_items, [dataset.testDict[u] for u in batch_users]))
        for metric in results:
            results[metric] += batch_result[metric]
    for metric in results:
        results[metric] /= float(len(users))
    return results


def evaluate_sgfcf_dan(model, dataset, components, homophily, batch_size=500):
    user_factors, item_factors = model.factors(components, homophily)
    normalized = components[3]
    normalized_csc = normalized.tocsc()
    users = list(dataset.testDict.keys())
    results = _empty_results()
    if not users:
        return results
    max_k = min(max(world.topks), model.base_model.m_items)
    for start in range(0, len(users), batch_size):
        batch_users = users[start:start + batch_size]
        indices = torch.as_tensor(batch_users, dtype=torch.long, device=model.base_model.device)
        spectral = user_factors[indices] @ item_factors.t()
        total = spectral.sum(dim=1, keepdim=True); total[total == 0] = 1.0
        spectral = spectral / total
        convolution = normalized[batch_users].dot(normalized_csc.T).dot(normalized).toarray()
        total = convolution.sum(axis=1, keepdims=True); total[total == 0] = 1.0
        convolution = torch.as_tensor(convolution / total, dtype=torch.float32, device=model.base_model.device)
        dan_scores = torch.as_tensor(model.dan_model.score_batch(model.dan_interactions[batch_users]), dtype=torch.float32, device=model.base_model.device)
        scores = torch.sigmoid(spectral + model.params["gamma"] * convolution + model.dan_coefficient * dan_scores)
        positive = dataset.getUserPosItems(batch_users)
        rows, cols = [], []
        for row, items in enumerate(positive):
            rows.extend([row] * len(items)); cols.extend(items)
        if rows:
            scores[rows, cols] = -(1 << 10)
        top_items = torch.topk(scores, k=max_k).indices.cpu()
        batch_result = model.base_model.test_one_batch((top_items, [dataset.testDict[u] for u in batch_users]))
        for metric in results:
            results[metric] += batch_result[metric]
    for metric in results:
        results[metric] /= float(len(users))
    return results


def search(base_model, dataset, coefficient_values=COEFFICIENT_VALUES,
           homophily=None):
    """Select all SGFCF variants using recall on the requested split."""
    if homophily is None:
        homophily = compute_homophily(base_model)
    best_original = None
    best_extended = None
    best_original_results = None
    best_extended_results = None
    best_two_component = None
    best_two_component_results = None
    best_original_score = float("-inf")
    best_extended_score = float("-inf")
    best_two_component_score = float("-inf")
    cached_band_key = None
    three_band_components = None
    two_band_components = None

    for index, base_params in enumerate(SGFCF_SEARCH_SPACE, start=1):
        params = dict(base_params)
        params["k"] = int(base_model.k)
        band_key = (params["alpha"], params["eps"], params["k"])
        if band_key != cached_band_key:
            three_band_components = compute_components(
                base_model, params, spectrum="three_bands"
            )
            if 3 * int(base_model.k) <= min(base_model.R.shape):
                two_band_components = select_spectral_bands(
                    three_band_components, (BAND_LOW, BAND_HIGH)
                )
            else:
                two_band_components = compute_components(
                    base_model, params, spectrum="two_bands"
                )
            cached_band_key = band_key

        original_results, extended_grid, two_component_grid = (
            evaluate_frequency_grids(
                base_model,
                dataset,
                three_band_components,
                two_band_components,
                homophily,
                params,
                coefficient_values=coefficient_values,
            )
        )
        original_score = _score(original_results)
        if original_score > best_original_score:
            best_original_score = original_score
            best_original = {"params": dict(params)}
            best_original_results = original_results

        for (mid_coefficient, high_coefficient), extended_results in (
            extended_grid.items()
        ):
            extended_score = _score(extended_results)
            if extended_score > best_extended_score:
                best_extended_score = extended_score
                best_extended = {
                    "params": dict(params),
                    "mid_coefficient": mid_coefficient,
                    "high_coefficient": high_coefficient,
                }
                best_extended_results = extended_results

        for (low_coefficient, high_coefficient), two_component_results in (
            two_component_grid.items()
        ):
            two_component_score = _score(two_component_results)
            if two_component_score > best_two_component_score:
                best_two_component_score = two_component_score
                best_two_component = {
                    "params": dict(params),
                    "low_coefficient": low_coefficient,
                    "high_coefficient": high_coefficient,
                }
                best_two_component_results = two_component_results

        print(
            f"SGFCF search {index}/{len(SGFCF_SEARCH_SPACE)}: "
            f"original={best_original_score:.6f}, "
            f"frequency_terms={best_extended_score:.6f}, "
            f"two_components={best_two_component_score:.6f}",
            flush=True,
        )

    three_band_components = None
    two_band_components = None

    best_generalized = None
    best_generalized_results = None
    best_generalized_score = float("-inf")
    for index, normalization in enumerate(
        ASYMMETRIC_NORMALIZATION_SPACE, start=1
    ):
        component_params = dict(SGFCF_SEARCH_SPACE[0])
        component_params["k"] = int(base_model.k)
        generalized_components = compute_components(
            base_model, component_params, normalization=normalization
        )
        generalized_params = []
        for base_params in SGFCF_SEARCH_SPACE:
            params = dict(base_params)
            params["k"] = int(base_model.k)
            generalized_params.append(params)
        generalized_grid_results = evaluate_parameter_grid(
            base_model,
            dataset,
            generalized_components,
            homophily,
            generalized_params,
        )
        for params, generalized_results in zip(
            generalized_params, generalized_grid_results
        ):
            generalized_score = _score(generalized_results)
            if generalized_score > best_generalized_score:
                best_generalized_score = generalized_score
                best_generalized = {
                    "params": dict(params),
                    "normalization": dict(normalization),
                }
                best_generalized_results = generalized_results
        print(
            f"Generalized-normalization search "
            f"{index}/{len(ASYMMETRIC_NORMALIZATION_SPACE)}: "
            f"best={best_generalized_score:.6f}",
            flush=True,
        )

    configs = OrderedDict(
        original=best_original,
        three_component=best_extended,
        two_component=best_two_component,
        generalized_normalization=best_generalized,
    )
    validation_results = OrderedDict(
        original=best_original_results,
        three_component=best_extended_results,
        two_component=best_two_component_results,
        generalized_normalization=best_generalized_results,
    )
    return configs, validation_results


def search(base_model, dataset, homophily=None, k_values=None,
           dan_params=None):
    """Search exactly four models: SGFCF, DAN, linear, and unified-normalized."""
    if homophily is None:
        homophily = compute_homophily(base_model)
    dan_params = {**DAN_DEFAULT_PARAMS, **(dan_params or {}), "model": "RLAE_DAN"}
    if k_values is None:
        k_values = [base_model.k]
    original_k = base_model.k
    candidates = []
    for k in k_values:
        for candidate in SGFCF_SEARCH_SPACE:
            params = dict(candidate); params["k"] = int(k)
            candidates.append(params)

    configs, results = OrderedDict(), OrderedDict()
    best = {name: (-np.inf, None, None) for name in ("SGFCF", "DAN", "SGFCF+DAN", "UnifiedNorm-SGFCF+DAN")}
    dan_model = None
    try:
        for index, params in enumerate(candidates, 1):
            base_model.k = int(params["k"])
            components = compute_components(base_model, params, spectrum="largest")
            sgfcf = SGFCFModel(base_model, params)
            sgfcf_result = evaluate(sgfcf, dataset, components, homophily)
            score = _score(sgfcf_result)
            if score > best["SGFCF"][0]:
                best["SGFCF"] = (score, {"params": dict(params)}, sgfcf_result)

            if dan_model is None:
                dan_model = DANScoringModel(base_model.R, **dan_params)
                dan_result = _evaluate_dan(dan_model, base_model, dataset)
                best["DAN"] = (_score(dan_result), {
                    "params": dict(params), "dan_params": dict(dan_params)
                }, dan_result)

            for name, unified in (("SGFCF+DAN", False), ("UnifiedNorm-SGFCF+DAN", True)):
                dan_input = components[3] if unified else base_model.R
                candidate_dan = DANScoringModel(dan_input, **dan_params) if unified else dan_model
                for dan_coefficient in COEFFICIENT_VALUES:
                    combined = SGFCFPlusDANModel(
                        base_model, params, candidate_dan, dan_coefficient,
                        dan_interactions=dan_input,
                    )
                    combined.params["gamma"] = params["gamma"]
                    combined_result = evaluate_sgfcf_dan(combined, dataset, components, homophily)
                    score = _score(combined_result)
                    if score > best[name][0]:
                        best[name] = (score, {
                            "params": dict(params),
                            "dan_coefficient": dan_coefficient,
                            "dan_params": dict(dan_params),
                            "unified_normalization": unified,
                        }, combined_result)
            print(f"Four-model search {index}/{len(candidates)}: "
                  f"SGFCF={best['SGFCF'][0]:.6f}, "
                  f"DAN={best['DAN'][0]:.6f}, "
                  f"linear={best['SGFCF+DAN'][0]:.6f}, "
                  f"unified={best['UnifiedNorm-SGFCF+DAN'][0]:.6f}", flush=True)
        for name in ("SGFCF", "DAN", "SGFCF+DAN", "UnifiedNorm-SGFCF+DAN"):
            configs[name], results[name] = best[name][1], best[name][2]
    finally:
        base_model.k = original_k
    return configs, results


def search(base_model, dataset, homophily=None, k_values=None,
           dan_params=None):
    """Search SGFCF, DAN, linear SGFCF+DAN, and unified SGFCF+DAN."""
    if homophily is None:
        homophily = compute_homophily(base_model)
    k_values = tuple(k_values or (64, 128, 256))
    candidates = [dict(p, k=int(k)) for k in k_values for p in SGFCF_SEARCH_SPACE]
    names = ("SGFCF", "DAN", "SGFCF+DAN", "UnifiedNorm-SGFCF+DAN")
    best = {name: (-np.inf, None, None) for name in names}
    component_cache = {}
    dan_cache = {}
    original_k = base_model.k

    def dan_for(params):
        key = tuple(sorted(params.items()))
        if key not in dan_cache:
            dan_cache[key] = DANScoringModel(base_model.R, **params)
        return dan_cache[key]

    try:
        # Standalone DAN search over its requested 54 configurations.
        for dan_index, dan_config in enumerate(DAN_SEARCH_SPACE, 1):
            dan_model = dan_for(dan_config)
            dan_result = _evaluate_dan(dan_model, base_model, dataset)
            score = _score(dan_result)
            if score > best["DAN"][0]:
                best["DAN"] = (score, {"params": dict(candidates[0]),
                                       "dan_params": dict(dan_config)}, dan_result)
            if dan_index % 9 == 0:
                print(f"DAN search {dan_index}/{len(DAN_SEARCH_SPACE)}: best={best['DAN'][0]:.6f}", flush=True)

        for index, params in enumerate(candidates, 1):
            base_model.k = int(params["k"])
            key = (params["alpha"], params["eps"], params["k"])
            if key not in component_cache:
                component_cache[key] = compute_components(base_model, params, spectrum="largest")
            components = component_cache[key]
            sgfcf_result = evaluate(SGFCFModel(base_model, params), dataset, components, homophily)
            if _score(sgfcf_result) > best["SGFCF"][0]:
                best["SGFCF"] = (_score(sgfcf_result), {"params": dict(params)}, sgfcf_result)

            for dan_config in DAN_SEARCH_SPACE:
                dan_model = dan_for(dan_config)
                for coefficient in DAN_COEFFICIENT_VALUES:
                    linear = SGFCFPlusDANModel(base_model, params, dan_model, coefficient)
                    result = evaluate_sgfcf_dan(linear, dataset, components, homophily)
                    if _score(result) > best["SGFCF+DAN"][0]:
                        best["SGFCF+DAN"] = (_score(result), {
                            "params": dict(params), "dan_params": dict(dan_config),
                            "dan_coefficient": coefficient}, result)

                normalized = components[3]
                unified_dan = DANScoringModel(normalized, **dan_config)
                for coefficient in DAN_COEFFICIENT_VALUES:
                    unified = UnifiedNormalizationSGFCFDANModel(
                        base_model, params, unified_dan, coefficient,
                        dan_interactions=normalized
                    )
                    result = evaluate_sgfcf_dan(unified, dataset, components, homophily)
                    if _score(result) > best["UnifiedNorm-SGFCF+DAN"][0]:
                        best["UnifiedNorm-SGFCF+DAN"] = (_score(result), {
                            "params": dict(params), "dan_params": dict(dan_config),
                            "dan_coefficient": coefficient}, result)
            print(f"Four-model search {index}/{len(candidates)}: "
                  f"SGFCF={best['SGFCF'][0]:.6f}, DAN={best['DAN'][0]:.6f}, "
                  f"linear={best['SGFCF+DAN'][0]:.6f}, "
                  f"unified={best['UnifiedNorm-SGFCF+DAN'][0]:.6f}", flush=True)
    finally:
        base_model.k = original_k
    configs = OrderedDict((name, best[name][1]) for name in names)
    validation = OrderedDict((name, best[name][2]) for name in names)
    return configs, validation


def search(base_model, dataset, homophily=None, k_values=None,
           dan_params=None):
    """Full SGFCF/DAN searches followed by top-10 hybrid searches."""
    if homophily is None:
        homophily = compute_homophily(base_model)
    k_values = tuple(k_values or (64, 128, 256))
    sgfcf_candidates = [dict(p, k=int(k)) for k in k_values for p in SGFCF_SEARCH_SPACE]
    component_cache = {}
    original_k = base_model.k

    sgfcf_ranked, dan_ranked = [], []
    try:
        # Full SGFCF search.
        for index, params in enumerate(sgfcf_candidates, 1):
            base_model.k = int(params["k"])
            key = (params["alpha"], params["eps"], params["k"])
            if key not in component_cache:
                component_cache[key] = compute_components(base_model, params, spectrum="largest")
            result = evaluate(SGFCFModel(base_model, params), dataset,
                              component_cache[key], homophily)
            sgfcf_ranked.append((_score(result), dict(params), result))
            if index % 20 == 0 or index == len(sgfcf_candidates):
                print(f"SGFCF search {index}/{len(sgfcf_candidates)}: "
                      f"best={max(x[0] for x in sgfcf_ranked):.6f}", flush=True)

        # Full DAN search.
        for index, dan_config in enumerate(DAN_SEARCH_SPACE, 1):
            dan_model = DANScoringModel(base_model.R, **dan_config)
            result = _evaluate_dan(dan_model, base_model, dataset)
            dan_ranked.append((_score(result), dict(dan_config), result))
            del dan_model
            gc.collect()
            if index % 9 == 0 or index == len(DAN_SEARCH_SPACE):
                print(f"DAN search {index}/{len(DAN_SEARCH_SPACE)}: "
                      f"best={max(x[0] for x in dan_ranked):.6f}", flush=True)

        top_sgfcf = sorted(sgfcf_ranked, key=lambda x: x[0], reverse=True)[:10]
        top_dan = sorted(dan_ranked, key=lambda x: x[0], reverse=True)[:10]
        hybrid_best = {
            "SGFCF+DAN": (-np.inf, None, None),
            "UnifiedNorm-SGFCF+DAN": (-np.inf, None, None),
        }
        total_hybrid = len(top_sgfcf) * len(top_dan) * len(DAN_COEFFICIENT_VALUES)
        hybrid_index = 0
        # Process one DAN configuration at a time so that at most one dense
        # DAN matrix is resident for each normalization type.
        for _, dan_config, _ in top_dan:
            raw_dan = DANScoringModel(base_model.R, **dan_config)
            for _, params, _ in top_sgfcf:
                base_model.k = int(params["k"])
                key = (params["alpha"], params["eps"], params["k"])
                components = component_cache[key]
                for coefficient in DAN_COEFFICIENT_VALUES:
                    hybrid = SGFCFPlusDANModel(
                        base_model, params, raw_dan, coefficient,
                        dan_interactions=base_model.R,
                    )
                    result = evaluate_sgfcf_dan(hybrid, dataset, components, homophily)
                    score = _score(result)
                    if score > hybrid_best["SGFCF+DAN"][0]:
                        hybrid_best["SGFCF+DAN"] = (score, {
                            "params": dict(params), "dan_params": dict(dan_config),
                            "dan_coefficient": coefficient}, result)
                    hybrid_index += 1
                    if hybrid_index % 100 == 0:
                        print(f"Hybrid search {hybrid_index}/{total_hybrid * 2}", flush=True)
            del raw_dan
            gc.collect()
            # Run the unified-normalization half after releasing the raw DAN
            # matrix; Yelp2018 cannot safely hold both dense weight matrices.
            for _, params, _ in top_sgfcf:
                base_model.k = int(params["k"])
                key = (params["alpha"], params["eps"], params["k"])
                components = component_cache[key]
                normalized = components[3]
                unified_dan = DANScoringModel(normalized, **dan_config)
                for coefficient in DAN_COEFFICIENT_VALUES:
                    hybrid = UnifiedNormalizationSGFCFDANModel(
                        base_model, params, unified_dan, coefficient,
                        dan_interactions=normalized,
                    )
                    result = evaluate_sgfcf_dan(hybrid, dataset, components, homophily)
                    score = _score(result)
                    if score > hybrid_best["UnifiedNorm-SGFCF+DAN"][0]:
                        hybrid_best["UnifiedNorm-SGFCF+DAN"] = (score, {
                            "params": dict(params), "dan_params": dict(dan_config),
                            "dan_coefficient": coefficient}, result)
                    hybrid_index += 1
                    if hybrid_index % 100 == 0 or hybrid_index == total_hybrid * 2:
                        print(f"Hybrid search {hybrid_index}/{total_hybrid * 2}", flush=True)
                del unified_dan
                gc.collect()

        best_sgfcf = max(sgfcf_ranked, key=lambda x: x[0])
        best_dan = max(dan_ranked, key=lambda x: x[0])
        configs = OrderedDict([
            ("SGFCF", {"params": best_sgfcf[1]}),
            ("DAN", {"params": dict(sgfcf_candidates[0]), "dan_params": best_dan[1]}),
            ("SGFCF+DAN", hybrid_best["SGFCF+DAN"][1]),
            ("UnifiedNorm-SGFCF+DAN", hybrid_best["UnifiedNorm-SGFCF+DAN"][1]),
        ])
        validation = OrderedDict([
            ("SGFCF", best_sgfcf[2]),
            ("DAN", best_dan[2]),
            ("SGFCF+DAN", hybrid_best["SGFCF+DAN"][2]),
            ("UnifiedNorm-SGFCF+DAN", hybrid_best["UnifiedNorm-SGFCF+DAN"][2]),
        ])
        return configs, validation
    finally:
        base_model.k = original_k
