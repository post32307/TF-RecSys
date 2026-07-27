"""SGFCF and spectral-filter variants sharing the complete SGFCF pipeline."""

from collections import OrderedDict

import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse.linalg import svds
from torch import nn


# Best SGFCF configurations recorded in the seed-2020 final sweep results.
SGFCF_SEED_2020_PARAMS = {
    "amazon-electronics": {"k": 32, "beta_1": 1.0, "beta_2": 1.3, "alpha": 19.0, "eps": 0.5, "gamma": 0.5},
    "lastfm": {"k": 128, "beta_1": 0.7, "beta_2": 1.1, "alpha": 0.0, "eps": 0.28, "gamma": 1.5},
    "citeulike": {"k": 1024, "beta_1": 0.7, "beta_2": 1.1, "alpha": 0.0, "eps": 0.28, "gamma": 1.5},
    "epinions": {"k": 128, "beta_1": 1.0, "beta_2": 1.3, "alpha": 19.0, "eps": 0.5, "gamma": 0.5},
    "yelp": {"k": 128, "beta_1": 1.0, "beta_2": 1.3, "alpha": 19.0, "eps": 0.5, "gamma": 0.5},
    "douban-book": {"k": 1024, "beta_1": 1.0, "beta_2": 1.3, "alpha": 19.0, "eps": 0.5, "gamma": 0.5},
    "amazon-CD": {"k": 512, "beta_1": 1.0, "beta_2": 1.3, "alpha": 19.0, "eps": 0.5, "gamma": 0.5},
    "news": {"k": 64, "beta_1": 0.9, "beta_2": 1.0, "alpha": 0.0, "eps": 0.37, "gamma": 0.3},
    "movielens": {"k": 128, "beta_1": 0.9, "beta_2": 1.0, "alpha": 0.0, "eps": 0.37, "gamma": 0.3},
    "gowalla": {"k": 1024, "beta_1": 0.7, "beta_2": 1.1, "alpha": 0.0, "eps": 0.28, "gamma": 1.5},
    "yelp2018": {"k": 512, "beta_1": 0.9, "beta_2": 1.0, "alpha": 0.0, "eps": 0.37, "gamma": 0.3},
}


def get_sgfcf_params(dataset_name, fallback_k):
    """Return an independent SGFCF parameter dictionary for a dataset."""
    fallback = {
        "k": int(fallback_k),
        "beta_1": 1.0,
        "beta_2": 1.0,
        "alpha": 0.0,
        "eps": 0.5,
        "gamma": 1.0,
    }
    return dict(SGFCF_SEED_2020_PARAMS.get(dataset_name, fallback))


class SGFCFFilter(nn.Module):
    """Base class for spectral responses inserted into the SGFCF pipeline."""

    def __init__(self, base_model, params):
        super().__init__()
        if base_model is None:
            raise ValueError("base_model must be an initialized SGFCF model.")
        self.base_model = base_model
        self.params = dict(params)

    def _as_tensor(self, singular_values):
        return torch.as_tensor(
            singular_values,
            dtype=torch.float32,
            device=self.base_model.device,
        )


class OriginalSGFCF(SGFCFFilter):
    """Original SGFCF response before node-specific homophily exponents."""

    def forward(self, singular_values):
        return self._as_tensor(singular_values)


class RidgeFilter(SGFCFFilter):
    """Gram-ridge response: h(lambda) = lambda / (rho + lambda)."""

    def __init__(self, base_model, params, rho=1.0):
        super().__init__(base_model, params)
        if rho <= 0:
            raise ValueError("rho must be greater than 0.")
        self.rho = float(rho)

    def forward(self, singular_values):
        singular_values = self._as_tensor(singular_values)
        return singular_values / (self.rho + singular_values)


class HeatKernelFilter(SGFCFFilter):
    """Heat-kernel response: h(lambda) = exp(-t * lambda)."""

    def __init__(self, base_model, params, t=1.0):
        super().__init__(base_model, params)
        if t < 0:
            raise ValueError("t must be non-negative.")
        self.t = float(t)

    def forward(self, singular_values):
        singular_values = self._as_tensor(singular_values)
        return torch.exp(-self.t * singular_values)


class PageRankFilter(SGFCFFilter):
    """Personalized PageRank response: h(lambda) = (1-eta)/(1-eta*lambda)."""

    def __init__(self, base_model, params, eta=0.85):
        super().__init__(base_model, params)
        if not 0 < eta < 1:
            raise ValueError("eta must satisfy 0 < eta < 1.")
        self.eta = float(eta)

    def forward(self, singular_values):
        singular_values = self._as_tensor(singular_values)
        return (1.0 - self.eta) / (1.0 - self.eta * singular_values)


def build_filter_models(base_model, params, rho=1.0, t=1.0, eta=0.85):
    """Build SGFCF and variants that differ only in spectral response."""
    return OrderedDict(
        sgfcf=OriginalSGFCF(base_model, params),
        ridge=RidgeFilter(base_model, params, rho=rho),
        heat_kernel=HeatKernelFilter(base_model, params, t=t),
        pagerank=PageRankFilter(base_model, params, eta=eta),
    )


def _threshold_offdiag_graph(matrix):
    graph = matrix.tocsr(copy=True)
    graph.setdiag(0)
    graph.eliminate_zeros()
    graph.data[graph.data <= 1] = 0
    graph.eliminate_zeros()
    graph.data = np.ones(graph.nnz, dtype=np.int8)
    return graph


def _count_induced_edges(graph, nodes, marker, stamp):
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


def _compute_homophily_ratios(base_model):
    interactions = base_model.R.astype(np.float32).tocsr()
    n_users, n_items = interactions.shape
    interactions_csc = interactions.tocsc()
    item_graph = _threshold_offdiag_graph(interactions.T.dot(interactions))
    user_graph = _threshold_offdiag_graph(interactions.dot(interactions.T))

    user_ratios = np.zeros(n_users, dtype=np.float32)
    item_marker = np.zeros(n_items, dtype=np.int32)
    item_stamp = 0
    for user in range(n_users):
        nodes = interactions[user].indices
        if len(nodes) > 1:
            edges, item_stamp = _count_induced_edges(item_graph, nodes, item_marker, item_stamp)
            user_ratios[user] = edges / (len(nodes) * (len(nodes) - 1))

    item_ratios = np.zeros(n_items, dtype=np.float32)
    user_marker = np.zeros(n_users, dtype=np.int32)
    user_stamp = 0
    for item in range(n_items):
        nodes = interactions_csc[:, item].indices
        if len(nodes) > 1:
            edges, user_stamp = _count_induced_edges(user_graph, nodes, user_marker, user_stamp)
            item_ratios[item] = edges / (len(nodes) * (len(nodes) - 1))

    device = base_model.device
    return (
        torch.as_tensor(user_ratios, dtype=torch.float32, device=device),
        torch.as_tensor(item_ratios, dtype=torch.float32, device=device),
    )


def _compute_sgfcf_components(base_model, params):
    interactions = base_model.R.astype(np.float32).tocsr()
    max_rank = min(base_model.n_users, base_model.m_items) - 1
    if max_rank < 1:
        raise ValueError("SGFCF requires at least two users and two items.")
    rank = max(1, min(int(params["k"]), max_rank))

    user_degree = np.asarray(interactions.sum(axis=1)).ravel()
    item_degree = np.asarray(interactions.sum(axis=0)).ravel()
    with np.errstate(divide="ignore"):
        d_user = 1.0 / np.power(user_degree + params["alpha"], params["eps"])
        d_item = 1.0 / np.power(item_degree + params["alpha"], params["eps"])
    d_user[~np.isfinite(d_user)] = 0.0
    d_item[~np.isfinite(d_item)] = 0.0

    normalized = sp.diags(d_user).dot(interactions).dot(sp.diags(d_item))
    normalized = normalized.astype(np.float32).tocsr()
    svd_rank = min(rank + 200, max_rank)
    user_vectors, singular_values, item_vectors_t = svds(normalized, k=svd_rank)
    order = np.argsort(singular_values)[::-1]

    device = base_model.device
    user_vectors = torch.as_tensor(user_vectors[:, order[:rank]], dtype=torch.float32, device=device)
    item_vectors = torch.as_tensor(item_vectors_t[order[:rank], :].T, dtype=torch.float32, device=device)
    singular_values = torch.as_tensor(singular_values[order[:rank]], dtype=torch.float32, device=device)
    singular_values /= torch.clamp(singular_values.max(), min=1e-12)
    return user_vectors, singular_values, item_vectors, normalized


def _individual_weight(response, homophily, beta_1, beta_2):
    response = torch.clamp(response, min=1e-12)
    minimum, maximum = homophily.min(), homophily.max()
    if torch.isclose(minimum, maximum).item():
        exponents = torch.full_like(homophily, float(beta_1))
    else:
        exponents = (beta_2 - beta_1) * homophily / (maximum - minimum)
        exponents += (maximum * beta_1 - beta_2 * minimum) / (maximum - minimum)
    return response.unsqueeze(0).pow(exponents.unsqueeze(1))


def _sgfcf_rating_matrix(filter_model, components, homophily, convolution):
    user_vectors, singular_values, item_vectors, _ = components
    user_homophily, item_homophily = homophily
    params = filter_model.params
    response = filter_model(singular_values)
    user_factors = user_vectors * _individual_weight(
        response, user_homophily, params["beta_1"], params["beta_2"]
    )
    item_factors = item_vectors * _individual_weight(
        response, item_homophily, params["beta_1"], params["beta_2"]
    )

    # Use the base-model rating function while preserving SGFCF's pre-sigmoid score.
    embeddings = torch.cat((user_factors, item_factors), dim=0)
    users = torch.arange(filter_model.base_model.n_users, device=filter_model.base_model.device)
    rating = filter_model.base_model.getUsersRating(embeddings, users, apply_sigmoid=False)
    rating_sum = rating.sum(dim=1, keepdim=True)
    rating_sum[rating_sum == 0] = 1.0
    rating = rating / rating_sum

    return torch.sigmoid(rating + params["gamma"] * convolution)


def evaluate_filter_models(filter_models, dataset, multicore=0):
    """Evaluate the original SGFCF and every response variant."""
    if not filter_models:
        return OrderedDict()
    base_model = next(iter(filter_models.values())).base_model
    params = next(iter(filter_models.values())).params
    components = _compute_sgfcf_components(base_model, params)
    homophily = _compute_homophily_ratios(base_model)
    normalized = components[3]
    convolution = normalized.dot(normalized.T).dot(normalized).toarray()
    convolution_sum = convolution.sum(axis=1, keepdims=True)
    convolution_sum[convolution_sum == 0] = 1.0
    convolution = torch.as_tensor(
        convolution / convolution_sum,
        dtype=torch.float32,
        device=base_model.device,
    )
    results = OrderedDict()

    with torch.no_grad():
        for name, filter_model in filter_models.items():
            if filter_model.base_model is not base_model or filter_model.params != params:
                raise ValueError("All filters must share the same SGFCF base and parameters.")
            rating_matrix = _sgfcf_rating_matrix(
                filter_model, components, homophily, convolution
            )
            results[name] = base_model.Test_RatingMatrix(
                rating_matrix, dataset, multicore=multicore
            )
    return results
