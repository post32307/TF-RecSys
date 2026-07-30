import numpy as np


def recall_at_k(ground_truth, relevance, k):
    """Match code/metrics.py::RecallPrecision_ATk."""
    hits = relevance[:, :k].sum(axis=1)
    denominators = np.asarray([len(items) for items in ground_truth])
    return float(np.sum(hits / denominators))


def ndcg_at_k(ground_truth, relevance, k):
    """Match code/metrics.py::NDCGatK_r."""
    predicted = relevance[:, :k]
    ideal = np.zeros((len(predicted), k), dtype=np.float64)
    for row, items in enumerate(ground_truth):
        ideal[row, : min(len(items), k)] = 1.0
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    idcg = np.sum(ideal * discounts, axis=1)
    dcg = np.sum(predicted * discounts, axis=1)
    idcg[idcg == 0.0] = 1.0
    values = dcg / idcg
    values[np.isnan(values)] = 0.0
    return float(np.sum(values))


def evaluate(model, train_matrix, eval_interactions, k=20, batch_size=256):
    if k > train_matrix.shape[1]:
        raise ValueError(
            f"Requested top-k {k} exceeds the number of items "
            f"({train_matrix.shape[1]})."
        )

    users = sorted(eval_interactions)
    recall_sum = 0.0
    ndcg_sum = 0.0
    for start in range(0, len(users), batch_size):
        batch_users = users[start : start + batch_size]
        observed = train_matrix[batch_users]
        scores = model.score_batch(observed)

        rows, cols = observed.nonzero()
        scores[rows, cols] = -np.inf
        candidate = np.argpartition(scores, -k, axis=1)[:, -k:]
        candidate_scores = np.take_along_axis(scores, candidate, axis=1)
        order = np.argsort(-candidate_scores, axis=1)
        ranked_items = np.take_along_axis(candidate, order, axis=1)

        ground_truth = [eval_interactions[user] for user in batch_users]
        relevance = np.asarray(
            [
                [item in labels for item in ranked]
                for ranked, labels in zip(ranked_items, ground_truth)
            ],
            dtype=np.float64,
        )
        recall_sum += recall_at_k(ground_truth, relevance, k)
        ndcg_sum += ndcg_at_k(ground_truth, relevance, k)

    num_users = len(users)
    return recall_sum / num_users, ndcg_sum / num_users
