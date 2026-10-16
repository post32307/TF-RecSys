import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu


class DAN:
    """Closed-form DAN linear recommender adapted to weak generalization."""

    SUPPORTED_MODELS = {"LAE_DAN", "EASE_DAN", "RLAE_DAN"}

    def __init__(
        self,
        model="EASE_DAN",
        reg_p=20.0,
        alpha=0.2,
        beta=0.3,
        drop_p=0.6,
        xi=0.1,
        solver="auto",
        block_size=256,
    ):
        if model not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unknown DAN base model {model!r}; choose one of "
                f"{sorted(self.SUPPORTED_MODELS)}"
            )
        if reg_p <= 0:
            raise ValueError("reg_p must be positive")
        if not 0 <= drop_p < 1:
            raise ValueError("drop_p must satisfy 0 <= drop_p < 1")
        if solver not in {"auto", "primal", "dual"}:
            raise ValueError("solver must be one of: auto, primal, dual")
        if block_size <= 0:
            raise ValueError("block_size must be positive")

        self.model = model
        self.reg_p = float(reg_p)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.drop_p = float(drop_p)
        self.xi = float(xi)
        self.solver = solver
        self.block_size = int(block_size)
        self.W = None

    def fit(self, interactions):
        X = interactions.astype(np.float64)
        self.interactions = X.tocsr()
        num_users = X.shape[0]
        num_items = X.shape[1]
        item_counts = np.asarray(X.sum(axis=0)).ravel()
        user_counts = np.asarray(X.sum(axis=1)).ravel()

        # Isolated users/items can occur when the held-out split contains IDs
        # absent from train. Clipping avoids infinities; cold item scores are
        # explicitly suppressed during evaluation.
        safe_user_counts = np.maximum(user_counts, 1.0)
        safe_item_counts = np.maximum(item_counts, 1.0)

        regularization = self.reg_p + (
            self.drop_p / (1.0 - self.drop_p)
        ) * item_counts
        self.left_scale = np.power(safe_item_counts, self.alpha)
        self.right_scale = np.power(safe_item_counts, -self.alpha)
        self.cold_items = item_counts == 0

        selected_solver = self.solver
        if selected_solver == "auto":
            selected_solver = (
                "dual"
                if (
                    self.model == "EASE_DAN"
                    and num_items > num_users
                    and num_items >= 50000
                )
                else "primal"
            )
        if selected_solver == "dual" and self.model != "EASE_DAN":
            raise ValueError("The exact dual solver currently supports EASE_DAN only")
        self.selected_solver = selected_solver

        if selected_solver == "dual":
            self._fit_dual(
                X, safe_user_counts, regularization, num_items
            )
        else:
            self._fit_primal(
                X, safe_user_counts, regularization, num_items
            )
        return self

    def _fit_primal(
        self, X, safe_user_counts, regularization, num_items
    ):
        normalized_xt = X.multiply(
            np.power(safe_user_counts, -self.beta)[:, None]
        ).T
        gram = normalized_xt.dot(X).toarray()
        diagonal = np.diag_indices(num_items)
        gram[diagonal] += regularization
        precision = np.linalg.inv(gram)

        if self.model == "EASE_DAN":
            base = np.eye(num_items) - precision / np.diag(precision)
        elif self.model == "LAE_DAN":
            base = np.eye(num_items) - precision * regularization
        else:
            diag_precision = np.diag(precision)
            condition = (1.0 - regularization * diag_precision) > self.xi
            lagrangian = (
                (1.0 - self.xi) / diag_precision - regularization
            ) * condition.astype(np.float64)
            base = np.eye(num_items) - precision * (
                regularization + lagrangian
            )

        self.W = (
            base
            * self.left_scale.reshape(-1, 1)
            * self.right_scale
        )
        self.W[diagonal] = 0.0

    def _fit_dual(
        self, X, safe_user_counts, regularization, num_items
    ):
        inverse_regularization = 1.0 / regularization
        middle = sp.diags(
            np.power(safe_user_counts, self.beta), format="csc"
        )
        middle += X.multiply(inverse_regularization).dot(X.T).tocsc()
        self.dual_lu = splu(middle)
        self.inverse_regularization = inverse_regularization
        self.precision_diag = np.empty(num_items, dtype=np.float64)

        for start in range(0, num_items, self.block_size):
            end = min(start + self.block_size, num_items)
            columns = X[:, start:end].toarray()
            solved = self.dual_lu.solve(columns)
            correction = np.einsum("ij,ij->j", columns, solved)
            inv_reg = inverse_regularization[start:end]
            self.precision_diag[start:end] = inv_reg - (
                inv_reg * inv_reg * correction
            )

        eps = np.finfo(np.float64).eps
        self.precision_diag[
            np.abs(self.precision_diag) < eps
        ] = eps

    def score_batch(self, interactions):
        if not hasattr(self, "selected_solver"):
            raise RuntimeError("fit must be called before score_batch")
        if self.selected_solver == "primal":
            scores = np.asarray(interactions.dot(self.W))
        else:
            scaled_input = interactions.multiply(self.left_scale)
            scaled_by_reg = scaled_input.multiply(
                self.inverse_regularization
            )
            overlap = scaled_by_reg.dot(self.interactions.T).toarray().T
            solved = self.dual_lu.solve(overlap)
            low_rank = np.asarray(self.interactions.T.dot(solved).T)
            input_precision = scaled_by_reg.toarray() - (
                low_rank * self.inverse_regularization
            )
            scores = (
                scaled_input.toarray()
                - input_precision / self.precision_diag
            ) * self.right_scale
        scores[:, self.cold_items] = -np.inf
        return scores
