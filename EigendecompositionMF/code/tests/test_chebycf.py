"""CPU regression checks against the repository's original ChebyCF.

Run with the lgn Python: python tests/test_chebycf.py
Definitions are extracted to avoid world.py's CLI parsing and dataset loading.
"""
import ast
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds
import torch
import torch.nn.functional as F

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE.parent / "TrainingFreeRecSys" / "ChebyCF"))
from src.model import ChebyCF as ReferenceChebyCF

torch.set_num_threads(4)
world = SimpleNamespace(config={"test_u_batch_size": 256}, topks=[5])
metric_namespace = {"np": np}
metric_tree = ast.parse((CODE / "metrics.py").read_text())
exec(compile(ast.Module(body=[n for n in metric_tree.body if isinstance(n, ast.FunctionDef)],
                        type_ignores=[]), "metrics.py", "exec"), metric_namespace)
namespace = {"torch": torch, "np": np, "sp": sp, "svds": svds,
             "math": math, "F": F, "BasicDataset": object, "world": world,
             "metrics": SimpleNamespace(**metric_namespace),
             "helpers": SimpleNamespace(minibatch=lambda users, batch_size:
                 (users[i:i + batch_size] for i in range(0, len(users), batch_size)))}
tree = ast.parse((CODE / "model.py").read_text())
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.ClassDef)],
                        type_ignores=[]), "model.py", "exec"), namespace)
TFmodel = namespace["TFmodel"]
LinearScoreMatrix = namespace["LinearScoreMatrix"]


def make_model(matrix):
    matrix = matrix.tocsr()
    dataset = SimpleNamespace(n_users=matrix.shape[0], m_items=matrix.shape[1],
                              UserItemNet=matrix)
    dataset.getUserPosItems = lambda users: [matrix[u].indices for u in users]
    dataset.testDict = {u: [int(np.flatnonzero(matrix[u].toarray().ravel() == 0)[0])]
                        for u in range(matrix.shape[0])}
    return TFmodel({"device": "cpu", "k": 4}, dataset)


class ChebyCFTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(2020)
        matrix = (rng.random((27, 33)) < 0.23).astype(np.float32)
        matrix[-1, :] = 0
        matrix[:, -1] = 0
        self.r = sp.csr_matrix(matrix)
        self.model = make_model(self.r)

    def test_reference_grid(self):
        cache = {}
        max_error = 0.0
        for eta in (3, 5):
            for K in (8, 32):
                for phi in (1.5, 10.0, 15.0):
                    for alpha in (0.1, 0.2):
                        for beta in (0.2, 0.4):
                            params = dict(K=K, phi=phi, eta=eta, alpha=alpha, beta=beta)
                            with self.subTest(**params):
                                ref = ReferenceChebyCF(**params)
                                ref.fit(self.r)
                                expected = ref(torch.from_numpy(self.r.toarray()))
                                actual = self.model.S_ChebyCF(**params, cache=cache,
                                                              batch_size=7, normalize=False)
                                max_error = max(max_error, float((expected - actual).abs().max()))
                                torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
                                self.assertEqual(actual.device.type, "cpu")
        print(f"48 reference configurations: max absolute score error = {max_error:.3g}")

    def test_disabled_components_and_batches(self):
        for eta, alpha, beta in ((0, 0.1, 0.2), (3, 0, 0.4), (3, 0.2, 0)):
            params = dict(K=32, phi=1.5, eta=eta, alpha=alpha, beta=beta)
            ref = ReferenceChebyCF(**params)
            ref.fit(self.r)
            expected = ref(torch.from_numpy(self.r.toarray()))
            for batch_size in (1, 7, 64):
                actual = self.model.S_ChebyCF(**params, batch_size=batch_size, normalize=False)
                torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)

    def test_cache_and_no_full_matrix_densification(self):
        cache = {}
        original_toarray = sp.csr_matrix.toarray
        batch_rows = []

        def checked_toarray(matrix, *args, **kwargs):
            batch_rows.append(matrix.shape[0])
            self.assertLessEqual(matrix.shape[0], 7)
            return original_toarray(matrix, *args, **kwargs)

        with patch.dict(namespace, {"svds": unittest.mock.Mock(wraps=svds)}):
            with patch.object(sp.csr_matrix, "toarray", checked_toarray):
                self.model.S_ChebyCF(eta=3, batch_size=7, cache=cache)
                self.model.S_ChebyCF(eta=3, phi=10, beta=0.4, batch_size=7, cache=cache)
                self.assertEqual(namespace["svds"].call_count, 1)
                self.model.S_ChebyCF(eta=5, batch_size=7, cache=cache)
                self.assertEqual(namespace["svds"].call_count, 2)
            other = make_model(self.r[:-1])
            other.S_ChebyCF(eta=5, cache=cache)
            self.assertEqual(namespace["svds"].call_count, 3)
        self.assertTrue(batch_rows)
        self.assertIs(cache["chebycf"]["owner"], other)
        for key in ("sparse", "degree", "ideal"):
            self.assertEqual(cache["chebycf"][key].device.type, "cpu")

    def test_evaluation_and_ensemble_do_not_mutate_scores(self):
        first = self.model.S_ChebyCF(eta=3, batch_size=7)
        second = torch.randn_like(first)
        first_copy, second_copy = first.clone(), second.clone()
        for weight in (0, 0.3, 1):
            combined = LinearScoreMatrix(first, second, weight)
            expected_scores = weight * first + (1 - weight) * second
            actual = self.model.Test_RatingMatrix(combined, self.model.dataset)
            expected = self.model.Test_RatingMatrix(expected_scores, self.model.dataset)
            for key in actual:
                np.testing.assert_array_equal(actual[key], expected[key])
        torch.testing.assert_close(first, first_copy)
        torch.testing.assert_close(second, second_copy)
        torch.testing.assert_close(first[:-1].sum(1), torch.ones(26), atol=1e-5, rtol=1e-5)
        self.assertTrue(torch.isfinite(first).all())
        self.assertEqual(float(first[-1].abs().sum()), 0)

    def test_old_saved_parameters_are_not_reused(self):
        main_tree = ast.parse((CODE / "main.py").read_text())
        loader = next(n for n in ast.walk(main_tree)
                      if isinstance(n, ast.FunctionDef) and n.name == "load_saved_parameters")
        import os
        with tempfile.TemporaryDirectory() as directory:
            params_file = Path(directory) / "params.txt"
            params_file.write_text(
                "toy : 2020 : DAN : [{'reg_p': 50}]\n"
                "toy : 2020 : ChebyCF : [{'K': 8}]\n"
                "toy : 2020 : ensemble:ChebyCF+DAN : {'ChebyCF': {'K': 8}, 'a_sum': 0.3}\n"
                "toy : 2020 : ensemble:ChebyCF+SGFCF : "
                "{'ChebyCF': {'K': 32, 'implementation_version': 2}, 'a_sum': 0.5}\n")
            env = {"params_file": params_file, "dataset_name": "toy", "ast": ast, "os": os,
                   "world": SimpleNamespace(seed=2020, cprint=lambda message: None)}
            exec(compile(ast.Module(body=[loader], type_ignores=[]), "main.py", "exec"), env)
            saved = env["load_saved_parameters"]()
            self.assertEqual(saved["DAN"], [{"reg_p": 50}])
            self.assertNotIn("ChebyCF", saved)
            self.assertNotIn("ensemble:ChebyCF+DAN", saved)
            self.assertIn("ensemble:ChebyCF+SGFCF", saved)


if __name__ == "__main__":
    unittest.main()
