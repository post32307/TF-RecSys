import world
from world import cprint
import torch
torch.set_num_threads(4)
import numpy as np
import os
import sys
from time import time
import time
from tensorboardX import SummaryWriter
from os.path import join
from math import log
# ==============================
import helpers
helpers.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import dataloader

from tqdm import tqdm

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


import model
Neg_k = 1

# init tensorboard
if world.tensorboard:
    w: SummaryWriter = SummaryWriter(
        join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
    )
else:
    w = None
    world.cprint("not enable tensorflowboard")

try:
    import scipy.sparse as sp
    import itertools
    from scipy.sparse.linalg import eigs
    from scipy.sparse.linalg import svds



    plot_true = 0
    #   0: skip eigenvalue plotting and continue to the model/search code
    #   1: plot real eigenvalues, save to plots/, and terminate with sys.exit()
    if plot_true not in (0, 1):
        raise ValueError("plot_true must be 0 or 1.")
    
    val_true = 1
    #   0: no hyperparameter search; test default params on test.txt
    #   1: train on train.txt, search on val.txt, test on test.txt
    #   2: train on train.txt, search on test.txt, test on test.txt
    if val_true not in (0, 1, 2):
        raise ValueError("val_true must be 0, 1, or 2.")



    if plot_true == 1:
        # ---------------------------------------------------------
        # Temporary diagnostic: plot asymmetric-normalized eigenvalues
        # ---------------------------------------------------------
        plot_data_root = os.environ.get(
            "TFRECSYS_DATA_ROOT",
            os.path.join(world.ROOT_PATH, "data", "split_dataset", f"00_dataset_seed_{world.seed}"),
        )
        plot_dataset_name = world.dataset
        plot_dataset_path = os.path.join(plot_data_root, plot_dataset_name)
        if not os.path.isdir(plot_dataset_path):
            raise FileNotFoundError(f"Dataset directory not found: {plot_dataset_path}")

        def plot_tau_delta_to_alpha_beta(tau, delta):
            norm_alpha = tau / 2.0 - delta
            norm_beta = tau / 2.0 + delta
            if norm_alpha <= 1e-12 or norm_beta <= 1e-12:
                raise ValueError
            return norm_alpha, norm_beta

        plot_tau_space = [0.2, 0.4, 0.6, 0.8]
        plot_delta_space = [0.0, 0.1, 0.2, 0.3]
        plot_pairs = []
        for plot_tau, plot_delta in itertools.product(plot_tau_space, plot_delta_space):
            try:
                plot_alpha, plot_beta = plot_tau_delta_to_alpha_beta(plot_tau, plot_delta)
            except ValueError:
                continue
            plot_pairs.append((plot_tau, plot_delta, plot_alpha, plot_beta))

        plot_dataset = dataloader.Loader(path=plot_dataset_path)
        plot_model = model.PSEmodel(world.config, plot_dataset)
        plot_k = max(1, min(int(world.config["k"]), plot_model.n_users + plot_model.m_items - 2))
        plot_rowsum = np.array(plot_model.A.sum(axis=1)).flatten()
        plot_rowsum[plot_rowsum == 0] = 1e-9
        plot_model.D_inv = sp.diags(1.0 / plot_rowsum)

        plot_asym_eigenvalues = []
        for plot_tau, plot_delta, plot_alpha, plot_beta in tqdm(
            plot_pairs,
            desc="Plotting asym eigenvalues",
            unit="cfg",
        ):
            plot_x_norm = plot_model.getNormAdj(
                plot_model.A,
                plot_model.D_inv,
                alpha1=plot_alpha,
                alpha2=plot_beta,
            )
            plot_eigs = eigs(plot_x_norm, k=plot_k, which="LR", return_eigenvectors=False)
            plot_asym_eigenvalues.append((plot_tau, plot_delta, plot_alpha, plot_beta, plot_eigs))

        plot_sgfcf_alpha_space = [0.0, 1.0, 5.0, 10.0, 20.0]
        plot_sgfcf_eps_space = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        plot_sgfcf_param_pairs = list(itertools.product(
            plot_sgfcf_alpha_space,
            plot_sgfcf_eps_space,
        ))
        plot_r = plot_model.R.astype(np.float32).tocsr()
        plot_user_degree = np.array(plot_r.sum(axis=1)).flatten()
        plot_item_degree = np.array(plot_r.sum(axis=0)).flatten()
        plot_sgfcf_eigenvalues = []
        for plot_sgfcf_alpha, plot_sgfcf_eps in tqdm(
            plot_sgfcf_param_pairs,
            desc="Plotting SGFCF eigenvalues",
            unit="cfg",
        ):
            with np.errstate(divide="ignore"):
                plot_d_u = 1.0 / np.power(plot_user_degree + plot_sgfcf_alpha, plot_sgfcf_eps)
                plot_d_i = 1.0 / np.power(plot_item_degree + plot_sgfcf_alpha, plot_sgfcf_eps)
            plot_d_u[np.isinf(plot_d_u)] = 0.0
            plot_d_i[np.isinf(plot_d_i)] = 0.0
            plot_r_tilde = sp.diags(plot_d_u).dot(plot_r).dot(sp.diags(plot_d_i)).tocsr()
            plot_sgfcf_adj = sp.bmat([[None, plot_r_tilde], [plot_r_tilde.T, None]], format="csr")
            plot_sgfcf_eigs = eigs(plot_sgfcf_adj, k=plot_k, which="LR", return_eigenvectors=False)
            plot_sgfcf_eigenvalues.append((plot_sgfcf_alpha, plot_sgfcf_eps, plot_sgfcf_eigs))

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plot_dir = os.path.join(os.path.dirname(__file__), "plots")
        os.makedirs(plot_dir, exist_ok=True)
        plot_file = os.path.join(
            plot_dir,
            f"{plot_dataset_name}_eignevalues_seed_{world.seed}_k_{plot_k}.png",
        )

        def normalized_spectrum(vals):
            vals = np.sort(np.abs(vals))[::-1]
            if len(vals) == 0 or vals[0] == 0:
                return vals
            return vals / vals[0]

        def energy_curve(vals):
            vals = np.sort(np.abs(vals))[::-1]
            energy = vals ** 2
            total_energy = np.sum(energy)
            if total_energy == 0:
                return np.zeros_like(energy)
            return np.cumsum(energy) / total_energy

        fig, axes = plt.subplots(2, 3, figsize=(21, 12))
        plot_tau_colors = {
            plot_tau: plt.get_cmap("tab10")(plot_idx % 10)
            for plot_idx, plot_tau in enumerate(plot_tau_space)
        }
        plot_delta_markers = {
            plot_delta: plot_marker
            for plot_delta, plot_marker in zip(
                plot_delta_space,
                ["o", "x", "+", "o"],
            )
        }
        for plot_tau, plot_delta, plot_alpha, plot_beta, plot_eigs in plot_asym_eigenvalues:
            plot_label = (
                f"tau={plot_tau}, delta={plot_delta} "
                f"(a={plot_alpha:.2f}, b={plot_beta:.2f})"
            )
            plot_color = plot_tau_colors[plot_tau]
            plot_marker = plot_delta_markers[plot_delta]
            plot_markerfacecolor = "none"
            plot_markersize = 8 if plot_delta == plot_delta_space[3] else 4.5
            sorted_real = np.sort(plot_eigs.real)[::-1]
            norm_vals = normalized_spectrum(plot_eigs)
            energy_vals = energy_curve(plot_eigs)
            axes[0, 0].plot(
                np.arange(1, len(sorted_real) + 1),
                sorted_real,
                color=plot_color,
                marker=plot_marker,
                markerfacecolor=plot_markerfacecolor,
                markeredgewidth=1.1,
                markersize=plot_markersize,
                linewidth=1,
                label=plot_label,
            )
            axes[0, 1].plot(
                np.arange(1, len(norm_vals) + 1),
                norm_vals,
                color=plot_color,
                marker=plot_marker,
                markerfacecolor=plot_markerfacecolor,
                markeredgewidth=1.1,
                markersize=plot_markersize,
                linewidth=1,
                label=plot_label,
            )
            axes[0, 2].plot(
                np.arange(1, len(energy_vals) + 1),
                energy_vals,
                color=plot_color,
                marker=plot_marker,
                markerfacecolor=plot_markerfacecolor,
                markeredgewidth=1.1,
                markersize=plot_markersize,
                linewidth=1,
                label=plot_label,
            )

        for plot_sgfcf_alpha, plot_sgfcf_eps, plot_eigs in plot_sgfcf_eigenvalues:
            plot_label = f"alpha={plot_sgfcf_alpha}, eps={plot_sgfcf_eps}"
            sorted_real = np.sort(plot_eigs.real)[::-1]
            norm_vals = normalized_spectrum(plot_eigs)
            energy_vals = energy_curve(plot_eigs)
            axes[1, 0].plot(np.arange(1, len(sorted_real) + 1), sorted_real, marker=".", linewidth=1, label=plot_label)
            axes[1, 1].plot(np.arange(1, len(norm_vals) + 1), norm_vals, marker=".", linewidth=1, label=plot_label)
            axes[1, 2].plot(np.arange(1, len(energy_vals) + 1), energy_vals, marker=".", linewidth=1, label=plot_label)

        axes[0, 0].set_title("Asymmetric Normalization: Real Eigenvalues")
        axes[0, 1].set_title("Asymmetric Normalization: Normalized Spectrum")
        axes[0, 2].set_title("Asymmetric Normalization: Energy Curve")
        axes[1, 0].set_title("SGFCF-Style Normalization: Real Eigenvalues")
        axes[1, 1].set_title("SGFCF-Style Normalization: Normalized Spectrum")
        axes[1, 2].set_title("SGFCF-Style Normalization: Energy Curve")
        for plot_ax in axes.flat:
            plot_ax.set_xlabel("Rank")
            plot_ax.grid(True, alpha=0.25)
            plot_ax.legend(fontsize=6, loc="best")
        axes[0, 0].set_ylabel("Eigenvalue Real Part")
        axes[0, 1].set_ylabel("lambda_i / lambda_1")
        axes[0, 2].set_ylabel("Cumulative Energy")
        axes[1, 0].set_ylabel("Eigenvalue Real Part")
        axes[1, 1].set_ylabel("lambda_i / lambda_1")
        axes[1, 2].set_ylabel("Cumulative Energy")
        fig.suptitle(f"{plot_dataset_name} eigenvalue spectra, seed={world.seed}, k={plot_k}")
        fig.tight_layout()
        fig.savefig(plot_file, dpi=200)
        plt.close(fig)
        print(f"Saved eigenvalue real-part plots to {plot_file}")
        sys.exit()



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

    max_k = world.config["k"]
    default_params = {
        "tau": 0.6,
        "delta": 0.1,
        "beta_1": 1.0,
        "beta_2": 1.0,
        "k": max_k,
    }
    sgfcf_seed_2020_defaults = {
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
    default_sgfcf_params = dict(sgfcf_seed_2020_defaults.get(
        dataset_name,
        {"k": max_k, "beta_1": 0.7, "beta_2": 1.1, "alpha": 0.0, "eps": 0.28, "gamma": 1.5},
    ))
    default_nonhomophily_sgfcf_params = {
        "k": default_sgfcf_params["k"],
        "beta": 1.0,
        "alpha": default_sgfcf_params["alpha"],
        "eps": default_sgfcf_params["eps"],
        "gamma": default_sgfcf_params["gamma"],
    }
    default_quantized_sgfcf_params = dict(default_sgfcf_params)
    beta_quantization_values = torch.tensor(
        [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3],
        # [0.7, 0.9, 1.1, 1.3],
        dtype=torch.float32,
        device=world.device,
    )
    tau_space = [0.2, 0.4, 0.6, 0.8]
    delta_space = [0.0, 0.1, 0.2, 0.3]
    individual_filter_space = [
        (1.0, 1.0),
        (0.7, 1.1),
        (0.9, 1.0),
        (1.0, 1.3),
        (0.3, 2.5),
        (2.0, 2.4),
        (4.3, 5.1),
        (2.5, 6.5),
    ]
    nonhomophily_sgfcf_beta_space = [0.0, 0.5, 1.0, 1.5, 2.0]

    fixed_asym_alpha = 0.2
    fixed_asym_beta = 0.4
    bpr_renewal_coeff_space = [0.0, 0.05, 0.1, 0.2, 0.5]
    bpr_hard_negative_k = max(world.topks)
    default_bpr_renewal_params = {
        "mqqt_coeff": 0.0,
        "pptm_coeff": 0.0,
        "hard_negative_k": bpr_hard_negative_k,
    }
    bpr_renewal_variant_labels = {
        "plain": "Baseline Asymmetric Eigenvector: U_R U_R^T",
        "lambda": "Baseline Asymmetric Eigenvector: U_R Lambda U_R^T",
        "lambda2": "Baseline Asymmetric Eigenvector: U_R Lambda^2 U_R^T",
    }

    def default_bpr_renewal_params_by_variant():
        return {
            variant_name: dict(default_bpr_renewal_params)
            for variant_name in bpr_renewal_variant_labels
        }

    def normalize_bpr_renewal_params_by_variant(params_by_variant):
        if "mqqt_coeff" in params_by_variant and "pptm_coeff" in params_by_variant:
            return {
                variant_name: dict(params_by_variant)
                for variant_name in bpr_renewal_variant_labels
            }
        return {
            variant_name: dict(params_by_variant.get(variant_name, default_bpr_renewal_params))
            for variant_name in bpr_renewal_variant_labels
        }

    def alpha_beta_to_tau_delta(norm_alpha, norm_beta):
        return norm_alpha + norm_beta, (norm_beta - norm_alpha) / 2.0

    def tau_delta_to_alpha_beta(tau, delta):
        norm_alpha = tau / 2.0 - delta
        norm_beta = tau / 2.0 + delta
        min_norm_power = 1e-12
        if norm_alpha <= min_norm_power or norm_beta <= min_norm_power:
            raise ValueError(
                f"Skipped tau/delta pair: tau={tau}, delta={delta} gives "
                f"alpha={norm_alpha}, beta={norm_beta}."
            )
        return norm_alpha, norm_beta

    def is_valid_tau_delta(tau, delta):
        try:
            tau_delta_to_alpha_beta(tau, delta)
            return True
        except ValueError:
            return False

    tau_delta_to_alpha_beta(default_params["tau"], default_params["delta"])

    def load_dataset(eval_file):
        previous_eval_file = os.environ.get("TFRECSYS_EVAL_FILE")
        os.environ["TFRECSYS_EVAL_FILE"] = eval_file
        try:
            return dataloader.Loader(path=dataset_path)
        finally:
            if previous_eval_file is None:
                os.environ.pop("TFRECSYS_EVAL_FILE", None)
            else:
                os.environ["TFRECSYS_EVAL_FILE"] = previous_eval_file

    def fprint(*args, **kwargs):
        print(*args, **kwargs)
        print(*args, **kwargs, file=out_file)

    def prepare_eval_batches(current_dataset):
        u_batch_size = world.config["test_u_batch_size"]
        test_dict = current_dataset.testDict
        users = list(test_dict.keys())
        if len(users) == 0:
            return [], 0

        u_batch_size = min(u_batch_size, max(1, len(users) // 10))
        eval_batches = []
        for batch_users in helpers.minibatch(users, batch_size=u_batch_size):
            eval_batches.append((
                batch_users,
                current_dataset.getUserPosItems(batch_users),
                [test_dict[u] for u in batch_users],
                torch.Tensor(batch_users).long().to(world.device),
            ))
        return eval_batches, len(users)

    def test_factorized_rating(current_model, current_dataset, user_factors, item_factors, eval_batches=None, user_count=None):
        max_k_eval = max(world.topks)
        results = {
            "precision": np.zeros(len(world.topks)),
            "recall": np.zeros(len(world.topks)),
            "ndcg": np.zeros(len(world.topks)),
        }
        if eval_batches is None or user_count is None:
            eval_batches, user_count = prepare_eval_batches(current_dataset)
        if user_count == 0:
            return results

        with torch.no_grad():
            for batch_users, all_pos, ground_true, batch_users_gpu in eval_batches:
                rating = user_factors[batch_users_gpu, :] @ item_factors.t()

                exclude_index = []
                exclude_items = []
                for range_i, items_pos in enumerate(all_pos):
                    exclude_index.extend([range_i] * len(items_pos))
                    exclude_items.extend(items_pos)
                if exclude_index:
                    rating[exclude_index, exclude_items] = -(1 << 10)

                _, rating_k = torch.topk(rating, k=max_k_eval)
                batch_result = current_model.test_one_batch((rating_k.cpu(), ground_true))
                results["recall"] += batch_result["recall"]
                results["precision"] += batch_result["precision"]
                results["ndcg"] += batch_result["ndcg"]
                del rating, rating_k

        results["recall"] /= float(user_count)
        results["precision"] /= float(user_count)
        results["ndcg"] /= float(user_count)
        return results

    def test_sparse_rating_batches(current_model, current_dataset, rating_batch_fn, eval_batches=None, user_count=None):
        max_k_eval = max(world.topks)
        results = {
            "precision": np.zeros(len(world.topks)),
            "recall": np.zeros(len(world.topks)),
            "ndcg": np.zeros(len(world.topks)),
        }
        if eval_batches is None or user_count is None:
            eval_batches, user_count = prepare_eval_batches(current_dataset)
        if user_count == 0:
            return results

        with torch.no_grad():
            for batch_users, all_pos, ground_true, _ in eval_batches:
                rating_np = rating_batch_fn(np.asarray(batch_users, dtype=np.int64))
                rating = torch.as_tensor(rating_np, dtype=torch.float32, device=world.device)

                exclude_index = []
                exclude_items = []
                for range_i, items_pos in enumerate(all_pos):
                    exclude_index.extend([range_i] * len(items_pos))
                    exclude_items.extend(items_pos)
                if exclude_index:
                    rating[exclude_index, exclude_items] = -(1 << 10)

                _, rating_k = torch.topk(rating, k=max_k_eval)
                batch_result = current_model.test_one_batch((rating_k.cpu(), ground_true))
                results["recall"] += batch_result["recall"]
                results["precision"] += batch_result["precision"]
                results["ndcg"] += batch_result["ndcg"]
                del rating, rating_k

        results["recall"] /= float(user_count)
        results["precision"] /= float(user_count)
        results["ndcg"] /= float(user_count)
        return results

    def row_normalize_sparse(mat):
        mat = mat.astype(np.float32).tocsr()
        row_sum = np.array(mat.sum(axis=1)).flatten()
        inv_row_sum = np.zeros_like(row_sum, dtype=np.float32)
        nonzero_rows = row_sum > 0
        inv_row_sum[nonzero_rows] = 1.0 / row_sum[nonzero_rows]
        return sp.diags(inv_row_sum).dot(mat).astype(np.float32).tocsr()

    def build_hard_negative_matrix(current_model, current_dataset, user_factors, item_factors, hard_negative_k):
        hard_negative_k = max(1, min(int(hard_negative_k), current_model.m_items))
        batch_size = world.config["test_u_batch_size"]
        rows = []
        cols = []

        with torch.no_grad():
            for batch_users in helpers.minibatch(list(range(current_model.n_users)), batch_size=batch_size):
                batch_users_gpu = torch.Tensor(batch_users).long().to(world.device)
                rating = user_factors[batch_users_gpu, :] @ item_factors.t()
                all_pos = current_dataset.getUserPosItems(batch_users)

                exclude_index = []
                exclude_items = []
                for range_i, items_pos in enumerate(all_pos):
                    exclude_index.extend([range_i] * len(items_pos))
                    exclude_items.extend(items_pos)
                if exclude_index:
                    rating[exclude_index, exclude_items] = -(1 << 10)

                _, hard_items = torch.topk(rating, k=hard_negative_k)
                hard_items = hard_items.cpu().numpy()
                for range_i, user_id in enumerate(batch_users):
                    pos_set = set(all_pos[range_i])
                    max_candidates = current_model.m_items - len(pos_set)
                    if max_candidates <= 0:
                        continue
                    selected = []
                    for item_id in hard_items[range_i]:
                        item_id = int(item_id)
                        if item_id not in pos_set:
                            selected.append(item_id)
                        if len(selected) >= min(hard_negative_k, max_candidates):
                            break
                    rows.extend([user_id] * len(selected))
                    cols.extend(selected)

                del rating, hard_items

        data = np.ones(len(rows), dtype=np.float32)
        return sp.csr_matrix(
            (data, (rows, cols)),
            shape=(current_model.n_users, current_model.m_items),
            dtype=np.float32,
        )

    def prepare_bpr_renewal_terms(current_model, current_dataset, user_factors, item_factors, hard_negative_k):
        fprint(f"Constructing hard negatives for one-step BPR renewal, top_k={hard_negative_k}.")
        hard_negative_matrix = build_hard_negative_matrix(
            current_model,
            current_dataset,
            user_factors,
            item_factors,
            hard_negative_k,
        )
        contrast_matrix = (
            row_normalize_sparse(current_model.R)
            - row_normalize_sparse(hard_negative_matrix)
        ).astype(np.float32).tocsr()

        user_factors_np = user_factors.detach().cpu().numpy().astype(np.float32)
        item_factors_np = item_factors.detach().cpu().numpy().astype(np.float32)
        p_t_m = contrast_matrix.T.dot(user_factors_np).T.astype(np.float32)
        p_t_m = torch.as_tensor(p_t_m, dtype=torch.float32, device=world.device)
        return contrast_matrix, user_factors_np, item_factors_np, p_t_m

    def evaluate_bpr_renewal_params(
        current_model,
        current_dataset,
        user_factors,
        item_factors,
        bpr_renewal_params,
        renewal_terms=None,
        eval_batches=None,
        user_count=None,
        desc="BPR Renewal Eval",
    ):
        hard_negative_k = int(bpr_renewal_params.get("hard_negative_k", bpr_hard_negative_k))
        mqqt_coeff = bpr_renewal_params["mqqt_coeff"]
        pptm_coeff = bpr_renewal_params["pptm_coeff"]
        if np.isclose(mqqt_coeff, 0.0) and np.isclose(pptm_coeff, 0.0):
            return test_factorized_rating(
                current_model,
                current_dataset,
                user_factors,
                item_factors,
                eval_batches,
                user_count,
            )

        if renewal_terms is None:
            renewal_terms = prepare_bpr_renewal_terms(
                current_model,
                current_dataset,
                user_factors,
                item_factors,
                hard_negative_k,
            )
        contrast_matrix, _, item_factors_np, p_t_m = renewal_terms
        item_factors_t = item_factors.t()

        results = {
            "precision": np.zeros(len(world.topks)),
            "recall": np.zeros(len(world.topks)),
            "ndcg": np.zeros(len(world.topks)),
        }
        if eval_batches is None or user_count is None:
            eval_batches, user_count = prepare_eval_batches(current_dataset)
        if user_count == 0:
            return results

        max_k_eval = max(world.topks)
        with torch.no_grad():
            with tqdm(total=len(eval_batches), desc=desc, unit="batch") as pbar:
                for batch_users, all_pos, ground_true, batch_users_gpu in eval_batches:
                    batch_users_np = np.asarray(batch_users, dtype=np.int64)
                    base_rating = user_factors[batch_users_gpu, :] @ item_factors_t
                    m_q = contrast_matrix[batch_users_np].dot(item_factors_np).astype(np.float32)
                    item_side_term = torch.as_tensor(m_q, dtype=torch.float32, device=world.device) @ item_factors_t
                    user_side_term = user_factors[batch_users_gpu, :] @ p_t_m
                    rating = (
                        base_rating
                        + mqqt_coeff * item_side_term
                        + pptm_coeff * user_side_term
                    )

                    exclude_index = []
                    exclude_items = []
                    for range_i, items_pos in enumerate(all_pos):
                        exclude_index.extend([range_i] * len(items_pos))
                        exclude_items.extend(items_pos)
                    if exclude_index:
                        rating[exclude_index, exclude_items] = -(1 << 10)

                    _, rating_k = torch.topk(rating, k=max_k_eval)
                    batch_result = current_model.test_one_batch((rating_k.cpu(), ground_true))
                    results["recall"] += batch_result["recall"]
                    results["precision"] += batch_result["precision"]
                    results["ndcg"] += batch_result["ndcg"]
                    del base_rating, item_side_term, user_side_term, rating, rating_k
                    pbar.update(1)

        results["recall"] /= float(user_count)
        results["precision"] /= float(user_count)
        results["ndcg"] /= float(user_count)
        return results

    def search_bpr_renewal_coefficients(current_model, current_dataset, user_factors, item_factors):
        eval_batches, user_count = prepare_eval_batches(current_dataset)
        param_pairs = list(itertools.product(bpr_renewal_coeff_space, bpr_renewal_coeff_space))
        renewal_terms = prepare_bpr_renewal_terms(
            current_model,
            current_dataset,
            user_factors,
            item_factors,
            bpr_hard_negative_k,
        )
        contrast_matrix, _, item_factors_np, p_t_m = renewal_terms
        item_factors_t = item_factors.t()
        results_by_params = {
            param_pair: {
                "precision": np.zeros(len(world.topks)),
                "recall": np.zeros(len(world.topks)),
                "ndcg": np.zeros(len(world.topks)),
            }
            for param_pair in param_pairs
        }
        max_k_eval = max(world.topks)

        if user_count == 0:
            empty_results = {
                "precision": np.zeros(len(world.topks)),
                "recall": np.zeros(len(world.topks)),
                "ndcg": np.zeros(len(world.topks)),
            }
            return {
                "mqqt_coeff": 0.0,
                "pptm_coeff": 0.0,
                "hard_negative_k": bpr_hard_negative_k,
            }, empty_results, 0.0

        with torch.no_grad():
            with tqdm(total=len(eval_batches) * len(param_pairs), desc="BPR Renewal Coeff Search", unit="cfg") as pbar:
                for batch_users, all_pos, ground_true, batch_users_gpu in eval_batches:
                    batch_users_np = np.asarray(batch_users, dtype=np.int64)
                    base_rating = user_factors[batch_users_gpu, :] @ item_factors_t

                    m_q = contrast_matrix[batch_users_np].dot(item_factors_np).astype(np.float32)
                    item_side_term = torch.as_tensor(m_q, dtype=torch.float32, device=world.device) @ item_factors_t
                    user_side_term = user_factors[batch_users_gpu, :] @ p_t_m

                    exclude_index = []
                    exclude_items = []
                    for range_i, items_pos in enumerate(all_pos):
                        exclude_index.extend([range_i] * len(items_pos))
                        exclude_items.extend(items_pos)

                    for mqqt_coeff, pptm_coeff in param_pairs:
                        rating = (
                            base_rating
                            + mqqt_coeff * item_side_term
                            + pptm_coeff * user_side_term
                        )
                        if exclude_index:
                            rating[exclude_index, exclude_items] = -(1 << 10)

                        _, rating_k = torch.topk(rating, k=max_k_eval)
                        batch_result = current_model.test_one_batch((rating_k.cpu(), ground_true))
                        current_results = results_by_params[(mqqt_coeff, pptm_coeff)]
                        current_results["recall"] += batch_result["recall"]
                        current_results["precision"] += batch_result["precision"]
                        current_results["ndcg"] += batch_result["ndcg"]
                        del rating, rating_k
                        pbar.update(1)

                    del base_rating, item_side_term, user_side_term

        best_params = None
        best_results = None
        best_score = -1.0
        for mqqt_coeff, pptm_coeff in param_pairs:
            current_results = results_by_params[(mqqt_coeff, pptm_coeff)]
            current_results["recall"] /= float(user_count)
            current_results["precision"] /= float(user_count)
            current_results["ndcg"] /= float(user_count)
            score = tuning_score(current_results)
            if best_results is None or score > best_score:
                best_params = {
                    "mqqt_coeff": mqqt_coeff,
                    "pptm_coeff": pptm_coeff,
                    "hard_negative_k": bpr_hard_negative_k,
                }
                best_results = current_results
                best_score = score

        return best_params, best_results, best_score

    def tuning_score(results):
        # return results["recall"][0] + results["ndcg"][0]
        return results["recall"][0]

    def print_metrics(label, results):
        fprint(label)
        fprint(f"Recall@20: {results['recall'][0]:.6f}")
        fprint(f"Precision: {results['precision'].tolist()}")
        fprint(f"NDCG:      {results['ndcg'].tolist()}")

    def individual_weight(eig_values, homo_ratio, beta_1, beta_2):
        eig_values = torch.clamp(eig_values, min=1e-12)
        y_min, y_max = beta_1, beta_2
        x_min, x_max = homo_ratio.min(), homo_ratio.max()
        if torch.isclose(x_min, x_max).item():
            homo_weight = torch.ones_like(homo_ratio) * y_min
        else:
            homo_weight = (y_max - y_min) / (x_max - x_min) * homo_ratio
            homo_weight = homo_weight + (x_max * y_min - y_max * x_min) / (x_max - x_min)
        return eig_values.unsqueeze(0).pow(homo_weight.unsqueeze(1))

    def quantized_individual_weight(eig_values, homo_ratio, beta_1, beta_2):
        eig_values = torch.clamp(eig_values, min=1e-12)
        y_min, y_max = beta_1, beta_2
        x_min, x_max = homo_ratio.min(), homo_ratio.max()
        if torch.isclose(x_min, x_max).item():
            homo_weight = torch.ones_like(homo_ratio) * y_min
        else:
            homo_weight = (y_max - y_min) / (x_max - x_min) * homo_ratio
            homo_weight = homo_weight + (x_max * y_min - y_max * x_min) / (x_max - x_min)
        quantization_values = beta_quantization_values.to(homo_weight.device)
        nearest_idx = torch.argmin(
            torch.abs(homo_weight.unsqueeze(1) - quantization_values.unsqueeze(0)),
            dim=1,
        )
        homo_weight = quantization_values[nearest_idx]
        return eig_values.unsqueeze(0).pow(homo_weight.unsqueeze(1))

    def threshold_offdiag_graph(mat):
        graph = mat.tocsr(copy=True)
        graph.setdiag(0)
        graph.eliminate_zeros()
        graph.data[graph.data <= 1] = 0
        graph.eliminate_zeros()
        graph.data = np.ones(graph.nnz, dtype=np.int8)
        return graph

    def count_induced_edges(graph, nodes, marker, stamp):
        size = len(nodes)
        if size <= 1:
            return 0, stamp
        stamp += 1
        if stamp == np.iinfo(marker.dtype).max:
            marker.fill(0)
            stamp = 1
        marker[nodes] = stamp
        count = 0
        indptr = graph.indptr
        indices = graph.indices
        for node in nodes:
            neighbors = indices[indptr[node]:indptr[node + 1]]
            count += np.count_nonzero(marker[neighbors] == stamp)
        return count, stamp

    def compute_homophily_ratios(current_model):
        freq_matrix = current_model.R.astype(np.float32).tocsr()
        n_users, n_items = freq_matrix.shape
        train_data = [freq_matrix[u].indices for u in range(n_users)]
        freq_matrix_csc = freq_matrix.tocsc()
        train_data_item = [freq_matrix_csc[:, i].indices for i in range(n_items)]

        item_graph = threshold_offdiag_graph(freq_matrix.T.dot(freq_matrix))
        user_graph = threshold_offdiag_graph(freq_matrix.dot(freq_matrix.T))

        homo_ratio_user = np.zeros(n_users, dtype=np.float32)
        item_marker = np.zeros(n_items, dtype=np.int32)
        item_stamp = 0
        for user_idx, items_u in enumerate(train_data):
            size = len(items_u)
            if size > 1:
                edge_count, item_stamp = count_induced_edges(item_graph, items_u, item_marker, item_stamp)
                homo_ratio_user[user_idx] = edge_count / (size * (size - 1))

        homo_ratio_item = np.zeros(n_items, dtype=np.float32)
        user_marker = np.zeros(n_users, dtype=np.int32)
        user_stamp = 0
        for item_idx, users_i in enumerate(train_data_item):
            size = len(users_i)
            if size > 1:
                edge_count, user_stamp = count_induced_edges(user_graph, users_i, user_marker, user_stamp)
                homo_ratio_item[item_idx] = edge_count / (size * (size - 1))

        return (
            torch.as_tensor(homo_ratio_user, dtype=torch.float32, device=world.device),
            torch.as_tensor(homo_ratio_item, dtype=torch.float32, device=world.device),
        )

    def safe_rank(current_model, requested_k):
        total_nodes = current_model.n_users + current_model.m_items
        return max(1, min(int(requested_k), total_nodes - 2))

    def compute_asym_components(current_model, tau, delta, k):
        current_k = safe_rank(current_model, k)
        original_k = current_model.k
        current_model.k = current_k
        norm_alpha, norm_beta = tau_delta_to_alpha_beta(tau, delta)
        rowsum = np.array(current_model.A.sum(axis=1)).flatten()
        rowsum[rowsum == 0] = 1e-9
        current_model.D_inv = sp.diags(1.0 / rowsum)
        X_norm = current_model.getNormAdj(
            current_model.A,
            current_model.D_inv,
            alpha1=norm_alpha,
            alpha2=norm_beta,
        )
        _, U_R, eigs = current_model.Eigendecomposition_Asym(X_norm)
        current_model.k = original_k
        eig_filter_values = torch.abs(eigs[:current_k])
        eig_filter_values = eig_filter_values / torch.clamp(torch.max(eig_filter_values), min=1e-12)
        return U_R[:, :current_k], eigs[:current_k], eig_filter_values

    def build_fixed_asym_lambda2_factors(current_model, k):
        current_k = safe_rank(current_model, k)
        original_k = current_model.k
        current_model.k = current_k
        rowsum = np.array(current_model.A.sum(axis=1)).flatten()
        rowsum[rowsum == 0] = 1e-9
        current_model.D_inv = sp.diags(1.0 / rowsum)
        X_norm = current_model.getNormAdj(
            current_model.A,
            current_model.D_inv,
            alpha1=fixed_asym_alpha,
            alpha2=fixed_asym_beta,
        )
        _, U_R, eigs = current_model.Eigendecomposition_Asym(X_norm)
        current_model.k = original_k
        U_R = U_R[:, :current_k]
        eigs = eigs[:current_k]
        user_U = U_R[:current_model.n_users, :]
        item_U = U_R[current_model.n_users:, :]
        lambda2_user_factors = user_U * (eigs ** 2).unsqueeze(0)
        lambda2_item_factors = item_U
        return user_U, item_U, eigs, lambda2_user_factors, lambda2_item_factors

    def build_fixed_asym_bpr_renewal_variants(current_model, k):
        user_U, item_U, eigs, lambda2_user_factors, lambda2_item_factors = build_fixed_asym_lambda2_factors(
            current_model,
            k,
        )
        return {
            "plain": (user_U, item_U),
            "lambda": (user_U * eigs.unsqueeze(0), item_U),
            "lambda2": (lambda2_user_factors, lambda2_item_factors),
        }

    def build_individual_filter_factors(current_model, params, homo_user, homo_item, component_cache):
        cache_key = (params["tau"], params["delta"], params["k"])
        if cache_key not in component_cache:
            component_cache[cache_key] = compute_asym_components(
                current_model,
                params["tau"],
                params["delta"],
                params["k"],
            )

        U_R, _, eig_filter_values = component_cache[cache_key]

        user_filter = individual_weight(eig_filter_values, homo_user, params["beta_1"], params["beta_2"])
        item_filter = individual_weight(eig_filter_values, homo_item, params["beta_1"], params["beta_2"])
        user_factors = U_R[:current_model.n_users, :] * user_filter
        item_factors = U_R[current_model.n_users:, :] * item_filter
        return user_factors, item_factors

    def compute_sgfcf_components(current_model, params, sgfcf_component_cache):
        current_k = max(1, min(int(params["k"]), min(current_model.n_users, current_model.m_items) - 1))
        cache_key = (current_k, params["alpha"], params["eps"])
        if cache_key not in sgfcf_component_cache:
            freq_matrix = current_model.R.astype(np.float32).tocsr()
            user_degree = np.array(freq_matrix.sum(axis=1)).flatten()
            item_degree = np.array(freq_matrix.sum(axis=0)).flatten()
            with np.errstate(divide="ignore"):
                d_u = 1.0 / np.power(user_degree + params["alpha"], params["eps"])
                d_i = 1.0 / np.power(item_degree + params["alpha"], params["eps"])
            d_u[np.isinf(d_u)] = 0.0
            d_i[np.isinf(d_i)] = 0.0

            norm_freq_matrix = sp.diags(d_u).dot(freq_matrix).dot(sp.diags(d_i)).astype(np.float32).tocsr()
            svd_rank = min(current_k + 200, min(current_model.n_users, current_model.m_items) - 1)
            if svd_rank < 1:
                raise ValueError("SGFCF requires at least two users/items for SVD.")
            u_mat, s_mat, vt_mat = svds(norm_freq_matrix, k=svd_rank)
            sort_idx = np.argsort(s_mat)[::-1]
            s_mat = s_mat[sort_idx]
            u_mat = u_mat[:, sort_idx]
            vt_mat = vt_mat[sort_idx, :]

            U = torch.as_tensor(u_mat, dtype=torch.float32, device=world.device)
            value = torch.as_tensor(s_mat, dtype=torch.float32, device=world.device)
            value = value / torch.clamp(torch.max(value), min=1e-12)
            V = torch.as_tensor(vt_mat.T, dtype=torch.float32, device=world.device)
            sgfcf_component_cache[cache_key] = (
                U,
                value,
                V,
                norm_freq_matrix.tocsr(),
                norm_freq_matrix.tocsc(),
                current_k,
            )

        return sgfcf_component_cache[cache_key]

    def build_sgfcf_factors(current_model, params, homo_user, homo_item, sgfcf_component_cache):
        U, value, V, norm_freq_csr, norm_freq_csc, current_k = compute_sgfcf_components(
            current_model,
            params,
            sgfcf_component_cache,
        )
        value_slice = value[:current_k]
        user_filter = individual_weight(value_slice, homo_user, params["beta_1"], params["beta_2"])
        item_filter = individual_weight(value_slice, homo_item, params["beta_1"], params["beta_2"])
        user_factors = U[:, :current_k] * user_filter
        item_factors = V[:, :current_k] * item_filter
        return user_factors, item_factors, norm_freq_csr, norm_freq_csc

    def build_quantized_sgfcf_factors(current_model, params, homo_user, homo_item, sgfcf_component_cache):
        U, value, V, norm_freq_csr, norm_freq_csc, current_k = compute_sgfcf_components(
            current_model,
            params,
            sgfcf_component_cache,
        )
        value_slice = value[:current_k]
        user_filter = quantized_individual_weight(value_slice, homo_user, params["beta_1"], params["beta_2"])
        item_filter = quantized_individual_weight(value_slice, homo_item, params["beta_1"], params["beta_2"])
        user_factors = U[:, :current_k] * user_filter
        item_factors = V[:, :current_k] * item_filter
        return user_factors, item_factors, norm_freq_csr, norm_freq_csc

    def build_nonhomophily_sgfcf_factors(current_model, params, sgfcf_component_cache):
        U, value, V, norm_freq_csr, norm_freq_csc, current_k = compute_sgfcf_components(
            current_model,
            params,
            sgfcf_component_cache,
        )
        global_filter = torch.clamp(value[:current_k], min=1e-12).pow(params["beta"]).unsqueeze(0)
        user_factors = U[:, :current_k] * global_filter
        item_factors = V[:, :current_k] * global_filter
        return user_factors, item_factors, norm_freq_csr, norm_freq_csc

    def test_sgfcf_rating(
        current_model,
        current_dataset,
        user_factors,
        item_factors,
        norm_freq_csr,
        norm_freq_csc,
        gamma,
        eval_batches=None,
        user_count=None,
    ):
        max_k_eval = max(world.topks)
        results = {
            "precision": np.zeros(len(world.topks)),
            "recall": np.zeros(len(world.topks)),
            "ndcg": np.zeros(len(world.topks)),
        }
        if eval_batches is None or user_count is None:
            eval_batches, user_count = prepare_eval_batches(current_dataset)
        if user_count == 0:
            return results

        with torch.no_grad():
            for batch_users, all_pos, ground_true, batch_users_gpu in eval_batches:
                rating = user_factors[batch_users_gpu, :] @ item_factors.t()
                rating_sum = rating.sum(dim=1, keepdim=True)
                rating_sum[rating_sum == 0] = 1.0
                rating = rating / rating_sum

                norm_conv = norm_freq_csr[batch_users].dot(norm_freq_csc.T).dot(norm_freq_csr).toarray()
                norm_conv_sum = norm_conv.sum(axis=1, keepdims=True)
                norm_conv_sum[norm_conv_sum == 0] = 1.0
                norm_conv = norm_conv / norm_conv_sum
                norm_conv_t = torch.as_tensor(norm_conv, dtype=torch.float32, device=world.device)
                rating = torch.sigmoid(rating + gamma * norm_conv_t)

                exclude_index = []
                exclude_items = []
                for range_i, items_pos in enumerate(all_pos):
                    exclude_index.extend([range_i] * len(items_pos))
                    exclude_items.extend(items_pos)
                if exclude_index:
                    rating[exclude_index, exclude_items] = -(1 << 10)

                _, rating_k = torch.topk(rating, k=max_k_eval)
                batch_result = current_model.test_one_batch((rating_k.cpu(), ground_true))
                results["recall"] += batch_result["recall"]
                results["precision"] += batch_result["precision"]
                results["ndcg"] += batch_result["ndcg"]
                del rating, rating_k

        results["recall"] /= float(user_count)
        results["precision"] /= float(user_count)
        results["ndcg"] /= float(user_count)
        return results

    def evaluate_individual_filter_model(current_model, current_dataset, params, homo_user, homo_item, component_cache, eval_batches=None, user_count=None):
        user_factors, item_factors = build_individual_filter_factors(
            current_model,
            params,
            homo_user,
            homo_item,
            component_cache,
        )
        return test_factorized_rating(
            current_model,
            current_dataset,
            user_factors,
            item_factors,
            eval_batches,
            user_count,
        )

    def evaluate_sgfcf_model(current_model, current_dataset, params, homo_user, homo_item, sgfcf_component_cache, eval_batches=None, user_count=None):
        user_factors, item_factors, norm_freq_csr, norm_freq_csc = build_sgfcf_factors(
            current_model,
            params,
            homo_user,
            homo_item,
            sgfcf_component_cache,
        )
        return test_sgfcf_rating(
            current_model,
            current_dataset,
            user_factors,
            item_factors,
            norm_freq_csr,
            norm_freq_csc,
            params["gamma"],
            eval_batches,
            user_count,
        )

    def evaluate_quantized_sgfcf_model(current_model, current_dataset, params, homo_user, homo_item, sgfcf_component_cache, eval_batches=None, user_count=None):
        user_factors, item_factors, norm_freq_csr, norm_freq_csc = build_quantized_sgfcf_factors(
            current_model,
            params,
            homo_user,
            homo_item,
            sgfcf_component_cache,
        )
        return test_sgfcf_rating(
            current_model,
            current_dataset,
            user_factors,
            item_factors,
            norm_freq_csr,
            norm_freq_csc,
            params["gamma"],
            eval_batches,
            user_count,
        )

    def evaluate_nonhomophily_sgfcf_model(current_model, current_dataset, params, sgfcf_component_cache, eval_batches=None, user_count=None):
        user_factors, item_factors, norm_freq_csr, norm_freq_csc = build_nonhomophily_sgfcf_factors(
            current_model,
            params,
            sgfcf_component_cache,
        )
        return test_sgfcf_rating(
            current_model,
            current_dataset,
            user_factors,
            item_factors,
            norm_freq_csr,
            norm_freq_csc,
            params["gamma"],
            eval_batches,
            user_count,
        )

    def search_hyperparameters(search_model, search_dataset):
        search_batches, search_user_count = prepare_eval_batches(search_dataset)
        homo_user, homo_item = compute_homophily_ratios(search_model)
        component_cache = {}
        best_params = None
        best_results = None
        best_score = -1.0

        tau_delta_pairs = [
            (tau, delta)
            for tau, delta in itertools.product(tau_space, delta_space)
            if is_valid_tau_delta(tau, delta)
        ]
        search_space = list(itertools.product(tau_delta_pairs, individual_filter_space))
        with tqdm(total=len(search_space), desc="Individual Filter Search", unit="cfg") as pbar:
            for (tau, delta), (beta_1, beta_2) in search_space:
                params = {
                    "tau": tau,
                    "delta": delta,
                    "beta_1": beta_1,
                    "beta_2": beta_2,
                    "k": max_k,
                }
                results = evaluate_individual_filter_model(
                    search_model,
                    search_dataset,
                    params,
                    homo_user,
                    homo_item,
                    component_cache,
                    search_batches,
                    search_user_count,
                )
                score = tuning_score(results)
                if best_results is None or score > best_score:
                    best_params = dict(params)
                    best_results = results
                    best_score = score
                pbar.set_postfix({
                    "best": f"{best_score:.4f}",
                    "cur": f"{score:.4f}",
                })
                pbar.update(1)

        return best_params, best_results, best_score

    def search_nonhomophily_sgfcf_hyperparameters(search_model, search_dataset):
        search_batches, search_user_count = prepare_eval_batches(search_dataset)
        sgfcf_component_cache = {}
        best_params = None
        best_results = None
        best_score = -1.0

        search_space = list(nonhomophily_sgfcf_beta_space)
        with tqdm(total=len(search_space), desc="nonhomophily_SGFCF Search", unit="cfg") as pbar:
            for beta in search_space:
                params = dict(default_nonhomophily_sgfcf_params)
                params["beta"] = beta
                results = evaluate_nonhomophily_sgfcf_model(
                    search_model,
                    search_dataset,
                    params,
                    sgfcf_component_cache,
                    search_batches,
                    search_user_count,
                )
                score = tuning_score(results)
                if best_results is None or score > best_score:
                    best_params = dict(params)
                    best_results = results
                    best_score = score
                pbar.set_postfix({
                    "best": f"{best_score:.4f}",
                    "cur": f"{score:.4f}",
                })
                pbar.update(1)

        return best_params, best_results, best_score

    def search_bpr_renewal_hyperparameters(search_model, search_dataset, k):
        factor_variants = build_fixed_asym_bpr_renewal_variants(
            search_model,
            k,
        )
        best_params_by_variant = {}
        best_results_by_variant = {}
        best_scores_by_variant = {}
        for variant_name in ("plain", "lambda", "lambda2"):
            user_factors, item_factors = factor_variants[variant_name]
            fprint(f"Searching BPR-renewal coefficients for {bpr_renewal_variant_labels[variant_name]}.")
            best_params, best_results, best_score = search_bpr_renewal_coefficients(
                search_model,
                search_dataset,
                user_factors,
                item_factors,
            )
            best_params_by_variant[variant_name] = best_params
            best_results_by_variant[variant_name] = best_results
            best_scores_by_variant[variant_name] = best_score
        return best_params_by_variant, best_results_by_variant, best_scores_by_variant

    def run_final_tests(test_dataset, individual_params, sgfcf_params, quantized_sgfcf_params, nonhomophily_sgfcf_params):
        test_model = model.PSEmodel(world.config, test_dataset)
        homo_user, homo_item = compute_homophily_ratios(test_model)
        component_cache = {}
        sgfcf_component_cache = {}
        individual_results = evaluate_individual_filter_model(
            test_model,
            test_dataset,
            individual_params,
            homo_user,
            homo_item,
            component_cache,
        )
        sgfcf_results = evaluate_sgfcf_model(
            test_model,
            test_dataset,
            sgfcf_params,
            homo_user,
            homo_item,
            sgfcf_component_cache,
        )
        quantized_sgfcf_results = evaluate_quantized_sgfcf_model(
            test_model,
            test_dataset,
            quantized_sgfcf_params,
            homo_user,
            homo_item,
            sgfcf_component_cache,
        )
        nonhomophily_sgfcf_results = evaluate_nonhomophily_sgfcf_model(
            test_model,
            test_dataset,
            nonhomophily_sgfcf_params,
            sgfcf_component_cache,
        )
        return individual_results, sgfcf_results, quantized_sgfcf_results, nonhomophily_sgfcf_results, test_model

    def run_baseline_tests(test_model, test_dataset, k, bpr_renewal_params_by_variant):
        bpr_renewal_params_by_variant = normalize_bpr_renewal_params_by_variant(
            bpr_renewal_params_by_variant
        )
        baseline_alpha = fixed_asym_alpha
        baseline_beta = fixed_asym_beta
        baseline_tau, baseline_delta = alpha_beta_to_tau_delta(baseline_alpha, baseline_beta)
        recovered_alpha, recovered_beta = tau_delta_to_alpha_beta(baseline_tau, baseline_delta)
        if not np.isclose(recovered_alpha, baseline_alpha) or not np.isclose(recovered_beta, baseline_beta):
            raise ValueError("alpha/beta and tau/delta conversion failed.")

        fprint(
            "Baseline normalization: "
            f"alpha={baseline_alpha}, beta={baseline_beta}, "
            f"tau=alpha+beta={baseline_tau}, "
            f"delta=(beta-alpha)/2={baseline_delta}"
        )

        user_U, item_U, eigs, lambda2_user_factors, lambda2_item_factors = build_fixed_asym_lambda2_factors(
            test_model,
            k,
        )
        bpr_factor_variants = {
            "plain": (user_U, item_U),
            "lambda": (user_U * eigs.unsqueeze(0), item_U),
            "lambda2": (lambda2_user_factors, lambda2_item_factors),
        }

        baseline_plain = test_factorized_rating(test_model, test_dataset, user_U, item_U)
        print_metrics("Baseline Asymmetric Eigenvector: U_R U_R^T", baseline_plain)

        baseline_lambda = test_factorized_rating(
            test_model,
            test_dataset,
            user_U * eigs.unsqueeze(0),
            item_U,
        )
        print_metrics("Baseline Asymmetric Eigenvector: U_R Lambda U_R^T", baseline_lambda)

        baseline_lambda2 = test_factorized_rating(
            test_model,
            test_dataset,
            lambda2_user_factors,
            lambda2_item_factors,
        )
        print_metrics("Baseline Asymmetric Eigenvector: U_R Lambda^2 U_R^T", baseline_lambda2)

        for variant_name in ("plain", "lambda", "lambda2"):
            bpr_user_factors, bpr_item_factors = bpr_factor_variants[variant_name]
            bpr_renewal_params = bpr_renewal_params_by_variant.get(
                variant_name,
                default_bpr_renewal_params,
            )
            bpr_renewal_results = evaluate_bpr_renewal_params(
                test_model,
                test_dataset,
                bpr_user_factors,
                bpr_item_factors,
                bpr_renewal_params,
                desc=f"BPR Renewal Final Eval ({variant_name})",
            )
            fprint(f"BPR-renewal coefficients for {bpr_renewal_variant_labels[variant_name]}: {bpr_renewal_params}")
            print_metrics(
                f"{bpr_renewal_variant_labels[variant_name]} + One-Step BPR Renewal",
                bpr_renewal_results,
            )

        R = test_model.R.astype(np.float32).tocsr()
        user_degree = np.array(R.sum(axis=1)).flatten()
        item_degree = np.array(R.sum(axis=0)).flatten()
        user_degree[user_degree == 0] = 1e-9
        item_degree[item_degree == 0] = 1e-9
        R_tilde = (
            sp.diags(np.power(user_degree, -baseline_alpha))
            .dot(R)
            .dot(sp.diags(np.power(item_degree, -baseline_beta)))
            .tocsr()
        )
        R_tilde_t = R_tilde.T.tocsr()

        def r_tilde_cubic_batch(batch_users):
            return R_tilde[batch_users].dot(R_tilde_t).dot(R_tilde).toarray()

        baseline_r_tilde = test_sparse_rating_batches(test_model, test_dataset, r_tilde_cubic_batch)
        print_metrics("Baseline R_tilde R_tilde^T R_tilde", baseline_r_tilde)

    # ---------------------------------------------------------
    # 2. Hyperparameter search
    # ---------------------------------------------------------
    world.cprint(f"Dataset root: {data_root}")
    world.cprint(f"Selected dataset: {dataset_name}")
    world.cprint(f"val_true: {val_true}")

    with open("main_out.txt", "w") as out_file:
        fprint(f"Seed: {world.seed}")
        fprint(f"Dataset path: {dataset_path}")
        fprint(f"val_true: {val_true}")
        fprint(f"Default individual-filter params: {default_params}")
        fprint(f"Default SGFCF params: {default_sgfcf_params}")
        fprint(f"Default quantized_SGFCF params: {default_quantized_sgfcf_params}")
        fprint(f"quantized_SGFCF beta set: {beta_quantization_values.detach().cpu().tolist()}")
        fprint(f"Default nonhomophily_SGFCF params: {default_nonhomophily_sgfcf_params}")
        fprint(f"Default BPR-renewal params: {default_bpr_renewal_params}")

        best_sgfcf_params = dict(default_sgfcf_params)
        best_quantized_sgfcf_params = dict(default_quantized_sgfcf_params)
        if val_true == 0:
            best_params = dict(default_params)
            best_nonhomophily_sgfcf_params = dict(default_nonhomophily_sgfcf_params)
            best_bpr_renewal_params = default_bpr_renewal_params_by_variant()
            best_search_results = None
            best_search_score = None
            best_nonhomophily_sgfcf_search_results = None
            best_nonhomophily_sgfcf_search_score = None
            best_bpr_renewal_search_results = {}
            best_bpr_renewal_search_score = {}
            fprint("Skipping hyperparameter search; using default params.")
        else:
            search_eval_file = "val.txt" if val_true == 1 else "test.txt"
            search_name = "validation" if val_true == 1 else "test"
            search_dataset = load_dataset(search_eval_file)
            search_model = model.PSEmodel(world.config, search_dataset)
            fprint(f"Starting individual-filter hyperparameter search on {search_name} dataset ({search_eval_file}).")
            search_start = time.time()
            best_params, best_search_results, best_search_score = search_hyperparameters(search_model, search_dataset)
            search_runtime = time.time() - search_start
            fprint(f"Individual-filter search complete in {search_runtime:.2f} seconds.")
            fprint(f"Best individual-filter search params: {best_params}")
            fprint(f"Best individual-filter search score: {best_search_score:.6f}")
            print_metrics("Best individual-filter search metrics", best_search_results)

            fprint()
            fprint(
                "Starting nonhomophily_SGFCF hyperparameter search "
                f"on {search_name} dataset ({search_eval_file})."
            )
            nonhomophily_sgfcf_search_start = time.time()
            best_nonhomophily_sgfcf_params, best_nonhomophily_sgfcf_search_results, best_nonhomophily_sgfcf_search_score = search_nonhomophily_sgfcf_hyperparameters(
                search_model,
                search_dataset,
            )
            nonhomophily_sgfcf_search_runtime = time.time() - nonhomophily_sgfcf_search_start
            fprint(f"nonhomophily_SGFCF search complete in {nonhomophily_sgfcf_search_runtime:.2f} seconds.")
            fprint(f"Best nonhomophily_SGFCF search params: {best_nonhomophily_sgfcf_params}")
            fprint(f"Best nonhomophily_SGFCF search score: {best_nonhomophily_sgfcf_search_score:.6f}")
            print_metrics("Best nonhomophily_SGFCF search metrics", best_nonhomophily_sgfcf_search_results)

            fprint()
            fprint(
                "Starting BPR-renewal coefficient search "
                f"on {search_name} dataset ({search_eval_file})."
            )
            bpr_renewal_search_start = time.time()
            best_bpr_renewal_params, best_bpr_renewal_search_results, best_bpr_renewal_search_score = search_bpr_renewal_hyperparameters(
                search_model,
                search_dataset,
                best_params["k"],
            )
            bpr_renewal_search_runtime = time.time() - bpr_renewal_search_start
            fprint(f"BPR-renewal coefficient search complete in {bpr_renewal_search_runtime:.2f} seconds.")
            for variant_name in ("plain", "lambda", "lambda2"):
                fprint(f"Best BPR-renewal search params for {bpr_renewal_variant_labels[variant_name]}: {best_bpr_renewal_params[variant_name]}")
                fprint(f"Best BPR-renewal search score for {bpr_renewal_variant_labels[variant_name]}: {best_bpr_renewal_search_score[variant_name]:.6f}")
                print_metrics(
                    f"Best BPR-renewal search metrics for {bpr_renewal_variant_labels[variant_name]}",
                    best_bpr_renewal_search_results[variant_name],
                )

        # ---------------------------------------------------------
        # 3. Final test
        # ---------------------------------------------------------
        test_dataset = load_dataset("test.txt")
        fprint()
        fprint("Final Test")
        final_test_results, final_sgfcf_test_results, final_quantized_sgfcf_test_results, final_nonhomophily_sgfcf_test_results, final_test_model = run_final_tests(
            test_dataset,
            best_params,
            best_sgfcf_params,
            best_quantized_sgfcf_params,
            best_nonhomophily_sgfcf_params,
        )
        fprint(f"Final individual-filter params: {best_params}")
        print_metrics("Right-Eigenvector Individual Filter Test", final_test_results)
        fprint()
        fprint(f"Final SGFCF params: {best_sgfcf_params}")
        print_metrics("SGFCF Test", final_sgfcf_test_results)
        fprint()
        fprint(f"Final quantized_SGFCF params: {best_quantized_sgfcf_params}")
        fprint("quantized_SGFCF score: homophily-derived beta rounded to nearest value in B")
        print_metrics("quantized_SGFCF Test", final_quantized_sgfcf_test_results)
        fprint()
        fprint(f"Final nonhomophily_SGFCF params: {best_nonhomophily_sgfcf_params}")
        fprint("nonhomophily_SGFCF score: global beta filter without homophily ratios")
        print_metrics("nonhomophily_SGFCF Test", final_nonhomophily_sgfcf_test_results)

        # ---------------------------------------------------------
        # 4. Baseline asymmetric eigenvector tests
        # ---------------------------------------------------------
        fprint()
        fprint("Baseline Eigenvector and R_tilde Model Tests")
        run_baseline_tests(
            final_test_model,
            test_dataset,
            best_params["k"],
            best_bpr_renewal_params,
        )

finally:
    if world.tensorboard:
        w.close()
