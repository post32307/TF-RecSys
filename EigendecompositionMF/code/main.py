import world
import torch
torch.set_num_threads(4)
import os
import sys
import time
from tensorboardX import SummaryWriter
from os.path import join
# ==============================
import helpers
helpers.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================

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
    plot_true = 0
    #   0: skip eigenvalue plotting and continue to the dataset-root setup
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
        import temp_plot
        temp_plot.plot_eigenvalues()
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

    world.cprint(f"Dataset root: {data_root}")
    world.cprint(f"Selected dataset: {dataset_name}")
    world.cprint(f"Dataset path: {dataset_path}")
    world.cprint(f"val_true: {val_true}")
    world.cprint(f"max_k: {max_k}")



    def Energy(A, k):
        """Return the fractional rank-k energy of matrix A."""
        if k <= 0:
            return 0.0
        if hasattr(A, "multiply"):
            from scipy.sparse.linalg import svds
            total_energy = float(A.multiply(A).sum())
            if total_energy == 0:
                return 0.0
            k = min(int(k), min(A.shape) - 1)
            if k <= 0:
                return 0.0
            singular_values = svds(A.astype("float32"), k=k, return_singular_vectors=False)
            top_k_energy = float((singular_values ** 2).sum())
            return top_k_energy / total_energy
        if hasattr(A, "toarray"):
            A = A.toarray()
        A = torch.as_tensor(A, dtype=torch.float32, device=world.device)
        singular_values = torch.linalg.svdvals(A)
        total_energy = torch.sum(singular_values ** 2)
        if total_energy == 0:
            return 0.0
        k = min(int(k), singular_values.numel())
        top_k_energy = torch.sum(singular_values[:k] ** 2)
        return float((top_k_energy / total_energy).detach().cpu())

    def PlotEnergy(A, max_k, output_file=None):
        if max_k < 32:
            raise ValueError("max_k must be at least 32.")
        k_values = list(range(32, int(max_k) + 1, 32))
        energy_values = [Energy(A, k) for k in k_values]
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if output_file is None:
            plot_dir = os.path.join(os.path.dirname(__file__), "plots")
            os.makedirs(plot_dir, exist_ok=True)
            output_file = os.path.join(
                plot_dir,
                f"{dataset_name}_energy_seed_{world.seed}_k_{int(max_k)}.png",
            )
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(k_values, energy_values, marker="o", linewidth=2)
        ax.set_xlabel("k")
        ax.set_ylabel("Fractional energy")
        ax.set_title(f"{dataset_name} fractional energy")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.02)
        fig.tight_layout()
        fig.savefig(output_file, dpi=200)
        plt.close(fig)
        return k_values, energy_values, output_file

    print()
    import dataloader
    import model
    dataset = dataloader.Loader(path=dataset_path)
    Recmodel = model.PSEmodel(world.config, dataset)
    PlotEnergy(Recmodel.A, Recmodel.k, output_file="Plot_A")
    PlotEnergy(Recmodel.getNormAdj(Recmodel.A,Recmodel.D_inv,alpha1=0.5,alpha2=0.5), Recmodel.k, output_file="Plot_A_tilde")
    PlotEnergy(Recmodel.getNormAdj(Recmodel.A,Recmodel.D_inv,alpha1=0.2,alpha2=0.4), Recmodel.k, output_file="Plot_A_norm")
    

finally:
    if world.tensorboard:
        w.close()
