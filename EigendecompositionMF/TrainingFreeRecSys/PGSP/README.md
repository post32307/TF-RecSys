# PGSP
Personalized Graph Signal Processing for Collaborative Filtering  
Jiahao Liu, Dongsheng Li, Hansu Gu, Tun Lu, Peng Zhang, Li Shang and Ning Gu  
The Web Conference (WWW). 2023  



## Runnable Python Implementation

`PGSP.py` is a command-line version of the original `PGSP.ipynb` notebook. It keeps the notebook computation structure:

1. Load the implicit train/test split from `../../data/<dataset>/`.
2. Build the user-item matrix and graph propagation matrices.
3. Compute the smallest eigenvectors of the normalized graph Laplacian with `scipy.sparse.linalg.eigsh`.
4. Compute the notebook's `P0` and `P1` score components.
5. Combine them as `P = phi * P0 + (1 - phi) * P1`.
6. Mask training interactions and report `Recall@20` and `NDCG@20`.

The script also caches expensive intermediate arrays under `PGSP/temp/` by default. The cache key includes the dataset, train-file fingerprint, dtype, and eigenvector dimension, so validation-search file swaps in `run_sweep.py` do not reuse stale matrices from the original train split.

## Direct Usage

Run from `TrainingFreeRecSys` after activating the project environment:

```bash
conda activate LGN
python PGSP/PGSP.py --dataset amazon-electronics --k 32 --phi 0.5 --seed 2020
```

Useful options:

```bash
--dataset DATASET    Dataset directory under ../../data.
--k K                Number of eigenvectors, called DIM in the notebook.
--phi PHI            Mixture weight for P0 versus P1.
--top_k K            Evaluation cutoff. Default: 20.
--seed SEED          Random seed passed to NumPy and PyTorch.
--dtype float32      Notebook-compatible default dtype.
--no_cache           Recompute eig/P0/P1 intermediates instead of using PGSP/temp/.
--cache_dir PATH     Store/read cache files in a custom directory.
--no_p0_norm         Use the notebook's raw P0 branch.
--no_p1_norm         Use the notebook's raw P1 branch.
```

## Running Through `run_sweep.py`

`run_sweep.py` now includes PGSP as a normal model. Verification mode runs the configured grid in dataset-size order and stops once valid metrics are produced:

```bash
conda activate LGN
python run_sweep.py --models PGSP --datasets amazon-electronics --seeds 2020 \
  --gpus 0 --verify_max_combinations 1 --timeout_per_model 300
```

Hyperparameter search is also supported. The sweep uses validation splits from `../../data/00_search_sets_seed_<seed>/<dataset>/`, writes validation results to `00_val_results_seed_<seed>.txt`, restores the original train/test files, and then writes final test results to `00_test_results_seed_<seed>.txt`:

```bash
conda activate LGN
python run_sweep.py --models PGSP --datasets amazon-electronics --seeds 2020 \
  --gpus 0 --search --max_combinations 1 --timeout_per_model 300
```

PGSP's sweep grid is currently:

```text
k:   32, 64, 128, 512, 1024
phi: 0.0, 0.25, 0.5, 0.75, 1.0
```

## Verification Notes

The converted script was checked against a direct execution of the notebook logic on `amazon-electronics` with `k=32`, `phi=0.5`, and seed `2020`. Both produced:

```text
Recall@20: 0.1387
NDCG@20:   0.1035
```

The script was also verified through `run_sweep.py` in both normal verification mode and search mode on `amazon-electronics`.
