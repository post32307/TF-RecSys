import argparse
import hashlib
import itertools
import json
import os
import tempfile
from pathlib import Path


def _convert_split_file(source, destination):
    """Convert one grouped user/item split to RecBole's interaction format."""
    with open(source, "r", encoding="utf-8") as src, open(
        destination, "w", encoding="utf-8"
    ) as dst:
        dst.write("user_id:token\titem_id:token\n")
        for line in src:
            fields = line.split()
            if not fields:
                continue
            user_id, *item_ids = fields
            for item_id in item_ids:
                dst.write(f"{user_id}\t{item_id}\n")


def _config_key(config_paths, run_options):
    """Fingerprint effective config sources and CLI overrides for result reuse."""
    digest = hashlib.sha256()
    for path in config_paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    digest.update(json.dumps(run_options, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()[:16]


def _append_result(results_dir, filename, dataset, seed, metrics, hyperparameters):
    results_dir.mkdir(parents=True, exist_ok=True)
    result_path = results_dir / filename
    result_text = json.dumps(metrics, sort_keys=True)
    params_text = json.dumps(hyperparameters, sort_keys=True)
    with result_path.open("a", encoding="utf-8") as output:
        output.write(f"{dataset} | {seed} | {result_text} | {params_text}\n")
    print(f"[RESULTS] appended to {result_path}")


def _read_search_cache(cache_path):
    cache = {}
    if cache_path.is_file():
        with cache_path.open("r", encoding="utf-8") as source:
            for line in source:
                try:
                    row = json.loads(line)
                    cache[row["config_key"]] = row["validation_results"]
                except (json.JSONDecodeError, KeyError):
                    continue
    return cache


def _append_search_cache(cache_path, config_key, hyperparameters, metrics):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8") as output:
        output.write(json.dumps({
            "config_key": config_key,
            "hyperparameters": hyperparameters,
            "validation_results": metrics,
        }, sort_keys=True) + "\n")


def _recall_at_20(metrics):
    for name, value in metrics.items():
        if name.lower() == "recall@20":
            return float(value)
    raise KeyError(f"Recall@20 missing from validation results: {metrics}")


def main(args):
    # Treat --gpu_ID as the physical CUDA device ID. RecBole then uses the
    # selected device as cuda:0 within this process. Use -1 to force CPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = "" if args.gpu_ID < 0 else str(args.gpu_ID)
    print(">>GPU_ID:", args.gpu_ID)
    from recbole.quick_start import run_recbole
    from recbole.utils import init_seed

    repo_dir = Path(__file__).resolve().parent
    if args.seed_folder is None:
        args.seed_folder = f"00_dataset_seed_{args.seed}"
    init_seed(args.seed, reproducibility=True)
    print(">>SEED:", args.seed)
    split_dir = Path(args.split_root) / args.seed_folder / args.dataset
    required = [split_dir / name for name in ("train.txt", "val.txt", "test.txt")]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing split file(s): " + ", ".join(missing))

    config_name = args.config or f"config_{args.dataset.lower()}.yaml"
    config_path = repo_dir / "configs" / config_name
    if not config_path.is_file():
        # JGCF model settings are shared across implicit-feedback datasets.
        config_path = repo_dir / "configs" / "config_gowalla.yaml"
    config_sources = [repo_dir / "recbole" / "properties" / "overall.yaml"]
    model_config_path = repo_dir / "recbole" / "properties" / "model" / f"{args.model}.yaml"
    if model_config_path.is_file():
        config_sources.append(model_config_path)
    config_sources.append(config_path)
    results_dir = repo_dir / "results"
    cache_path = results_dir / "search_trials.jsonl"
    cached_trials = _read_search_cache(cache_path)
    split_digest = hashlib.sha256()
    for split_file in required:
        split_digest.update(split_file.read_bytes())
    grid = itertools.product(
        [32, 64, 128], [2, 3], [1.0, 2.0], [1.0, 2.0], [1e-5, 1e-4]
    )
    best_params = None
    best_metrics = None
    best_score = None
    for embedding_size, n_layers, a, b, reg_weight in grid:
        hyperparameters = {
            "embedding_size": embedding_size,
            "n_layers": n_layers,
            "a": a,
            "b": b,
            "reg_weight": reg_weight,
        }
        trial_key = _config_key(config_sources, {
            "model": args.model,
            "dataset": args.dataset,
            "seed": args.seed,
            "search_epochs": 20,
            "eval_step": 10,
            "stopping_step": 20,
            "split_digest": split_digest.hexdigest(),
            "hyperparameters": hyperparameters,
        })
        metrics = cached_trials.get(trial_key)
        if metrics is None:
            print(f"[SEARCH] JGCF validation trial: {hyperparameters}")
            trial_config = {
                "data_path": str(split_dir.parent.parent),
                "benchmark_filename": ["train", "valid", "test"],
                "seed": args.seed,
                "gpu_id": "" if args.gpu_ID < 0 else "0",
                "epochs": 20,
                "eval_step": 10,
                "stopping_step": 20,
                **hyperparameters,
            }
            # Convert the grouped splits into RecBole format for this trial.
            with tempfile.TemporaryDirectory(prefix="jgcf-splits-") as temp_dir:
                dataset_dir = Path(temp_dir) / args.dataset
                dataset_dir.mkdir(parents=True)
                for source, suffix in zip(required, ("train", "valid", "test")):
                    _convert_split_file(source, dataset_dir / f"{args.dataset}.{suffix}.inter")
                trial_config["data_path"] = temp_dir
                trial = run_recbole(
                    model=args.model, dataset=args.dataset,
                    config_file_list=[str(config_path)], config_dict=trial_config,
                    evaluate_test=False,
                )
            metrics = trial["best_valid_result"]
            _append_search_cache(cache_path, trial_key, hyperparameters, metrics)
        else:
            print(f"[CACHE] using JGCF validation trial: {hyperparameters}")

        score = _recall_at_20(metrics)
        if best_score is None or score > best_score:
            best_params, best_metrics, best_score = hyperparameters, metrics, score

    print(f"[SEARCH] best Recall@20={best_score:.6f}, params={best_params}")
    _append_result(
        results_dir, "hyperparameter_search.txt", args.dataset, args.seed,
        best_metrics, best_params,
    )

    # Full training of the selected parameters uses the original epoch budget.
    with tempfile.TemporaryDirectory(prefix="jgcf-splits-") as temp_dir:
        dataset_dir = Path(temp_dir) / args.dataset
        dataset_dir.mkdir(parents=True)
        for source, suffix in zip(required, ("train", "valid", "test")):
            target = dataset_dir / f"{args.dataset}.{suffix}.inter"
            _convert_split_file(source, target)

        config_dict = {
            "data_path": temp_dir,
            "benchmark_filename": ["train", "valid", "test"],
            "seed": args.seed,
            "gpu_id": "" if args.gpu_ID < 0 else "0",
            **best_params,
            "eval_step": 10,
            "stopping_step": 20,
        }
        if args.epochs is not None:
            config_dict["epochs"] = args.epochs
        result = run_recbole(
            model=args.model,
            dataset=args.dataset,
            config_file_list=[str(config_path)],
            config_dict=config_dict,
        )
        print(f"[FINAL] test results={result['test_result']}")
        _append_result(
            results_dir, "test_results.txt", args.dataset, args.seed,
            result["test_result"], best_params,
        )
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   default="JGCF", help="choose the model to use. ")
    parser.add_argument("--dataset", default="gowalla", help="choose the dataset to use. ")
    parser.add_argument("--split_root", default=None,
                        help="Parent directory containing seed folders (default: ../data/split_dataset)")
    parser.add_argument("--seed_folder", default=None,
                        help="Split folder, for example 00_dataset_seed_2026")
    parser.add_argument("--seed", "--random_seed", dest="seed", type=int, default=2026,
                        help="Random seed used for model initialization")
    parser.add_argument("--gpu_ID", type=int, default=0,
                        help="Physical GPU ID to use; use -1 to force CPU")
    parser.add_argument("--config", default=None,
                        help="Optional config filename inside configs/")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override training epochs; useful for smoke checks")
    parser.add_argument("--eval_step", type=int, default=None,
                        help="Override validation interval")
    parser.add_argument("--stopping_step", type=int, default=None,
                        help="Override early stopping patience")
    args = parser.parse_args()
    if args.split_root is None:
        args.split_root = str(Path(__file__).resolve().parent.parent / "data" / "split_dataset")
    main(args)
