import os

# Limit CPU threads to at most 8 for openmp, mkl, openblas, etc.
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'
os.environ['OPENBLAS_NUM_THREADS'] = '8'
os.environ['VECLIB_MAXIMUM_THREADS'] = '8'
os.environ['NUMEXPR_NUM_THREADS'] = '8'

import subprocess
import sys
import time
import argparse
import re
import itertools
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import signal

def sig_handler(signum, frame):
    print(f"\n[!] Process terminated with signal {signum}")
    raise KeyboardInterrupt

signal.signal(signal.SIGINT, sig_handler)
signal.signal(signal.SIGTERM, sig_handler)

# Base directory of this script
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Datasets directory (relative to repository root)
current_dir = Path(__file__).resolve().parent
DATA_ROOT = current_dir.parent / "data"

# Datasets list dynamically retrieved from the data folder
if DATA_ROOT.exists() and DATA_ROOT.is_dir():
    DATASETS = sorted([
        d for d in os.listdir(DATA_ROOT)
        if os.path.isdir(os.path.join(DATA_ROOT, d)) and not d.startswith('.') and not d.endswith('_val') and not d.startswith("00")
    ])
else:
    DATASETS = []

print("All Datasets:", DATASETS)

# Model configurations with relative paths
MODELS = {
    'BSPM': {
        'cwd': os.path.join(BASE_DIR, 'BSPM', 'bspm'),
        'cmd': [sys.executable, 'main.py', '--simple_model', 'bspm']
    },
    'EASE': {
        'cwd': os.path.join(BASE_DIR, 'EASE'),
        'cmd': [sys.executable, 'run.py']
    },
    'GF-CF': {
        'cwd': os.path.join(BASE_DIR, 'GF_CF', 'Table3&4'),
        'cmd': [sys.executable, 'main.py', '--simple_model', 'gf-cf']
    },
    'PGSP': {
        
    },
    'TurboCF': {
        'cwd': os.path.join(BASE_DIR, 'Turbo_CF'),
        'cmd': [sys.executable, 'main.py']
    },
    'FaGSP': {
        'cwd': os.path.join(BASE_DIR, 'FaGSP'),
        'cmd': [sys.executable, 'main.py']
    },
    'HiGSP': {
        'cwd': os.path.join(BASE_DIR, 'HiGSP'),
        'cmd': [sys.executable, 'main.py']
    },
    'SGFCF': {
        'cwd': os.path.join(BASE_DIR, 'SGFCF'),
        'cmd': [sys.executable, 'sgfcf.py']
    }
}

# Example parameter grids for search
PARAM_GRIDS = {
    'BSPM': {
        'idl_beta': [0.1, 0.3, 0.5, 0.7, 0.9],
        'factor_dim': [32, 64, 128, 256, 512, 1024],
    },
    'EASE': {
        'lambda_': list(range(50, 2050, 50)),
    },
    'GF-CF': {
        'k': list(range(32, 1025, 30)),
    },
    'TurboCF': {
        'alpha': [0.1, 0.3, 0.5, 0.7, 0.9],
        'power': [0.5, 1.0, 1.5, 2.0],
        'filter': [1, 2, 3],
    },
    'FaGSP': {
        'alpha1': [0.3],
        'alpha2': [0.5],
        'q': [0.7],
        'order1': [4, 8, 12, 16],
        'k1': [32, 128, 512, 1024],
        'k2': [32, 128, 512, 1024],
    },
    'HiGSP': {
        'alpha1': [0.08],
        'alpha2': [0.5, 0.73, 0.9],
        'order1': [2],
        'order2': [4, 8, 12, 16],
        'pri_factor': [32, 128, 512, 1024],
    },
    'SGFCF': {
        'k': [32, 128, 512, 1024],
        'gamma': [1.0],
        'beta_1': [0.5, 1.0, 1.5, 2.0],
        'eps': [0.2, 0.4, 0.6, 0.8],
    }
}

def get_param_grid(model_name, dataset):
    grid = PARAM_GRIDS.get(model_name, {})
    # If the grid defines a 'default' key, we assume it's structured for dataset-specific overrides
    if 'default' in grid:
        return grid.get(dataset, grid.get('default', {}))
    return grid

def parse_metrics(log_file, model_name):
    if not os.path.exists(log_file):
        return "N/A"
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    metrics = {}
    
    if model_name in ['BSPM', 'GF-CF']:
        matches = list(re.finditer(r"'recall': array\(\[(.*?)\]\),.*?'ndcg': array\(\[(.*?)\]\)", content, re.DOTALL))
        if matches:
            match = matches[-1]
            recall = match.group(1).split(',')[0].strip()
            ndcg = match.group(2).split(',')[0].strip()
            metrics['Recall@20'] = recall
            metrics['NDCG@20'] = ndcg
            
    elif model_name == 'FaGSP':
        match_recall = list(re.finditer(r"Recall@20:(\d+\.\d+)", content))
        if match_recall:
            metrics['Recall@20'] = match_recall[-1].group(1)
        match_ndcg = list(re.finditer(r"NDCG@20:(\d+\.\d+)", content))
        if match_ndcg:
            metrics['NDCG@20'] = match_ndcg[-1].group(1)

    elif model_name == 'HiGSP':
        match_recall = list(re.finditer(r"Recall@20:(\d+\.\d+)", content))
        if match_recall:
            metrics['Recall@20'] = match_recall[-1].group(1)
        match_ndcg = list(re.finditer(r"NDCG@20:(\d+\.\d+)", content))
        if match_ndcg:
            metrics['NDCG@20'] = match_ndcg[-1].group(1)

    elif model_name == 'TurboCF':
        m_recall = list(re.finditer(r"Recall@20: (\d+\.\d+)", content))
        m_ndcg = list(re.finditer(r"NDCG@20: (\d+\.\d+)", content))
        if m_recall: metrics['Recall@20'] = m_recall[-1].group(1)
        if m_ndcg: metrics['NDCG@20'] = m_ndcg[-1].group(1)

    elif model_name == 'EASE':
        m_recall = list(re.finditer(r"recall@\[20\]:\s*\[([\d\.]+)\]", content))
        m_ndcg = list(re.finditer(r"ndcg@\[20\]:\s*\[([\d\.]+)\]", content))
        if m_recall: metrics['Recall@20'] = m_recall[-1].group(1)
        if m_ndcg: metrics['NDCG@20'] = m_ndcg[-1].group(1)

    elif model_name == 'SGFCF':
        lines = content.strip().split('\n')
        for line in reversed(lines):
            parts = line.strip().split()
            if len(parts) == 4:
                try:
                    vals = [float(p) for p in parts]
                    metrics['Recall@20'] = str(vals[3])
                    metrics['NDCG@20'] = str(vals[1])
                    break
                except ValueError:
                    continue

    if not metrics:
        return "N/A"
    
    return ", ".join([f"{k}: {v}" for k, v in metrics.items()])

def get_recall_value(metrics_str):
    if not metrics_str or metrics_str == "N/A" or metrics_str == "ERROR":
        return -1.0
    match = re.search(r"Recall@20:\s*([+-]?\d*\.?\d+(?:[eE][-+]?\d+)?)", metrics_str)
    if match:
        return float(match.group(1))
    return -1.0

def run_experiment(model_name, dataset, log_dir, gpu=None, params=None, seed=None):
    if model_name not in MODELS:
        print(f"Unknown model: {model_name}")
        return None

    model_info = MODELS[model_name]
    if 'cmd' not in model_info:
        print(f"Skipping {model_name}: No 'cmd' defined.")
        return "N/A"
    
    cmd = list(model_info['cmd']) + ['--dataset', dataset]
    
    if seed is not None:
        cmd += ['--seed', str(seed)]
        
    if params:
        for k, v in params.items():
            cmd += [f'--{k}', str(v)]
    
    if gpu is not None and model_name in ['BSPM']:
        cmd += ['--gpu', '0']
    
    env = os.environ.copy()
    if gpu is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    env['OMP_NUM_THREADS'] = '8'
    env['MKL_NUM_THREADS'] = '8'
    env['OPENBLAS_NUM_THREADS'] = '8'
    env['VECLIB_MAXIMUM_THREADS'] = '8'
    env['NUMEXPR_NUM_THREADS'] = '8'
    
    param_suffix = "_" + "_".join([f"{k}{v}" for k, v in params.items()]) if params else ""
    log_file = os.path.join(log_dir, f"{model_name}_{dataset}{param_suffix}.log")

    print(f"  -> Running {model_name} on {dataset} on GPU {gpu} with {params if params else 'defaults'}... (Logging to {log_file})")
    
    with open(log_file, 'a') as f:
        start_time = time.time()
        try:
            subprocess.run(cmd, cwd=model_info['cwd'], stdout=f, stderr=subprocess.STDOUT, check=True, env=env)
            duration = time.time() - start_time
            print(f"  -> Finished {model_name} on GPU {gpu} in {duration:.2f}s")
            return parse_metrics(log_file, model_name)
        except subprocess.CalledProcessError as e:
            print(f"  !! Error running {model_name} on GPU {gpu}. Check {log_file}")
            return "ERROR"

def main():
    parser = argparse.ArgumentParser(description="Run multiple recommendation models with parameter search.")
    
    parser.add_argument("model", nargs="?", help="Specific model to run.")
    parser.add_argument("--models", nargs="+", help="Models to run.")
    parser.add_argument("--datasets", nargs="+", help="Datasets to run on.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use.")
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2, 3], help="GPU IDs to use for round-robin parallel grid search.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[2020], help="List of seeds to run.")
    parser.add_argument("--search", action="store_true", help="Perform parameter search using a validation split, then test.")
    parser.add_argument("--test_from_results", action="store_true", help="Parse the results text file to find the best validation parameters and run the final test")
    
    args = parser.parse_args()
    
    # 1. Validation: check if all seeds are in the data folder
    missing_seeds = []
    for seed in args.seeds:
        seed_dir = os.path.join(DATA_ROOT, f"00_search_sets_seed_{seed}")
        if not os.path.exists(seed_dir):
            missing_seeds.append(seed)
            
    if missing_seeds:
        print(f"Error: The following seeds are not found in the data folder: {missing_seeds}")
        sys.exit(1)
        
    if args.models:
        models_to_run = args.models
    elif args.model:
        models_to_run = [args.model]
    else:
        models_to_run = list(MODELS.keys())
        
    datasets_to_run = args.datasets if args.datasets else DATASETS
    
    print(f"Starting experiments at {time.ctime()}")
    print(f"Search mode: {args.search}")
    print(f"Seeds: {args.seeds}")
    print("-" * 50)
    
    # Loop over all seeds sequentially
    if args.test_from_results:
        print(f"=== [TEST FROM RESULTS MODE] Parsing results file for best parameters ===")
        import ast
        
        for seed in args.seeds:
            results_file = os.path.join(BASE_DIR, f"00_val_results_seed_{seed}.txt")
            log_dir = os.path.join(BASE_DIR, f"00_exp_logs_seed_{seed}")
            test_results_file = os.path.join(BASE_DIR, f"00_test_results_seed_{seed}.txt")
            if not os.path.exists(results_file):
                print(f"No results file found for seed {seed}")
                continue
                
            best_params_parsed = {m: {d: {'best_recall': -1.0, 'best_params': None} for d in args.datasets} for m in models_to_run}
            with open(results_file, 'r') as f:
                for line in f:
                    if "(Val)" not in line or "ERROR" in line:
                        continue
                    parts = [p.strip() for p in line.split('|')]
                    if len(parts) >= 4:
                        m_name = parts[0]
                        d_name = parts[1]
                        param_str = parts[2]
                        metrics_str = parts[3].replace("(Val)", "").strip()
                        
                        if m_name in best_params_parsed and d_name in best_params_parsed[m_name]:
                            recall = get_recall_value(metrics_str)
                            if recall > best_params_parsed[m_name][d_name]['best_recall']:
                                best_params_parsed[m_name][d_name]['best_recall'] = recall
                                
                                params = {}
                                if param_str and param_str != "default":
                                    for pair in param_str.split(','):
                                        if '=' in pair:
                                            k, v = pair.split('=')
                                            k = k.strip()
                                            v = v.strip()
                                            try:
                                                v = ast.literal_eval(v)
                                            except:
                                                pass
                                            params[k] = v
                                best_params_parsed[m_name][d_name]['best_params'] = params
            
            for dataset in args.datasets:
                print(f"\n--- Testing on {dataset} (Seed {seed}) ---")
                for model_name in models_to_run:
                    best_params = best_params_parsed[model_name][dataset]['best_params']
                    if best_params is not None:
                        print(f"  -> Best validation parameters for {model_name}: {best_params} (Recall: {best_params_parsed[model_name][dataset]['best_recall']})")
                        metrics_str = run_experiment(model_name, dataset, log_dir, args.gpus[0], best_params, seed)
                        param_str = ", ".join([f"{k}={v}" for k, v in best_params.items()])
                        result_line = f"{model_name:<10} | {dataset:<20} | {param_str:<30} | {metrics_str} (Test)"
                        with open(test_results_file, 'a') as f:
                            f.write(result_line + "\n")
                    else:
                        print(f"  -> No valid validation results found for {model_name} on {dataset}.")
        print(f"Test complete. Results appended to results file.")
        sys.exit(0)

    for seed in args.seeds:
        all_best_params = {m: {} for m in models_to_run}
        all_test_results = {m: {} for m in models_to_run}
        
        print(f"\n==================================================")
        print(f"=== Processing Seed: {seed} ===")
        print(f"==================================================\n")
        
        log_dir = os.path.join(BASE_DIR, f"00_exp_logs_seed_{seed}")
        results_file = os.path.join(BASE_DIR, f"00_val_results_seed_{seed}.txt")
        test_results_file = os.path.join(BASE_DIR, f"00_test_results_seed_{seed}.txt")
        os.makedirs(log_dir, exist_ok=True)
        
        with open(results_file, 'a') as f:
            f.write(f"\nExperiment Results Summary - {time.ctime()} (Search Mode: {args.search}, Seed: {seed})\n")
            f.write("=" * 80 + "\n")
            f.write(f"{'Model':<10} | {'Dataset':<20} | {'Parameters':<30} | Metrics\n")
            f.write("-" * 80 + "\n")
            
        with open(test_results_file, 'a') as f:
            f.write(f"\nFinal Test Results Summary - {time.ctime()} (Seed: {seed})\n")
            f.write("=" * 80 + "\n")
            f.write(f"{'Model':<10} | {'Dataset':<20} | {'Parameters':<30} | Metrics\n")
            f.write("-" * 80 + "\n")
        
        try:
            for dataset in datasets_to_run:
                swapped = False
                current_search = args.search
            
                train_filename = "data1.txt" if dataset == "lastfm" else "train.txt"
                test_filename = "test1.txt" if dataset == "lastfm" else "test.txt"
            
                orig_train = os.path.join(DATA_ROOT, dataset, train_filename)
                orig_test = os.path.join(DATA_ROOT, dataset, test_filename)
                bak_train = orig_train + ".bak"
                bak_test = orig_test + ".bak"
            
                search_train = os.path.join(DATA_ROOT, f"00_search_sets_seed_{seed}", dataset, "train_val.txt")
                search_val = os.path.join(DATA_ROOT, f"00_search_sets_seed_{seed}", dataset, "validation_val.txt")
            
                best_params_dict = {}
            
                try:
                    if current_search:
                        if not os.path.exists(search_train) or not os.path.exists(search_val):
                            print(f"Error: Search sets for {dataset} not found in 00_search_sets_seed_{seed}. Skipping search for this dataset.")
                            current_search = False
                        else:
                            print(f"  -> [SEARCH MODE] Swapping train and test files with validation splits for {dataset}")
                            # Backup original train/test files
                            if os.path.exists(orig_train):
                                os.rename(orig_train, bak_train)
                            if os.path.exists(orig_test):
                                os.rename(orig_test, bak_test)
                            
                            # Copy validation files to original names
                            shutil.copy(search_train, orig_train)
                            shutil.copy(search_val, orig_test)
                        
                            # Backup any npz files
                            npz_files = [f for f in os.listdir(os.path.join(DATA_ROOT, dataset)) if f.endswith('.npz') and not f.endswith('.bak')]
                            for npz in npz_files:
                                orig_npz = os.path.join(DATA_ROOT, dataset, npz)
                                os.rename(orig_npz, orig_npz + ".bak")
                            
                            swapped = True

                    for model_name in models_to_run:
                        if model_name == "PGSP": continue

                        print(f"Model: {model_name} | Dataset: {dataset}")
                    
                        grid = get_param_grid(model_name, dataset)
                        if current_search and grid:
                            keys = list(grid.keys())
                            values = list(grid.values())
                            combinations = list(itertools.product(*values))
                        
                            best_recall = -1.0
                            best_params = None
                        
                            if combinations:
                                # 1. Run the first combination sequentially to pre-generate any shareable assets (like npz files)
                                first_combo = combinations[0]
                                first_params = dict(zip(keys, first_combo))
                                metrics_str = run_experiment(
                                    model_name, dataset, log_dir, args.gpus[0], first_params, seed
                                )
                                param_str = ", ".join([f"{k}={v}" for k, v in first_params.items()])
                                result_line = f"{model_name:<10} | {dataset:<20} | {param_str:<30} | {metrics_str} (Val)"
                                with open(results_file, 'a') as f:
                                    f.write(result_line + "\n")
                                best_recall = get_recall_value(metrics_str)
                                best_params = first_params
                            
                                # 2. Run the rest of the combinations in parallel across the GPUs
                                remaining_combos = combinations[1:]
                                if remaining_combos:
                                    import queue
                                    from concurrent.futures import ThreadPoolExecutor
                                
                                    gpu_queue = queue.Queue()
                                    for g in args.gpus:
                                        for _ in range(3):
                                            gpu_queue.put(g)
                                    
                                    results = []
                                
                                    def worker(combo):
                                        gpu = gpu_queue.get()
                                        try:
                                            params = dict(zip(keys, combo))
                                            metrics = run_experiment(
                                                model_name, dataset, log_dir, gpu, params, seed
                                            )
                                            return combo, params, metrics
                                        finally:
                                            gpu_queue.put(gpu)
                                            gpu_queue.task_done()
                                        
                                    with ThreadPoolExecutor(max_workers=len(args.gpus) * 3) as executor:
                                        futures = [executor.submit(worker, combo) for combo in remaining_combos]
                                        from concurrent.futures import as_completed
                                        for future in as_completed(futures):
                                            try:
                                                combo, params, metrics_str = future.result()
                                                param_str = ", ".join([f"{k}={v}" for k, v in params.items()])
                                                result_line = f"{model_name:<10} | {dataset:<20} | {param_str:<30} | {metrics_str} (Val)"
                                                with open(results_file, 'a') as f:
                                                    f.write(result_line + "\n")
                                                
                                                recall = get_recall_value(metrics_str)
                                                if recall > best_recall:
                                                    best_recall = recall
                                                    best_params = params
                                            except Exception as e:
                                                print(f"Exception in worker thread: {e}")
                                            
                            best_params_dict[model_name] = best_params
                            if best_params is not None:
                                all_best_params[model_name][dataset] = str(best_params)
                            else:
                                all_best_params[model_name][dataset] = "N/A"
                            print(f"  -> Best validation parameters for {model_name} on {dataset}: {best_params} (Recall@20: {best_recall:.4f})")
                    
                        elif not current_search:
                            # Test mode execution
                            metrics_str = run_experiment(model_name, dataset, log_dir, args.gpu, None, seed)
                            result_line = f"{model_name:<10} | {dataset:<20} | {'default':<30} | {metrics_str}"
                            with open(test_results_file, 'a') as f:
                                f.write(result_line + "\n")
                            all_test_results[model_name][dataset] = get_recall_value(metrics_str)
                            
                finally:
                    if swapped:
                        print(f"  -> [SEARCH MODE] Restoring original files for {dataset}")
                        # Remove validation train/test files
                        if os.path.exists(orig_train):
                            os.remove(orig_train)
                        if os.path.exists(orig_test):
                            os.remove(orig_test)
                        
                        # Restore original train/test files from backup
                        if os.path.exists(bak_train):
                            os.rename(bak_train, orig_train)
                        if os.path.exists(bak_test):
                            os.rename(bak_test, orig_test)
                        
                        # Remove generated npz files for validation set
                        for f in os.listdir(os.path.join(DATA_ROOT, dataset)):
                            if f.endswith('.npz') and not f.endswith('.npz.bak'):
                                try:
                                    os.remove(os.path.join(DATA_ROOT, dataset, f))
                                except Exception as e:
                                    print(f"Warning: Could not remove generated npz file {f}: {e}")
                                
                        # Restore original npz files
                        for f in os.listdir(os.path.join(DATA_ROOT, dataset)):
                            if f.endswith('.npz.bak'):
                                orig_npz = os.path.join(DATA_ROOT, dataset, f[:-4])
                                os.rename(os.path.join(DATA_ROOT, dataset, f), orig_npz)
                            
                # Now run final test on test set using the best parameters found
                if current_search:
                    print(f"=== [FINAL TEST MODE] Running models on {dataset} with best found parameters ===")
                    for model_name, best_params in best_params_dict.items():
                        if best_params is not None:
                            metrics_str = run_experiment(model_name, dataset, log_dir, args.gpu, best_params, seed)
                            param_str = ", ".join([f"{k}={v}" for k, v in best_params.items()])
                            result_line = f"{model_name:<10} | {dataset:<20} | {param_str:<30} | {metrics_str} (Test)"
                            with open(test_results_file, 'a') as f:
                                f.write(result_line + "\n")
                            all_test_results[model_name][dataset] = get_recall_value(metrics_str)
                        
                print("-" * 50)
                
        except (KeyboardInterrupt, SystemExit):
            print(f"\n[!] Execution interrupted. Partial results for seed {seed} are saved to {results_file}.")
            sys.exit(1)

        print(f"All experiments for seed {seed} finished. Search results saved to {results_file}, Final Test results saved to {test_results_file}")

if __name__ == "__main__":
    main()