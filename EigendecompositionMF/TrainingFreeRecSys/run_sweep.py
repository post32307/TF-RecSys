
import os
import subprocess
import sys
import time
import argparse
import re
import random
import itertools
import shutil
from pathlib import Path

# Base directory of this script
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Datasets directory (relative to repository root)
current_dir = Path(__file__).resolve().parent
DATA_ROOT = current_dir.parent / "data"

# Datasets list dynamically retrieved from the data folder
if DATA_ROOT.exists() and DATA_ROOT.is_dir():
    DATASETS = sorted([
        d for d in os.listdir(DATA_ROOT)
        if os.path.isdir(os.path.join(DATA_ROOT, d)) and not d.startswith('.') and not d.endswith('_val') and d != "00 search_sets"
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
        'lr': [0.0001, 0.0005, 0.001, 0.005],               # default: 0.001
        'decay': [1e-5, 1e-4, 1e-3, 1e-2],                  # default: 1e-4
        'idl_beta': [0.1, 0.2, 0.3, 0.5, 0.7],               # default: 0.3
        'factor_dim': [128, 256, 512],                        # default: 256
    },
    'EASE': {
        'lambda_': [50.0, 100.0, 250.0, 500.0, 750.0, 1000.0, 2000.0],  # default: 500.0
    },
    'GF-CF': {
        'lr': [0.0001, 0.0005, 0.001, 0.005],                # default: 0.001
        'decay': [1e-5, 1e-4, 1e-3, 1e-2],                   # default: 1e-4
        'layer': [1, 2, 3, 4, 5],                             # default: 3
        'recdim': [32, 64, 128, 256],                         # default: 64
    },
    'PGSP': {

    },
    'TurboCF': {
        'alpha': [0.3, 0.4, 0.5, 0.6, 0.7],                 # default: 0.5
        'power': [0.5, 0.75, 1.0, 1.5, 2.0],                 # default: 1.0
        'filter': [1, 2, 3],                                  # default: 1
    },
    'FaGSP': {
        'alpha1': [0.1, 0.2, 0.3, 0.4, 0.5],                # default: 0.3
        'alpha2': [0.3, 0.4, 0.5, 0.6, 0.7],                 # default: 0.5
        'q': [0.5, 0.6, 0.7, 0.8, 0.9],                     # default: 0.7
        'order1': [6, 8, 12, 16, 20],                         # default: 12
    },
    'HiGSP': {
        'alpha1': [0.03, 0.05, 0.08, 0.12, 0.15],            # default: 0.08
        'alpha2': [0.5, 0.6, 0.73, 0.8, 0.9],                # default: 0.73
        'order1': [1, 2, 3, 4],                               # default: 2
        'order2': [6, 8, 12, 16, 20],                         # default: 12
    },
    'SGFCF': {
        'k': [50, 100, 150, 250, 400],                       # default: 100
        'gamma': [0.25, 0.5, 0.75, 1.0, 1.5],                # default: 1.0
        'beta_1': [0.5, 0.75, 1.0, 1.5, 2.0],                # default: 1.0
        'eps': [0.25, 0.5, 0.75, 1.0],                       # default: 0.5
    }
}

def create_split_datastructures(dataset_name, data_root=DATA_ROOT, split_ratio=0.8):
    """
    Reads the original train.txt and creates in-memory split data structures 
    WITHOUT writing new files to the disk.
    """
    dataset_path = os.path.join(data_root, dataset_name)
    train_file = os.path.join(dataset_path, 'data1.txt' if dataset_name == 'lastfm' else 'train.txt')
    
    if not os.path.exists(train_file):
        print(f"Error: {train_file} not found. Skipping split.")
        return None, None
    
    train_data = {}
    val_data = {}
    
    with open(train_file, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 2:
            if len(parts) == 1:
                # User has no interacting items
                train_data[parts[0]] = []
            continue
            
        uid = parts[0]
        items = parts[1:]
        
        # Shuffle and split
        items_copy = list(items)
        random.shuffle(items_copy)
        split_point = int(len(items_copy) * split_ratio)
        
        tr_items = items_copy[:split_point]
        val_items = items_copy[split_point:]
        
        if tr_items:
            train_data[uid] = tr_items
        if val_items:
            val_data[uid] = val_items
            
    print(f"  -> Created in-memory validation split datastructures for: {dataset_name}")
    return train_data, val_data

def parse_metrics(log_file, model_name):
    if not os.path.exists(log_file):
        return "N/A"
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    metrics = {}
    
    if model_name in ['BSPM', 'GF-CF']:
        # Format: {'precision': array([0.0034]), 'recall': array([0.0157]), 'ndcg': array([0.0084])}
        match = re.search(r"'recall': array\(\[(.*?)\]\),.*?'ndcg': array\(\[(.*?)\]\)", content, re.DOTALL)
        if match:
            recall = match.group(1).split(',')[0].strip()
            ndcg = match.group(2).split(',')[0].strip()
            metrics['Recall@20'] = recall
            metrics['NDCG@20'] = ndcg
            
    elif model_name == 'FaGSP':
        # Format: Recall@20:0.1234  F1@20:0.0123    MRR@20:0.0456   NDCG@20:0.0789
        match_recall = re.search(r"Recall@20:(\d+\.\d+)", content)
        if match_recall:
            metrics['Recall@20'] = match_recall.group(1)
        match_ndcg = re.search(r"NDCG@20:(\d+\.\d+)", content)
        if match_ndcg:
            metrics['NDCG@20'] = match_ndcg.group(1)

    elif model_name == 'HiGSP':
        # Format: Recall@20:0.1234  F1@20:0.0123    MRR@20:0.0456   NDCG@20:0.0789
        match_recall = re.search(r"Recall@20:(\d+\.\d+)", content)
        if match_recall:
            metrics['Recall@20'] = match_recall.group(1)
        match_ndcg = re.search(r"NDCG@20:(\d+\.\d+)", content)
        if match_ndcg:
            metrics['NDCG@20'] = match_ndcg.group(1)

    elif model_name == 'TurboCF':
        # Format: Recall@20: 0.0157, NDCG@20: 0.0084
        m_recall = re.search(r"Recall@20: (\d+\.\d+)", content)
        m_ndcg = re.search(r"NDCG@20: (\d+\.\d+)", content)
        if m_recall: metrics['Recall@20'] = m_recall.group(1)
        if m_ndcg: metrics['NDCG@20'] = m_ndcg.group(1)

    elif model_name == 'EASE':
        # Format: recall@[20]: [0.1431]
        m_recall = re.search(r"recall@\[20\]:\s*\[([\d\.]+)\]", content)
        m_ndcg = re.search(r"ndcg@\[20\]:\s*\[([\d\.]+)\]", content)
        if m_recall: metrics['Recall@20'] = m_recall.group(1)
        if m_ndcg: metrics['NDCG@20'] = m_ndcg.group(1)

    elif model_name == 'SGFCF':
        # Format: ndcg10 ndcg20 recall10 recall20 (space separated)
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

def run_experiment(model_name, dataset, log_dir, gpu=None, params=None, seed=None, train_data=None, val_data=None):
    if model_name not in MODELS:
        print(f"Unknown model: {model_name}")
        return None

    # Construct the base command shared by both Search and Test modes
    model_info = MODELS[model_name]
    cmd = list(model_info['cmd']) + ['--dataset', dataset]
    
    if seed is not None:
        cmd += ['--seed', str(seed)]
        
    if params:
        for k, v in params.items():
            cmd += [f'--{k}', str(v)]
    
    if gpu is not None and model_name in ['BSPM']:
        cmd += ['--gpu', str(gpu)]
    
    env = os.environ.copy()
    if gpu is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    
    param_suffix = "_" + "_".join([f"{k}{v}" for k, v in params.items()]) if params else ""
    log_file = os.path.join(log_dir, f"{model_name}_{dataset}{param_suffix}.log")

    # -------------------------------------------------------------------------
    # SEARCH MODE (Using Subprocess Commands)
    # -------------------------------------------------------------------------
    if train_data is not None and val_data is not None:
        print(f"  -> [SEARCH MODE] Running {model_name} on {dataset} with {params}... (Logging to {log_file})")
        # print(f"     [WARNING]: Subprocess cannot read in-memory dictionaries. It will default to disk files.")
        
        with open(log_file, 'a') as f:
            start_time = time.time()
            try:
                subprocess.run(cmd, cwd=model_info['cwd'], stdout=f, stderr=subprocess.STDOUT, check=True, env=env)
                duration = time.time() - start_time
                print(f"  -> Finished in {duration:.2f}s")
                return parse_metrics(log_file, model_name)
            except subprocess.CalledProcessError as e:
                print(f"  !! Error running search for {model_name}. Check {log_file}")
                return "ERROR"

    # -------------------------------------------------------------------------
    # TEST MODE (Original Subprocess Execution)
    # -------------------------------------------------------------------------
    print(f"  -> [TEST MODE] Running {model_name} on {dataset} with {params if params else 'defaults'}... (Logging to {log_file})")
    
    with open(log_file, 'a') as f:
        start_time = time.time()
        try:
            subprocess.run(cmd, cwd=model_info['cwd'], stdout=f, stderr=subprocess.STDOUT, check=True, env=env)
            duration = time.time() - start_time
            print(f"  -> Finished in {duration:.2f}s")
            return parse_metrics(log_file, model_name)
        except subprocess.CalledProcessError as e:
            print(f"  !! Error running {model_name}. Check {log_file}")
            return "ERROR"

def main():
    parser = argparse.ArgumentParser(description="Run multiple recommendation models with parameter search.")
    
    parser.add_argument("model", nargs="?", help="Specific model to run.")
    parser.add_argument("--models", nargs="+", help="Models to run.")
    parser.add_argument("--datasets", nargs="+", help="Datasets to run on.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use.")
    parser.add_argument("--seed", type=int, default=2020)

    parser.add_argument("--log_dir", default=BASE_DIR+"/00_experiment_logs", help="Directory to save logs.")
    parser.add_argument("--results_file", default=BASE_DIR+"/00_results.txt", help="File to save summarized results.")
    parser.add_argument("--search", action="store_true", help="Perform parameter search using a validation split.")
    
    args = parser.parse_args()
    os.makedirs(args.log_dir, exist_ok=True)
    
    if args.models:
        models_to_run = args.models
    elif args.model:
        models_to_run = [args.model]
    else:
        models_to_run = list(MODELS.keys())
        
    datasets_to_run = args.datasets if args.datasets else DATASETS
    
    print(f"Starting experiments at {time.ctime()}")
    print(f"Search mode: {args.search}")
    print(f"Seed: {args.seed}")
    print("-" * 50)
    
    with open(args.results_file, 'a') as f:
        f.write(f"\nExperiment Results Summary - {time.ctime()} (Search Mode: {args.search}, Seed: {args.seed})\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'Model':<10} | {'Dataset':<20} | {'Parameters':<30} | Metrics\n")
        f.write("-" * 80 + "\n")
    
    for dataset in datasets_to_run:
        train_data, val_data = None, None
        
        # Only populate the memory datastructures if search is enabled
        if args.search:
            train_data, val_data = create_split_datastructures(dataset)
            if train_data is None: 
                continue # Skip if file wasn't found
            
        swapped = False
        train_filename = "data1.txt" if dataset == "lastfm" else "train.txt"
        test_filename = "test1.txt" if dataset == "lastfm" else "test.txt"
        
        orig_train = os.path.join(DATA_ROOT, dataset, train_filename)
        orig_test = os.path.join(DATA_ROOT, dataset, test_filename)
        bak_train = orig_train + ".bak"
        bak_test = orig_test + ".bak"
        
        search_train = os.path.join(DATA_ROOT, "00 search_sets", dataset, "train_val.txt")
        search_val = os.path.join(DATA_ROOT, "00 search_sets", dataset, "validation_val.txt")
        
        best_params_dict = {}
        
        try:
            if args.search:
                if not os.path.exists(search_train) or not os.path.exists(search_val):
                    print(f"Error: Search sets for {dataset} not found in 00 search_sets. Skipping.")
                    continue
                
                print(f"  -> [SEARCH MODE] Swapping train.txt and test.txt with validation splits for {dataset}")
                # Backup original train/test files
                if os.path.exists(orig_train):
                    os.rename(orig_train, bak_train)
                if os.path.exists(orig_test):
                    os.rename(orig_test, bak_test)
                    
                # Copy validation files to original names
                shutil.copy(search_train, orig_train)
                shutil.copy(search_val, orig_test)
                
                # Backup any npz files
                npz_files = [f for f in os.listdir(os.path.join(DATA_ROOT, dataset)) if f.endswith('.npz')]
                for npz in npz_files:
                    orig_npz = os.path.join(DATA_ROOT, dataset, npz)
                    os.rename(orig_npz, orig_npz + ".bak")
                    
                swapped = True

            for model_name in models_to_run:

                if model_name=="PGSP": continue

                print(f"Model: {model_name} | Dataset: {dataset}")
                
                if args.search and model_name in PARAM_GRIDS:
                    grid = PARAM_GRIDS[model_name]
                    keys = list(grid.keys())
                    values = list(grid.values())
                    combinations = list(itertools.product(*values))
                    
                    best_recall = -1.0
                    best_params = None
                    
                    for combo in combinations:
                        params = dict(zip(keys, combo))
                        metrics_str = run_experiment(
                            model_name, dataset, args.log_dir, args.gpu, params, args.seed,
                            train_data=train_data, val_data=val_data
                        )
                        param_str = ", ".join([f"{k}={v}" for k, v in params.items()])
                        result_line = f"{model_name:<10} | {dataset:<20} | {param_str:<30} | {metrics_str} (Val)"
                        with open(args.results_file, 'a') as f:
                            f.write(result_line + "\n")
                            
                        recall = get_recall_value(metrics_str)
                        if recall > best_recall:
                            best_recall = recall
                            best_params = params
                    
                    best_params_dict[model_name] = best_params
                    print(f"  -> Best validation parameters for {model_name} on {dataset}: {best_params} (Recall@20: {best_recall:.4f})")
                else:
                    metrics_str = run_experiment(model_name, dataset, args.log_dir, args.gpu, None, args.seed)
                    result_line = f"{model_name:<10} | {dataset:<20} | {'default':<30} | {metrics_str}"
                    with open(args.results_file, 'a') as f:
                        f.write(result_line + "\n")
                        
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
        if args.search:
            print(f"=== [FINAL TEST MODE] Running models on {dataset} with best found parameters ===")
            for model_name, best_params in best_params_dict.items():
                if best_params is not None:
                    metrics_str = run_experiment(model_name, dataset, args.log_dir, args.gpu, best_params, args.seed)
                    param_str = ", ".join([f"{k}={v}" for k, v in best_params.items()])
                    result_line = f"{model_name:<10} | {dataset:<20} | {param_str:<30} | {metrics_str} (Test)"
                    with open(args.results_file, 'a') as f:
                        f.write(result_line + "\n")
                    
        print("-" * 50)
            
    print(f"All experiments finished. Results saved to {args.results_file}")

if __name__ == "__main__":
    main()