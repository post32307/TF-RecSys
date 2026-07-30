import os

# Limit CPU threads for openmp, mkl, openblas, etc.  Verification mode
# can override this per subprocess so the scheduler can pack more jobs.
DEFAULT_THREADS_PER_TASK = int(os.environ.get('SWEEP_THREADS_PER_TASK', '8'))
os.environ['OMP_NUM_THREADS'] = str(DEFAULT_THREADS_PER_TASK)
os.environ['MKL_NUM_THREADS'] = str(DEFAULT_THREADS_PER_TASK)
os.environ['OPENBLAS_NUM_THREADS'] = str(DEFAULT_THREADS_PER_TASK)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(DEFAULT_THREADS_PER_TASK)
os.environ['NUMEXPR_NUM_THREADS'] = str(DEFAULT_THREADS_PER_TASK)

import subprocess
import sys
import time
import argparse
import re
import itertools
from pathlib import Path
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import signal

def sig_handler(signum, frame):
    print(f"\n[!] Process terminated with signal {signum}")
    raise KeyboardInterrupt

signal.signal(signal.SIGINT, sig_handler)
signal.signal(signal.SIGTERM, sig_handler)

# Base directory of this script
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
SWEEP_RESULTS_DIR = os.path.join(BASE_DIR, "sweep_results")

def ensure_sweep_results_dir():
    os.makedirs(SWEEP_RESULTS_DIR, exist_ok=True)

def seed_log_dir(seed):
    return os.path.join(SWEEP_RESULTS_DIR, f"00_exp_logs_seed_{seed}")

def val_results_file(seed):
    return os.path.join(SWEEP_RESULTS_DIR, f"00_val_results_seed_{seed}.txt")

def test_results_file(seed):
    return os.path.join(SWEEP_RESULTS_DIR, f"00_test_results_seed_{seed}.txt")

# Datasets directory (relative to repository root)
current_dir = Path(__file__).resolve().parent
DATA_ROOT = current_dir.parent / "data"
OG_DATA_ROOT = DATA_ROOT / "og_dataset"
COMBINED_DATA_ROOT = DATA_ROOT / "combined_dataset"
SPLIT_DATA_ROOT = DATA_ROOT / "split_dataset"
SPLIT_DATA_ROOT_PREFIX = "00_dataset_seed"

def discover_datasets():
    if COMBINED_DATA_ROOT.exists():
        names = [
            path.name.removesuffix("_combined.txt")
            for path in COMBINED_DATA_ROOT.glob("*_combined.txt")
        ]
        return sorted(names)
    if OG_DATA_ROOT.exists():
        return sorted([
            path.name for path in OG_DATA_ROOT.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        ])
    if DATA_ROOT.exists() and DATA_ROOT.is_dir():
        return sorted([
            d for d in os.listdir(DATA_ROOT)
            if os.path.isdir(os.path.join(DATA_ROOT, d))
            and not d.startswith(".")
            and not d.endswith("_val")
            and not d.startswith("00")
            and d not in {"combined_dataset", "og_dataset"}
        ])
    return []

# Datasets list dynamically retrieved from the combined/original data folder.
DATASETS = discover_datasets()

print("All Datasets:", DATASETS)

# Model configurations with relative paths
MODELS = {
    'BSPM': {
        'cwd': os.path.join(BASE_DIR, 'BSPM', 'bspm'),
        'cmd': [sys.executable, 'main.py', '--simple_model', 'bspm', '--tensorboard', '0']
    },
    'EASE': {
        'cwd': os.path.join(BASE_DIR, 'EASE'),
        'cmd': [sys.executable, 'run.py']
    },
    'GF-CF': {
        'cwd': os.path.join(BASE_DIR, 'GF_CF', 'Table3&4'),
        'cmd': [sys.executable, 'main.py', '--simple_model', 'gf-cf', '--tensorboard', '0']
    },
    'PGSP': {
        'cwd': os.path.join(BASE_DIR, 'PGSP'),
        'cmd': [sys.executable, 'PGSP.py']
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
    },
    'ChebyCF': {
        'cwd': os.path.join(BASE_DIR, 'ChebyCF'),
        'cmd': [sys.executable, 'main.py']
    },
    'DAN': {
        'cwd': os.path.join(BASE_DIR, 'DAN', 'code'),
        'cmd': [sys.executable, 'main.py']
    }
}

# Shared rank grid for factor dimensions / truncated spectral dimensions.
RANK_GRID = [32, 64, 128, 256, 512, 1024]

BSPM_BASE_COMBOS = [
    # Defaults plus the parameter families highlighted in the BSPM README.
    {'solver_shr': 'euler', 'K_s': 1, 'T_s': 1.0, 'final_sharpening': True, 't_point_combination': False, 'idl_beta': 0.3},
    {'solver_shr': 'rk4', 'K_s': 1, 'T_s': 2.5, 'final_sharpening': True, 't_point_combination': False, 'idl_beta': 0.2},
    {'solver_shr': 'euler', 'K_s': 1, 'T_s': 1.2, 'final_sharpening': True, 't_point_combination': True, 'idl_beta': 0.3},
    {'solver_shr': 'rk4', 'K_s': 2, 'T_s': 2.2, 'final_sharpening': False, 't_point_combination': False, 'idl_beta': 0.3},
    {'solver_shr': 'rk4', 'K_s': 1, 'T_s': 2.5, 'final_sharpening': True, 't_point_combination': False, 'idl_beta': 0.1},
    {'solver_shr': 'euler', 'K_s': 1, 'T_s': 1.2, 'final_sharpening': True, 't_point_combination': True, 'idl_beta': 0.5},
    {'solver_shr': 'rk4', 'K_s': 2, 'T_s': 2.2, 'final_sharpening': False, 't_point_combination': True, 'idl_beta': 0.7},
]

SGFCF_BASE_COMBOS = [
    # Curated from the SGFCF README dense/sparse settings, adapted to beta_1/beta_2.
    {'beta_1': 1.0, 'beta_2': 1.0, 'alpha': 0.0, 'eps': 0.5, 'gamma': 1.0},
    {'beta_1': 0.7, 'beta_2': 1.1, 'alpha': 0.0, 'eps': 0.28, 'gamma': 1.5},
    {'beta_1': 4.3, 'beta_2': 5.1, 'alpha': 2.3, 'eps': 0.5, 'gamma': 0.05},
    {'beta_1': 0.9, 'beta_2': 1.0, 'alpha': 0.0, 'eps': 0.37, 'gamma': 0.3},
    {'beta_1': 2.0, 'beta_2': 2.4, 'alpha': 0.0, 'eps': 0.37, 'gamma': 0.07},
    {'beta_1': 1.0, 'beta_2': 1.3, 'alpha': 19.0, 'eps': 0.5, 'gamma': 0.5},
    {'beta_1': 2.0, 'beta_2': 5.0, 'alpha': 5.3, 'eps': 0.5, 'gamma': 0.025},
    {'beta_1': 0.3, 'beta_2': 2.5, 'alpha': 0.0, 'eps': 0.34, 'gamma': 1.6},
    {'beta_1': 2.5, 'beta_2': 6.5, 'alpha': 4.8, 'eps': 0.5, 'gamma': 0.025},
]

# Example parameter grids for search
PARAM_GRIDS = {
    'BSPM': {
        '__combos__': [
            {**base, 'factor_dim': factor_dim}
            for base in BSPM_BASE_COMBOS
            for factor_dim in RANK_GRID
        ],
    },
    'EASE': {
        'lambda_': [50, 100, 200, 500, 1000, 2000],
    },
    'GF-CF': {
        'k': [32, 64, 128, 256, 512, 1024],
    },
    'PGSP': {
        'k': [32, 64, 128, 256, 512, 1024],
        'phi': [0.0, 0.25, 0.5, 0.75, 1.0],
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
        'k1': [32, 64, 128, 256, 512, 1024],
        'k2': [32, 64, 128, 256, 512, 1024],
    },
    'HiGSP': {
        'alpha1': [0.08],
        'alpha2': [0.5, 0.73, 0.9],
        'order1': [2],
        'order2': [4, 8, 12, 16],
        'pri_factor': [32, 64, 128, 256, 512, 1024],
    },

    'SGFCF': {
        '__combos__': [
            {'k': k, **base}
            for base in SGFCF_BASE_COMBOS
            for k in RANK_GRID
        ],
    },
    'ChebyCF': {
        'K': [8,32],
        'phi': [1.5, 10.0, 15.0],
        'eta': [256, 512],
        'alpha': [0.1, 0.2],
        'beta': [0.2, 0.4],
    },
    'DAN': {
        'model': ['RLAE_DAN'],
        'reg_p': [10.0, 20.0, 50.0],
        'alpha': [0.2, 0.3],
        'beta': [0.2, 0.3],
        'drop_p': [0.4, 0.6, 0.7],
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

    elif model_name == 'PGSP':
        m_recall = list(re.finditer(r"Recall@20:\s*([+-]?\d*\.?\d+(?:[eE][-+]?\d+)?)", content))
        m_ndcg = list(re.finditer(r"NDCG@20:\s*([+-]?\d*\.?\d+(?:[eE][-+]?\d+)?)", content))
        if m_recall:
            metrics['Recall@20'] = m_recall[-1].group(1)
        if m_ndcg:
            metrics['NDCG@20'] = m_ndcg[-1].group(1)

    elif model_name in ['ChebyCF', 'DAN']:
        m_recall = list(re.finditer(
            r"Recall@20:\s*([+-]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
            content,
        ))
        m_ndcg = list(re.finditer(
            r"NDCG@20:\s*([+-]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
            content,
        ))
        if m_recall:
            metrics['Recall@20'] = m_recall[-1].group(1)
        if m_ndcg:
            metrics['NDCG@20'] = m_ndcg[-1].group(1)

    if not metrics:
        return "N/A"
    
    return ", ".join([f"{k}: {v}" for k, v in metrics.items()])

def get_recall_value(metrics_str):
    if not metrics_str or metrics_str == "N/A" or metrics_str == "ERROR":
        return -1.0
    metrics_str = re.sub(r"^\((?:Val|Test)\)\s*", "", metrics_str)
    match = re.search(r"Recall@20:\s*([+-]?\d*\.?\d+(?:[eE][-+]?\d+)?)", metrics_str)
    if match:
        return float(match.group(1))
    return -1.0

def format_metric_values(metrics_str, phase=None):
    if metrics_str is None:
        body = "N/A"
    else:
        body = str(metrics_str)

    def repl(match):
        label, value = match.groups()
        try:
            return f"{label}{float(value):.6g}"
        except ValueError:
            return match.group(0)

    body = re.sub(
        r"(:\s*)([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)",
        repl,
        body,
    )
    if phase:
        return f"({phase}) {body}"
    return body

def format_search_result_line(model_name, dataset, params, metrics, duration, phase):
    return (
        f"{model_name:<10} | {dataset:<20} | "
        f"{format_metric_values(metrics, phase):<60} | "
        f"{format_duration(duration):<8} | {format_params(params)}"
    )

def is_result_data_line(line):
    if "|" not in line:
        return False
    parts = [part.strip() for part in line.rstrip("\n").split("|")]
    if len(parts) < 5:
        return False
    return parts[0] in MODELS and parts[2].startswith(("(", "TIMEOUT", "ERROR", "N/A"))

def result_line_dataset(line):
    if not is_result_data_line(line):
        return None
    return line.split("|", 2)[1].strip()

def header_list(values, max_items=12):
    values = [str(value) for value in values]
    if not values:
        return "N/A"
    if len(values) <= max_items:
        return ", ".join(values)
    shown = ", ".join(values[:max_items])
    return f"{shown}, ... ({len(values)} total)"

def describe_param_grid(model_name, dataset=None, limit=None):
    grid = get_param_grid(model_name, dataset) if dataset else PARAM_GRIDS.get(model_name, {})
    if not grid:
        return "default"
    if "__combos__" in grid:
        combo_count = len(grid["__combos__"])
        if limit is not None:
            combo_count = min(combo_count, limit)
        keys = []
        for combo in grid["__combos__"][:combo_count]:
            for key in combo:
                if key not in keys:
                    keys.append(key)
        return f"{combo_count} combos over {', '.join(keys)}"
    parts = []
    combo_count = 1
    for key, values in grid.items():
        value_list = list(values)
        combo_count *= max(1, len(value_list))
        parts.append(f"{key}=[{header_list(value_list, max_items=8)}]")
    if limit is not None:
        combo_count = min(combo_count, limit)
    return f"{combo_count} combos: " + "; ".join(parts)

def describe_hyperparameters(models, datasets, limit=None, phase=None):
    if phase == "Test":
        return "Final test uses the best validation parameters found in the validation results file."

    summaries = []
    datasets = list(datasets or [])
    for model_name in models or []:
        if not datasets:
            summaries.append(f"{model_name}: {describe_param_grid(model_name, limit=limit)}")
            continue
        descriptions = {
            describe_param_grid(model_name, dataset, limit=limit)
            for dataset in datasets
        }
        if len(descriptions) == 1:
            summaries.append(f"{model_name}: {next(iter(descriptions))}")
        else:
            summaries.append(f"{model_name}: dataset-specific grids")
    return " | ".join(summaries) if summaries else "N/A"

def pending_param_dicts(results_file, model_name, dataset, limit=None):
    params_to_run = list(iter_param_dicts(model_name, dataset, limit=None))
    if params_to_run == [None]:
        return []

    completed_keys = existing_param_keys(results_file, "Val", model_name, dataset)
    pending = [
        params for params in params_to_run
        if format_params(params) not in completed_keys
    ]
    if limit is not None:
        pending = pending[:limit]
    return pending

RUN_HISTORY_RE = re.compile(
    r"^[A-Z][a-z]{2} [A-Z][a-z]{2}\s+\d{1,2} "
    r"\d{2}:\d{2}:\d{2} \d{4} : "
)

def first_pending_param(results_file, models, datasets):
    for dataset in datasets:
        for model_name in models:
            pending = pending_param_dicts(results_file, model_name, dataset, limit=1)
            if not pending:
                continue
            return model_name, dataset, pending[0]
    return None, None, None

def first_pending_test(test_results_file, models, datasets):
    for dataset in datasets:
        for model_name in models:
            if not has_test_result(test_results_file, model_name, dataset):
                return model_name, dataset, None
    return None, None, None

def format_run_history_line(started_at, model_name, dataset, params, include_params=True):
    if model_name is None or dataset is None:
        return f"{time.ctime(started_at)} : none"
    if not include_params:
        return f"{time.ctime(started_at)} : {model_name} / {dataset}"
    params_text = format_params(params)
    return f"{time.ctime(started_at)} : {model_name} / {dataset} / {params_text}"

def describe_pending_test_runs(test_results_file, models, datasets):
    lines = []
    for dataset in datasets:
        pending_models = [
            model_name for model_name in models
            if not has_test_result(test_results_file, model_name, dataset)
        ]
        if pending_models:
            lines.append(
                f"  {dataset} | {len(pending_models)} model tests | {', '.join(pending_models)}"
            )
    if not lines:
        return ["  none"]
    return lines

def build_results_header(title, seed, mode, started_at, start_model, start_dataset,
                         start_phase, models=None, datasets=None,
                         hyperparameter_summary=None, run_history=None):
    lines = [
        f"{title} (Seed: {seed})",
        "Run started:",
        *(run_history or []),
    ]
    lines.extend([
        "=" * 120,
        f"{'Model':<10} | {'Dataset':<20} | {'Metrics':<60} | {'Time':<8} | Parameters",
        "-" * 120,
    ])
    return "\n".join(lines) + "\n"

def normalize_results_file(results_file, title, seed, mode, started_at,
                           start_model=None, start_dataset=None, start_phase=None,
                           models=None, datasets=None,
                           hyperparameter_summary=None, run_history_line=None):
    rows = []
    run_history = []
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            for raw_line in f:
                line = raw_line.rstrip("\n")
                if is_result_data_line(line):
                    rows.append(line)
                elif RUN_HISTORY_RE.match(line):
                    run_history.append(line)

    if run_history_line:
        run_history.append(run_history_line)

    with open(results_file, "w") as f:
        f.write(build_results_header(
            title,
            seed,
            mode,
            started_at,
            start_model,
            start_dataset,
            start_phase,
            models,
            datasets,
            hyperparameter_summary,
            run_history,
        ))
        last_dataset = None
        for row in rows:
            dataset = result_line_dataset(row)
            if last_dataset is not None and dataset != last_dataset:
                f.write("\n")
            f.write(row + "\n")
            last_dataset = dataset

def append_result_line(results_file, result_line, dataset):
    existing = ""
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            existing = f.read()

    last_dataset = None
    for line in reversed(existing.splitlines()):
        last_dataset = result_line_dataset(line)
        if last_dataset is not None:
            break

    if existing:
        desired_suffix = "\n\n" if last_dataset is not None and last_dataset != dataset else "\n"
        normalized = existing.rstrip("\n") + desired_suffix
        if normalized != existing:
            with open(results_file, "w") as f:
                f.write(normalized)

    with open(results_file, "a") as f:
        f.write(result_line + "\n")

def parse_result_rows(results_file, phase=None):
    rows = []
    if not os.path.exists(results_file):
        return rows

    with open(results_file, "r") as f:
        for line in f:
            parts = [part.strip() for part in line.rstrip("\n").split("|")]
            if len(parts) < 5:
                continue
            metrics = parts[2]
            if phase and not metrics.startswith(f"({phase})"):
                continue
            rows.append({
                "model": parts[0],
                "dataset": parts[1],
                "metrics": metrics,
                "duration": parts[3],
                "params": parts[4],
            })
    return rows

def existing_param_keys(results_file, phase, model_name, dataset):
    return {
        row["params"]
        for row in parse_result_rows(results_file, phase)
        if row["model"] == model_name and row["dataset"] == dataset
    }

def best_existing_params(results_file, model_name, dataset):
    best_recall = -1.0
    best_params = None
    for row in parse_result_rows(results_file, "Val"):
        if row["model"] != model_name or row["dataset"] != dataset:
            continue
        recall = get_recall_value(row["metrics"])
        if recall > best_recall:
            best_recall = recall
            best_params = parse_params_string(row["params"])
    return best_params, best_recall

def has_test_result(test_results_file, model_name, dataset):
    return any(
        row["model"] == model_name and row["dataset"] == dataset
        for row in parse_result_rows(test_results_file, "Test")
    )

def parse_params_string(param_str):
    if not param_str or param_str == "default":
        return None
    import ast
    params = {}
    for pair in param_str.split(","):
        if "=" not in pair:
            continue
        key, value = pair.split("=", 1)
        key = key.strip()
        value = value.strip()
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass
        params[key] = value
    return params

def seed_data_root(seed):
    return SPLIT_DATA_ROOT / f"{SPLIT_DATA_ROOT_PREFIX}_{seed}"

def seed_dataset_dir(seed, dataset):
    return seed_data_root(seed) / dataset

def seed_dataset_exists(seed, dataset=None):
    root = seed_data_root(seed)
    if dataset is None:
        return root.exists()
    dataset_dir = root / dataset
    return (
        dataset_dir.exists()
        and (dataset_dir / "train.txt").exists()
        and (dataset_dir / "val.txt").exists()
        and (dataset_dir / "test.txt").exists()
    )

def validation_env(seed, dataset):
    if not seed_dataset_exists(seed, dataset):
        return None
    return {
        "TFRECSYS_DATA_ROOT": str(seed_data_root(seed)),
        "TFRECSYS_EVAL_FILE": "val.txt",
    }

def test_env(seed):
    return {
        "TFRECSYS_DATA_ROOT": str(seed_data_root(seed)),
        "TFRECSYS_EVAL_FILE": "test.txt",
    }

def format_duration(seconds):
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"

def format_bytes(num_bytes):
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.1f}{unit}" if unit != "B" else f"{int(value)}B"
        value /= 1024.0
    return f"{value:.1f}TB"

def format_params(params):
    if not params:
        return "default"
    return ", ".join([f"{k}={v}" for k, v in params.items()])

def is_valid_metrics(metrics_str):
    return get_recall_value(metrics_str) >= 0

def get_dataset_size_bytes(dataset):
    combined_file = COMBINED_DATA_ROOT / f"{dataset}_combined.txt"
    if combined_file.exists():
        try:
            return os.path.getsize(combined_file)
        except OSError:
            pass

    dataset_dir = OG_DATA_ROOT / dataset
    if not dataset_dir.exists():
        for split_root in sorted(SPLIT_DATA_ROOT.glob(f"{SPLIT_DATA_ROOT_PREFIX}_*")):
            candidate = split_root / dataset
            if candidate.exists():
                dataset_dir = candidate
                break

    total = 0
    if not dataset_dir.exists():
        return total
    for root, _, files in os.walk(dataset_dir):
        for filename in files:
            if filename.endswith(".bak"):
                continue
            path = os.path.join(root, filename)
            try:
                total += os.path.getsize(path)
            except OSError:
                pass
    return total

def get_sorted_datasets(datasets):
    sizes = {dataset: get_dataset_size_bytes(dataset) for dataset in datasets}
    return sorted(datasets, key=lambda dataset: (sizes[dataset], dataset)), sizes

def read_mem_available_mb():
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        return None
    return None

def get_load_average():
    try:
        return os.getloadavg()[0]
    except (AttributeError, OSError):
        return 0.0

def get_visible_gpu_indices():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return []

    gpu_ids = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            gpu_ids.append(int(line))
        except ValueError:
            continue
    return gpu_ids

def normalize_gpus(requested_gpus, fallback_gpu=0):
    requested = list(requested_gpus or [fallback_gpu])
    visible = get_visible_gpu_indices()
    if not visible:
        return requested

    visible_set = set(visible)
    valid = [gpu for gpu in requested if gpu in visible_set]
    invalid = [gpu for gpu in requested if gpu not in visible_set]
    if invalid:
        print(
            f"Warning: requested GPU IDs {invalid} are not visible on this server. "
            f"Using {valid or [visible[0]]}."
        )
    return valid or [visible[0]]

def estimate_task_memory_mb(dataset_size_bytes, min_task_memory_mb, dataset_memory_multiplier):
    dataset_mb = dataset_size_bytes / (1024.0 * 1024.0)
    return max(float(min_task_memory_mb), dataset_mb * float(dataset_memory_multiplier))

def current_parallel_limit(max_workers, threads_per_task):
    cpu_count = os.cpu_count() or 1
    load_avg = get_load_average()
    available_cpu = max(1.0, cpu_count - load_avg)
    cpu_slots = max(1, int(available_cpu // max(1, threads_per_task)))
    return max(1, min(max_workers, cpu_slots))

def progress_bar(done, total, width=28):
    if total <= 0:
        return "[" + "-" * width + "]"
    filled = int(width * done / total)
    return "[" + "#" * filled + "-" * (width - filled) + "]"

def iter_param_dicts(model_name, dataset, limit=None):
    grid = get_param_grid(model_name, dataset)
    if not grid:
        yield None
        return

    if '__combos__' in grid:
        for count, params in enumerate(grid['__combos__']):
            if limit is not None and count >= limit:
                break
            yield dict(params)
        return

    keys = list(grid.keys())
    count = 0
    for combo in itertools.product(*grid.values()):
        yield dict(zip(keys, combo))
        count += 1
        if limit is not None and count >= limit:
            break

def run_experiment(model_name, dataset, log_dir, gpu=None, params=None, seed=None,
                   timeout=300, threads_per_task=None, return_details=False,
                   extra_env=None):
    if model_name not in MODELS:
        print(f"Unknown model: {model_name}")
        if return_details:
            return {"metrics": None, "status": "ERROR", "duration": 0.0, "log_file": None}
        return None

    model_info = MODELS[model_name]
    if 'cmd' not in model_info:
        print(f"Skipping {model_name}: No 'cmd' defined.")
        if return_details:
            return {"metrics": "N/A", "status": "SKIPPED", "duration": 0.0, "log_file": None}
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
    if extra_env:
        env.update({key: str(value) for key, value in extra_env.items()})
    if gpu is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    worker_threads = threads_per_task if threads_per_task is not None else DEFAULT_THREADS_PER_TASK
    env['OMP_NUM_THREADS'] = str(worker_threads)
    env['MKL_NUM_THREADS'] = str(worker_threads)
    env['OPENBLAS_NUM_THREADS'] = str(worker_threads)
    env['VECLIB_MAXIMUM_THREADS'] = str(worker_threads)
    env['NUMEXPR_NUM_THREADS'] = str(worker_threads)
    
    param_suffix = "_" + "_".join([f"{k}{v}" for k, v in params.items()]) if params else ""
    log_file = os.path.join(log_dir, f"{model_name}_{dataset}{param_suffix}.log")

    print(f"  -> Running {model_name} on {dataset} on GPU {gpu} with {params if params else 'defaults'}... (Logging to {log_file})")
    
    with open(log_file, 'a') as f:
        start_time = time.time()
        try:
            f.write(f"\n=== {time.ctime()} ===\n")
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', '')} threads={worker_threads}\n")
            if extra_env:
                f.write("Extra env: " + " ".join([f"{key}={value}" for key, value in sorted(extra_env.items())]) + "\n")
            f.flush()
            proc = subprocess.Popen(
                cmd,
                cwd=model_info['cwd'],
                stdout=f,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
            try:
                returncode = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                    proc.wait(timeout=5)
                except Exception:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except Exception:
                        pass
                duration = time.time() - start_time
                print(f"  !! Timeout running {model_name} on {dataset} after {duration:.2f}s. Check {log_file}")
                metrics = "TIMEOUT"
                if return_details:
                    return {"metrics": metrics, "status": "TIMEOUT", "duration": duration, "log_file": log_file}
                return metrics

            if returncode != 0:
                duration = time.time() - start_time
                print(f"  !! Error running {model_name} on GPU {gpu}. Check {log_file}")
                if return_details:
                    return {"metrics": "ERROR", "status": "ERROR", "duration": duration, "log_file": log_file}
                return "ERROR"

            duration = time.time() - start_time
            print(f"  -> Finished {model_name} on GPU {gpu} in {duration:.2f}s")
            metrics = parse_metrics(log_file, model_name)
            if return_details:
                return {"metrics": metrics, "status": "OK", "duration": duration, "log_file": log_file}
            return metrics
        except Exception as e:
            duration = time.time() - start_time
            print(f"  !! Error running {model_name} on GPU {gpu}. Check {log_file}")
            f.write(f"\nException: {e}\n")
            if return_details:
                return {"metrics": "ERROR", "status": "ERROR", "duration": duration, "log_file": log_file}
            return "ERROR"

def verify_model_dataset(model_name, dataset, log_dir, gpu, seed, timeout, threads_per_task, combination_limit):
    started_at = time.time()
    attempts = []
    last_detail = None
    split_env = test_env(seed)

    for params in iter_param_dicts(model_name, dataset, combination_limit):
        detail = run_experiment(
            model_name,
            dataset,
            log_dir,
            gpu,
            params,
            seed,
            timeout=timeout,
            threads_per_task=threads_per_task,
            return_details=True,
            extra_env=split_env,
        )
        detail["params"] = params
        attempts.append(detail)
        last_detail = detail

        if detail["status"] == "TIMEOUT":
            break
        if detail["status"] == "OK" and is_valid_metrics(detail["metrics"]):
            break

    if last_detail is None:
        last_detail = {
            "metrics": "N/A",
            "status": "SKIPPED",
            "duration": 0.0,
            "log_file": None,
            "params": None,
        }

    worked = last_detail["status"] == "OK" and is_valid_metrics(last_detail["metrics"])
    status = "OK" if worked else last_detail["status"]
    if status == "OK" and not worked:
        status = "NO_METRICS"

    return {
        "model": model_name,
        "dataset": dataset,
        "status": status,
        "metrics": last_detail["metrics"],
        "params": last_detail.get("params"),
        "duration": time.time() - started_at,
        "attempts": len(attempts),
        "log_file": last_detail.get("log_file"),
        "gpu": gpu,
    }

def print_progress(done_count, total_count, started_at, active, last_result=None):
    pct = (done_count / total_count * 100.0) if total_count else 100.0
    active_bits = []
    now = time.time()
    for meta in list(active.values())[:4]:
        active_bits.append(f"{meta['model']}/{meta['dataset']} {format_duration(now - meta['start_time'])}")
    active_str = ", ".join(active_bits) if active_bits else "none"
    last_str = ""
    if last_result:
        last_str = (
            f" | last: {last_result['model']}/{last_result['dataset']} "
            f"{last_result['status']} {format_duration(last_result['duration'])}"
        )
    print(
        f"{progress_bar(done_count, total_count)} "
        f"{done_count}/{total_count} {pct:5.1f}% "
        f"elapsed {format_duration(time.time() - started_at)}"
        f"{last_str} | running: {active_str}",
        flush=True,
    )

def report_dataset_complete(dataset, dataset_results, dataset_started_at, dataset_size):
    print(
        f"\n=== Dataset complete: {dataset} ({format_bytes(dataset_size)}) "
        f"in {format_duration(time.time() - dataset_started_at)} ==="
    )
    for result in sorted(dataset_results, key=lambda item: item["model"]):
        print(
            f"  {result['model']:<10} {result['status']:<10} "
            f"{format_duration(result['duration']):>8} "
            f"{format_params(result['params']):<45} {result['metrics']}"
        )
    print("=" * 80, flush=True)

def format_verification_result_line(result):
    return (
        f"{result['model']:<10} | {result['dataset']:<20} | "
        f"{format_metric_values(result['metrics']):<60} | "
        f"{format_duration(result['duration']):<8} | "
        f"{format_params(result['params'])}"
    )

def write_dataset_chunk(test_results_file, seed, dataset, dataset_results, dataset_started_at, dataset_size):
    sorted_results = sorted(dataset_results, key=lambda item: item["model"])
    with open(test_results_file, 'a') as f:
        f.write(f"\nDataset Verification Chunk - {time.ctime()} (Seed: {seed})\n")
        f.write("=" * 120 + "\n")
        f.write(
            f"Dataset: {dataset} | Size: {format_bytes(dataset_size)} | "
            f"Elapsed: {format_duration(time.time() - dataset_started_at)}\n"
        )
        f.write("-" * 120 + "\n")
        f.write(f"{'Model':<10} | {'Dataset':<20} | {'Metrics':<60} | {'Time':<8} | Parameters\n")
        f.write("-" * 120 + "\n")
        for result in sorted_results:
            f.write(format_verification_result_line(result) + "\n")
        f.write("\n")
        f.write("=" * 120 + "\n")

def get_completed_datasets_from_results(results_file, models_to_run):
    if not os.path.exists(results_file):
        return set()

    required_models = set(models_to_run)
    completed = set()
    current_dataset = None
    current_models = set()

    def close_chunk():
        if current_dataset and required_models.issubset(current_models):
            completed.add(current_dataset)

    with open(results_file, 'r') as f:
        for line in f:
            if line.startswith("Dataset: "):
                close_chunk()
                current_dataset = line.split("|", 1)[0].split(":", 1)[1].strip()
                current_models = set()
                continue

            if current_dataset and "|" in line:
                model_name = line.split("|", 1)[0].strip()
                if model_name in required_models:
                    current_models.add(model_name)

    close_chunk()
    return completed

def run_verification_sweep(args, models_to_run, datasets_to_run):
    requested_models = list(models_to_run)
    unknown_models = [m for m in requested_models if m not in MODELS]
    skipped_models = [m for m in requested_models if m in MODELS and 'cmd' not in MODELS[m]]
    models_to_run = [m for m in requested_models if m in MODELS and 'cmd' in MODELS[m]]
    if unknown_models:
        print(f"Unknown models skipped: {unknown_models}")
    if skipped_models:
        print(f"Skipping models without runnable commands: {skipped_models}")

    datasets_to_run, dataset_sizes = get_sorted_datasets(datasets_to_run)
    print("\nDatasets sorted by size:")
    for dataset in datasets_to_run:
        print(f"  {dataset:<20} {format_bytes(dataset_sizes[dataset])}")

    gpus = normalize_gpus(args.gpus, args.gpu)
    threads_per_task = max(1, args.threads_per_task)
    combination_limit = args.verify_max_combinations
    if combination_limit is None:
        combination_limit = args.max_combinations

    cpu_count = os.cpu_count() or 1
    gpu_cap = max(1, len(gpus) * max(1, args.gpu_slots_per_gpu))
    cpu_cap = max(1, int(max(1.0, cpu_count - get_load_average()) // threads_per_task))
    min_estimated_memory = min(
        estimate_task_memory_mb(dataset_sizes[d], args.min_task_memory_mb, args.dataset_memory_multiplier)
        for d in datasets_to_run
    ) if datasets_to_run else args.min_task_memory_mb
    mem_available_mb = read_mem_available_mb()
    if mem_available_mb is None:
        mem_cap = cpu_cap
    else:
        mem_cap = max(1, int((mem_available_mb * args.memory_fraction) // min_estimated_memory))

    auto_workers = max(1, min(gpu_cap, cpu_cap, mem_cap))
    max_workers = args.max_parallel if args.max_parallel and args.max_parallel > 0 else auto_workers

    scheduler_summary = (
        "\nVerification scheduler: "
        f"max_parallel={max_workers}, threads_per_task={threads_per_task}, "
        f"timeout={args.timeout_per_model}s, load1={get_load_average():.2f}"
    )
    if mem_available_mb is not None:
        scheduler_summary += f", mem_available={mem_available_mb:.0f}MB"
    print(scheduler_summary)

    for seed in args.seeds:
        print(f"\n==================================================")
        print(f"=== Verification Seed: {seed} ===")
        print(f"==================================================\n")

        ensure_sweep_results_dir()
        log_dir = seed_log_dir(seed)
        test_results_path = test_results_file(seed)
        os.makedirs(log_dir, exist_ok=True)

        completed_datasets = set()
        if args.resume_completed:
            completed_datasets = get_completed_datasets_from_results(test_results_path, models_to_run)
            if completed_datasets:
                skipped = [d for d in datasets_to_run if d in completed_datasets]
                print(f"Resuming: skipping completed dataset chunks: {skipped}")

        remaining_datasets = [d for d in datasets_to_run if d not in completed_datasets]

        with open(test_results_path, 'a') as f:
            f.write(f"\nVerification Results Summary - {time.ctime()} (Seed: {seed})\n")
            f.write("=" * 100 + "\n")
            f.write("Datasets are written below as complete chunks when all runnable models finish.\n")
            f.write("Chunk order follows dataset size order used by this run.\n")
            if completed_datasets:
                f.write(f"Resume mode skipped completed datasets: {', '.join([d for d in datasets_to_run if d in completed_datasets])}\n")
            f.write("-" * 100 + "\n")

        tasks = []
        for dataset in remaining_datasets:
            for model_name in models_to_run:
                task_memory = estimate_task_memory_mb(
                    dataset_sizes[dataset],
                    args.min_task_memory_mb,
                    args.dataset_memory_multiplier,
                )
                tasks.append({
                    "model": model_name,
                    "dataset": dataset,
                    "memory_mb": task_memory,
                })

        if not tasks:
            print(f"All requested datasets already have complete chunks for seed {seed}.")
            with open(test_results_path, 'a') as f:
                f.write(f"All requested datasets already complete for seed {seed}.\n")
            continue

        pending = deque(tasks)
        active = {}
        active_memory_mb = 0.0
        gpu_index = 0
        total_count = len(tasks)
        done_count = 0
        started_at = time.time()
        dataset_started_at = {dataset: time.time() for dataset in remaining_datasets}
        dataset_remaining = {dataset: len(models_to_run) for dataset in remaining_datasets}
        dataset_results = {dataset: [] for dataset in remaining_datasets}
        last_progress_at = 0.0
        active_slot_seconds = 0.0
        last_active_sample_at = time.time()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            while pending or active:
                now = time.time()
                active_slot_seconds += len(active) * (now - last_active_sample_at)
                last_active_sample_at = now

                launch_limit = current_parallel_limit(max_workers, threads_per_task)
                launched_any = False

                while pending and len(active) < launch_limit:
                    next_task = pending[0]
                    mem_available_mb = read_mem_available_mb()
                    mem_limit_mb = None
                    if mem_available_mb is not None:
                        mem_limit_mb = mem_available_mb * args.memory_fraction
                    if active and mem_limit_mb is not None and active_memory_mb + next_task["memory_mb"] > mem_limit_mb:
                        break

                    task = pending.popleft()
                    gpu = gpus[gpu_index % len(gpus)] if gpus else None
                    gpu_index += 1
                    future = executor.submit(
                        verify_model_dataset,
                        task["model"],
                        task["dataset"],
                        log_dir,
                        gpu,
                        seed,
                        args.timeout_per_model,
                        threads_per_task,
                        combination_limit,
                    )
                    active[future] = {
                        "model": task["model"],
                        "dataset": task["dataset"],
                        "start_time": time.time(),
                        "memory_mb": task["memory_mb"],
                    }
                    active_memory_mb += task["memory_mb"]
                    launched_any = True

                now = time.time()
                if launched_any and now - last_progress_at > 5:
                    print_progress(done_count, total_count, started_at, active)
                    last_progress_at = now

                if not active:
                    time.sleep(1)
                    continue

                done_futures, _ = wait(active.keys(), timeout=1, return_when=FIRST_COMPLETED)
                if not done_futures:
                    if time.time() - last_progress_at >= 30:
                        print_progress(done_count, total_count, started_at, active)
                        last_progress_at = time.time()
                    continue

                for future in done_futures:
                    meta = active.pop(future)
                    active_memory_mb -= meta["memory_mb"]
                    try:
                        result = future.result()
                    except Exception as e:
                        result = {
                            "model": meta["model"],
                            "dataset": meta["dataset"],
                            "status": "ERROR",
                            "metrics": f"ERROR: {e}",
                            "params": None,
                            "duration": time.time() - meta["start_time"],
                            "attempts": 0,
                            "log_file": None,
                            "gpu": None,
                        }

                    done_count += 1
                    dataset = result["dataset"]
                    dataset_results[dataset].append(result)
                    dataset_remaining[dataset] -= 1

                    result_line = format_verification_result_line(result)
                    print(result_line)

                    print_progress(done_count, total_count, started_at, active, result)
                    last_progress_at = time.time()

                    if dataset_remaining[dataset] == 0:
                        report_dataset_complete(
                            dataset,
                            dataset_results[dataset],
                            dataset_started_at[dataset],
                            dataset_sizes[dataset],
                        )
                        write_dataset_chunk(
                            test_results_path,
                            seed,
                            dataset,
                            dataset_results[dataset],
                            dataset_started_at[dataset],
                            dataset_sizes[dataset],
                        )

        now = time.time()
        active_slot_seconds += len(active) * (now - last_active_sample_at)
        elapsed = max(1e-9, now - started_at)
        avg_active = active_slot_seconds / elapsed
        print(
            f"Verification for seed {seed} finished. "
            f"Average active programs: {avg_active:.2f}. "
            f"Results saved to {test_results_path}"
        )
        with open(test_results_path, 'a') as f:
            f.write(f"\nSeed {seed} average active programs: {avg_active:.2f}\n")

def main():
    parser = argparse.ArgumentParser(description="Run multiple recommendation models with parameter search.")
    
    parser.add_argument("model", nargs="?", help="Specific model to run.")
    parser.add_argument("--models", nargs="+", help="Models to run.")
    parser.add_argument("--datasets", nargs="+", help="Datasets to run on.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use.")
    parser.add_argument("--gpus", nargs="+", type=int, default=[0], help="GPU IDs to use for round-robin parallel runs.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[2020], help="List of seeds to run.")
    parser.add_argument("--search", action="store_true", help="Perform parameter search using a validation split, then test.")
    parser.add_argument("--test_from_results", action="store_true", help="Parse the results text file to find the best validation parameters and run the final test")
    parser.add_argument("--max_combinations", type=int, default=None, help="Limit parameter combinations per model for smoke tests.")
    parser.add_argument("--timeout_per_model", type=int, default=300, help="Seconds before a model/dataset run is skipped.")
    parser.add_argument("--max_parallel", type=int, default=0, help="Maximum concurrent programs. 0 lets the scheduler choose from CPU, memory, load, and GPU slots.")
    parser.add_argument("--threads_per_task", type=int, default=DEFAULT_THREADS_PER_TASK, help="BLAS/OpenMP threads assigned to each subprocess in verification mode.")
    parser.add_argument("--gpu_slots_per_gpu", type=int, default=2, help="Maximum concurrent verification tasks to schedule per listed GPU.")
    parser.add_argument("--search_slots_per_gpu", type=int, default=1, help="Maximum concurrent search trials to schedule per listed GPU.")
    parser.add_argument("--min_task_memory_mb", type=int, default=1024, help="Minimum memory reservation per scheduled verification task.")
    parser.add_argument("--dataset_memory_multiplier", type=float, default=30.0, help="Estimated memory multiplier applied to each dataset size.")
    parser.add_argument("--memory_fraction", type=float, default=0.75, help="Fraction of currently available memory the scheduler may reserve.")
    parser.add_argument("--verify_max_combinations", type=int, default=None, help="Limit hyperparameter attempts per model/dataset in verification mode.")
    parser.add_argument("--resume_completed", action="store_true", help="Skip datasets that already have complete verification chunks in the test results file.")
    
    args = parser.parse_args()
    args.gpus = normalize_gpus(args.gpus, args.gpu)
    
    missing_seeds = []
    for seed in args.seeds:
        if not seed_dataset_exists(seed):
            missing_seeds.append(seed)

    if missing_seeds:
        print(
            "Error: The following seed split folders are not found in "
            f"data/split_dataset: {missing_seeds}. Run data/split_dataset.py for each seed."
        )
        sys.exit(1)
        
    if args.models:
        models_to_run = args.models
    elif args.model:
        models_to_run = [args.model]
    else:
        models_to_run = list(MODELS.keys())
        
    datasets_to_run = args.datasets if args.datasets else DATASETS
    datasets_to_run, dataset_sizes = get_sorted_datasets(datasets_to_run)

    missing_dataset_splits = [
        (seed, dataset)
        for seed in args.seeds
        for dataset in datasets_to_run
        if not seed_dataset_exists(seed, dataset)
    ]
    if missing_dataset_splits:
        print("Error: Missing train.txt/val.txt/test.txt for requested seed datasets:")
        for seed, dataset in missing_dataset_splits:
            print(f"  split_dataset/{SPLIT_DATA_ROOT_PREFIX}_{seed}/{dataset}")
        print("Run data/split_dataset.py for the missing seed/dataset pairs.")
        sys.exit(1)
    
    print(f"Starting experiments at {time.ctime()}")
    print(f"Search mode: {args.search}")
    print(f"Seeds: {args.seeds}")
    print("Datasets sorted by size:")
    for dataset in datasets_to_run:
        print(f"  {dataset:<20} {format_bytes(dataset_sizes[dataset])}")
    print("-" * 50)
    
    # Loop over all seeds sequentially
    if args.test_from_results:
        print(f"=== [TEST FROM RESULTS MODE] Parsing results file for best parameters ===")
        import ast
        
        for seed in args.seeds:
            ensure_sweep_results_dir()
            results_file = val_results_file(seed)
            log_dir = seed_log_dir(seed)
            test_results_path = test_results_file(seed)
            final_eval_env = test_env(seed)
            if not os.path.exists(results_file):
                print(f"No results file found for seed {seed}")
                continue

            started_at = time.time()
            pending_model, pending_dataset, pending_params = first_pending_test(
                test_results_path,
                models_to_run,
                datasets_to_run,
            )
            run_history_line = format_run_history_line(
                started_at,
                pending_model,
                pending_dataset,
                pending_params,
                include_params=False,
            )
            normalize_results_file(
                test_results_path,
                "Final Test Results Summary",
                seed,
                "test_from_results",
                started_at,
                models_to_run[0] if models_to_run else None,
                datasets_to_run[0] if datasets_to_run else None,
                "Test",
                models_to_run,
                datasets_to_run,
                None,
                run_history_line,
            )
                
            best_params_parsed = {m: {d: {'best_recall': -1.0, 'best_params': None} for d in datasets_to_run} for m in models_to_run}
            with open(results_file, 'r') as f:
                for line in f:
                    if "(Val)" not in line or "ERROR" in line:
                        continue
                    parts = [p.strip() for p in line.split('|')]
                    if len(parts) >= 4:
                        m_name = parts[0]
                        d_name = parts[1]
                        if parts[2].startswith("(Val)"):
                            metrics_str = parts[2].replace("(Val)", "", 1).strip()
                            param_str = parts[4] if len(parts) >= 5 else ""
                        else:
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
            
            for dataset in datasets_to_run:
                print(f"\n--- Testing on {dataset} (Seed {seed}) ---")
                for model_name in models_to_run:
                    best_params = best_params_parsed[model_name][dataset]['best_params']
                    if best_params is not None:
                        print(f"  -> Best validation parameters for {model_name}: {best_params} (Recall: {best_params_parsed[model_name][dataset]['best_recall']})")
                        detail = run_experiment(
                            model_name, dataset, log_dir, args.gpus[0], best_params, seed,
                            timeout=args.timeout_per_model,
                            return_details=True,
                            extra_env=final_eval_env,
                        )
                        result_line = format_search_result_line(
                            model_name,
                            dataset,
                            best_params,
                            detail["metrics"],
                            detail["duration"],
                            "Test",
                        )
                        append_result_line(test_results_path, result_line, dataset)
                    else:
                        print(f"  -> No valid validation results found for {model_name} on {dataset}.")
        print(f"Test complete. Results appended to results file.")
        sys.exit(0)

    if not args.search:
        run_verification_sweep(args, models_to_run, datasets_to_run)
        sys.exit(0)

    for seed in args.seeds:
        all_best_params = {m: {} for m in models_to_run}
        all_test_results = {m: {} for m in models_to_run}
        
        print(f"\n==================================================")
        print(f"=== Processing Seed: {seed} ===")
        print(f"==================================================\n")
        
        ensure_sweep_results_dir()
        log_dir = seed_log_dir(seed)
        results_file = val_results_file(seed)
        test_results_path = test_results_file(seed)
        final_eval_env = test_env(seed)
        os.makedirs(log_dir, exist_ok=True)

        started_at = time.time()
        start_model = models_to_run[0] if models_to_run else None
        start_dataset = datasets_to_run[0] if datasets_to_run else None
        pending_model, pending_dataset, pending_params = first_pending_param(
            results_file,
            models_to_run,
            datasets_to_run,
        )
        search_run_history_line = format_run_history_line(
            started_at,
            pending_model,
            pending_dataset,
            pending_params,
        )
        pending_model, pending_dataset, pending_params = first_pending_test(
            test_results_path,
            models_to_run,
            datasets_to_run,
        )
        test_run_history_line = format_run_history_line(
            started_at,
            pending_model,
            pending_dataset,
            pending_params,
            include_params=False,
        )
        normalize_results_file(
            results_file,
            "Validation Search Results Summary",
            seed,
            "search" if args.search else "test",
            started_at,
            start_model,
            start_dataset,
            "Val" if args.search else "Test",
            models_to_run,
            datasets_to_run,
            None,
            search_run_history_line,
        )
        normalize_results_file(
            test_results_path,
            "Final Test Results Summary",
            seed,
            "search-final-test" if args.search else "test",
            started_at,
            start_model,
            start_dataset,
            "Test",
            models_to_run,
            datasets_to_run,
            None,
            test_run_history_line,
        )
        
        try:
            for dataset in datasets_to_run:
                current_search = args.search
                search_env = None
                best_params_dict = {}

                try:
                    if current_search:
                        search_env = validation_env(seed, dataset)
                        if search_env is None:
                            print(
                                f"Error: train.txt/val.txt for {dataset} not found in "
                                f"split_dataset/{SPLIT_DATA_ROOT_PREFIX}_{seed}. Skipping search for this dataset."
                            )
                            current_search = False
                        else:
                            print(
                                f"  -> [SEARCH MODE] Using {seed_dataset_dir(seed, dataset)} "
                                f"with TFRECSYS_EVAL_FILE=val.txt"
                            )

                    for model_name in models_to_run:
                        print(f"Model: {model_name} | Dataset: {dataset}")

                        if current_search:
                            all_params = list(iter_param_dicts(model_name, dataset, limit=None))
                            if all_params == [None]:
                                all_params = []
                            completed_keys = existing_param_keys(results_file, "Val", model_name, dataset)
                            completed_count = sum(
                                1 for params in all_params
                                if format_params(params) in completed_keys
                            )
                            params_to_run = [
                                params for params in all_params
                                if format_params(params) not in completed_keys
                            ]
                            if args.max_combinations is not None:
                                params_to_run = params_to_run[:args.max_combinations]
                            if completed_count:
                                print(
                                    f"  -> Resume: skipping {completed_count} completed validation "
                                    f"runs for {model_name} on {dataset}."
                                )

                            best_params, best_recall = best_existing_params(
                                results_file,
                                model_name,
                                dataset,
                            )

                            if not params_to_run:
                                best_params_dict[model_name] = best_params
                                if best_params is not None:
                                    all_best_params[model_name][dataset] = str(best_params)
                                else:
                                    all_best_params[model_name][dataset] = "N/A"
                                print(
                                    f"  -> Resume: no remaining validation parameters for "
                                    f"{model_name} on {dataset}. Best existing parameters: "
                                    f"{best_params} (Recall@20: {best_recall:.4f})"
                                )
                                continue

                            first_params = params_to_run[0]
                            detail = run_experiment(
                                model_name, dataset, log_dir, args.gpus[0], first_params, seed,
                                timeout=args.timeout_per_model,
                                return_details=True,
                                extra_env=search_env,
                            )
                            metrics_str = detail["metrics"]
                            result_line = format_search_result_line(
                                model_name,
                                dataset,
                                first_params,
                                metrics_str,
                                detail["duration"],
                                "Val",
                            )
                            append_result_line(results_file, result_line, dataset)
                            recall = get_recall_value(metrics_str)
                            if recall > best_recall:
                                best_recall = recall
                                best_params = first_params

                            remaining_params = params_to_run[1:]
                            if remaining_params:
                                from concurrent.futures import ThreadPoolExecutor

                                search_slots = max(1, args.search_slots_per_gpu)
                                slot_gpus = [
                                    gpu
                                    for gpu in args.gpus
                                    for _ in range(search_slots)
                                ] or [args.gpu]
                                max_workers = len(slot_gpus)

                                def worker(params, gpu):
                                    return run_experiment(
                                        model_name, dataset, log_dir, gpu, params, seed,
                                        timeout=args.timeout_per_model,
                                        return_details=True,
                                        extra_env=search_env,
                                    )

                                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                                    active = {}
                                    completed = {}
                                    next_submit = 0
                                    next_write = 0
                                    next_slot = 0

                                    def submit_next():
                                        nonlocal next_submit, next_slot
                                        if next_submit >= len(remaining_params):
                                            return
                                        params = remaining_params[next_submit]
                                        gpu = slot_gpus[next_slot % len(slot_gpus)]
                                        next_slot += 1
                                        future = executor.submit(worker, params, gpu)
                                        active[future] = {
                                            "order": next_submit,
                                            "params": params,
                                            "start_time": time.time(),
                                        }
                                        next_submit += 1

                                    def flush_ready_results():
                                        nonlocal next_write, best_recall, best_params
                                        while next_write in completed:
                                            params, detail = completed.pop(next_write)
                                            metrics_str = detail["metrics"]
                                            result_line = format_search_result_line(
                                                model_name,
                                                dataset,
                                                params,
                                                metrics_str,
                                                detail["duration"],
                                                "Val",
                                            )
                                            append_result_line(results_file, result_line, dataset)

                                            recall = get_recall_value(metrics_str)
                                            if recall > best_recall:
                                                best_recall = recall
                                                best_params = params
                                            next_write += 1

                                    for _ in range(min(max_workers, len(remaining_params))):
                                        submit_next()

                                    while active:
                                        done_futures, _ = wait(
                                            active.keys(),
                                            return_when=FIRST_COMPLETED,
                                        )
                                        for future in done_futures:
                                            meta = active.pop(future)
                                            params = meta["params"]
                                            try:
                                                detail = future.result()
                                            except Exception as e:
                                                print(f"Exception in worker thread: {e}")
                                                detail = {
                                                    "metrics": "ERROR",
                                                    "status": "ERROR",
                                                    "duration": time.time() - meta["start_time"],
                                                    "log_file": None,
                                                }
                                            completed[meta["order"]] = (params, detail)
                                            submit_next()
                                        flush_ready_results()

                            best_params_dict[model_name] = best_params
                            if best_params is not None:
                                all_best_params[model_name][dataset] = str(best_params)
                            else:
                                all_best_params[model_name][dataset] = "N/A"
                            print(f"  -> Best validation parameters for {model_name} on {dataset}: {best_params} (Recall@20: {best_recall:.4f})")

                        elif not current_search:
                            detail = run_experiment(
                                model_name, dataset, log_dir, args.gpu, None, seed,
                                timeout=args.timeout_per_model,
                                return_details=True,
                                extra_env=final_eval_env,
                            )
                            metrics_str = detail["metrics"]
                            result_line = format_search_result_line(
                                model_name,
                                dataset,
                                None,
                                metrics_str,
                                detail["duration"],
                                "Test",
                            )
                            append_result_line(test_results_path, result_line, dataset)
                            all_test_results[model_name][dataset] = get_recall_value(metrics_str)
                finally:
                    pass

                if current_search:
                    print(f"=== [FINAL TEST MODE] Running models on {dataset} with best found parameters ===")
                    for model_name, best_params in best_params_dict.items():
                        if best_params is not None:
                            if has_test_result(test_results_path, model_name, dataset):
                                print(
                                    f"  -> Resume: skipping existing final test for "
                                    f"{model_name} on {dataset}."
                                )
                                continue
                            detail = run_experiment(
                                model_name, dataset, log_dir, args.gpu, best_params, seed,
                                timeout=args.timeout_per_model,
                                return_details=True,
                                extra_env=final_eval_env,
                            )
                            metrics_str = detail["metrics"]
                            result_line = format_search_result_line(
                                model_name,
                                dataset,
                                best_params,
                                metrics_str,
                                detail["duration"],
                                "Test",
                            )
                            append_result_line(test_results_path, result_line, dataset)
                            all_test_results[model_name][dataset] = get_recall_value(metrics_str)

                print("-" * 50)
                
        except (KeyboardInterrupt, SystemExit):
            print(f"\n[!] Execution interrupted. Partial results for seed {seed} are saved to {results_file}.")
            sys.exit(1)

        print(f"All experiments for seed {seed} finished. Search results saved to {results_file}, Final Test results saved to {test_results_path}")

if __name__ == "__main__":
    main()
