import os
import sys
import queue
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from run_sweep import MODELS, DATASETS, run_experiment, BASE_DIR

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", nargs="+", type=int, default=list(range(8)))
    parser.add_argument("--seed", type=int, default=2020)
    args = parser.parse_args()

    models_to_run = [m for m in MODELS.keys() if 'cmd' in MODELS[m]]
    seed = args.seed

    log_dir = os.path.join(BASE_DIR, f"00_exp_logs_seed_{seed}")
    test_results_file = os.path.join(BASE_DIR, f"00_test_results_seed_{seed}.txt")
    os.makedirs(log_dir, exist_ok=True)

    with open(test_results_file, 'a') as f:
        f.write(f"\nFinal Test Results Summary - {time.ctime()} (Seed: {seed})\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'Model':<10} | {'Dataset':<20} | {'Parameters':<30} | Metrics\n")
        f.write("-" * 80 + "\n")

    gpu_queue = queue.Queue()
    for g in args.gpus:
        for _ in range(1): # 1 task per GPU to be safe
            gpu_queue.put(g)

    def worker(task):
        model_name, dataset = task
        gpu = gpu_queue.get()
        try:
            metrics_str = run_experiment(model_name, dataset, log_dir, gpu, None, seed)
            return model_name, dataset, metrics_str, False
        except Exception as e:
            return model_name, dataset, "ERROR", True
        finally:
            gpu_queue.put(gpu)
            gpu_queue.task_done()

    completed_pairs = set()
    if os.path.exists(test_results_file):
        with open(test_results_file, 'r') as f:
            for line in f:
                if ' | ' in line and 'ERROR' not in line and 'Metrics' not in line:
                    parts = line.strip().split('|')
                    if len(parts) >= 2:
                        completed_pairs.add((parts[0].strip(), parts[1].strip()))

    # Iterate dataset by dataset
    for dataset in DATASETS:
        print(f"\n--- Testing Dataset: {dataset} ---")
        tasks = [(m, dataset) for m in models_to_run if (m, dataset) not in completed_pairs]
        if not tasks:
            print("All models already completed for this dataset. Skipping.")
            continue
        
        has_error = False
        with ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
            futures = [executor.submit(worker, task) for task in tasks]
            for future in as_completed(futures):
                model_name, ds, metrics_str, err = future.result()
                result_line = f"{model_name:<10} | {ds:<20} | {'default':<30} | {metrics_str}"
                print(result_line)
                with open(test_results_file, 'a') as f:
                    f.write(result_line + "\n")
                if err or "ERROR" in metrics_str:
                    has_error = True

        if has_error:
            print(f"\nAn ERROR occurred while testing dataset {dataset}. Terminating code so you can check and resolve.")
            sys.exit(1)

if __name__ == '__main__':
    main()
