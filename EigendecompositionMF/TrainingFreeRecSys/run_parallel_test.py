import os
import time
import argparse
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from run_sweep import MODELS, DATASETS, run_experiment, get_recall_value, BASE_DIR

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", nargs="+", type=int, default=list(range(8)))
    parser.add_argument("--seeds", nargs="+", type=int, default=[2020])
    args = parser.parse_args()

    models_to_run = list(MODELS.keys())
    
    # Remove PGSP if it has no cmd
    models_to_run = [m for m in models_to_run if 'cmd' in MODELS[m]]
    
    datasets_to_run = DATASETS
    print(f"Running models: {models_to_run}")
    print(f"Running datasets: {datasets_to_run}")

    for seed in args.seeds:
        log_dir = os.path.join(BASE_DIR, f"00_exp_logs_seed_{seed}")
        test_results_file = os.path.join(BASE_DIR, f"00_test_results_seed_{seed}.txt")
        os.makedirs(log_dir, exist_ok=True)

        with open(test_results_file, 'a') as f:
            f.write(f"\nFinal Test Results Summary - {time.ctime()} (Seed: {seed})\n")
            f.write("=" * 80 + "\n")
            f.write(f"{'Model':<10} | {'Dataset':<20} | {'Parameters':<30} | Metrics\n")
            f.write("-" * 80 + "\n")

        tasks = []
        for dataset in datasets_to_run:
            for model_name in models_to_run:
                tasks.append((model_name, dataset))

        gpu_queue = queue.Queue()
        # Add each GPU multiple times to the queue if we want concurrent runs on the same GPU, 
        # but for default tests, it's safer to have 1 or 2 tasks per GPU
        for g in args.gpus:
            for _ in range(2):
                gpu_queue.put(g)

        print(f"Starting {len(tasks)} tasks on {len(args.gpus)} GPUs")

        def worker(task):
            model_name, dataset = task
            gpu = gpu_queue.get()
            try:
                metrics_str = run_experiment(model_name, dataset, log_dir, gpu, None, seed)
                return model_name, dataset, metrics_str
            except Exception as e:
                return model_name, dataset, "ERROR"
            finally:
                gpu_queue.put(gpu)
                gpu_queue.task_done()

        with ThreadPoolExecutor(max_workers=len(args.gpus) * 2) as executor:
            futures = [executor.submit(worker, task) for task in tasks]
            for future in as_completed(futures):
                model_name, dataset, metrics_str = future.result()
                result_line = f"{model_name:<10} | {dataset:<20} | {'default':<30} | {metrics_str}"
                print(result_line)
                with open(test_results_file, 'a') as f:
                    f.write(result_line + "\n")

if __name__ == '__main__':
    main()
