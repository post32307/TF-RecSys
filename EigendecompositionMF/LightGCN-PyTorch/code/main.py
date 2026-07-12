import os
import sys
import json
import subprocess
import itertools
import re
import time
from os.path import join

import torch
from tensorboardX import SummaryWriter

import Procedure
import dataloader
import model
import utils
import world
from world import cprint


Neg_k = 1
DATASET_COL_WIDTH = 20
MODELS = {
    'mf': model.PureMF,
    'lgn': model.LightGCN
}


def build_dataset(eval_file):
    return dataloader.Loader(
        path=join(world.SPLIT_PATH, world.dataset),
        eval_file=eval_file
    )


def configure_params(recdim, lr, decay, layer):
    world.config['latent_dim_rec'] = recdim
    world.config['lr'] = lr
    world.config['decay'] = decay
    world.config['lightGCN_n_layers'] = layer


def build_training_objects(dataset):
    utils.set_seed(world.seed)
    recmodel = MODELS[world.model_name](world.config, dataset)
    recmodel = recmodel.to(world.device)
    bpr = utils.BPRLoss(recmodel, world.config)
    return recmodel, bpr


def maybe_load_weights(recmodel, weight_file):
    if not world.LOAD or weight_file is None:
        return
    try:
        recmodel.load_state_dict(torch.load(weight_file, map_location=torch.device('cpu')))
        world.cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")


def snapshot_state(recmodel):
    return {key: value.detach().cpu().clone() for key, value in recmodel.state_dict().items()}


def restore_state(recmodel, state):
    if state is not None:
        recmodel.load_state_dict(state)


def train_model(dataset, validation_dataset, recmodel, bpr, writer, save_weight_file=None):
    best_score = None
    best_epoch = None
    best_state = None
    best_results = None
    stale_checks = 0
    completed_epochs = 0

    for epoch in range(world.TRAIN_epochs):
        completed_epochs = epoch + 1
        output_information = Procedure.BPR_train_original(
            dataset, recmodel, bpr, epoch, neg_k=Neg_k, w=writer
        )
        print(f'EPOCH[{epoch + 1}/{world.TRAIN_epochs}] {output_information}')

        should_validate = (epoch + 1) % world.EARLY_STOP_INTERVAL == 0
        should_validate = should_validate or (epoch + 1) == world.TRAIN_epochs
        if world.EARLY_STOP and should_validate:
            cprint("[VALIDATION:val.txt]")
            validation_results = Procedure.Test(
                validation_dataset, recmodel, epoch + 1, writer, world.config['multicore']
            )
            validation_score = score_results(validation_results)
            improved = best_score is None or validation_score > best_score + world.EARLY_STOP_DELTA
            if improved:
                best_score = validation_score
                best_epoch = epoch + 1
                best_results = validation_results
                best_state = snapshot_state(recmodel)
                stale_checks = 0
                print(f"[EARLY STOP] improved Recall@{world.topks[0]}={best_score:.6f} at epoch {best_epoch}")
            else:
                if epoch + 1 >= world.EARLY_STOP_MIN_EPOCHS:
                    stale_checks += 1
                print(
                    f"[EARLY STOP] no improvement: recall={validation_score:.6f}, "
                    f"best={best_score:.6f}, stale={stale_checks}/{world.EARLY_STOP_PATIENCE}"
                )
                if stale_checks >= world.EARLY_STOP_PATIENCE:
                    print(
                        f"[EARLY STOP] stopping at epoch {epoch + 1}; "
                        f"best validation Recall@{world.topks[0]}={best_score:.6f} at epoch {best_epoch}"
                    )
                    break

        if save_weight_file is not None:
            torch.save(recmodel.state_dict(), save_weight_file)

    restore_state(recmodel, best_state)
    if save_weight_file is not None:
        torch.save(recmodel.state_dict(), save_weight_file)
    return best_results, completed_epochs, best_epoch


def train_and_eval(params, test_eval_file, writer, save_weight_file=None):
    run_start = time.time()
    recdim, lr, decay, layer = params
    configure_params(recdim, lr, decay, layer)
    if save_weight_file == "auto":
        save_weight_file = utils.getFileName()
    train_dataset = build_dataset("val.txt")
    validation_dataset = build_dataset("val.txt")
    test_dataset = build_dataset(test_eval_file)
    recmodel, bpr = build_training_objects(train_dataset)
    maybe_load_weights(recmodel, save_weight_file)
    validation_results, completed_epochs, best_epoch = train_model(
        train_dataset, validation_dataset, recmodel, bpr, writer, save_weight_file
    )
    if validation_results is None:
        cprint("[VALIDATION:val.txt]")
        validation_results = Procedure.Test(
            validation_dataset, recmodel, world.TRAIN_epochs, writer, world.config['multicore']
        )
        best_epoch = completed_epochs
    runtime = time.time() - run_start
    if test_eval_file == "val.txt":
        return validation_results, validation_results, completed_epochs, best_epoch, runtime
    cprint(f"[TEST:{test_eval_file}]")
    test_results = Procedure.Test(test_dataset, recmodel, world.TRAIN_epochs, writer, world.config['multicore'])
    runtime = time.time() - run_start
    return validation_results, test_results, completed_epochs, best_epoch, runtime


def score_results(results):
    return float(results['recall'][0])


def format_metric_array(values):
    return "[" + ", ".join(f"{float(value):.6f}" for value in values) + "]"


def format_results(results):
    return (
        f"recall={format_metric_array(results['recall'])}, "
        f"precision={format_metric_array(results['precision'])}, "
        f"ndcg={format_metric_array(results['ndcg'])}"
    )


def results_path():
    return join(world.ROOT_PATH, "results", "main_out.txt")


def validation_results_path():
    return join(world.ROOT_PATH, "results", "validation.txt")


def test_results_path():
    return join(world.ROOT_PATH, "results", "test.txt")


def mode_results_path(mode):
    if mode == "val":
        return validation_results_path()
    if mode == "test":
        return test_results_path()
    return results_path()


def param_key(params):
    return (
        int(params[0]),
        f"{float(params[1]):.12g}",
        f"{float(params[2]):.12g}",
        int(params[3]),
    )


def parse_metric_list(metric_text, name):
    match = re.search(rf"{name}=\[([^\]]*)\]", metric_text)
    if match is None:
        return []
    values = []
    for value in match.group(1).split(","):
        value = value.strip()
        if value:
            values.append(float(value))
    return values


def parse_params_text(params_text):
    match = re.search(
        r"recdim=([^,\)]+),\s*lr=([^,\)]+),\s*decay=([^,\)]+),\s*layer=([^,\)]+)",
        params_text,
    )
    if match is None:
        return None
    return (
        int(match.group(1)),
        float(match.group(2)),
        float(match.group(3)),
        int(match.group(4)),
    )


def parse_result_rows(path):
    rows = []
    if not os.path.exists(path):
        return rows

    with open(path, "r") as f:
        for raw_line in f:
            parts = [part.strip() for part in raw_line.rstrip("\n").split("|")]
            if len(parts) < 7 or parts[0] not in ("(val)", "(test)"):
                continue

            params_part = next((part for part in parts if part.startswith("parameters=")), None)
            seed_part = next((part for part in parts if part.startswith("seed=")), None)
            split_part = next((part for part in parts if part.startswith("split_path=")), "")
            if params_part is None or seed_part is None:
                continue

            params = parse_params_text(params_part)
            if params is None:
                continue

            metrics = {
                "recall": parse_metric_list(parts[3], "recall"),
                "precision": parse_metric_list(parts[3], "precision"),
                "ndcg": parse_metric_list(parts[3], "ndcg"),
            }
            if not metrics["recall"]:
                continue

            rows.append({
                "mode": parts[0].strip("()"),
                "dataset": parts[1],
                "seed": int(seed_part.split("=", 1)[1]),
                "results": metrics,
                "params": params,
                "params_key": param_key(params),
                "split_path": split_part.split("=", 1)[1] if "=" in split_part else "",
                "line": raw_line,
            })
    return rows


def existing_results(mode, dataset, seed, params=None):
    key = param_key(params) if params is not None else None
    return [
        row for row in parse_result_rows(mode_results_path(mode))
        if row["mode"] == mode
        and row["dataset"] == dataset
        and row["seed"] == seed
        and (key is None or row["params_key"] == key)
    ]


def first_existing_result(mode, dataset, seed, params):
    rows = existing_results(mode, dataset, seed, params)
    return rows[-1] if rows else None


def best_existing_validation(dataset, seed, allowed_params=None):
    allowed_keys = None
    if allowed_params is not None:
        allowed_keys = {param_key(params) for params in allowed_params}

    best_row = None
    best_score = None
    for row in existing_results("val", dataset, seed):
        if allowed_keys is not None and row["params_key"] not in allowed_keys:
            continue
        score = score_results(row["results"])
        if best_score is None or score > best_score:
            best_score = score
            best_row = row
    return best_row, best_score


def append_mode_result(mode, line):
    path = mode_results_path(mode)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(line)


def append_result(mode, params, results, epochs, runtime, split_path):
    line = (
        f"({mode}) | "
        f"{world.dataset:<{DATASET_COL_WIDTH}} | "
        f"seed={world.seed} | "
        f"{format_results(results)} | "
        f"epochs={epochs} | "
        f"parameters=(recdim={params[0]}, lr={params[1]}, decay={params[2]}, layer={params[3]}, "
        f"bpr_batch={world.config['bpr_batch_size']}, topks={world.topks}) | "
        f"runtime={runtime:.2f}s | "
        f"split_path={split_path}\n"
    )
    child_result_file = world.args.child_result_file
    if child_result_file:
        os.makedirs(os.path.dirname(child_result_file), exist_ok=True)
        with open(child_result_file, "a") as f:
            f.write(json.dumps({"mode": mode, "line": line}) + "\n")
        return

    path = results_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(line)
    append_mode_result(mode, line)
    print(f"[RESULTS] appended to {path}")


def search_params(writer):
    params_grid = list(itertools.product(
        world.SEARCH_RECDIMS,
        world.SEARCH_LRS,
        world.SEARCH_DECAYS,
        world.SEARCH_LAYERS
    ))
    completed_keys = {
        row["params_key"]
        for row in existing_results("val", world.dataset, world.seed)
    }
    pending_params = [
        params for params in params_grid
        if param_key(params) not in completed_keys
    ]

    skipped_count = len(params_grid) - len(pending_params)
    if skipped_count:
        print(
            f"[SEARCH] resume: skipping {skipped_count} completed validation "
            f"runs for {world.dataset} seed={world.seed}"
        )

    best_row, best_score = best_existing_validation(world.dataset, world.seed, params_grid)
    best_params = best_row["params"] if best_row is not None else None
    best_results = best_row["results"] if best_row is not None else None
    if best_row is not None:
        print(
            f"[SEARCH] resume: best existing params={best_params}, "
            f"Recall@{world.topks[0]}={best_score:.6f}"
        )

    for params in pending_params:
        print(f"\n[SEARCH] params=(recdim={params[0]}, lr={params[1]}, decay={params[2]}, layer={params[3]})")
        validation_results, _, epochs, _, runtime = train_and_eval(params, test_eval_file="val.txt", writer=writer)
        current_score = score_results(validation_results)
        append_result("val", params, validation_results, epochs, runtime, join(world.SPLIT_PATH, world.dataset))
        print(f"[SEARCH] Recall@{world.topks[0]}={current_score:.6f}")
        if best_score is None or current_score > best_score:
            best_score = current_score
            best_params = params
            best_results = validation_results
            print(f"[SEARCH] new best params={best_params}, recall={best_score:.6f}")

    if best_params is None:
        raise RuntimeError(
            f"No validation results found for {world.dataset} seed={world.seed}; "
            "check the search grid and validation output."
        )

    print(f"\n[SEARCH] best params={best_params}, validation results={best_results}")
    return best_params


def expand_list(values):
    expanded = []
    for value in values:
        if isinstance(value, str) and "," in value:
            expanded.extend(part.strip() for part in value.split(",") if part.strip())
        else:
            expanded.append(value)
    return expanded


def scheduler_requested():
    return (
        not world.args.single_run
        and (world.args.datasets is not None or world.args.seeds is not None)
    )


def scheduler_datasets():
    if world.args.datasets is None:
        return [world.dataset]
    return expand_list(world.args.datasets)


def scheduler_seeds():
    if world.args.seeds is None:
        return [world.seed]
    return world.args.seeds


def add_arg(cmd, name, value):
    cmd.extend([f"--{name}", str(value)])


def child_command(dataset, seed, result_file):
    cmd = [sys.executable, os.path.abspath(__file__)]
    args = world.args
    for name in (
        "bpr_batch", "recdim", "layer", "lr", "decay", "dropout", "keepprob",
        "a_fold", "testbatch", "path", "topks", "tensorboard", "comment",
        "load", "epochs", "multicore", "pretrain", "model", "search",
        "search_recdims", "search_lrs", "search_decays", "search_layers",
        "early_stop", "early_stop_patience", "early_stop_delta",
        "early_stop_interval", "early_stop_min_epochs"
    ):
        add_arg(cmd, name, getattr(args, name))
    add_arg(cmd, "dataset", dataset)
    add_arg(cmd, "seed", seed)
    add_arg(cmd, "single_run", 1)
    add_arg(cmd, "child_result_file", result_file)
    return cmd


def append_ordered_child_results(result_file, val_file, test_file, task_seed,
                                 previous_seed=None, error_line=None):
    val_lines = []
    test_lines = []
    if os.path.exists(result_file):
        with open(result_file, "r") as f:
            for raw in f:
                record = json.loads(raw)
                if record["mode"] == "val":
                    val_lines.append(record["line"])
                elif record["mode"] == "test":
                    test_lines.append(record["line"])

    if error_line:
        val_lines.append(error_line)
        test_lines.append(error_line)

    os.makedirs(os.path.dirname(val_file), exist_ok=True)
    add_seed_separator = previous_seed is not None and previous_seed != task_seed
    with open(val_file, "a") as f:
        if add_seed_separator and val_lines:
            f.write("\n")
        for line in val_lines:
            f.write(line)
    with open(test_file, "a") as f:
        if add_seed_separator and test_lines:
            f.write("\n")
        for line in test_lines:
            f.write(line)


def run_scheduler():
    datasets = scheduler_datasets()
    seeds = scheduler_seeds()
    unknown = [dataset for dataset in datasets if dataset not in world.all_dataset]
    if unknown:
        raise NotImplementedError(f"Haven't supported {unknown} yet!, try {world.all_dataset}")

    tasks = []
    for seed in seeds:
        for dataset in datasets:
            split_path = join(world.DATA_PATH, f"00_dataset_seed_{seed}", dataset)
            if not os.path.isdir(split_path):
                raise FileNotFoundError(f"Split dataset not found: {split_path}")
            tasks.append({"seed": seed, "dataset": dataset, "split_path": split_path})

    max_concurrent = max(1, world.args.concurrent_runs)
    results_dir = join(world.ROOT_PATH, "results")
    tmp_dir = join(results_dir, "tmp")
    log_dir = join(results_dir, "logs")
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    val_file = validation_results_path()
    test_file = test_results_path()

    print(
        f"[SCHEDULER] launching {len(tasks)} runs with concurrent_runs={max_concurrent}; "
        f"validation={val_file}, test={test_file}"
    )

    active = []
    next_task = 0
    completed = {}
    next_flush = 0
    last_flushed_seed = None
    started_at = time.time()

    while next_task < len(tasks) or active:
        while next_task < len(tasks) and len(active) < max_concurrent:
            task = tasks[next_task]
            result_file = join(tmp_dir, f"run_{next_task:04d}.jsonl")
            log_file = join(log_dir, f"run_{next_task:04d}_{task['dataset']}_seed_{task['seed']}.log")
            for path in (result_file, log_file):
                if os.path.exists(path):
                    os.remove(path)
            cmd = child_command(task["dataset"], task["seed"], result_file)
            log_handle = open(log_file, "w")
            proc = subprocess.Popen(
                cmd,
                cwd=world.CODE_PATH,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            print(
                f"[SCHEDULER] started #{next_task}: {task['dataset']} seed={task['seed']} "
                f"log={log_file}"
            )
            active.append({
                "index": next_task,
                "proc": proc,
                "log_handle": log_handle,
                "result_file": result_file,
                "log_file": log_file,
                "task": task,
                "start": time.time(),
            })
            next_task += 1

        still_active = []
        for run in active:
            returncode = run["proc"].poll()
            if returncode is None:
                still_active.append(run)
                continue
            run["log_handle"].close()
            run["returncode"] = returncode
            run["runtime"] = time.time() - run["start"]
            completed[run["index"]] = run
            status = "OK" if returncode == 0 else f"FAILED({returncode})"
            print(
                f"[SCHEDULER] finished #{run['index']}: {run['task']['dataset']} "
                f"seed={run['task']['seed']} {status} runtime={run['runtime']:.2f}s"
            )
        active = still_active

        while next_flush in completed:
            run = completed.pop(next_flush)
            error_line = None
            if run["returncode"] != 0:
                task = run["task"]
                error_line = (
                    f"(error) | {task['dataset']:<{DATASET_COL_WIDTH}} | seed={task['seed']} | "
                    f"runtime={run['runtime']:.2f}s | split_path={task['split_path']} | "
                    f"log={run['log_file']}\n"
                )
            append_ordered_child_results(
                run["result_file"],
                val_file,
                test_file,
                task_seed=run["task"]["seed"],
                previous_seed=last_flushed_seed,
                error_line=error_line,
            )
            last_flushed_seed = run["task"]["seed"]
            if os.path.exists(run["result_file"]):
                os.remove(run["result_file"])
            print(f"[SCHEDULER] flushed #{next_flush} results in initiation order")
            next_flush += 1

        if active:
            time.sleep(1)

    for path in (val_file, test_file):
        with open(path, "a") as f:
            f.write("\n")

    print(f"[SCHEDULER] all runs complete in {time.time() - started_at:.2f}s")


def main():
    if scheduler_requested():
        run_scheduler()
        return

    utils.set_seed(world.seed)
    print(">>SEED:", world.seed)
    print(">>SPLIT:", join(world.SPLIT_PATH, world.dataset))

    if world.tensorboard:
        run_name = (
            f"{time.strftime('%m-%d-%Hh%Mm%Ss')}-"
            f"{world.dataset}-seed{world.seed}-pid{os.getpid()}-{world.comment}"
        )
        writer = SummaryWriter(
            join(world.BOARD_PATH, run_name)
        )
    else:
        writer = None
        world.cprint("not enable tensorflowboard")

    try:
        if world.search:
            best_params = search_params(writer)
            existing_test = first_existing_result("test", world.dataset, world.seed, best_params)
            if existing_test is not None:
                final_results = existing_test["results"]
                print(
                    f"\n[FINAL] resume: skipping existing test for {world.dataset} "
                    f"seed={world.seed} params={best_params}"
                )
            else:
                print(f"\n[FINAL] retraining and testing best params={best_params}")
                validation_results, final_results, epochs, best_epoch, runtime = train_and_eval(
                    best_params,
                    test_eval_file="test.txt",
                    writer=writer,
                    save_weight_file="auto"
                )
                logged_epochs = best_epoch if best_epoch is not None else epochs
                split_path = join(world.SPLIT_PATH, world.dataset)
                if first_existing_result("val", world.dataset, world.seed, best_params) is None:
                    append_result("val", best_params, validation_results, logged_epochs, runtime, split_path)
                append_result("test", best_params, final_results, logged_epochs, runtime, split_path)
        else:
            params = (
                world.config['latent_dim_rec'],
                world.config['lr'],
                world.config['decay'],
                world.config['lightGCN_n_layers']
            )
            existing_val = first_existing_result("val", world.dataset, world.seed, params)
            existing_test = first_existing_result("test", world.dataset, world.seed, params)
            if existing_val is not None and existing_test is not None:
                final_results = existing_test["results"]
                print(
                    f"[FINAL] resume: skipping existing validation and test for "
                    f"{world.dataset} seed={world.seed} params={params}"
                )
            else:
                print(f"[FINAL] testing split with params={params}")
                validation_results, final_results, epochs, best_epoch, runtime = train_and_eval(
                    params,
                    test_eval_file="test.txt",
                    writer=writer,
                    save_weight_file="auto"
                )
                logged_epochs = best_epoch if best_epoch is not None else epochs
                split_path = join(world.SPLIT_PATH, world.dataset)
                if existing_val is None:
                    append_result("val", params, validation_results, logged_epochs, runtime, split_path)
                if existing_test is None:
                    append_result("test", params, final_results, logged_epochs, runtime, split_path)
        print(f"[FINAL] results={final_results}\n")
    finally:
        if world.tensorboard and writer is not None:
            writer.close()


if __name__ == "__main__":
    main()
