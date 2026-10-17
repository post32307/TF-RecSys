import os
import sys
import json
import subprocess
import itertools
import re
import time
from os.path import join

import torch
try:
    from tensorboardX import SummaryWriter
except ModuleNotFoundError:
    class SummaryWriter:
        """No-op fallback when optional TensorBoard logging is unavailable."""

        def __init__(self, *args, **kwargs):
            pass

        def add_scalar(self, *args, **kwargs):
            pass

        def add_scalars(self, *args, **kwargs):
            pass

        def close(self):
            pass

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


def format_output_path(path):
    return os.path.relpath(path, world.ROOT_PATH)


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


def tmp_results_dir():
    return join(world.ROOT_PATH, "results", "tmp")


def tmp_result_paths():
    paths = []
    tmp_dir = tmp_results_dir()
    if os.path.isdir(tmp_dir):
        paths.extend(
            join(tmp_dir, name)
            for name in sorted(os.listdir(tmp_dir))
            if name.endswith(".jsonl")
        )
    if world.args.child_result_file:
        paths.append(world.args.child_result_file)
    return sorted(set(paths))


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


def parse_result_line(raw_line):
    parts = [part.strip() for part in raw_line.rstrip("\n").split("|")]
    if len(parts) < 7 or parts[0] not in ("(val)", "(test)"):
        return None

    params_part = next((part for part in parts if part.startswith("parameters=")), None)
    seed_part = next((part for part in parts if part.startswith("seed=")), None)
    split_part = next((part for part in parts if part.startswith("split_path=")), "")
    if params_part is None or seed_part is None:
        return None

    params = parse_params_text(params_part)
    if params is None:
        return None

    metrics = {
        "recall": parse_metric_list(parts[3], "recall"),
        "precision": parse_metric_list(parts[3], "precision"),
        "ndcg": parse_metric_list(parts[3], "ndcg"),
    }
    if not metrics["recall"]:
        return None

    return {
        "mode": parts[0].strip("()"),
        "dataset": parts[1],
        "seed": int(seed_part.split("=", 1)[1]),
        "results": metrics,
        "params": params,
        "params_key": param_key(params),
        "epochs": int(next((part.split("=", 1)[1] for part in parts if part.startswith("epochs=")), "0")),
        "split_path": split_part.split("=", 1)[1] if "=" in split_part else "",
        "line": raw_line,
    }


def parse_result_rows(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r") as f:
        for raw_line in f:
            row = parse_result_line(raw_line)
            if row is not None:
                rows.append(row)
    return rows


def parse_json_result_rows(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r") as f:
        for raw in f:
            try:
                record = json.loads(raw)
            except json.JSONDecodeError:
                continue
            line = record.get("line")
            if not line:
                continue
            row = parse_result_line(line)
            if row is not None:
                rows.append(row)
    return rows


def available_result_rows(mode, include_tmp=True):
    rows = parse_result_rows(mode_results_path(mode))
    if include_tmp:
        for path in tmp_result_paths():
            rows.extend(
                row for row in parse_json_result_rows(path)
                if row["mode"] == mode
            )
    return rows


def result_rows_match(rows, mode, dataset, seed, params=None):
    key = param_key(params) if params is not None else None
    return [
        row for row in rows
        if row["mode"] == mode
        and row["dataset"] == dataset
        and row["seed"] == seed
        and (key is None or row["params_key"] == key)
    ]


def existing_results(mode, dataset, seed, params=None, include_tmp=True):
    return result_rows_match(
        available_result_rows(mode, include_tmp=include_tmp),
        mode, dataset, seed, params
    )


def first_existing_result(mode, dataset, seed, params):
    rows = existing_results(mode, dataset, seed, params)
    return rows[-1] if rows else None


def first_existing_final_result(mode, dataset, seed, params):
    rows = existing_results(mode, dataset, seed, params, include_tmp=False)
    return rows[-1] if rows else None


def best_existing_validation(dataset, seed, allowed_params=None):
    allowed_keys = None
    if allowed_params is not None:
        allowed_keys = {param_key(params) for params in allowed_params}

    best_row = None
    best_score = None
    for row in existing_results("val", dataset, seed):
        # Search trials are defined as 20-epoch validation runs. Do not reuse
        # results from earlier runs that used a different training budget.
        if row.get("epochs") != 20:
            continue
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
        f"split_path={format_output_path(split_path)}\n"
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


def search_param_grid():
    return list(itertools.product(
        world.SEARCH_RECDIMS,
        world.SEARCH_LRS,
        world.SEARCH_DECAYS,
        world.SEARCH_LAYERS
    ))


def search_params(writer):
    params_grid = search_param_grid()
    completed_keys = {
        row["params_key"]
        for row in existing_results("val", world.dataset, world.seed)
        if row.get("epochs") == 20
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

    full_epoch_budget = world.TRAIN_epochs
    world.TRAIN_epochs = 20
    try:
        for params in pending_params:
            print(f"\n[SEARCH] params=(recdim={params[0]}, lr={params[1]}, decay={params[2]}, layer={params[3]})")
            validation_results, _, epochs, _, runtime = train_and_eval(params, test_eval_file="val.txt", writer=writer)
            current_score = score_results(validation_results)
            append_result("val", params, validation_results, 20, runtime, join(world.SPLIT_PATH, world.dataset))
            print(f"[SEARCH] Recall@{world.topks[0]}={current_score:.6f}")
            if best_score is None or current_score > best_score:
                best_score = current_score
                best_params = params
                best_results = validation_results
                print(f"[SEARCH] new best params={best_params}, recall={best_score:.6f}")
    finally:
        world.TRAIN_epochs = full_epoch_budget

    if best_params is None:
        raise RuntimeError(
            f"No validation results found for {world.dataset} seed={world.seed}; "
            "check the search grid and validation output."
        )

    print(f"\n[SEARCH] best params={best_params}, validation results={best_results}")
    append_selected_results("hyperparameter_search.txt", best_results, best_params)
    return best_params


def append_selected_results(filename, results, params, dataset=None, seed=None):
    path = join(world.ROOT_PATH, "results", filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    params_dict = {
        "recdim": params[0], "lr": params[1], "decay": params[2],
        "layer": params[3], "bpr_batch": world.config['bpr_batch_size'],
        "topks": world.topks,
    }
    with open(path, "a") as output:
        output.write(
            f"{dataset or world.dataset} | {world.seed if seed is None else seed} | {format_results(results)} | "
            f"{json.dumps(params_dict, sort_keys=True)}\n"
        )
    print(f"[RESULTS] appended to {path}")


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


def scheduler_gpu_ids():
    if world.args.gpu_ids is None:
        return []
    return [str(gpu) for gpu in expand_list(world.args.gpu_ids)]


def child_env(gpu_ids, launch_index):
    env = os.environ.copy()
    if gpu_ids:
        env["CUDA_VISIBLE_DEVICES"] = gpu_ids[launch_index % len(gpu_ids)]
    return env


def task_gpu_id(gpu_ids, launch_index):
    if not gpu_ids:
        return None
    return gpu_ids[launch_index % len(gpu_ids)]


def add_arg(cmd, name, value):
    cmd.extend([f"--{name}", str(value)])


def child_command(dataset, seed, result_file, task_mode="full", params=None):
    cmd = [sys.executable, "main.py"]
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
    # The scheduler masks each child's assigned physical GPU in child_env;
    # inside that child the selected device is therefore logical cuda:0.
    add_arg(cmd, "gpu_ID", 0)
    add_arg(cmd, "dataset", dataset)
    add_arg(cmd, "seed", seed)
    add_arg(cmd, "single_run", 1)
    add_arg(cmd, "child_result_file", result_file)
    add_arg(cmd, "task_mode", task_mode)
    if task_mode == "val":
        # Every grid candidate uses the same fixed 20-epoch budget.
        add_arg(cmd, "epochs", 20)
    if params is not None:
        add_arg(cmd, "task_recdim", params[0])
        add_arg(cmd, "task_lr", params[1])
        add_arg(cmd, "task_decay", params[2])
        add_arg(cmd, "task_layer", params[3])
    return cmd


def file_line_set(path):
    if not os.path.exists(path):
        return set()
    with open(path, "r") as f:
        return set(f.readlines())


def cached_json_lines(mode, dataset, seed, params):
    key = param_key(params)
    lines = []
    seen = set()
    for path in tmp_result_paths():
        for row in parse_json_result_rows(path):
            if (
                row["mode"] == mode
                and row["dataset"] == dataset
                and row["seed"] == seed
                and row["params_key"] == key
                and row["line"] not in seen
            ):
                seen.add(row["line"])
                lines.append(row["line"])
    return lines


def write_cached_child_result(result_file, task):
    lines = cached_json_lines(
        task["task_mode"], task["dataset"], task["seed"], task["params"]
    )
    if not lines:
        return
    existing = set()
    if os.path.exists(result_file):
        existing = {row["line"] for row in parse_json_result_rows(result_file)}
    os.makedirs(os.path.dirname(result_file), exist_ok=True)
    with open(result_file, "a") as f:
        for line in lines:
            if line not in existing:
                f.write(json.dumps({"mode": task["task_mode"], "line": line}) + "\n")
                existing.add(line)


def append_ordered_child_results(result_file, val_file, test_file, task_seed,
                                 previous_seed=None, error_line=None):
    val_lines = []
    test_lines = []
    if os.path.exists(result_file):
        with open(result_file, "r") as f:
            for raw in f:
                try:
                    record = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if record["mode"] == "val":
                    val_lines.append(record["line"])
                elif record["mode"] == "test":
                    test_lines.append(record["line"])

    if error_line:
        val_lines.append(error_line)
        test_lines.append(error_line)

    val_existing = file_line_set(val_file)
    test_existing = file_line_set(test_file)
    val_lines = [line for line in val_lines if line not in val_existing]
    test_lines = [line for line in test_lines if line not in test_existing]

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


def safe_token(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def params_slug(params):
    return (
        f"rd{params[0]}_lr{safe_token(params[1])}_"
        f"dc{safe_token(params[2])}_ly{params[3]}"
    )


def task_file_base(task):
    return (
        f"{task['task_mode']}_seed_{task['seed']}_"
        f"{safe_token(task['dataset'])}_{params_slug(task['params'])}"
    )


def attach_task_files(task, tmp_dir, log_dir):
    base = task_file_base(task)
    task["result_file"] = join(tmp_dir, f"{base}.jsonl")
    task["log_file"] = join(log_dir, f"{base}.log")
    return task


def task_existing_result(task, include_tmp=True):
    row = first_existing_result(
        task["task_mode"], task["dataset"], task["seed"], task["params"]
    ) if include_tmp else first_existing_final_result(
        task["task_mode"], task["dataset"], task["seed"], task["params"]
    )
    if task["task_mode"] == "val" and row is not None and row.get("epochs") != 20:
        return None
    return row


def task_param_text(task):
    params = task["params"]
    return (
        f"params=(recdim={params[0]}, lr={params[1]}, "
        f"decay={params[2]}, layer={params[3]})"
    )


def run_task_batch(tasks, max_concurrent, tmp_dir, log_dir, val_file, test_file,
                   last_flushed_seed, batch_label, gpu_ids=None):
    if gpu_ids is None:
        gpu_ids = []
    active = []
    next_task = 0
    completed = {}
    next_flush = 0

    while next_task < len(tasks) or active:
        while next_task < len(tasks) and len(active) < max_concurrent:
            task = attach_task_files(tasks[next_task], tmp_dir, log_dir)
            existing = task_existing_result(task, include_tmp=True)
            if existing is not None:
                if task_existing_result(task, include_tmp=False) is None:
                    write_cached_child_result(task["result_file"], task)
                completed[next_task] = {
                    "index": next_task,
                    "returncode": 0,
                    "runtime": 0.0,
                    "result_file": task["result_file"],
                    "log_file": task["log_file"],
                    "task": task,
                    "skipped": True,
                }
                print(
                    f"[SCHEDULER] resume: skipping {batch_label} #{next_task}: "
                    f"{task['task_mode']} {task['dataset']} seed={task['seed']} "
                    f"{task_param_text(task)}"
                )
                next_task += 1
                continue

            cmd = child_command(
                task["dataset"], task["seed"], task["result_file"],
                task_mode=task["task_mode"], params=task["params"]
            )
            gpu_id = task_gpu_id(gpu_ids, next_task)
            env = child_env(gpu_ids, next_task)
            log_handle = open(task["log_file"], "a")
            gpu_text = f" gpu={gpu_id}" if gpu_id is not None else ""
            log_handle.write(
                f"\n[SCHEDULER] start {task['task_mode']} "
                f"{task['dataset']} seed={task['seed']} {task_param_text(task)}{gpu_text}\n"
            )
            log_handle.flush()
            proc = subprocess.Popen(
                cmd,
                cwd=world.CODE_PATH,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            print(
                f"[SCHEDULER] started {batch_label} #{next_task}: "
                f"{task['task_mode']} {task['dataset']} seed={task['seed']} "
                f"{task_param_text(task)}{gpu_text} log={task['log_file']}"
            )
            active.append({
                "index": next_task,
                "gpu_id": gpu_id,
                "proc": proc,
                "log_handle": log_handle,
                "result_file": task["result_file"],
                "log_file": task["log_file"],
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
                f"[SCHEDULER] finished {batch_label} #{run['index']}: "
                f"{run['task']['task_mode']} {run['task']['dataset']} "
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
                    f"{task_param_text(task)} | runtime={run['runtime']:.2f}s | "
                    f"split_path={format_output_path(task['split_path'])} | "
                    f"log={format_output_path(run['log_file'])}\n"
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
            print(f"[SCHEDULER] flushed {batch_label} #{next_flush} results in initiation order")
            next_flush += 1

        if active:
            time.sleep(1)

    return last_flushed_seed


def search_unit_key(task):
    return task["dataset"], task["seed"]


def make_search_task(task_mode, dataset, seed, split_path, params):
    return {
        "task_mode": task_mode,
        "seed": seed,
        "dataset": dataset,
        "params": params,
        "split_path": split_path,
    }


def write_error_child_result(run):
    task = run["task"]
    error_line = (
        f"(error) | {task['dataset']:<{DATASET_COL_WIDTH}} | seed={task['seed']} | "
        f"{task_param_text(task)} | runtime={run['runtime']:.2f}s | "
        f"split_path={format_output_path(task['split_path'])} | "
        f"log={format_output_path(run['log_file'])}\n"
    )
    os.makedirs(os.path.dirname(run["result_file"]), exist_ok=True)
    with open(run["result_file"], "a") as f:
        f.write(json.dumps({"mode": task["task_mode"], "line": error_line}) + "\n")


def flush_task_result(task, val_file, test_file, previous_seed, flush_label):
    if task_existing_result(task, include_tmp=False) is not None:
        if os.path.exists(task["result_file"]):
            os.remove(task["result_file"])
        return True, previous_seed

    if not os.path.exists(task["result_file"]):
        write_cached_child_result(task["result_file"], task)
    if not os.path.exists(task["result_file"]):
        return False, previous_seed

    append_ordered_child_results(
        task["result_file"],
        val_file,
        test_file,
        task_seed=task["seed"],
        previous_seed=previous_seed,
    )
    if os.path.exists(task["result_file"]):
        os.remove(task["result_file"])
    print(f"[SCHEDULER] flushed {flush_label} result in output order")
    return True, task["seed"]


def start_scheduler_task(task, launch_index, gpu_ids):
    cmd = child_command(
        task["dataset"], task["seed"], task["result_file"],
        task_mode=task["task_mode"], params=task["params"]
    )
    gpu_id = task_gpu_id(gpu_ids, launch_index)
    env = child_env(gpu_ids, launch_index)
    log_handle = open(task["log_file"], "a")
    gpu_text = f" gpu={gpu_id}" if gpu_id is not None else ""
    log_handle.write(
        f"\n[SCHEDULER] start {task['task_mode']} "
        f"{task['dataset']} seed={task['seed']} {task_param_text(task)}{gpu_text}\n"
    )
    log_handle.flush()
    proc = subprocess.Popen(
        cmd,
        cwd=world.CODE_PATH,
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )
    print(
        f"[SCHEDULER] started #{launch_index}: {task['task_mode']} "
        f"{task['dataset']} seed={task['seed']} {task_param_text(task)}"
        f"{gpu_text} log={task['log_file']}"
    )
    return {
        "index": launch_index,
        "gpu_id": gpu_id,
        "proc": proc,
        "log_handle": log_handle,
        "result_file": task["result_file"],
        "log_file": task["log_file"],
        "task": task,
        "start": time.time(),
    }


def maybe_enqueue_search_test(unit_key, unit_states, params_grid, ready_tests,
                              tmp_dir, log_dir, val_file, test_file):
    state = unit_states[unit_key]
    if state["remaining_val"] != 0 or state["test_enqueued"] or state["test_done"]:
        return

    dataset, seed = unit_key
    best_row, best_score = best_existing_validation(dataset, seed, params_grid)
    if best_row is None:
        raise RuntimeError(
            f"No validation results found for {dataset} seed={seed}; "
            "cannot choose best hyperparameters."
        )

    best_params = best_row["params"]
    append_selected_results(
        "hyperparameter_search.txt", best_row["results"], best_params,
        dataset=dataset, seed=seed,
    )
    state["best_params"] = best_params
    print(
        f"[SCHEDULER] best params for {dataset} seed={seed}: "
        f"{best_params} Recall@{world.topks[0]}={best_score:.6f}"
    )

    task = make_search_task("test", dataset, seed, state["split_path"], best_params)
    task = attach_task_files(task, tmp_dir, log_dir)
    state["test_task"] = task
    existing = task_existing_result(task, include_tmp=True)
    if existing is not None:
        if task_existing_result(task, include_tmp=False) is None:
            write_cached_child_result(task["result_file"], task)
        state["test_done"] = True
        append_selected_results(
            "test_results.txt", existing["results"], best_params,
            dataset=dataset, seed=seed,
        )
        print(
            f"[SCHEDULER] resume: skipping existing test for "
            f"{dataset} seed={seed} {task_param_text(task)}"
        )
        return

    ready_tests.append(task)
    state["test_enqueued"] = True
    print(
        f"[SCHEDULER] queued test for {dataset} seed={seed} "
        f"{task_param_text(task)}"
    )


def flush_ready_search_results(unit_order, unit_states, val_file, test_file, flush_state):
    while flush_state["val_unit_index"] < len(unit_order):
        unit_key = unit_order[flush_state["val_unit_index"]]
        state = unit_states[unit_key]
        val_tasks = state["val_tasks"]
        while state["val_flush_index"] < len(val_tasks):
            task = val_tasks[state["val_flush_index"]]
            flushed, previous_seed = flush_task_result(
                task,
                val_file,
                test_file,
                flush_state["last_val_seed"],
                f"validation {task['dataset']} seed={task['seed']}",
            )
            if not flushed:
                break
            flush_state["last_val_seed"] = previous_seed
            state["val_flush_index"] += 1

        if state["val_flush_index"] < len(val_tasks):
            break
        flush_state["val_unit_index"] += 1

    while flush_state["test_unit_index"] < len(unit_order):
        unit_key = unit_order[flush_state["test_unit_index"]]
        state = unit_states[unit_key]
        task = state.get("test_task")
        if task is None:
            break
        flushed, previous_seed = flush_task_result(
            task,
            val_file,
            test_file,
            flush_state["last_test_seed"],
            f"test {task['dataset']} seed={task['seed']}",
        )
        if not flushed:
            break
        flush_state["last_test_seed"] = previous_seed
        flush_state["test_unit_index"] += 1


def validate_scheduler_inputs(datasets, seeds):
    unknown = [dataset for dataset in datasets if dataset not in world.all_dataset]
    if unknown:
        raise NotImplementedError(f"Haven't supported {unknown} yet!, try {world.all_dataset}")
    for seed in seeds:
        for dataset in datasets:
            split_path = join(world.DATA_PATH, f"00_dataset_seed_{seed}", dataset)
            if not os.path.isdir(split_path):
                raise FileNotFoundError(f"Split dataset not found: {split_path}")


def last_result_seed():
    rows = parse_result_rows(validation_results_path())
    if rows:
        return rows[-1]["seed"]
    rows = parse_result_rows(test_results_path())
    if rows:
        return rows[-1]["seed"]
    return None


def last_result_seed_for_path(path):
    rows = parse_result_rows(path)
    return rows[-1]["seed"] if rows else None


def run_search_scheduler(datasets, seeds):
    validate_scheduler_inputs(datasets, seeds)
    max_concurrent = max(1, world.args.concurrent_runs)
    gpu_ids = scheduler_gpu_ids()
    results_dir = join(world.ROOT_PATH, "results")
    tmp_dir = tmp_results_dir()
    log_dir = join(results_dir, "logs")
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    val_file = validation_results_path()
    test_file = test_results_path()
    params_grid = search_param_grid()
    started_at = time.time()

    print(
        f"[SCHEDULER] launching global parameter search with concurrent_runs={max_concurrent}; "
        f"gpu_ids={gpu_ids or 'default'}; datasets={datasets}; seeds={seeds}; "
        f"validation={val_file}, test={test_file}"
    )

    unit_states = {}
    unit_order = []
    pending_validations = []
    existing_validations = 0
    cached_validations = 0

    for seed in seeds:
        for dataset in datasets:
            split_path = join(world.DATA_PATH, f"00_dataset_seed_{seed}", dataset)
            unit_key = (dataset, seed)
            unit_order.append(unit_key)
            unit_states[unit_key] = {
                "dataset": dataset,
                "seed": seed,
                "split_path": split_path,
                "remaining_val": 0,
                "test_enqueued": False,
                "test_done": False,
                "best_params": None,
                "val_tasks": [],
                "val_flush_index": 0,
                "test_task": None,
            }

            completed_for_unit = 0
            for params in params_grid:
                task = make_search_task("val", dataset, seed, split_path, params)
                task = attach_task_files(task, tmp_dir, log_dir)
                unit_states[unit_key]["val_tasks"].append(task)
                existing = task_existing_result(task, include_tmp=True)
                if existing is not None:
                    existing_validations += 1
                    completed_for_unit += 1
                    if task_existing_result(task, include_tmp=False) is None:
                        cached_validations += 1
                        write_cached_child_result(task["result_file"], task)
                    continue

                unit_states[unit_key]["remaining_val"] += 1
                pending_validations.append(task)

            print(
                f"[SCHEDULER] validation grid: {dataset} seed={seed} "
                f"({completed_for_unit}/{len(params_grid)} complete, "
                f"{unit_states[unit_key]['remaining_val']} pending)"
            )

    ready_tests = []
    for unit_key in unit_states:
        maybe_enqueue_search_test(
            unit_key, unit_states, params_grid, ready_tests,
            tmp_dir, log_dir, val_file, test_file
        )
    flush_state = {
        "val_unit_index": 0,
        "test_unit_index": 0,
        "last_val_seed": last_result_seed_for_path(val_file),
        "last_test_seed": last_result_seed_for_path(test_file),
    }
    flush_ready_search_results(unit_order, unit_states, val_file, test_file, flush_state)

    print(
        f"[SCHEDULER] queued {len(pending_validations)} pending validation tasks; "
        f"{existing_validations} validation tasks already complete "
        f"({cached_validations} staged from tmp); {len(ready_tests)} tests ready"
    )

    active = []
    next_validation = 0
    launch_index = 0

    while next_validation < len(pending_validations) or ready_tests or active:
        while len(active) < max_concurrent:
            if ready_tests:
                task = ready_tests.pop(0)
            elif next_validation < len(pending_validations):
                task = pending_validations[next_validation]
                next_validation += 1
            else:
                break

            existing = task_existing_result(task, include_tmp=True)
            if existing is not None:
                if task_existing_result(task, include_tmp=False) is None:
                    write_cached_child_result(task["result_file"], task)

                unit_key = search_unit_key(task)
                if task["task_mode"] == "val":
                    unit_states[unit_key]["remaining_val"] = max(
                        0, unit_states[unit_key]["remaining_val"] - 1
                    )
                    maybe_enqueue_search_test(
                        unit_key, unit_states, params_grid, ready_tests,
                        tmp_dir, log_dir, val_file, test_file
                    )
                elif task["task_mode"] == "test":
                    unit_states[unit_key]["test_done"] = True
                print(
                    f"[SCHEDULER] resume: skipping existing {task['task_mode']} "
                    f"{task['dataset']} seed={task['seed']} {task_param_text(task)}"
                )
                flush_ready_search_results(unit_order, unit_states, val_file, test_file, flush_state)
                continue

            active.append(start_scheduler_task(task, launch_index, gpu_ids))
            launch_index += 1

        still_active = []
        for run in active:
            returncode = run["proc"].poll()
            if returncode is None:
                still_active.append(run)
                continue

            run["log_handle"].close()
            run["returncode"] = returncode
            run["runtime"] = time.time() - run["start"]
            status = "OK" if returncode == 0 else f"FAILED({returncode})"
            task = run["task"]
            print(
                f"[SCHEDULER] finished #{run['index']}: {task['task_mode']} "
                f"{task['dataset']} seed={task['seed']} {status} "
                f"runtime={run['runtime']:.2f}s"
            )
            if run["returncode"] != 0:
                write_error_child_result(run)

            unit_key = search_unit_key(task)
            if task["task_mode"] == "val":
                unit_states[unit_key]["remaining_val"] = max(
                    0, unit_states[unit_key]["remaining_val"] - 1
                )
                maybe_enqueue_search_test(
                    unit_key, unit_states, params_grid, ready_tests,
                    tmp_dir, log_dir, val_file, test_file
                )
            elif task["task_mode"] == "test":
                unit_states[unit_key]["test_done"] = True
            flush_ready_search_results(unit_order, unit_states, val_file, test_file, flush_state)

        active = still_active

        if active:
            time.sleep(1)

    for path in (val_file, test_file):
        with open(path, "a") as f:
            f.write("\n")

    print(f"[SCHEDULER] all search runs complete in {time.time() - started_at:.2f}s")


def task_params_from_args():
    values = (
        world.args.task_recdim,
        world.args.task_lr,
        world.args.task_decay,
        world.args.task_layer,
    )
    if any(value is None for value in values):
        raise ValueError("--task_mode val/test requires task_recdim, task_lr, task_decay, and task_layer")
    return values


def run_internal_task(writer):
    params = task_params_from_args()
    split_path = join(world.SPLIT_PATH, world.dataset)

    if world.args.task_mode == "val":
        existing_val = first_existing_result("val", world.dataset, world.seed, params)
        if existing_val is not None and existing_val.get("epochs") != 20:
            existing_val = None
        if existing_val is not None:
            print(
                f"[TASK] resume: skipping existing validation for "
                f"{world.dataset} seed={world.seed} params={params}"
            )
            return existing_val["results"]
        print(f"[TASK] validating params={params}")
        validation_results, _, epochs, best_epoch, runtime = train_and_eval(
            params, test_eval_file="val.txt", writer=writer
        )
        append_result("val", params, validation_results, 20, runtime, split_path)
        return validation_results

    if world.args.task_mode == "test":
        existing_test = first_existing_result("test", world.dataset, world.seed, params)
        if existing_test is not None:
            print(
                f"[TASK] resume: skipping existing test for "
                f"{world.dataset} seed={world.seed} params={params}"
            )
            return existing_test["results"]
        print(f"[TASK] testing params={params}")
        validation_results, final_results, epochs, best_epoch, runtime = train_and_eval(
            params,
            test_eval_file="test.txt",
            writer=writer,
            save_weight_file="auto"
        )
        logged_epochs = best_epoch if best_epoch is not None else epochs
        if first_existing_result("val", world.dataset, world.seed, params) is None:
            append_result("val", params, validation_results, logged_epochs, runtime, split_path)
        append_result("test", params, final_results, logged_epochs, runtime, split_path)
        append_selected_results("test_results.txt", final_results, params)
        return final_results

    raise ValueError(f"Unknown task mode: {world.args.task_mode}")


def run_scheduler():
    datasets = scheduler_datasets()
    seeds = scheduler_seeds()
    if world.search:
        run_search_scheduler(datasets, seeds)
        return

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
    gpu_ids = scheduler_gpu_ids()
    results_dir = join(world.ROOT_PATH, "results")
    tmp_dir = join(results_dir, "tmp")
    log_dir = join(results_dir, "logs")
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    val_file = validation_results_path()
    test_file = test_results_path()

    print(
        f"[SCHEDULER] launching {len(tasks)} runs with concurrent_runs={max_concurrent}; "
        f"gpu_ids={gpu_ids or 'default'}; validation={val_file}, test={test_file}"
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
            if os.path.exists(log_file):
                os.remove(log_file)
            cmd = child_command(task["dataset"], task["seed"], result_file)
            gpu_id = task_gpu_id(gpu_ids, next_task)
            env = child_env(gpu_ids, next_task)
            log_handle = open(log_file, "w")
            proc = subprocess.Popen(
                cmd,
                cwd=world.CODE_PATH,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            gpu_text = f" gpu={gpu_id}" if gpu_id is not None else ""
            print(
                f"[SCHEDULER] started #{next_task}: {task['dataset']} seed={task['seed']} "
                f"{gpu_text} log={log_file}"
            )
            active.append({
                "index": next_task,
                "gpu_id": gpu_id,
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
                    f"runtime={run['runtime']:.2f}s | split_path={format_output_path(task['split_path'])} | "
                    f"log={format_output_path(run['log_file'])}\n"
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
    utils.set_seed(world.seed)
    print(">>SEED:", world.seed)
    if scheduler_requested():
        run_scheduler()
        return

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
        if world.args.task_mode != "full":
            final_results = run_internal_task(writer)
            print(f"[FINAL] results={final_results}\n")
            return

        if world.search:
            best_params = search_params(writer)
            existing_test = first_existing_result("test", world.dataset, world.seed, best_params)
            if existing_test is not None:
                final_results = existing_test["results"]
                append_selected_results("test_results.txt", final_results, best_params)
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
                append_selected_results("test_results.txt", final_results, best_params)
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
