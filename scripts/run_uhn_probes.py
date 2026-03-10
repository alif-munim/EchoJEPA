"""
Run all UHN linear probes: HP selection on val, evaluation on test.
47 standard tasks x 2 models = 94 runs (+ 6 trajectory tasks per model).

Follows same protocol as run_mimic_probes_v2.py:
  - StandardScaler on embeddings (fit on train)
  - Classification: LogisticRegression with C grid, HP selected on val balanced_accuracy
  - Regression: Ridge with alpha grid, HP selected on val R2
  - Final metrics computed on test set
  - Results saved as test_metrics.json per model/task (skip if exists)
"""

import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

BASE = "/mnt/custom-file-systems/efs/fs-0049217cdf69186d7_fsap-0fa7145b64eaa046b/vjepa2"
UHN_DIR = f"{BASE}/experiments/nature_medicine/uhn"
OUT_DIR = f"{BASE}/results/probes/nature_medicine/uhn"

MODELS = [
    "echojepa_g",
    "echojepa_l",
]

# Classification tasks (int labels)
CLS_TASKS = [
    "ar_severity",
    "as_severity",
    "cardiac_rhythm",
    "diastolic_function",
    "disease_amyloidosis",
    "disease_bicuspid_av",
    "disease_dcm",
    "disease_endocarditis",
    "disease_hcm",
    "disease_myxomatous_mv",
    "disease_rheumatic_mv",
    "disease_stemi",
    "disease_takotsubo",
    "la_size",
    "lv_cavity_size",
    "lv_hypertrophy",
    "lv_systolic_function",
    "mr_severity",
    "pa_pressure",
    "pericardial_effusion",
    "pr_severity",
    "ra_size",
    "rv_function",
    "rv_size",
    "rwma",
    "tr_severity",
]

# Regression tasks (float labels)
REG_TASKS = [
    "ao_root",
    "aov_area",
    "aov_mean_grad",
    "aov_vmax",
    "cardiac_output",
    "edv",
    "esv",
    "gls",
    "ivsd",
    "la_vol",
    "lv_mass",
    "lvef",
    "lvot_vti",
    "mv_dt",
    "mv_ea",
    "mv_ee",
    "mv_ee_medial",
    "rv_fac",
    "rv_sp",
    "rvsp",
    "tapse",
]

# Trajectory tasks (paired embeddings, predict delta)
TRAJECTORY_TASKS = [
    "trajectory_lvef",
    "trajectory_tapse",
    "trajectory_lv_mass",
    "trajectory_rv_sp",
    "trajectory_mr_severity",
]

C_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
ALPHA_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]


def run_classification(model, task):
    split_dir = os.path.join(UHN_DIR, f"{model}_splits", task)
    out_dir = os.path.join(OUT_DIR, model, task)
    out_file = os.path.join(out_dir, "test_metrics.json")

    if os.path.exists(out_file):
        return json.load(open(out_file)), True

    train_data = np.load(os.path.join(split_dir, "train.npz"), allow_pickle=True)
    val_data = np.load(os.path.join(split_dir, "val.npz"), allow_pickle=True)
    test_data = np.load(os.path.join(split_dir, "test.npz"), allow_pickle=True)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_data["embeddings"])
    X_val = scaler.transform(val_data["embeddings"])
    X_test = scaler.transform(test_data["embeddings"])
    y_train, y_val, y_test = train_data["labels"], val_data["labels"], test_data["labels"]

    n_train = len(y_train)
    n_classes = len(np.unique(y_train))

    # HP search on val
    best_score, best_C, best_model = -1, None, None
    for C in C_GRID:
        m = LogisticRegression(C=C, max_iter=2000, solver="lbfgs", random_state=42)
        m.fit(X_train, y_train)
        score = balanced_accuracy_score(y_val, m.predict(X_val))
        if score > best_score:
            best_score, best_C, best_model = score, C, m

    # Evaluate on test
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)

    if n_classes == 2:
        auc = roc_auc_score(y_test, y_proba[:, 1])
    else:
        try:
            auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
        except ValueError:
            auc = float("nan")

    metrics = {
        "task": task,
        "task_type": "classification",
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "auc": float(auc),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "n_train": int(n_train),
        "n_test": int(len(y_test)),
        "n_classes": int(n_classes),
        "best_C": float(best_C),
        "val_balanced_accuracy": float(best_score),
    }

    os.makedirs(out_dir, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics, False


def run_regression(model, task):
    split_dir = os.path.join(UHN_DIR, f"{model}_splits", task)
    out_dir = os.path.join(OUT_DIR, model, task)
    out_file = os.path.join(out_dir, "test_metrics.json")

    if os.path.exists(out_file):
        return json.load(open(out_file)), True

    train_data = np.load(os.path.join(split_dir, "train.npz"), allow_pickle=True)
    val_data = np.load(os.path.join(split_dir, "val.npz"), allow_pickle=True)
    test_data = np.load(os.path.join(split_dir, "test.npz"), allow_pickle=True)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_data["embeddings"])
    X_val = scaler.transform(val_data["embeddings"])
    X_test = scaler.transform(test_data["embeddings"])
    y_train, y_val, y_test = train_data["labels"], val_data["labels"], test_data["labels"]

    n_train = len(y_train)

    # HP search on val
    best_score, best_alpha, best_model = -np.inf, None, None
    for alpha in ALPHA_GRID:
        m = Ridge(alpha=alpha)
        m.fit(X_train, y_train)
        score = r2_score(y_val, m.predict(X_val))
        if score > best_score:
            best_score, best_alpha, best_model = score, alpha, m

    # Evaluate on test
    y_pred = best_model.predict(X_test)

    metrics = {
        "task": task,
        "task_type": "regression",
        "r2": float(r2_score(y_test, y_pred)),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "pearson_r": float(pearsonr(y_test, y_pred)[0]),
        "spearman_rho": float(spearmanr(y_test, y_pred)[0]),
        "n_train": int(n_train),
        "n_test": int(len(y_test)),
        "best_alpha": float(best_alpha),
        "val_r2": float(best_score),
    }

    os.makedirs(out_dir, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics, False


def run_trajectory(model, task):
    """Train a probe to predict measurement delta from paired embeddings."""
    split_dir = os.path.join(UHN_DIR, f"{model}_splits", "trajectory", task)
    out_dir = os.path.join(OUT_DIR, model, task)
    out_file = os.path.join(out_dir, "test_metrics.json")

    if os.path.exists(out_file):
        return json.load(open(out_file)), True

    train_data = np.load(os.path.join(split_dir, "train.npz"), allow_pickle=True)
    val_data = np.load(os.path.join(split_dir, "val.npz"), allow_pickle=True)
    test_data = np.load(os.path.join(split_dir, "test.npz"), allow_pickle=True)

    # Input: concatenation of both embeddings
    X_train = np.concatenate([train_data["embeddings_1"], train_data["embeddings_2"]], axis=1)
    X_val = np.concatenate([val_data["embeddings_1"], val_data["embeddings_2"]], axis=1)
    X_test = np.concatenate([test_data["embeddings_1"], test_data["embeddings_2"]], axis=1)

    # Target: delta (label_2 - label_1)
    y_train = train_data["delta"]
    y_val = val_data["delta"]
    y_test = test_data["delta"]

    n_train = len(y_train)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # HP search on val
    best_score, best_alpha, best_model = -np.inf, None, None
    for alpha in ALPHA_GRID:
        m = Ridge(alpha=alpha)
        m.fit(X_train, y_train)
        score = r2_score(y_val, m.predict(X_val))
        if score > best_score:
            best_score, best_alpha, best_model = score, alpha, m

    # Evaluate on test
    y_pred = best_model.predict(X_test)

    metrics = {
        "task": task,
        "task_type": "trajectory_regression",
        "r2": float(r2_score(y_test, y_pred)),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "pearson_r": float(pearsonr(y_test, y_pred)[0]),
        "spearman_rho": float(spearmanr(y_test, y_pred)[0]),
        "n_train": int(n_train),
        "n_test": int(len(y_test)),
        "best_alpha": float(best_alpha),
        "val_r2": float(best_score),
    }

    os.makedirs(out_dir, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics, False


def run_task(args):
    """Worker function for parallel execution."""
    model, task, task_type = args
    try:
        if task_type == "classification":
            metrics, skipped = run_classification(model, task)
        elif task_type == "regression":
            metrics, skipped = run_regression(model, task)
        elif task_type == "trajectory":
            metrics, skipped = run_trajectory(model, task)
        else:
            return model, task, None, False, "unknown task_type"
        return model, task, metrics, skipped, None
    except Exception as e:
        return model, task, None, False, str(e)


def print_summary(all_results):
    """Print summary tables."""
    for model in MODELS:
        model_results = {k: v for (m, k), v in all_results.items() if m == model}
        if not model_results:
            continue

        cls_results = {k: v for k, v in model_results.items() if v.get("task_type") == "classification"}
        reg_results = {k: v for k, v in model_results.items() if v.get("task_type") == "regression"}
        traj_results = {k: v for k, v in model_results.items() if v.get("task_type") == "trajectory_regression"}

        print(f"\n{'='*90}")
        print(f"  {model.upper()}")
        print(f"{'='*90}")

        if cls_results:
            print(f"\n  Classification ({len(cls_results)} tasks)")
            print(f"  {'Task':<30} {'AUC':>8} {'BalAcc':>8} {'F1mac':>8} {'N_test':>8} {'C':>8}")
            print(f"  {'-'*70}")
            for task in sorted(cls_results):
                m = cls_results[task]
                print(f"  {task:<30} {m['auc']:>8.3f} {m['balanced_accuracy']:>8.3f} "
                      f"{m['f1_macro']:>8.3f} {m['n_test']:>8d} {m['best_C']:>8.4g}")

        if reg_results:
            print(f"\n  Regression ({len(reg_results)} tasks)")
            print(f"  {'Task':<30} {'R2':>8} {'MAE':>8} {'Pearson':>8} {'N_test':>8} {'alpha':>8}")
            print(f"  {'-'*70}")
            for task in sorted(reg_results):
                m = reg_results[task]
                print(f"  {task:<30} {m['r2']:>8.3f} {m['mae']:>8.3f} "
                      f"{m['pearson_r']:>8.3f} {m['n_test']:>8d} {m['best_alpha']:>8.4g}")

        if traj_results:
            print(f"\n  Trajectory ({len(traj_results)} tasks)")
            print(f"  {'Task':<30} {'R2':>8} {'MAE':>8} {'Pearson':>8} {'N_test':>8} {'alpha':>8}")
            print(f"  {'-'*70}")
            for task in sorted(traj_results):
                m = traj_results[task]
                print(f"  {task:<30} {m['r2']:>8.3f} {m['mae']:>8.3f} "
                      f"{m['pearson_r']:>8.3f} {m['n_test']:>8d} {m['best_alpha']:>8.4g}")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=MODELS, help="Models to run")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers (default=1 sequential)")
    parser.add_argument("--skip-trajectory", action="store_true", help="Skip trajectory tasks")
    args = parser.parse_args()

    t0 = time.time()

    # Build task list
    jobs = []
    for model in args.models:
        split_base = os.path.join(UHN_DIR, f"{model}_splits")
        if not os.path.isdir(split_base):
            logger.warning(f"SKIP {model}: no splits directory")
            continue

        for task in CLS_TASKS:
            if os.path.exists(os.path.join(split_base, task, "train.npz")):
                jobs.append((model, task, "classification"))

        for task in REG_TASKS:
            if os.path.exists(os.path.join(split_base, task, "train.npz")):
                jobs.append((model, task, "regression"))

        if not args.skip_trajectory:
            for task in TRAJECTORY_TASKS:
                if os.path.exists(os.path.join(split_base, "trajectory", task, "train.npz")):
                    jobs.append((model, task, "trajectory"))

    logger.info(f"Total jobs: {len(jobs)} ({len(args.models)} models)")

    all_results = {}
    done, skipped, failed = 0, 0, 0

    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_task, job): job for job in jobs}
            for future in as_completed(futures):
                model, task, metrics, was_skipped, error = future.result()
                if error:
                    logger.error(f"  FAILED {model}/{task}: {error}")
                    failed += 1
                elif was_skipped:
                    skipped += 1
                else:
                    all_results[(model, task)] = metrics
                    task_type = metrics.get("task_type", "?")
                    if task_type == "classification":
                        logger.info(f"  {model}/{task}: AUC={metrics['auc']:.3f} BalAcc={metrics['balanced_accuracy']:.3f}")
                    else:
                        logger.info(f"  {model}/{task}: R2={metrics['r2']:.3f} MAE={metrics['mae']:.3f}")
                    done += 1
    else:
        for job in jobs:
            model, task, metrics, was_skipped, error = run_task(job)
            if error:
                logger.error(f"  FAILED {model}/{task}: {error}")
                failed += 1
            elif was_skipped:
                logger.info(f"  SKIP {model}/{task}: already done")
                all_results[(model, task)] = metrics
                skipped += 1
            else:
                all_results[(model, task)] = metrics
                task_type = metrics.get("task_type", "?")
                if task_type == "classification":
                    logger.info(f"  {model}/{task}: AUC={metrics['auc']:.3f} BalAcc={metrics['balanced_accuracy']:.3f}")
                else:
                    logger.info(f"  {model}/{task}: R2={metrics['r2']:.3f} MAE={metrics['mae']:.3f}")
                done += 1

    elapsed = time.time() - t0
    logger.info(f"\nCOMPLETE: {done} new + {skipped} skipped + {failed} failed = {done + skipped + failed}/{len(jobs)} in {elapsed/60:.1f} min")

    # Load skipped results for summary
    for model in args.models:
        for task_list, task_type in [(CLS_TASKS, "classification"), (REG_TASKS, "regression")]:
            for task in task_list:
                if (model, task) not in all_results:
                    f = os.path.join(OUT_DIR, model, task, "test_metrics.json")
                    if os.path.exists(f):
                        all_results[(model, task)] = json.load(open(f))
        if not args.skip_trajectory:
            for task in TRAJECTORY_TASKS:
                if (model, task) not in all_results:
                    f = os.path.join(OUT_DIR, model, task, "test_metrics.json")
                    if os.path.exists(f):
                        all_results[(model, task)] = json.load(open(f))

    print_summary(all_results)

    # Save combined results
    combined_file = os.path.join(OUT_DIR, "all_results.json")
    combined = {}
    for (model, task), metrics in sorted(all_results.items()):
        if model not in combined:
            combined[model] = {}
        combined[model][task] = metrics
    with open(combined_file, "w") as f:
        json.dump(combined, f, indent=2)
    logger.info(f"Saved combined results to {combined_file}")


if __name__ == "__main__":
    main()
