"""
Run all MIMIC probes: HP selection on val, evaluation on test.
23 tasks × 7 models = 161 runs.
"""

import json
import logging
import os
import time

import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

MIMIC_DIR = "experiments/nature_medicine/mimic"
OUT_DIR = "results/probes/nature_medicine/mimic_v2"

MODELS = [
    "echojepa_g",
    "echojepa_l",
    "echojepa_l_kinetics",
    "echomae",
    "echoprime",
    "panecho",
    "echofm",
]

CLS_TASKS = [
    "discharge_destination",
    "disease_afib",
    "disease_amyloidosis",
    "disease_dcm",
    "disease_hcm",
    "disease_hf",
    "disease_stemi",
    "disease_takotsubo",
    "disease_tamponade",
    "drg_severity",
    "icu_transfer",
    "in_hospital_mortality",
    "mortality_1yr",
    "mortality_30d",
    "mortality_90d",
    "readmission_30d",
    "triage_acuity",
]

REG_TASKS = [
    "creatinine",
    "ef_note_extracted",
    "lactate",
    "los_remaining",
    "nt_probnp",
    "troponin_t",
]

C_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
ALPHA_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]


def run_classification(model, task):
    split_dir = os.path.join(MIMIC_DIR, f"{model}_splits", task)
    out_dir = os.path.join(OUT_DIR, model, task)
    out_file = os.path.join(out_dir, "test_metrics.json")

    if os.path.exists(out_file):
        logger.info(f"  SKIP {model}/{task}: already done")
        return json.load(open(out_file))

    train_data = np.load(os.path.join(split_dir, "train.npz"), allow_pickle=True)
    val_data = np.load(os.path.join(split_dir, "val.npz"), allow_pickle=True)
    test_data = np.load(os.path.join(split_dir, "test.npz"), allow_pickle=True)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_data["embeddings"])
    X_val = scaler.transform(val_data["embeddings"])
    X_test = scaler.transform(test_data["embeddings"])
    y_train, y_val, y_test = train_data["labels"], val_data["labels"], test_data["labels"]

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
    n_classes = len(np.unique(y_test))

    if n_classes == 2:
        auc = roc_auc_score(y_test, y_proba[:, 1])
    else:
        try:
            auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
        except ValueError:
            auc = float("nan")

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "auc": float(auc),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "n_test": int(len(y_test)),
        "n_classes": int(n_classes),
        "best_C": float(best_C),
        "val_balanced_accuracy": float(best_score),
    }

    os.makedirs(out_dir, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"  {model}/{task}: AUC={auc:.3f}, BalAcc={metrics['balanced_accuracy']:.3f} (C={best_C})")
    return metrics


def run_regression(model, task):
    split_dir = os.path.join(MIMIC_DIR, f"{model}_splits", task)
    out_dir = os.path.join(OUT_DIR, model, task)
    out_file = os.path.join(out_dir, "test_metrics.json")

    if os.path.exists(out_file):
        logger.info(f"  SKIP {model}/{task}: already done")
        return json.load(open(out_file))

    train_data = np.load(os.path.join(split_dir, "train.npz"), allow_pickle=True)
    val_data = np.load(os.path.join(split_dir, "val.npz"), allow_pickle=True)
    test_data = np.load(os.path.join(split_dir, "test.npz"), allow_pickle=True)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_data["embeddings"])
    X_val = scaler.transform(val_data["embeddings"])
    X_test = scaler.transform(test_data["embeddings"])
    y_train, y_val, y_test = train_data["labels"], val_data["labels"], test_data["labels"]

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
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r, p = pearsonr(y_test, y_pred)

    metrics = {
        "r2": float(r2),
        "mae": float(mae),
        "pearson_r": float(r),
        "pearson_p": float(p),
        "n_test": int(len(y_test)),
        "best_alpha": float(best_alpha),
        "val_r2": float(best_score),
    }

    os.makedirs(out_dir, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"  {model}/{task}: R2={r2:.3f}, MAE={mae:.3f}, r={r:.3f} (alpha={best_alpha})")
    return metrics


def main():
    t0 = time.time()
    total, done, skipped = 0, 0, 0

    for model in MODELS:
        split_base = os.path.join(MIMIC_DIR, f"{model}_splits")
        if not os.path.isdir(split_base):
            logger.warning(f"SKIP {model}: no splits directory")
            continue

        logger.info(f"=== {model} ===")

        for task in CLS_TASKS:
            if not os.path.exists(os.path.join(split_base, task, "train.npz")):
                continue
            total += 1
            try:
                run_classification(model, task)
                done += 1
            except Exception as e:
                logger.error(f"  FAILED {model}/{task}: {e}")

        for task in REG_TASKS:
            if not os.path.exists(os.path.join(split_base, task, "train.npz")):
                continue
            total += 1
            try:
                run_regression(model, task)
                done += 1
            except Exception as e:
                logger.error(f"  FAILED {model}/{task}: {e}")

    elapsed = time.time() - t0
    logger.info(f"\nCOMPLETE: {done}/{total} runs in {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
