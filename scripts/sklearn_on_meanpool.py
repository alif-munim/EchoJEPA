#!/usr/bin/env python3
"""
Run sklearn classifiers on CY's mean-pooled frozen embeddings.
Reproduces CY's pipeline: clip-level mean-pool → study-level mean-pool → sklearn.

Usage:
    python scripts/sklearn_on_meanpool.py
"""
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

EFS = "/mnt/custom-file-systems/efs/fs-0049217cdf69186d7_fsap-0fa7145b64eaa046b/vjepa2"
EMB_PATH = f"{EFS}/experiments/nature_medicine/mimic/archived/echojepa_g_mimic_embeddings.npz"
CSV_DIR = f"{EFS}/experiments/nature_medicine/mimic/probe_csvs"

TASKS = {
    # Classification
    "mortality_1yr": "classification",
    "mortality_90d": "classification",
    "mortality_30d": "classification",
    "readmission_30d": "classification",
    "discharge_destination": "classification",
    # Regression
    "los_remaining": "regression",
    "troponin_t": "regression",
    "nt_probnp": "regression",
    "creatinine": "regression",
    "lactate": "regression",
}


def extract_study_id(path: str) -> str:
    m = re.search(r"/s(\d+)/", path)
    return m.group(1) if m else "unknown"


def load_embeddings():
    """Load CY's mean-pooled embeddings, index by path."""
    print("Loading embeddings...", end=" ", flush=True)
    t0 = time.time()
    d = np.load(EMB_PATH, allow_pickle=True)
    emb = d["embeddings"]  # (525312, 1408)
    paths = d["paths"]  # (525312,)
    path_to_idx = {str(p): i for i, p in enumerate(paths)}
    print(f"done ({len(emb)} clips, {time.time()-t0:.1f}s)")
    return emb, path_to_idx


def load_csv(csv_path: str):
    """Load a probe CSV: 'path label' per line."""
    paths, labels = [], []
    with open(csv_path) as f:
        for line in f:
            parts = line.strip().split()
            paths.append(parts[0])
            labels.append(float(parts[-1]))
    return paths, np.array(labels)


def build_study_features(clip_paths, clip_labels, embeddings, path_to_idx):
    """Average clip embeddings per study → study-level features + labels."""
    study_embs = defaultdict(list)
    study_labels = {}
    missing = 0

    for path, label in zip(clip_paths, clip_labels):
        sid = extract_study_id(path)
        idx = path_to_idx.get(path)
        if idx is None:
            missing += 1
            continue
        study_embs[sid].append(embeddings[idx])
        study_labels[sid] = label

    if missing > 0:
        print(f"  WARNING: {missing}/{len(clip_paths)} clips not found in embeddings")

    study_ids = sorted(study_embs.keys())
    X = np.array([np.mean(study_embs[sid], axis=0) for sid in study_ids])
    y = np.array([study_labels[sid] for sid in study_ids])
    return X, y, study_ids


def run_classification(task_name, embeddings, path_to_idx):
    """Run LogisticRegression with HP sweep on classification task."""
    csv_dir = f"{CSV_DIR}/{task_name}"

    train_paths, train_labels = load_csv(f"{csv_dir}/train.csv")
    val_paths, val_labels = load_csv(f"{csv_dir}/val.csv")
    test_paths, test_labels = load_csv(f"{csv_dir}/test.csv")

    X_train, y_train, _ = build_study_features(train_paths, train_labels, embeddings, path_to_idx)
    X_val, y_val, _ = build_study_features(val_paths, val_labels, embeddings, path_to_idx)
    X_test, y_test, _ = build_study_features(test_paths, test_labels, embeddings, path_to_idx)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    print(f"  Train: {X_train.shape[0]} studies, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    print(f"  Prevalence — train: {y_train.mean():.3f}, test: {y_test.mean():.3f}")

    best_auroc = -1
    best_cfg = None
    best_model = None

    for C in [0.001, 0.01, 0.1, 1.0, 10.0]:
        for penalty in ["l1", "l2"]:
            solver = "saga" if penalty == "l1" else "lbfgs"
            try:
                clf = LogisticRegression(
                    C=C, penalty=penalty, solver=solver,
                    max_iter=2000, random_state=42, n_jobs=-1
                )
                clf.fit(X_train, y_train)
                val_probs = clf.predict_proba(X_val)[:, 1]
                val_auroc = roc_auc_score(y_val, val_probs)
                if val_auroc > best_auroc:
                    best_auroc = val_auroc
                    best_cfg = f"C={C}, {penalty}"
                    best_model = clf
            except Exception:
                continue

    test_probs = best_model.predict_proba(X_test)[:, 1]
    test_auroc = roc_auc_score(y_test, test_probs)

    print(f"  Best val AUROC: {best_auroc:.4f} ({best_cfg})")
    print(f"  Test AUROC: {test_auroc:.4f}")

    return {
        "task": task_name,
        "type": "classification",
        "test_auroc": test_auroc,
        "val_auroc": best_auroc,
        "best_cfg": best_cfg,
        "n_train": X_train.shape[0],
        "n_test": X_test.shape[0],
    }


def run_regression(task_name, embeddings, path_to_idx):
    """Run Ridge regression with HP sweep on regression task."""
    csv_dir = f"{CSV_DIR}/{task_name}"

    # Load zscore params (CSVs store raw values, need to z-score for fair comparison)
    with open(f"{csv_dir}/zscore_params.json") as f:
        zp = json.load(f)
    target_mean, target_std = zp["target_mean"], zp["target_std"]

    train_paths, train_labels = load_csv(f"{csv_dir}/train.csv")
    val_paths, val_labels = load_csv(f"{csv_dir}/val.csv")
    test_paths, test_labels = load_csv(f"{csv_dir}/test.csv")

    X_train, y_train_raw, _ = build_study_features(train_paths, train_labels, embeddings, path_to_idx)
    X_val, y_val_raw, _ = build_study_features(val_paths, val_labels, embeddings, path_to_idx)
    X_test, y_test_raw, _ = build_study_features(test_paths, test_labels, embeddings, path_to_idx)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    print(f"  Train: {X_train.shape[0]} studies, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

    best_r2 = -999
    best_cfg = None
    best_model = None

    for alpha in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
        reg = Ridge(alpha=alpha, random_state=42)
        reg.fit(X_train, y_val_raw if len(y_train_raw) == 0 else y_train_raw)
        val_preds = reg.predict(X_val)
        ss_res = np.sum((y_val_raw - val_preds) ** 2)
        ss_tot = np.sum((y_val_raw - np.mean(y_val_raw)) ** 2)
        val_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        if val_r2 > best_r2:
            best_r2 = val_r2
            best_cfg = f"alpha={alpha}"
            best_model = reg

    test_preds = best_model.predict(X_test)
    ss_res = np.sum((y_test_raw - test_preds) ** 2)
    ss_tot = np.sum((y_test_raw - np.mean(y_test_raw)) ** 2)
    test_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    test_r, _ = pearsonr(y_test_raw, test_preds)

    print(f"  Best val R²: {best_r2:.4f} ({best_cfg})")
    print(f"  Test R²: {test_r2:.4f}, Pearson r: {test_r:.4f}")

    return {
        "task": task_name,
        "type": "regression",
        "test_r2": test_r2,
        "test_r": test_r,
        "val_r2": best_r2,
        "best_cfg": best_cfg,
        "n_train": X_train.shape[0],
        "n_test": X_test.shape[0],
    }


def main():
    embeddings, path_to_idx = load_embeddings()

    results = []
    for task_name, task_type in TASKS.items():
        print(f"\n{'='*60}")
        print(f"Task: {task_name} ({task_type})")
        print(f"{'='*60}")

        if task_type == "classification":
            r = run_classification(task_name, embeddings, path_to_idx)
        else:
            r = run_regression(task_name, embeddings, path_to_idx)
        results.append(r)

    # Summary table
    print(f"\n{'='*80}")
    print("SUMMARY — sklearn on mean-pooled G embeddings (study-level)")
    print(f"{'='*80}")
    print(f"{'Task':<25} {'Type':<15} {'Test Metric':<15} {'Value':<10} {'Best HP':<20}")
    print("-" * 85)
    for r in results:
        if r["type"] == "classification":
            print(f"{r['task']:<25} {'AUROC':<15} {'AUROC':<15} {r['test_auroc']:<10.4f} {r['best_cfg']:<20}")
        else:
            print(f"{r['task']:<25} {'R²/r':<15} {'R²':<15} {r['test_r2']:<10.4f} {r['best_cfg']:<20}")
            print(f"{'':<25} {'':<15} {'Pearson r':<15} {r['test_r']:<10.4f}")


if __name__ == "__main__":
    main()
