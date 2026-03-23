#!/usr/bin/env python3
"""
Run sklearn on mean-pooled embeddings with CLIP-LEVEL training + study-level pred avg.
This matches CY's pipeline more closely: train on clip embeddings, predict per clip,
average predictions per study.

Also adds prediction averaging to the study-level pipeline for comparison.

Usage:
    python scripts/sklearn_on_meanpool_predavg.py
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
    "mortality_1yr": "classification",
    "mortality_90d": "classification",
    "mortality_30d": "classification",
    "readmission_30d": "classification",
    "discharge_destination": "classification",
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
    print("Loading embeddings...", end=" ", flush=True)
    t0 = time.time()
    d = np.load(EMB_PATH, allow_pickle=True)
    emb = d["embeddings"]
    paths = d["paths"]
    path_to_idx = {str(p): i for i, p in enumerate(paths)}
    print(f"done ({len(emb)} clips, {time.time()-t0:.1f}s)")
    return emb, path_to_idx


def load_csv(csv_path: str):
    paths, labels = [], []
    with open(csv_path) as f:
        for line in f:
            parts = line.strip().split()
            paths.append(parts[0])
            labels.append(float(parts[-1]))
    return paths, np.array(labels)


def get_clip_data(clip_paths, clip_labels, embeddings, path_to_idx):
    """Get clip-level embeddings and labels, filtering missing clips."""
    X, y, sids = [], [], []
    missing = 0
    for path, label in zip(clip_paths, clip_labels):
        idx = path_to_idx.get(path)
        if idx is None:
            missing += 1
            continue
        X.append(embeddings[idx])
        y.append(label)
        sids.append(extract_study_id(path))
    if missing > 0:
        print(f"  WARNING: {missing} clips not found")
    return np.array(X), np.array(y), np.array(sids)


def study_level_pred_avg_clf(model, X_clips, y_clips, study_ids):
    """Predict per clip, average probabilities per study, compute AUROC."""
    clip_probs = model.predict_proba(X_clips)[:, 1]

    study_probs = defaultdict(list)
    study_labels = {}
    for sid, prob, label in zip(study_ids, clip_probs, y_clips):
        study_probs[sid].append(prob)
        study_labels[sid] = int(label)

    sids = sorted(study_probs.keys())
    avg_probs = np.array([np.mean(study_probs[s]) for s in sids])
    labels = np.array([study_labels[s] for s in sids])
    return roc_auc_score(labels, avg_probs), len(sids)


def study_level_pred_avg_reg(model, X_clips, y_clips, study_ids):
    """Predict per clip, average predictions per study, compute R²/r."""
    clip_preds = model.predict(X_clips)

    study_preds = defaultdict(list)
    study_labels = {}
    for sid, pred, label in zip(study_ids, clip_preds, y_clips):
        study_preds[sid].append(pred)
        study_labels[sid] = label

    sids = sorted(study_preds.keys())
    avg_preds = np.array([np.mean(study_preds[s]) for s in sids])
    labels = np.array([study_labels[s] for s in sids])

    ss_res = np.sum((labels - avg_preds) ** 2)
    ss_tot = np.sum((labels - np.mean(labels)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    r, _ = pearsonr(labels, avg_preds)
    return r2, r, len(sids)


def run_classification(task_name, embeddings, path_to_idx):
    csv_dir = f"{CSV_DIR}/{task_name}"

    train_paths, train_labels = load_csv(f"{csv_dir}/train.csv")
    val_paths, val_labels = load_csv(f"{csv_dir}/val.csv")
    test_paths, test_labels = load_csv(f"{csv_dir}/test.csv")

    # Clip-level data
    X_train, y_train, sid_train = get_clip_data(train_paths, train_labels, embeddings, path_to_idx)
    X_val, y_val, sid_val = get_clip_data(val_paths, val_labels, embeddings, path_to_idx)
    X_test, y_test, sid_test = get_clip_data(test_paths, test_labels, embeddings, path_to_idx)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    n_train_studies = len(set(sid_train))
    n_test_studies = len(set(sid_test))
    print(f"  Clips — train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")
    print(f"  Studies — train: {n_train_studies}, test: {n_test_studies}")

    # Subsample training clips: 1 per study for speed (like study_sampling)
    # Actually, use all clips — CY trained on all clips
    best_auroc = -1
    best_cfg = None
    best_model = None

    for C in [0.001, 0.01, 0.1, 1.0, 10.0]:
        # Skip L1 — too slow on 300K+ clips × 1408 features
        clf = LogisticRegression(
            C=C, penalty="l2", solver="lbfgs",
            max_iter=500, random_state=42, n_jobs=-1
        )
        clf.fit(X_train, y_train)

        # Val: study-level pred avg
        val_auroc, _ = study_level_pred_avg_clf(clf, X_val, y_val, sid_val)
        if val_auroc > best_auroc:
            best_auroc = val_auroc
            best_cfg = f"C={C}, l2"
            best_model = clf
        print(f"    C={C}, l2 → val PA AUROC: {val_auroc:.4f}")

    # Test: study-level pred avg
    test_auroc, n_studies = study_level_pred_avg_clf(best_model, X_test, y_test, sid_test)

    # Also clip-level test AUROC for reference
    clip_auroc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])

    print(f"  Best: {best_cfg}")
    print(f"  Test clip AUROC: {clip_auroc:.4f}")
    print(f"  Test PA AUROC:   {test_auroc:.4f} ({n_studies} studies)")

    return {
        "task": task_name,
        "type": "classification",
        "test_auroc_clip": clip_auroc,
        "test_auroc_pa": test_auroc,
        "best_cfg": best_cfg,
        "n_test_studies": n_studies,
    }


def run_regression(task_name, embeddings, path_to_idx):
    csv_dir = f"{CSV_DIR}/{task_name}"

    train_paths, train_labels = load_csv(f"{csv_dir}/train.csv")
    val_paths, val_labels = load_csv(f"{csv_dir}/val.csv")
    test_paths, test_labels = load_csv(f"{csv_dir}/test.csv")

    X_train, y_train, sid_train = get_clip_data(train_paths, train_labels, embeddings, path_to_idx)
    X_val, y_val, sid_val = get_clip_data(val_paths, val_labels, embeddings, path_to_idx)
    X_test, y_test, sid_test = get_clip_data(test_paths, test_labels, embeddings, path_to_idx)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    print(f"  Clips — train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")
    print(f"  Studies — train: {len(set(sid_train))}, test: {len(set(sid_test))}")

    best_r2 = -999
    best_cfg = None
    best_model = None

    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
        reg = Ridge(alpha=alpha, random_state=42)
        reg.fit(X_train, y_train)

        val_r2, val_r, _ = study_level_pred_avg_reg(reg, X_val, y_val, sid_val)
        if val_r2 > best_r2:
            best_r2 = val_r2
            best_cfg = f"alpha={alpha}"
            best_model = reg
        print(f"    alpha={alpha} → val PA R²: {val_r2:.4f}, r: {val_r:.4f}")

    test_r2, test_r, n_studies = study_level_pred_avg_reg(best_model, X_test, y_test, sid_test)

    print(f"  Best: {best_cfg}")
    print(f"  Test PA R²: {test_r2:.4f}, r: {test_r:.4f} ({n_studies} studies)")

    return {
        "task": task_name,
        "type": "regression",
        "test_r2_pa": test_r2,
        "test_r_pa": test_r,
        "best_cfg": best_cfg,
        "n_test_studies": n_studies,
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

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY — sklearn clip-level train + study-level pred avg")
    print(f"{'='*80}")
    print(f"{'Task':<25} {'Clip AUROC':<12} {'PA AUROC/R²':<12} {'PA r':<10} {'Studies':<8} {'HP':<15}")
    print("-" * 82)
    for r in results:
        if r["type"] == "classification":
            print(f"{r['task']:<25} {r['test_auroc_clip']:<12.4f} {r['test_auroc_pa']:<12.4f} {'—':<10} {r['n_test_studies']:<8} {r['best_cfg']:<15}")
        else:
            print(f"{r['task']:<25} {'—':<12} {r['test_r2_pa']:<12.4f} {r['test_r_pa']:<10.4f} {r['n_test_studies']:<8} {r['best_cfg']:<15}")


if __name__ == "__main__":
    main()
