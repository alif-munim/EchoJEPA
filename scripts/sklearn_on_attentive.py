#!/usr/bin/env python3
"""
Run sklearn on attentive probe predictions to compare with mean-pooled sklearn.

Since each HP head has its own attentive pooler, we can't mix 1408-dim features
across heads. Instead, we use clip_probs_all_heads (N, 15, C) and pick the same
head for train and test. For classification, the probe probability IS the feature;
for regression, the z-scored prediction is the feature.

This tests: given the same sklearn classifier, do attentive probe clip outputs
beat mean-pooled clip embeddings for study-level prediction averaging?

Also: we compare using ALL 15 heads' outputs as a 15-dim (or 30-dim) feature
vector, which ensembles the attentive probes via sklearn.

Usage:
    python scripts/sklearn_on_attentive.py
"""
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

EFS = "/mnt/custom-file-systems/efs/fs-0049217cdf69186d7_fsap-0fa7145b64eaa046b/vjepa2"
OUT_DIR = f"{EFS}/evals/vitg-384/nature_medicine/mimic/video_classification_frozen"
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


def load_npz(npz_path):
    """Load clip_outputs.npz."""
    d = np.load(npz_path, allow_pickle=True)
    return d


def split_train_val(probs, labels, study_ids, val_frac=0.15, seed=42):
    """Split train data by study into train_sub and val."""
    rng = np.random.RandomState(seed)
    unique_studies = np.unique(study_ids)
    rng.shuffle(unique_studies)
    n_val = max(1, int(len(unique_studies) * val_frac))
    val_studies = set(unique_studies[:n_val])
    train_mask = np.array([s not in val_studies for s in study_ids])
    val_mask = ~train_mask
    return (
        probs[train_mask], labels[train_mask], study_ids[train_mask],
        probs[val_mask], labels[val_mask], study_ids[val_mask],
    )


def study_pred_avg_clf(probs, y, sids):
    """Clip probabilities -> study-level average -> AUROC."""
    study_probs = defaultdict(list)
    study_labels = {}
    for sid, p, lab in zip(sids, probs, y):
        study_probs[sid].append(p)
        study_labels[sid] = int(lab)
    keys = sorted(study_probs.keys())
    avg = np.array([np.mean(study_probs[s]) for s in keys])
    labs = np.array([study_labels[s] for s in keys])
    return roc_auc_score(labs, avg), len(keys)


def study_pred_avg_reg(preds, y, sids):
    """Clip predictions -> study-level average -> R^2 and r."""
    study_preds = defaultdict(list)
    study_labels = {}
    for sid, p, lab in zip(sids, preds, y):
        study_preds[sid].append(p)
        study_labels[sid] = lab
    keys = sorted(study_preds.keys())
    avg = np.array([np.mean(study_preds[s]) for s in keys])
    labs = np.array([study_labels[s] for s in keys])
    ss_res = np.sum((labs - avg) ** 2)
    ss_tot = np.sum((labs - np.mean(labs)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    r, _ = pearsonr(labs, avg)
    return r2, r, len(keys)


def study_pred_avg_sklearn_clf(model, X, y, sids):
    """sklearn clip predictions -> study-level average -> AUROC."""
    probs = model.predict_proba(X)[:, 1]
    return study_pred_avg_clf(probs, y, sids)


def study_pred_avg_sklearn_reg(model, X, y, sids):
    """sklearn clip predictions -> study-level average -> R^2, r."""
    preds = model.predict(X)
    return study_pred_avg_reg(preds, y, sids)


def run_classification(task_name):
    train_npz_path = f"{OUT_DIR}/{task_name}-trainfeat-echojepa-g/clip_outputs.npz"
    test_npz_path = f"{OUT_DIR}/{task_name}-predavg-echojepa-g/clip_outputs.npz"

    if not Path(train_npz_path).exists() or not Path(test_npz_path).exists():
        print(f"  SKIP: missing NPZ")
        return None

    train_npz = load_npz(train_npz_path)
    test_npz = load_npz(test_npz_path)

    # clip_probs_all_heads: (N, 15, 2) — class probabilities per head
    train_probs_all = train_npz["clip_probs_all_heads"]  # (N_train, 15, 2)
    test_probs_all = test_npz["clip_probs_all_heads"]    # (N_test, 15, 2)
    train_labels = train_npz["clip_labels"]
    test_labels = test_npz["clip_labels"]
    train_sids = train_npz["clip_study_ids"]
    test_sids = test_npz["clip_study_ids"]

    n_heads = train_probs_all.shape[1]
    print(f"  Train: {len(train_labels)} clips ({len(set(train_sids))} studies)")
    print(f"  Test:  {len(test_labels)} clips ({len(set(test_sids))} studies)")

    # --- Method 1: Direct probe pred avg (no sklearn) per head ---
    print(f"\n  --- Direct probe pred avg (no sklearn) ---")
    best_val_auroc, best_head = -1, -1
    head_test_aurocs = []

    for h in range(n_heads):
        # Split train for val
        tr_probs_h = train_probs_all[:, h, 1]
        _, _, _, val_probs_h, val_labels, val_sids = split_train_val(
            tr_probs_h, train_labels, train_sids
        )
        val_auroc, _ = study_pred_avg_clf(val_probs_h, val_labels, val_sids)

        test_probs_h = test_probs_all[:, h, 1]
        test_auroc, n_studies = study_pred_avg_clf(test_probs_h, test_labels, test_sids)
        head_test_aurocs.append(test_auroc)

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_head = h

    best_test_auroc = head_test_aurocs[best_head]
    oracle_test_auroc = max(head_test_aurocs)
    oracle_head = int(np.argmax(head_test_aurocs))
    print(f"  Best head (by val): {best_head}, test PA AUROC: {best_test_auroc:.4f}")
    print(f"  Oracle head (by test): {oracle_head}, test PA AUROC: {oracle_test_auroc:.4f}")

    # --- Method 2: sklearn on all-heads probs as features (30-dim) ---
    print(f"\n  --- sklearn on 15-head probs (30-dim features) ---")
    X_train_all = train_probs_all.reshape(len(train_labels), -1)  # (N, 30)
    X_test_all = test_probs_all.reshape(len(test_labels), -1)    # (N, 30)

    X_tr, y_tr, sid_tr, X_val, y_val, sid_val = split_train_val(
        X_train_all, train_labels, train_sids
    )
    X_test = X_test_all
    y_test = test_labels
    sid_test = test_sids

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    best_auroc_sk, best_cfg, best_model = -1, None, None
    for C in [0.001, 0.01, 0.1, 1.0, 10.0]:
        clf = LogisticRegression(
            C=C, penalty="l2", solver="lbfgs",
            max_iter=1000, random_state=42, n_jobs=-1
        )
        clf.fit(X_tr_s, y_tr)
        val_auroc, _ = study_pred_avg_sklearn_clf(clf, X_val_s, y_val, sid_val)
        if val_auroc > best_auroc_sk:
            best_auroc_sk = val_auroc
            best_cfg = f"C={C}"
            best_model = clf

    sk_test_auroc, n_studies = study_pred_avg_sklearn_clf(best_model, X_test_s, y_test, sid_test)
    print(f"  Best: {best_cfg}, test PA AUROC: {sk_test_auroc:.4f} ({n_studies} studies)")

    return {
        "task": task_name, "type": "classification",
        "direct_pa": best_test_auroc,
        "direct_oracle": oracle_test_auroc,
        "sklearn_30d_pa": sk_test_auroc,
        "best_head": best_head,
        "best_cfg": best_cfg,
        "n_test_studies": n_studies,
    }


def run_regression(task_name):
    train_npz_path = f"{OUT_DIR}/{task_name}-trainfeat-echojepa-g/clip_outputs.npz"
    test_npz_path = f"{OUT_DIR}/{task_name}-predavg-echojepa-g/clip_outputs.npz"

    if not Path(train_npz_path).exists() or not Path(test_npz_path).exists():
        print(f"  SKIP: missing NPZ")
        return None

    train_npz = load_npz(train_npz_path)
    test_npz = load_npz(test_npz_path)

    # For regression: clip_predictions_all_heads (N, 15) — z-scored predictions
    train_preds_all = train_npz["clip_predictions_all_heads"]  # (N_train, 15)
    test_preds_all = test_npz["clip_predictions_all_heads"]    # (N_test, 15)
    train_labels = train_npz["clip_labels"].astype(np.float64)
    test_labels = test_npz["clip_labels"].astype(np.float64)
    train_sids = train_npz["clip_study_ids"]
    test_sids = test_npz["clip_study_ids"]

    n_heads = train_preds_all.shape[1]
    print(f"  Train: {len(train_labels)} clips ({len(set(train_sids))} studies)")
    print(f"  Test:  {len(test_labels)} clips ({len(set(test_sids))} studies)")

    # --- Method 1: Direct probe pred avg per head ---
    print(f"\n  --- Direct probe pred avg (no sklearn) ---")
    best_val_r2, best_head = -999, -1
    head_test_r2s = []

    for h in range(n_heads):
        tr_preds_h = train_preds_all[:, h]
        _, _, _, val_preds_h, val_labels, val_sids = split_train_val(
            tr_preds_h, train_labels, train_sids
        )
        val_r2, _, _ = study_pred_avg_reg(val_preds_h, val_labels, val_sids)

        test_preds_h = test_preds_all[:, h]
        test_r2, test_r, n_studies = study_pred_avg_reg(test_preds_h, test_labels, test_sids)
        head_test_r2s.append(test_r2)

        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_head = h

    best_test_r2 = head_test_r2s[best_head]
    oracle_test_r2 = max(head_test_r2s)
    oracle_head = int(np.argmax(head_test_r2s))

    # Get r for best head
    test_preds_best = test_preds_all[:, best_head]
    _, best_test_r, _ = study_pred_avg_reg(test_preds_best, test_labels, test_sids)

    print(f"  Best head (by val): {best_head}, test PA R2: {best_test_r2:.4f}, r: {best_test_r:.4f}")
    print(f"  Oracle head (by test): {oracle_head}, test PA R2: {oracle_test_r2:.4f}")

    # --- Method 2: sklearn Ridge on all-heads preds (15-dim) ---
    print(f"\n  --- sklearn on 15-head predictions (15-dim features) ---")
    X_train_all = train_preds_all  # (N, 15)
    X_test_all = test_preds_all    # (N, 15)

    X_tr, y_tr, sid_tr, X_val, y_val, sid_val = split_train_val(
        X_train_all, train_labels, train_sids
    )
    X_test = X_test_all
    y_test = test_labels
    sid_test = test_sids

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    best_r2_sk, best_cfg, best_model = -999, None, None
    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
        reg = Ridge(alpha=alpha, random_state=42)
        reg.fit(X_tr_s, y_tr)
        val_r2, _, _ = study_pred_avg_sklearn_reg(reg, X_val_s, y_val, sid_val)
        if val_r2 > best_r2_sk:
            best_r2_sk = val_r2
            best_cfg = f"alpha={alpha}"
            best_model = reg

    sk_test_r2, sk_test_r, n_studies = study_pred_avg_sklearn_reg(
        best_model, X_test_s, y_test, sid_test
    )
    print(f"  Best: {best_cfg}, test PA R2: {sk_test_r2:.4f}, r: {sk_test_r:.4f} ({n_studies} studies)")

    return {
        "task": task_name, "type": "regression",
        "direct_pa_r2": best_test_r2,
        "direct_pa_r": best_test_r,
        "direct_oracle_r2": oracle_test_r2,
        "sklearn_15d_r2": sk_test_r2,
        "sklearn_15d_r": sk_test_r,
        "best_head": best_head,
        "best_cfg": best_cfg,
        "n_test_studies": n_studies,
    }


def main():
    results = []
    for task_name, task_type in TASKS.items():
        print(f"\n{'='*60}")
        print(f"Task: {task_name} ({task_type})")
        print(f"{'='*60}")
        t0 = time.time()
        if task_type == "classification":
            r = run_classification(task_name)
        else:
            r = run_regression(task_name)
        if r:
            results.append(r)
        print(f"  ({time.time()-t0:.0f}s)")

    # Summary
    print(f"\n{'='*90}")
    print("SUMMARY: Attentive probe outputs — direct PA vs sklearn ensemble")
    print(f"{'='*90}")
    print(f"{'Task':<25} {'Direct PA':<12} {'Oracle PA':<12} {'sklearn PA':<12} {'Head':<6} {'Studies':<8}")
    print("-" * 75)
    for r in results:
        if r["type"] == "classification":
            print(f"{r['task']:<25} {r['direct_pa']:<12.4f} {r['direct_oracle']:<12.4f} {r['sklearn_30d_pa']:<12.4f} {r['best_head']:<6} {r['n_test_studies']:<8}")
        else:
            print(f"{r['task']:<25} {r['direct_pa_r2']:<12.4f} {r['direct_oracle_r2']:<12.4f} {r['sklearn_15d_r2']:<12.4f} {r['best_head']:<6} {r['n_test_studies']:<8}")

    # Also print the comparison we care about
    print(f"\n{'='*90}")
    print("KEY COMPARISON: Direct attentive PA vs CY mean-pool sklearn PA")
    print("(Direct PA = val-selected best head, no sklearn)")
    print(f"{'='*90}")
    print(f"{'Task':<25} {'Attentive PA':<14} {'sklearn ens.':<14}")
    print("-" * 53)
    for r in results:
        if r["type"] == "classification":
            print(f"{r['task']:<25} {r['direct_pa']:<14.4f} {r['sklearn_30d_pa']:<14.4f}")
        else:
            print(f"{r['task']:<25} {r['direct_pa_r2']:<14.4f} {r['sklearn_15d_r2']:<14.4f}")


if __name__ == "__main__":
    main()
