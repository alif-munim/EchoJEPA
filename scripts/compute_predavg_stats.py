#!/usr/bin/env python3
"""Compute statistics from study_predictions.csv files (prediction averaging results).

Usage:
    python scripts/compute_predavg_stats.py                              # point estimates
    python scripts/compute_predavg_stats.py --bootstrap 2000             # + 95% CIs
    python scripts/compute_predavg_stats.py --compare                    # + pairwise DeLong / bootstrap
    python scripts/compute_predavg_stats.py --confusion                  # + confusion matrices
    python scripts/compute_predavg_stats.py --plot figures/predavg       # generate all plots
    python scripts/compute_predavg_stats.py --task mr_severity --plot .  # plots for one task
    python scripts/compute_predavg_stats.py --csv output.csv             # save to CSV

Scans for study_predictions.csv under the results directory, auto-detects
classification vs regression, and computes standard metrics plus optional
bootstrap CIs, DeLong tests, Bland-Altman analysis, calibration, and plots.
"""

import argparse
import concurrent.futures
import itertools
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    average_precision_score,
)


DEFAULT_RESULTS_DIR = "evals/vitg-384/nature_medicine/uhn/video_classification_frozen"


# ============================================================
# DeLong test for comparing two AUROCs (binary only)
# ============================================================

def _delong_structural_components(y_true, y_score):
    """Compute DeLong structural components V10, V01 for a single predictor.

    V10[i] = fraction of negatives that positive i beats (+ 0.5 for ties).
    V01[j] = fraction of positives that beat negative j (+ 0.5 for ties).

    Uses vectorized broadcasting. Memory: O(m*n) where m=positives, n=negatives.
    For 20K studies this is fine; for 100K+ consider the rank-based method.
    """
    pos_scores = y_score[y_true == 1]
    neg_scores = y_score[y_true == 0]
    m, n = len(pos_scores), len(neg_scores)
    if m == 0 or n == 0:
        return np.nan, np.array([]), np.array([])

    # (m, n) comparison matrix
    comp = (pos_scores[:, None] > neg_scores[None, :]).astype(np.float64)
    comp += 0.5 * (pos_scores[:, None] == neg_scores[None, :]).astype(np.float64)

    V10 = comp.mean(axis=1)   # (m,) — per-positive placement values
    V01 = comp.mean(axis=0)   # (n,) — per-negative placement values
    auc = V10.mean()
    return auc, V10, V01


def delong_test(y_true, scores_a, scores_b):
    """DeLong test comparing two AUROCs on the same binary dataset.

    Returns (auc_a, auc_b, z_stat, p_value).
    Reference: DeLong et al. (1988), Biometrics 44(3):837-845.
    """
    auc_a, V10_a, V01_a = _delong_structural_components(y_true, scores_a)
    auc_b, V10_b, V01_b = _delong_structural_components(y_true, scores_b)

    if np.isnan(auc_a) or np.isnan(auc_b):
        return auc_a, auc_b, np.nan, np.nan

    m = (y_true == 1).sum()
    n = (y_true == 0).sum()

    # Covariance of structural components
    S10 = np.cov(np.stack([V10_a, V10_b]))  # (2, 2)
    S01 = np.cov(np.stack([V01_a, V01_b]))  # (2, 2)

    # Variance of AUC_a - AUC_b
    var_diff = (S10[0, 0] + S10[1, 1] - 2 * S10[0, 1]) / m + \
               (S01[0, 0] + S01[1, 1] - 2 * S01[0, 1]) / n

    if var_diff <= 0:
        return auc_a, auc_b, 0.0, 1.0

    z = (auc_a - auc_b) / np.sqrt(var_diff)
    p = 2 * scipy_stats.norm.sf(abs(z))
    return auc_a, auc_b, z, p


# ============================================================
# Bootstrap CIs
# ============================================================

def bootstrap_ci(values, labels, metric_fn, n_boot=2000, ci=0.95, seed=42):
    """Bootstrap a metric over (values, labels) pairs, return (lo, hi) CI.

    metric_fn(labels, values) -> scalar.
    Kept serial — outer loops (per-model, per-pair) are parallelized instead.
    """
    rng = np.random.RandomState(seed)
    n = len(values)
    alpha = (1 - ci) / 2
    stats = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        try:
            s = metric_fn(labels[idx], values[idx])
            if not np.isnan(s):
                stats.append(s)
        except (ValueError, IndexError):
            continue
    if len(stats) < 10:
        return np.nan, np.nan
    stats = np.array(stats)
    return float(np.percentile(stats, 100 * alpha)), float(np.percentile(stats, 100 * (1 - alpha)))


# ============================================================
# Calibration (ECE)
# ============================================================

def expected_calibration_error(labels, probs, n_bins=10):
    """Compute Expected Calibration Error for binary or multi-class.

    For multi-class, uses the predicted-class probability (confidence).
    """
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = accuracies[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += mask.sum() / len(labels) * abs(bin_acc - bin_conf)
    return ece


# ============================================================
# Bland-Altman (bias + limits of agreement)
# ============================================================

def bland_altman(labels, predictions):
    """Compute bias and 95% limits of agreement.

    Returns (bias, loa_lower, loa_upper, sd_diff).
    bias = mean(pred - label), LoA = bias ± 1.96 * SD(pred - label).
    """
    diff = predictions - labels
    bias = float(diff.mean())
    sd = float(diff.std(ddof=1))
    return bias, bias - 1.96 * sd, bias + 1.96 * sd, sd


# ============================================================
# Metric computation
# ============================================================

def parse_tag(dirname: str):
    """Parse '{task}-predavg-{model}' directory name into (task, model)."""
    if "-predavg-" not in dirname:
        return None, None
    parts = dirname.split("-predavg-")
    return parts[0], parts[1]


def compute_classification_metrics(df: pd.DataFrame, n_boot=0) -> dict:
    """Compute metrics for classification study_predictions.csv."""
    labels = df["label"].values
    prob_cols = [c for c in df.columns if c.startswith("prob_class_")]
    n_classes = len(prob_cols)
    probs = df[prob_cols].values
    preds = df["predicted_class"].values

    metrics = {
        "type": "classification",
        "n_studies": len(df),
        "n_classes": n_classes,
        "accuracy": accuracy_score(labels, preds),
        "balanced_accuracy": balanced_accuracy_score(labels, preds),
        "kappa": cohen_kappa_score(labels, preds),
        "ece": expected_calibration_error(labels, probs),
    }

    # AUROC (one-vs-rest)
    try:
        if n_classes == 2:
            metrics["auroc"] = roc_auc_score(labels, probs[:, 1])
            metrics["auprc"] = average_precision_score(labels, probs[:, 1])
            metrics["prevalence"] = float(labels.mean())
        else:
            metrics["auroc"] = roc_auc_score(
                labels, probs, multi_class="ovr", average="macro"
            )
            for c in range(n_classes):
                binary_labels = (labels == c).astype(int)
                if 0 < binary_labels.sum() < len(binary_labels):
                    metrics[f"auroc_class_{c}"] = roc_auc_score(binary_labels, probs[:, c])
    except ValueError as e:
        metrics["auroc"] = np.nan
        metrics["auroc_error"] = str(e)

    # Bootstrap CIs
    if n_boot > 0 and not np.isnan(metrics.get("auroc", np.nan)):
        if n_classes == 2:
            lo, hi = bootstrap_ci(
                probs[:, 1], labels,
                lambda y, s: roc_auc_score(y, s),
                n_boot=n_boot,
            )
            metrics["auroc_ci_lo"] = lo
            metrics["auroc_ci_hi"] = hi

            lo, hi = bootstrap_ci(
                probs[:, 1], labels,
                lambda y, s: average_precision_score(y, s),
                n_boot=n_boot,
            )
            metrics["auprc_ci_lo"] = lo
            metrics["auprc_ci_hi"] = hi
        else:
            lo, hi = bootstrap_ci(
                probs, labels,
                lambda y, s: roc_auc_score(y, s, multi_class="ovr", average="macro"),
                n_boot=n_boot,
            )
            metrics["auroc_ci_lo"] = lo
            metrics["auroc_ci_hi"] = hi

    # Class distribution
    unique, counts = np.unique(labels, return_counts=True)
    for u, c in zip(unique, counts):
        metrics[f"n_class_{int(u)}"] = int(c)

    metrics["median_clips"] = float(df["n_clips"].median())
    return metrics


def compute_regression_metrics(df: pd.DataFrame, n_boot=0) -> dict:
    """Compute metrics for regression study_predictions.csv."""
    labels = df["label"].values
    preds = df["prediction"].values

    r2 = r2_score(labels, preds)
    pearson_r, pearson_p = scipy_stats.pearsonr(labels, preds)
    mae = mean_absolute_error(labels, preds)
    rmse = float(np.sqrt(np.mean((labels - preds) ** 2)))
    bias, loa_lo, loa_hi, sd_diff = bland_altman(labels, preds)

    metrics = {
        "type": "regression",
        "n_studies": len(df),
        "r2": r2,
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "loa_lower": loa_lo,
        "loa_upper": loa_hi,
        "sd_diff": sd_diff,
        "label_mean": float(labels.mean()),
        "label_std": float(labels.std()),
        "pred_mean": float(preds.mean()),
        "pred_std": float(preds.std()),
        "median_clips": float(df["n_clips"].median()),
    }

    # Bootstrap CIs
    if n_boot > 0:
        lo, hi = bootstrap_ci(
            preds, labels, lambda y, s: r2_score(y, s), n_boot=n_boot,
        )
        metrics["r2_ci_lo"] = lo
        metrics["r2_ci_hi"] = hi

        lo, hi = bootstrap_ci(
            preds, labels, lambda y, s: scipy_stats.pearsonr(y, s)[0], n_boot=n_boot,
        )
        metrics["pearson_ci_lo"] = lo
        metrics["pearson_ci_hi"] = hi

        lo, hi = bootstrap_ci(
            preds, labels, lambda y, s: mean_absolute_error(y, s), n_boot=n_boot,
        )
        metrics["mae_ci_lo"] = lo
        metrics["mae_ci_hi"] = hi

    return metrics


def process_file(csv_path: str, n_boot: int = 0) -> dict:
    """Load study_predictions.csv and compute appropriate metrics."""
    df = pd.read_csv(csv_path)
    if "prediction" in df.columns:
        return compute_regression_metrics(df, n_boot=n_boot)
    elif "predicted_class" in df.columns:
        return compute_classification_metrics(df, n_boot=n_boot)
    else:
        return {"type": "unknown", "error": f"Unrecognized columns: {list(df.columns)}"}


# ============================================================
# Pairwise model comparisons
# ============================================================

def _pairwise_regression_bootstrap(labels, preds_a, preds_b, n_boot, seed=42):
    """Bootstrap R² difference for a regression pair. Runs in a worker process."""
    rng = np.random.RandomState(seed)
    n = len(labels)
    diffs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        d = r2_score(labels[idx], preds_a[idx]) - r2_score(labels[idx], preds_b[idx])
        diffs.append(d)
    diffs = np.array(diffs)
    diff_mean = diffs.mean()
    p_val = 2 * min(np.mean(diffs > 0), np.mean(diffs < 0))
    ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
    return diff_mean, ci_lo, ci_hi, p_val


def _pairwise_multiclass_bootstrap(labels, probs_a, probs_b, n_boot, seed=42):
    """Bootstrap multi-class AUROC difference for a pair. Runs in a worker process."""
    rng = np.random.RandomState(seed)
    n = len(labels)
    diffs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        try:
            d = roc_auc_score(labels[idx], probs_a[idx], multi_class="ovr", average="macro") - \
                roc_auc_score(labels[idx], probs_b[idx], multi_class="ovr", average="macro")
            diffs.append(d)
        except ValueError:
            continue
    diffs = np.array(diffs)
    if len(diffs) == 0:
        return np.nan, np.nan, np.nan, np.nan
    diff_mean = diffs.mean()
    p_val = 2 * min(np.mean(diffs > 0), np.mean(diffs < 0))
    ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
    return diff_mean, ci_lo, ci_hi, p_val


def run_pairwise_comparisons(results_dir: Path, task_filter=None, n_boot=2000):
    """Run DeLong (binary) or bootstrap (multi-class/regression) pairwise comparisons."""
    n_workers = min(os.cpu_count() or 1, 12)

    # Group files by task
    task_files = {}
    for subdir in sorted(results_dir.iterdir()):
        if not subdir.is_dir():
            continue
        task, model = parse_tag(subdir.name)
        if task is None:
            continue
        if task_filter and task_filter not in task:
            continue
        csv_path = subdir / "study_predictions.csv"
        if not csv_path.exists():
            continue
        task_files.setdefault(task, {})[model] = csv_path

    for task, model_paths in sorted(task_files.items()):
        models = sorted(model_paths.keys())
        if len(models) < 2:
            continue

        # Load all model predictions, align by study_id
        dfs = {}
        for m in models:
            dfs[m] = pd.read_csv(model_paths[m])

        is_regression = "prediction" in dfs[models[0]].columns
        is_binary = False
        if not is_regression:
            prob_cols = [c for c in dfs[models[0]].columns if c.startswith("prob_class_")]
            is_binary = len(prob_cols) == 2

        print(f"\n{'='*80}")
        task_type = "regression" if is_regression else f"{'binary' if is_binary else 'multi-class'} classification"
        print(f"PAIRWISE COMPARISONS: {task} ({task_type})")
        print(f"{'='*80}")

        pairs = list(itertools.combinations(models, 2))
        rows = []

        if is_binary:
            # DeLong is fast (no bootstrap), run serially
            for m_a, m_b in pairs:
                merged = dfs[m_a].merge(dfs[m_b], on="study_id", suffixes=("_a", "_b"))
                if len(merged) < 10:
                    print(f"  {m_a} vs {m_b}: only {len(merged)} shared studies, skipping")
                    continue
                labels = merged["label_a"].values
                scores_a = merged["prob_class_1_a"].values
                scores_b = merged["prob_class_1_b"].values
                auc_a, auc_b, z, p_val = delong_test(labels, scores_a, scores_b)
                rows.append({
                    "model_a": m_a, "model_b": m_b, "n_shared": len(merged),
                    "metric_a": f"AUC={auc_a:.4f}", "metric_b": f"AUC={auc_b:.4f}",
                    "diff": f"{auc_a - auc_b:+.4f}", "z": f"{z:.2f}",
                    "p": p_val, "method": "DeLong",
                })

        elif is_regression:
            # Bootstrap R² difference — parallelize across pairs
            pair_data = []
            for m_a, m_b in pairs:
                merged = dfs[m_a].merge(dfs[m_b], on="study_id", suffixes=("_a", "_b"))
                if len(merged) < 10:
                    print(f"  {m_a} vs {m_b}: only {len(merged)} shared studies, skipping")
                    continue
                pair_data.append((m_a, m_b, merged))

            with concurrent.futures.ProcessPoolExecutor(max_workers=min(len(pair_data), n_workers)) as pool:
                futures = {}
                for m_a, m_b, merged in pair_data:
                    labels = merged["label_a"].values
                    preds_a = merged["prediction_a"].values
                    preds_b = merged["prediction_b"].values
                    f = pool.submit(_pairwise_regression_bootstrap, labels, preds_a, preds_b, n_boot)
                    futures[f] = (m_a, m_b, merged, labels, preds_a, preds_b)

                for f in concurrent.futures.as_completed(futures):
                    m_a, m_b, merged, labels, preds_a, preds_b = futures[f]
                    diff_mean, ci_lo, ci_hi, p_val = f.result()
                    r2_a = r2_score(labels, preds_a)
                    r2_b = r2_score(labels, preds_b)
                    rows.append({
                        "model_a": m_a, "model_b": m_b, "n_shared": len(merged),
                        "metric_a": f"R²={r2_a:.4f}", "metric_b": f"R²={r2_b:.4f}",
                        "diff": f"{diff_mean:+.4f}", "ci": f"[{ci_lo:+.4f}, {ci_hi:+.4f}]",
                        "p": p_val,
                    })

        else:
            # Multi-class bootstrap — parallelize across pairs
            pair_data = []
            for m_a, m_b in pairs:
                merged = dfs[m_a].merge(dfs[m_b], on="study_id", suffixes=("_a", "_b"))
                if len(merged) < 10:
                    print(f"  {m_a} vs {m_b}: only {len(merged)} shared studies, skipping")
                    continue
                labels = merged["label_a"].values
                prob_cols_a = [c for c in merged.columns if c.startswith("prob_class_") and c.endswith("_a")]
                prob_cols_b = [c for c in merged.columns if c.startswith("prob_class_") and c.endswith("_b")]
                probs_a = merged[prob_cols_a].values
                probs_b = merged[prob_cols_b].values
                try:
                    auc_a = roc_auc_score(labels, probs_a, multi_class="ovr", average="macro")
                    auc_b = roc_auc_score(labels, probs_b, multi_class="ovr", average="macro")
                except ValueError:
                    continue
                pair_data.append((m_a, m_b, merged, labels, probs_a, probs_b, auc_a, auc_b))

            with concurrent.futures.ProcessPoolExecutor(max_workers=min(len(pair_data) or 1, n_workers)) as pool:
                futures = {}
                for m_a, m_b, merged, labels, probs_a, probs_b, auc_a, auc_b in pair_data:
                    f = pool.submit(_pairwise_multiclass_bootstrap, labels, probs_a, probs_b, n_boot)
                    futures[f] = (m_a, m_b, merged, auc_a, auc_b)

                for f in concurrent.futures.as_completed(futures):
                    m_a, m_b, merged, auc_a, auc_b = futures[f]
                    diff_mean, ci_lo, ci_hi, p_val = f.result()
                    rows.append({
                        "model_a": m_a, "model_b": m_b, "n_shared": len(merged),
                        "metric_a": f"AUC={auc_a:.4f}", "metric_b": f"AUC={auc_b:.4f}",
                        "diff": f"{diff_mean:+.4f}", "ci": f"[{ci_lo:+.4f}, {ci_hi:+.4f}]",
                        "p": p_val, "method": "bootstrap",
                    })

        if rows:
            comp_df = pd.DataFrame(rows)
            comp_df["sig"] = comp_df["p"].apply(
                lambda x: "***" if x < 0.001 else ("**" if x < 0.01 else ("*" if x < 0.05 else ""))
            )
            display_cols = [c for c in ["model_a", "model_b", "n_shared", "metric_a", "metric_b",
                                        "diff", "ci", "z", "p", "sig", "method"] if c in comp_df.columns]
            print(comp_df[display_cols].to_string(index=False, float_format="%.4f"))


# ============================================================
# Confusion matrix display
# ============================================================

def print_confusion_matrices(results_dir: Path, task_filter=None, model_filter=None):
    """Print confusion matrices for classification tasks."""
    for subdir in sorted(results_dir.iterdir()):
        if not subdir.is_dir():
            continue
        task, model = parse_tag(subdir.name)
        if task is None:
            continue
        if task_filter and task_filter not in task:
            continue
        if model_filter and model_filter not in model:
            continue
        csv_path = subdir / "study_predictions.csv"
        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path)
        if "predicted_class" not in df.columns:
            continue

        labels = df["label"].values
        preds = df["predicted_class"].values
        classes = sorted(np.unique(np.concatenate([labels, preds])))
        cm = confusion_matrix(labels, preds, labels=classes)

        print(f"\n--- {task} / {model} ---")
        print(f"{'':>8s}", end="")
        for c in classes:
            print(f"  pred_{c:d}", end="")
        print("   recall")
        for i, c in enumerate(classes):
            row_sum = cm[i].sum()
            recall = cm[i, i] / row_sum if row_sum > 0 else 0
            print(f"true_{c:<3d}", end="")
            for j in range(len(classes)):
                print(f"  {cm[i, j]:6d}", end="")
            print(f"   {recall:.3f}")
        print(f"{'prec':>8s}", end="")
        for j in range(len(classes)):
            col_sum = cm[:, j].sum()
            prec = cm[j, j] / col_sum if col_sum > 0 else 0
            print(f"  {prec:6.3f}", end="")
        print()


# ============================================================
# Plotting
# ============================================================

# Colorblind-friendly palette (Wong 2011, Nature Methods)
MODEL_COLORS = {
    "echojepa-g": "#0072B2",     # blue
    "echojepa-l-k": "#009E73",   # green
    "echojepa-l": "#56B4E9",     # light blue
    "echojepa-b": "#CC79A7",     # pink
    "echoprime": "#E69F00",      # orange
    "panecho": "#D55E00",        # vermillion
}
MODEL_ORDER = ["echojepa-g", "echojepa-l-k", "echojepa-l", "echojepa-b", "echoprime", "panecho"]

MODEL_LABELS = {
    "echojepa-g": "EchoJEPA-G",
    "echojepa-l-k": "EchoJEPA-L-K",
    "echojepa-l": "EchoJEPA-L",
    "echojepa-b": "EchoJEPA-B",
    "echoprime": "EchoPrime",
    "panecho": "PanEcho",
}

TASK_LABELS = {
    "lvef": "LVEF (%)",
    "tapse": "TAPSE (cm)",
    "rvsp": "RVSP (mmHg)",
    "rv_sp": "RV s' (m/s)",
    "rv_fac": "RV FAC (%)",
    "edv": "EDV (mL)",
    "esv": "ESV (mL)",
    "aov_vmax": "AoV Vmax (m/s)",
    "aov_mean_grad": "AoV Mean Gradient (mmHg)",
    "mv_ee_medial": "MV e' Medial (cm/s)",
    "cardiac_output": "Cardiac Output (L/min)",
    "mr_severity": "MR Severity",
    "as_severity": "AS Severity",
    "ar_severity": "AR Severity",
    "tr_severity": "TR Severity",
    "diastolic_function": "Diastolic Function",
    "disease_hcm": "HCM",
    "disease_dcm": "DCM",
    "disease_amyloidosis": "Amyloidosis",
    "disease_bicuspid_av": "Bicuspid AV",
    "disease_myxomatous_mv": "Myxomatous MV",
    "disease_rheumatic_mv": "Rheumatic MV",
    "disease_stemi": "STEMI",
    "trajectory_lvef": "LVEF Trajectory",
    "trajectory_lvef_onset": "New-onset Cardiomyopathy",
    "trajectory_mr_severity_onset": "MR Progression",
}


def _setup_style():
    """Set publication-quality matplotlib defaults."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    return plt


def _get_color(model):
    return MODEL_COLORS.get(model, "#999999")


def _get_label(model):
    return MODEL_LABELS.get(model, model)


def _task_label(task):
    return TASK_LABELS.get(task, task.replace("_", " ").title())


def _ordered_models(models):
    """Sort models by the canonical order."""
    return sorted(models, key=lambda m: MODEL_ORDER.index(m) if m in MODEL_ORDER else 99)


def _load_task_data(results_dir: Path, task_filter=None, model_filter=None):
    """Load all study_predictions.csv, grouped by task -> model -> DataFrame."""
    task_data = {}
    for subdir in sorted(results_dir.iterdir()):
        if not subdir.is_dir():
            continue
        task, model = parse_tag(subdir.name)
        if task is None:
            continue
        if task_filter and task_filter not in task:
            continue
        if model_filter and model_filter not in model:
            continue
        csv_path = subdir / "study_predictions.csv"
        if not csv_path.exists():
            continue
        task_data.setdefault(task, {})[model] = pd.read_csv(csv_path)
    return task_data


def plot_roc_curves(task_data, out_dir, plt):
    """ROC curves per classification task, one curve per model."""
    from sklearn.metrics import roc_curve

    for task, model_dfs in sorted(task_data.items()):
        sample_df = next(iter(model_dfs.values()))
        if "predicted_class" not in sample_df.columns:
            continue
        prob_cols = [c for c in sample_df.columns if c.startswith("prob_class_")]
        n_classes = len(prob_cols)

        fig, ax = plt.subplots(figsize=(5, 5))
        models = _ordered_models(model_dfs.keys())

        for model in models:
            df = model_dfs[model]
            labels = df["label"].values
            probs = df[prob_cols].values

            if n_classes == 2:
                fpr, tpr, _ = roc_curve(labels, probs[:, 1])
                auc = roc_auc_score(labels, probs[:, 1])
                ax.plot(fpr, tpr, color=_get_color(model), lw=1.5,
                        label=f"{_get_label(model)} ({auc:.3f})")
            else:
                # Macro-average ROC: compute per-class, then average
                from sklearn.preprocessing import label_binarize
                classes = list(range(n_classes))
                y_bin = label_binarize(labels, classes=classes)
                all_fpr = np.linspace(0, 1, 200)
                mean_tpr = np.zeros_like(all_fpr)
                for c in classes:
                    if y_bin[:, c].sum() == 0 or y_bin[:, c].sum() == len(y_bin):
                        continue
                    fpr_c, tpr_c, _ = roc_curve(y_bin[:, c], probs[:, c])
                    mean_tpr += np.interp(all_fpr, fpr_c, tpr_c)
                mean_tpr /= n_classes
                try:
                    auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
                except ValueError:
                    auc = np.nan
                ax.plot(all_fpr, mean_tpr, color=_get_color(model), lw=1.5,
                        label=f"{_get_label(model)} ({auc:.3f})")

        ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{_task_label(task)} — ROC")
        ax.legend(loc="lower right")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")
        fig.savefig(out_dir / f"roc_{task}.pdf")
        fig.savefig(out_dir / f"roc_{task}.png")
        plt.close(fig)
        print(f"  roc_{task}.pdf")


def plot_pr_curves(task_data, out_dir, plt):
    """Precision-recall curves for binary classification tasks."""
    from sklearn.metrics import precision_recall_curve

    for task, model_dfs in sorted(task_data.items()):
        sample_df = next(iter(model_dfs.values()))
        if "predicted_class" not in sample_df.columns:
            continue
        prob_cols = [c for c in sample_df.columns if c.startswith("prob_class_")]
        if len(prob_cols) != 2:
            continue

        fig, ax = plt.subplots(figsize=(5, 5))
        models = _ordered_models(model_dfs.keys())

        for model in models:
            df = model_dfs[model]
            labels = df["label"].values
            probs = df[prob_cols].values[:, 1]
            prec, rec, _ = precision_recall_curve(labels, probs)
            ap = average_precision_score(labels, probs)
            ax.plot(rec, prec, color=_get_color(model), lw=1.5,
                    label=f"{_get_label(model)} ({ap:.3f})")

        prevalence = next(iter(model_dfs.values()))["label"].mean()
        ax.axhline(prevalence, color="gray", ls="--", lw=0.8, alpha=0.5, label=f"Prevalence ({prevalence:.3f})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"{_task_label(task)} — Precision-Recall")
        ax.legend(loc="upper right", fontsize=7)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.05)
        fig.savefig(out_dir / f"pr_{task}.pdf")
        fig.savefig(out_dir / f"pr_{task}.png")
        plt.close(fig)
        print(f"  pr_{task}.pdf")


def plot_calibration(task_data, out_dir, plt, n_bins=10):
    """Calibration (reliability) diagrams for classification tasks."""
    for task, model_dfs in sorted(task_data.items()):
        sample_df = next(iter(model_dfs.values()))
        if "predicted_class" not in sample_df.columns:
            continue
        prob_cols = [c for c in sample_df.columns if c.startswith("prob_class_")]

        fig, ax = plt.subplots(figsize=(5, 5))
        models = _ordered_models(model_dfs.keys())

        for model in models:
            df = model_dfs[model]
            labels = df["label"].values
            probs = df[prob_cols].values
            confidences = probs.max(axis=1)
            predictions = probs.argmax(axis=1)
            correct = (predictions == labels).astype(float)

            bin_edges = np.linspace(0, 1, n_bins + 1)
            bin_centers = []
            bin_accs = []
            for i in range(n_bins):
                mask = (confidences > bin_edges[i]) & (confidences <= bin_edges[i + 1])
                if mask.sum() < 5:
                    continue
                bin_centers.append(confidences[mask].mean())
                bin_accs.append(correct[mask].mean())

            ece = expected_calibration_error(labels, probs, n_bins=n_bins)
            ax.plot(bin_centers, bin_accs, "o-", color=_get_color(model), lw=1.5, ms=4,
                    label=f"{_get_label(model)} (ECE={ece:.3f})")

        ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4, label="Perfect")
        ax.set_xlabel("Mean Predicted Confidence")
        ax.set_ylabel("Observed Accuracy")
        ax.set_title(f"{_task_label(task)} — Calibration")
        ax.legend(loc="upper left", fontsize=7)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.05)
        ax.set_aspect("equal")
        fig.savefig(out_dir / f"calibration_{task}.pdf")
        fig.savefig(out_dir / f"calibration_{task}.png")
        plt.close(fig)
        print(f"  calibration_{task}.pdf")


def plot_confusion_heatmaps(task_data, out_dir, plt):
    """Confusion matrix heatmaps for classification tasks."""
    for task, model_dfs in sorted(task_data.items()):
        sample_df = next(iter(model_dfs.values()))
        if "predicted_class" not in sample_df.columns:
            continue

        models = _ordered_models(model_dfs.keys())
        n_models = len(models)
        fig, axes = plt.subplots(1, n_models, figsize=(3.5 * n_models, 3.5), squeeze=False)

        for i, model in enumerate(models):
            ax = axes[0, i]
            df = model_dfs[model]
            labels = df["label"].values
            preds = df["predicted_class"].values
            classes = sorted(np.unique(np.concatenate([labels, preds])))
            cm = confusion_matrix(labels, preds, labels=classes)
            # Normalize by row (true class)
            cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            cm_norm = np.nan_to_num(cm_norm)

            im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="equal")
            ax.set_xticks(range(len(classes)))
            ax.set_yticks(range(len(classes)))
            ax.set_xticklabels(classes, fontsize=8)
            ax.set_yticklabels(classes, fontsize=8)
            ax.set_xlabel("Predicted")
            if i == 0:
                ax.set_ylabel("True")
            ax.set_title(_get_label(model), fontsize=9)

            # Annotate cells
            for r in range(len(classes)):
                for c_idx in range(len(classes)):
                    val = cm_norm[r, c_idx]
                    color = "white" if val > 0.5 else "black"
                    ax.text(c_idx, r, f"{val:.2f}", ha="center", va="center",
                            fontsize=7, color=color)

        fig.suptitle(f"{_task_label(task)} — Confusion Matrices (row-normalized)", fontsize=11)
        fig.tight_layout()
        fig.savefig(out_dir / f"confusion_{task}.pdf")
        fig.savefig(out_dir / f"confusion_{task}.png")
        plt.close(fig)
        print(f"  confusion_{task}.pdf")


def plot_scatter(task_data, out_dir, plt):
    """Predicted vs true scatter plots for regression tasks."""
    for task, model_dfs in sorted(task_data.items()):
        sample_df = next(iter(model_dfs.values()))
        if "prediction" not in sample_df.columns:
            continue

        models = _ordered_models(model_dfs.keys())
        n_models = len(models)
        ncols = min(n_models, 3)
        nrows = (n_models + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)

        for i, model in enumerate(models):
            ax = axes[i // ncols, i % ncols]
            df = model_dfs[model]
            labels = df["label"].values
            preds = df["prediction"].values
            r2 = r2_score(labels, preds)

            ax.scatter(labels, preds, s=1, alpha=0.15, color=_get_color(model), rasterized=True)
            lims = [min(labels.min(), preds.min()), max(labels.max(), preds.max())]
            margin = (lims[1] - lims[0]) * 0.05
            lims = [lims[0] - margin, lims[1] + margin]
            ax.plot(lims, lims, "k--", lw=0.8, alpha=0.4)
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_xlabel("True")
            ax.set_ylabel("Predicted")
            ax.set_title(f"{_get_label(model)} (R\u00b2={r2:.3f})", fontsize=9)
            ax.set_aspect("equal")

        # Hide empty subplots
        for j in range(n_models, nrows * ncols):
            axes[j // ncols, j % ncols].set_visible(False)

        fig.suptitle(f"{_task_label(task)} — Predicted vs True", fontsize=11)
        fig.tight_layout()
        fig.savefig(out_dir / f"scatter_{task}.pdf")
        fig.savefig(out_dir / f"scatter_{task}.png")
        plt.close(fig)
        print(f"  scatter_{task}.pdf")


def plot_bland_altman(task_data, out_dir, plt):
    """Bland-Altman plots for regression tasks."""
    for task, model_dfs in sorted(task_data.items()):
        sample_df = next(iter(model_dfs.values()))
        if "prediction" not in sample_df.columns:
            continue

        models = _ordered_models(model_dfs.keys())
        n_models = len(models)
        ncols = min(n_models, 3)
        nrows = (n_models + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows), squeeze=False)

        for i, model in enumerate(models):
            ax = axes[i // ncols, i % ncols]
            df = model_dfs[model]
            labels = df["label"].values
            preds = df["prediction"].values
            mean_vals = (labels + preds) / 2
            diff = preds - labels

            bias = diff.mean()
            sd = diff.std(ddof=1)
            loa_lo = bias - 1.96 * sd
            loa_hi = bias + 1.96 * sd

            ax.scatter(mean_vals, diff, s=1, alpha=0.15, color=_get_color(model), rasterized=True)
            ax.axhline(bias, color="black", lw=1, ls="-", label=f"Bias={bias:.2f}")
            ax.axhline(loa_hi, color="gray", lw=0.8, ls="--", label=f"+1.96 SD={loa_hi:.2f}")
            ax.axhline(loa_lo, color="gray", lw=0.8, ls="--", label=f"-1.96 SD={loa_lo:.2f}")
            ax.axhline(0, color="black", lw=0.5, alpha=0.3)
            ax.set_xlabel("Mean of True & Predicted")
            ax.set_ylabel("Predicted - True")
            ax.set_title(_get_label(model), fontsize=9)
            ax.legend(fontsize=6, loc="upper right")

        for j in range(n_models, nrows * ncols):
            axes[j // ncols, j % ncols].set_visible(False)

        fig.suptitle(f"{_task_label(task)} — Bland-Altman", fontsize=11)
        fig.tight_layout()
        fig.savefig(out_dir / f"bland_altman_{task}.pdf")
        fig.savefig(out_dir / f"bland_altman_{task}.png")
        plt.close(fig)
        print(f"  bland_altman_{task}.pdf")


def plot_heatmap(task_data, metrics_rows, out_dir, plt):
    """Cross-task heatmap: models x tasks colored by primary metric."""
    cls_rows = [r for r in metrics_rows if r["type"] == "classification"]
    reg_rows = [r for r in metrics_rows if r["type"] == "regression"]

    for label, rows, metric, cmap, vrange in [
        ("AUROC", cls_rows, "auroc", "YlOrRd", (0.5, 1.0)),
        ("R\u00b2", reg_rows, "r2", "YlGnBu", (0.0, 1.0)),
    ]:
        if not rows:
            continue
        df = pd.DataFrame(rows)
        tasks = sorted(df["task"].unique(), key=lambda t: df[df["task"] == t][metric].max(), reverse=True)
        models = _ordered_models(df["model"].unique())

        pivot = df.pivot_table(index="task", columns="model", values=metric)
        pivot = pivot.reindex(index=tasks, columns=models)

        fig, ax = plt.subplots(figsize=(max(6, len(models) * 1.2), max(4, len(tasks) * 0.45)))
        im = ax.imshow(pivot.values, cmap=cmap, vmin=vrange[0], vmax=vrange[1], aspect="auto")

        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([_get_label(m) for m in models], rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(tasks)))
        ax.set_yticklabels([_task_label(t) for t in tasks], fontsize=8)

        for r in range(len(tasks)):
            for c_idx in range(len(models)):
                val = pivot.values[r, c_idx]
                if np.isnan(val):
                    ax.text(c_idx, r, "---", ha="center", va="center", fontsize=7, color="gray")
                else:
                    color = "white" if val > (vrange[0] + vrange[1]) / 2 else "black"
                    ax.text(c_idx, r, f"{val:.3f}", ha="center", va="center", fontsize=7, color=color)

        fig.colorbar(im, ax=ax, shrink=0.8, label=label)
        ax.set_title(f"Model Comparison — {label}", fontsize=11)
        fig.tight_layout()
        tag = "auroc" if metric == "auroc" else "r2"
        fig.savefig(out_dir / f"heatmap_{tag}.pdf")
        fig.savefig(out_dir / f"heatmap_{tag}.png")
        plt.close(fig)
        print(f"  heatmap_{tag}.pdf")


def plot_forest(metrics_rows, out_dir, plt):
    """Forest plots: per-model metric with CI whiskers, grouped by task."""
    cls_rows = [r for r in metrics_rows if r["type"] == "classification" and "auroc_ci_lo" in r]
    reg_rows = [r for r in metrics_rows if r["type"] == "regression" and "r2_ci_lo" in r]

    for label, rows, metric, ci_lo_key, ci_hi_key in [
        ("AUROC", cls_rows, "auroc", "auroc_ci_lo", "auroc_ci_hi"),
        ("R\u00b2", reg_rows, "r2", "r2_ci_lo", "r2_ci_hi"),
    ]:
        if not rows:
            continue
        df = pd.DataFrame(rows)
        tasks = sorted(df["task"].unique())

        fig, ax = plt.subplots(figsize=(8, max(4, len(tasks) * 0.6)))
        y_pos = 0
        y_ticks = []
        y_labels = []

        for task in tasks:
            group = df[df["task"] == task]
            models = _ordered_models(group["model"].unique())
            for model in reversed(models):
                row = group[group["model"] == model].iloc[0]
                val = row[metric]
                lo = row.get(ci_lo_key, val)
                hi = row.get(ci_hi_key, val)
                ax.errorbar(val, y_pos, xerr=[[val - lo], [hi - val]],
                            fmt="o", ms=5, color=_get_color(model), capsize=3, lw=1.2,
                            label=_get_label(model) if task == tasks[0] else None)
                y_ticks.append(y_pos)
                y_labels.append(f"{_get_label(model)}")
                y_pos += 1
            # Add task separator
            if task != tasks[-1]:
                ax.axhline(y_pos - 0.5, color="gray", lw=0.3, alpha=0.5)
            y_pos += 0.5

        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels, fontsize=7)
        ax.set_xlabel(label)
        ax.set_title(f"Forest Plot — {label} with 95% CI", fontsize=11)
        ax.invert_yaxis()

        # Add task labels on the right
        y_pos = 0
        for task in tasks:
            n_models = len(df[df["task"] == task]["model"].unique())
            mid = y_pos + (n_models - 1) / 2
            ax.text(ax.get_xlim()[1] + 0.005, mid, _task_label(task),
                    fontsize=7, va="center", ha="left", style="italic")
            y_pos += n_models + 0.5

        fig.tight_layout()
        tag = "auroc" if metric == "auroc" else "r2"
        fig.savefig(out_dir / f"forest_{tag}.pdf")
        fig.savefig(out_dir / f"forest_{tag}.png")
        plt.close(fig)
        print(f"  forest_{tag}.pdf")


def plot_nclips_vs_performance(task_data, out_dir, plt):
    """Performance vs number of clips per study (does averaging help?)."""
    for task, model_dfs in sorted(task_data.items()):
        # Pick the best model (echojepa-g if available)
        model = "echojepa-g" if "echojepa-g" in model_dfs else next(iter(model_dfs.keys()))
        df = model_dfs[model]

        is_regression = "prediction" in df.columns
        if is_regression:
            labels = df["label"].values
            preds = df["prediction"].values
            errors = np.abs(preds - labels)
            ylabel = "Mean Absolute Error"
        else:
            prob_cols = [c for c in df.columns if c.startswith("prob_class_")]
            probs = df[prob_cols].values
            correct = (probs.argmax(axis=1) == df["label"].values).astype(float)
            errors = 1.0 - correct  # misclassification
            ylabel = "Error Rate"

        n_clips = df["n_clips"].values

        # Bin by n_clips quantiles
        n_bins = min(10, len(np.unique(n_clips)))
        try:
            bins = np.percentile(n_clips, np.linspace(0, 100, n_bins + 1))
            bins = np.unique(bins)
        except Exception:
            continue
        if len(bins) < 3:
            continue

        bin_centers = []
        bin_errors = []
        bin_counts = []
        for i in range(len(bins) - 1):
            mask = (n_clips >= bins[i]) & (n_clips < bins[i + 1])
            if i == len(bins) - 2:
                mask = (n_clips >= bins[i]) & (n_clips <= bins[i + 1])
            if mask.sum() < 10:
                continue
            bin_centers.append(n_clips[mask].mean())
            bin_errors.append(errors[mask].mean())
            bin_counts.append(mask.sum())

        if len(bin_centers) < 3:
            continue

        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.plot(bin_centers, bin_errors, "o-", color=_get_color(model), lw=1.5, ms=5)
        ax.set_xlabel("Number of Clips per Study")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{_task_label(task)} — {_get_label(model)}", fontsize=10)

        # Add bin counts as text
        for x, y, n in zip(bin_centers, bin_errors, bin_counts):
            ax.annotate(f"n={n}", (x, y), textcoords="offset points",
                        xytext=(0, 8), fontsize=6, ha="center", color="gray")

        fig.tight_layout()
        fig.savefig(out_dir / f"nclips_{task}.pdf")
        fig.savefig(out_dir / f"nclips_{task}.png")
        plt.close(fig)
        print(f"  nclips_{task}.pdf")


def generate_all_plots(results_dir, out_dir, metrics_rows, task_filter=None, model_filter=None):
    """Generate all plot types."""
    plt = _setup_style()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    task_data = _load_task_data(results_dir, task_filter=task_filter, model_filter=model_filter)
    if not task_data:
        print("No data found for plotting.")
        return

    print(f"\nGenerating plots in {out_dir}/")

    print("\n  ROC curves:")
    plot_roc_curves(task_data, out_dir, plt)

    print("\n  Precision-recall curves:")
    plot_pr_curves(task_data, out_dir, plt)

    print("\n  Calibration diagrams:")
    plot_calibration(task_data, out_dir, plt)

    print("\n  Confusion matrix heatmaps:")
    plot_confusion_heatmaps(task_data, out_dir, plt)

    print("\n  Scatter plots (predicted vs true):")
    plot_scatter(task_data, out_dir, plt)

    print("\n  Bland-Altman plots:")
    plot_bland_altman(task_data, out_dir, plt)

    print("\n  Cross-task heatmaps:")
    plot_heatmap(task_data, metrics_rows, out_dir, plt)

    print("\n  Forest plots (requires --bootstrap):")
    plot_forest(metrics_rows, out_dir, plt)

    print("\n  Performance vs n_clips:")
    plot_nclips_vs_performance(task_data, out_dir, plt)

    n_plots = len(list(out_dir.glob("*.pdf")))
    print(f"\nDone: {n_plots} PDF + PNG plots saved to {out_dir}/")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--results_dir", default=DEFAULT_RESULTS_DIR,
        help="Root directory containing *-predavg-* subdirectories",
    )
    parser.add_argument("--task", default=None, help="Filter to specific task (substring match)")
    parser.add_argument("--model", default=None, help="Filter to specific model (substring match)")
    parser.add_argument("--csv", default=None, help="Save results to CSV")
    parser.add_argument("--compact", action="store_true", help="Compact table output")
    parser.add_argument(
        "--bootstrap", type=int, default=0, metavar="N",
        help="Bootstrap iterations for 95%% CIs (0=disabled, typical: 2000)",
    )
    parser.add_argument("--compare", action="store_true", help="Pairwise model comparisons (DeLong + bootstrap)")
    parser.add_argument("--confusion", action="store_true", help="Print confusion matrices")
    parser.add_argument("--plot", metavar="DIR", default=None, help="Generate plots to DIR (PDF + PNG)")
    parser.add_argument("--workers", type=int, default=None, metavar="N",
                        help="Max parallel workers for bootstrap (default: min(cpu_count, 12))")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"ERROR: {results_dir} does not exist")
        sys.exit(1)

    n_boot = args.bootstrap
    n_workers = args.workers or min(os.cpu_count() or 1, 12)
    if n_boot > 0:
        print(f"Bootstrap CIs enabled: {n_boot} iterations (parallel: {n_workers} workers)")

    # Collect matching CSV paths
    jobs = []
    for subdir in sorted(results_dir.iterdir()):
        if not subdir.is_dir():
            continue
        task, model = parse_tag(subdir.name)
        if task is None:
            continue
        if args.task and args.task not in task:
            continue
        if args.model and args.model not in model:
            continue

        csv_path = subdir / "study_predictions.csv"
        if not csv_path.exists():
            continue
        jobs.append((str(csv_path), task, model))

    # Compute metrics — parallel across models when bootstrap is enabled
    rows = []
    if n_boot > 0 and len(jobs) > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=min(len(jobs), n_workers)) as pool:
            futures = {pool.submit(process_file, csv_path, n_boot): (task, model)
                       for csv_path, task, model in jobs}
            for f in concurrent.futures.as_completed(futures):
                task, model = futures[f]
                metrics = f.result()
                metrics["task"] = task
                metrics["model"] = model
                rows.append(metrics)
                print(f"  Done: {task} / {model}")
        # Restore deterministic ordering
        rows.sort(key=lambda r: (r["task"], r["model"]))
    else:
        for csv_path, task, model in jobs:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                metrics = process_file(csv_path, n_boot=n_boot)
            metrics["task"] = task
            metrics["model"] = model
            rows.append(metrics)

    if not rows:
        print("No study_predictions.csv files found matching filters.")
        sys.exit(0)

    # Separate classification and regression
    cls_rows = [r for r in rows if r["type"] == "classification"]
    reg_rows = [r for r in rows if r["type"] == "regression"]

    # Print classification results
    if cls_rows:
        print("=" * 110)
        print("CLASSIFICATION TASKS (study-level prediction averaging)")
        print("=" * 110)

        cls_df = pd.DataFrame(cls_rows)
        cls_df = cls_df.sort_values(["task", "model"])

        if args.compact:
            cols = ["task", "model", "n_studies", "n_classes", "auroc", "balanced_accuracy", "kappa", "ece"]
            if n_boot > 0:
                cols.insert(5, "auroc_ci_lo")
                cols.insert(6, "auroc_ci_hi")
            cols = [c for c in cols if c in cls_df.columns]
            print(cls_df[cols].to_string(index=False, float_format="%.4f"))
        else:
            for task_name, group in cls_df.groupby("task"):
                n_cls = group["n_classes"].iloc[0]
                print(f"\n--- {task_name} ({n_cls}-class, {group['n_studies'].iloc[0]} studies) ---")

                display_cols = ["model", "auroc"]
                if n_boot > 0 and "auroc_ci_lo" in group.columns:
                    display_cols += ["auroc_ci_lo", "auroc_ci_hi"]
                display_cols += ["balanced_accuracy", "accuracy", "kappa", "ece"]
                if n_cls == 2:
                    idx = display_cols.index("balanced_accuracy")
                    display_cols.insert(idx, "auprc")
                    if n_boot > 0 and "auprc_ci_lo" in group.columns:
                        display_cols.insert(idx + 1, "auprc_ci_lo")
                        display_cols.insert(idx + 2, "auprc_ci_hi")
                    display_cols.append("prevalence")
                display_cols = [c for c in display_cols if c in group.columns]
                print(group[display_cols].to_string(index=False, float_format="%.4f"))

    # Print regression results
    if reg_rows:
        print("\n" + "=" * 110)
        print("REGRESSION TASKS (study-level prediction averaging)")
        print("=" * 110)

        reg_df = pd.DataFrame(reg_rows)
        reg_df = reg_df.sort_values(["task", "model"])

        if args.compact:
            cols = ["task", "model", "n_studies", "r2", "pearson_r", "mae", "rmse", "bias", "loa_lower", "loa_upper"]
            if n_boot > 0:
                cols.insert(4, "r2_ci_lo")
                cols.insert(5, "r2_ci_hi")
            cols = [c for c in cols if c in reg_df.columns]
            print(reg_df[cols].to_string(index=False, float_format="%.4f"))
        else:
            for task_name, group in reg_df.groupby("task"):
                n_studies = group["n_studies"].iloc[0]
                label_mean = group["label_mean"].iloc[0]
                label_std = group["label_std"].iloc[0]
                print(f"\n--- {task_name} ({n_studies} studies, label mean={label_mean:.2f} std={label_std:.2f}) ---")

                display_cols = ["model", "r2"]
                if n_boot > 0 and "r2_ci_lo" in group.columns:
                    display_cols += ["r2_ci_lo", "r2_ci_hi"]
                display_cols += ["pearson_r"]
                if n_boot > 0 and "pearson_ci_lo" in group.columns:
                    display_cols += ["pearson_ci_lo", "pearson_ci_hi"]
                display_cols += ["mae"]
                if n_boot > 0 and "mae_ci_lo" in group.columns:
                    display_cols += ["mae_ci_lo", "mae_ci_hi"]
                display_cols += ["rmse", "bias", "loa_lower", "loa_upper"]
                display_cols = [c for c in display_cols if c in group.columns]
                print(group[display_cols].to_string(index=False, float_format="%.4f"))

    # Save to CSV if requested
    if args.csv:
        all_df = pd.DataFrame(rows)
        all_df.to_csv(args.csv, index=False)
        print(f"\nSaved {len(rows)} results to {args.csv}")

    # Summary
    print(f"\n{'=' * 110}")
    print(f"SUMMARY: {len(cls_rows)} classification + {len(reg_rows)} regression = {len(rows)} total")
    tasks = sorted(set(r["task"] for r in rows))
    models = sorted(set(r["model"] for r in rows))
    print(f"Tasks ({len(tasks)}): {', '.join(tasks)}")
    print(f"Models ({len(models)}): {', '.join(models)}")

    # Pairwise comparisons
    if args.compare:
        print(f"\n{'#' * 110}")
        print("PAIRWISE MODEL COMPARISONS")
        print(f"{'#' * 110}")
        run_pairwise_comparisons(results_dir, task_filter=args.task, n_boot=max(n_boot, 2000))

    # Confusion matrices
    if args.confusion:
        print(f"\n{'#' * 110}")
        print("CONFUSION MATRICES")
        print(f"{'#' * 110}")
        print_confusion_matrices(results_dir, task_filter=args.task, model_filter=args.model)

    # Plots
    if args.plot:
        generate_all_plots(
            results_dir, args.plot, rows,
            task_filter=args.task, model_filter=args.model,
        )


if __name__ == "__main__":
    main()
