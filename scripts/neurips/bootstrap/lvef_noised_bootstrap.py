"""
Bootstrap CIs for LVEF noised inference from per-sample data.

R² is a set-level metric, so bootstrap resamples (prediction, label) pairs
and recomputes R² for each resample.

Usage:
    python scripts/neurips/lvef_noised_bootstrap.py
"""

import csv
from pathlib import Path

import numpy as np

MODELS = {
    "JEPA": "jepa_in21k_e100",
    "BYOL": "byol_e100",
    "MAE": "mae_e99",
    "SALT": "salt_v1_e79",
}
N_BOOTSTRAP = 10_000
CI_LEVEL = 0.95
SAMPLES_DIR = Path("scripts/neurips/samples")


def load_persample(label):
    path = SAMPLES_DIR / f"{label}_noised_lvef_persample.csv"
    with open(path) as f:
        return list(csv.DictReader(f))


def get_condition_data(rows, condition):
    """Get paired (predictions, labels) for a condition, ordered by sample_idx."""
    cond_rows = [r for r in rows if r["condition"] == condition]
    cond_rows.sort(key=lambda r: int(r["sample_idx"]))
    preds = np.array([float(r["prediction"]) for r in cond_rows])
    labels = np.array([float(r["label"]) for r in cond_rows])
    return preds, labels


def r2_score(preds, labels):
    ss_res = np.sum((labels - preds) ** 2)
    ss_tot = np.sum((labels - labels.mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0


def mae_score(preds, labels):
    return np.mean(np.abs(labels - preds))


def pearson(preds, labels):
    if len(preds) < 3:
        return 0.0
    return np.corrcoef(preds, labels)[0, 1]


def bootstrap_metric(preds, labels, metric_fn, n_boot=N_BOOTSTRAP, ci=CI_LEVEL):
    """Bootstrap CI for a set-level metric."""
    rng = np.random.RandomState(42)
    n = len(preds)
    boot_vals = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        boot_vals.append(metric_fn(preds[idx], labels[idx]))
    boot_vals = np.array(boot_vals)
    alpha = (1 - ci) / 2
    lo, hi = np.percentile(boot_vals, [alpha * 100, (1 - alpha) * 100])
    return metric_fn(preds, labels), lo, hi


def paired_bootstrap_delta(clean_preds, clean_labels, pert_preds, pert_labels,
                           metric_fn, n_boot=N_BOOTSTRAP, ci=CI_LEVEL):
    """Bootstrap CI for metric degradation (clean_metric - perturbed_metric).

    Paired: same resampled indices for both conditions.
    """
    assert len(clean_preds) == len(pert_preds)
    rng = np.random.RandomState(42)
    n = len(clean_preds)
    deltas = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        clean_val = metric_fn(clean_preds[idx], clean_labels[idx])
        pert_val = metric_fn(pert_preds[idx], pert_labels[idx])
        deltas.append(clean_val - pert_val)
    deltas = np.array(deltas)
    alpha = (1 - ci) / 2
    lo, hi = np.percentile(deltas, [alpha * 100, (1 - alpha) * 100])
    actual = metric_fn(clean_preds, clean_labels) - metric_fn(pert_preds, pert_labels)
    return actual, lo, hi


def fmt(mean, lo, hi):
    return f"{mean:.3f} [{lo:.3f}, {hi:.3f}]"


def fmt_pct(mean, lo, hi):
    return f"{mean*100:.1f}% [{lo*100:.1f}, {hi*100:.1f}]"


def main():
    perturbation_types = ["depth_attenuation", "gaussian_shadow", "haze_artifact"]
    severity_levels = ["mild", "moderate", "severe"]

    all_data = {}
    for name, label in MODELS.items():
        all_data[name] = load_persample(label)

    # =========================================================
    # 1. Absolute R² with CIs
    # =========================================================
    print("=" * 105)
    print("1. R² WITH 95% BOOTSTRAP CIs")
    print("=" * 105)

    conditions = ["clean"]
    for pt in perturbation_types:
        for sev in severity_levels:
            conditions.append(f"{pt}/{sev}")

    header = f"{'Condition':<30}"
    for name in MODELS:
        header += f" {name:<22}"
    print(header)
    print("-" * 105)

    for cond in conditions:
        line = f"{cond:<30}"
        for name in MODELS:
            preds, labels = get_condition_data(all_data[name], cond)
            mean, lo, hi = bootstrap_metric(preds, labels, r2_score)
            line += f" {fmt(mean, lo, hi):<22}"
        print(line)

    # =========================================================
    # 2. Paired degradation — severe
    # =========================================================
    print()
    print("=" * 105)
    print("2. PAIRED R² DEGRADATION: clean → severe (absolute drop with 95% CIs)")
    print("=" * 105)

    header = f"{'Perturbation':<25}"
    for name in MODELS:
        header += f" {name:<22}"
    print(header)
    print("-" * 105)

    for pt in perturbation_types:
        line = f"{pt:<25}"
        for name in MODELS:
            c_preds, c_labels = get_condition_data(all_data[name], "clean")
            p_preds, p_labels = get_condition_data(all_data[name], f"{pt}/severe")
            mean, lo, hi = paired_bootstrap_delta(c_preds, c_labels, p_preds, p_labels, r2_score)
            line += f" {fmt(mean, lo, hi):<22}"
        print(line)

    # =========================================================
    # 3. All severities
    # =========================================================
    print()
    print("=" * 105)
    print("3. R² DEGRADATION BY SEVERITY")
    print("=" * 105)

    for pt in perturbation_types:
        print(f"\n  {pt}:")
        header = f"  {'Severity':<15}"
        for name in MODELS:
            header += f" {name:<22}"
        print(header)
        print("  " + "-" * 100)
        for sev in severity_levels:
            line = f"  {sev:<15}"
            for name in MODELS:
                c_preds, c_labels = get_condition_data(all_data[name], "clean")
                p_preds, p_labels = get_condition_data(all_data[name], f"{pt}/{sev}")
                mean, lo, hi = paired_bootstrap_delta(c_preds, c_labels, p_preds, p_labels, r2_score)
                line += f" {fmt(mean, lo, hi):<22}"
            print(line)

    # =========================================================
    # 4. MAE and Pearson for severe (supplementary)
    # =========================================================
    print()
    print("=" * 105)
    print("4. ADDITIONAL METRICS — SEVERE (MAE and Pearson with CIs)")
    print("=" * 105)

    for metric_name, metric_fn in [("MAE", mae_score), ("Pearson", pearson)]:
        print(f"\n  {metric_name}:")
        header = f"  {'Condition':<25}"
        for name in MODELS:
            header += f" {name:<22}"
        print(header)
        print("  " + "-" * 100)
        for cond in ["clean"] + [f"{pt}/severe" for pt in perturbation_types]:
            line = f"  {cond:<25}"
            for name in MODELS:
                preds, labels = get_condition_data(all_data[name], cond)
                mean, lo, hi = bootstrap_metric(preds, labels, metric_fn)
                line += f" {fmt(mean, lo, hi):<22}"
            print(line)

    print(f"\n\nAll CIs: 95% percentile bootstrap, n=1277 samples, {N_BOOTSTRAP} resamples, seed=42.")


if __name__ == "__main__":
    main()
