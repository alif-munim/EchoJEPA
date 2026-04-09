"""
Bootstrap CIs for CAMUS noised segmentation from per-sample data.

Computes:
1. Absolute Dice with 95% CI for each condition
2. Paired degradation (clean - perturbed) with 95% CI for each perturbation
3. Per-structure breakdown for severe conditions

Usage:
    python scripts/rebuttal/camus_noised_bootstrap.py
"""

import csv
import sys
from pathlib import Path

import numpy as np

MODELS = {
    "JEPA": "jepa_in21k_e100",
    "BYOL": "byol_e100",
    "MAE": "mae_e99",
    "SALT": "salt_s2v1_e79",
}
STRUCTURE_NAMES = ["LV", "MYO", "LA"]
N_BOOTSTRAP = 10_000
CI_LEVEL = 0.95
SAMPLES_DIR = Path("scripts/rebuttal/samples")


def load_persample(label):
    path = SAMPLES_DIR / f"{label}_noised_seg_persample.csv"
    with open(path) as f:
        return list(csv.DictReader(f))


def get_condition_samples(rows, condition):
    """Get per-sample mean_dice for a condition, ordered by sample_idx."""
    cond_rows = [r for r in rows if r["condition"] == condition]
    cond_rows.sort(key=lambda r: int(r["sample_idx"]))
    return np.array([float(r["mean_dice"]) for r in cond_rows])


def get_structure_samples(rows, condition, structure):
    """Get per-sample structure dice for a condition."""
    cond_rows = [r for r in rows if r["condition"] == condition]
    cond_rows.sort(key=lambda r: int(r["sample_idx"]))
    return np.array([float(r[f"mean_{structure}_dice"]) for r in cond_rows])


def bootstrap_ci(values, n_boot=N_BOOTSTRAP, ci=CI_LEVEL):
    """Bootstrap CI for the mean."""
    rng = np.random.RandomState(42)
    n = len(values)
    means = np.array([np.mean(rng.choice(values, size=n, replace=True)) for _ in range(n_boot)])
    alpha = (1 - ci) / 2
    lo, hi = np.percentile(means, [alpha * 100, (1 - alpha) * 100])
    return np.mean(values), lo, hi


def paired_bootstrap_ci(clean, perturbed, n_boot=N_BOOTSTRAP, ci=CI_LEVEL):
    """Paired bootstrap CI for mean degradation (clean - perturbed)."""
    assert len(clean) == len(perturbed)
    deltas = clean - perturbed
    rng = np.random.RandomState(42)
    n = len(deltas)
    means = np.array([np.mean(rng.choice(deltas, size=n, replace=True)) for _ in range(n_boot)])
    alpha = (1 - ci) / 2
    lo, hi = np.percentile(means, [alpha * 100, (1 - alpha) * 100])
    return np.mean(deltas), lo, hi


def fmt_ci(mean, lo, hi, as_pct=False):
    if as_pct:
        return f"{mean*100:.1f}% [{lo*100:.1f}, {hi*100:.1f}]"
    return f"{mean:.3f} [{lo:.3f}, {hi:.3f}]"


def main():
    perturbation_types = ["depth_attenuation", "gaussian_shadow", "haze_artifact"]
    severity_levels = ["mild", "moderate", "severe"]

    # =========================================================
    # 1. Absolute Dice with CIs
    # =========================================================
    print("=" * 100)
    print("1. ABSOLUTE DICE WITH 95% BOOTSTRAP CIs")
    print("=" * 100)

    header = f"{'Condition':<30}"
    for name in MODELS:
        header += f" {name:<22}"
    print(header)
    print("-" * 100)

    all_data = {}
    for name, label in MODELS.items():
        rows = load_persample(label)
        all_data[name] = rows

    conditions = ["clean"]
    for pt in perturbation_types:
        for sev in severity_levels:
            conditions.append(f"{pt}/{sev}")

    for cond in conditions:
        line = f"{cond:<30}"
        for name in MODELS:
            vals = get_condition_samples(all_data[name], cond)
            mean, lo, hi = bootstrap_ci(vals)
            line += f" {fmt_ci(mean, lo, hi):<22}"
        print(line)

    # =========================================================
    # 2. Paired degradation for severe conditions
    # =========================================================
    print()
    print("=" * 100)
    print("2. PAIRED DEGRADATION: clean → severe (% Dice drop with 95% CIs)")
    print("=" * 100)

    header = f"{'Perturbation':<25}"
    for name in MODELS:
        header += f" {name:<22}"
    print(header)
    print("-" * 100)

    for pt in perturbation_types:
        line = f"{pt:<25}"
        for name in MODELS:
            clean = get_condition_samples(all_data[name], "clean")
            severe = get_condition_samples(all_data[name], f"{pt}/severe")
            mean, lo, hi = paired_bootstrap_ci(clean, severe)
            clean_mean = np.mean(clean)
            pct_mean = mean / clean_mean
            pct_lo = lo / clean_mean
            pct_hi = hi / clean_mean
            line += f" {fmt_ci(pct_mean, pct_lo, pct_hi, as_pct=True):<22}"
        print(line)

    # Average severe drop
    line = f"{'avg severe':<25}"
    for name in MODELS:
        clean = get_condition_samples(all_data[name], "clean")
        drops = []
        for pt in perturbation_types:
            severe = get_condition_samples(all_data[name], f"{pt}/severe")
            drops.append(clean - severe)
        avg_delta = np.mean(drops, axis=0)  # per-sample average across 3 perturbations
        mean, lo, hi = bootstrap_ci(avg_delta)
        clean_mean = np.mean(clean)
        line += f" {fmt_ci(mean/clean_mean, lo/clean_mean, hi/clean_mean, as_pct=True):<22}"
    print(line)

    # =========================================================
    # 3. Paired degradation across all severity levels
    # =========================================================
    print()
    print("=" * 100)
    print("3. DEGRADATION BY SEVERITY (% Dice drop from clean)")
    print("=" * 100)

    for pt in perturbation_types:
        print(f"\n  {pt}:")
        header = f"  {'Severity':<15}"
        for name in MODELS:
            header += f" {name:<22}"
        print(header)
        print("  " + "-" * 95)
        for sev in severity_levels:
            line = f"  {sev:<15}"
            for name in MODELS:
                clean = get_condition_samples(all_data[name], "clean")
                pert = get_condition_samples(all_data[name], f"{pt}/{sev}")
                mean, lo, hi = paired_bootstrap_ci(clean, pert)
                clean_mean = np.mean(clean)
                line += f" {fmt_ci(mean/clean_mean, lo/clean_mean, hi/clean_mean, as_pct=True):<22}"
            print(line)

    # =========================================================
    # 4. Per-structure breakdown (severe only)
    # =========================================================
    print()
    print("=" * 100)
    print("4. PER-STRUCTURE BREAKDOWN (severe perturbation, absolute Dice)")
    print("=" * 100)

    for pt in perturbation_types:
        print(f"\n  {pt}/severe:")
        header = f"  {'Structure':<10}"
        for name in MODELS:
            header += f" {name:<22}"
        print(header)
        print("  " + "-" * 95)
        for struct in STRUCTURE_NAMES:
            line = f"  {struct:<10}"
            for name in MODELS:
                vals = get_structure_samples(all_data[name], f"{pt}/severe", struct)
                mean, lo, hi = bootstrap_ci(vals)
                line += f" {fmt_ci(mean, lo, hi):<22}"
            print(line)

    print("\n\nAll CIs: 95% percentile bootstrap, n=100 samples, 10K resamples, seed=42.")


if __name__ == "__main__":
    main()
