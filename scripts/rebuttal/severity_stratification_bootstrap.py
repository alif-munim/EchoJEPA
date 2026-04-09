"""
Severity-stratified LVEF analysis with bootstrap CIs.

Stratifies EchoNet-Dynamic test predictions by EF severity bin,
computes per-bin R², Pearson, MAE, prediction bias, and 95% bootstrap CIs.

Input: *_noised_lvef_persample.csv files (uses 'clean' condition only).
"""

import argparse
import csv
import os

import numpy as np
from scipy import stats


BINS = [
    ("Reduced (<40%)", 0, 40),
    ("Mildly reduced (40-54%)", 40, 55),
    ("Normal (≥55%)", 55, 200),
]

MODELS = [
    ("JEPA e100", "jepa_in21k_e100_noised_lvef_persample.csv"),
    ("BYOL e100", "byol_e100_noised_lvef_persample.csv"),
    ("MAE e99", "mae_e99_noised_lvef_persample.csv"),
    ("SALT S2 e79", "salt_v1_e79_noised_lvef_persample.csv"),
]


def load_clean_predictions(csv_path):
    preds, labels = [], []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["condition"] == "clean":
                preds.append(float(row["prediction"]))
                labels.append(float(row["label"]))
    return np.array(preds), np.array(labels)


def compute_metrics(preds, labels):
    if len(preds) < 3:
        return {"r2": np.nan, "pearson": np.nan, "mae": np.nan, "pred_mean": np.nan, "bias": np.nan}
    ss_res = np.sum((labels - preds) ** 2)
    ss_tot = np.sum((labels - labels.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    pearson = np.corrcoef(preds, labels)[0, 1] if len(preds) > 2 else np.nan
    mae = np.mean(np.abs(preds - labels))
    pred_mean = np.mean(preds)
    bias = pred_mean - np.mean(labels)
    return {"r2": r2, "pearson": pearson, "mae": mae, "pred_mean": pred_mean, "bias": bias}


def bootstrap_ci(preds, labels, n_boot=10000, seed=42):
    rng = np.random.RandomState(seed)
    n = len(preds)
    boot_metrics = {"r2": [], "pearson": [], "mae": []}
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        m = compute_metrics(preds[idx], labels[idx])
        for k in boot_metrics:
            boot_metrics[k].append(m[k])
    ci = {}
    for k in boot_metrics:
        arr = np.array(boot_metrics[k])
        arr = arr[~np.isnan(arr)]
        if len(arr) > 0:
            ci[k] = (np.percentile(arr, 2.5), np.percentile(arr, 97.5))
        else:
            ci[k] = (np.nan, np.nan)
    return ci


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samples_dir", default="scripts/rebuttal/samples")
    p.add_argument("--n_boot", type=int, default=10000)
    p.add_argument("--output", default="scripts/rebuttal/samples/severity_stratification_e100.csv")
    args = p.parse_args()

    all_results = []

    for model_name, csv_file in MODELS:
        csv_path = os.path.join(args.samples_dir, csv_file)
        if not os.path.exists(csv_path):
            print(f"  SKIP {model_name}: {csv_path} not found")
            continue

        preds, labels = load_clean_predictions(csv_path)
        print(f"\n{'='*60}")
        print(f"{model_name}: {len(preds)} videos")

        # Overall
        m = compute_metrics(preds, labels)
        ci = bootstrap_ci(preds, labels, args.n_boot)
        print(f"  Overall: R²={m['r2']:.3f} [{ci['r2'][0]:.3f}, {ci['r2'][1]:.3f}], "
              f"Pearson={m['pearson']:.3f} [{ci['pearson'][0]:.3f}, {ci['pearson'][1]:.3f}], "
              f"MAE={m['mae']:.1f}")
        all_results.append({
            "model": model_name, "bin": "Overall", "n": len(preds),
            "true_mean": f"{labels.mean():.1f}",
            "r2": f"{m['r2']:.4f}", "r2_lo": f"{ci['r2'][0]:.4f}", "r2_hi": f"{ci['r2'][1]:.4f}",
            "pearson": f"{m['pearson']:.4f}", "pearson_lo": f"{ci['pearson'][0]:.4f}", "pearson_hi": f"{ci['pearson'][1]:.4f}",
            "mae": f"{m['mae']:.2f}", "mae_lo": f"{ci['mae'][0]:.2f}", "mae_hi": f"{ci['mae'][1]:.2f}",
            "pred_mean": f"{m['pred_mean']:.1f}", "bias": f"{m['bias']:.1f}",
        })

        # Per-bin
        for bin_name, lo, hi in BINS:
            mask = (labels >= lo) & (labels < hi)
            bp = preds[mask]
            bl = labels[mask]
            m = compute_metrics(bp, bl)
            ci = bootstrap_ci(bp, bl, args.n_boot)
            print(f"  {bin_name} (n={len(bp)}): R²={m['r2']:.3f} [{ci['r2'][0]:.3f}, {ci['r2'][1]:.3f}], "
                  f"Pearson={m['pearson']:.3f} [{ci['pearson'][0]:.3f}, {ci['pearson'][1]:.3f}], "
                  f"MAE={m['mae']:.1f} [{ci['mae'][0]:.1f}, {ci['mae'][1]:.1f}], "
                  f"pred_mean={m['pred_mean']:.1f}, bias={m['bias']:+.1f}")
            all_results.append({
                "model": model_name, "bin": bin_name, "n": len(bp),
                "true_mean": f"{bl.mean():.1f}",
                "r2": f"{m['r2']:.4f}", "r2_lo": f"{ci['r2'][0]:.4f}", "r2_hi": f"{ci['r2'][1]:.4f}",
                "pearson": f"{m['pearson']:.4f}", "pearson_lo": f"{ci['pearson'][0]:.4f}", "pearson_hi": f"{ci['pearson'][1]:.4f}",
                "mae": f"{m['mae']:.2f}", "mae_lo": f"{ci['mae'][0]:.2f}", "mae_hi": f"{ci['mae'][1]:.2f}",
                "pred_mean": f"{m['pred_mean']:.1f}", "bias": f"{m['bias']:.1f}",
            })

    # Pairwise comparisons on ALL bins + overall
    pairs = [("JEPA e100", "MAE e99"), ("JEPA e100", "BYOL e100"), ("JEPA e100", "SALT S2 e79"),
             ("BYOL e100", "MAE e99"), ("BYOL e100", "SALT S2 e79"), ("MAE e99", "SALT S2 e79")]
    all_bins = [("Overall", None, None)] + [(name, lo, hi) for name, lo, hi in BINS]

    # Load all predictions once
    model_data = {}
    for model_name, csv_file in MODELS:
        csv_path = os.path.join(args.samples_dir, csv_file)
        if not os.path.exists(csv_path):
            continue
        model_data[model_name] = load_clean_predictions(csv_path)

    for bin_name, blo, bhi in all_bins:
        print(f"\n{'='*60}")
        print(f"Pairwise bootstrap tests — {bin_name}:")
        rng = np.random.RandomState(42)
        for m1, m2 in pairs:
            if m1 not in model_data or m2 not in model_data:
                continue
            p1, l1 = model_data[m1]
            p2, l2 = model_data[m2]
            if blo is not None:
                mask = (l1 >= blo) & (l1 < bhi)
                p1, l1 = p1[mask], l1[mask]
                mask = (l2 >= blo) & (l2 < bhi)
                p2, l2 = p2[mask], l2[mask]
            n = len(l1)
            delta_mae = []
            for _ in range(args.n_boot):
                idx = rng.randint(0, n, size=n)
                mae1 = np.mean(np.abs(p1[idx] - l1[idx]))
                mae2 = np.mean(np.abs(p2[idx] - l2[idx]))
                delta_mae.append(mae2 - mae1)  # positive = m1 better
            delta_mae = np.array(delta_mae)
            lo, hi = np.percentile(delta_mae, [2.5, 97.5])
            mean_d = delta_mae.mean()
            sig = "*sig*" if lo > 0 or hi < 0 else "n.s."
            print(f"  {m2} − {m1}: ΔMAE={mean_d:+.2f} [{lo:+.2f}, {hi:+.2f}] {sig}")

    # Save CSV
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
