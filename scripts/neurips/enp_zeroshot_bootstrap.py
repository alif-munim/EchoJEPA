"""Bootstrap CIs for zero-shot pediatric LVEF predictions (4 models).

Reads per-clip prediction CSVs from the inference runs, un-normalizes using
the pediatric scaler, computes study-level averages, then bootstraps R², MAE,
and Pearson correlation with 95% CIs.

Usage:
    python scripts/neurips/enp_zeroshot_bootstrap.py \
        --results_dir /opt/dlami/nvme/evals/neurips/enp_zeroshot \
        --scaler_path data/scalers/pediatric_ef_scaler.pkl \
        --n_bootstrap 10000 --seed 42
"""
import argparse
import os
import pickle

import numpy as np
import pandas as pd
from scipy import stats


def load_scaler(path):
    with open(path, "rb") as f:
        scaler = pickle.load(f)
    return float(scaler.mean_[0]), float(scaler.scale_[0])


def bootstrap_metrics(labels, preds, n_bootstrap=10000, seed=42):
    rng = np.random.RandomState(seed)
    n = len(labels)

    r2s, maes, pearsons = [], [], []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        y, yhat = labels[idx], preds[idx]

        ss_res = np.sum((y - yhat) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        mae = np.mean(np.abs(y - yhat))
        r, _ = stats.pearsonr(y, yhat) if len(y) > 2 else (float("nan"), 1.0)

        r2s.append(r2)
        maes.append(mae)
        pearsons.append(r)

    return {
        "R2": (np.median(r2s), np.percentile(r2s, 2.5), np.percentile(r2s, 97.5)),
        "MAE": (np.median(maes), np.percentile(maes, 2.5), np.percentile(maes, 97.5)),
        "Pearson": (np.median(pearsons), np.percentile(pearsons, 2.5), np.percentile(pearsons, 97.5)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--scaler_path", default="data/scalers/pediatric_ef_scaler.pkl")
    parser.add_argument("--n_bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    mean, scale = load_scaler(args.scaler_path)
    print(f"Scaler: mean={mean:.2f}, std={scale:.2f}")

    models = {
        "JEPA IN21K e100": "jepa_e100_predictions.csv",
        "BYOL e100": "byol_e100_predictions.csv",
        "MAE e99": "mae_e99_predictions.csv",
        "SALT S2v1 e79": "salt_e79_predictions.csv",
    }

    all_results = []
    for model_name, csv_name in models.items():
        csv_path = os.path.join(args.results_dir, csv_name)
        if not os.path.exists(csv_path):
            print(f"\n[SKIP] {model_name}: {csv_path} not found")
            continue

        df = pd.read_csv(csv_path)
        print(f"\n{'='*60}")
        print(f"Model: {model_name} ({len(df)} clips)")

        # Un-normalize: real = z * scale + mean
        labels_real = df["label"].values * scale + mean
        preds_real = df["prediction"].values * scale + mean

        # Point estimates
        ss_res = np.sum((labels_real - preds_real) ** 2)
        ss_tot = np.sum((labels_real - np.mean(labels_real)) ** 2)
        r2_point = 1 - ss_res / ss_tot
        mae_point = np.mean(np.abs(labels_real - preds_real))
        pearson_point, _ = stats.pearsonr(labels_real, preds_real)

        print(f"  R²:      {r2_point:.4f}")
        print(f"  MAE:     {mae_point:.4f}")
        print(f"  Pearson: {pearson_point:.4f}")

        # Bootstrap
        metrics = bootstrap_metrics(labels_real, preds_real, args.n_bootstrap, args.seed)
        for metric_name, (med, lo, hi) in metrics.items():
            print(f"  {metric_name} 95% CI: [{lo:.4f}, {hi:.4f}] (median {med:.4f})")
            all_results.append({
                "model": model_name,
                "metric": metric_name,
                "point": {"R2": r2_point, "MAE": mae_point, "Pearson": pearson_point}[metric_name],
                "median": med,
                "ci_lo": lo,
                "ci_hi": hi,
                "n_clips": len(df),
            })

    if all_results:
        out_df = pd.DataFrame(all_results)
        out_path = os.path.join(args.results_dir, "enp_zeroshot_bootstrap.csv")
        out_df.to_csv(out_path, index=False)
        print(f"\nSaved bootstrap results to {out_path}")

        # Print summary table
        print(f"\n{'='*60}")
        print("SUMMARY: Zero-shot Pediatric LVEF (END probes → ENP test)")
        print(f"{'='*60}")
        print(f"{'Model':<20} {'R²':>12} {'MAE':>12} {'Pearson':>12}")
        print("-" * 60)
        for model_name in models:
            rows = [r for r in all_results if r["model"] == model_name]
            if not rows:
                continue
            r2 = next(r for r in rows if r["metric"] == "R2")
            mae = next(r for r in rows if r["metric"] == "MAE")
            pear = next(r for r in rows if r["metric"] == "Pearson")
            print(f"{model_name:<20} {r2['point']:.3f} [{r2['ci_lo']:.3f},{r2['ci_hi']:.3f}]"
                  f"  {mae['point']:.2f} [{mae['ci_lo']:.2f},{mae['ci_hi']:.2f}]"
                  f"  {pear['point']:.3f} [{pear['ci_lo']:.3f},{pear['ci_hi']:.3f}]")


if __name__ == "__main__":
    main()
