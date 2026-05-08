"""Per-stratum paired bootstrap CIs — V-JEPA†-e125 vs V4-e25 on EchoNet-Dynamic LVEF.

Strata (AHA 2022 / ASE, raw EF):
  Reduced (≤40, HFrEF) | Mildly reduced (41–49) | Normal (50–70) | Hyperdynamic (>70)

For each stratum:
  - V-JEPA† MAE, Pearson, R² with marginal 95% CI
  - V4 MAE, Pearson, R² with marginal 95% CI
  - Paired ΔMAE, ΔPearson, ΔR² (V4 − base) with 95% CI and P(V4 better)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, r2_score

RNG = np.random.default_rng(0)
B = 10000

BASE = "/tmp/lvef_strat/base_e125.csv"
V4   = "/tmp/lvef_strat/v4_e25.csv"

STRATA = [
    ("Reduced ≤40",         lambda ef: ef <= 40),
    ("Mild reduced 41–49",  lambda ef: (ef > 40) & (ef < 50)),
    ("Normal 50–70",        lambda ef: (ef >= 50) & (ef <= 70)),
    ("Hyperdynamic >70",    lambda ef: ef > 70),
]


def load(path):
    df = pd.read_csv(path).sort_values("video_path").reset_index(drop=True)
    return df


def metrics(y, yhat):
    mae = mean_absolute_error(y, yhat)
    if len(y) >= 3 and y.std() > 0 and yhat.std() > 0:
        r2 = r2_score(y, yhat)
        pr = pearsonr(y, yhat)[0]
    else:
        r2 = np.nan
        pr = np.nan
    return mae, pr, r2


def boot_ci(y, yhat, B=B, rng=RNG):
    n = len(y)
    maes, prs, r2s = [], [], []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        yy = y[idx]; yh = yhat[idx]
        m, p, r = metrics(yy, yh)
        maes.append(m); prs.append(p); r2s.append(r)
    def ci(v):
        v = np.array(v)
        v = v[~np.isnan(v)]
        if len(v) == 0:
            return (np.nan, np.nan)
        return (np.percentile(v, 2.5), np.percentile(v, 97.5))
    return ci(maes), ci(prs), ci(r2s)


def paired_delta(y, yhat_v4, yhat_base, B=B, rng=RNG):
    n = len(y)
    d_mae, d_pr, d_r2 = [], [], []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        yy = y[idx]; a = yhat_v4[idx]; b = yhat_base[idx]
        # MAE: lower = better → ΔMAE(base − V4) so positive = V4 better
        d_mae.append(mean_absolute_error(yy, b) - mean_absolute_error(yy, a))
        if yy.std() > 0 and a.std() > 0 and b.std() > 0:
            d_pr.append(pearsonr(yy, a)[0] - pearsonr(yy, b)[0])
            d_r2.append(r2_score(yy, a) - r2_score(yy, b))
        else:
            d_pr.append(np.nan); d_r2.append(np.nan)
    def summ(d):
        d = np.array(d); d = d[~np.isnan(d)]
        if len(d) == 0: return "—"
        p = (d > 0).mean()
        return f"{d.mean():+.3f} [{np.percentile(d,2.5):+.3f}, {np.percentile(d,97.5):+.3f}] P(V4>base)={p:.3f}"
    return summ(d_mae), summ(d_pr), summ(d_r2)


def main():
    b = load(BASE)
    v = load(V4)
    assert (b["video_path"].to_numpy() == v["video_path"].to_numpy()).all(), "misaligned"
    y = b["label_real"].to_numpy()
    yh_b = b["pred_real"].to_numpy()
    yh_v = v["pred_real"].to_numpy()

    print("=" * 100)
    print(f"EchoNet-Dynamic LVEF — per-stratum paired bootstrap (B={B})")
    print(f"Total N = {len(y)}")
    print("=" * 100)

    for name, pred in STRATA:
        mask = pred(y)
        n = mask.sum()
        y_s = y[mask]; yh_b_s = yh_b[mask]; yh_v_s = yh_v[mask]
        print(f"\n### {name} (N={n})")
        if n < 10:
            print(f"  Too few samples for bootstrap.")
            continue
        mae_b, pr_b, r2_b = metrics(y_s, yh_b_s)
        mae_v, pr_v, r2_v = metrics(y_s, yh_v_s)
        (mae_b_ci, pr_b_ci, r2_b_ci) = boot_ci(y_s, yh_b_s)
        (mae_v_ci, pr_v_ci, r2_v_ci) = boot_ci(y_s, yh_v_s)
        print(f"  V-JEPA†-e125:  MAE {mae_b:.3f} [{mae_b_ci[0]:.3f}, {mae_b_ci[1]:.3f}]   Pearson {pr_b:.3f} [{pr_b_ci[0]:.3f}, {pr_b_ci[1]:.3f}]   R² {r2_b:.3f} [{r2_b_ci[0]:.3f}, {r2_b_ci[1]:.3f}]")
        print(f"  V4 MV-PhaseRel: MAE {mae_v:.3f} [{mae_v_ci[0]:.3f}, {mae_v_ci[1]:.3f}]   Pearson {pr_v:.3f} [{pr_v_ci[0]:.3f}, {pr_v_ci[1]:.3f}]   R² {r2_v:.3f} [{r2_v_ci[0]:.3f}, {r2_v_ci[1]:.3f}]")
        d_mae, d_pr, d_r2 = paired_delta(y_s, yh_v_s, yh_b_s)
        print(f"  Paired Δ:")
        print(f"    ΔMAE (base−V4, pos=V4 better):    {d_mae}")
        print(f"    ΔPearson (V4−base, pos=V4 better): {d_pr}")
        print(f"    ΔR² (V4−base, pos=V4 better):      {d_r2}")

    # Full cohort
    print(f"\n### Full cohort (N={len(y)})")
    mae_b, pr_b, r2_b = metrics(y, yh_b)
    mae_v, pr_v, r2_v = metrics(y, yh_v)
    (mae_b_ci, pr_b_ci, r2_b_ci) = boot_ci(y, yh_b)
    (mae_v_ci, pr_v_ci, r2_v_ci) = boot_ci(y, yh_v)
    print(f"  V-JEPA†-e125:  MAE {mae_b:.3f} [{mae_b_ci[0]:.3f}, {mae_b_ci[1]:.3f}]   Pearson {pr_b:.3f} [{pr_b_ci[0]:.3f}, {pr_b_ci[1]:.3f}]   R² {r2_b:.3f} [{r2_b_ci[0]:.3f}, {r2_b_ci[1]:.3f}]")
    print(f"  V4 MV-PhaseRel: MAE {mae_v:.3f} [{mae_v_ci[0]:.3f}, {mae_v_ci[1]:.3f}]   Pearson {pr_v:.3f} [{pr_v_ci[0]:.3f}, {pr_v_ci[1]:.3f}]   R² {r2_v:.3f} [{r2_v_ci[0]:.3f}, {r2_v_ci[1]:.3f}]")
    d_mae, d_pr, d_r2 = paired_delta(y, yh_v, yh_b)
    print(f"  Paired Δ:")
    print(f"    ΔMAE (base−V4, pos=V4 better):    {d_mae}")
    print(f"    ΔPearson (V4−base, pos=V4 better): {d_pr}")
    print(f"    ΔR² (V4−base, pos=V4 better):      {d_r2}")


if __name__ == "__main__":
    main()
