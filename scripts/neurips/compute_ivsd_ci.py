"""IVSD test metrics + paired bootstrap CIs.

Columns: video_path, label_real, pred_real, abs_error
Metrics: MAE, R^2, Pearson
Paired deltas: V4 vs base_e130, TokenRel vs base_e130, TokenRel vs V4
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, r2_score

RNG = np.random.default_rng(0)
B = 10000

FILES = {
    "base_e130":         "/tmp/ivsd_test/base_e130.csv",
    "V4-e25":            "/tmp/ivsd_test/v4_e25.csv",
    "TokenRel-Motion-e25": "/tmp/ivsd_test/tokenrel_r2_e25.csv",
}


def load(path):
    df = pd.read_csv(path)
    return df.sort_values("video_path").reset_index(drop=True)


def metrics(y, yhat):
    return (
        mean_absolute_error(y, yhat),
        r2_score(y, yhat),
        pearsonr(y, yhat)[0],
    )


def bootstrap_ci(y, yhat, B=B, rng=RNG):
    n = len(y)
    maes, r2s, prs = [], [], []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        yh = yhat[idx]
        maes.append(mean_absolute_error(yy, yh))
        r2s.append(r2_score(yy, yh))
        if yy.std() > 0 and yh.std() > 0:
            prs.append(pearsonr(yy, yh)[0])
    def ci(v):
        v = np.array(v)
        return v.mean(), np.percentile(v, 2.5), np.percentile(v, 97.5)
    return ci(maes), ci(r2s), ci(prs)


def paired_delta(y, yhat_a, yhat_b, B=B, rng=RNG):
    """Compute paired bootstrap on a vs b (a − b). Positive metric → a better."""
    n = len(y)
    d_mae, d_r2, d_pr = [], [], []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        a = yhat_a[idx]; b = yhat_b[idx]
        # MAE is lower=better, so reverse sign: a<b → a better → report (b−a)_MAE
        d_mae.append(mean_absolute_error(yy, b) - mean_absolute_error(yy, a))
        d_r2.append(r2_score(yy, a) - r2_score(yy, b))
        if yy.std() > 0 and a.std() > 0 and b.std() > 0:
            d_pr.append(pearsonr(yy, a)[0] - pearsonr(yy, b)[0])
    def summ(d):
        d = np.array(d)
        p = (d > 0).mean()
        return f"{d.mean():+.4f} [{np.percentile(d,2.5):+.4f}, {np.percentile(d,97.5):+.4f}]  P(a>b)={p:.3f}"
    return {
        "ΔMAE (b−a, pos=a better)": summ(d_mae),
        "ΔR² (a−b, pos=a better)": summ(d_r2),
        "ΔPearson (a−b, pos=a better)": summ(d_pr),
    }


def main():
    dfs = {k: load(v) for k, v in FILES.items()}
    # Check alignment
    keys = list(dfs.keys())
    ref_vp = dfs[keys[0]]["video_path"].to_numpy()
    for k in keys[1:]:
        assert np.array_equal(dfs[k]["video_path"].to_numpy(), ref_vp), f"{k} misaligned"
    print(f"N = {len(ref_vp)} test clips, aligned")
    y = dfs[keys[0]]["label_real"].to_numpy()
    print()
    print("=" * 72)
    print(f"IVSD test metrics (N={len(y)}, bootstrap B={B})")
    print("=" * 72)
    print(f"{'model':<22} {'MAE':<22} {'R²':<22} {'Pearson':<22}")
    preds = {}
    for name, df in dfs.items():
        yhat = df["pred_real"].to_numpy()
        preds[name] = yhat
        (m_mae, mae_lo, mae_hi), (m_r2, r2_lo, r2_hi), (m_pr, pr_lo, pr_hi) = bootstrap_ci(y, yhat)
        print(f"{name:<22} {m_mae:.4f} [{mae_lo:.4f},{mae_hi:.4f}]  {m_r2:.4f} [{r2_lo:.4f},{r2_hi:.4f}]  {m_pr:.4f} [{pr_lo:.4f},{pr_hi:.4f}]")

    print()
    print("=" * 72)
    print(f"Paired ΔR² (positive = a wins), B={B}")
    print("=" * 72)
    pairs = [
        ("V4-e25", "base_e130"),
        ("TokenRel-Motion-e25", "base_e130"),
        ("TokenRel-Motion-e25", "V4-e25"),
    ]
    for a, b in pairs:
        print(f"\n{a}  vs  {b}:")
        for k, v in paired_delta(y, preds[a], preds[b]).items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
