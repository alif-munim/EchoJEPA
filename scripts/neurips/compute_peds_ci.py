"""EchoNet-Pediatric LVEF test metrics + paired bootstrap CIs.

4 variants available: base_e125 (722), V4-e25 (723), MCC-e25 (798), FJ-30k (803).
TokenRel-e5 (726) stalled at ep 6 — no test inference.
TokenRel-e25 (852) currently running on node 56.

Columns: video_path, label_real, pred_real, abs_error (labels on raw EF scale)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, r2_score

RNG = np.random.default_rng(0)
B = 10000

FILES = {
    "V-JEPA†-e125":      "/tmp/peds/base_e125.csv",
    "V4-e25":            "/tmp/peds/v4_e25.csv",
    "MCC-Anchored-e25":  "/tmp/peds/mcc_e25.csv",
    "FullJoint-30k":     "/tmp/peds/fj_30k_803.csv",
}


def load(path):
    df = pd.read_csv(path)
    return df.sort_values("video_path").reset_index(drop=True)


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
    n = len(y)
    d_mae, d_r2, d_pr = [], [], []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        yy = y[idx]; a = yhat_a[idx]; b = yhat_b[idx]
        d_mae.append(mean_absolute_error(yy, b) - mean_absolute_error(yy, a))  # pos = a better
        d_r2.append(r2_score(yy, a) - r2_score(yy, b))                          # pos = a better
        if yy.std() > 0 and a.std() > 0 and b.std() > 0:
            d_pr.append(pearsonr(yy, a)[0] - pearsonr(yy, b)[0])
    def summ(d):
        d = np.array(d); p = (d > 0).mean()
        return f"{d.mean():+.4f} [{np.percentile(d,2.5):+.4f}, {np.percentile(d,97.5):+.4f}]  P(a>b)={p:.3f}"
    return {
        "ΔMAE (b−a, pos=a better)": summ(d_mae),
        "ΔR² (a−b, pos=a better)": summ(d_r2),
        "ΔPearson (a−b, pos=a better)": summ(d_pr),
    }


def main():
    dfs = {k: load(v) for k, v in FILES.items()}
    keys = list(dfs.keys())
    ref_vp = dfs[keys[0]]["video_path"].to_numpy()
    for k in keys[1:]:
        assert np.array_equal(dfs[k]["video_path"].to_numpy(), ref_vp), f"{k} misaligned"
    print(f"N = {len(ref_vp)} clips, all 4 variants aligned on video_path")
    y = dfs[keys[0]]["label_real"].to_numpy()
    preds = {k: dfs[k]["pred_real"].to_numpy() for k in keys}

    print()
    print("=" * 78)
    print(f"EchoNet-Pediatric LVEF test metrics (N={len(y)}, bootstrap B={B})")
    print("=" * 78)
    for name in keys:
        (m_mae, mae_lo, mae_hi), (m_r2, r2_lo, r2_hi), (m_pr, pr_lo, pr_hi) = bootstrap_ci(y, preds[name])
        print(f"{name:<22}  MAE {m_mae:.3f} [{mae_lo:.3f},{mae_hi:.3f}]   R² {m_r2:.4f} [{r2_lo:.4f},{r2_hi:.4f}]   r {m_pr:.4f} [{pr_lo:.4f},{pr_hi:.4f}]")

    print()
    print("=" * 78)
    print(f"Paired deltas (B={B})")
    print("=" * 78)
    pairs = [
        ("V4-e25", "V-JEPA†-e125"),
        ("MCC-Anchored-e25", "V-JEPA†-e125"),
        ("FullJoint-30k", "V-JEPA†-e125"),
        ("MCC-Anchored-e25", "V4-e25"),
        ("FullJoint-30k", "V4-e25"),
        ("MCC-Anchored-e25", "FullJoint-30k"),
    ]
    for a, b in pairs:
        print(f"\n{a}  vs  {b}:")
        for k, v in paired_delta(y, preds[a], preds[b]).items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
