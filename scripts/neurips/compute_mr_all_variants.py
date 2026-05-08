"""True-softmax AUROC + paired bootstrap for all 6 MR variants.

Uses patched-eval CSVs from jobs 862 (V4, MV-PairedIntra, TokenRel-e5) and
864 (V-JEPA†-e130, MCC-848, TokenRel-e25).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

RNG = np.random.default_rng(0)
B = 2000

FILES_ALL = {
    "V-JEPA†-e130":        "/tmp/rerun864/base_e130_mr.csv",
    "V4-e25":              "/tmp/rerun862/v4_e25_mr_a4c.csv",
    "MV-PairedIntra":      "/tmp/rerun862/paired_iv25_mr_a4c.csv",
    "TokenRel-Motion-e5":  "/tmp/rerun862/tokenrel_r2_e5_mr_a4c.csv",
    "MCC-Anchored-e25":    "/tmp/rerun864/mcc_e25_mr_848.csv",
    "TokenRel-Motion-e25": "/tmp/rerun864/tokenrel_r2_e25_mr_853.csv",
}


def main():
    dfs = {k: pd.read_csv(v).sort_values("video_path").reset_index(drop=True) for k, v in FILES_ALL.items()}
    ref_vp = dfs["V-JEPA†-e130"]["video_path"].to_numpy()
    for k, df in dfs.items():
        assert np.array_equal(df["video_path"].to_numpy(), ref_vp), f"{k} misaligned"
    y = dfs["V-JEPA†-e130"]["true_label"].to_numpy()
    Ps = {}
    for k, df in dfs.items():
        P = df[[f"prob_class_{c}" for c in range(4)]].to_numpy()
        s = P.sum(axis=1)
        if not np.allclose(s, 1.0, atol=1e-3):
            P = P - P.max(axis=1, keepdims=True)
            P = np.exp(P)
            P = P / P.sum(axis=1, keepdims=True)
        Ps[k] = P

    def boot_bin(y, s, rng=RNG, B=B):
        out = []
        for _ in range(B):
            idx = rng.integers(0, len(y), len(y))
            yy = y[idx]
            if len(np.unique(yy)) < 2:
                continue
            out.append(roc_auc_score(yy, s[idx]))
        return np.array(out)

    def boot_ovr(y, P, rng=RNG, B=B):
        out = []
        for _ in range(B):
            idx = rng.integers(0, len(y), len(y))
            yy = y[idx]
            if len(np.unique(yy)) < 4:
                continue
            try:
                out.append(roc_auc_score(yy, P[idx], multi_class="ovr", average="macro"))
            except Exception:
                pass
        return np.array(out)

    print("=" * 90)
    print("MR A4C test: true softmax AUROC + 95% CI per variant (N=4,482, B=2000)")
    print("=" * 90)
    for name in FILES_ALL.keys():
        P = Ps[name]
        ovr = roc_auc_score(y, P, multi_class="ovr", average="macro")
        v = boot_ovr(y, P)
        ovr_ci = f"{ovr:.4f} [{np.percentile(v,2.5):.4f}, {np.percentile(v,97.5):.4f}]"
        y_any = (y > 0).astype(int); s_any = 1 - P[:, 0]
        a_any = roc_auc_score(y_any, s_any); v_any = boot_bin(y_any, s_any)
        any_ci = f"{a_any:.4f} [{np.percentile(v_any,2.5):.4f}, {np.percentile(v_any,97.5):.4f}]"
        y_mod = (y >= 2).astype(int); s_mod = P[:, 2] + P[:, 3]
        a_mod = roc_auc_score(y_mod, s_mod); v_mod = boot_bin(y_mod, s_mod)
        mod_ci = f"{a_mod:.4f} [{np.percentile(v_mod,2.5):.4f}, {np.percentile(v_mod,97.5):.4f}]"
        y_sev = (y == 3).astype(int); s_sev = P[:, 3]
        a_sev = roc_auc_score(y_sev, s_sev); v_sev = boot_bin(y_sev, s_sev)
        sev_ci = f"{a_sev:.4f} [{np.percentile(v_sev,2.5):.4f}, {np.percentile(v_sev,97.5):.4f}]"
        print(f"  {name:22s}  4cls={ovr_ci}  any={any_ci}  mod={mod_ci}  sev={sev_ci}")

    print()
    print("=" * 90)
    print("Paired ΔAUROC vs V-JEPA†-e130 (B=2000, aligned per-video)")
    print("=" * 90)
    ref = "V-JEPA†-e130"
    Pr = Ps[ref]
    for name in ["V4-e25", "MV-PairedIntra", "TokenRel-Motion-e5", "MCC-Anchored-e25", "TokenRel-Motion-e25"]:
        Pv = Ps[name]
        d4, da, dm = [], [], []
        for _ in range(B):
            idx = RNG.integers(0, len(y), len(y))
            yy = y[idx]
            if len(np.unique(yy)) < 4:
                continue
            try:
                d4.append(roc_auc_score(yy, Pv[idx], multi_class="ovr", average="macro") - roc_auc_score(yy, Pr[idx], multi_class="ovr", average="macro"))
            except Exception:
                pass
            ya = (yy > 0).astype(int)
            if len(np.unique(ya)) == 2:
                da.append(roc_auc_score(ya, 1 - Pv[idx, 0]) - roc_auc_score(ya, 1 - Pr[idx, 0]))
            ym = (yy >= 2).astype(int)
            if len(np.unique(ym)) == 2:
                dm.append(roc_auc_score(ym, Pv[idx, 2] + Pv[idx, 3]) - roc_auc_score(ym, Pr[idx, 2] + Pr[idx, 3]))

        def summ(d):
            if not d:
                return "—"
            a = np.array(d); p = (a > 0).mean()
            return f"{a.mean():+.4f} [{np.percentile(a,2.5):+.4f}, {np.percentile(a,97.5):+.4f}]  P(>0)={p:.3f}"

        print(f"\n{name}:")
        print(f"  Δ 4-cls OVR: {summ(d4)}")
        print(f"  Δ any-MR:    {summ(da)}")
        print(f"  Δ ≥mod:      {summ(dm)}")


if __name__ == "__main__":
    main()
