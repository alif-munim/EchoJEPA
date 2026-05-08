"""Compute true multi-class OVR AUROC + paired bootstrap CIs for 862 reruns.

For MR (4-class):
  - 4-class macro OVR AUROC from softmax probabilities
  - Binary 'any-MR' (None=0 vs rest) from prob_class_0
  - Binary '>=moderate' (class>=2) from prob_class_2+prob_class_3
  - Binary 'severe only' (class=3) from prob_class_3

For HCM (2-class): directly from prob_class_1 (positive class).

Also computes paired-bootstrap CIs for variant-vs-reference deltas,
where variants are aligned row-by-row via video_path.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, cohen_kappa_score

RNG = np.random.default_rng(0)

MR_FILES = {
    "V4-e25":           "/tmp/rerun862/v4_e25_mr_a4c.csv",
    "MV-PairedIntra":   "/tmp/rerun862/paired_iv25_mr_a4c.csv",
    "TokenRel-Motion-e5": "/tmp/rerun862/tokenrel_r2_e5_mr_a4c.csv",
}
HCM_FILES = {
    "V-JEPA†-e125":       "/tmp/rerun862/base_e125_hcm_a4c.csv",
    "TokenRel-Motion-e5": "/tmp/rerun862/tokenrel_r2_e5_hcm_a4c.csv",
}


def load_mr(path):
    df = pd.read_csv(path)
    y = df["true_label"].to_numpy()
    P = df[[f"prob_class_{c}" for c in range(4)]].to_numpy()
    # softmax-normalize if the rows don't already sum to 1 (they may be logits saved before softmax)
    row_sums = P.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-3):
        # apply softmax
        P = P - P.max(axis=1, keepdims=True)
        P = np.exp(P)
        P = P / P.sum(axis=1, keepdims=True)
    return df["video_path"].to_numpy(), y, P


def load_hcm(path):
    df = pd.read_csv(path)
    y = df["true_label"].to_numpy()
    P = df[[f"prob_class_{c}" for c in range(2)]].to_numpy()
    row_sums = P.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-3):
        P = P - P.max(axis=1, keepdims=True)
        P = np.exp(P)
        P = P / P.sum(axis=1, keepdims=True)
    return df["video_path"].to_numpy(), y, P


def boot_ci_auroc(y, scores, is_binary, B=2000, rng=RNG):
    n = len(y)
    vals = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        if is_binary:
            if len(np.unique(yy)) < 2:
                continue
            vals.append(roc_auc_score(yy, scores[idx]))
        else:
            ss = scores[idx]
            uq = np.unique(yy)
            if len(uq) < len(np.unique(y)):
                continue
            try:
                vals.append(roc_auc_score(yy, ss, multi_class="ovr", average="macro"))
            except Exception:
                continue
    vals = np.array(vals)
    return np.percentile(vals, 2.5), np.percentile(vals, 97.5), vals


def main():
    print("=" * 70)
    print("MR A4C test — 4-class macro OVR AUROC (true softmax-based)")
    print("=" * 70)
    mr_rows = []
    mr_cache = {}
    for name, path in MR_FILES.items():
        vp, y, P = load_mr(path)
        mr_cache[name] = (vp, y, P)
        # 4-class macro OVR
        ovr4 = roc_auc_score(y, P, multi_class="ovr", average="macro")
        lo, hi, _ = boot_ci_auroc(y, P, is_binary=False)
        # Binary: any-MR = 1 - prob_class_0  (None is class 0)
        y_any = (y > 0).astype(int)
        s_any = 1 - P[:, 0]
        any_auroc = roc_auc_score(y_any, s_any)
        lo_any, hi_any, _ = boot_ci_auroc(y_any, s_any, is_binary=True)
        # Binary: >=moderate (class >= 2)
        y_mod = (y >= 2).astype(int)
        s_mod = P[:, 2] + P[:, 3]
        mod_auroc = roc_auc_score(y_mod, s_mod)
        lo_mod, hi_mod, _ = boot_ci_auroc(y_mod, s_mod, is_binary=True)
        # Binary: severe (class == 3)
        y_sev = (y == 3).astype(int)
        s_sev = P[:, 3]
        if y_sev.sum() > 0:
            sev_auroc = roc_auc_score(y_sev, s_sev)
            lo_sev, hi_sev, _ = boot_ci_auroc(y_sev, s_sev, is_binary=True)
        else:
            sev_auroc = np.nan; lo_sev = hi_sev = np.nan
        # top-1 kappa, bal_acc
        pred = P.argmax(axis=1)
        bacc = balanced_accuracy_score(y, pred)
        kappa = cohen_kappa_score(y, pred)
        mr_rows.append({
            "variant": name, "N": len(y),
            "4cls_OVR_macro": ovr4, "4cls_lo": lo, "4cls_hi": hi,
            "any_MR_AUROC": any_auroc, "any_lo": lo_any, "any_hi": hi_any,
            "ge_mod_AUROC": mod_auroc, "mod_lo": lo_mod, "mod_hi": hi_mod,
            "severe_AUROC": sev_auroc, "sev_lo": lo_sev, "sev_hi": hi_sev,
            "bal_acc": bacc, "kappa": kappa,
        })

    df = pd.DataFrame(mr_rows)
    # format
    for c in df.columns:
        if df[c].dtype == float:
            df[c] = df[c].map(lambda x: f"{x:.4f}" if pd.notna(x) else "nan")
    print(df.to_string(index=False))
    print()

    # Paired bootstrap: variant vs V4-e25 (reference) on 4-class OVR + any-MR + >=mod
    print("=" * 70)
    print("Paired bootstrap ΔAUROC vs V4-e25 reference, B=10000")
    print("=" * 70)
    # Align on video_path
    ref_name = "V4-e25"
    vp_ref, y_ref, P_ref = mr_cache[ref_name]
    for name in MR_FILES:
        if name == ref_name:
            continue
        vp_v, y_v, P_v = mr_cache[name]
        # Sort both by video_path
        order_ref = np.argsort(vp_ref)
        order_v = np.argsort(vp_v)
        assert np.array_equal(vp_ref[order_ref], vp_v[order_v]), f"misaligned {name}"
        y = y_ref[order_ref]
        Pr = P_ref[order_ref]
        Pv = P_v[order_v]
        n = len(y)
        B = 2000
        deltas_4cls, deltas_any, deltas_mod = [], [], []
        for _ in range(B):
            idx = RNG.integers(0, n, n)
            yy = y[idx]
            if len(np.unique(yy)) < 4:
                continue
            try:
                a = roc_auc_score(yy, Pv[idx], multi_class="ovr", average="macro")
                b = roc_auc_score(yy, Pr[idx], multi_class="ovr", average="macro")
                deltas_4cls.append(a - b)
            except Exception:
                pass
            y_any = (yy > 0).astype(int)
            if len(np.unique(y_any)) == 2:
                a = roc_auc_score(y_any, 1 - Pv[idx, 0])
                b = roc_auc_score(y_any, 1 - Pr[idx, 0])
                deltas_any.append(a - b)
            y_mod = (yy >= 2).astype(int)
            if len(np.unique(y_mod)) == 2:
                a = roc_auc_score(y_mod, Pv[idx, 2] + Pv[idx, 3])
                b = roc_auc_score(y_mod, Pr[idx, 2] + Pr[idx, 3])
                deltas_mod.append(a - b)
        def summarize(d):
            if not d:
                return "—"
            a = np.array(d)
            p = (a > 0).mean()
            return f"{a.mean():+.4f} [{np.percentile(a,2.5):+.4f}, {np.percentile(a,97.5):+.4f}]  P(>0)={p:.3f}"
        print(f"{name} − {ref_name}:")
        print(f"  Δ 4-cls OVR:   {summarize(deltas_4cls)}")
        print(f"  Δ any-MR:      {summarize(deltas_any)}")
        print(f"  Δ ≥moderate:   {summarize(deltas_mod)}")
        print()

    print("=" * 70)
    print("HCM A4C test (binary) — V-JEPA†-e125 (from rerun 862)")
    print("=" * 70)
    for name, path in HCM_FILES.items():
        vp, y, P = load_hcm(path)
        auroc = roc_auc_score(y, P[:, 1])
        lo, hi, _ = boot_ci_auroc(y, P[:, 1], is_binary=True)
        pred = P.argmax(axis=1)
        bacc = balanced_accuracy_score(y, pred)
        kappa = cohen_kappa_score(y, pred)
        print(f"{name}: N={len(y)} pos={int(y.sum())} ({y.mean()*100:.2f}%)  AUROC={auroc:.4f} [{lo:.4f}, {hi:.4f}]  bal_acc={bacc:.4f}  kappa={kappa:.4f}")


if __name__ == "__main__":
    main()
