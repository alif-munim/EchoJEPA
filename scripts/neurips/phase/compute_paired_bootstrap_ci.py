#!/usr/bin/env python3
"""Paired bootstrap CIs for downstream probe Δ comparisons.

For two prediction CSVs representing the same held-out set of clips
evaluated by two different encoders, computes the paired Δ on common
metrics (MAE, R², Pearson for regression; acc, macro-acc, Moderate+
binary acc/sens/spec for classification) and a 95% bootstrap CI via
10,000 paired resamples.

Usage:
    python compute_paired_bootstrap_ci.py \
        --pred-a /path/to/pilot_rvsp_test.csv \
        --pred-b /path/to/ctrl_rvsp_test.csv \
        --task regression \
        --a-label "pilot 655 e5 encoder_pool" \
        --b-label "ctrl 658 e5 encoder_pool" \
        --out /tmp/rvsp_delta.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


def _load_reg(path: Path) -> list[tuple[str, float, float]]:
    rows = []
    with path.open() as fh:
        r = csv.DictReader(fh)
        for row in r:
            try:
                key = row.get("video_path") or row.get("path") or row.get("s3_uri")
                y = float(row["label_real"])
                p = float(row["pred_real"])
                rows.append((key, y, p))
            except (ValueError, KeyError):
                continue
    return rows


def _load_cls(path: Path) -> list[tuple[str, int, int, float]]:
    rows = []
    with path.open() as fh:
        r = csv.DictReader(fh)
        for row in r:
            try:
                key = row.get("video_path") or row.get("path") or row.get("s3_uri")
                y = int(float(row["true_label"]))
                p = int(float(row["predicted_class"]))
                conf = float(row.get("prediction_confidence", 0.0))
                rows.append((key, y, p, conf))
            except (ValueError, KeyError):
                continue
    return rows


def _pair(a, b):
    """Inner-join two rowsets on video_path (first field)."""
    bd = {r[0]: r for r in b}
    paired = []
    for r in a:
        if r[0] in bd:
            paired.append((r, bd[r[0]]))
    return paired


def _reg_metrics(y, p):
    y = np.asarray(y); p = np.asarray(p)
    mae = float(np.mean(np.abs(y - p)))
    sse = float(np.sum((y - p) ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - sse / max(sst, 1e-9)
    # Pearson
    yz = y - y.mean(); pz = p - p.mean()
    denom = (np.sqrt((yz ** 2).sum()) * np.sqrt((pz ** 2).sum()))
    pearson = float((yz * pz).sum() / max(denom, 1e-9))
    return {"MAE": mae, "R2": r2, "Pearson": pearson}


def _cls_metrics(y, p):
    y = np.asarray(y); p = np.asarray(p)
    n = len(y)
    acc = float((y == p).mean())
    # Macro per-class recall
    classes = sorted(set(y.tolist()) | set(p.tolist()))
    recalls = []
    for c in classes:
        m = y == c
        if m.sum() > 0:
            recalls.append(float((p[m] == c).mean()))
    macro = float(np.mean(recalls)) if recalls else 0.0
    # Binary Moderate+ (class >=2) if we have at least 3 classes
    if max(classes) >= 2:
        pos = y >= 2
        pp = p >= 2
        tp = int(((pos) & (pp)).sum()); tn = int(((~pos) & (~pp)).sum())
        fp = int(((~pos) & (pp)).sum()); fn = int(((pos) & (~pp)).sum())
        sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
        return {"acc": acc, "macro_recall": macro, "mod_plus_sens": sens,
                "mod_plus_spec": spec, "mod_plus_acc": (tp + tn) / n}
    return {"acc": acc, "macro_recall": macro}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-a", required=True, type=Path)
    parser.add_argument("--pred-b", required=True, type=Path)
    parser.add_argument("--task", choices=["regression", "classification"], required=True)
    parser.add_argument("--a-label", default="A")
    parser.add_argument("--b-label", default="B")
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.task == "regression":
        a = _load_reg(args.pred_a); b = _load_reg(args.pred_b)
        paired = _pair(a, b)
        print(f"[pair] A={len(a)}, B={len(b)}, paired={len(paired)}")
        if len(paired) < 50:
            print("[ERROR] too few paired rows"); return 1
        y = np.array([r[0][1] for r in paired])
        pa = np.array([r[0][2] for r in paired])
        pb = np.array([r[1][2] for r in paired])
        m_a = _reg_metrics(y, pa)
        m_b = _reg_metrics(y, pb)
        # Bootstrap deltas
        rng = np.random.default_rng(args.seed)
        n = len(y)
        deltas = {k: [] for k in m_a.keys()}
        for _ in range(args.n_boot):
            idx = rng.integers(0, n, size=n)
            ya, paa, pba = y[idx], pa[idx], pb[idx]
            ma = _reg_metrics(ya, paa); mb = _reg_metrics(ya, pba)
            for k in m_a.keys():
                deltas[k].append(ma[k] - mb[k])
        cis = {}
        for k, vs in deltas.items():
            arr = np.array(vs)
            cis[k] = {
                "delta_point": m_a[k] - m_b[k],
                "ci95_low": float(np.percentile(arr, 2.5)),
                "ci95_high": float(np.percentile(arr, 97.5)),
                "p_ne_0": float(min((arr < 0).mean(), (arr > 0).mean()) * 2),
            }
        out = {
            "task": "regression",
            "a_label": args.a_label, "b_label": args.b_label,
            "n_paired": len(paired),
            "a_metrics": m_a, "b_metrics": m_b,
            "delta_a_minus_b": cis,
            "note": "Positive Δ_MAE = A worse; Positive Δ_R² = A better. Paired Δ is per-row diff bootstrap.",
        }
    else:
        a = _load_cls(args.pred_a); b = _load_cls(args.pred_b)
        paired = _pair(a, b)
        print(f"[pair] A={len(a)}, B={len(b)}, paired={len(paired)}")
        if len(paired) < 50:
            print("[ERROR] too few paired rows"); return 1
        y = np.array([r[0][1] for r in paired])
        pa = np.array([r[0][2] for r in paired])
        pb = np.array([r[1][2] for r in paired])
        m_a = _cls_metrics(y, pa); m_b = _cls_metrics(y, pb)
        rng = np.random.default_rng(args.seed)
        n = len(y)
        deltas = {k: [] for k in m_a.keys()}
        for _ in range(args.n_boot):
            idx = rng.integers(0, n, size=n)
            ya, paa, pba = y[idx], pa[idx], pb[idx]
            ma = _cls_metrics(ya, paa); mb = _cls_metrics(ya, pba)
            for k in m_a.keys():
                deltas[k].append(ma[k] - mb[k])
        cis = {}
        for k, vs in deltas.items():
            arr = np.array(vs)
            cis[k] = {
                "delta_point": m_a[k] - m_b[k],
                "ci95_low": float(np.percentile(arr, 2.5)),
                "ci95_high": float(np.percentile(arr, 97.5)),
                "p_ne_0": float(min((arr < 0).mean(), (arr > 0).mean()) * 2),
            }
        out = {
            "task": "classification",
            "a_label": args.a_label, "b_label": args.b_label,
            "n_paired": len(paired),
            "a_metrics": m_a, "b_metrics": m_b,
            "delta_a_minus_b": cis,
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
