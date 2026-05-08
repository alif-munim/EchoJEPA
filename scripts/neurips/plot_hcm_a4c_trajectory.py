"""HCM A4C probe training trajectory (val AUROC per epoch) with test AUROC
horizontal reference lines.

Striking val/test flip: V-JEPA†-e125 wins val (~0.65) but collapses on test
(0.546, N=2165, prev 2.82%); TokenRel-Motion-e5 loses val (~0.57) but wins
test by +0.214 AUROC (0.760, non-overlapping CIs). Test AUROCs are
softmax-based from rerun 862 with the patched eval.py.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

COLOR = {
    "base_e125":   "#1f77b4",  # blue
    "TokenRel-e5": "#2ca02c",  # green
}
LABEL = {
    "base_e125":   "V-JEPA†-e125 (705)",
    "TokenRel-e5": "TokenRel-Motion-e5 (704, 3.6 EF)",
}
MARKER = {"base_e125": "o", "TokenRel-e5": "^"}

CSV = "/mnt/custom-file-systems/efs/fs-0049217cdf69186d7_fsap-0fa7145b64eaa046b/vjepa2/claude/neurips/figures/hcm/hcm_a4c_trajectories.csv"
OUT = "/mnt/custom-file-systems/efs/fs-0049217cdf69186d7_fsap-0fa7145b64eaa046b/vjepa2/claude/neurips/figures/hcm/hcm_a4c_trajectories.png"


def load_traj(path):
    by_model_val = defaultdict(list)
    test = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            m = r["model"]
            if r["phase"] == "val":
                by_model_val[m].append((int(r["epoch"]), float(r["val_auroc"])))
            elif r["phase"] == "test_ovr_862":
                test[m] = float(r["val_auroc"])
    for m in by_model_val:
        by_model_val[m].sort()
    return by_model_val, test


def main():
    by_model, test_auroc = load_traj(CSV)

    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    for m, rows in by_model.items():
        xs = [e for e, _ in rows]
        ys = [a for _, a in rows]
        ax.plot(xs, ys, marker=MARKER[m], color=COLOR[m], label=f"{LABEL[m]} (val)", lw=1.8, ms=5)
    for m, auroc in test_auroc.items():
        ax.axhline(auroc, color=COLOR[m], ls="--", lw=1.5, alpha=0.75,
                   label=f"{LABEL[m]} test AUROC = {auroc:.3f} (rerun 862)")
    ax.set_xlabel("Probe training epoch")
    ax.set_ylabel("AUROC")
    ax.set_title("HCM A4C (N=2,165, prev 2.82%): val vs test AUROC — striking val/test flip", fontsize=11)
    ax.set_ylim(0.42, 0.82)
    ax.grid(alpha=0.3)
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(loc="lower right", fontsize=9, frameon=True)
    # annotate
    ax.text(1, 0.76, "Test: TokenRel-e5 wins by +0.214 AUROC\n[non-overlapping 95% CIs]",
            fontsize=9, color="#2ca02c", va="top")
    ax.text(1, 0.55, "Val: V-JEPA†-e125 wins by ~+0.08 AUROC",
            fontsize=9, color="#1f77b4", va="top")

    fig.tight_layout()
    out = Path(OUT)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"wrote {out}")
    print(f"wrote {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
