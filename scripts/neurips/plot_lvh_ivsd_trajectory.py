"""3-panel LVH IVSD trajectory: Val MAE, Val R², Val Pearson.

Three encoders under comparison on EchoNet-LVH IVSD regression:
  - base_e130:    V-JEPA IN21K continuation checkpoint at e130 (~19.3 EFLOPs post-e100)
  - V4-e25:       MV-PhaseRel +25 (job 593 e25) (~18 EFLOPs post-e100)
  - TokenRel-e25: token-level phase-relational head +25 (job 703 e25) (~18 EFLOPs post-e100)

Mechanism test: V4 lost IVSD decisively (ΔR² = −0.225) — does TokenRel's
token-level (vs V4's pooled) phase supervision preserve fine spatial
features? If TokenRel ≈ base, pooled InfoNCE is specifically damaging;
if TokenRel ≈ V4, phase supervision in general is the issue.

Reads trajectory CSV written by the probes' log_r0.csv files.
Writes PNG + PDF into claude/neurips/figures/lvh/.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

COLOR = {
    "base_e130":    "#1f77b4",  # blue — V-JEPA IN21K baseline
    "V4-e25":       "#d62728",  # red — MV-PhaseRel (pooled InfoNCE head)
    "TokenRel-e25": "#2ca02c",  # green — TokenRel-Motion (token-level head)
}
LABEL = {
    "base_e130":    "V-JEPA† e130 (~19.3 EFLOPs)",
    "V4-e25":       "MV-PhaseRel e25 (~18 EFLOPs)",
    "TokenRel-e25": "TokenRel-Motion e25 (~18 EFLOPs, running, partial)",
}
MARKER = {"base_e130": "o", "V4-e25": "s", "TokenRel-e25": "^"}


def load_traj(csv_path: Path):
    by_model = defaultdict(list)
    with csv_path.open() as f:
        for r in csv.DictReader(f):
            if r["phase"] != "val":
                continue
            by_model[r["model"]].append((
                int(r["epoch"]),
                float(r["train_mae"]),
                float(r["val_mae"]),
                float(r["val_r2"]),
                float(r["val_pearson"]),
            ))
    for m in by_model:
        by_model[m].sort(key=lambda t: t[0])
    return by_model


def plot_metric(ax, by_model, col_idx, ylabel, lower_better=False):
    for model, rows in by_model.items():
        xs = [r[0] for r in rows]
        ys = [r[col_idx] for r in rows]
        ax.plot(
            xs, ys,
            marker=MARKER[model], color=COLOR[model], label=LABEL[model],
            lw=1.8, ms=5,
        )
    ax.set_xlabel("Probe training epoch")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    # Integer x-ticks
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if lower_better:
        ax.set_title(f"{ylabel} (lower = better)", fontsize=11)
    else:
        ax.set_title(f"{ylabel} (higher = better)", fontsize=11)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        default="/mnt/custom-file-systems/efs/fs-0049217cdf69186d7_fsap-0fa7145b64eaa046b/vjepa2/claude/neurips/figures/lvh/lvh_ivsd_trajectories.csv",
    )
    ap.add_argument(
        "--out",
        default="/mnt/custom-file-systems/efs/fs-0049217cdf69186d7_fsap-0fa7145b64eaa046b/vjepa2/claude/neurips/figures/lvh/lvh_ivsd_trajectories.png",
    )
    args = ap.parse_args()

    by_model = load_traj(Path(args.csv))
    if not by_model:
        raise SystemExit(f"No val rows loaded from {args.csv}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    plot_metric(axes[0], by_model, col_idx=2, ylabel="Val MAE (cm)", lower_better=True)
    plot_metric(axes[1], by_model, col_idx=3, ylabel="Val R²")
    plot_metric(axes[2], by_model, col_idx=4, ylabel="Val Pearson")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center", ncol=3, frameon=False, fontsize=10,
        bbox_to_anchor=(0.5, 1.03),
    )
    fig.suptitle(
        "EchoNet-LVH IVSD probe trajectories (TokenRel-e25 running, partial)",
        fontsize=12, y=1.08,
    )
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"wrote {out}")
    print(f"wrote {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
