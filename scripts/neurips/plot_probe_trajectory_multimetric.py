"""Generalized 3-panel probe trajectory plot (Val MAE / Val R² / Val Pearson).

Consumes a trajectory CSV with schema:
    model,phase,epoch,train_mae,val_mae,val_r2,val_pearson

Plots one line per model across the three val metrics in one PNG (+PDF).
Used across claude/neurips/figures/ for echonet, peds, tapse, lvh, etc.

Example:
    python plot_probe_trajectory_multimetric.py \\
        --csv claude/neurips/figures/echonet/echonet_lvef_trajectories.csv \\
        --out claude/neurips/figures/echonet/echonet_lvef_trajectories.png \\
        --title "EchoNet-Dynamic LVEF probe trajectories" \\
        --ylabel-mae "Val MAE (EF %)"
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Canonical colour / marker / label map — extend as new models are added.
MODEL_STYLE = {
    # baselines
    "V-JEPA†-e125":          {"color": "#1f77b4", "marker": "o", "label": "V-JEPA† e125"},
    "V-JEPA†":               {"color": "#1f77b4", "marker": "o", "label": "V-JEPA†"},
    "base_e125":             {"color": "#1f77b4", "marker": "o", "label": "V-JEPA† e125"},
    "base_e130":             {"color": "#17becf", "marker": "o", "label": "V-JEPA† e130"},
    "V-JEPA‡":               {"color": "#1f77b4", "marker": "P", "label": "V-JEPA‡ e148"},
    # V4 family
    "MV-PhaseRel":           {"color": "#d62728", "marker": "s", "label": "MV-PhaseRel"},
    "V4-e25":                {"color": "#d62728", "marker": "s", "label": "MV-PhaseRel e25"},
    # V3 family
    "MV-PairedIntra":        {"color": "#ff7f0e", "marker": "D", "label": "MV-PairedIntra"},
    # TokenRel
    "TokenRel-Motion-e25":   {"color": "#2ca02c", "marker": "^", "label": "TokenRel-Motion e25"},
    "TokenRel-Motion-e5":    {"color": "#8fbc8f", "marker": "^", "label": "TokenRel-Motion e5"},
    # MCC / FJ
    "MCC-Anchored-794":      {"color": "#9467bd", "marker": "X", "label": "MCC-Anchored e25"},
    "MCC-Anchored":          {"color": "#9467bd", "marker": "X", "label": "MCC-Anchored e25"},
    "FullJoint-Study-795":   {"color": "#8c564b", "marker": "v", "label": "FullJoint-Study 30k"},
    "FullJoint-Study":       {"color": "#8c564b", "marker": "v", "label": "FullJoint-Study 30k"},
}


def style_for(model: str) -> dict:
    """Return style dict for a model, falling back to a grey line on unknowns."""
    # Strip trailing parenthetical (e.g. "(partial)") or "_lvef_test" suffixes.
    key = model.split(" (")[0].strip().strip('"')
    if key in MODEL_STYLE:
        return MODEL_STYLE[key]
    # Unknown — use neutral style
    return {"color": "#7f7f7f", "marker": "*", "label": model}


def load_traj(csv_path: Path):
    """Read the CSV (handles UTF-8 BOM + CRLF). Returns {model: [(epoch, tr_mae, val_mae, val_r2, val_pearson), ...]}."""
    by_model = defaultdict(list)
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if r.get("phase") != "val":
                continue
            model = r["model"].strip()
            try:
                epoch = int(r["epoch"])
            except (ValueError, TypeError):
                continue
            if epoch < 1:
                continue
            by_model[model].append(
                (
                    epoch,
                    float(r["train_mae"]),
                    float(r["val_mae"]),
                    float(r["val_r2"]),
                    float(r["val_pearson"]),
                )
            )
    for m in by_model:
        by_model[m].sort(key=lambda t: t[0])
    return by_model


def plot_metric(ax, by_model, col_idx, ylabel, lower_better=False):
    for model, rows in by_model.items():
        st = style_for(model)
        xs = [r[0] for r in rows]
        ys = [r[col_idx] for r in rows]
        ax.plot(
            xs,
            ys,
            marker=st["marker"],
            color=st["color"],
            label=st["label"],
            lw=1.8,
            ms=5,
        )
    ax.set_xlabel("Probe training epoch")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    direction = "lower = better" if lower_better else "higher = better"
    ax.set_title(f"{ylabel} ({direction})", fontsize=11)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument(
        "--title",
        default="Probe trajectories",
        help="Figure suptitle.",
    )
    ap.add_argument(
        "--ylabel-mae",
        default="Val MAE",
        help="Y-axis label for the MAE panel (units vary by task).",
    )
    args = ap.parse_args()

    by_model = load_traj(args.csv)
    if not by_model:
        raise SystemExit(f"No val rows loaded from {args.csv}")

    ncols = 3
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4.5))
    plot_metric(axes[0], by_model, col_idx=2, ylabel=args.ylabel_mae, lower_better=True)
    plot_metric(axes[1], by_model, col_idx=3, ylabel="Val R²")
    plot_metric(axes[2], by_model, col_idx=4, ylabel="Val Pearson")

    # Shared legend — dedupe + preserve insertion order per model
    handles, labels = axes[0].get_legend_handles_labels()
    seen = set()
    uniq = []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l)
            uniq.append((h, l))
    fig.legend(
        [h for h, _ in uniq],
        [l for _, l in uniq],
        loc="upper center",
        ncol=min(len(uniq), 4),
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, 1.03),
    )
    fig.suptitle(args.title, fontsize=12, y=1.10)
    fig.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"wrote {args.out}")
    print(f"wrote {args.out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
