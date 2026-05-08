import csv
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

STYLE = {
    "V-JEPA†-e130":        {"color": "#17becf", "marker": "o", "label": "V-JEPA† e130 (19.3 EFLOPs)"},
    "MCC-Anchored":        {"color": "#9467bd", "marker": "X", "label": "MCC-Anchored e25"},
    "TokenRel-Motion-e5":  {"color": "#8fbc8f", "marker": "^", "label": "TokenRel-Motion e5 (3.6 EFLOPs)"},
    "TokenRel-Motion-e25": {"color": "#2ca02c", "marker": "^", "label": "TokenRel-Motion e25 (partial, running)"},
}
ORDER = ["V-JEPA†-e130", "MCC-Anchored", "TokenRel-Motion-e5", "TokenRel-Motion-e25"]

def load(p):
    by = defaultdict(list)
    with open(p) as f:
        for r in csv.DictReader(f):
            if r["phase"] != "val": continue
            try:
                by[r["model"]].append((int(r["epoch"]), float(r["val_acc"]), float(r["val_auroc"]), float(r["val_kappa"])))
            except (ValueError, KeyError):
                continue
    for k in by: by[k].sort()
    return by

def panel(ax, by, col, ylabel):
    for m in ORDER:
        if m not in by: continue
        rows = by[m]
        st = STYLE[m]
        xs = [r[0] for r in rows]
        ys = [r[col] for r in rows]
        ax.plot(xs, ys, marker=st["marker"], color=st["color"], label=st["label"], lw=1.7, ms=5)
    ax.set_xlabel("Probe training epoch")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title(f"{ylabel} (higher = better)", fontsize=11)

by = load("/mnt/custom-file-systems/efs/fs-0049217cdf69186d7_fsap-0fa7145b64eaa046b/vjepa2/claude/neurips/figures/mr/mr_all_variants_trajectories.csv")
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
panel(axes[0], by, 1, "Val accuracy (%)")
panel(axes[1], by, 2, "Val AUROC (4-class macro OVR)")
panel(axes[2], by, 3, "Val kappa")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.03))
fig.suptitle("MIMIC MR A4C 4-class probe — val trajectories across variants", fontsize=12, y=1.08)
fig.tight_layout()
out = "/mnt/custom-file-systems/efs/fs-0049217cdf69186d7_fsap-0fa7145b64eaa046b/vjepa2/claude/neurips/figures/mr/mr_all_variants_trajectories.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
print(f"wrote {out}")
