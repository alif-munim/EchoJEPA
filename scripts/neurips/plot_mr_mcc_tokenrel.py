import csv
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

STYLE = {
    "MCC-Anchored":        {"color": "#9467bd", "marker": "X", "label": "MCC-Anchored e25"},
    "TokenRel-Motion-e25": {"color": "#2ca02c", "marker": "^", "label": "TokenRel-Motion e25"},
}

def load(p):
    by = defaultdict(list)
    with open(p) as f:
        for r in csv.DictReader(f):
            if r["phase"] != "val": continue
            by[r["model"]].append((int(r["epoch"]),
                                    float(r["val_acc"]),
                                    float(r["val_auroc"]),
                                    float(r["val_kappa"])))
    for k in by: by[k].sort()
    return by

def panel(ax, by, col, ylabel, note=""):
    for m, rows in by.items():
        st = STYLE[m]
        xs = [r[0] for r in rows]
        ys = [r[col] for r in rows]
        ax.plot(xs, ys, marker=st["marker"], color=st["color"], label=st["label"], lw=1.8, ms=5)
    ax.set_xlabel("Probe training epoch")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title(f"{ylabel} (higher = better){note}", fontsize=11)

by = load("/mnt/custom-file-systems/efs/fs-0049217cdf69186d7_fsap-0fa7145b64eaa046b/vjepa2/claude/neurips/figures/mr/mr_mcc_tokenrel_trajectories.csv")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
panel(axes[0], by, 1, "Val accuracy (%)")
panel(axes[1], by, 2, "Val AUROC (4-class macro OVR)")
panel(axes[2], by, 3, "Val kappa")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 1.03))
fig.suptitle("MIMIC MR A4C 4-class probe — MCC vs TokenRel-Motion e25 (val trajectories)", fontsize=12, y=1.08)
fig.tight_layout()

out = "/mnt/custom-file-systems/efs/fs-0049217cdf69186d7_fsap-0fa7145b64eaa046b/vjepa2/claude/neurips/figures/mr/mr_mcc_tokenrel_trajectories.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
print(f"wrote {out}")
