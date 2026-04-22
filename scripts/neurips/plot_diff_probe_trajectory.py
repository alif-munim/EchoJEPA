"""4-panel trajectory plot: diff probe, raw probe, matched_frame clean, matched_frame shuffle.

Pulls diff/raw probe R² (mean ± 95% bootstrap CI from job 314) from primary_all.csv
and matched_frame clean/shuffle R² (single-value from jobs 216/220) from a hardcoded
dict mirroring claude/neurips/experiments/frame-shuffling-results.md.
"""

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ---- matched_frame numbers from claude/neurips/experiments/frame-shuffling-results.md ----
# Protocol D: depth=4 attentive probes, RoPE remap, num_segments=2, prediction averaging.
MATCHED_FRAME = {
    "jepa": {
        25:  {"clean": 0.460, "mf": 0.416},
        50:  {"clean": 0.612, "mf": 0.328},
        75:  {"clean": 0.629, "mf": 0.439},
        100: {"clean": 0.650, "mf": 0.507},
    },
    "byol": {
        24:  {"clean": 0.437, "mf": -0.294},
        50:  {"clean": 0.477, "mf": 0.108},
        75:  {"clean": 0.505, "mf": 0.287},
        100: {"clean": 0.527, "mf": 0.249},
    },
    "mae": {
        25:  {"clean": 0.225, "mf": 0.257},
        50:  {"clean": 0.413, "mf": 0.281},
        75:  {"clean": 0.435, "mf": 0.356},
        99:  {"clean": 0.467, "mf": 0.440},
        124: {"clean": 0.469, "mf": 0.428},
        149: {"clean": 0.527, "mf": 0.491},
        174: {"clean": 0.500, "mf": 0.448},
        194: {"clean": 0.526, "mf": 0.460},
    },
    "salt": {
        4:  {"clean": 0.028, "mf": -0.020},
        29: {"clean": 0.356, "mf": -0.443},
        54: {"clean": 0.431, "mf": -0.417},
        79: {"clean": 0.402, "mf": -0.404},
    },
}

COLOR = {
    "diff":     "#d62728",  # red
    "raw":      "#1f77b4",  # blue
    "mf_clean": "#2ca02c",  # green
    "mf_shuf":  "#ff7f0e",  # orange
}
LABEL = {
    "diff":     "Diff probe (linear-A, wd=1e-4)",
    "raw":      "Raw probe (linear-A, wd=1e-4)",
    "mf_clean": "Attn-probe clean (jobs 216/220)",
    "mf_shuf":  "Attn-probe matched-frame",
}

MODEL_EPOCHS = {
    "jepa": [25, 50, 75, 100],
    "mae":  [25, 50, 75, 99, 124, 149, 174, 194],
    "byol": [24, 50, 75, 100],
    "salt": [4, 29, 54, 79],
}


def load_primary(csv_path: Path):
    rows = []
    with csv_path.open() as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def mean_and_bootstrap_ci(vals, n_boot=10000, seed=0, alpha=0.05):
    rng = np.random.default_rng(seed)
    vals = np.asarray(vals, dtype=float)
    if len(vals) == 0:
        return float("nan"), float("nan"), float("nan")
    means = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(vals), len(vals))
        means.append(vals[idx].mean())
    means = np.sort(np.asarray(means))
    lo = means[int(alpha / 2 * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot)]
    return float(vals.mean()), float(lo), float(hi)


def family_numbers(rows, family, inp):
    """Return sorted list of (epoch, mean, lo, hi) for linear-A, wd=1e-4, given input."""
    per_epoch = defaultdict(list)
    for r in rows:
        model = r["model"]
        if not model.startswith(f"{family}_e"):
            continue
        if r["arch"] != "linear-A" or r["input"] != inp or r["wd"] != "0.0001":
            continue
        epoch = int(model.split("_e")[1])
        per_epoch[epoch].append(float(r["test_r2"]))
    out = []
    for ep in sorted(per_epoch):
        mu, lo, hi = mean_and_bootstrap_ci(per_epoch[ep])
        out.append((ep, mu, lo, hi))
    return out


def plot_panel(ax, family, rows, ylim):
    diff_pts = family_numbers(rows, family, "diff")
    raw_pts  = family_numbers(rows, family, "raw")

    # --- diff / raw with bootstrap CI bands ---
    for pts, key in [(diff_pts, "diff"), (raw_pts, "raw")]:
        if not pts:
            continue
        xs  = [p[0] for p in pts]
        mu  = [p[1] for p in pts]
        lo  = [p[2] for p in pts]
        hi  = [p[3] for p in pts]
        ax.plot(xs, mu, "-o", color=COLOR[key], label=LABEL[key], lw=1.6, ms=4)
        ax.fill_between(xs, lo, hi, color=COLOR[key], alpha=0.18, linewidth=0)

    # --- matched_frame clean / shuffle (no CI; single aggregate value) ---
    mf = MATCHED_FRAME[family]
    xs_mf = sorted(mf.keys())
    clean = [mf[e]["clean"] for e in xs_mf]
    shuf  = [mf[e]["mf"]    for e in xs_mf]
    ax.plot(xs_mf, clean, "--s", color=COLOR["mf_clean"], label=LABEL["mf_clean"], lw=1.2, ms=4, alpha=0.9)
    ax.plot(xs_mf, shuf,  "--^", color=COLOR["mf_shuf"],  label=LABEL["mf_shuf"],  lw=1.2, ms=4, alpha=0.9)

    ax.axhline(0.0, color="k", lw=0.6, alpha=0.4)
    ax.set_title(family.upper(), fontsize=11)
    ax.set_xlabel("Training epoch")
    ax.set_ylim(*ylim)
    ax.grid(alpha=0.3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="/tmp/diff_probe_314/primary_all.csv")
    ap.add_argument("--out", default="/tmp/diff_probe_314/trajectory.png")
    args = ap.parse_args()

    rows = load_primary(Path(args.csv))

    # Determine shared Y-axis: clamp to [-0.4, 0.8] to include BYOL/SALT matched_frame negatives
    ylim = (-0.5, 0.8)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), sharey=True)
    for ax, fam in zip(axes, ["jepa", "mae", "byol", "salt"]):
        plot_panel(ax, fam, rows, ylim)
    axes[0].set_ylabel("LVEF test R²")

    # Shared legend at the top
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 1.02))

    fig.suptitle("Diff-probe trajectory vs matched-frame attentive probe (EchoNet-Dynamic LVEF)",
                 fontsize=12, y=1.06)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    fig.savefig(args.out.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"wrote {args.out}")
    print(f"wrote {args.out.replace('.png', '.pdf')}")


if __name__ == "__main__":
    main()
