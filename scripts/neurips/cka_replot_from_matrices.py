"""
Replot CKA heatmaps from a saved cka_matrices*.npz (no recomputation).

Use this to iterate on layout without re-running CKA. Places the colorbar in
its own reserved strip on the right via GridSpec so it cannot overlap the
rightmost subplots.

Usage:
    python scripts/neurips/cka_replot_from_matrices.py \
        --npz figures/neurips/cka_matrices_pertoken.npz \
        --out_name cka_layerwise_pertoken \
        --title_suffix "per-token K=32, LN" \
        --vmin 0.3 --vmax 1.0
"""

import argparse
import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

DISPLAY_NAMES = {
    "JEPA-IN21K-e100": "JEPA (e100)",
    "BYOL-L-e100": "BYOL (e100)",
    "MAE-L-e99": "MAE (e99)",
    "SALT-S2v1-e79": "SALT (e79)",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True)
    parser.add_argument("--out_dir", default="figures/neurips")
    parser.add_argument("--out_name", required=True)
    parser.add_argument("--title_suffix", default="")
    parser.add_argument("--vmin", type=float, default=0.3)
    parser.add_argument("--vmax", type=float, default=1.0)
    parser.add_argument("--cmap", default="viridis")
    args = parser.parse_args()

    data = np.load(args.npz, allow_pickle=False)
    n_common = int(data["n_common"])
    models = list(data["models"])
    pairs = list(itertools.combinations(models, 2))

    ncols = 3
    nrows = 2

    # Reserve a dedicated colorbar column (thin, far right). The cbar_col
    # never hosts heatmaps, so no overlap is possible.
    fig = plt.figure(figsize=(ncols * 3.4 + 0.8, nrows * 3.6 + 0.6))
    gs = GridSpec(
        nrows, ncols + 1, figure=fig,
        width_ratios=[1.0] * ncols + [0.06],
        wspace=0.35, hspace=0.45,
    )

    im = None
    for idx, (a, b) in enumerate(pairs):
        r, c = divmod(idx, ncols)
        ax = fig.add_subplot(gs[r, c])
        M = data[f"{a}__{b}"]
        La, Lb = M.shape
        im = ax.imshow(
            M, origin="lower", vmin=args.vmin, vmax=args.vmax, cmap=args.cmap,
            extent=[0.5, Lb + 0.5, 0.5, La + 0.5], aspect="equal",
        )
        diag = np.diag(M).mean()
        ax.set_title(
            f"{DISPLAY_NAMES.get(a, a)} vs {DISPLAY_NAMES.get(b, b)}\n"
            f"diag-mean = {diag:.2f}",
            fontsize=10,
        )
        ax.set_xlabel(f"{DISPLAY_NAMES.get(b, b)} layer")
        ax.set_ylabel(f"{DISPLAY_NAMES.get(a, a)} layer")
        ticks = [1, 6, 12, 18, 24] if La == 24 else list(range(1, La + 1, max(1, La // 5)))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

    cax = fig.add_subplot(gs[:, ncols])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Linear CKA")

    suffix = f" — {args.title_suffix}" if args.title_suffix else ""
    fig.suptitle(
        f"Layer-wise CKA — IN21K-init ViT-L, ~ep100  "
        f"(n={n_common} EchoNet-Dynamic test){suffix}",
        fontsize=11,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    pdf = os.path.join(args.out_dir, f"{args.out_name}.pdf")
    png = os.path.join(args.out_dir, f"{args.out_name}.png")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=200)
    print(f"[save] {pdf}")
    print(f"[save] {png}")


if __name__ == "__main__":
    main()
