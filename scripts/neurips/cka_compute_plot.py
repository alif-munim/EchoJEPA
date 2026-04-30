"""
Compute pairwise layer-vs-layer CKA matrices and plot as a 2×3 heatmap grid.

Reads:  scripts/neurips/cka/features/{MODEL}.npz   (from cka_merge_shards.py)
Writes: scripts/neurips/cka/cka_matrices.npz       (6 pairs × 24 × 24)
        figures/neurips/cka_layerwise.{pdf,png}

Uses linear CKA (Kornblith et al. 2019) computed on the common subset of
valid videos across all four models.

Usage:
    python scripts/neurips/cka_compute_plot.py \
        --models JEPA-IN21K-e100 BYOL-L-e100 MAE-L-e99 SALT-S2v1-e79 \
        --device cuda:0
"""

import argparse
import itertools
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, ".")

DISPLAY_NAMES = {
    "JEPA-IN21K-e100": "JEPA (e100)",
    "BYOL-L-e100": "BYOL (e100)",
    "MAE-L-e99": "MAE (e99)",
    "SALT-S2v1-e79": "SALT (e79)",
}


def linear_cka(X, Y):
    """Linear CKA (Kornblith 2019). X: [N,D1], Y: [N,D2]. Returns scalar in [0,1]."""
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    hsic_xy = torch.norm(Y.T @ X, p="fro") ** 2
    hsic_xx = torch.norm(X.T @ X, p="fro") ** 2
    hsic_yy = torch.norm(Y.T @ Y, p="fro") ** 2
    return (hsic_xy / (torch.sqrt(hsic_xx * hsic_yy) + 1e-10)).item()


def load_model_features(model, feat_dir):
    path = os.path.join(feat_dir, f"{model}.npz")
    data = np.load(path, allow_pickle=False)
    return data["features"], data["valid_idx"]  # [L, N, D], [N]


def compute_cka_matrix(feats_a, feats_b, device):
    """feats_*: torch tensor with leading dim = layers. Returns [L_a, L_b] CKA.

    Accepts either [L, N, D] (mean-pooled) or [L, N, K, D] (per-token). For
    per-token, N×K are flattened so each layer compares [N*K, D] rows.
    """
    def _flat(t):
        if t.dim() == 4:
            return t.reshape(t.shape[0], -1, t.shape[-1])
        return t

    fa = _flat(feats_a)
    fb = _flat(feats_b)
    La = fa.shape[0]
    Lb = fb.shape[0]
    M = np.zeros((La, Lb), dtype=np.float32)
    for i in range(La):
        Xi = fa[i].to(device, dtype=torch.float32)
        for j in range(Lb):
            Yj = fb[j].to(device, dtype=torch.float32)
            M[i, j] = linear_cka(Xi, Yj)
    return M


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--feat_dir", default="scripts/neurips/cka/features")
    parser.add_argument("--out_npz", default="scripts/neurips/cka/cka_matrices.npz")
    parser.add_argument("--fig_dir", default="figures/neurips")
    parser.add_argument("--fig_name", default="cka_layerwise")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--vmin", type=float, default=0.0)
    parser.add_argument("--vmax", type=float, default=1.0)
    parser.add_argument("--title_suffix", default="")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_npz), exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load all models + intersect valid_idx so CKA uses the same videos.
    feats_by_model = {}
    valid_sets = []
    for m in args.models:
        f, vi = load_model_features(m, args.feat_dir)
        feats_by_model[m] = (f, vi)
        valid_sets.append(set(vi.tolist()))
        print(f"[load] {m}: features={f.shape}, valid={len(vi)}")

    common = sorted(set.intersection(*valid_sets))
    print(f"[intersect] {len(common)} videos valid across all {len(args.models)} models")
    if len(common) == 0:
        raise RuntimeError("No videos valid across all models")

    # Restrict each model's features to common order (handles [L,N,D] and [L,N,K,D]).
    common_arr = np.array(common, dtype=np.int64)
    feats_t = {}
    for m in args.models:
        f, vi = feats_by_model[m]
        idx_map = {v: k for k, v in enumerate(vi.tolist())}
        take = np.array([idx_map[v] for v in common], dtype=np.int64)
        # np.take along axis 1 covers both 3D and 4D shapes. Keep fp16 on CPU
        # to bound RAM for per-token mode; cast to fp32 inside CKA per layer.
        f_aligned = np.take(f, take, axis=1)
        feats_t[m] = torch.from_numpy(f_aligned)  # keeps source dtype (fp16)
        print(f"[align] {m}: {feats_t[m].shape}  dtype={feats_t[m].dtype}")

    # Compute CKA for all pairs
    pairs = list(itertools.combinations(args.models, 2))
    cka_mats = {}
    for a, b in pairs:
        print(f"[cka] {a} vs {b} …")
        M = compute_cka_matrix(feats_t[a], feats_t[b], device)
        cka_mats[f"{a}__{b}"] = M
        print(f"       mean={M.mean():.3f}  diag_mean={np.diag(M).mean():.3f}  max={M.max():.3f}")

    # Save matrices
    np.savez_compressed(
        args.out_npz,
        n_common=len(common),
        common_idx=common_arr,
        models=np.array(args.models),
        **cka_mats,
    )
    print(f"[save] {args.out_npz}")

    # Plot: 2×3 grid for 6 pairs
    n_pairs = len(pairs)
    ncols = 3
    nrows = int(np.ceil(n_pairs / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.4, nrows * 3.6))
    axes = np.atleast_1d(axes).flatten()

    for ax, (a, b) in zip(axes, pairs):
        M = cka_mats[f"{a}__{b}"]
        La, Lb = M.shape
        im = ax.imshow(M, origin="lower", vmin=args.vmin, vmax=args.vmax,
                       cmap="viridis",
                       extent=[0.5, Lb + 0.5, 0.5, La + 0.5], aspect="equal")
        diag = np.diag(M).mean()
        ax.set_title(f"{DISPLAY_NAMES.get(a, a)} vs {DISPLAY_NAMES.get(b, b)}\n"
                     f"diag-mean = {diag:.2f}",
                     fontsize=10)
        ax.set_xlabel(f"{DISPLAY_NAMES.get(b, b)} layer")
        ax.set_ylabel(f"{DISPLAY_NAMES.get(a, a)} layer")
        ticks = [1, 6, 12, 18, 24] if La == 24 else list(range(1, La + 1, max(1, La // 5)))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

    # Hide any unused axes
    for ax in axes[len(pairs):]:
        ax.axis("off")

    cbar = fig.colorbar(im, ax=axes.tolist(), fraction=0.02, pad=0.02, shrink=0.9)
    cbar.set_label("Linear CKA")
    fig.suptitle(
        f"Layer-wise CKA — IN21K-init ViT-L, ~ep100  "
        f"(n={len(common)} EchoNet-Dynamic test){args.title_suffix}",
        y=1.0, fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 0.93, 0.97])

    pdf_path = os.path.join(args.fig_dir, f"{args.fig_name}.pdf")
    png_path = os.path.join(args.fig_dir, f"{args.fig_name}.png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=200)
    print(f"[save] {pdf_path}")
    print(f"[save] {png_path}")


if __name__ == "__main__":
    main()
