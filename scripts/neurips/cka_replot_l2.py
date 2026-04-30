"""
Quick replot (Option A): L2-normalize each merged feature vector per layer per
sample, then recompute CKA. Approximates what a final LayerNorm would have
contributed. Uses already-merged features from cka_layerwise_439.

Usage:
    python scripts/neurips/cka_replot_l2.py \
        --feat_dir /tmp/cka_merged \
        --out_dir /home/sagemaker-user/user-default-efs/vjepa2/figures/neurips \
        --suffix _l2
"""

import argparse
import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

DISPLAY_NAMES = {
    "JEPA-IN21K-e100": "JEPA (e100)",
    "BYOL-L-e100": "BYOL (e100)",
    "MAE-L-e99": "MAE (e99)",
    "SALT-S2v1-e79": "SALT (e79)",
}


def linear_cka(X, Y):
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    hsic_xy = torch.norm(Y.T @ X, p="fro") ** 2
    hsic_xx = torch.norm(X.T @ X, p="fro") ** 2
    hsic_yy = torch.norm(Y.T @ Y, p="fro") ** 2
    return (hsic_xy / (torch.sqrt(hsic_xx * hsic_yy) + 1e-10)).item()


def l2_normalize(X):
    # X: [N, D] float tensor
    n = X.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return X / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feat_dir", default="/tmp/cka_merged")
    parser.add_argument("--out_dir", default="figures/neurips")
    parser.add_argument("--suffix", default="_l2")
    parser.add_argument("--vmin", type=float, default=0.3)
    parser.add_argument("--vmax", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    models = ["JEPA-IN21K-e100", "BYOL-L-e100", "MAE-L-e99", "SALT-S2v1-e79"]
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    feats = {}
    valid_sets = []
    for m in models:
        d = np.load(os.path.join(args.feat_dir, f"{m}.npz"), allow_pickle=False)
        feats[m] = (d["features"], d["valid_idx"])
        valid_sets.append(set(d["valid_idx"].tolist()))
        print(f"{m}: features={d['features'].shape}, valid={len(d['valid_idx'])}")

    common = sorted(set.intersection(*valid_sets))
    common_arr = np.array(common, dtype=np.int64)
    print(f"common: {len(common)}")

    # Align + L2-normalize per layer per sample
    aligned = {}
    for m in models:
        f, vi = feats[m]
        idx_map = {v: k for k, v in enumerate(vi.tolist())}
        take = np.array([idx_map[v] for v in common], dtype=np.int64)
        f_aligned = torch.from_numpy(f[:, take, :].astype(np.float32))  # [L, N, D]
        L, N, D = f_aligned.shape
        f_norm = torch.empty_like(f_aligned)
        for l in range(L):
            f_norm[l] = l2_normalize(f_aligned[l])
        aligned[m] = f_norm
        print(f"{m}: aligned+l2 {f_norm.shape}")

    pairs = list(itertools.combinations(models, 2))
    cka_mats = {}
    for a, b in pairs:
        La = aligned[a].shape[0]
        Lb = aligned[b].shape[0]
        M = np.zeros((La, Lb), dtype=np.float32)
        for i in range(La):
            Xi = aligned[a][i].to(device)
            for j in range(Lb):
                Yj = aligned[b][j].to(device)
                M[i, j] = linear_cka(Xi, Yj)
        cka_mats[f"{a}__{b}"] = M
        print(f"{a} vs {b}: mean={M.mean():.3f} diag={np.diag(M).mean():.3f} "
              f"min={M.min():.3f} max={M.max():.3f}")

    np.savez_compressed(
        os.path.join(args.out_dir, f"cka_matrices{args.suffix}.npz"),
        n_common=len(common),
        common_idx=common_arr,
        models=np.array(models),
        **cka_mats,
    )

    # Plot
    ncols, nrows = 3, 2
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
                     f"diag-mean = {diag:.2f}", fontsize=10)
        ax.set_xlabel(f"{DISPLAY_NAMES.get(b, b)} layer")
        ax.set_ylabel(f"{DISPLAY_NAMES.get(a, a)} layer")
        ax.set_xticks([1, 6, 12, 18, 24])
        ax.set_yticks([1, 6, 12, 18, 24])

    for ax in axes[len(pairs):]:
        ax.axis("off")

    cbar = fig.colorbar(im, ax=axes.tolist(), fraction=0.02, pad=0.02, shrink=0.9)
    cbar.set_label("Linear CKA (L2-normalized)")
    fig.suptitle(
        f"Layer-wise CKA (L2-normalized features) — IN21K ViT-L, ~ep100  "
        f"(n={len(common)} EchoNet-Dynamic test)",
        y=1.0, fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 0.93, 0.97])

    pdf = os.path.join(args.out_dir, f"cka_layerwise{args.suffix}.pdf")
    png = os.path.join(args.out_dir, f"cka_layerwise{args.suffix}.png")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=200)
    print(f"[save] {pdf}")
    print(f"[save] {png}")


if __name__ == "__main__":
    main()
