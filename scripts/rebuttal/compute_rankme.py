"""
Compute RankMe (effective rank / dimensionality) from saved encoder features.

RankMe (Garrido et al. 2023, arXiv:2210.02885):
    RankMe(Z) = exp(-Σ_i p_i log p_i)
    where p_i = σ_i / (Σ_j σ_j + ε) and σ_i are singular values of Z.

This is the spectral entropy of normalized singular values, then exponentiated
to give an "effective number of dimensions". The result is bounded above by
the smaller of (n_samples, embed_dim).

Usage:
    python scripts/rebuttal/compute_rankme.py path/to/clip_outputs.npz [--label NAME]
    python scripts/rebuttal/compute_rankme.py path/to/features.npz --feature-key embeddings

The script auto-detects the feature key:
- "clip_features_best_head" (from eval pipeline clip_outputs.npz)
- "embeddings" (from speckle_probing_*.npz)
"""

import argparse
import numpy as np
import sys
from pathlib import Path


def rankme(Z: np.ndarray, eps: float = 1e-7) -> float:
    """
    Compute RankMe effective rank of an [N, D] feature matrix.

    Args:
        Z: feature matrix of shape (N, D), N samples each with D dimensions.
        eps: small constant for numerical stability when computing log.

    Returns:
        Effective rank (float). Bounded by min(N, D).
    """
    if Z.ndim != 2:
        raise ValueError(f"Expected 2D feature matrix, got shape {Z.shape}")

    # SVD: σ_i are the singular values, sorted in descending order
    sigma = np.linalg.svd(Z, compute_uv=False)

    # Normalize to a probability distribution: p_i = σ_i / Σ σ_j
    sigma_sum = sigma.sum()
    if sigma_sum < eps:
        return 0.0
    p = sigma / sigma_sum

    # Spectral entropy then exponentiate
    # RankMe = exp(-Σ p_i log p_i)
    p_safe = p + eps
    entropy = -np.sum(p * np.log(p_safe))
    return float(np.exp(entropy))


def load_features(path: Path, feature_key: str = None) -> tuple[np.ndarray, str]:
    """
    Load a [N, D] feature matrix from an npz file.

    Tries auto-detection if feature_key is not specified.
    """
    data = np.load(path, allow_pickle=True)
    keys = list(data.keys())

    if feature_key is not None:
        if feature_key not in keys:
            raise KeyError(f"Key '{feature_key}' not in {path.name}. Available: {keys}")
        feats = data[feature_key]
        return feats, feature_key

    # Auto-detect
    candidates = [
        "clip_features_best_head",  # eval pipeline clip_outputs.npz
        "embeddings",                # speckle_probing_*.npz
        "features",                  # generic
    ]
    for k in candidates:
        if k in keys:
            return data[k], k

    raise KeyError(
        f"No feature key found in {path.name}. "
        f"Available: {keys}. "
        f"Try --feature-key to specify."
    )


def main():
    parser = argparse.ArgumentParser(description="Compute RankMe effective rank")
    parser.add_argument("npz_path", type=Path, help="Path to .npz file with features")
    parser.add_argument("--label", default=None, help="Friendly label for output")
    parser.add_argument("--feature-key", default=None, help="Key in npz (auto-detected if omitted)")
    parser.add_argument(
        "--aggregate",
        choices=["none", "study"],
        default="none",
        help="Aggregation: 'none' = use features as-is; 'study' = average per study_id",
    )
    args = parser.parse_args()

    if not args.npz_path.exists():
        print(f"ERROR: file not found: {args.npz_path}", file=sys.stderr)
        sys.exit(1)

    feats, key = load_features(args.npz_path, args.feature_key)
    label = args.label or args.npz_path.stem

    # Optional study-level aggregation (average all clips per study)
    if args.aggregate == "study":
        data = np.load(args.npz_path, allow_pickle=True)
        if "clip_study_ids" not in data.keys():
            print("ERROR: --aggregate=study requires 'clip_study_ids' in npz", file=sys.stderr)
            sys.exit(1)
        study_ids = data["clip_study_ids"]
        unique_ids, inv = np.unique(study_ids, return_inverse=True)
        study_feats = np.zeros((len(unique_ids), feats.shape[1]), dtype=np.float32)
        for i in range(len(unique_ids)):
            study_feats[i] = feats[inv == i].mean(axis=0)
        feats = study_feats
        print(f"Aggregated {len(study_ids)} clips → {len(unique_ids)} studies")

    n, d = feats.shape
    rank = rankme(feats)

    print(f"=== {label} ===")
    print(f"  Source: {args.npz_path}")
    print(f"  Feature key: {key}")
    print(f"  Shape: ({n}, {d})")
    print(f"  RankMe (effective dim): {rank:.4f}")
    print(f"  Usage: {rank/d*100:.1f}% of embed_dim")
    print(f"  Theoretical max: {min(n, d)}")


if __name__ == "__main__":
    main()
