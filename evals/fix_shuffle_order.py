#!/usr/bin/env python
"""
Fix embedding order for extractions that used DistributedSampler(shuffle=True).

The original extraction used shuffle=True, which permuted clip order relative to
the CSV/clip_index.npz. The merge step assumed CSV-order alignment, so
study_embeddings.npz has incorrect study assignments.

This script:
1. Reproduces the exact permutation (deterministic: seed=0, epoch=0)
2. Reorders clip_embeddings.npz to match CSV order
3. Re-pools to study level

Usage:
    python -m evals.fix_shuffle_order \
        --embeddings_dir experiments/nature_medicine/uhn/echojepa_g_embeddings \
        --clip_index experiments/nature_medicine/uhn/uhn_clip_index.npz \
        --n_clips 18111412 \
        --world_size 8
"""

import argparse
import logging
import math
import os

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def reconstruct_permutation(n_clips: int, world_size: int) -> np.ndarray:
    """Reproduce the exact DistributedSampler permutation order.

    Returns an array where result[i] = the CSV index of the clip at position i
    in the shuffled merge order.
    """
    # DistributedSampler pads to even distribution
    total_size = math.ceil(n_clips / world_size) * world_size

    # Reproduce the shuffled permutation (seed=0, epoch=0)
    g = torch.Generator()
    g.manual_seed(0)  # seed + epoch = 0 + 0
    perm = torch.randperm(n_clips, generator=g).tolist()

    # Pad (same as DistributedSampler)
    perm += perm[: (total_size - len(perm))]

    # Reconstruct the interleaved merge order.
    # Each rank R gets perm[R::world_size], and the global_idx formula
    # interleaves ranks: position 0=rank0[0], 1=rank1[0], ..., W-1=rankW-1[0],
    # W=rank0[1], W+1=rank1[1], ...
    # So the merged order (sorted by global_idx) is:
    # [rank0[0], rank1[0], ..., rankW-1[0], rank0[1], rank1[1], ...]
    # which is just perm[0], perm[1], perm[2], ... = the full permutation.
    return np.array(perm[:n_clips])


def main():
    parser = argparse.ArgumentParser(description="Fix shuffle ordering in extracted embeddings")
    parser.add_argument("--embeddings_dir", required=True, help="Dir with clip_embeddings.npz")
    parser.add_argument("--clip_index", required=True, help="uhn_clip_index.npz")
    parser.add_argument("--n_clips", type=int, required=True, help="Total clips in dataset")
    parser.add_argument("--world_size", type=int, default=8)
    args = parser.parse_args()

    clip_emb_path = os.path.join(args.embeddings_dir, "clip_embeddings.npz")
    logger.info(f"Loading {clip_emb_path}...")
    data = np.load(clip_emb_path)
    embeddings = data["embeddings"]
    logger.info(f"Loaded embeddings: {embeddings.shape}")

    # Reconstruct the permutation
    logger.info("Reconstructing shuffle permutation...")
    perm = reconstruct_permutation(args.n_clips, args.world_size)

    n_emb = len(embeddings)
    n_perm = len(perm)
    n = min(n_emb, n_perm)
    if n_emb != n_perm:
        logger.warning(f"Embeddings ({n_emb}) != perm ({n_perm}), using min={n}")

    # Reorder: embeddings[i] is the embedding of perm[i], we want CSV order.
    # Some perm values may exceed n (when extraction is incomplete), so filter.
    logger.info("Reordering to CSV order...")
    perm_n = perm[:n]
    valid_mask = perm_n < n  # only keep assignments where target index is in range
    n_invalid = n - int(valid_mask.sum())
    if n_invalid > 0:
        logger.warning(f"{n_invalid} clips have perm target >= {n}, skipping (will be zero-filled)")

    reordered = np.zeros_like(embeddings[:n])
    valid_indices = np.where(valid_mask)[0]
    reordered[perm_n[valid_mask]] = embeddings[valid_indices]

    # Save fixed clip embeddings
    backup_path = clip_emb_path + ".shuffled_backup"
    if not os.path.exists(backup_path):
        os.rename(clip_emb_path, backup_path)
        logger.info(f"Backed up original to {backup_path}")

    np.savez(clip_emb_path, embeddings=reordered)
    logger.info(f"Saved reordered clip embeddings: {reordered.shape}")

    # Re-pool to study level
    logger.info("Re-pooling to study level...")
    clip_index = np.load(args.clip_index, allow_pickle=True)
    study_uids = clip_index["study_uids"][:n]

    unique_studies, inverse = np.unique(study_uids, return_inverse=True)
    n_studies = len(unique_studies)
    embed_dim = reordered.shape[1]

    study_embeddings = np.zeros((n_studies, embed_dim), dtype=np.float64)
    clips_per_study = np.zeros(n_studies, dtype=np.int32)

    for i in range(n):
        study_embeddings[inverse[i]] += reordered[i]
        clips_per_study[inverse[i]] += 1

    study_embeddings = (study_embeddings / clips_per_study[:, None]).astype(np.float32)

    study_path = os.path.join(args.embeddings_dir, "study_embeddings.npz")
    backup_study = study_path + ".shuffled_backup"
    if not os.path.exists(backup_study):
        os.rename(study_path, backup_study)
        logger.info(f"Backed up original study embeddings to {backup_study}")

    np.savez(
        study_path,
        embeddings=study_embeddings,
        study_ids=unique_studies,
        clips_per_study=clips_per_study,
    )
    logger.info(f"Saved fixed study embeddings: {study_embeddings.shape} ({n_studies} studies)")
    logger.info("Done!")


if __name__ == "__main__":
    main()
