"""
Merge per-shard CKA feature NPZs into one file per model.

Reads scripts/neurips/cka/features/{MODEL}.shard{i}of{N}.npz for all i,
concatenates along the video axis in global CSV order, writes
scripts/neurips/cka/features/{MODEL}.npz.

Usage:
    python scripts/neurips/cka_merge_shards.py \
        --models JEPA-IN21K-e100 BYOL-L-e100 MAE-L-e99 SALT-S2v1-e79 \
        --num_shards 4
"""

import argparse
import glob
import os

import numpy as np


def merge_model(model, num_shards, feat_dir):
    shards = []
    for s in range(num_shards):
        path = os.path.join(feat_dir, f"{model}.shard{s}of{num_shards}.npz")
        if not os.path.exists(path):
            pattern = os.path.join(feat_dir, f"{model}.shard*of{num_shards}.npz")
            raise FileNotFoundError(
                f"Missing shard {s}: {path}\n"
                f"  existing: {sorted(glob.glob(pattern))}"
            )
        shards.append(np.load(path, allow_pickle=False))

    # [L, sum(N_s), D] for mean-pool or [L, sum(N_s), K, D] for per-token.
    feats = np.concatenate([s["features"] for s in shards], axis=1)
    valid_idx = np.concatenate([s["valid_idx"] for s in shards], axis=0)
    order = np.argsort(valid_idx)
    feats = feats[:, order, ...]
    valid_idx = valid_idx[order]

    out_path = os.path.join(feat_dir, f"{model}.npz")
    np.savez_compressed(
        out_path,
        features=feats,
        model=model,
        valid_idx=valid_idx,
    )
    print(f"[{model}] merged {num_shards} shards → {out_path}  shape={feats.shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--num_shards", type=int, required=True)
    parser.add_argument("--feat_dir", default="scripts/neurips/cka/features")
    args = parser.parse_args()

    for m in args.models:
        merge_model(m, args.num_shards, args.feat_dir)


if __name__ == "__main__":
    main()
