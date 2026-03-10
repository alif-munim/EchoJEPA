"""
Build view classification NPZs for linear probe training.

Matches view-labeled clips (from classifier/data_prep pipeline) to clip-level
embeddings (from UHN 18M extraction) by SOP Instance UID, then saves
train/val/test NPZs per model.

Usage:
    python scripts/build_view_probe_npzs.py \
        --master_csv experiments/nature_medicine/uhn/uhn_all_clips.csv \
        --view_train data/csv/uhn_views_22k_train_cleaned.csv \
        --view_val data/csv/uhn_views_22k_val_cleaned.csv \
        --view_test data/csv/uhn_views_22k_test_cleaned.csv \
        --models echojepa_g echojepa_l \
        --emb_dir experiments/nature_medicine/uhn \
        --output_dir experiments/nature_medicine/uhn/view_splits
"""

import argparse
import logging
import os
import time

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def extract_sop_filename(s3_path):
    """Extract SOP Instance UID filename from S3 path."""
    return s3_path.strip().split("/")[-1]


def build_sop_to_index(master_csv):
    """Build SOP filename -> row index mapping from master clip CSV."""
    logger.info(f"Building SOP->index mapping from {master_csv}...")
    sop_to_idx = {}
    t0 = time.time()
    with open(master_csv) as f:
        for i, line in enumerate(f):
            path = line.strip().split()[0]
            sop = extract_sop_filename(path)
            sop_to_idx[sop] = i
            if (i + 1) % 5_000_000 == 0:
                logger.info(f"  {(i+1)/1e6:.0f}M lines ({time.time()-t0:.0f}s)")
    logger.info(f"  Done: {len(sop_to_idx)} unique SOPs in {time.time()-t0:.0f}s")
    return sop_to_idx


def load_view_csv(csv_path):
    """Load view CSV, return list of (sop_filename, int_label)."""
    entries = []
    with open(csv_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            path, label = parts[0], int(parts[1])
            sop = extract_sop_filename(path)
            entries.append((sop, label))
    return entries


def match_view_to_indices(view_entries, sop_to_idx):
    """Match view entries to master CSV row indices."""
    matched_indices = []
    matched_labels = []
    missing = 0
    for sop, label in view_entries:
        if sop in sop_to_idx:
            matched_indices.append(sop_to_idx[sop])
            matched_labels.append(label)
        else:
            missing += 1
    if missing > 0:
        logger.warning(f"  {missing}/{len(view_entries)} clips not found in master index")
    return np.array(matched_indices, dtype=np.int64), np.array(matched_labels, dtype=np.int64)


def load_clip_embeddings_by_indices(emb_path, indices):
    """Load clip embeddings for specific row indices from merged clip_embeddings.npz."""
    logger.info(f"Loading embeddings from {emb_path} for {len(indices)} clips...")
    t0 = time.time()
    data = np.load(emb_path, mmap_mode="r")
    keys = list(data.keys())

    if "embeddings" in keys:
        emb_key = "embeddings"
    elif "clip_embeddings" in keys:
        emb_key = "clip_embeddings"
    else:
        raise KeyError(f"No embedding key found. Keys: {keys}")

    all_embs = data[emb_key]
    logger.info(f"  Full array shape: {all_embs.shape}")

    sorted_order = np.argsort(indices)
    sorted_indices = indices[sorted_order]
    selected = np.array(all_embs[sorted_indices])

    # Unsort back to original order
    unsort = np.argsort(sorted_order)
    selected = selected[unsort]

    logger.info(f"  Loaded {selected.shape} in {time.time()-t0:.0f}s")
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--master_csv", required=True)
    parser.add_argument("--view_train", required=True)
    parser.add_argument("--view_val", required=True)
    parser.add_argument("--view_test", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--emb_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    # Step 1: Build SOP -> master row index
    sop_to_idx = build_sop_to_index(args.master_csv)

    # Step 2: Match view clips to master indices
    splits = {}
    for name, path in [("train", args.view_train), ("val", args.view_val), ("test", args.view_test)]:
        entries = load_view_csv(path)
        indices, labels = match_view_to_indices(entries, sop_to_idx)
        splits[name] = {"indices": indices, "labels": labels}
        logger.info(f"  {name}: {len(indices)} matched clips, {len(np.unique(labels))} classes")

    # Step 3: For each model, extract embeddings and save NPZs
    for model in args.models:
        emb_file = os.path.join(args.emb_dir, f"{model}_embeddings", "clip_embeddings.npz")
        if not os.path.exists(emb_file):
            logger.warning(f"Skipping {model}: {emb_file} not found")
            continue

        model_dir = os.path.join(args.output_dir, model)
        os.makedirs(model_dir, exist_ok=True)

        for split_name, split_data in splits.items():
            indices = split_data["indices"]
            labels = split_data["labels"]

            embeddings = load_clip_embeddings_by_indices(emb_file, indices)

            out_path = os.path.join(model_dir, f"{split_name}.npz")
            np.savez(out_path, embeddings=embeddings, labels=labels)
            logger.info(f"  Saved {out_path}: embeddings={embeddings.shape}, labels={labels.shape}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
