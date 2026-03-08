#!/usr/bin/env python
"""Generate UHN per-task train/val/test split NPZs for probe training.

Joins study-level embeddings with label NPZs on study_ids, then splits
by the split assignment already present in each label file.

Usage:
    # All available models, all tasks (including trajectory)
    python -m evals.regenerate_uhn_downstream \
        --uhn_dir experiments/nature_medicine/uhn

    # Single model
    python -m evals.regenerate_uhn_downstream \
        --uhn_dir experiments/nature_medicine/uhn \
        --model echojepa_g
"""

import argparse
import logging
import os

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODELS = [
    "echojepa_g",
    "echojepa_l",
    "echojepa_l_kinetics",
    "echomae",
    "random_init",
]


def collect_label_files(labels_dir):
    """Collect all label NPZ files, including trajectory/ subdirectory."""
    tasks = []
    for f in sorted(os.listdir(labels_dir)):
        path = os.path.join(labels_dir, f)
        if f.endswith(".npz"):
            tasks.append((f.replace(".npz", ""), path))
        elif os.path.isdir(path):
            for sf in sorted(os.listdir(path)):
                if sf.endswith(".npz"):
                    task_name = f"{f}/{sf.replace('.npz', '')}"
                    tasks.append((task_name, os.path.join(path, sf)))
    return tasks


def process_model(model, uhn_dir, tasks):
    """Generate splits for one model across all tasks."""
    emb_path = os.path.join(uhn_dir, f"{model}_embeddings", "study_embeddings.npz")
    if not os.path.exists(emb_path):
        logger.warning(f"Skipping {model}: {emb_path} not found")
        return False

    logger.info(f"\n{'='*60}")
    logger.info(f"Processing model: {model}")
    logger.info(f"{'='*60}")

    emb_data = np.load(emb_path, allow_pickle=True)
    emb_study_ids = emb_data["study_ids"]
    embeddings = emb_data["embeddings"]
    logger.info(f"  Embeddings: {embeddings.shape}")

    # Build study_id -> row index lookup
    sid_to_idx = {sid: i for i, sid in enumerate(emb_study_ids)}

    splits_dir = os.path.join(uhn_dir, f"{model}_splits")
    total_tasks = 0

    for task_name, label_path in tasks:
        label_data = np.load(label_path, allow_pickle=True)

        # Detect trajectory (paired) vs standard (single-study) format
        is_trajectory = "study_id_1" in label_data

        if is_trajectory:
            sids_1 = label_data["study_id_1"]
            sids_2 = label_data["study_id_2"]
            splits = label_data["splits"]
            patient_ids = label_data["patient_ids"]

            mask = np.array(
                [(s1 in sid_to_idx and s2 in sid_to_idx) for s1, s2 in zip(sids_1, sids_2)]
            )
            if mask.sum() == 0:
                logger.warning(f"  {task_name}: no overlapping study pairs, skipping")
                continue

            task_dir = os.path.join(splits_dir, task_name)
            os.makedirs(task_dir, exist_ok=True)

            emb_1 = np.array([embeddings[sid_to_idx[s]] for s in sids_1[mask]])
            emb_2 = np.array([embeddings[sid_to_idx[s]] for s in sids_2[mask]])

            split_counts = {}
            for split_name in ["train", "val", "test"]:
                split_mask = splits[mask] == split_name
                n = int(split_mask.sum())
                split_counts[split_name] = n
                if n == 0:
                    continue
                save_dict = dict(
                    embeddings_1=emb_1[split_mask],
                    embeddings_2=emb_2[split_mask],
                    study_id_1=sids_1[mask][split_mask],
                    study_id_2=sids_2[mask][split_mask],
                    patient_ids=patient_ids[mask][split_mask],
                    days_between=label_data["days_between"][mask][split_mask],
                )
                for opt_key in ["label_1", "label_2", "delta"]:
                    if opt_key in label_data:
                        save_dict[opt_key] = label_data[opt_key][mask][split_mask]
                np.savez(os.path.join(task_dir, f"{split_name}.npz"), **save_dict)

            dropped_n = len(sids_1) - int(mask.sum())
            drop_str = f" ({dropped_n} dropped)" if dropped_n > 0 else ""
            logger.info(
                f"  {task_name}: {mask.sum()} pairs{drop_str} → "
                f"{split_counts.get('train', 0)}/{split_counts.get('val', 0)}/{split_counts.get('test', 0)}"
            )
        else:
            label_sids = label_data["study_ids"]
            labels = label_data["labels"]
            splits = label_data["splits"]
            patient_ids = label_data["patient_ids"]

            # Find studies present in both labels and embeddings
            mask = np.array([sid in sid_to_idx for sid in label_sids])
            if mask.sum() == 0:
                logger.warning(f"  {task_name}: no overlapping studies, skipping")
                continue

            matched_sids = label_sids[mask]
            matched_labels = labels[mask]
            matched_splits = splits[mask]
            matched_pids = patient_ids[mask]
            matched_emb = np.array([embeddings[sid_to_idx[sid]] for sid in matched_sids])

            # Create per-split NPZs
            task_dir = os.path.join(splits_dir, task_name)
            os.makedirs(task_dir, exist_ok=True)

            split_counts = {}
            for split_name in ["train", "val", "test"]:
                split_mask = matched_splits == split_name
                n = int(split_mask.sum())
                split_counts[split_name] = n
                if n == 0:
                    continue
                np.savez(
                    os.path.join(task_dir, f"{split_name}.npz"),
                    embeddings=matched_emb[split_mask],
                    labels=matched_labels[split_mask],
                    study_ids=matched_sids[split_mask],
                    patient_ids=matched_pids[split_mask],
                )

            dropped_n = len(label_sids) - int(mask.sum())
            drop_str = f" ({dropped_n} dropped)" if dropped_n > 0 else ""
            logger.info(
                f"  {task_name}: {mask.sum()} studies{drop_str} → "
                f"{split_counts.get('train', 0)}/{split_counts.get('val', 0)}/{split_counts.get('test', 0)}"
            )
        total_tasks += 1

    logger.info(f"Done with {model}: {total_tasks} tasks written to {splits_dir}/")
    return True


def main():
    parser = argparse.ArgumentParser(description="Generate UHN per-task splits")
    parser.add_argument("--uhn_dir", required=True, help="UHN embeddings directory")
    parser.add_argument("--model", help="Single model to process (default: all available)")
    args = parser.parse_args()

    labels_dir = os.path.join(args.uhn_dir, "labels")
    tasks = collect_label_files(labels_dir)
    logger.info(f"Found {len(tasks)} tasks in {labels_dir}")

    models = [args.model] if args.model else MODELS
    processed = 0
    for model in models:
        if process_model(model, args.uhn_dir, tasks):
            processed += 1

    logger.info(f"\nAll done! Processed {processed} model(s).")


if __name__ == "__main__":
    main()
