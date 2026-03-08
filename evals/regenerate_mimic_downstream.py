#!/usr/bin/env python
"""Regenerate all MIMIC study-level NPZs and train/val/test splits.

After fixing the shuffle bug in master NPZs, the downstream study-level
pooled embeddings and patient-level splits need regeneration. This script
handles all 7 models × 23 tasks in one pass.

Usage:
    python -m evals.regenerate_mimic_downstream \
        --mimic_dir experiments/nature_medicine/mimic

    # Single model only
    python -m evals.regenerate_mimic_downstream \
        --mimic_dir experiments/nature_medicine/mimic \
        --model echojepa_g
"""

import argparse
import json
import logging
import os

import numpy as np

from evals.pool_embeddings import pool_to_study_level

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODELS = [
    "echojepa_g",
    "echojepa_l",
    "echojepa_l_kinetics",
    "echomae",
    "echofm",
    "panecho",
    "echoprime",
]


def main():
    parser = argparse.ArgumentParser(description="Regenerate MIMIC study-level and splits")
    parser.add_argument("--mimic_dir", required=True, help="MIMIC embeddings directory")
    parser.add_argument("--model", help="Single model to regenerate (default: all)")
    args = parser.parse_args()

    mimic = args.mimic_dir
    models = [args.model] if args.model else MODELS

    # Load shared files
    clip_index_path = os.path.join(mimic, "clip_index.npz")
    split_path = os.path.join(mimic, "patient_split.json")
    labels_dir = os.path.join(mimic, "labels")

    logger.info(f"Loading clip index from {clip_index_path}...")
    clip_index = np.load(clip_index_path, allow_pickle=True)
    all_study_ids = clip_index["study_ids"]
    all_patient_ids = clip_index["patient_ids"]

    logger.info(f"Loading patient split from {split_path}...")
    with open(split_path) as f:
        patient_split = json.load(f)

    # Get all task label NPZs
    task_files = sorted(f for f in os.listdir(labels_dir) if f.endswith(".npz"))
    logger.info(f"Found {len(task_files)} tasks in {labels_dir}")

    for model in models:
        master_path = os.path.join(mimic, f"{model}_mimic_embeddings.npz")
        if not os.path.exists(master_path):
            logger.warning(f"Skipping {model}: {master_path} not found")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Processing model: {model}")
        logger.info(f"{'='*60}")

        logger.info(f"Loading {master_path}...")
        master = np.load(master_path)
        all_embeddings = master["embeddings"]
        embed_dim = all_embeddings.shape[1]
        logger.info(f"  Shape: {all_embeddings.shape}")

        study_dir = os.path.join(mimic, f"{model}_study_level")
        splits_dir = os.path.join(mimic, f"{model}_splits")
        os.makedirs(study_dir, exist_ok=True)

        for task_file in task_files:
            task_name = task_file.replace(".npz", "")
            label_data = np.load(os.path.join(labels_dir, task_file))
            indices = label_data["indices"]
            labels = label_data["labels"]

            # Subset to this task
            task_embeddings = all_embeddings[indices]
            task_study_ids = all_study_ids[indices]
            task_patient_ids = all_patient_ids[indices]

            # Pool to study level
            study_emb, study_labels, unique_study_ids, clips_per_study = pool_to_study_level(
                task_embeddings, task_study_ids, labels
            )

            # Map study_id -> patient_id
            study_to_patient = {}
            for sid, pid in zip(task_study_ids, task_patient_ids):
                if sid not in study_to_patient:
                    study_to_patient[sid] = pid
            study_patient_ids = np.array([study_to_patient[sid] for sid in unique_study_ids])

            # Save study-level NPZ
            study_out = os.path.join(study_dir, task_file)
            np.savez(
                study_out,
                embeddings=study_emb,
                labels=study_labels,
                study_ids=unique_study_ids,
                patient_ids=study_patient_ids,
                clips_per_study=clips_per_study,
            )

            # Create patient-level splits
            task_splits_dir = os.path.join(splits_dir, task_name)
            os.makedirs(task_splits_dir, exist_ok=True)

            split_assignments = np.array([patient_split.get(str(pid), "unknown") for pid in study_patient_ids])

            for split_name in ["train", "val", "test"]:
                mask = split_assignments == split_name
                split_out = os.path.join(task_splits_dir, f"{split_name}.npz")
                np.savez(
                    split_out,
                    embeddings=study_emb[mask],
                    labels=study_labels[mask],
                    study_ids=unique_study_ids[mask],
                    patient_ids=study_patient_ids[mask],
                )

            n_train = int((split_assignments == "train").sum())
            n_val = int((split_assignments == "val").sum())
            n_test = int((split_assignments == "test").sum())
            logger.info(f"  {task_name}: {len(study_emb)} studies → {n_train}/{n_val}/{n_test}")

        logger.info(f"Done with {model}")

    logger.info("\nAll done!")


if __name__ == "__main__":
    main()
