"""
Create {model}_splits_allclips folders for each model.

For each model, expands study-level splits into clip-level splits by looking up
all clip embeddings for each study_id from the model's full embeddings file.

Usage:
    python create_allclips.py
"""

import numpy as np
import os
import sys
from collections import defaultdict

BASE = "/mnt/pool/datasets/CY/EchoJEPA/embeddings/mimic/experiments/nature_medicine/mimic"
MODELS = ["echofm", "echojepa_l_kinetics", "echojepa_l", "echoprime", "panecho"]


def get_study_id(path):
    """Extract study_id from S3 path like .../s90001295/90001295_0001.mp4"""
    return int(path.split('/')[-2][1:])


def process_model(model):
    print(f"\n{'='*60}")
    print(f"Processing {model}...")
    print(f"{'='*60}")

    # Load clip-level embeddings
    emb_file = os.path.join(BASE, f"{model}_mimic_embeddings.npz")
    data = np.load(emb_file, allow_pickle=True)
    all_embeddings = data['embeddings']
    all_paths = data['paths']
    print(f"  Loaded {len(all_paths)} clips, embedding dim={all_embeddings.shape[1]}")

    # Build study_id -> clip indices mapping (skip padding entries)
    study_to_clips = defaultdict(list)
    for i, p in enumerate(all_paths):
        if not p.startswith('s3://'):
            continue
        sid = get_study_id(p)
        study_to_clips[sid].append(i)
    print(f"  Built index: {len(study_to_clips)} unique studies")

    # Process each task
    splits_dir = os.path.join(BASE, f"{model}_splits")
    allclips_dir = os.path.join(BASE, f"{model}_splits_allclips")
    tasks = sorted(os.listdir(splits_dir))

    for task in tasks:
        task_out = os.path.join(allclips_dir, task)
        os.makedirs(task_out, exist_ok=True)

        for split in ["train", "val", "test"]:
            split_file = os.path.join(splits_dir, task, f"{split}.npz")
            if not os.path.exists(split_file):
                print(f"  WARNING: {split_file} not found, skipping")
                continue

            sp = np.load(split_file, allow_pickle=True)
            study_ids = sp['study_ids']
            patient_ids = sp['patient_ids']
            labels = sp['labels']

            # Expand to all clips
            exp_embeddings = []
            exp_labels = []
            exp_study_ids = []
            exp_patient_ids = []
            missing = 0

            for j in range(len(study_ids)):
                sid = int(study_ids[j])
                clip_indices = study_to_clips.get(sid, [])
                if len(clip_indices) == 0:
                    missing += 1
                    continue
                n_clips = len(clip_indices)
                exp_embeddings.append(all_embeddings[clip_indices])
                exp_labels.append(np.full(n_clips, labels[j]))
                exp_study_ids.append(np.full(n_clips, study_ids[j]))
                exp_patient_ids.append(np.full(n_clips, patient_ids[j]))

            if len(exp_embeddings) == 0:
                print(f"  WARNING: {task}/{split} has no clips, skipping")
                continue

            out = {
                'embeddings': np.concatenate(exp_embeddings, axis=0),
                'labels': np.concatenate(exp_labels, axis=0),
                'study_ids': np.concatenate(exp_study_ids, axis=0),
                'patient_ids': np.concatenate(exp_patient_ids, axis=0),
            }

            out_path = os.path.join(task_out, f"{split}.npz")
            np.savez(out_path, **out)

            if missing > 0:
                print(f"  {task}/{split}: {out['embeddings'].shape[0]} clips "
                      f"from {len(study_ids)-missing}/{len(study_ids)} studies "
                      f"({missing} missing)")

        sys.stdout.flush()

    print(f"  Done: {model} -> {allclips_dir}")
    # Free memory
    del data, all_embeddings, all_paths, study_to_clips


if __name__ == "__main__":
    for model in MODELS:
        process_model(model)
    print("\nAll models complete!")
