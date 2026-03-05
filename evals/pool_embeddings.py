"""Pool clip-level embeddings to study-level by averaging across clips per study.

Produces a study-level NPZ with one embedding per study, ready for linear probing.

Usage:
    # Pool master embeddings to study level
    python -m evals.pool_embeddings \
        --embeddings embeddings/nature_medicine/mimic/echojepa_g_mimic_embeddings.npz \
        --clip_index embeddings/nature_medicine/mimic/clip_index.npz \
        --output embeddings/nature_medicine/mimic/echojepa_g_mimic_study.npz

    # Pool with task-specific labels (subset of clips)
    python -m evals.pool_embeddings \
        --embeddings embeddings/nature_medicine/mimic/echojepa_g_mimic_embeddings.npz \
        --clip_index embeddings/nature_medicine/mimic/clip_index.npz \
        --labels embeddings/nature_medicine/mimic/labels/mortality_1yr.npz \
        --output embeddings/nature_medicine/mimic/study_level/mortality_1yr.npz
"""

import argparse
import os

import numpy as np


def pool_to_study_level(embeddings, study_ids, labels=None):
    """Mean-pool clip embeddings to study-level.

    Args:
        embeddings: (N_clips, D) array
        study_ids: (N_clips,) array of study IDs
        labels: (N_clips,) array of labels (must be constant within study)

    Returns:
        study_embeddings: (N_studies, D) array
        study_labels: (N_studies,) array (if labels provided)
        unique_study_ids: (N_studies,) array of study IDs
        clips_per_study: (N_studies,) array of clip counts
    """
    unique_ids, inverse = np.unique(study_ids, return_inverse=True)
    n_studies = len(unique_ids)
    embed_dim = embeddings.shape[1]

    study_embeddings = np.zeros((n_studies, embed_dim), dtype=np.float32)
    clips_per_study = np.zeros(n_studies, dtype=np.int32)
    study_labels = np.zeros(n_studies, dtype=np.float64) if labels is not None else None

    for clip_idx in range(len(embeddings)):
        study_idx = inverse[clip_idx]
        study_embeddings[study_idx] += embeddings[clip_idx]
        clips_per_study[study_idx] += 1
        if labels is not None:
            study_labels[study_idx] = labels[clip_idx]  # Constant within study

    # Mean pool
    study_embeddings /= clips_per_study[:, None]

    return study_embeddings, study_labels, unique_ids, clips_per_study


def main():
    parser = argparse.ArgumentParser(description="Pool clip embeddings to study level")
    parser.add_argument("--embeddings", required=True, help="Master clip-level NPZ")
    parser.add_argument("--clip_index", required=True, help="Clip index NPZ (study_ids, patient_ids, s3_paths)")
    parser.add_argument("--labels", help="Labels-only NPZ from remap_embeddings (optional, subsets clips)")
    parser.add_argument("--output", required=True, help="Output study-level NPZ")
    args = parser.parse_args()

    print(f"Loading embeddings from {args.embeddings}...")
    master = np.load(args.embeddings)
    all_embeddings = master["embeddings"]
    print(f"  Clips: {all_embeddings.shape[0]}, Dim: {all_embeddings.shape[1]}")

    print(f"Loading clip index from {args.clip_index}...")
    index = np.load(args.clip_index, allow_pickle=True)
    all_study_ids = index["study_ids"]
    all_patient_ids = index["patient_ids"]

    if args.labels:
        print(f"Loading task labels from {args.labels}...")
        label_data = np.load(args.labels)
        indices = label_data["indices"]
        labels = label_data["labels"]
        embeddings = all_embeddings[indices]
        study_ids = all_study_ids[indices]
        patient_ids = all_patient_ids[indices]
        print(f"  Subset: {len(indices)} clips")
    else:
        embeddings = all_embeddings
        study_ids = all_study_ids
        patient_ids = all_patient_ids
        labels = master["labels"]

    print("Pooling to study level...")
    study_emb, study_labels, unique_study_ids, clips_per_study = pool_to_study_level(
        embeddings, study_ids, labels
    )

    # Map study_id -> patient_id (take first occurrence)
    study_to_patient = {}
    for sid, pid in zip(study_ids, patient_ids):
        if sid not in study_to_patient:
            study_to_patient[sid] = pid
    study_patient_ids = np.array([study_to_patient[sid] for sid in unique_study_ids])

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.savez(
        args.output,
        embeddings=study_emb,
        labels=study_labels,
        study_ids=unique_study_ids,
        patient_ids=study_patient_ids,
        clips_per_study=clips_per_study,
    )

    print(f"\nSaved to {args.output}:")
    print(f"  Studies: {study_emb.shape[0]}")
    print(f"  Embed dim: {study_emb.shape[1]}")
    print(f"  Patients: {len(np.unique(study_patient_ids))}")
    print(f"  Clips/study: min={clips_per_study.min()}, max={clips_per_study.max()}, "
          f"median={int(np.median(clips_per_study))}, mean={clips_per_study.mean():.1f}")
    if study_labels is not None:
        n_unique = len(np.unique(study_labels))
        print(f"  Labels: {n_unique} unique, range [{study_labels.min():.3f}, {study_labels.max():.3f}]")


if __name__ == "__main__":
    main()
