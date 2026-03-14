#!/usr/bin/env python3
"""Build label NPZ files for UHN probe evaluation tasks from source data.

Produces NPZ files with keys: study_ids, patient_ids, labels, splits.
These are consumed by build_probe_csvs.py to produce the final training CSVs.

This script replaces the deleted interactive notebook that originally built
the labels. See DATASET_PROVENANCE.md for full data provenance documentation.

Data sources:
    - echo.db: syngo_measures (regression measurements)
    - data/aws/aws_syngo.csv: StudyRef -> DeidentifiedStudyID mapping (320K rows)
    - data/aws/aws_syngo_findings_v2.csv: Syngo observations with common_label
    - data/aws/aws_heartlab_findings_v2.csv: HeartLab findings with common_label
    - study_deid_map (in echo.db): DeidentifiedStudyID -> OriginalPatientID
    - patient_split.json: patient_id -> train/val/test split

Usage:
    # Build all 4 tasks
    python build_labels.py --all

    # Build specific task
    python build_labels.py --task lvef

    # Validate against existing NPZs
    python build_labels.py --all --validate

    # Specify output directory
    python build_labels.py --all --output_dir labels_v2
"""

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Paths relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
ECHO_DB = REPO_ROOT / "uhn_echo" / "nature_medicine" / "data_exploration" / "echo.db"
AWS_SYNGO_CSV = REPO_ROOT / "data" / "aws" / "aws_syngo.csv"
AWS_SYNGO_FINDINGS = REPO_ROOT / "data" / "aws" / "aws_syngo_findings_v2.csv"
AWS_HEARTLAB_FINDINGS = REPO_ROOT / "data" / "aws" / "aws_heartlab_findings_v2.csv"
PATIENT_SPLIT = SCRIPT_DIR / "patient_split.json"
EXISTING_LABELS = SCRIPT_DIR / "labels"

TASKS = ["lvef", "tapse", "mr_severity", "as_severity"]


def load_studyref_mapping():
    """Load StudyRef -> (DeidentifiedStudyID, patient_id) from aws_syngo.csv.

    This CSV maps Syngo integer StudyRefs to deidentified DICOM UIDs.
    320K entries covering all Syngo studies with videos on S3.
    """
    df = pd.read_csv(AWS_SYNGO_CSV, usecols=["STUDY_REF", "PATIENT_ID", "DeidentifiedStudyID"])
    ref_to_deid = dict(zip(df["STUDY_REF"].astype(str), df["DeidentifiedStudyID"]))
    ref_to_patient = dict(zip(df["STUDY_REF"].astype(str), df["PATIENT_ID"].astype(str)))
    print(f"  Loaded {len(ref_to_deid):,} StudyRef->DeidID mappings from aws_syngo.csv")
    return ref_to_deid, ref_to_patient


def load_deid_to_patient():
    """Build DeidentifiedStudyID -> patient_id from both Syngo and HeartLab sources.

    Combines:
    - aws_syngo.csv (PATIENT_ID, numeric, for Syngo studies)
    - study_deid_map in echo.db (OriginalPatientID, numeric, for HeartLab studies)
    aws_syngo.csv takes priority for studies present in both.
    """
    # HeartLab path: study_deid_map
    conn = sqlite3.connect(str(ECHO_DB))
    rows = conn.execute("SELECT DeidentifiedStudyID, OriginalPatientID FROM study_deid_map").fetchall()
    conn.close()
    hl_map = {r[0]: r[1] for r in rows}

    # Syngo path: aws_syngo.csv
    df = pd.read_csv(AWS_SYNGO_CSV, usecols=["DeidentifiedStudyID", "PATIENT_ID"])
    syngo_map = dict(zip(df["DeidentifiedStudyID"], df["PATIENT_ID"].astype(str)))

    # Merge (Syngo takes priority)
    combined = {**hl_map, **syngo_map}
    print(f"  Loaded {len(combined):,} DeidID->patient_id mappings (Syngo: {len(syngo_map):,}, HeartLab: {len(hl_map):,})")
    return combined


def load_patient_splits():
    """Load patient_id -> split (train/val/test) mapping."""
    with open(PATIENT_SPLIT) as f:
        splits = json.load(f)
    print(f"  Loaded {len(splits):,} patient splits")
    return splits


def build_lvef(ref_to_deid, ref_to_patient, patient_splits):
    """Build LVEF labels from syngo_measures.

    Source: MeasurementName = 'LV EF, MOD BP' (biplane modified Simpson's).
    This is the gold-standard LVEF measurement method.

    - 92K Syngo StudyRefs, ~51K map to deidentified studies with S3 videos
    - Multiple measurements per study averaged
    - Values <= 0 excluded (measurement artifacts)
    - NPZ stores raw EF percentage (not z-scored); z-scoring done at CSV stage
    """
    print("\n--- Building LVEF ---")
    conn = sqlite3.connect(str(ECHO_DB))
    rows = conn.execute(
        """
        SELECT StudyRef, CAST(Value AS REAL) as val
        FROM syngo_measures
        WHERE MeasurementName = 'LV EF, MOD BP'
          AND Value IS NOT NULL
        """
    ).fetchall()
    conn.close()
    print(f"  Raw rows from syngo_measures: {len(rows):,}")

    # Map to deidentified IDs
    records = []
    for ref, val in rows:
        deid = ref_to_deid.get(str(ref))
        patient = ref_to_patient.get(str(ref))
        if deid and patient:
            records.append({"study_id": deid, "patient_id": patient, "label": val})

    df = pd.DataFrame(records)
    print(f"  After StudyRef mapping: {len(df):,} rows, {df['study_id'].nunique():,} studies")

    # Average multiple measurements per study
    df = df.groupby(["study_id", "patient_id"])["label"].mean().reset_index()

    # Filter: exclude values <= 0 (artifact from measurement system)
    n_before = len(df)
    df = df[df["label"] > 0]
    n_filtered = n_before - len(df)
    if n_filtered > 0:
        print(f"  Filtered {n_filtered} studies with EF <= 0")

    # Filter to patients in split
    df = df[df["patient_id"].isin(patient_splits)]
    print(f"  After patient filter: {len(df):,} studies")

    # Assign splits
    df["split"] = df["patient_id"].map(patient_splits)

    return _to_npz_arrays(df, dtype_label=np.float32)


def build_tapse(ref_to_deid, ref_to_patient, patient_splits):
    """Build TAPSE labels from syngo_measures.

    Source: MeasurementName = 'TAPSE (M-mode)' (tricuspid annular plane
    systolic excursion measured by M-mode).

    - 104K Syngo rows, ~55K map to deidentified studies
    - Multiple measurements per study averaged
    - Values outside [0.3, 5.0] cm excluded (unit errors: some stored in mm
      instead of cm, or contain impossible values like 202 cm)
    - NPZ stores raw TAPSE in cm; z-scoring done at CSV stage
    """
    print("\n--- Building TAPSE ---")
    conn = sqlite3.connect(str(ECHO_DB))
    rows = conn.execute(
        """
        SELECT StudyRef, CAST(Value AS REAL) as val
        FROM syngo_measures
        WHERE MeasurementName = 'TAPSE (M-mode)'
          AND Value IS NOT NULL
        """
    ).fetchall()
    conn.close()
    print(f"  Raw rows from syngo_measures: {len(rows):,}")

    records = []
    for ref, val in rows:
        deid = ref_to_deid.get(str(ref))
        patient = ref_to_patient.get(str(ref))
        if deid and patient:
            records.append({"study_id": deid, "patient_id": patient, "label": val})

    df = pd.DataFrame(records)
    print(f"  After StudyRef mapping: {len(df):,} rows, {df['study_id'].nunique():,} studies")

    df = df.groupby(["study_id", "patient_id"])["label"].mean().reset_index()

    # Filter: [0.3, 5.0] cm — values outside this are unit errors
    # (e.g., 202 cm = stored in mm, 0.0 = failed measurement)
    n_before = len(df)
    df = df[(df["label"] >= 0.3) & (df["label"] <= 5.0)]
    n_filtered = n_before - len(df)
    if n_filtered > 0:
        print(f"  Filtered {n_filtered} studies outside [0.3, 5.0] cm")

    df = df[df["patient_id"].isin(patient_splits)]
    print(f"  After patient filter: {len(df):,} studies")

    df["split"] = df["patient_id"].map(patient_splits)

    return _to_npz_arrays(df, dtype_label=np.float32)


def build_mr_severity(deid_to_patient, patient_splits):
    """Build MR severity labels by merging Syngo observations and HeartLab findings.

    Classes (5):
        0 = none
        1 = trace (Syngo only; HeartLab has no trace grade)
        2 = mild (includes mild_to_moderate)
        3 = moderate (includes moderate_to_severe)
        4 = severe

    Sources:
        - aws_syngo_findings_v2.csv: common_label in {MR_none, MR_trace, MR_mild, MR_moderate, MR_severe}
          Syngo observation Name = 'MV_Regurgitation_obs'. Intermediate grades (mild_to_moderate,
          moderate_to_severe) are already collapsed into the adjacent class in common_label.
        - aws_heartlab_findings_v2.csv: same common_label values (no MR_trace in HeartLab)
          HeartLab Finding Group 93 ("mitral valve/function/regurgitation").

    Merge: union by DeidentifiedStudyID, highest severity class wins.
    """
    print("\n--- Building MR severity ---")
    CLASS_MAP = {"MR_none": 0, "MR_trace": 1, "MR_mild": 2, "MR_moderate": 3, "MR_severe": 4}
    return _build_classification_from_csvs(CLASS_MAP, deid_to_patient, patient_splits, "MR severity")


def build_as_severity(deid_to_patient, patient_splits):
    """Build AS severity labels by merging Syngo observations and HeartLab findings.

    Classes (4):
        0 = none/sclerosis (non-obstructive calcification merged with none)
        1 = mild (includes mild_to_moderate)
        2 = moderate (includes moderate_to_severe)
        3 = severe

    Sources:
        - aws_syngo_findings_v2.csv: common_label in {AS_none, AS_mild, AS_moderate, AS_severe}
          Syngo observation Name = 'AoV_sten_degree_SD_obs'. Note: the raw syngo_observations
          table also has 'AoV_Sclerosis_Stenosis_obs' (78K rows) but those are NOT in the
          source CSV — the source CSV uses 'AoV_sten_degree_SD_obs' (21K studies).
        - aws_heartlab_findings_v2.csv: same common_label values. HeartLab contributes the
          majority (~155K vs ~21K from Syngo).

    Merge: union by DeidentifiedStudyID, highest severity class wins.
    """
    print("\n--- Building AS severity ---")
    CLASS_MAP = {"AS_none": 0, "AS_mild": 1, "AS_moderate": 2, "AS_severe": 3}
    return _build_classification_from_csvs(CLASS_MAP, deid_to_patient, patient_splits, "AS severity")


def _build_classification_from_csvs(class_map, deid_to_patient, patient_splits, task_name):
    """Shared logic for classification tasks built from source CSVs."""
    # Syngo source
    syngo = pd.read_csv(AWS_SYNGO_FINDINGS)
    syngo_task = syngo[syngo["common_label"].isin(class_map)][["DeidentifiedStudyID", "common_label"]].copy()
    syngo_task["class"] = syngo_task["common_label"].map(class_map)
    syngo_task = syngo_task.groupby("DeidentifiedStudyID")["class"].max().reset_index()
    print(f"  Syngo: {len(syngo_task):,} unique studies")

    # HeartLab source
    hl = pd.read_csv(AWS_HEARTLAB_FINDINGS)
    hl_task = hl[hl["common_label"].isin(class_map)][["DeidentifiedStudyID", "common_label"]].copy()
    hl_task["class"] = hl_task["common_label"].map(class_map)
    hl_task = hl_task.groupby("DeidentifiedStudyID")["class"].max().reset_index()
    print(f"  HeartLab: {len(hl_task):,} unique studies")

    # Merge: union, highest severity wins
    merged = pd.concat([syngo_task, hl_task]).groupby("DeidentifiedStudyID")["class"].max().reset_index()
    print(f"  Merged: {len(merged):,} unique studies")

    # Map to patient_id
    merged["patient_id"] = merged["DeidentifiedStudyID"].map(deid_to_patient)
    n_unmapped = merged["patient_id"].isna().sum()
    if n_unmapped > 0:
        print(f"  Warning: {n_unmapped} studies have no patient_id mapping (dropped)")
    merged = merged.dropna(subset=["patient_id"])

    # Filter to patients in split
    merged = merged[merged["patient_id"].isin(patient_splits)]
    print(f"  After patient filter: {len(merged):,} studies")

    # Assign splits
    merged["split"] = merged["patient_id"].map(patient_splits)

    # Report class distribution
    for cls_val in sorted(class_map.values()):
        cls_name = [k for k, v in class_map.items() if v == cls_val][0]
        count = (merged["class"] == cls_val).sum()
        print(f"    class {cls_val} ({cls_name}): {count:,}")

    df = merged.rename(columns={"DeidentifiedStudyID": "study_id", "class": "label"})
    return _to_npz_arrays(df, dtype_label=np.int32)


def _to_npz_arrays(df, dtype_label):
    """Convert DataFrame to NPZ-ready arrays."""
    return {
        "study_ids": df["study_id"].values.astype(object),
        "patient_ids": df["patient_id"].values.astype(object),
        "labels": df["label"].values.astype(dtype_label),
        "splits": df["split"].values.astype(object),
    }


def validate_against_existing(task, arrays, existing_dir):
    """Compare reconstructed arrays to existing NPZ."""
    existing_path = existing_dir / f"{task}.npz"
    if not existing_path.exists():
        print(f"  [VALIDATE] No existing NPZ at {existing_path}")
        return

    existing = np.load(str(existing_path), allow_pickle=True)
    n_new = len(arrays["study_ids"])
    n_old = len(existing["study_ids"])

    old_sids = set(existing["study_ids"])
    new_sids = set(arrays["study_ids"])
    overlap = old_sids & new_sids
    in_old_not_new = old_sids - new_sids
    in_new_not_old = new_sids - old_sids

    print(f"  [VALIDATE] Existing: {n_old:,}, New: {n_new:,}")
    print(f"  [VALIDATE] Overlap: {len(overlap):,}, In old not new: {len(in_old_not_new)}, In new not old: {len(in_new_not_old)}")

    # Value comparison for overlapping studies
    if overlap:
        old_dict = dict(zip(existing["study_ids"], existing["labels"]))
        new_dict = dict(zip(arrays["study_ids"], arrays["labels"]))
        diffs = [abs(float(old_dict[s]) - float(new_dict[s])) for s in overlap]
        exact = sum(1 for d in diffs if d < 0.001)
        print(f"  [VALIDATE] Value match: {exact:,}/{len(overlap):,} exact, max_diff={max(diffs):.6f}")

    # Split comparison
    old_splits = dict(zip(existing["study_ids"], existing["splits"]))
    new_splits = dict(zip(arrays["study_ids"], arrays["splits"]))
    split_match = sum(1 for s in overlap if old_splits[s] == new_splits[s])
    print(f"  [VALIDATE] Split match: {split_match:,}/{len(overlap):,}")


def save_npz(task, arrays, output_dir):
    """Save arrays to NPZ file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{task}.npz"
    np.savez(str(path), **arrays)
    print(f"  Saved: {path} ({len(arrays['study_ids']):,} studies)")


def main():
    parser = argparse.ArgumentParser(description="Build UHN label NPZ files from source data")
    parser.add_argument("--task", type=str, choices=TASKS, help="Build a single task")
    parser.add_argument("--all", action="store_true", help="Build all tasks")
    parser.add_argument("--output_dir", type=str, default="labels_v2", help="Output directory name (relative to script dir)")
    parser.add_argument("--validate", action="store_true", help="Compare against existing NPZs in labels/")
    args = parser.parse_args()

    if not args.task and not args.all:
        parser.error("Specify --task or --all")

    tasks = TASKS if args.all else [args.task]
    output_dir = SCRIPT_DIR / args.output_dir

    print("Loading shared data...")
    patient_splits = load_patient_splits()

    # Regression tasks need StudyRef mapping
    needs_regression = any(t in tasks for t in ["lvef", "tapse"])
    ref_to_deid, ref_to_patient = (None, None)
    if needs_regression:
        ref_to_deid, ref_to_patient = load_studyref_mapping()

    # Classification tasks need deid->patient mapping
    needs_classification = any(t in tasks for t in ["mr_severity", "as_severity"])
    deid_to_patient = None
    if needs_classification:
        deid_to_patient = load_deid_to_patient()

    builders = {
        "lvef": lambda: build_lvef(ref_to_deid, ref_to_patient, patient_splits),
        "tapse": lambda: build_tapse(ref_to_deid, ref_to_patient, patient_splits),
        "mr_severity": lambda: build_mr_severity(deid_to_patient, patient_splits),
        "as_severity": lambda: build_as_severity(deid_to_patient, patient_splits),
    }

    for task in tasks:
        arrays = builders[task]()
        save_npz(task, arrays, output_dir)
        if args.validate:
            validate_against_existing(task, arrays, EXISTING_LABELS)

    # Summary
    print("\n=== Summary ===")
    for task in tasks:
        path = output_dir / f"{task}.npz"
        npz = np.load(str(path), allow_pickle=True)
        splits = npz["splits"]
        n_train = (splits == "train").sum()
        n_val = (splits == "val").sum()
        n_test = (splits == "test").sum()
        labels = npz["labels"]
        if labels.dtype in [np.float32, np.float64]:
            stats = f"mean={labels.mean():.3f}, std={labels.std():.3f}, range=[{labels.min():.3f}, {labels.max():.3f}]"
        else:
            unique, counts = np.unique(labels, return_counts=True)
            stats = ", ".join(f"c{u}={c:,}" for u, c in zip(unique, counts))
        print(f"  {task}: {len(labels):,} studies (train={n_train:,}, val={n_val:,}, test={n_test:,}) | {stats}")


if __name__ == "__main__":
    main()
