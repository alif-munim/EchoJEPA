"""
Build MV E prime medial dataset.

Regression task: predict mitral annular e' velocity (medial/septal) from B-mode video.
A4C only, B-mode only.

Output: probe_csvs/mv_e_prime_medial/{train,val,test}_vf.csv + viewfilter_meta.json + zscore_params.json
"""

import csv
import json
import sqlite3
import numpy as np
from collections import defaultdict
from pathlib import Path

DB_PATH = '/mnt/custom-file-systems/efs/fs-0049217cdf69186d7_fsap-0fa7145b64eaa046b/vjepa2/uhn_echo/nature_medicine/data_exploration/echo.db'
AWS_SYNGO = '/mnt/custom-file-systems/efs/fs-0049217cdf69186d7_fsap-0fa7145b64eaa046b/vjepa2/data/aws/aws_syngo.csv'
VIEW_PREDS = '/home/sagemaker-user/user-default-efs/vjepa2/classifier/output/view_inference_18m/master_predictions.csv'
COLOR_PREDS = '/home/sagemaker-user/user-default-efs/vjepa2/classifier/output/color_inference_18m/master_predictions.csv'
OUT_DIR = Path('/mnt/custom-file-systems/efs/fs-0049217cdf69186d7_fsap-0fa7145b64eaa046b/vjepa2/experiments/nature_medicine/uhn/probe_csvs/mv_e_prime_medial')

ALLOWED_VIEWS = ["A4C"]
BMODE_ONLY = True
VALID_RANGE = (0.02, 0.25)  # cm/s — typical range 0.03-0.20


def extract_study_uid(s3_path):
    parts = s3_path.split('/')
    for p in parts:
        if p.startswith('1.2.276') and '.3.1.2.' in p:
            return p
    return None


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # 1. Get study-level e' medial labels
    print("1. Loading MV E prime medial measurements...")
    cur.execute("""SELECT StudyRef, AVG(CAST(Value AS REAL)) as mean_val
    FROM syngo_measures
    WHERE MeasurementName = 'MV E prime medial' AND CAST(Value AS REAL) >= ? AND CAST(Value AS REAL) <= ?
    GROUP BY StudyRef""", VALID_RANGE)
    study_ref_labels = {r[0]: round(r[1], 4) for r in cur.fetchall()}
    print(f"   Studies with valid e' medial: {len(study_ref_labels):,}")

    # 2. Get patient IDs for splitting
    print("2. Loading patient IDs...")
    refs = list(study_ref_labels.keys())
    patient_map = {}
    batch_size = 500
    for i in range(0, len(refs), batch_size):
        batch = refs[i:i+batch_size]
        placeholders = ','.join(['?'] * len(batch))
        cur.execute(f"""SELECT STUDY_REF, PATIENT_ID FROM syngo_study_details
        WHERE STUDY_REF IN ({placeholders})""", batch)
        for row in cur.fetchall():
            patient_map[row[0]] = row[1]
    print(f"   Mapped {len(patient_map):,} studies to patients")

    conn.close()

    # 3. Load aws_syngo mapping
    print("3. Loading aws_syngo mapping...")
    ref_to_uids = defaultdict(list)
    uid_to_ref = {}
    with open(AWS_SYNGO) as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid_to_ref[row['DeidentifiedStudyID']] = row['STUDY_REF']
            ref_to_uids[row['STUDY_REF']].append(row['DeidentifiedStudyID'])
    print(f"   {len(ref_to_uids):,} study_refs")

    # 4. Load view predictions (A4C only)
    print("4. Loading view predictions (A4C only)...")
    view_clips = set()
    with open(VIEW_PREDS) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['prediction'] in ALLOWED_VIEWS:
                view_clips.add(row['s3_uri'])
    print(f"   A4C clips: {len(view_clips):,}")

    # 5. Load color predictions (B-mode only)
    print("5. Loading color predictions (B-mode only)...")
    bmode_clips = set()
    with open(COLOR_PREDS) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['is_color'] == 'No':
                bmode_clips.add(row['s3_uri'])
    print(f"   B-mode clips: {len(bmode_clips):,}")

    # 6. Find valid clips
    print("6. Building clip list...")
    valid_clips = view_clips & bmode_clips
    print(f"   A4C + B-mode clips: {len(valid_clips):,}")

    study_clips = defaultdict(list)
    for clip in valid_clips:
        uid = extract_study_uid(clip)
        if uid and uid in uid_to_ref:
            ref = uid_to_ref[uid]
            if ref in study_ref_labels:
                study_clips[ref].append(clip)

    print(f"   Studies with A4C B-mode clips AND e' medial label: {len(study_clips):,}")
    clip_counts = [len(v) for v in study_clips.values()]
    print(f"   Clips per study: median={np.median(clip_counts):.0f}, mean={np.mean(clip_counts):.1f}, total={sum(clip_counts):,}")

    # 7. Patient-level split (70/15/15)
    print("7. Splitting by patient (70/15/15)...")
    patient_studies = defaultdict(list)
    for ref in study_clips:
        pid = patient_map.get(ref, f"unknown_{ref}")
        patient_studies[pid].append(ref)

    patients = sorted(patient_studies.keys())
    np.random.seed(42)
    np.random.shuffle(patients)

    n = len(patients)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)

    train_patients = set(patients[:n_train])
    val_patients = set(patients[n_train:n_train+n_val])
    test_patients = set(patients[n_train+n_val:])

    splits = {'train': [], 'val': [], 'test': []}
    train_studies = set()
    val_studies = set()
    test_studies = set()
    for pid in train_patients:
        train_studies.update(patient_studies[pid])
    for pid in val_patients:
        val_studies.update(patient_studies[pid])
    for pid in test_patients:
        test_studies.update(patient_studies[pid])

    for ref in train_studies:
        for clip in study_clips[ref]:
            splits['train'].append((clip, study_ref_labels[ref]))
    for ref in val_studies:
        for clip in study_clips[ref]:
            splits['val'].append((clip, study_ref_labels[ref]))
    for ref in test_studies:
        for clip in study_clips[ref]:
            splits['test'].append((clip, study_ref_labels[ref]))

    print(f"   Train: {len(train_patients):,} patients, {len(train_studies):,} studies, {len(splits['train']):,} clips")
    print(f"   Val:   {len(val_patients):,} patients, {len(val_studies):,} studies, {len(splits['val']):,} clips")
    print(f"   Test:  {len(test_patients):,} patients, {len(test_studies):,} studies, {len(splits['test']):,} clips")

    # 8. Write CSVs
    print("8. Writing CSVs...")
    for split_name, data in splits.items():
        csv_path = OUT_DIR / f"{split_name}_vf.csv"
        with open(csv_path, 'w') as f:
            for clip, label in sorted(data):
                f.write(f"{clip} {label:.4f}\n")
        print(f"   {csv_path.name}: {len(data):,} lines")

    # 9. Z-score params from training set
    train_study_labels = [study_ref_labels[ref] for ref in train_studies]
    mean_val = np.mean(train_study_labels)
    std_val = np.std(train_study_labels)
    zscore_params = {"target_mean": float(mean_val), "target_std": float(std_val)}
    with open(OUT_DIR / "zscore_params.json", 'w') as f:
        json.dump(zscore_params, f)
    print(f"   Z-score params: mean={mean_val:.6f}, std={std_val:.6f}")

    # 10. Metadata
    meta = {
        "task": "mv_e_prime_medial",
        "measurement_name": "MV E prime medial",
        "allowed_views": ALLOWED_VIEWS,
        "bmode_only": BMODE_ONLY,
        "unit": "cm/s",
        "valid_range": list(VALID_RANGE),
        "n_patients": {"train": len(train_patients), "val": len(val_patients), "test": len(test_patients)},
        "n_studies": {"train": len(train_studies), "val": len(val_studies), "test": len(test_studies)},
        "n_clips": {"train": len(splits['train']), "val": len(splits['val']), "test": len(splits['test'])}
    }
    with open(OUT_DIR / "viewfilter_meta.json", 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"\n   Saved viewfilter_meta.json")

    # Summary stats
    all_labels = train_study_labels + [study_ref_labels[ref] for ref in val_studies] + [study_ref_labels[ref] for ref in test_studies]
    print(f"\n   Overall: mean={np.mean(all_labels):.4f}, std={np.std(all_labels):.4f}, median={np.median(all_labels):.4f}")
    print(f"   Range: [{np.min(all_labels):.4f}, {np.max(all_labels):.4f}]")
    print(f"   Percentiles: 5th={np.percentile(all_labels,5):.4f}, 25th={np.percentile(all_labels,25):.4f}, 50th={np.percentile(all_labels,50):.4f}, 75th={np.percentile(all_labels,75):.4f}, 95th={np.percentile(all_labels,95):.4f}")


if __name__ == "__main__":
    main()
