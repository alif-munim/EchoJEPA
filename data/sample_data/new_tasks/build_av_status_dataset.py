"""
Build 4-class AV valve status dataset: Mechanical / Surgical Bioprosthetic / TAVR / Native.

Combines Syngo (2015-2019) and HeartLab (2005-2014) labels, filters to PLAX/A4C/A3C
clips with S3 video data, and splits by patient (70/15/15).

Classes:
  0 = MECHANICAL (prosthetic mechanical valve)
  1 = SURGICAL BIOPROSTHETIC (surgical tissue/bioprosthetic valve)
  2 = TAVR (transcatheter aortic valve replacement — Edwards Sapien, CoreValve, Evolut)
  3 = NATIVE (structurally normal AV, no surgery)

Output: probe_csvs/av_status/{train,val,test}_vf.csv + viewfilter_meta.json
"""

import csv
import json
import sqlite3
import numpy as np
from collections import defaultdict
from pathlib import Path

DB_PATH = '/mnt/custom-file-systems/efs/fs-0049217cdf69186d7_fsap-0fa7145b64eaa046b/vjepa2/uhn_echo/nature_medicine/data_exploration/echo.db'
AWS_SYNGO = '/mnt/custom-file-systems/efs/fs-0049217cdf69186d7_fsap-0fa7145b64eaa046b/vjepa2/data/aws/aws_syngo.csv'
AWS_HEARTLAB = '/mnt/custom-file-systems/efs/fs-0049217cdf69186d7_fsap-0fa7145b64eaa046b/vjepa2/data/aws/aws_heartlab_0806.csv'
VIEW_PREDS = '/home/sagemaker-user/user-default-efs/vjepa2/classifier/output/view_inference_18m/master_predictions.csv'
COLOR_PREDS = '/home/sagemaker-user/user-default-efs/vjepa2/classifier/output/color_inference_18m/master_predictions.csv'
OUT_DIR = Path('/mnt/custom-file-systems/efs/fs-0049217cdf69186d7_fsap-0fa7145b64eaa046b/vjepa2/experiments/nature_medicine/uhn/probe_csvs/av_status')

ALLOWED_VIEWS = ["PLAX", "A4C", "A3C"]
BMODE_ONLY = False  # color allowed (prosthetic jets visible)
CLASS_NAMES = {0: "mechanical", 1: "surgical_bioprosthetic", 2: "tavr", 3: "native"}

# Syngo manufacturer → class mapping
SYNGO_MFGR_MECHANICAL = {'St._Jude', 'Unkown_mech', 'Carbomedics', 'Bjork-Shiley',
                          'Starr_Edwards', 'On-X mechanical', 'On-X'}
SYNGO_MFGR_BIO = {'Unkown_bio', 'Hancock_II', 'Magna_Perimount', 'Carpentier',
                   'Freestyle', 'Magna Ease', 'Mosaic', 'Toronto_SPV',
                   'Edwards Magna', 'Trifecta', 'Magna', 'Perigon'}
SYNGO_MFGR_TAVR = {'Sapien', 'Corevalve', 'Sapien 3', 'Evolut R', 'TAVI',
                    'Sapien 3 Ultra', 'Portico', 'Perceval'}

# Syngo mechanical type values
SYNGO_MECH_TYPES = {'bileaflet', 'mechanical', 'tilting_disk', 'SJM', 'On-X',
                    'mechanical AVR', 'St. Jude', 'Bjork-Shiley', 'Bjork Shiley',
                    'Mechanical prosthesis is in the aortic valve position.'}

# HeartLab findings
HL_MECHANICAL_FINDINGS = {'278', '275', '276', '277'}  # Group 75
HL_BIO_FINDINGS = {'283', '279', '1426', '281', '1522', '280', '282', '1523', '100270', '1427'}  # Group 76
HL_TAVR_FINDINGS = {'100762', '100763', '100764', '100766'}  # Group 100179 + post-op TAVR
HL_NATIVE_FINDINGS = {'100439', '243', '100460', '242', '310', '100445'}  # Tricuspid/trileaflet/normal (grps 68, 83)


def get_syngo_labels(cur):
    """Get study_ref -> class from Syngo observations."""
    labels = {}

    # AoV_Prosthetic_mfgr-ASE_obs (manufacturer — primary source)
    cur.execute("SELECT StudyRef, Value FROM syngo_observations WHERE Name='AoV_Prosthetic_mfgr-ASE_obs'")
    for ref, val in cur.fetchall():
        if val in SYNGO_MFGR_MECHANICAL:
            labels[ref] = 0
        elif val in SYNGO_MFGR_BIO:
            labels[ref] = 1
        elif val in SYNGO_MFGR_TAVR:
            labels[ref] = 2

    # AoV_Mechanical_type-ASE_obs (backup for mechanical)
    cur.execute("SELECT StudyRef, Value FROM syngo_observations WHERE Name='AoV_Mechanical_type-ASE_obs'")
    for ref, val in cur.fetchall():
        if ref not in labels and val in SYNGO_MECH_TYPES:
            labels[ref] = 0

    # AoV_Bioprosthetictype-ASE_obs (backup for bio, check for TAVR keywords)
    cur.execute("SELECT StudyRef, Value FROM syngo_observations WHERE Name='AoV_Bioprosthetictype-ASE_obs'")
    for ref, val in cur.fetchall():
        if ref not in labels:
            val_lower = val.lower()
            if 'tavi' in val_lower or 'tavr' in val_lower or 'sapien' in val_lower or 'corevalve' in val_lower or 'evolut' in val_lower:
                labels[ref] = 2
            else:
                labels[ref] = 1

    # Native: any AV structure observation (tricuspid, bicuspid, normal) without prosthetic labels
    native_refs = set()
    cur.execute("SELECT DISTINCT StudyRef FROM syngo_observations WHERE Name='AoV_structure_uhn_obs'")
    native_refs.update(r[0] for r in cur.fetchall())
    cur.execute("SELECT DISTINCT StudyRef FROM syngo_observations WHERE Name='AoV_Normal_obs'")
    native_refs.update(r[0] for r in cur.fetchall())
    for ref in native_refs:
        if ref not in labels:
            labels[ref] = 3

    return labels


def get_heartlab_labels(cur):
    """Get study_instance_uid -> class from HeartLab findings."""
    labels = {}

    def get_studies_for_findings(finding_ids):
        studies = set()
        for fid in finding_ids:
            cur.execute("""SELECT DISTINCT hs.STUDY_INSTANCE_UID
            FROM heartlab_finding_intersects fi
            JOIN heartlab_reports hr ON hr.ID = fi.REP_ID
            JOIN heartlab_series hser ON hser.ID = hr.SERI_ID
            JOIN heartlab_studies hs ON hs.ID = hser.STUD_ID
            WHERE fi.FIN_ID = ?""", (fid,))
            studies.update(r[0] for r in cur.fetchall())
        return studies

    # Mechanical
    mech_studies = get_studies_for_findings(HL_MECHANICAL_FINDINGS)
    for uid in mech_studies:
        labels[uid] = 0

    # Surgical bioprosthetic (exclude TAVR)
    bio_studies = get_studies_for_findings(HL_BIO_FINDINGS)
    for uid in bio_studies:
        if uid not in labels:
            labels[uid] = 1

    # TAVR
    tavr_studies = get_studies_for_findings(HL_TAVR_FINDINGS)
    for uid in tavr_studies:
        if uid not in labels:
            labels[uid] = 2

    # Native (tricuspid/trileaflet/normal — any native AV structure)
    native_studies = get_studies_for_findings(HL_NATIVE_FINDINGS)
    for uid in native_studies:
        if uid not in labels:
            labels[uid] = 3

    return labels


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

    # 1. Get Syngo labels
    print("1. Loading Syngo labels...")
    syngo_labels = get_syngo_labels(cur)
    for cls in range(4):
        n = sum(1 for v in syngo_labels.values() if v == cls)
        print(f"   {CLASS_NAMES[cls]}: {n:,}")

    # 2. Get HeartLab labels
    print("\n2. Loading HeartLab labels...")
    heartlab_labels = get_heartlab_labels(cur)
    for cls in range(4):
        n = sum(1 for v in heartlab_labels.values() if v == cls)
        print(f"   {CLASS_NAMES[cls]}: {n:,}")

    # 3. Get patient IDs
    print("\n3. Loading patient IDs...")
    all_refs = list(syngo_labels.keys())
    patient_map = {}
    batch_size = 500
    for i in range(0, len(all_refs), batch_size):
        batch = all_refs[i:i+batch_size]
        placeholders = ','.join(['?'] * len(batch))
        cur.execute(f"SELECT STUDY_REF, PATIENT_ID FROM syngo_study_details WHERE STUDY_REF IN ({placeholders})", batch)
        for row in cur.fetchall():
            patient_map[row[0]] = row[1]
    print(f"   Syngo: {len(patient_map):,}")

    conn.close()

    # 4. Load S3 mappings
    print("\n4. Loading S3 mappings...")
    ref_to_uids = defaultdict(list)
    uid_to_ref = {}
    with open(AWS_SYNGO) as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid_to_ref[row['DeidentifiedStudyID']] = row['STUDY_REF']
            ref_to_uids[row['STUDY_REF']].append(row['DeidentifiedStudyID'])
    print(f"   Syngo: {len(ref_to_uids):,}")

    hl_orig_to_s3 = {}
    hl_s3_to_patient = {}
    with open(AWS_HEARTLAB) as f:
        reader = csv.DictReader(f)
        for row in reader:
            orig_uid = row.get('OriginalStudyID', '')
            s3_uid = row.get('DeidentifiedStudyID_hl', '')
            patient_id = row.get('PATIENT_ID', '')
            if orig_uid and s3_uid:
                hl_orig_to_s3[orig_uid] = s3_uid
                hl_s3_to_patient[s3_uid] = f"HL_{patient_id}"
    print(f"   HeartLab: {len(hl_orig_to_s3):,}")

    # 5. Load view predictions
    print("\n5. Loading view predictions (PLAX + A4C + A3C)...")
    view_clips = set()
    with open(VIEW_PREDS) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['prediction'] in ALLOWED_VIEWS:
                view_clips.add(row['s3_uri'])
    print(f"   PLAX+A4C+A3C clips: {len(view_clips):,}")

    # 6. Color filter (not applied — color allowed)
    valid_clips = view_clips
    print("\n6. Color filter: not applied (color allowed)")

    # 7. Build unified label map
    print("\n7. Building unified label map...")
    uid_labels = {}
    for ref, cls in syngo_labels.items():
        for uid in ref_to_uids.get(ref, []):
            uid_labels[uid] = cls

    hl_mapped = 0
    for hl_uid, cls in heartlab_labels.items():
        s3_uid = hl_orig_to_s3.get(hl_uid)
        if s3_uid and s3_uid not in uid_labels:
            uid_labels[s3_uid] = cls
            hl_mapped += 1
    print(f"   HeartLab mapped: {hl_mapped:,}")
    for cls in range(4):
        n = sum(1 for v in uid_labels.values() if v == cls)
        print(f"   {CLASS_NAMES[cls]}: {n:,} S3 study_uids")

    # 8. Map clips to studies
    print("\n8. Mapping clips to labeled studies...")
    study_clips = defaultdict(list)
    for clip in valid_clips:
        uid = extract_study_uid(clip)
        if uid and uid in uid_labels:
            study_clips[uid].append(clip)

    for cls in range(4):
        studies = [uid for uid in study_clips if uid_labels[uid] == cls]
        clips = sum(len(study_clips[uid]) for uid in studies)
        print(f"   {CLASS_NAMES[cls]}: {len(studies):,} studies, {clips:,} clips")

    # 9. Downsample native: 50/50 healthy/diseased mix, capped to largest prosthetic class
    np.random.seed(42)
    class_uids = {cls: [uid for uid in study_clips if uid_labels[uid] == cls] for cls in range(4)}
    largest_other = max(len(class_uids[cls]) for cls in range(3))

    # Split native into healthy (normal/tricuspid structure) vs diseased (bicuspid, stenotic, etc.)
    # Use syngo AoV_structure_uhn_obs to classify: tricuspid/normal = healthy, everything else = diseased
    native_healthy = []
    native_diseased = []
    for uid in class_uids[3]:
        ref = uid_to_ref.get(uid)
        if ref:
            # Check if this study has a "diseased" native valve label
            is_diseased = False
            # We'll mark as diseased if NOT in the simple tricuspid/normal set
            # For simplicity, default to healthy (most native AV are tricuspid)
            # and mark diseased if they have AS severity, bicuspid, or sclerosis labels
            native_healthy.append(uid)
        else:
            native_healthy.append(uid)

    # Re-query diseased status from DB (reopening since we closed earlier)
    conn2 = sqlite3.connect(DB_PATH)
    cur2 = conn2.cursor()
    # Get all study_refs with bicuspid or AS-related observations
    diseased_refs = set()
    cur2.execute("""SELECT DISTINCT StudyRef FROM syngo_observations
    WHERE Name='AoV_structure_uhn_obs' AND (Value LIKE '%bicuspid%' OR Value LIKE '%unicuspid%'
    OR Value LIKE '%calcif%' OR Value LIKE '%restricted%' OR Value LIKE '%rheumatic%')""")
    diseased_refs.update(r[0] for r in cur2.fetchall())
    cur2.execute("""SELECT DISTINCT StudyRef FROM syngo_observations
    WHERE Name='AoV_structure_sD_obs' AND Value != 'normal' AND Value != 'tricuspid'""")
    diseased_refs.update(r[0] for r in cur2.fetchall())
    conn2.close()

    native_healthy = []
    native_diseased = []
    for uid in class_uids[3]:
        ref = uid_to_ref.get(uid)
        if ref and ref in diseased_refs:
            native_diseased.append(uid)
        else:
            native_healthy.append(uid)

    print(f"\n   Native healthy: {len(native_healthy):,}, diseased: {len(native_diseased):,}")

    # 50/50 mix capped to largest prosthetic class
    native_cap_per_group = largest_other // 2
    native_cap_per_group = min(native_cap_per_group, len(native_healthy), len(native_diseased))
    sampled_healthy = list(np.random.choice(native_healthy, native_cap_per_group, replace=False))
    sampled_diseased = list(np.random.choice(native_diseased, native_cap_per_group, replace=False))
    native_sampled = sampled_healthy + sampled_diseased
    print(f"   Native sampled: {len(native_sampled):,} ({native_cap_per_group:,} healthy + {native_cap_per_group:,} diseased)")

    all_uids = class_uids[0] + class_uids[1] + class_uids[2] + native_sampled
    for cls in range(3):
        print(f"   {CLASS_NAMES[cls]}: {len(class_uids[cls]):,} studies (full)")
    print(f"   {CLASS_NAMES[3]}: {len(native_sampled):,} studies (50/50 healthy/diseased)")

    # 10. Patient-level split
    print("\n9. Patient-level split (70/15/15)...")
    uid_to_patient = {}
    for uid in all_uids:
        ref = uid_to_ref.get(uid)
        if ref and ref in patient_map:
            uid_to_patient[uid] = patient_map[ref]
        elif uid in hl_s3_to_patient:
            uid_to_patient[uid] = hl_s3_to_patient[uid]
        else:
            uid_to_patient[uid] = f"unknown_{uid[:20]}"

    patient_studies = defaultdict(list)
    for uid in all_uids:
        patient_studies[uid_to_patient[uid]].append(uid)

    patients = sorted(patient_studies.keys())
    np.random.shuffle(patients)

    n = len(patients)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)

    train_patients = set(patients[:n_train])
    val_patients = set(patients[n_train:n_train+n_val])
    test_patients = set(patients[n_train+n_val:])

    splits = {'train': [], 'val': [], 'test': []}
    split_study_counts = {'train': defaultdict(int), 'val': defaultdict(int), 'test': defaultdict(int)}

    for split_name, pat_set in [('train', train_patients), ('val', val_patients), ('test', test_patients)]:
        for pid in pat_set:
            for uid in patient_studies[pid]:
                cls = uid_labels[uid]
                split_study_counts[split_name][cls] += 1
                for clip in study_clips[uid]:
                    splits[split_name].append((clip, cls))

    for split_name in ['train', 'val', 'test']:
        total_clips = len(splits[split_name])
        print(f"   {split_name}: {total_clips:,} clips")
        for cls in range(4):
            print(f"     {CLASS_NAMES[cls]}: {split_study_counts[split_name][cls]:,} studies")

    # 11. Write CSVs
    print("\n10. Writing CSVs...")
    for split_name, data in splits.items():
        csv_path = OUT_DIR / f"{split_name}_vf.csv"
        with open(csv_path, 'w') as f:
            for clip, label in sorted(data):
                f.write(f"{clip} {label}\n")
        print(f"   {csv_path.name}: {len(data):,} lines")

    # 12. Metadata
    meta = {
        "task": "av_status",
        "classes": CLASS_NAMES,
        "num_classes": 4,
        "allowed_views": ALLOWED_VIEWS,
        "bmode_only": BMODE_ONLY,
        "native_mix": "50/50 healthy (tricuspid/normal) vs diseased (bicuspid/calcified/stenotic)",
        "label_sources": {
            "syngo": {
                "mechanical": "AoV_Prosthetic_mfgr-ASE_obs IN (St._Jude, Unkown_mech, Carbomedics, ...) OR AoV_Mechanical_type-ASE_obs",
                "surgical_bioprosthetic": "AoV_Prosthetic_mfgr-ASE_obs IN (Hancock_II, Magna_Perimount, ...) OR AoV_Bioprosthetictype-ASE_obs (excl TAVR)",
                "tavr": "AoV_Prosthetic_mfgr-ASE_obs IN (Sapien, Corevalve, Evolut R, ...) OR AoV_Bioprosthetictype with TAVR keywords",
                "native": "AoV_structure_uhn_obs (any value: tricuspid, bicuspid, etc.) OR AoV_Normal_obs, MINUS all prosthetic"
            },
            "heartlab": {
                "mechanical": "Group 75 findings (278, 275, 276, 277)",
                "surgical_bioprosthetic": "Group 76 findings (283, 279, 1426, 281, 1522, 280, 282, 1523, 100270, 1427)",
                "tavr": "Group 100179 findings (100762, 100763, 100764) + finding 100766 (post-op TAVR)",
                "native": "Findings 100439/243/100460/242/310/100445 (tricuspid/trileaflet/normal, grps 68+83), MINUS all prosthetic"
            }
        },
        "split_studies": {k: dict(v) for k, v in split_study_counts.items()},
    }
    with open(OUT_DIR / "viewfilter_meta.json", 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"\n   Saved viewfilter_meta.json")


if __name__ == "__main__":
    main()
