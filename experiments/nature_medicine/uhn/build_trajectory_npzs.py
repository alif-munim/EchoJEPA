"""
Build trajectory NPZ files from cross-sectional severity labels + study dates.

Produces paired (baseline, follow-up) study entries with the same format as
the existing trajectory NPZs (trajectory_lvef.npz, trajectory_mr_severity.npz, etc.):

    study_id_1, study_id_2, patient_ids, label_1, label_2, delta, days_between, splits

Usage:
    python build_trajectory_npzs.py --task tr_severity
    python build_trajectory_npzs.py --task as_severity
    python build_trajectory_npzs.py --task tr_severity --min_days 30 --max_days 365
"""

import argparse
import os

import numpy as np
import pandas as pd


# --- Paths ---
LABELS_DIR = os.path.join(os.path.dirname(__file__), "labels")
TRAJECTORY_DIR = os.path.join(LABELS_DIR, "trajectory")
MAPPING_DIR = os.path.join(os.path.dirname(__file__), "mapping")


def load_study_dates():
    """Load study dates from uhn_uid_to_studyref.csv + ecs_master.csv."""
    # Primary: uhn_uid_to_studyref.csv (YYYYMMDD, date-only, one row per study)
    uid_path = os.path.join(MAPPING_DIR, "uhn_uid_to_studyref.csv")
    uid_ref = pd.read_csv(uid_path, usecols=["deidentified_study_id", "study_date"])
    uid_ref["date"] = pd.to_datetime(uid_ref["study_date"].astype(str), format="%Y%m%d", errors="coerce")
    uid_ref = uid_ref[["deidentified_study_id", "date"]].rename(columns={"deidentified_study_id": "study_id"})
    uid_ref = uid_ref.dropna(subset=["date"]).set_index("study_id")

    # Fallback: ecs_master.csv (MM/DD/YY HH:MM:SS, deduplicated to study-level)
    ecs_path = os.path.join(MAPPING_DIR, "ecs_master.csv")
    ecs = pd.read_csv(ecs_path, usecols=["deidentified_study_id", "STUDY_DATE"])
    ecs = ecs.drop_duplicates("deidentified_study_id")
    ecs["date"] = pd.to_datetime(ecs["STUDY_DATE"], format="mixed", errors="coerce")
    ecs = ecs[["deidentified_study_id", "date"]].rename(columns={"deidentified_study_id": "study_id"})
    ecs_only = ecs[~ecs["study_id"].isin(uid_ref.index)].dropna(subset=["date"]).set_index("study_id")

    combined = pd.concat([uid_ref, ecs_only])
    print(f"Loaded {len(combined)} study dates ({len(uid_ref)} from uid_ref, {len(ecs_only)} from ecs_master)")
    return combined


def load_cross_sectional_labels(task):
    """Load cross-sectional severity NPZ: study_ids, patient_ids, labels, splits."""
    npz_path = os.path.join(LABELS_DIR, f"{task}.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"No cross-sectional labels at {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    df = pd.DataFrame(
        {
            "study_id": data["study_ids"],
            "patient_id": data["patient_ids"],
            "label": data["labels"],
            "split": data["splits"],
        }
    )
    print(f"Loaded {len(df)} studies for {task}")
    print(f"  Label distribution: {dict(df['label'].value_counts().sort_index())}")
    print(f"  Splits: {dict(df['split'].value_counts())}")
    return df


def build_trajectory_pairs(labels_df, dates_df, min_days=30, max_days=365):
    """
    Build trajectory pairs from cross-sectional labels + dates.

    For each patient with 2+ dated studies, form all (baseline, follow-up) pairs
    where days_between is in [min_days, max_days]. Baseline is the earlier study.

    Splits are patient-level: all pairs from the same patient get the same split.
    """
    # Join dates
    dated = labels_df.merge(dates_df, left_on="study_id", right_index=True, how="inner")
    print(f"  {len(dated)}/{len(labels_df)} studies have dates ({len(dated) / len(labels_df) * 100:.1f}%)")

    # Build patient-level split mapping (majority vote from cross-sectional splits)
    patient_splits = labels_df.groupby("patient_id")["split"].agg(lambda x: x.mode().iloc[0])

    # Group by patient and form pairs
    pairs = []
    for pid, group in dated.groupby("patient_id"):
        if len(group) < 2:
            continue

        group = group.sort_values("date")
        studies = group[["study_id", "label", "date"]].values

        for i in range(len(studies)):
            for j in range(i + 1, len(studies)):
                sid_1, lab_1, date_1 = studies[i]
                sid_2, lab_2, date_2 = studies[j]
                days = (date_2 - date_1).days

                if min_days <= days <= max_days:
                    pairs.append(
                        {
                            "study_id_1": sid_1,
                            "study_id_2": sid_2,
                            "patient_id": pid,
                            "label_1": int(lab_1),
                            "label_2": int(lab_2),
                            "delta": int(lab_2) - int(lab_1),
                            "days_between": days,
                            "split": patient_splits.get(pid, "train"),
                        }
                    )

    pairs_df = pd.DataFrame(pairs)
    print(f"  Built {len(pairs_df)} pairs from {pairs_df['patient_id'].nunique()} patients")
    return pairs_df


def save_trajectory_npz(pairs_df, task, output_dir=None):
    """Save trajectory pairs in the standard NPZ format."""
    if output_dir is None:
        output_dir = TRAJECTORY_DIR
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"trajectory_{task}.npz")

    np.savez(
        output_path,
        study_id_1=np.array(pairs_df["study_id_1"].values, dtype=object),
        study_id_2=np.array(pairs_df["study_id_2"].values, dtype=object),
        patient_ids=np.array(pairs_df["patient_id"].values, dtype=object),
        label_1=np.array(pairs_df["label_1"].values, dtype=np.int32),
        label_2=np.array(pairs_df["label_2"].values, dtype=np.int32),
        delta=np.array(pairs_df["delta"].values, dtype=np.int32),
        days_between=np.array(pairs_df["days_between"].values, dtype=np.int32),
        splits=np.array(pairs_df["split"].values, dtype=object),
    )
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"  Saved {output_path} ({size_mb:.1f} MB)")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Build trajectory NPZ from cross-sectional severity labels")
    parser.add_argument("--task", required=True, help="Task name (e.g., tr_severity, as_severity)")
    parser.add_argument("--min_days", type=int, default=30, help="Min days between studies (default: 30)")
    parser.add_argument("--max_days", type=int, default=365, help="Max days between studies (default: 365)")
    parser.add_argument("--output_dir", default=None, help="Override output directory")
    args = parser.parse_args()

    print(f"=== Building trajectory NPZ for {args.task} ===")
    print(f"  Time window: {args.min_days}-{args.max_days} days")

    dates_df = load_study_dates()
    labels_df = load_cross_sectional_labels(args.task)
    pairs_df = build_trajectory_pairs(labels_df, dates_df, args.min_days, args.max_days)

    if len(pairs_df) == 0:
        print("ERROR: No pairs found!")
        return

    # Summary
    splits = pairs_df["split"].value_counts()
    print(f"\n  Split distribution:")
    for sp in ["train", "val", "test"]:
        if sp in splits.index:
            print(f"    {sp}: {splits[sp]}")

    print(f"\n  Delta distribution:")
    print(f"    mean={pairs_df['delta'].mean():.3f}, std={pairs_df['delta'].std():.3f}")
    print(f"    min={pairs_df['delta'].min()}, max={pairs_df['delta'].max()}")

    print(f"\n  Days between:")
    print(f"    mean={pairs_df['days_between'].mean():.0f}, median={pairs_df['days_between'].median():.0f}")

    save_trajectory_npz(pairs_df, args.task, args.output_dir)
    print("\nDone!")


if __name__ == "__main__":
    main()
