"""
Build stratified data efficiency subsets for VSD and RV basal diameter.

Fractions: 50%, 25%, 12.5%, 6.25%, 3.125% (log2 halving series)
Each smaller subset is nested within the next larger one.
Stratification:
  - VSD (classification): preserves class distribution (positive/negative ratio)
  - RV basal diam (regression): preserves label distribution via quantile binning

Output: train_vf_{50pct,25pct,12pct,6pct,3pct}.csv for each task
"""

import numpy as np
from pathlib import Path
from collections import defaultdict

PROBE_CSV_DIR = Path('/mnt/custom-file-systems/efs/fs-0049217cdf69186d7_fsap-0fa7145b64eaa046b/vjepa2/experiments/nature_medicine/uhn/probe_csvs')

FRACTIONS = [0.5, 0.25, 0.125, 0.0625, 0.03125]
FRACTION_NAMES = ['50pct', '25pct', '12pct', '6pct', '3pct']


def extract_study_uid(path):
    parts = path.split('/')
    for p in parts:
        if p.startswith('1.2.276') and '.3.1.2.' in p:
            return p
    return None


def load_train_csv(task_dir):
    """Load training CSV, group clips by study, return study-level data."""
    studies = defaultdict(list)  # study_uid -> [(clip_path, label)]
    with open(task_dir / 'train_vf.csv') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            path, label = parts[0], parts[-1]
            uid = extract_study_uid(path)
            if uid:
                studies[uid].append((path, label))
    return studies


def build_stratified_subsets_classification(task_dir, task_name):
    """Build nested stratified subsets for binary classification."""
    print(f"\n{'='*60}")
    print(f"Building data efficiency subsets for {task_name} (classification)")
    print(f"{'='*60}")

    studies = load_train_csv(task_dir)
    # Group studies by class
    class_studies = defaultdict(list)
    for uid, clips in studies.items():
        label = clips[0][1]  # all clips in a study have the same label
        class_studies[label].append(uid)

    print(f"  Total studies: {len(studies):,}")
    for cls, uids in sorted(class_studies.items()):
        print(f"    Class {cls}: {len(uids):,} studies ({100*len(uids)/len(studies):.1f}%)")

    np.random.seed(42)
    # Shuffle each class independently
    for cls in class_studies:
        np.random.shuffle(class_studies[cls])

    for frac, name in zip(FRACTIONS, FRACTION_NAMES):
        # Take frac of each class (preserves ratio)
        selected_uids = set()
        for cls, uids in class_studies.items():
            n_select = max(1, int(len(uids) * frac))
            selected_uids.update(uids[:n_select])

        # Write CSV
        out_path = task_dir / f'train_vf_{name}.csv'
        n_clips = 0
        with open(out_path, 'w') as f:
            for uid in selected_uids:
                for clip_path, label in studies[uid]:
                    f.write(f"{clip_path} {label}\n")
                    n_clips += 1

        print(f"  {name}: {len(selected_uids):,} studies, {n_clips:,} clips")


def build_stratified_subsets_regression(task_dir, task_name, n_bins=10):
    """Build nested stratified subsets for regression (quantile binning)."""
    print(f"\n{'='*60}")
    print(f"Building data efficiency subsets for {task_name} (regression)")
    print(f"{'='*60}")

    studies = load_train_csv(task_dir)
    # Get study-level labels
    study_labels = {}
    for uid, clips in studies.items():
        study_labels[uid] = float(clips[0][1])

    uids = list(study_labels.keys())
    labels = np.array([study_labels[uid] for uid in uids])

    print(f"  Total studies: {len(uids):,}")
    print(f"  Label range: [{labels.min():.3f}, {labels.max():.3f}], mean={labels.mean():.3f}")

    # Quantile binning for stratification
    bin_edges = np.percentile(labels, np.linspace(0, 100, n_bins + 1))
    bin_indices = np.digitize(labels, bin_edges[1:-1])  # 0 to n_bins-1

    # Group by bin
    bin_studies = defaultdict(list)
    for i, uid in enumerate(uids):
        bin_studies[bin_indices[i]].append(uid)

    np.random.seed(42)
    for bin_idx in bin_studies:
        np.random.shuffle(bin_studies[bin_idx])

    for frac, name in zip(FRACTIONS, FRACTION_NAMES):
        selected_uids = set()
        for bin_idx, bin_uids in bin_studies.items():
            n_select = max(1, int(len(bin_uids) * frac))
            selected_uids.update(bin_uids[:n_select])

        # Write CSV
        out_path = task_dir / f'train_vf_{name}.csv'
        n_clips = 0
        with open(out_path, 'w') as f:
            for uid in selected_uids:
                for clip_path, label in studies[uid]:
                    f.write(f"{clip_path} {label}\n")
                    n_clips += 1

        print(f"  {name}: {len(selected_uids):,} studies, {n_clips:,} clips")


def main():
    # VSD (binary classification)
    vsd_dir = PROBE_CSV_DIR / 'disease_vsd'
    build_stratified_subsets_classification(vsd_dir, "VSD Detection")

    # RV Basal Diam (regression)
    rv_dir = PROBE_CSV_DIR / 'rv_basal_diam'
    build_stratified_subsets_regression(rv_dir, "RV Basal Diameter")

    print("\n\nDone. Subsets are nested (each smaller ⊂ next larger).")


if __name__ == "__main__":
    main()
