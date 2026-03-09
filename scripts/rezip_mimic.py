#!/usr/bin/env python
"""Regenerate per-model MIMIC zip files with current (verified) embeddings.

Each zip is self-contained with:
  - Master embedding NPZ (with paths column)
  - clip_index.npz, patient_split.json (shared)
  - labels/*.npz (23 task label files)
  - {model}_study_level/*.npz (23 study-level files)
  - {model}_splits/{task}/train|val|test.npz (23×3 split files)
  - data/csv/nature_medicine/mimic/*.csv (23 source CSVs)
"""

import os
import zipfile

BASE = "/mnt/custom-file-systems/efs/fs-0049217cdf69186d7_fsap-0fa7145b64eaa046b/vjepa2"
MIMIC = f"{BASE}/experiments/nature_medicine/mimic"
CSV_DIR = f"{BASE}/data/csv/nature_medicine/mimic"

MODELS = {
    "echojepa_g": "echojepa_g_mimic_embeddings.npz",
    "echojepa_l": "echojepa_l_mimic_embeddings.npz",
    "echojepa_l_kinetics": "echojepa_l_kinetics_mimic_embeddings.npz",
    "echomae": "echomae_mimic_embeddings.npz",
    "panecho": "panecho_mimic_embeddings.npz",
    "echoprime": "echoprime_mimic_embeddings.npz",
    "echofm": "echofm_mimic_embeddings.npz",
}


def add_file(zf, local_path, archive_path):
    """Add a file to the zip, printing progress."""
    size_mb = os.path.getsize(local_path) / 1e6
    print(f"    {archive_path} ({size_mb:.1f} MB)")
    zf.write(local_path, archive_path)


def create_model_zip(model_name, master_npz_name):
    zip_path = os.path.join(MIMIC, f"{model_name}_mimic_all.zip")
    zip_tmp = zip_path + ".tmp"
    prefix = "experiments/nature_medicine/mimic"

    print(f"\n{'='*60}")
    print(f"  {model_name} -> {os.path.basename(zip_path)}")
    print(f"{'='*60}")

    with zipfile.ZipFile(zip_tmp, "w", compression=zipfile.ZIP_STORED) as zf:
        # Master embedding NPZ (with paths)
        add_file(zf, os.path.join(MIMIC, master_npz_name), f"{prefix}/{master_npz_name}")

        # Shared files
        add_file(zf, os.path.join(MIMIC, "clip_index.npz"), f"{prefix}/clip_index.npz")
        add_file(zf, os.path.join(MIMIC, "patient_split.json"), f"{prefix}/patient_split.json")

        # Labels (23 files)
        labels_dir = os.path.join(MIMIC, "labels")
        for f in sorted(os.listdir(labels_dir)):
            if f.endswith(".npz"):
                add_file(zf, os.path.join(labels_dir, f), f"{prefix}/labels/{f}")

        # Study-level (23 files)
        study_dir = os.path.join(MIMIC, f"{model_name}_study_level")
        if os.path.exists(study_dir):
            for f in sorted(os.listdir(study_dir)):
                if f.endswith(".npz"):
                    add_file(zf, os.path.join(study_dir, f), f"{prefix}/{model_name}_study_level/{f}")

        # Splits (23 tasks × 3 splits)
        splits_dir = os.path.join(MIMIC, f"{model_name}_splits")
        if os.path.exists(splits_dir):
            for task in sorted(os.listdir(splits_dir)):
                task_dir = os.path.join(splits_dir, task)
                if not os.path.isdir(task_dir):
                    continue
                for split_file in ["train.npz", "val.npz", "test.npz"]:
                    split_path = os.path.join(task_dir, split_file)
                    if os.path.exists(split_path):
                        add_file(zf, split_path, f"{prefix}/{model_name}_splits/{task}/{split_file}")

        # Source CSVs (23 files)
        csv_prefix = "data/csv/nature_medicine/mimic"
        for f in sorted(os.listdir(CSV_DIR)):
            if f.endswith(".csv"):
                add_file(zf, os.path.join(CSV_DIR, f), f"{csv_prefix}/{f}")

    # Atomic replace
    old_size = os.path.getsize(zip_path) / 1e9 if os.path.exists(zip_path) else 0
    new_size = os.path.getsize(zip_tmp) / 1e9
    os.replace(zip_tmp, zip_path)
    print(f"\n  Written: {zip_path}")
    print(f"  Size: {new_size:.2f} GB (was {old_size:.2f} GB)")


if __name__ == "__main__":
    for model_name, master_npz in MODELS.items():
        create_model_zip(model_name, master_npz)

    # Also re-zip covariates
    cov_zip = os.path.join(MIMIC, "mimic_covariates.zip")
    cov_dir = os.path.join(MIMIC, "covariates")
    print(f"\n{'='*60}")
    print(f"  covariates -> mimic_covariates.zip")
    print(f"{'='*60}")
    with zipfile.ZipFile(cov_zip + ".tmp", "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(os.listdir(cov_dir)):
            local = os.path.join(cov_dir, f)
            arc = f"experiments/nature_medicine/mimic/covariates/{f}"
            add_file(zf, local, arc)
    os.replace(cov_zip + ".tmp", cov_zip)
    print(f"  Written: {cov_zip}")

    print(f"\nAll zips regenerated.")
