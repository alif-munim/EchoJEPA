#!/usr/bin/env python3
"""
Patient-disjoint train/val/test split for echo classifiers (image or video).

Assumptions / Inputs
- labels CSV has at least: filename,label
  (it can be per-video, per-frame, per-image; each row is a training sample)
- syngo CSV has at least: DeidentifiedStudyID,PATIENT_ID
- The study ID (DeidentifiedStudyID) can be extracted from filename by taking the
  directory immediately after "h4h_ef_frames/".

What it does
- Maps each sample -> study_id -> patient_id
- Splits *patients* into train/val/test with fixed random seed
- Uses a greedy stratified assignment (by label counts) at patient level, so label
  distribution is roughly preserved while staying patient-disjoint
- Writes:
  1) output CSV = original rows + columns: study_id,patient_id,split
  2) patients_train.txt / patients_val.txt / patients_test.txt
  3) split_summary.txt with counts

Example
python3 make_patient_split.py \
  --labels labels_masked_inplace.csv \
  --syngo syngo-deid.dedup.csv \
  --out labels_patient_split.csv \
  --seed 42 \
  --train 0.80 --val 0.10 --test 0.10
"""

import argparse
import csv
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def extract_study_id(path: str) -> Optional[str]:
    key = "h4h_ef_frames/"
    i = path.find(key)
    if i == -1:
        return None
    rest = path[i + len(key):]
    study = rest.split("/", 1)[0]
    return study if study else None


def load_syngo_map(syngo_path: Path) -> Dict[str, str]:
    """DeidentifiedStudyID -> PATIENT_ID"""
    m = {}
    with syngo_path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        fields = set(r.fieldnames or [])
        need = {"DeidentifiedStudyID", "PATIENT_ID"}
        if not need.issubset(fields):
            raise SystemExit(f"{syngo_path}: missing required columns {need}. Found: {r.fieldnames}")

        for row in r:
            sid = (row.get("DeidentifiedStudyID") or "").strip().strip('"')
            pid = (row.get("PATIENT_ID") or "").strip()
            if sid and pid:
                m[sid] = pid
    return m


def l2_objective(split_label_counts: Dict[str, Counter],
                 target_label_counts: Dict[str, Dict[str, float]]) -> float:
    """Sum of squared errors across splits and labels."""
    err = 0.0
    for split, target in target_label_counts.items():
        actual = split_label_counts[split]
        for lab, t in target.items():
            a = float(actual.get(lab, 0))
            d = a - t
            err += d * d
    return err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, type=Path, help="labels CSV with at least filename,label")
    ap.add_argument("--syngo", required=True, type=Path, help="syngo-deid.dedup.csv (must have DeidentifiedStudyID,PATIENT_ID)")
    ap.add_argument("--out", required=True, type=Path, help="output CSV with split column")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train", type=float, default=0.80)
    ap.add_argument("--val", type=float, default=0.10)
    ap.add_argument("--test", type=float, default=0.10)
    ap.add_argument("--strict", action="store_true",
                    help="Fail if any row cannot be mapped to a patient (missing study_id or syngo entry).")
    args = ap.parse_args()

    # sanity ratios
    s = args.train + args.val + args.test
    if abs(s - 1.0) > 1e-6:
        raise SystemExit(f"train+val+test must sum to 1.0 (got {s})")

    rng = random.Random(args.seed)

    study_to_patient = load_syngo_map(args.syngo)

    # Read label rows
    rows = []
    missing_study = 0
    missing_patient = 0

    with args.labels.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        fields = set(r.fieldnames or [])
        if "filename" not in fields or "label" not in fields:
            raise SystemExit(f"{args.labels}: must contain columns filename,label. Found: {r.fieldnames}")

        for row in r:
            fn = (row.get("filename") or "").strip()
            lab = (row.get("label") or "").strip()

            sid = extract_study_id(fn) if fn else None
            if not sid:
                missing_study += 1
                if args.strict:
                    raise SystemExit(f"Could not extract study_id from filename: {fn}")
                pid = ""
            else:
                pid = study_to_patient.get(sid, "")
                if not pid:
                    missing_patient += 1
                    if args.strict:
                        raise SystemExit(f"study_id {sid} not found in {args.syngo}")
            rows.append((row, sid or "", pid, lab))

    # Build patient -> label counts (based on sample rows)
    patient_label_counts: Dict[str, Counter] = defaultdict(Counter)
    patient_total_samples: Counter = Counter()
    all_labels: Counter = Counter()

    valid_patients = set()
    for _, _, pid, lab in rows:
        if not pid:
            continue
        valid_patients.add(pid)
        patient_label_counts[pid][lab] += 1
        patient_total_samples[pid] += 1
        all_labels[lab] += 1

    patients = list(valid_patients)
    if not patients:
        raise SystemExit("No patients found (all rows missing patient mapping?).")

    # Target sample counts per split (by samples, not by patients)
    total_samples = sum(all_labels.values())
    split_ratios = {"train": args.train, "val": args.val, "test": args.test}
    target_samples = {k: total_samples * v for k, v in split_ratios.items()}

    # Target per-label sample counts per split
    target_label_counts: Dict[str, Dict[str, float]] = {}
    for split, ratio in split_ratios.items():
        target_label_counts[split] = {lab: all_labels[lab] * ratio for lab in all_labels}

    # Greedy patient assignment: place largest patients first
    patients_sorted = sorted(patients, key=lambda p: patient_total_samples[p], reverse=True)

    # Initialize split structures
    split_patients = {"train": set(), "val": set(), "test": set()}
    split_label_counts = {"train": Counter(), "val": Counter(), "test": Counter()}
    split_sample_counts = {"train": 0, "val": 0, "test": 0}

    # Small random tie-breaking for reproducibility
    def split_order():
        keys = ["train", "val", "test"]
        rng.shuffle(keys)
        return keys

    for pid in patients_sorted:
        plc = patient_label_counts[pid]
        p_samples = patient_total_samples[pid]

        best_split = None
        best_obj = None

        for split in split_order():
            # Soft constraint: keep sample counts near target
            # Try candidate assignment
            split_label_counts[split].update(plc)
            split_sample_counts[split] += p_samples

            obj = l2_objective(split_label_counts, target_label_counts)
            # Add sample-count penalty to keep totals near ratio
            for sp in ("train", "val", "test"):
                d = split_sample_counts[sp] - target_samples[sp]
                obj += (d * d) * 0.01  # small weight; label stratification dominates

            # revert
            split_label_counts[split].subtract(plc)
            # remove zeros from Counter after subtract
            split_label_counts[split] += Counter()
            split_sample_counts[split] -= p_samples

            if best_obj is None or obj < best_obj:
                best_obj = obj
                best_split = split

        # commit
        split_patients[best_split].add(pid)
        split_label_counts[best_split].update(plc)
        split_sample_counts[best_split] += p_samples

    # Build patient -> split map
    patient_to_split = {}
    for sp in ("train", "val", "test"):
        for pid in split_patients[sp]:
            patient_to_split[pid] = sp

    # Write output CSV
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        # keep original columns, append new ones
        orig_fieldnames = list(rows[0][0].keys())
        out_fields = orig_fieldnames + ["study_id", "patient_id", "split"]
        w = csv.DictWriter(f, fieldnames=out_fields)
        w.writeheader()

        for orig_row, sid, pid, _lab in rows:
            split = patient_to_split.get(pid, "") if pid else ""
            out_row = dict(orig_row)
            out_row["study_id"] = sid
            out_row["patient_id"] = pid
            out_row["split"] = split
            w.writerow(out_row)

    # Write patient lists
    for sp in ("train", "val", "test"):
        pfile = args.out.with_name(f"patients_{sp}.txt")
        with pfile.open("w", encoding="utf-8") as f:
            for pid in sorted(split_patients[sp]):
                f.write(pid + "\n")

    # Summary
    summary_path = args.out.with_name("split_summary.txt")
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"seed={args.seed}\n")
        f.write(f"ratios train/val/test={args.train}/{args.val}/{args.test}\n\n")
        f.write(f"total_rows_in_labels={len(rows)}\n")
        f.write(f"rows_missing_study_id={missing_study}\n")
        f.write(f"rows_missing_patient_id={missing_patient}\n\n")
        f.write(f"unique_patients_total={len(patients)}\n")
        for sp in ("train", "val", "test"):
            f.write(f"\n[{sp}]\n")
            f.write(f"unique_patients={len(split_patients[sp])}\n")
            f.write(f"samples={split_sample_counts[sp]}\n")
            # per-label sample counts
            for lab, cnt in split_label_counts[sp].most_common():
                f.write(f"  {lab}: {cnt}\n")

    # Minimal terminal output
    print(f"Wrote: {args.out}")
    print(f"Wrote: {summary_path}")
    print(f"Patients: train={len(split_patients['train'])}, val={len(split_patients['val'])}, test={len(split_patients['test'])}")
    print(f"Samples:  train={split_sample_counts['train']}, val={split_sample_counts['val']}, test={split_sample_counts['test']}")


if __name__ == "__main__":
    main()
