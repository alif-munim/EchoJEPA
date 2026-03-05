#!/usr/bin/env python3
"""
Rewrite labels_patient_split.csv so each row points to the corresponding MP4 path,
and (optionally) drop rows whose MP4 path does not exist.

Input (per-row JPG path pattern):
  .../h4h_ef_frames/<STUDY_ID>/<SERIES_ID>/mp4/<SOP_UID>_masked.jpg

Output MP4 path (matches your extracted layout):
  <ROOT>/<STUDY_ID>/unmasked/<SERIES_ID>/<SOP_UID>.mp4

Creates:
  - <out>.csv (adds column: mp4_path; may drop invalid rows if --drop-missing)
  - prints basic stats

Usage:
  python3 map_labels_to_mp4.py \
    --in labels_patient_split.csv \
    --root /absolute/path/to/uhn_studies_22k_585 \
    --out labels_patient_split_mp4.csv \
    --check-exists \
    --drop-missing

Notes:
- If you use --drop-missing, the script will only keep rows where mp4_path exists.
- If you omit --root, it defaults to ./uhn_studies_22k_585 (resolved to absolute).
"""

import argparse
import csv
from pathlib import Path
from typing import Optional, Tuple


KEY = "h4h_ef_frames/"


def extract_ids(p: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Returns (study_id, series_id, filename) from a path containing .../h4h_ef_frames/<study>/<series>/..."""
    i = p.find(KEY)
    if i == -1:
        return None, None, None
    rest = p[i + len(KEY):]
    parts = rest.split("/")
    if len(parts) < 3:
        return None, None, None
    return parts[0], parts[1], parts[-1]


def jpg_to_sop_uid(fname: str) -> Optional[str]:
    if fname.endswith("_masked.jpg"):
        return fname[:-len("_masked.jpg")]
    if fname.endswith(".jpg"):
        return fname[:-len(".jpg")]
    return None


def jpg_to_mp4_path(jpg_path: str, root: Path) -> Optional[Path]:
    study_id, series_id, fname = extract_ids(jpg_path)
    if not study_id or not series_id or not fname:
        return None
    sop = jpg_to_sop_uid(fname)
    if not sop:
        return None
    return root / study_id / series_id / f"{sop}.mp4"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_csv", required=True, help="labels_patient_split.csv")
    ap.add_argument("--root", default="uhn_studies_22k_585", help="Root dir containing extracted study folders")
    ap.add_argument("--out", required=True, help="Output CSV (adds mp4_path)")
    ap.add_argument("--check-exists", action="store_true", help="Check whether mp4_path exists on disk")
    ap.add_argument("--drop-missing", action="store_true",
                    help="If set, drop rows whose derived mp4_path does not exist (requires --check-exists)")
    args = ap.parse_args()

    if args.drop_missing and not args.check_exists:
        raise SystemExit("--drop-missing requires --check-exists (we need to test filesystem existence).")

    in_csv = Path(args.in_csv)
    out_csv = Path(args.out)
    root = Path(args.root).expanduser().resolve()

    if not in_csv.exists():
        raise SystemExit(f"Input CSV not found: {in_csv}")

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    exists = 0
    missing = 0
    unparsable = 0
    total = 0
    kept = 0
    dropped = 0

    with in_csv.open("r", newline="", encoding="utf-8") as f_in:
        r = csv.DictReader(f_in)
        if not r.fieldnames:
            raise SystemExit("Input CSV has no header/fieldnames")

        fieldnames = list(r.fieldnames)
        out_fields = fieldnames + (["mp4_path"] if "mp4_path" not in fieldnames else [])

        with out_csv.open("w", newline="", encoding="utf-8") as f_out:
            w = csv.DictWriter(f_out, fieldnames=out_fields)
            w.writeheader()

            for row in r:
                total += 1
                jpg = (row.get("filename") or "").strip()
                mp4 = jpg_to_mp4_path(jpg, root)

                if mp4 is None:
                    row["mp4_path"] = ""
                    unparsable += 1
                    if args.drop_missing:
                        dropped += 1
                        continue
                    w.writerow(row)
                    kept += 1
                    continue

                row["mp4_path"] = str(mp4)

                if args.check_exists:
                    if mp4.exists():
                        exists += 1
                        w.writerow(row)
                        kept += 1
                    else:
                        missing += 1
                        if args.drop_missing:
                            dropped += 1
                            continue
                        w.writerow(row)
                        kept += 1
                else:
                    w.writerow(row)
                    kept += 1

    print(f"ROOT: {root}")
    print(f"Wrote: {out_csv}")
    print(f"Total input rows: {total}")
    print(f"Kept rows: {kept}")
    if args.drop_missing:
        print(f"Dropped rows (missing/unparsable): {dropped}")
    print(f"Unparsable rows: {unparsable}")
    if args.check_exists:
        print(f"MP4 exists: {exists}")
        print(f"MP4 missing: {missing}")


if __name__ == "__main__":
    main()
