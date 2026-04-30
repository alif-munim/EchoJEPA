#!/usr/bin/env python3
"""Download DICOMs from studies with multiple clips for xcorr alignment testing.

Groups the MIMIC-IV-Echo record list by study_id, picks studies with at least
--clips-per-study entries, and downloads all clips for --n-studies of them.
Reuses the same S3 path / flat-naming convention as ``download_and_convert.py``
so downstream extract/crop scripts work unchanged.
"""

from __future__ import annotations

import argparse
import csv
import random
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
DICOM_DIR = HERE / "dicoms"
S3_BUCKET = "s3://echodata25/mimic-raw-staging"
RECORD_LIST = Path(
    "/mnt/custom-file-systems/efs/fs-0049217cdf69186d7_fsap-0fa7145b64eaa046b"
    "/vjepa2/uhn_echo/nature_medicine/data_exploration/mimic/mimic-iv-echo"
    "/echo-record-list.csv"
)


def download(row: dict) -> tuple[dict, Path | None, str | None]:
    rel = row["dicom_filepath"]
    fname = Path(rel).name
    local = DICOM_DIR / fname
    if local.exists() and local.stat().st_size > 0:
        return row, local, None
    s3_uri = f"{S3_BUCKET}/{rel}"
    r = subprocess.run(
        ["aws", "s3", "cp", s3_uri, str(local), "--quiet"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        return row, None, r.stderr.strip() or f"exit {r.returncode}"
    return row, local, None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-studies", type=int, default=30,
                    help="Number of multi-clip studies to download (default 30)")
    ap.add_argument("--clips-per-study", type=int, default=4,
                    help="Minimum clips required to count a study (default 4). "
                         "Higher -> more cine-likely clips after single-frame skips.")
    ap.add_argument("--max-clips", type=int, default=8,
                    help="Cap clips taken per study (default 8)")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    DICOM_DIR.mkdir(parents=True, exist_ok=True)

    existing_studies = set()
    for p in DICOM_DIR.glob("*.dcm"):
        existing_studies.add(p.stem.split("_")[0])

    print(f"Reading {RECORD_LIST.name}")
    by_study: dict[str, list[dict]] = defaultdict(list)
    with RECORD_LIST.open() as f:
        for row in csv.DictReader(f):
            by_study[row["study_id"]].append(row)

    candidates = [(s, rs) for s, rs in by_study.items()
                  if len(rs) >= args.clips_per_study and s not in existing_studies]
    print(f"Studies with >={args.clips_per_study} clips (not already downloaded): "
          f"{len(candidates)}")

    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    picked = candidates[:args.n_studies]

    to_download: list[dict] = []
    for study, rows in picked:
        rng.shuffle(rows)
        to_download.extend(rows[:args.max_clips])
    print(f"Selected {len(picked)} studies, {len(to_download)} clips total")

    print(f"Downloading to {DICOM_DIR}/  (workers={args.workers})")
    n_ok = n_fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(download, r) for r in to_download]
        for i, fut in enumerate(as_completed(futs), 1):
            row, local, err = fut.result()
            if err:
                n_fail += 1
                print(f"  [{i:3d}/{len(to_download)}] FAIL  {row['dicom_filepath']}: {err}")
            else:
                n_ok += 1
                if i % 25 == 0 or i == len(to_download):
                    print(f"  [{i:3d}/{len(to_download)}] ok    {local.name}")

    print(f"\nDone: {n_ok} downloaded, {n_fail} failed")
    print(f"Next: extract last frames and cropped ECG strips for the new clips.")


if __name__ == "__main__":
    main()
