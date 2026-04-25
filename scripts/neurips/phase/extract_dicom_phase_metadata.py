"""Gate 1: DICOM phase-metadata extraction pilot for phi-JEPA.

Reads HeartRate (0018,1088), FrameTime (0018,1063), NumberOfFrames
(0028,0008) from MIMIC-IV-Echo raw DICOMs.

Operates on S3 paths (s3://echodata25/mimic-raw-staging/files/pXX/pPID/sSID/*.dcm)
with optional local caching.

Output: one CSV row per clip (study_id, clip_id, hr_bpm, frame_time_ms,
num_frames, fps, present_hr, present_ft, present_nf, dicom_path).
Plus a human-readable report at the end.

Pilot-scope design:
  - Sample N studies uniformly from the S3 key space (default 1000 studies).
  - Pull ALL clips in each sampled study (not just a random subset) because
    the arrhythmia filter works at study level and needs per-study spread.
  - Parallel I/O via a thread pool; pydicom parse in-memory (no local disk
    cache by default).
  - Record both the raw tag values AND which tags were missing, so missing-
    rate becomes a first-class statistic rather than a silent drop.

Gate 1 acceptance criteria (checked at end of run):
  - HR present on >= 95% of sampled clips
  - HR distribution centered on 60-90 bpm, spread 40-120 at tails
  - Within-study HR stdev <= 15 bpm on >= 85% of sampled studies

Usage:
  python scripts/neurips/phase/extract_dicom_phase_metadata.py \
    --out /opt/dlami/nvme/phase_pilot/clip_metadata.csv \
    --n-studies 1000 \
    --workers 32
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from typing import Optional

import boto3
import botocore.exceptions
import numpy as np
import pydicom
from pydicom.tag import Tag

# ---------------------------------------------------------------------------
# S3 layout constants
# ---------------------------------------------------------------------------

BUCKET = "echodata25"
PREFIX = "mimic-raw-staging/files/"

TAG_HR = Tag(0x0018, 0x1088)
TAG_FRAME_TIME = Tag(0x0018, 0x1063)
TAG_NUM_FRAMES = Tag(0x0028, 0x0008)
TAG_ACQ_DT = Tag(0x0008, 0x002A)
TAG_SOP_CLASS = Tag(0x0008, 0x0016)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass
class ClipRecord:
    study_id: str
    clip_id: str
    dicom_path: str
    hr_bpm: Optional[float]
    frame_time_ms: Optional[float]
    num_frames: Optional[int]
    fps: Optional[float]
    acquisition_dt: Optional[str]
    sop_class: Optional[str]
    present_hr: bool
    present_ft: bool
    present_nf: bool
    error: Optional[str]


# ---------------------------------------------------------------------------
# S3 listing
# ---------------------------------------------------------------------------


def list_studies(s3, n_studies: int, seed: int = 0) -> list[str]:
    """Return study prefixes under PREFIX.

    If n_studies < 0: return ALL studies (full corpus, no subsampling).
    Otherwise: random sample of n_studies (1 study per patient for diversity).

    A study prefix looks like:
      mimic-raw-staging/files/p10/p10002221/s94106955/
    """
    print(f"[list] enumerating patient groups under s3://{BUCKET}/{PREFIX} ...",
          flush=True)
    paginator = s3.get_paginator("list_objects_v2")

    # Level 1: pXX group prefixes.
    groups = []
    for page in paginator.paginate(Bucket=BUCKET, Prefix=PREFIX, Delimiter="/"):
        for cp in page.get("CommonPrefixes", []):
            groups.append(cp["Prefix"])
    print(f"[list] {len(groups)} patient groups (pXX/)", flush=True)

    full_corpus = n_studies < 0

    if full_corpus:
        # Enumerate every patient in every group, every study per patient.
        selected: list[str] = []
        for gi, group in enumerate(groups):
            patients = []
            for page in paginator.paginate(Bucket=BUCKET, Prefix=group, Delimiter="/"):
                patients.extend(cp["Prefix"] for cp in page.get("CommonPrefixes", []))
            for pat in patients:
                for page in paginator.paginate(Bucket=BUCKET, Prefix=pat, Delimiter="/"):
                    selected.extend(cp["Prefix"] for cp in page.get("CommonPrefixes", []))
            print(f"[list] group {gi+1}/{len(groups)} ({group}): "
                  f"{len(patients)} patients, total studies so far = {len(selected)}",
                  flush=True)
        print(f"[list] FULL CORPUS: {len(selected)} studies", flush=True)
        return selected

    # Subsampled: shuffle groups, shuffle patients, take 1 study per patient.
    rng = random.Random(seed)
    rng.shuffle(groups)

    selected = []
    for group in groups:
        if len(selected) >= n_studies:
            break
        # list patients in this group
        patients = []
        for page in paginator.paginate(Bucket=BUCKET, Prefix=group, Delimiter="/"):
            patients.extend(cp["Prefix"] for cp in page.get("CommonPrefixes", []))
        rng.shuffle(patients)

        for pat in patients:
            if len(selected) >= n_studies:
                break
            studies = []
            for page in paginator.paginate(Bucket=BUCKET, Prefix=pat, Delimiter="/"):
                studies.extend(cp["Prefix"] for cp in page.get("CommonPrefixes", []))
            rng.shuffle(studies)
            # take at most 1 study per patient to maximize patient diversity
            if studies:
                selected.append(studies[0])

    selected = selected[:n_studies]
    print(f"[list] selected {len(selected)} studies", flush=True)
    return selected


def list_clips(s3, study_prefix: str) -> list[str]:
    """Return all .dcm keys under a study prefix."""
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=BUCKET, Prefix=study_prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".dcm"):
                keys.append(obj["Key"])
    return keys


# ---------------------------------------------------------------------------
# DICOM fetch + parse
# ---------------------------------------------------------------------------

_STUDY_RE = re.compile(r"/s(\d+)/")
_CLIP_RE = re.compile(r"/s\d+/([^/]+)\.dcm$")


def parse_ids(key: str) -> tuple[str, str]:
    """Extract (study_id, clip_id) from the full S3 key."""
    s = _STUDY_RE.search(key)
    c = _CLIP_RE.search(key)
    study_id = f"s{s.group(1)}" if s else ""
    clip_id = c.group(1) if c else os.path.basename(key).rsplit(".", 1)[0]
    return study_id, clip_id


def fetch_and_parse(s3, key: str) -> ClipRecord:
    """Download the DICOM header only (no pixel data) and extract tags."""
    study_id, clip_id = parse_ids(key)
    full_path = f"s3://{BUCKET}/{key}"
    try:
        # Read only the first ~200KB — headers + metadata; pixel data is later.
        resp = s3.get_object(Bucket=BUCKET, Key=key, Range="bytes=0-262143")
        buf = io.BytesIO(resp["Body"].read())
        ds = pydicom.dcmread(buf, stop_before_pixels=True, force=True)
    except botocore.exceptions.ClientError as e:
        return _empty_record(study_id, clip_id, full_path, f"s3_error:{e.response['Error']['Code']}")
    except pydicom.errors.InvalidDicomError as e:
        return _empty_record(study_id, clip_id, full_path, f"dicom_parse:{e}")
    except Exception as e:
        return _empty_record(study_id, clip_id, full_path, f"other:{type(e).__name__}:{e}")

    e_hr = ds.get(TAG_HR)
    e_ft = ds.get(TAG_FRAME_TIME)
    e_nf = ds.get(TAG_NUM_FRAMES)
    e_dt = ds.get(TAG_ACQ_DT)
    e_sop = ds.get(TAG_SOP_CLASS)

    hr = _as_float(e_hr.value) if e_hr is not None else None
    ft = _as_float(e_ft.value) if e_ft is not None else None
    nf = _as_int(e_nf.value) if e_nf is not None else None
    fps = (1000.0 / ft) if ft and ft > 0 else None

    return ClipRecord(
        study_id=study_id,
        clip_id=clip_id,
        dicom_path=full_path,
        hr_bpm=hr,
        frame_time_ms=ft,
        num_frames=nf,
        fps=fps,
        acquisition_dt=str(e_dt.value) if e_dt is not None else None,
        sop_class=str(e_sop.value) if e_sop is not None else None,
        present_hr=e_hr is not None,
        present_ft=e_ft is not None,
        present_nf=e_nf is not None,
        error=None,
    )


def _empty_record(study_id: str, clip_id: str, path: str, error: str) -> ClipRecord:
    return ClipRecord(
        study_id=study_id, clip_id=clip_id, dicom_path=path,
        hr_bpm=None, frame_time_ms=None, num_frames=None, fps=None,
        acquisition_dt=None, sop_class=None,
        present_hr=False, present_ft=False, present_nf=False, error=error,
    )


def _as_float(v) -> Optional[float]:
    try:
        if isinstance(v, (list, tuple)):
            v = v[0]
        return float(v)
    except (TypeError, ValueError):
        return None


def _as_int(v) -> Optional[int]:
    try:
        if isinstance(v, (list, tuple)):
            v = v[0]
        return int(v)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _valid_hr(h: Optional[float]) -> bool:
    return h is not None and 40.0 <= h <= 180.0


def write_report(records: list[ClipRecord], report_path: str) -> dict:
    """Compute Gate 1 acceptance statistics and write a human-readable report."""
    n_clips = len(records)
    n_err = sum(1 for r in records if r.error is not None)
    n_has_hr = sum(1 for r in records if r.present_hr)
    n_has_ft = sum(1 for r in records if r.present_ft)
    n_has_nf = sum(1 for r in records if r.present_nf)
    n_valid_hr = sum(1 for r in records if _valid_hr(r.hr_bpm))

    hr_bpm_arr = np.array([r.hr_bpm for r in records if _valid_hr(r.hr_bpm)])
    ft_arr = np.array([r.frame_time_ms for r in records if r.frame_time_ms])

    # study-level stats
    studies: dict[str, list[float]] = {}
    for r in records:
        if _valid_hr(r.hr_bpm):
            studies.setdefault(r.study_id, []).append(r.hr_bpm)

    study_stds = np.array([np.std(v, ddof=0) for v in studies.values() if len(v) >= 2])
    n_studies = len(studies)
    n_studies_with_2plus = len(study_stds)
    n_studies_tight = int((study_stds <= 15).sum()) if len(study_stds) else 0

    gate_hr_coverage = (n_valid_hr / n_clips) if n_clips else 0.0
    gate_study_tight = (n_studies_tight / n_studies_with_2plus) if n_studies_with_2plus else 0.0

    report_lines = [
        "=" * 72,
        "GATE 1: DICOM PHASE-METADATA EXTRACTION PILOT",
        "=" * 72,
        f"Total clips scanned:           {n_clips}",
        f"Parse errors (S3 or DICOM):    {n_err}  ({100*n_err/max(1,n_clips):.1f}%)",
        "",
        f"Tag presence:",
        f"  HeartRate   (0018,1088):     {n_has_hr}/{n_clips}  ({100*n_has_hr/max(1,n_clips):.2f}%)",
        f"  FrameTime   (0018,1063):     {n_has_ft}/{n_clips}  ({100*n_has_ft/max(1,n_clips):.2f}%)",
        f"  NumFrames   (0028,0008):     {n_has_nf}/{n_clips}  ({100*n_has_nf/max(1,n_clips):.2f}%)",
        "",
        f"HR valid (40-180 bpm):         {n_valid_hr}/{n_clips}  ({100*gate_hr_coverage:.2f}%)",
        f"  Criterion: >= 95%            {'PASS' if gate_hr_coverage >= 0.95 else 'FAIL'}",
        "",
    ]

    if len(hr_bpm_arr):
        report_lines += [
            f"HR distribution (valid clips, n={len(hr_bpm_arr)}):",
            f"  mean   {hr_bpm_arr.mean():.1f} bpm",
            f"  median {np.median(hr_bpm_arr):.1f} bpm",
            f"  std    {hr_bpm_arr.std():.1f}",
            f"  p05    {np.percentile(hr_bpm_arr, 5):.1f}",
            f"  p95    {np.percentile(hr_bpm_arr, 95):.1f}",
            f"  min    {hr_bpm_arr.min():.1f}",
            f"  max    {hr_bpm_arr.max():.1f}",
            "",
            f"Histogram (10-bpm bins, 40-180):",
        ]
        hist, edges = np.histogram(hr_bpm_arr, bins=np.arange(40, 181, 10))
        for i, count in enumerate(hist):
            bar = "#" * int(60 * count / max(hist.max(), 1))
            report_lines.append(f"  {int(edges[i]):3d}-{int(edges[i+1]):3d}  {count:6d}  {bar}")
        report_lines.append("")

    if len(ft_arr):
        report_lines += [
            f"FrameTime distribution (n={len(ft_arr)}):",
            f"  mean   {ft_arr.mean():.2f} ms  ({1000/ft_arr.mean():.1f} fps)",
            f"  median {np.median(ft_arr):.2f} ms",
            f"  p05    {np.percentile(ft_arr, 5):.2f}",
            f"  p95    {np.percentile(ft_arr, 95):.2f}",
            "",
        ]

    report_lines += [
        f"Study-level arrhythmia filter:",
        f"  studies with ≥2 valid-HR clips:     {n_studies_with_2plus}",
        f"  studies with within-study std ≤15:  {n_studies_tight}  ({100*gate_study_tight:.2f}%)",
        f"  Criterion: ≥ 85%                    {'PASS' if gate_study_tight >= 0.85 else 'FAIL'}",
        "",
    ]

    if len(study_stds):
        report_lines += [
            f"Within-study HR stdev distribution:",
            f"  mean   {study_stds.mean():.2f}",
            f"  median {np.median(study_stds):.2f}",
            f"  p95    {np.percentile(study_stds, 95):.2f}",
            f"  max    {study_stds.max():.2f}",
            "",
        ]

    overall_pass = gate_hr_coverage >= 0.95 and gate_study_tight >= 0.85
    report_lines += [
        "=" * 72,
        f"OVERALL GATE 1:  {'PASS — proceed to Gate 2 (Run D)' if overall_pass else 'REVIEW — see criteria above'}",
        "=" * 72,
    ]

    report = "\n".join(report_lines)
    print(report)
    with open(report_path, "w") as fh:
        fh.write(report + "\n")
    print(f"[report] wrote {report_path}", flush=True)

    return {
        "n_clips": n_clips,
        "gate_hr_coverage": gate_hr_coverage,
        "gate_study_tight": gate_study_tight,
        "overall_pass": overall_pass,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output CSV (one row per clip).")
    ap.add_argument("--report", default=None,
                    help="Report path (default: <out>.report.txt).")
    ap.add_argument("--n-studies", type=int, default=1000)
    ap.add_argument("--max-clips-per-study", type=int, default=0,
                    help="0 = no cap; otherwise subsample up to N clips per study.")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--region", default="us-west-2")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    report_path = args.report or (args.out + ".report.txt")

    s3 = boto3.client("s3", region_name=args.region)

    t0 = time.time()
    studies = list_studies(s3, args.n_studies, seed=args.seed)

    # Gather clip keys for each selected study.
    print(f"[list] fetching clip keys for {len(studies)} studies ...", flush=True)
    all_keys: list[str] = []
    rng = random.Random(args.seed + 1)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(list_clips, s3, sp): sp for sp in studies}
        done = 0
        for fut in as_completed(futs):
            keys = fut.result()
            if args.max_clips_per_study and len(keys) > args.max_clips_per_study:
                keys = rng.sample(keys, args.max_clips_per_study)
            all_keys.extend(keys)
            done += 1
            if done % 100 == 0:
                print(f"[list]   {done}/{len(studies)} studies  "
                      f"({len(all_keys)} clips so far)", flush=True)
    print(f"[list] total clips: {len(all_keys)}  "
          f"(avg {len(all_keys)/max(1,len(studies)):.1f}/study)", flush=True)

    # Parse each clip's header (in parallel).
    records: list[ClipRecord] = []
    t_parse = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(fetch_and_parse, s3, k) for k in all_keys]
        for i, fut in enumerate(as_completed(futs)):
            records.append(fut.result())
            if (i + 1) % 500 == 0:
                rate = (i + 1) / max(0.1, time.time() - t_parse)
                eta = (len(futs) - (i + 1)) / max(0.1, rate)
                print(f"[parse]  {i+1}/{len(futs)} clips  "
                      f"({rate:.1f}/s  ETA {eta/60:.1f} min)", flush=True)

    # Write per-clip CSV.
    fieldnames = list(asdict(records[0]).keys()) if records else []
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in records:
            w.writerow(asdict(r))
    print(f"[csv] wrote {args.out} ({len(records)} rows)", flush=True)

    stats = write_report(records, report_path)

    dt = time.time() - t0
    print(f"[done] {len(records)} clips scanned in {dt/60:.1f} min", flush=True)
    return 0 if stats["overall_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
