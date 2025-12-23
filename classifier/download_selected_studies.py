#!/usr/bin/env python3
"""
Download all study directories listed in selected_study_dirs.csv with a clean terminal progress bar.
- Terminal: ONLY a progress bar (tqdm)
- Logs: full aws cli output (stdout+stderr) to a log file

Usage (macOS/Linux):
  python download_selected_studies.py \
    --csv selected_study_dirs.csv \
    --dest "/Users/alifmunim/Documents/Classification" \
    --log download.log \
    --jobs 1

Optional:
  --requester-pays
  --only-show-errors
  --resume   (skip already-downloaded studies if marker exists OR dir is non-empty)
  --skip-if-exists  (skip if destination dir exists and is non-empty; ignores marker)
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    print("Missing dependency: tqdm. Install with: pip install tqdm", file=sys.stderr)
    sys.exit(1)


def run_aws_cp(uri: str, out_dir: Path, log_fh, requester_pays: bool, only_show_errors: bool) -> int:
    cmd = ["aws", "s3", "cp", uri, str(out_dir), "--recursive"]
    if requester_pays:
        cmd += ["--request-payer", "requester"]
    if only_show_errors:
        cmd += ["--only-show-errors"]
    proc = subprocess.run(cmd, stdout=log_fh, stderr=log_fh)
    return proc.returncode


def is_non_empty_dir(p: Path) -> bool:
    # Fast non-emptiness check without materializing a full list
    return p.exists() and p.is_dir() and any(p.iterdir())


def study_already_present(out_dir: Path, marker: Path, resume: bool, skip_if_exists: bool) -> bool:
    """
    - --skip-if-exists: skip if out_dir exists and is non-empty (regardless of marker)
    - --resume: skip if marker exists OR out_dir exists and is non-empty
    """
    if skip_if_exists:
        return is_non_empty_dir(out_dir)
    if resume:
        return marker.exists() or is_non_empty_dir(out_dir)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=Path, help="selected_study_dirs.csv")
    ap.add_argument("--dest", required=True, type=Path, help="Local destination directory")
    ap.add_argument("--log", required=True, type=Path, help="Log file path")
    ap.add_argument("--jobs", type=int, default=1, help="Keep at 1 for clean progress. (Parallel not supported here.)")
    ap.add_argument("--requester-pays", action="store_true")
    ap.add_argument("--only-show-errors", action="store_true", help="Reduce AWS chatter in logs (still logs errors).")
    ap.add_argument("--resume", action="store_true",
                    help="Skip a study if '.download_complete' exists OR destination directory is non-empty.")
    ap.add_argument("--skip-if-exists", action="store_true",
                    help="Skip a study if destination directory exists and is non-empty (ignores marker).")
    args = ap.parse_args()

    if args.jobs != 1:
        print("This script supports --jobs 1 only (to keep a single clean progress bar).", file=sys.stderr)
        sys.exit(2)

    args.dest.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)

    # Read rows
    rows = []
    with args.csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            study_id = (r.get("study_id") or "").strip()
            s3_uri = (r.get("s3_uri") or "").strip()
            if study_id and s3_uri:
                rows.append((study_id, s3_uri))

    total = len(rows)
    if total == 0:
        print("No rows found in CSV.", file=sys.stderr)
        sys.exit(1)

    failed = []
    skipped = 0

    with args.log.open("a", encoding="utf-8") as log_fh:
        log_fh.write(f"\n=== RUN START: csv={args.csv} dest={args.dest} total={total} ===\n")

        for study_id, uri in tqdm(rows, total=total, unit="study", dynamic_ncols=True):
            out_dir = args.dest / study_id
            marker = out_dir / ".download_complete"

            if study_already_present(out_dir, marker, resume=args.resume, skip_if_exists=args.skip_if_exists):
                skipped += 1
                continue

            out_dir.mkdir(parents=True, exist_ok=True)

            log_fh.write(f"\n--- START {study_id} {uri} -> {out_dir} ---\n")
            rc = run_aws_cp(
                uri=uri,
                out_dir=out_dir,
                log_fh=log_fh,
                requester_pays=args.requester_pays,
                only_show_errors=args.only_show_errors,
            )
            log_fh.write(f"--- END {study_id} rc={rc} ---\n")

            if rc == 0:
                marker.write_text("ok\n", encoding="utf-8")
            else:
                failed.append(study_id)

        log_fh.write(f"\n=== RUN END: failed={len(failed)} skipped={skipped} ===\n")

    if failed:
        fail_path = args.log.with_suffix(args.log.suffix + ".failed_studies.txt")
        fail_path.write_text("\n".join(failed) + "\n", encoding="utf-8")
        sys.exit(3)


if __name__ == "__main__":
    main()
