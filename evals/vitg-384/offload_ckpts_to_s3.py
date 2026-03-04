#!/usr/bin/env python3
"""
Offload checkpoints to S3 while keeping the best ones locally for testing.

- Parses log_r0.csv (handles repeated header lines).
- Selects top-K epochs by accuracy.
- Keeps: latest.pt (if present) + epoch_XXX.pt for those top-K epochs.
- Uploads all other epoch_*.pt (and optionally other *.pt) to S3.
- Optionally deletes local files after successful upload.

Example:
  python3 offload_ckpts_to_s3.py \
    --exp-dir . \
    --s3-prefix s3://echodata25/results/uhn22k-classifier-vitg16-336-16f-pt279-a81-fs2-ns2-nvs1/checkpoints \
    --topk 3 \
    --delete-local

Dry-run:
  python3 offload_ckpts_to_s3.py --exp-dir . --s3-prefix s3://.../checkpoints --topk 3 --dry-run
"""

import argparse
import csv
import glob
import os
import subprocess
from pathlib import Path
from typing import List, Tuple


def parse_log(log_path: Path) -> List[Tuple[int, float, float]]:
    """Return list of (epoch, loss, acc). Skips repeated header lines."""
    rows: List[Tuple[int, float, float]] = []
    with log_path.open(newline="") as f:
        r = csv.reader(f)
        for row in r:
            if not row:
                continue
            if row[0].strip().lower() == "epoch":
                continue
            if len(row) < 3:
                continue
            try:
                ep = int(row[0])
                loss = float(row[1])
                acc = float(row[2])
            except Exception:
                continue
            rows.append((ep, loss, acc))
    return rows


def aws_s3_cp(src: Path, dst_s3: str, dry_run: bool):
    cmd = ["aws", "s3", "cp", str(src), dst_s3]
    if dry_run:
        print("[DRY-RUN]", " ".join(cmd))
        return
    subprocess.check_call(cmd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", required=True, help="Experiment directory containing epoch_*.pt and log_r0.csv")
    ap.add_argument("--s3-prefix", required=True, help="e.g. s3://echodata25/results/<exp>/checkpoints")
    ap.add_argument("--topk", type=int, default=3, help="Keep top-K epochs by acc (plus latest.pt if present)")
    ap.add_argument("--log", default="log_r0.csv", help="Log filename (default: log_r0.csv)")
    ap.add_argument("--keep-latest", action="store_true", default=True, help="Keep latest.pt locally (default: true)")
    ap.add_argument("--no-keep-latest", dest="keep_latest", action="store_false", help="Do not keep latest.pt")
    ap.add_argument("--include-non-epoch-pt", action="store_true",
                    help="Also upload other *.pt files (besides latest.pt) that are not epoch_*.pt")
    ap.add_argument("--delete-local", action="store_true", help="Delete local files after successful upload")
    ap.add_argument("--dry-run", action="store_true", help="Print actions without uploading/deleting")
    args = ap.parse_args()

    exp_dir = Path(args.exp_dir).expanduser().resolve()
    if not exp_dir.exists():
        raise SystemExit(f"exp-dir not found: {exp_dir}")

    log_path = exp_dir / args.log
    if not log_path.exists():
        raise SystemExit(f"log file not found: {log_path}")

    rows = parse_log(log_path)
    if not rows:
        raise SystemExit(f"No usable rows found in {log_path} (expected epoch,loss,acc)")

    # Top-K by accuracy (desc). Tie-breaker: lower loss.
    rows_sorted = sorted(rows, key=lambda x: (x[2], -x[1]), reverse=True)
    top = rows_sorted[: max(1, args.topk)]
    keep_epochs = sorted({ep for ep, _, _ in top})

    keep_files = set()
    for ep in keep_epochs:
        keep_files.add(f"epoch_{ep:03d}.pt")
    if args.keep_latest and (exp_dir / "latest.pt").exists():
        keep_files.add("latest.pt")

    # Discover files
    epoch_pts = sorted([Path(p).name for p in glob.glob(str(exp_dir / "epoch_*.pt"))])
    other_pts = []
    if args.include_non_epoch_pt:
        other_pts = sorted(
            [Path(p).name for p in glob.glob(str(exp_dir / "*.pt")) if Path(p).name not in epoch_pts]
        )

    candidates = epoch_pts + other_pts
    to_upload = [f for f in candidates if f not in keep_files]

    print(f"Experiment: {exp_dir}")
    print(f"S3 prefix:   {args.s3_prefix.rstrip('/')}/")
    print("\nTop epochs by acc (desc):")
    for ep, loss, acc in top:
        print(f"  epoch {ep:3d}  acc {acc:.5f}  loss {loss:.5f}")

    print("\nKeeping locally:")
    for f in sorted(keep_files):
        print(f"  {f}")

    print(f"\nWill upload {len(to_upload)} file(s):")
    for f in to_upload:
        print(f"  {f}")

    s3_prefix = args.s3_prefix.rstrip("/")

    # Upload
    for f in to_upload:
        src = exp_dir / f
        dst = f"{s3_prefix}/{f}"
        aws_s3_cp(src, dst, args.dry_run)

    # Delete local if requested
    if args.delete_local:
        for f in to_upload:
            p = exp_dir / f
            if args.dry_run:
                print("[DRY-RUN] rm", p)
            else:
                p.unlink()
        print(f"\nDeleted {len(to_upload)} local file(s).")
    else:
        print("\nDone (local files retained).")


if __name__ == "__main__":
    main()
