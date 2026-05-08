"""Build PLAX-only train/val/test splits for MIMIC HCM detection.

Filters the canonical `disease_hcm_v4.1` splits down to clips the view classifier
labelled as PLAX (view_status == 'OK'). Studies with zero PLAX clips are dropped.

PLAX is the dominant HCM-imaging view (septal wall thickening, SAM, LVOT
gradient) and aligns with the UHN disease_hcm probe's allowed_views
(PLAX, PSAX-PM, PSAX-MV, A4C).

Output: disease_hcm_plax/{train,val,test}.csv + viewfilter_meta.json
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path("/mnt/custom-file-systems/efs/fs-0049217cdf69186d7_fsap-0fa7145b64eaa046b/vjepa2")
SRC_SPLIT_DIR = REPO_ROOT / "experiments/nature_medicine/mimic/probe_csvs/disease_hcm_v4.1"
OUT_SPLIT_DIR = REPO_ROOT / "experiments/nature_medicine/mimic/probe_csvs/disease_hcm_plax"
VIEW_MANIFEST = Path("/home/sagemaker-user/user-default-efs/vjepa2/classifier/output/mimic_view_predictions.csv")

ALLOWED_VIEWS = {"PLAX"}
SPLITS = ("train", "val", "test")


def load_allowed_uris(manifest_path: Path) -> set[str]:
    allowed = set()
    total = 0
    with manifest_path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            total += 1
            if row["view_status"] != "OK":
                continue
            if row["view"] not in ALLOWED_VIEWS:
                continue
            allowed.add(row["s3_uri"])
    print(f"  manifest rows read: {total:,}")
    print(f"  {sorted(ALLOWED_VIEWS)} OK clips: {len(allowed):,}")
    return allowed


def study_id_from_path(path: str) -> str:
    parts = path.split("/")
    for p in parts:
        if p.startswith("s") and p[1:].isdigit():
            return p
    return ""


def filter_split(src: Path, allowed: set[str]) -> tuple[list[tuple[str, str]], dict]:
    rows_in = 0
    rows_kept = 0
    studies_in: set[str] = set()
    studies_kept: set[str] = set()
    pos_clips_in = pos_clips_kept = 0
    pos_studies_in: set[str] = set()
    pos_studies_kept: set[str] = set()

    kept: list[tuple[str, str]] = []
    with src.open("r") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            path, label = line.rsplit(" ", 1)
            rows_in += 1
            sid = study_id_from_path(path)
            if sid:
                studies_in.add(sid)
            if label == "1":
                pos_clips_in += 1
                if sid:
                    pos_studies_in.add(sid)
            if path in allowed:
                rows_kept += 1
                kept.append((path, label))
                if sid:
                    studies_kept.add(sid)
                if label == "1":
                    pos_clips_kept += 1
                    if sid:
                        pos_studies_kept.add(sid)

    stats = {
        "clips_in": rows_in,
        "clips_out": rows_kept,
        "studies_in": len(studies_in),
        "studies_out": len(studies_kept),
        "studies_dropped_no_view": len(studies_in - studies_kept),
        "pos_clips_in": pos_clips_in,
        "pos_clips_out": pos_clips_kept,
        "pos_studies_in": len(pos_studies_in),
        "pos_studies_out": len(pos_studies_kept),
        "pos_studies_dropped": len(pos_studies_in - pos_studies_kept),
    }
    return kept, stats


def write_split(rows: list[tuple[str, str]], dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w") as fh:
        for path, label in rows:
            fh.write(f"{path} {label}\n")


def per_study_clip_counts(rows: list[tuple[str, str]]) -> dict:
    counts: dict[str, int] = defaultdict(int)
    for path, _ in rows:
        sid = study_id_from_path(path)
        if sid:
            counts[sid] += 1
    vals = sorted(counts.values())
    if not vals:
        return {"min": 0, "median": 0, "max": 0, "mean": 0.0}
    n = len(vals)
    median = vals[n // 2] if n % 2 == 1 else (vals[n // 2 - 1] + vals[n // 2]) / 2
    return {"min": vals[0], "median": median, "max": vals[-1], "mean": round(sum(vals) / n, 2)}


def main() -> None:
    print(f"Loading view manifest: {VIEW_MANIFEST}")
    allowed = load_allowed_uris(VIEW_MANIFEST)

    meta: dict = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_splits": str(SRC_SPLIT_DIR.relative_to(REPO_ROOT)),
        "view_manifest": str(VIEW_MANIFEST),
        "view_filter": sorted(ALLOWED_VIEWS),
        "view_status_filter": "OK",
        "allowed_ok_clips_in_manifest": len(allowed),
        "splits": {},
    }

    for split in SPLITS:
        src = SRC_SPLIT_DIR / f"{split}.csv"
        print(f"\n[{split}] filtering {src.name}")
        rows, stats = filter_split(src, allowed)
        dst = OUT_SPLIT_DIR / f"{split}.csv"
        write_split(rows, dst)
        stats["clips_per_study"] = per_study_clip_counts(rows)
        meta["splits"][split] = stats
        print(f"  clips:    {stats['clips_in']:>7,} -> {stats['clips_out']:>6,}")
        print(f"  studies:  {stats['studies_in']:>7,} -> {stats['studies_out']:>6,}  (dropped {stats['studies_dropped_no_view']})")
        print(f"  pos studies: {stats['pos_studies_in']:>4,} -> {stats['pos_studies_out']:>4,}  (dropped {stats['pos_studies_dropped']})")
        print(f"  pos clips:   {stats['pos_clips_in']:>4,} -> {stats['pos_clips_out']:>4,}")
        print(f"  clips/study: min={stats['clips_per_study']['min']} median={stats['clips_per_study']['median']} max={stats['clips_per_study']['max']}")
        print(f"  -> {dst}")

    meta_path = OUT_SPLIT_DIR / "viewfilter_meta.json"
    with meta_path.open("w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"\nWrote metadata: {meta_path}")


if __name__ == "__main__":
    main()
