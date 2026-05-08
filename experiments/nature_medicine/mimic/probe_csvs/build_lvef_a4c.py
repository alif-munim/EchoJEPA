"""Build A4C-only train/val/test splits for LVEF regression on MIMIC echos.

Seeded from `lvef_structured` (TTE + B-mode + value_range [10, 85] + matching window 1d,
already filtered to {A2C, A4C}). This script narrows further to A4C-only clips using the
ConvNeXt view manifest (view_status == OK).

Labels are carried over verbatim from lvef_structured (no relabelling); z-score params
are re-derived from the A4C-filtered train split so predictions are calibrated to the
same distribution the probe sees.

Output: lvef_a4c/{train,val,test}.csv + zscore_params.json + label_meta.json
"""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path("/mnt/custom-file-systems/efs/fs-0049217cdf69186d7_fsap-0fa7145b64eaa046b/vjepa2")
SRC_SPLIT_DIR = REPO_ROOT / "experiments/nature_medicine/mimic/probe_csvs/lvef_structured"
OUT_SPLIT_DIR = REPO_ROOT / "experiments/nature_medicine/mimic/probe_csvs/lvef_a4c"
VIEW_MANIFEST = Path("/home/sagemaker-user/user-default-efs/vjepa2/classifier/output/mimic_view_predictions.csv")

ALLOWED_VIEWS = {"A4C"}
SPLITS = ("train", "val", "test")
STUDY_ID_RE = re.compile(r"/s(\d+)/")


def load_a4c_uris(manifest_path: Path) -> set[str]:
    a4c: set[str] = set()
    with manifest_path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row["view_status"] == "OK" and row["view"] in ALLOWED_VIEWS:
                a4c.add(row["s3_uri"])
    return a4c


def study_id_from_path(path: str) -> str:
    m = STUDY_ID_RE.search(path)
    return m.group(1) if m else ""


def filter_split(src: Path, a4c: set[str]) -> tuple[list[tuple[str, float]], dict]:
    clips_in = 0
    clips_out = 0
    studies_in: set[str] = set()
    studies_out: set[str] = set()
    kept: list[tuple[str, float]] = []

    with src.open("r") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            path, label = line.rsplit(" ", 1)
            clips_in += 1
            sid = study_id_from_path(path)
            if sid:
                studies_in.add(sid)
            if path not in a4c:
                continue
            try:
                val = float(label)
            except ValueError:
                continue
            kept.append((path, val))
            clips_out += 1
            if sid:
                studies_out.add(sid)

    vals = [v for _, v in kept]
    stats = {
        "clips_in": clips_in,
        "clips_out": clips_out,
        "studies_in": len(studies_in),
        "studies_out": len(studies_out),
        "studies_dropped_no_a4c": len(studies_in - studies_out),
        "lvef_stats": _stats(vals),
        "clips_per_study": _clips_per_study(kept),
    }
    return kept, stats


def _stats(vals: list[float]) -> dict:
    if not vals:
        return {"n": 0}
    s = sorted(vals)
    n = len(s)
    mean = sum(s) / n
    var = sum((v - mean) ** 2 for v in s) / n
    median = s[n // 2] if n % 2 == 1 else (s[n // 2 - 1] + s[n // 2]) / 2
    return {
        "n": n,
        "min": round(s[0], 2),
        "p25": round(s[n // 4], 2),
        "median": round(median, 2),
        "p75": round(s[3 * n // 4], 2),
        "max": round(s[-1], 2),
        "mean": round(mean, 4),
        "std": round(var ** 0.5, 4),
    }


def _clips_per_study(rows: list[tuple[str, float]]) -> dict:
    counts: dict[str, int] = defaultdict(int)
    for p, _ in rows:
        sid = study_id_from_path(p)
        if sid:
            counts[sid] += 1
    vals = sorted(counts.values())
    if not vals:
        return {"min": 0, "median": 0, "max": 0, "mean": 0.0}
    n = len(vals)
    median = vals[n // 2] if n % 2 == 1 else (vals[n // 2 - 1] + vals[n // 2]) / 2
    return {"min": vals[0], "median": median, "max": vals[-1], "mean": round(sum(vals) / n, 2)}


def write_split(rows: list[tuple[str, float]], dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w") as fh:
        for path, label in rows:
            fh.write(f"{path} {label:.6f}\n")


def main() -> None:
    print(f"Loading A4C view manifest: {VIEW_MANIFEST.name}")
    a4c = load_a4c_uris(VIEW_MANIFEST)
    print(f"  A4C OK clips: {len(a4c):,}")

    src_meta_path = SRC_SPLIT_DIR / "task_meta.json"
    src_task_meta = json.loads(src_meta_path.read_text()) if src_meta_path.exists() else {}

    meta = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "task_type": "regression",
        "target": "lvef_percent",
        "source_splits": str(SRC_SPLIT_DIR.relative_to(REPO_ROOT)),
        "source_task_meta": src_task_meta,
        "view_manifest": str(VIEW_MANIFEST),
        "view_filter": sorted(ALLOWED_VIEWS),
        "view_status_filter": "OK",
        "splits": {},
    }

    split_rows: dict[str, list[tuple[str, float]]] = {}
    for split in SPLITS:
        src = SRC_SPLIT_DIR / f"{split}.csv"
        print(f"\n[{split}] filtering {src.name}")
        rows, stats = filter_split(src, a4c)
        split_rows[split] = rows
        meta["splits"][split] = stats
        dst = OUT_SPLIT_DIR / f"{split}.csv"
        write_split(rows, dst)
        s = stats["lvef_stats"]
        cps = stats["clips_per_study"]
        print(f"  clips:    {stats['clips_in']:>7,} -> kept {stats['clips_out']:>6,}")
        print(f"  studies:  {stats['studies_in']:>7,} -> kept {stats['studies_out']:>6,}  "
              f"(dropped no-A4C: {stats['studies_dropped_no_a4c']})")
        print(f"  LVEF:     mean={s['mean']} std={s['std']} median={s['median']}  range=[{s['min']}, {s['max']}]")
        print(f"  clips/study: min={cps['min']} median={cps['median']} max={cps['max']}")
        print(f"  -> {dst}")

    train_vals = [v for _, v in split_rows["train"]]
    n = len(train_vals)
    mean = sum(train_vals) / n
    std = (sum((v - mean) ** 2 for v in train_vals) / n) ** 0.5
    zscore_path = OUT_SPLIT_DIR / "zscore_params.json"
    with zscore_path.open("w") as fh:
        json.dump({"target_mean": mean, "target_std": std}, fh)
    print(f"\nZ-score (from A4C-filtered train):")
    print(f"  target_mean = {mean:.10f}")
    print(f"  target_std  = {std:.10f}")
    print(f"  -> {zscore_path}")

    meta["zscore_params"] = {"target_mean": mean, "target_std": std, "computed_from": "train clips (A4C-filtered)"}
    meta_path = OUT_SPLIT_DIR / "label_meta.json"
    with meta_path.open("w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"Wrote metadata: {meta_path}")


if __name__ == "__main__":
    main()
