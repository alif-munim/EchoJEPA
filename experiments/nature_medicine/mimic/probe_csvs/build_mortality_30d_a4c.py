"""Build A4C-only train/val/test splits for 30-day all-cause mortality prediction on MIMIC echos.

Label source: mortality_30d.csv from data_exploration/mimic/csv/ (prebuilt by
build_all_labels.py). Binary: 1 = death within 30 days of echo study, 0 otherwise.
Coverage is 100% of the 7,243 echo cohort; prevalence 5.7% (411 positive).

View filter: A4C clips with view_status == OK from the ConvNeXt view manifest.
Splits: patient-level partition reused from disease_hf_v4.1 (zero subject leakage).
Output format: space-delimited "<s3_path> <int_label>".
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path("/mnt/custom-file-systems/efs/fs-0049217cdf69186d7_fsap-0fa7145b64eaa046b/vjepa2")
LABEL_CSV = REPO_ROOT / "uhn_echo/nature_medicine/data_exploration/mimic/csv/mortality_30d.csv"
SRC_SPLIT_DIR = REPO_ROOT / "experiments/nature_medicine/mimic/probe_csvs/disease_hf_v4.1"
OUT_SPLIT_DIR = REPO_ROOT / "experiments/nature_medicine/mimic/probe_csvs/mortality_30d_a4c"
VIEW_MANIFEST = Path("/home/sagemaker-user/user-default-efs/vjepa2/classifier/output/mimic_view_predictions.csv")

ALLOWED_VIEWS = {"A4C"}
SPLITS = ("train", "val", "test")
STUDY_ID_RE = re.compile(r"/s(\d+)/")


def load_labels(path: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    with path.open("r", newline="") as fh:
        r = csv.DictReader(fh)
        for row in r:
            sid = row["study_id"].strip()
            v = row["mortality_30d"].strip()
            if sid and v in ("0", "1"):
                out[sid] = int(v)
    return out


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


def filter_split(src: Path, a4c: set[str], labels: dict[str, int]) -> tuple[list[tuple[str, int]], dict]:
    clips_in = clips_a4c = clips_out = 0
    studies_in: set[str] = set()
    studies_out: set[str] = set()
    studies_unlabelled: set[str] = set()
    pos_studies: set[str] = set()
    neg_studies: set[str] = set()
    kept: list[tuple[str, int]] = []
    with src.open("r") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            path, _ = line.rsplit(" ", 1)
            clips_in += 1
            sid = study_id_from_path(path)
            if sid:
                studies_in.add(sid)
            if path not in a4c:
                continue
            clips_a4c += 1
            lbl = labels.get(sid)
            if lbl is None:
                studies_unlabelled.add(sid)
                continue
            kept.append((path, lbl))
            clips_out += 1
            if sid:
                studies_out.add(sid)
                (pos_studies if lbl == 1 else neg_studies).add(sid)
    stats = {
        "clips_in": clips_in,
        "clips_a4c": clips_a4c,
        "clips_out": clips_out,
        "studies_in": len(studies_in),
        "studies_out": len(studies_out),
        "studies_unlabelled": len(studies_unlabelled),
        "pos_studies": len(pos_studies),
        "neg_studies": len(neg_studies),
        "pos_clips": sum(1 for _, l in kept if l == 1),
        "neg_clips": sum(1 for _, l in kept if l == 0),
        "clips_per_study": _clips_per_study(kept),
    }
    return kept, stats


def _clips_per_study(rows: list[tuple[str, int]]) -> dict:
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


def write_split(rows: list[tuple[str, int]], dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w") as fh:
        for path, lbl in rows:
            fh.write(f"{path} {lbl}\n")


def main() -> None:
    print(f"Loading labels from {LABEL_CSV.name}")
    labels = load_labels(LABEL_CSV)
    label_dist = Counter(labels.values())
    print(f"  studies labelled: {len(labels):,}  (neg: {label_dist[0]:,} / pos: {label_dist[1]:,})")

    print(f"\nLoading A4C view manifest: {VIEW_MANIFEST.name}")
    a4c = load_a4c_uris(VIEW_MANIFEST)
    print(f"  A4C OK clips: {len(a4c):,}")

    meta = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "task_type": "classification",
        "num_classes": 2,
        "class_labels": ["alive_at_30d", "dead_within_30d"],
        "target": "mortality_30d",
        "label_source": str(LABEL_CSV.relative_to(REPO_ROOT)),
        "cohort_skeleton": str(SRC_SPLIT_DIR.relative_to(REPO_ROOT)),
        "view_manifest": str(VIEW_MANIFEST),
        "view_filter": sorted(ALLOWED_VIEWS),
        "view_status_filter": "OK",
        "label_distribution_full_cohort": dict(label_dist),
        "splits": {},
    }

    for split in SPLITS:
        src = SRC_SPLIT_DIR / f"{split}.csv"
        print(f"\n[{split}] filtering {src.name}")
        rows, stats = filter_split(src, a4c, labels)
        meta["splits"][split] = stats
        dst = OUT_SPLIT_DIR / f"{split}.csv"
        write_split(rows, dst)
        cps = stats["clips_per_study"]
        print(f"  clips:    {stats['clips_in']:>7,} -> A4C {stats['clips_a4c']:>6,} -> kept {stats['clips_out']:>6,}")
        print(f"  studies:  {stats['studies_in']:>7,} -> kept {stats['studies_out']:>6,}  (unlabelled: {stats['studies_unlabelled']})")
        print(f"  labels:   pos {stats['pos_studies']:>3,} studies / {stats['pos_clips']:>4,} clips"
              f"   neg {stats['neg_studies']:>5,} studies / {stats['neg_clips']:>6,} clips")
        print(f"  clips/study: min={cps['min']} median={cps['median']} max={cps['max']}")
        print(f"  -> {dst}")

    meta_path = OUT_SPLIT_DIR / "label_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"\nWrote metadata: {meta_path}")


if __name__ == "__main__":
    main()
