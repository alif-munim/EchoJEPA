"""Build MIMIC qualitative RV function splits (5-class, UHN-matched scheme) + a binary 10k subset.

Label source: echo_structured_measurement.rv_function (TTE, non-stress), matched to
each echo study within ±1 day, closest measurement wins. MIMIC's free-form result
strings are mapped to UHN's 5-class scheme:
    0 = normal
    1 = low normal
    2 = mildly reduced
    3 = moderately reduced
    4 = severely reduced
Ambiguous/focal categories (`RV not well seen`, `Cannot assess`, `Apical free wall hypo`,
`Basal RV hypo (McConnell's)`, `Hyperdynamic`) are dropped.

View filter: A4C only, view_status == OK.
Splits: patient-level partition reused from disease_hf_v4.1.
Output format: space-delimited "<s3_path> <int_class>".

Two output dirs:
  * rv_function_a4c/            — full 5-class splits (train / val / test)
  * rv_function_binary_10k_a4c/ — binary label (0,1 -> 0; 2,3,4 -> 1). train stratified
                                  subsample of 10,000 clips. val/test scaled by the same
                                  factor (clips_10k / clips_full_train) and stratified.
"""

from __future__ import annotations

import csv
import json
import random
import re
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path("/mnt/custom-file-systems/efs/fs-0049217cdf69186d7_fsap-0fa7145b64eaa046b/vjepa2")
MIMIC_DB = REPO_ROOT / "uhn_echo/nature_medicine/data_exploration/mimic/mimic.db"
SRC_SPLIT_DIR = REPO_ROOT / "experiments/nature_medicine/mimic/probe_csvs/disease_hf_v4.1"
OUT_SPLIT_DIR = REPO_ROOT / "experiments/nature_medicine/mimic/probe_csvs/rv_function_a4c"
SUBSET_DIR = REPO_ROOT / "experiments/nature_medicine/mimic/probe_csvs/rv_function_binary_10k_a4c"
VIEW_MANIFEST = Path("/home/sagemaker-user/user-default-efs/vjepa2/classifier/output/mimic_view_predictions.csv")

ALLOWED_VIEWS = {"A4C"}
MATCH_WINDOW_DAYS = 1
SUBSET_TRAIN_SIZE = 10_000
SUBSET_SEED = 42
SPLITS = ("train", "val", "test")
STUDY_ID_RE = re.compile(r"/s(\d+)/")

CLASS_MAP = {
    "Nl RV function": 0,
    "Low normal function": 1,
    "Mild global RV hypo": 2,
    "Moderate global RV hypo": 3,
    "Severe global hypo": 4,
    "RV function depressed": 4,
}
EXCLUDE_STRINGS = {
    "RV not well seen",
    "Cannot assess RV function",
    "Apical free wall hypo",
    "Basal RV hypo (McConnell's sign)",
    "Hyperdynamic",
}
CLASS_LABELS_5 = ["normal", "low_normal", "mildly_reduced", "moderately_reduced", "severely_reduced"]
# Binary mapping: {0,1} -> 0 (normal), {2,3,4} -> 1 (any dysfunction)
BINARY_MAP = {0: 0, 1: 0, 2: 1, 3: 1, 4: 1}
BINARY_LABELS = ["no_dysfunction", "any_dysfunction"]


def build_study_labels(db_path: Path) -> tuple[dict[str, int], dict]:
    con = sqlite3.connect(str(db_path))
    q = """
    WITH rv AS (
      SELECT subject_id, measurement_id, measurement_datetime, result
      FROM echo_structured_measurement
      WHERE measurement = 'rv_function' AND test_type = 'tte'
        AND result IS NOT NULL AND result != ''
    ),
    matched AS (
      SELECT e.study_id, e.subject_id, rv.result,
             ROW_NUMBER() OVER (
               PARTITION BY e.study_id
               ORDER BY ABS(julianday(e.study_datetime) - julianday(rv.measurement_datetime))
             ) AS rn
      FROM echo_study_list e
      JOIN rv ON e.subject_id = rv.subject_id
      WHERE ABS(julianday(e.study_datetime) - julianday(rv.measurement_datetime)) <= ?
    )
    SELECT study_id, result FROM matched WHERE rn = 1
    """
    raw_counts: Counter = Counter()
    mapped: dict[str, int] = {}
    excluded = 0
    unmapped = 0
    for study_id, result in con.execute(q, (MATCH_WINDOW_DAYS,)):
        raw_counts[result] += 1
        if result in EXCLUDE_STRINGS:
            excluded += 1
            continue
        if result in CLASS_MAP:
            mapped[study_id] = CLASS_MAP[result]
        else:
            unmapped += 1
    con.close()
    stats = {
        "raw_result_counts": dict(raw_counts.most_common()),
        "mapped_studies": len(mapped),
        "excluded_ambiguous_or_focal": excluded,
        "unmapped_other_strings": unmapped,
        "class_distribution": dict(Counter(mapped.values())),
    }
    return mapped, stats


def load_allowed_uris(manifest_path: Path) -> set[str]:
    allowed: set[str] = set()
    with manifest_path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row["view_status"] == "OK" and row["view"] in ALLOWED_VIEWS:
                allowed.add(row["s3_uri"])
    return allowed


def study_id_from_path(path: str) -> str:
    m = STUDY_ID_RE.search(path)
    return m.group(1) if m else ""


def filter_split(src: Path, allowed: set[str], labels: dict[str, int]) -> tuple[list[tuple[str, int]], dict]:
    clips_in = clips_view_filtered = clips_out = 0
    studies_in: set[str] = set()
    studies_out: set[str] = set()
    studies_unlabelled: set[str] = set()
    kept: list[tuple[str, int]] = []
    with src.open("r") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            path, _orig = line.rsplit(" ", 1)
            clips_in += 1
            sid = study_id_from_path(path)
            if sid:
                studies_in.add(sid)
            if path not in allowed:
                continue
            clips_view_filtered += 1
            cls = labels.get(sid)
            if cls is None:
                studies_unlabelled.add(sid)
                continue
            kept.append((path, cls))
            clips_out += 1
            studies_out.add(sid)
    stats = {
        "clips_in": clips_in,
        "clips_view_filtered": clips_view_filtered,
        "clips_out": clips_out,
        "studies_in": len(studies_in),
        "studies_out": len(studies_out),
        "studies_unlabelled": len(studies_unlabelled),
        "class_distribution_clips": dict(Counter(c for _, c in kept)),
        "class_distribution_studies": _class_dist_studies(kept),
        "clips_per_study": _clips_per_study(kept),
    }
    return kept, stats


def _class_dist_studies(rows: list[tuple[str, int]]) -> dict:
    sid_to_class: dict[str, int] = {}
    for path, cls in rows:
        sid = study_id_from_path(path)
        if sid:
            sid_to_class[sid] = cls
    return dict(Counter(sid_to_class.values()))


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
        for path, cls in rows:
            fh.write(f"{path} {cls}\n")


def stratified_subsample(rows: list[tuple[str, int]], n_target: int, rng: random.Random) -> list[tuple[str, int]]:
    """Stratified random sample of ~n_target items, proportional to class counts."""
    by_class: dict[int, list[tuple[str, int]]] = defaultdict(list)
    for r in rows:
        by_class[r[1]].append(r)
    total = len(rows)
    if total == 0 or n_target >= total:
        out = list(rows)
        rng.shuffle(out)
        return out
    allocations: list[tuple[int, int, float]] = []
    for cls, items in by_class.items():
        prop = len(items) / total * n_target
        allocations.append((cls, int(prop), prop - int(prop)))
    remainder = n_target - sum(a[1] for a in allocations)
    allocations_sorted = sorted(allocations, key=lambda x: x[2], reverse=True)
    alloc_map: dict[int, int] = {}
    for i, (cls, floor, _) in enumerate(allocations_sorted):
        alloc_map[cls] = floor + (1 if i < remainder else 0)
    # Guarantee at least 1 per non-empty class if possible without exceeding total
    for cls, items in by_class.items():
        if items and alloc_map[cls] == 0:
            # Borrow from the class with largest surplus
            donor = max(alloc_map, key=lambda c: alloc_map[c] - (1 if c == cls else 0))
            if alloc_map[donor] > 1:
                alloc_map[donor] -= 1
                alloc_map[cls] += 1
    out: list[tuple[str, int]] = []
    for cls, items in by_class.items():
        k = min(alloc_map[cls], len(items))
        if k > 0:
            out.extend(rng.sample(items, k))
    rng.shuffle(out)
    return out


def binarize(rows: list[tuple[str, int]]) -> list[tuple[str, int]]:
    return [(p, BINARY_MAP[c]) for p, c in rows]


def main() -> None:
    print(f"Building labels from {MIMIC_DB.name}")
    labels, label_stats = build_study_labels(MIMIC_DB)
    print(f"  mapped studies: {label_stats['mapped_studies']:,}")
    print(f"  excluded (ambiguous/focal): {label_stats['excluded_ambiguous_or_focal']:,}")
    print(f"  unmapped other strings: {label_stats['unmapped_other_strings']:,}")
    print(f"  class distribution (studies): {label_stats['class_distribution']}")

    print(f"\nLoading view manifest: {VIEW_MANIFEST.name}")
    allowed = load_allowed_uris(VIEW_MANIFEST)
    print(f"  allowed views ({sorted(ALLOWED_VIEWS)}) OK clips: {len(allowed):,}")

    meta = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "task_type": "classification",
        "num_classes": 5,
        "class_labels": CLASS_LABELS_5,
        "label_source": "echo_structured_measurement.rv_function (TTE, non-stress)",
        "source_db": str(MIMIC_DB.relative_to(REPO_ROOT)),
        "class_map": CLASS_MAP,
        "excluded_strings": sorted(EXCLUDE_STRINGS),
        "match_window_days": MATCH_WINDOW_DAYS,
        "cohort_skeleton": str(SRC_SPLIT_DIR.relative_to(REPO_ROOT)),
        "view_manifest": str(VIEW_MANIFEST),
        "view_filter": sorted(ALLOWED_VIEWS),
        "view_status_filter": "OK",
        "raw_result_counts": label_stats["raw_result_counts"],
        "splits": {},
    }

    split_rows: dict[str, list[tuple[str, int]]] = {}
    for split in SPLITS:
        src = SRC_SPLIT_DIR / f"{split}.csv"
        print(f"\n[{split}] filtering {src.name}")
        rows, stats = filter_split(src, allowed, labels)
        split_rows[split] = rows
        meta["splits"][split] = stats
        dst = OUT_SPLIT_DIR / f"{split}.csv"
        write_split(rows, dst)
        cps = stats["clips_per_study"]
        print(f"  clips:    {stats['clips_in']:>7,} -> view {stats['clips_view_filtered']:>6,} -> kept {stats['clips_out']:>6,}")
        print(f"  studies:  {stats['studies_in']:>7,} -> kept {stats['studies_out']:>6,}  (unlabelled: {stats['studies_unlabelled']})")
        print(f"  class dist (studies): {stats['class_distribution_studies']}")
        print(f"  clips/study: min={cps['min']} median={cps['median']} max={cps['max']}")
        print(f"  -> {dst}")

    meta_path = OUT_SPLIT_DIR / "label_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"\nWrote metadata: {meta_path}")

    # ---- Binary 10k subset ----
    print(f"\n=== Building binary 10k subset (train -> {SUBSET_TRAIN_SIZE:,} clips, val/test scaled) ===")
    train_full = split_rows["train"]
    train_full_bin = binarize(train_full)
    full_train_clips = len(train_full_bin)
    if full_train_clips == 0:
        raise RuntimeError("Full train split is empty; cannot build subset.")
    scale = SUBSET_TRAIN_SIZE / full_train_clips
    print(f"  scale factor (subset/full) = {scale:.4f}  ({SUBSET_TRAIN_SIZE:,} / {full_train_clips:,})")

    rng = random.Random(SUBSET_SEED)
    subset_train = stratified_subsample(train_full_bin, SUBSET_TRAIN_SIZE, rng)

    val_full_bin = binarize(split_rows["val"])
    test_full_bin = binarize(split_rows["test"])
    val_target = max(1, round(len(val_full_bin) * scale))
    test_target = max(1, round(len(test_full_bin) * scale))
    subset_val = stratified_subsample(val_full_bin, val_target, rng)
    subset_test = stratified_subsample(test_full_bin, test_target, rng)

    for name, rows in (("train", subset_train), ("val", subset_val), ("test", subset_test)):
        dst = SUBSET_DIR / f"{name}.csv"
        write_split(rows, dst)
        dist = dict(Counter(c for _, c in rows))
        print(f"  [{name}] {len(rows):,} clips   class dist: {dist}")
        print(f"       -> {dst}")

    subset_meta = {
        **meta,
        "task_type": "classification",
        "num_classes": 2,
        "class_labels": BINARY_LABELS,
        "binary_map_from_5class": {str(k): v for k, v in BINARY_MAP.items()},
        "subset_of": str(OUT_SPLIT_DIR.relative_to(REPO_ROOT)),
        "subset_train_size_target": SUBSET_TRAIN_SIZE,
        "subset_train_size_actual": len(subset_train),
        "subset_val_size_actual": len(subset_val),
        "subset_test_size_actual": len(subset_test),
        "subset_scale_factor": scale,
        "subset_seed": SUBSET_SEED,
        "subset_method": "stratified random sample by binary class, proportional allocation; val/test scaled by train subset factor",
        "subset_class_distribution_clips": {
            "train": dict(Counter(c for _, c in subset_train)),
            "val": dict(Counter(c for _, c in subset_val)),
            "test": dict(Counter(c for _, c in subset_test)),
        },
    }
    (SUBSET_DIR / "label_meta.json").write_text(json.dumps(subset_meta, indent=2))
    print(f"\n  -> {SUBSET_DIR}")


if __name__ == "__main__":
    main()
