"""Build a 10k-clip subset of mortality_30d_a4c.

Train: stratified random sample by label (proportional allocation) to 10,000 clips.
Val/test: scaled by the same factor (10,000 / full_train_clips) and stratified.

Matches the approach used for rv_function_binary_10k_a4c: class-proportional
stratification on clip labels, seed=42 for reproducibility. Consistent with the
~10% sampling rate applied to the full A4C mortality splits.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path("/mnt/custom-file-systems/efs/fs-0049217cdf69186d7_fsap-0fa7145b64eaa046b/vjepa2")
SRC_DIR = REPO_ROOT / "experiments/nature_medicine/mimic/probe_csvs/mortality_30d_a4c"
OUT_DIR = REPO_ROOT / "experiments/nature_medicine/mimic/probe_csvs/mortality_30d_a4c_10k"

TRAIN_TARGET = 10_000
SEED = 42
SPLITS = ("train", "val", "test")
STUDY_ID_RE = re.compile(r"/s(\d+)/")


def read_split(path: Path) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    with path.open("r") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            p, l = line.rsplit(" ", 1)
            rows.append((p, int(l)))
    return rows


def write_split(rows: list[tuple[str, int]], dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w") as fh:
        for p, l in rows:
            fh.write(f"{p} {l}\n")


def stratified_subsample(rows: list[tuple[str, int]], n_target: int, rng: random.Random) -> list[tuple[str, int]]:
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
    for cls, items in by_class.items():
        if items and alloc_map[cls] == 0:
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


def study_id_from_path(path: str) -> str:
    m = STUDY_ID_RE.search(path)
    return m.group(1) if m else ""


def _study_counts(rows: list[tuple[str, int]]) -> dict:
    sid_to_class: dict[str, int] = {}
    for p, c in rows:
        sid = study_id_from_path(p)
        if sid:
            sid_to_class[sid] = c
    return {
        "n_studies": len(sid_to_class),
        "n_pos_studies": sum(1 for c in sid_to_class.values() if c == 1),
        "n_neg_studies": sum(1 for c in sid_to_class.values() if c == 0),
    }


def main() -> None:
    train_full = read_split(SRC_DIR / "train.csv")
    val_full = read_split(SRC_DIR / "val.csv")
    test_full = read_split(SRC_DIR / "test.csv")
    print(f"Source clip counts: train={len(train_full):,}  val={len(val_full):,}  test={len(test_full):,}")
    print(f"Train class dist: {dict(Counter(c for _, c in train_full))}")

    scale = TRAIN_TARGET / len(train_full)
    val_target = max(1, round(len(val_full) * scale))
    test_target = max(1, round(len(test_full) * scale))
    print(f"Scale factor = {scale:.4f}  (train {TRAIN_TARGET:,} / full {len(train_full):,})")
    print(f"Targets: val={val_target:,}  test={test_target:,}")

    rng = random.Random(SEED)
    subsets = {
        "train": stratified_subsample(train_full, TRAIN_TARGET, rng),
        "val": stratified_subsample(val_full, val_target, rng),
        "test": stratified_subsample(test_full, test_target, rng),
    }

    meta = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "task_type": "classification",
        "num_classes": 2,
        "class_labels": ["alive_at_30d", "dead_within_30d"],
        "subset_of": str(SRC_DIR.relative_to(REPO_ROOT)),
        "seed": SEED,
        "train_target": TRAIN_TARGET,
        "scale_factor": scale,
        "method": "stratified random sample by binary class, proportional allocation; val/test scaled by same factor",
        "splits": {},
    }

    for split, rows in subsets.items():
        dst = OUT_DIR / f"{split}.csv"
        write_split(rows, dst)
        clip_dist = dict(Counter(c for _, c in rows))
        studies = _study_counts(rows)
        meta["splits"][split] = {
            "clips": len(rows),
            "class_distribution_clips": clip_dist,
            **studies,
        }
        print(f"  [{split}] {len(rows):,} clips  class dist: {clip_dist}  "
              f"studies: {studies['n_studies']:,} ({studies['n_pos_studies']} pos)")
        print(f"       -> {dst}")

    (OUT_DIR / "label_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nWrote metadata: {OUT_DIR / 'label_meta.json'}")


if __name__ == "__main__":
    main()
