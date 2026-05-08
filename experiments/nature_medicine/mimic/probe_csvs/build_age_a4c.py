"""Build A4C-only train/val/test splits for age regression on MIMIC echos.

Label: age_at_echo (years) from demographics_fairness.csv (available for all 7,243 studies).
Cohort skeleton: disease_hf_v4.1 splits (patient-level partition covering all 4,579 subjects).
View filter: A4C clips with view_status == OK from the ConvNeXt view manifest.
Z-score params: computed on the A4C-filtered train split and written to zscore_params.json.

Output format: space-delimited "<path> <raw_age_float>" (matches other regression tasks).
"""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path("/mnt/custom-file-systems/efs/fs-0049217cdf69186d7_fsap-0fa7145b64eaa046b/vjepa2")
DEMOGRAPHICS_CSV = REPO_ROOT / "uhn_echo/nature_medicine/data_exploration/mimic/csv/demographics_fairness.csv"
SRC_SPLIT_DIR = REPO_ROOT / "experiments/nature_medicine/mimic/probe_csvs/disease_hf_v4.1"
OUT_SPLIT_DIR = REPO_ROOT / "experiments/nature_medicine/mimic/probe_csvs/age_a4c"
VIEW_MANIFEST = Path("/home/sagemaker-user/user-default-efs/vjepa2/classifier/output/mimic_view_predictions.csv")

ALLOWED_VIEWS = {"A4C"}
SPLITS = ("train", "val", "test")
STUDY_ID_RE = re.compile(r"/s(\d+)/")


def load_age_by_study(path: Path) -> dict[str, float]:
    ages: dict[str, float] = {}
    with path.open("r", newline="") as fh:
        r = csv.DictReader(fh)
        for row in r:
            sid = row["study_id"].strip()
            v = row["age_at_echo"].strip()
            if sid and v:
                try:
                    ages[sid] = float(v)
                except ValueError:
                    continue
    return ages


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


def filter_split(src: Path, a4c: set[str], ages: dict[str, float]) -> tuple[list[tuple[str, float]], dict]:
    clips_in = 0
    clips_a4c = 0
    clips_out = 0
    studies_in: set[str] = set()
    studies_out: set[str] = set()
    studies_no_age: set[str] = set()
    kept: list[tuple[str, float]] = []

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
            age = ages.get(sid)
            if age is None:
                studies_no_age.add(sid)
                continue
            kept.append((path, age))
            clips_out += 1
            studies_out.add(sid)

    ages_kept = [a for _, a in kept]
    stats = {
        "clips_in": clips_in,
        "clips_a4c": clips_a4c,
        "clips_out": clips_out,
        "studies_in": len(studies_in),
        "studies_out": len(studies_out),
        "studies_missing_age": len(studies_no_age),
        "age_stats": _age_stats(ages_kept),
        "clips_per_study": _clips_per_study(kept),
    }
    return kept, stats


def _age_stats(vals: list[float]) -> dict:
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
    print(f"Loading ages from {DEMOGRAPHICS_CSV.name}")
    ages = load_age_by_study(DEMOGRAPHICS_CSV)
    print(f"  studies with age: {len(ages):,}")

    print(f"\nLoading A4C view manifest: {VIEW_MANIFEST.name}")
    a4c = load_a4c_uris(VIEW_MANIFEST)
    print(f"  A4C OK clips: {len(a4c):,}")

    meta = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "task_type": "regression",
        "target": "age_at_echo_years",
        "label_source": str(DEMOGRAPHICS_CSV.relative_to(REPO_ROOT)),
        "cohort_skeleton": str(SRC_SPLIT_DIR.relative_to(REPO_ROOT)),
        "view_manifest": str(VIEW_MANIFEST),
        "view_filter": sorted(ALLOWED_VIEWS),
        "view_status_filter": "OK",
        "splits": {},
    }

    split_rows: dict[str, list[tuple[str, float]]] = {}
    for split in SPLITS:
        src = SRC_SPLIT_DIR / f"{split}.csv"
        print(f"\n[{split}] filtering {src.name}")
        rows, stats = filter_split(src, a4c, ages)
        split_rows[split] = rows
        meta["splits"][split] = stats
        dst = OUT_SPLIT_DIR / f"{split}.csv"
        write_split(rows, dst)
        a = stats["age_stats"]
        cps = stats["clips_per_study"]
        print(f"  clips:    {stats['clips_in']:>7,} -> A4C {stats['clips_a4c']:>6,} -> kept {stats['clips_out']:>6,}")
        print(f"  studies:  {stats['studies_in']:>7,} -> kept {stats['studies_out']:>6,}  (missing age: {stats['studies_missing_age']})")
        print(f"  age:      mean={a['mean']} std={a['std']} median={a['median']}  range=[{a['min']}, {a['max']}]")
        print(f"  clips/study: min={cps['min']} median={cps['median']} max={cps['max']}")
        print(f"  -> {dst}")

    train_ages = [a for _, a in split_rows["train"]]
    n = len(train_ages)
    mean = sum(train_ages) / n
    std = (sum((v - mean) ** 2 for v in train_ages) / n) ** 0.5
    zscore = {"target_mean": mean, "target_std": std}
    zscore_path = OUT_SPLIT_DIR / "zscore_params.json"
    with zscore_path.open("w") as fh:
        json.dump(zscore, fh)
    print(f"\nZ-score (computed from train split, clip-level):")
    print(f"  target_mean = {mean:.10f}")
    print(f"  target_std  = {std:.10f}")
    print(f"  -> {zscore_path}")

    meta["zscore_params"] = {"target_mean": mean, "target_std": std, "computed_from": "train clips"}
    meta_path = OUT_SPLIT_DIR / "label_meta.json"
    with meta_path.open("w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"Wrote metadata: {meta_path}")


if __name__ == "__main__":
    main()
