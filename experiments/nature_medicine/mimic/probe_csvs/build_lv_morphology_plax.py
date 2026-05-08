"""Build PLAX-only train/val/test splits for LV morphology regression on MIMIC echos.

Four parasternal M-mode-style measurements (all in cm, all measured in PLAX):
  * septal_thickness    — IVS (interventricular septum) thickness
  * inf_lat_thickness   — LVPW (left ventricular posterior wall / inferolateral wall) thickness
  * lvedd               — LVIDd (LV end-diastolic internal dimension)
  * lvesd               — LVIDs (LV end-systolic internal dimension)

Label source: echo_structured_measurement (TTE, closest measurement within ±1 day,
closest wins), then filter to PLAX clips using the ConvNeXt view manifest with
view_status == OK. Patient-level partition from disease_hf_v4.1.

Also emits a 10k-clip stratified subset per task (seed=42, val/test scaled by same
factor). Subset: quantile-stratified on label (10 bins) so the tail isn't
under-represented.

Output per task:
  <task>_plax/{train,val,test}.csv + zscore_params.json + label_meta.json
  <task>_plax_10k/{train,val,test}.csv + label_meta.json
"""

from __future__ import annotations

import csv
import json
import math
import random
import re
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path("/mnt/custom-file-systems/efs/fs-0049217cdf69186d7_fsap-0fa7145b64eaa046b/vjepa2")
MIMIC_DB = REPO_ROOT / "uhn_echo/nature_medicine/data_exploration/mimic/mimic.db"
SRC_SPLIT_DIR = REPO_ROOT / "experiments/nature_medicine/mimic/probe_csvs/disease_hf_v4.1"
VIEW_MANIFEST = Path("/home/sagemaker-user/user-default-efs/vjepa2/classifier/output/mimic_view_predictions.csv")

ALLOWED_VIEWS = {"PLAX"}
MATCH_WINDOW_DAYS = 1
SUBSET_TRAIN_SIZE = 10_000
SUBSET_SEED = 42
SUBSET_BINS = 10
SPLITS = ("train", "val", "test")
STUDY_ID_RE = re.compile(r"/s(\d+)/")

TASKS = {
    # name: (physiological min, physiological max) in cm
    "septal_thickness":   (0.4, 3.0),
    "inf_lat_thickness":  (0.4, 3.0),
    "lvedd":              (2.0, 8.5),
    "lvesd":              (1.0, 7.5),
}
DISPLAY_NAMES = {
    "septal_thickness":   "IVS thickness (diastolic)",
    "inf_lat_thickness":  "LVPW / inferolateral wall thickness (diastolic)",
    "lvedd":              "LVIDd (LV end-diastolic internal dimension)",
    "lvesd":              "LVIDs (LV end-systolic internal dimension)",
}


def build_label_map(task: str, value_range: tuple[float, float]) -> dict[str, float]:
    """{study_id: value} for the closest match within ±1 day.

    Fetches measurements and studies separately (indexed queries), then does the
    per-study closest-match matching in Python to avoid expensive SQL window
    functions on the 28M-row structured_measurement table.
    """
    lo, hi = value_range
    con = sqlite3.connect(str(MIMIC_DB))

    # Index-friendly: (test_type='tte', measurement=task) hits idx_esm_test_meas
    meas_rows = con.execute(
        "SELECT subject_id, measurement_datetime, CAST(result AS REAL) "
        "FROM echo_structured_measurement "
        "WHERE test_type='tte' AND measurement = ? AND result IS NOT NULL AND result != ''",
        (task,),
    ).fetchall()

    study_rows = con.execute(
        "SELECT study_id, subject_id, study_datetime FROM echo_study_list"
    ).fetchall()
    con.close()

    # Index measurements by subject
    import datetime as _dt
    def parse_dt(s: str) -> float:
        # Fallback-friendly parse; return seconds-since-epoch
        try:
            return _dt.datetime.fromisoformat(s).timestamp()
        except Exception:
            return float("nan")

    by_subject: dict[str, list[tuple[float, float]]] = {}
    for subj, mdt, val in meas_rows:
        try:
            v = float(val)
        except (TypeError, ValueError):
            continue
        if not (lo <= v <= hi):
            continue
        t = parse_dt(mdt)
        if t != t:  # nan
            continue
        by_subject.setdefault(subj, []).append((t, v))

    out: dict[str, float] = {}
    window_sec = MATCH_WINDOW_DAYS * 86400.0
    for study_id, subj, sdt in study_rows:
        ms = by_subject.get(subj)
        if not ms:
            continue
        t0 = parse_dt(sdt)
        if t0 != t0:
            continue
        # Find measurement with smallest |t - t0| within window
        best_dt = None; best_val = None
        for t, v in ms:
            d = abs(t - t0)
            if d > window_sec:
                continue
            if best_dt is None or d < best_dt:
                best_dt = d; best_val = v
        if best_val is not None:
            out[study_id] = best_val
    return out


def load_plax_uris(manifest_path: Path) -> set[str]:
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


def filter_split(src: Path, plax: set[str], labels: dict[str, float]) -> tuple[list[tuple[str, float]], dict]:
    clips_in = clips_plax = clips_out = 0
    studies_in: set[str] = set()
    studies_out: set[str] = set()
    studies_unlabelled: set[str] = set()
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
            if path not in plax:
                continue
            clips_plax += 1
            v = labels.get(sid)
            if v is None:
                studies_unlabelled.add(sid)
                continue
            kept.append((path, v))
            clips_out += 1
            studies_out.add(sid)
    stats = {
        "clips_in": clips_in,
        "clips_plax": clips_plax,
        "clips_out": clips_out,
        "studies_in": len(studies_in),
        "studies_out": len(studies_out),
        "studies_unlabelled": len(studies_unlabelled),
        "label_stats": _numeric_stats([v for _, v in kept]),
        "clips_per_study": _clips_per_study(kept),
    }
    return kept, stats


def _numeric_stats(vals: list[float]) -> dict:
    if not vals:
        return {"n": 0}
    s = sorted(vals)
    n = len(s)
    mean = sum(s) / n
    std = (sum((v - mean) ** 2 for v in s) / n) ** 0.5
    return {
        "n": n,
        "min": round(s[0], 3),
        "p5": round(s[n // 20], 3) if n >= 20 else round(s[0], 3),
        "p25": round(s[n // 4], 3),
        "median": round(s[n // 2], 3),
        "p75": round(s[3 * n // 4], 3),
        "p95": round(s[19 * n // 20], 3) if n >= 20 else round(s[-1], 3),
        "max": round(s[-1], 3),
        "mean": round(mean, 4),
        "std": round(std, 4),
    }


def _clips_per_study(rows: list[tuple[str, float]]) -> dict:
    c: dict[str, int] = defaultdict(int)
    for p, _ in rows:
        sid = study_id_from_path(p)
        if sid:
            c[sid] += 1
    vals = sorted(c.values())
    if not vals:
        return {"min": 0, "median": 0, "max": 0, "mean": 0.0}
    n = len(vals)
    median = vals[n // 2] if n % 2 == 1 else (vals[n // 2 - 1] + vals[n // 2]) / 2
    return {"min": vals[0], "median": median, "max": vals[-1], "mean": round(sum(vals) / n, 2)}


def write_rows(rows: list[tuple[str, float]], dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w") as fh:
        for p, v in rows:
            fh.write(f"{p} {v:.6f}\n")


def quantile_stratified_subsample(rows: list[tuple[str, float]], target: int, n_bins: int, rng: random.Random) -> list[tuple[str, float]]:
    if not rows or target >= len(rows):
        out = list(rows)
        rng.shuffle(out)
        return out
    vals = sorted(v for _, v in rows)
    n = len(vals)
    raw_edges = [vals[int(i * n / n_bins)] for i in range(n_bins)]
    raw_edges.append(vals[-1] + 1e-6)
    edges: list[float] = [raw_edges[0]]
    for v in raw_edges[1:]:
        if v > edges[-1]:
            edges.append(v)
    effective_bins = len(edges) - 1
    edges[-1] += 1e-6

    def bin_of(x: float) -> int:
        for i in range(effective_bins):
            if edges[i] <= x < edges[i + 1]:
                return i
        return effective_bins - 1

    by_bin: dict[int, list[tuple[str, float]]] = defaultdict(list)
    for r in rows:
        by_bin[bin_of(r[1])].append(r)

    base = target // effective_bins
    caps = {b: base for b in range(effective_bins)}
    remainder = target - base * effective_bins
    size_ord = sorted(range(effective_bins), key=lambda b: -len(by_bin[b]))
    for i in range(remainder):
        caps[size_ord[i % effective_bins]] += 1

    out: list[tuple[str, float]] = []
    shortfall = 0
    for b in range(effective_bins):
        items = by_bin[b]
        cap = caps[b]
        if len(items) <= cap:
            out.extend(items)
            shortfall += cap - len(items)
        else:
            out.extend(rng.sample(items, cap))
    if shortfall > 0:
        # redistribute to bins with headroom
        for b in sorted(range(effective_bins), key=lambda b: len(by_bin[b]), reverse=True):
            already = sum(1 for r in out if bin_of(r[1]) == b)
            avail = len(by_bin[b]) - already
            if avail <= 0 or shortfall <= 0:
                continue
            take = min(avail, shortfall)
            # pick items not already in out
            chosen_ids = {id(r) for r in out}
            pool = [r for r in by_bin[b] if id(r) not in chosen_ids]
            out.extend(rng.sample(pool, take))
            shortfall -= take

    rng.shuffle(out)
    return out


def main() -> None:
    print(f"Loading PLAX view manifest: {VIEW_MANIFEST.name}")
    plax = load_plax_uris(VIEW_MANIFEST)
    print(f"  PLAX OK clips: {len(plax):,}")

    for task, (lo, hi) in TASKS.items():
        print(f"\n========== {task} ({DISPLAY_NAMES[task]}) ==========")
        labels = build_label_map(task, (lo, hi))
        print(f"  labelled studies: {len(labels):,}  value_range=[{lo}, {hi}] cm")
        if labels:
            lvals = sorted(labels.values())
            print(f"  label stats: min={lvals[0]:.2f} p25={lvals[len(lvals)//4]:.2f} median={lvals[len(lvals)//2]:.2f} p75={lvals[3*len(lvals)//4]:.2f} max={lvals[-1]:.2f}")

        out_dir = REPO_ROOT / "experiments/nature_medicine/mimic/probe_csvs" / f"{task}_plax"
        subset_dir = REPO_ROOT / "experiments/nature_medicine/mimic/probe_csvs" / f"{task}_plax_10k"

        meta = {
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "task_type": "regression",
            "target": task,
            "display_name": DISPLAY_NAMES[task],
            "label_source": "echo_structured_measurement",
            "test_type": "tte",
            "matching_window_days": MATCH_WINDOW_DAYS,
            "value_range": [lo, hi],
            "unit": "cm",
            "view_manifest": str(VIEW_MANIFEST),
            "view_filter": sorted(ALLOWED_VIEWS),
            "view_status_filter": "OK",
            "cohort_skeleton": str(SRC_SPLIT_DIR.relative_to(REPO_ROOT)),
            "splits": {},
        }
        split_rows: dict[str, list[tuple[str, float]]] = {}
        for split in SPLITS:
            src = SRC_SPLIT_DIR / f"{split}.csv"
            rows, stats = filter_split(src, plax, labels)
            split_rows[split] = rows
            meta["splits"][split] = stats
            dst = out_dir / f"{split}.csv"
            write_rows(rows, dst)
            ls = stats["label_stats"]; cps = stats["clips_per_study"]
            print(f"  [{split:5s}] {stats['clips_in']:>7,} -> PLAX {stats['clips_plax']:>6,} -> kept {stats['clips_out']:>6,}  "
                  f"| studies {stats['studies_in']:>6,} -> {stats['studies_out']:>6,}  "
                  f"| mean={ls.get('mean','?')} std={ls.get('std','?')}  "
                  f"| clips/study med={cps['median']}  -> {dst.name}")

        # z-score from PLAX-filtered train
        train_vals = [v for _, v in split_rows["train"]]
        if train_vals:
            mean = sum(train_vals) / len(train_vals)
            std = (sum((v - mean) ** 2 for v in train_vals) / len(train_vals)) ** 0.5
        else:
            mean, std = 0.0, 1.0
        (out_dir / "zscore_params.json").write_text(json.dumps({"target_mean": mean, "target_std": std}))
        meta["zscore_params"] = {"target_mean": mean, "target_std": std, "computed_from": "PLAX-filtered train clips"}
        (out_dir / "label_meta.json").write_text(json.dumps(meta, indent=2))
        print(f"  zscore: mean={mean:.6f}  std={std:.6f}  -> {out_dir.name}/zscore_params.json")

        # 10k subset (quantile-stratified by label)
        rng = random.Random(SUBSET_SEED)
        full_train_n = len(split_rows["train"])
        if full_train_n == 0:
            print(f"  SKIP 10k subset: empty train")
            continue
        subset_train = quantile_stratified_subsample(split_rows["train"], SUBSET_TRAIN_SIZE, SUBSET_BINS, rng)
        scale = len(subset_train) / full_train_n
        val_target = max(1, round(len(split_rows["val"]) * scale))
        test_target = max(1, round(len(split_rows["test"]) * scale))
        subset_val = quantile_stratified_subsample(split_rows["val"], val_target, SUBSET_BINS, rng)
        subset_test = quantile_stratified_subsample(split_rows["test"], test_target, SUBSET_BINS, rng)

        for split, rs in (("train", subset_train), ("val", subset_val), ("test", subset_test)):
            dst = subset_dir / f"{split}.csv"
            write_rows(rs, dst)
        # subset uses SAME zscore as full parent (same distribution, consistency for probe training)
        (subset_dir / "zscore_params.json").write_text(json.dumps({"target_mean": mean, "target_std": std}))
        subset_meta = {
            **meta,
            "subset_of": str(out_dir.relative_to(REPO_ROOT)),
            "subset_train_size_target": SUBSET_TRAIN_SIZE,
            "subset_train_size_actual": len(subset_train),
            "subset_scale_factor": scale,
            "subset_seed": SUBSET_SEED,
            "subset_method": "quantile-stratified on label (10 bins), proportional allocation",
            "subset_val_size_actual": len(subset_val),
            "subset_test_size_actual": len(subset_test),
        }
        (subset_dir / "label_meta.json").write_text(json.dumps(subset_meta, indent=2))
        print(f"  10k subset: train {len(subset_train):,}  val {len(subset_val):,}  test {len(subset_test):,}  "
              f"(scale={scale:.4f})  -> {subset_dir.name}/")


if __name__ == "__main__":
    main()
