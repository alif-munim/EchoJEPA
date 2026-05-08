"""Trim A4C train splits down to ~10k clips for matched-budget probes.

Reference baseline: EchoNet-Dynamic LVEF train ~= 9,867 clips, MIMIC
RVSP/MR 10k variants. Val and test splits are untouched.

Strategies:
  - Classification (disease_hcm_a4c, disease_hf_incident_1yr_a4c):
    keep ALL positive-study clips (positives are load-bearing);
    downsample negative studies to fill the clip budget; cap clips
    per study to preserve prediction-averaging structure while
    avoiding over-represented high-clip studies.
  - Regression (age_a4c, lvef_a4c):
    label-decile stratified sampling. Allocate TARGET/10 clips per
    decile, preserving every clip in deciles with fewer than the
    per-decile quota (tail coverage).

Per-study cap: classification tasks use a cap of 4 clips/study post-
downsample; regression tasks use 6 clips/study (lvef_a4c starts with
median ~6, so higher cap is fine).

Outputs per task:
  <task>_a4c_10k/
    train.csv         — ~10k-clip trimmed train split
    trim_meta.json    — provenance, strategy, per-stratum counts

Val and test CSVs are NOT copied (callers should reference the full
<task>_a4c/ val.csv and test.csv; the trim is strictly a train-side
efficiency choice).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
from collections import defaultdict
from pathlib import Path


STUDY_RE = re.compile(r"/s(\d+)/")
TARGET = 10_000
CLIP_CAP_CLS = 4
CLIP_CAP_REG = 6
SEED = 42


def study_of(path: str) -> str:
    m = STUDY_RE.search(path)
    return m.group(1) if m else path  # fallback: whole path as key


def load_split(csv_path: Path) -> list[tuple[str, str]]:
    """Parse space-delimited `<path> <label>` → list of (path, label_str)."""
    rows = []
    with csv_path.open() as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            idx = line.rfind(" ")
            if idx < 0:
                raise ValueError(f"malformed line (no space): {line!r}")
            rows.append((line[:idx], line[idx + 1 :]))
    return rows


def write_split(out_path: Path, rows: list[tuple[str, str]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        for path, label in rows:
            fh.write(f"{path} {label}\n")


def trim_classification(
    rows: list[tuple[str, str]],
    target: int,
    clip_cap: int,
    seed: int,
) -> tuple[list[tuple[str, str]], dict]:
    """Keep all positive-study clips, downsample negative studies."""
    rng = random.Random(seed)
    per_study_clips: dict[str, list[tuple[str, str]]] = defaultdict(list)
    study_label: dict[str, int] = {}
    for path, label in rows:
        sid = study_of(path)
        per_study_clips[sid].append((path, label))
        # Study-level label = max over clips (any positive clip → positive study).
        lbl = int(float(label))
        study_label[sid] = max(study_label.get(sid, 0), lbl)

    pos_studies = [s for s, l in study_label.items() if l == 1]
    neg_studies = [s for s, l in study_label.items() if l == 0]

    # Keep all positives (cap clips per study).
    pos_out: list[tuple[str, str]] = []
    for s in pos_studies:
        clips = per_study_clips[s]
        if len(clips) > clip_cap:
            clips = rng.sample(clips, clip_cap)
        pos_out.extend(clips)

    # Budget for negatives.
    neg_budget = target - len(pos_out)
    if neg_budget <= 0:
        # Positives alone exceed target (unlikely but handle). Just return positives.
        return pos_out, {
            "strategy": "classification",
            "target": target,
            "clip_cap": clip_cap,
            "pos_studies_kept": len(pos_studies),
            "neg_studies_kept": 0,
            "pos_clips_kept": len(pos_out),
            "neg_clips_kept": 0,
            "total_clips": len(pos_out),
            "note": "positives exceeded target; returning positives only",
        }

    # Sample enough neg studies to cover budget assuming ~clip_cap clips/study.
    est_studies_needed = math.ceil(neg_budget / clip_cap)
    # Sample more than needed to avoid running out; we'll trim clips at the end.
    neg_sample_size = min(len(neg_studies), int(est_studies_needed * 1.2) + 50)
    rng.shuffle(neg_studies)
    neg_candidates = neg_studies[:neg_sample_size]

    # Collect clips round-robin, capping per-study.
    neg_out: list[tuple[str, str]] = []
    for s in neg_candidates:
        clips = per_study_clips[s]
        if len(clips) > clip_cap:
            clips = rng.sample(clips, clip_cap)
        neg_out.extend(clips)
        if len(neg_out) >= neg_budget:
            break

    # Fine trim if over budget.
    if len(neg_out) > neg_budget:
        neg_out = rng.sample(neg_out, neg_budget)

    out = pos_out + neg_out
    rng.shuffle(out)  # so training doesn't see class-sorted order
    return out, {
        "strategy": "classification_study_stratified",
        "target": target,
        "clip_cap": clip_cap,
        "pos_studies_kept": len(pos_studies),
        "neg_studies_kept": sum(
            1 for s in neg_candidates if any(cp in per_study_clips[s] for cp in neg_out)
        ),
        "pos_clips_kept": len(pos_out),
        "neg_clips_kept": len(neg_out),
        "total_clips": len(out),
    }


def trim_regression(
    rows: list[tuple[str, str]],
    target: int,
    clip_cap: int,
    seed: int,
    n_bins: int = 10,
) -> tuple[list[tuple[str, str]], dict]:
    """Label-stratified sampling on the continuous label.

    Handles discrete-ish labels (e.g. LVEF clusters at 55/60/65) by
    binning on unique quantile edges and reallocating skipped budget to
    the remaining bins.
    """
    rng = random.Random(seed)

    # Per-study label.
    per_study_clips: dict[str, list[tuple[str, str]]] = defaultdict(list)
    study_label: dict[str, float] = {}
    for path, label in rows:
        sid = study_of(path)
        per_study_clips[sid].append((path, label))
        study_label[sid] = float(label)

    # Compute bin edges on study-level labels using quantile positions,
    # then dedupe adjacent ties so we don't produce zero-width bins
    # (degenerate for discrete labels like LVEF at 55, 60, 65 ...).
    vals = sorted(study_label.values())
    n = len(vals)
    if n < n_bins:
        n_bins = max(1, n)
    raw_edges = [vals[int(i * n / n_bins)] for i in range(n_bins)]
    raw_edges.append(vals[-1] + 1e-6)

    # Dedupe: merge adjacent equal edges.
    bin_edges: list[float] = [raw_edges[0]]
    for v in raw_edges[1:]:
        if v > bin_edges[-1]:
            bin_edges.append(v)
    effective_bins = len(bin_edges) - 1
    # Ensure open upper bound.
    bin_edges[-1] += 1e-6

    def bin_of(x: float) -> int:
        for i in range(effective_bins):
            if bin_edges[i] <= x < bin_edges[i + 1]:
                return i
        return effective_bins - 1

    studies_by_bin: dict[int, list[str]] = defaultdict(list)
    for s, v in study_label.items():
        studies_by_bin[bin_of(v)].append(s)

    # Two-pass allocation: first pass hands each bin an equal slice;
    # any bin that can't spend its slice gives the remainder to a
    # second-pass re-allocation across the other bins.
    base_budget = target // effective_bins
    per_bin_caps = {b: base_budget for b in range(effective_bins)}
    remainder = target - base_budget * effective_bins
    # Distribute remainder to the largest-bins so big ones absorb it.
    bins_by_size = sorted(range(effective_bins), key=lambda b: -len(studies_by_bin[b]))
    for i in range(remainder):
        per_bin_caps[bins_by_size[i % effective_bins]] += 1

    def sample_from_bin(b: int, cap: int) -> list[tuple[str, str]]:
        d_studies = list(studies_by_bin[b])
        rng.shuffle(d_studies)
        d_clips: list[tuple[str, str]] = []
        for s in d_studies:
            clips = per_study_clips[s]
            if len(clips) > clip_cap:
                clips = rng.sample(clips, clip_cap)
            d_clips.extend(clips)
            if len(d_clips) >= cap:
                break
        if len(d_clips) > cap:
            d_clips = rng.sample(d_clips, cap)
        return d_clips

    # Pass 1: try to fill each bin's base budget; track shortfall.
    pass1: dict[int, list[tuple[str, str]]] = {}
    shortfalls: dict[int, int] = {}
    leftover_per_bin: dict[int, int] = {}
    for b in range(effective_bins):
        cap = per_bin_caps[b]
        clips = sample_from_bin(b, cap)
        pass1[b] = clips
        shortfalls[b] = max(0, cap - len(clips))
        # Max headroom a bin can still absorb (all clips from all studies, capped).
        max_bin_clips = sum(
            min(len(per_study_clips[s]), clip_cap) for s in studies_by_bin[b]
        )
        leftover_per_bin[b] = max_bin_clips - len(clips)

    # Redistribute shortfall to bins with headroom.
    total_shortfall = sum(shortfalls.values())
    if total_shortfall > 0:
        # Rank bins by remaining headroom, biggest first.
        candidates = sorted(
            range(effective_bins), key=lambda b: -leftover_per_bin[b]
        )
        per_pass_add = 0
        for b in candidates:
            if total_shortfall <= 0:
                break
            extra = min(leftover_per_bin[b], total_shortfall)
            if extra <= 0:
                continue
            new_cap = len(pass1[b]) + extra
            fresh = sample_from_bin(b, new_cap)
            pass1[b] = fresh
            total_shortfall -= extra

    # Assemble output.
    out: list[tuple[str, str]] = []
    per_bin_stats = {}
    for b in range(effective_bins):
        clips = pass1[b]
        out.extend(clips)
        per_bin_stats[f"bin_{b}"] = {
            "label_range": [bin_edges[b], bin_edges[b + 1]],
            "studies_available": len(studies_by_bin[b]),
            "clips_kept": len(clips),
        }

    rng.shuffle(out)
    return out, {
        "strategy": "regression_quantile_stratified",
        "target": target,
        "clip_cap": clip_cap,
        "n_bins_requested": n_bins,
        "n_bins_effective": effective_bins,
        "per_bin": per_bin_stats,
        "total_clips": len(out),
    }


TASKS = {
    "disease_hcm_plax": "classification",
    "disease_hf_incident_1yr_a4c": "classification",
    "age_a4c": "regression",
    "lvef_a4c": "regression",
}


def cap_clips_per_study(
    rows: list[tuple[str, str]], cap: int, seed: int
) -> tuple[list[tuple[str, str]], dict]:
    """Keep every study; cap clips-per-study at `cap`. For val/test.

    Preserves every study's label (positives untouched for classification;
    tail coverage untouched for regression). Only reduces inference cost.
    """
    rng = random.Random(seed)
    per_study_clips: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for path, label in rows:
        per_study_clips[study_of(path)].append((path, label))

    out: list[tuple[str, str]] = []
    caps_hit = 0
    under_cap = 0
    for s, clips in per_study_clips.items():
        if len(clips) > cap:
            clips = rng.sample(clips, cap)
            caps_hit += 1
        else:
            under_cap += 1
        out.extend(clips)

    rng.shuffle(out)
    return out, {
        "strategy": "valtest_per_study_cap",
        "cap": cap,
        "studies_total": len(per_study_clips),
        "studies_capped": caps_hit,
        "studies_under_cap": under_cap,
        "input_clips": len(rows),
        "output_clips": len(out),
    }


VAL_TEST_CAP = 3  # clips per study for val/test (prediction averaging saturates early)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default="/mnt/custom-file-systems/efs/fs-0049217cdf69186d7_fsap-0fa7145b64eaa046b/vjepa2/experiments/nature_medicine/mimic/probe_csvs",
    )
    ap.add_argument("--target", type=int, default=TARGET)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--task", default=None, help="restrict to one task (default: all 4)")
    ap.add_argument(
        "--valtest-cap",
        type=int,
        default=VAL_TEST_CAP,
        help="clips-per-study cap for val/test; keeps all studies, only downsamples clips",
    )
    ap.add_argument(
        "--skip-valtest", action="store_true", help="skip val/test trimming"
    )
    args = ap.parse_args()

    root = Path(args.root)
    tasks = [args.task] if args.task else list(TASKS.keys())

    for task in tasks:
        strategy = TASKS[task]
        src = root / task / "train.csv"
        out_dir = root / f"{task}_10k"
        dst = out_dir / "train.csv"
        meta_dst = out_dir / "trim_meta.json"

        print(f"=== {task} ({strategy}) ===")
        if not src.exists():
            print(f"  [SKIP] missing source: {src}")
            continue

        rows = load_split(src)
        print(f"  train input:  {len(rows):,} clips")
        cap = CLIP_CAP_CLS if strategy == "classification" else CLIP_CAP_REG
        if strategy == "classification":
            out, meta = trim_classification(rows, args.target, cap, args.seed)
        else:
            out, meta = trim_regression(rows, args.target, cap, args.seed)

        write_split(dst, out)
        meta["source_csv"] = str(src.relative_to(root.parent))
        meta["output_csv"] = str(dst.relative_to(root.parent))
        meta["input_clips"] = len(rows)
        meta["seed"] = args.seed
        meta_dst.write_text(json.dumps(meta, indent=2))

        print(f"  train output: {len(out):,} clips -> {dst}")

        if not args.skip_valtest:
            for split in ("val", "test"):
                vsrc = root / task / f"{split}.csv"
                if not vsrc.exists():
                    print(f"  [SKIP] missing {split}: {vsrc}")
                    continue
                vrows = load_split(vsrc)
                vout, vmeta = cap_clips_per_study(vrows, args.valtest_cap, args.seed)
                vdst = out_dir / f"{split}.csv"
                vmeta_dst = out_dir / f"{split}_trim_meta.json"
                write_split(vdst, vout)
                vmeta["source_csv"] = str(vsrc.relative_to(root.parent))
                vmeta["output_csv"] = str(vdst.relative_to(root.parent))
                vmeta["seed"] = args.seed
                vmeta_dst.write_text(json.dumps(vmeta, indent=2))
                print(
                    f"  {split} output:  {len(vout):,} clips "
                    f"({vmeta['studies_capped']} capped, "
                    f"{vmeta['studies_under_cap']} under-cap) -> {vdst}"
                )
        print("")


if __name__ == "__main__":
    main()
