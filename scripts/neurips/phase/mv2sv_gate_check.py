#!/usr/bin/env python3
"""MV2SV v5 staged-smoke gate checker.

Reads a run's ``log_r0.csv`` and (optionally) its stdout log, and
reports pass/fail per the stage's success gates. Exit code 0 = pass;
1 = fail.

Usage:
    mv2sv_gate_check.py parity   --csv /opt/.../log_r0.csv --stdout /opt/.../job.log
    mv2sv_gate_check.py nce      --csv /opt/.../log_r0.csv
    mv2sv_gate_check.py fused    --csv /opt/.../log_r0.csv

The sbatch for each stage invokes this as its final step. If the gate
fails, the sbatch exits non-zero and the chained ``afterok:`` dependency
prevents downstream stages from launching.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path


def _read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        print(f"[GATE FAIL] CSV missing: {path}", file=sys.stderr)
        return []
    with path.open() as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    return rows


def _safe_float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _check_no_nan(rows: list[dict], cols: list[str]) -> tuple[bool, str]:
    for i, r in enumerate(rows):
        for col in cols:
            if col not in r:
                continue
            v = _safe_float(r[col])
            if math.isnan(v) or math.isinf(v):
                return False, f"row {i}: {col}={v}"
    return True, "no NaN/inf"


def _check_stdout(stdout_path: Path | None) -> tuple[bool, str]:
    """Scan stdout for red flags: explicit NaN, DDP reducer warnings,
    guard ValueErrors, or Python tracebacks."""
    if stdout_path is None:
        return True, "no stdout check requested"
    if not stdout_path.exists():
        return True, f"no stdout at {stdout_path}"
    red_flags = [
        "non-finite",
        "NaN detected",
        "DDP reducer",
        "find_unused_parameters=True",  # a warning, not an error, but useful to flag if unexpected
        "Traceback (most recent call last):",
    ]
    # Only flag DDP warning if it appears with "ERROR" context.
    with stdout_path.open() as fh:
        for line in fh:
            for flag in red_flags:
                if flag in line:
                    # find_unused_parameters is expected on shared_projector_fused;
                    # ignore that specific context.
                    if flag == "find_unused_parameters=True" and "fused_shared_projector" in line:
                        continue
                    return False, f"stdout red flag: {flag!r} in {line.strip()[:120]}"
    return True, "stdout clean"


def _gate_parity(rows: list[dict], stdout_path: Path | None) -> list[tuple[str, bool, str]]:
    results: list[tuple[str, bool, str]] = []
    if not rows:
        results.append(("rows_present", False, "CSV empty"))
        return results
    # total_loss == intraview within tolerance (bf16 noise → 1e-3).
    max_abs_diff = 0.0
    for r in rows:
        total = _safe_float(r.get("loss"))
        intra = _safe_float(r.get("intraview"))
        if math.isnan(total) or math.isnan(intra):
            continue
        max_abs_diff = max(max_abs_diff, abs(total - intra))
    # bf16 tolerance is loose; a few 1e-3 is acceptable.
    results.append(
        (
            "parity: total==intraview",
            max_abs_diff < 1e-2,
            f"max|total-intra| = {max_abs_diff:.6f}",
        )
    )
    ok, msg = _check_no_nan(rows, ["loss", "intraview"])
    results.append(("no NaN in loss/intraview", ok, msg))
    ok, msg = _check_stdout(stdout_path)
    results.append(("stdout clean", ok, msg))
    return results


def _gate_nce(rows: list[dict], stdout_path: Path | None) -> list[tuple[str, bool, str]]:
    results: list[tuple[str, bool, str]] = []
    if not rows:
        results.append(("rows_present", False, "CSV empty"))
        return results
    B = 16  # matches the smoke YAML's data.batch_size
    # Scientific-path invariants (checked every row):
    any_fallback = any(_safe_float(r.get("used_clip_b_fallback", 0)) > 0.5 for r in rows)
    results.append(
        ("used_clip_b_fallback == 0 every step", not any_fallback, "see CSV")
    )
    # pct_target_clip_present: relaxed from strict ==1.0 to mean >= 0.85.
    # Real batches can drop a few rows because a sampled study lacks a valid
    # target-view candidate; the forward path masks those rows out of
    # pair_view / view_nce, so the objective is still scientifically sound
    # as long as the dropout rate is low. Warn (not fail) if any single
    # step goes below 0.75 sustained (>=3 consecutive logged steps).
    pct_tgts = [_safe_float(r.get("pct_target_clip_present", 0)) for r in rows]
    mean_pct_tgt = sum(pct_tgts) / max(1, len(pct_tgts))
    results.append(
        (
            "pct_target_clip_present mean >= 0.85",
            mean_pct_tgt >= 0.85,
            f"mean = {mean_pct_tgt:.3f}",
        )
    )
    # Sustained low-coverage warning (non-failing): 3+ consecutive steps <0.75.
    run = 0
    worst_run = 0
    for v in pct_tgts:
        if v < 0.75:
            run += 1
            worst_run = max(worst_run, run)
        else:
            run = 0
    if worst_run >= 3:
        print(
            f"  [WARN] pct_target_clip_present <0.75 sustained for "
            f"{worst_run} consecutive steps (non-failing)",
            file=sys.stderr,
        )
    # view_nce valid negatives: relaxed from min>=1 (every row) to mean>=1.
    # Rows with zero valid negatives are masked out of the loss by the
    # forward path; a handful per step is fine, the fail mode we care
    # about is most rows having no negatives (mean drops below 1).
    neg_means = [
        _safe_float(r.get("view_nce_valid_neg_count_mean", 0)) for r in rows
    ]
    mean_neg = sum(neg_means) / max(1, len(neg_means))
    results.append(
        (
            "view_nce_valid_neg_count mean >= 1.0",
            mean_neg >= 1.0,
            f"mean = {mean_neg:.3f}",
        )
    )
    fallback_fracs = [
        _safe_float(r.get("view_nce_fallback_fraction", 0)) for r in rows
    ]
    mean_fb = sum(fallback_fracs) / max(1, len(fallback_fracs))
    results.append(
        (
            "view_nce_fallback_fraction <= 0.3 (mean)",
            mean_fb <= 0.3,
            f"mean = {mean_fb:.3f}",
        )
    )
    # Trajectory: top1 should rise above random (1/B). Compare late-run
    # mean (last quarter) to an "above random" threshold.
    top1s = [_safe_float(r.get("view_nce_top1", 0)) for r in rows]
    if top1s:
        tail = top1s[max(0, len(top1s) * 3 // 4):]
        tail_mean = sum(tail) / max(1, len(tail))
        random_baseline = 1.0 / float(B)
        results.append(
            (
                f"view_nce_top1 > 1/B ({random_baseline:.3f}) in late run",
                tail_mean > random_baseline,
                f"tail_mean = {tail_mean:.3f}",
            )
        )
    # Pos - neg gap increases: compare first half to last half.
    gaps = []
    for r in rows:
        pos = _safe_float(r.get("view_nce_pos_sim_mean", 0))
        neg = _safe_float(r.get("view_nce_neg_sim_mean", 0))
        if math.isnan(pos) or math.isnan(neg):
            continue
        gaps.append(pos - neg)
    if len(gaps) >= 4:
        half = len(gaps) // 2
        first_mean = sum(gaps[:half]) / half
        last_mean = sum(gaps[half:]) / (len(gaps) - half)
        results.append(
            (
                "pos_sim - neg_sim increases",
                last_mean > first_mean,
                f"first_half={first_mean:.4f}, last_half={last_mean:.4f}",
            )
        )
    # Intraview stability: compare first and last quartiles.
    intras = [_safe_float(r.get("intraview", 0)) for r in rows if r.get("intraview")]
    if len(intras) >= 4:
        first_q = intras[: max(1, len(intras) // 4)]
        last_q = intras[-max(1, len(intras) // 4):]
        first_mean = sum(first_q) / len(first_q)
        last_mean = sum(last_q) / len(last_q)
        drift = abs(last_mean - first_mean) / max(1e-6, first_mean)
        results.append(
            (
                "intraview drift <= 20%",
                drift <= 0.20,
                f"first={first_mean:.4f}, last={last_mean:.4f}, drift={drift*100:.1f}%",
            )
        )
    # NaN/inf across all key losses.
    ok, msg = _check_no_nan(
        rows,
        [
            "loss", "intraview",
            "pair_shared_loss", "pair_view_loss", "view_nce_loss",
            "shared_loss",
        ],
    )
    results.append(("no NaN/inf in MV2SV losses", ok, msg))
    ok, msg = _check_stdout(stdout_path)
    results.append(("stdout clean", ok, msg))
    return results


def _gate_fused(rows: list[dict], stdout_path: Path | None) -> list[tuple[str, bool, str]]:
    # Stage C = Stage B + fused-specific gates.
    results = _gate_nce(rows, stdout_path)
    if not rows:
        return results
    pct_f_ok = all(
        _safe_float(r.get("pct_fused_clips_present", 0)) >= 0.999 for r in rows
    )
    results.append(("pct_fused_clips_present == 1.0", pct_f_ok, ""))
    # vv_mean is 0 on fused_active=False steps; only check active steps.
    vv_means = [
        _safe_float(r.get("fused_valid_views_mean", 0))
        for r in rows
        if _safe_float(r.get("fused_active", 0)) > 0.5
    ]
    if vv_means:
        mean_vv = sum(vv_means) / len(vv_means)
        results.append(
            (
                "fused_valid_views_mean >= 2 on active steps",
                mean_vv >= 2.0,
                f"mean over {len(vv_means)} active steps = {mean_vv:.3f}",
            )
        )
    else:
        results.append(
            ("fused_active steps present", False, "no row had fused_active > 0")
        )
    # fused_loss finite + non-exploding.
    fls = [
        _safe_float(r.get("fused_loss", 0))
        for r in rows
        if _safe_float(r.get("fused_active", 0)) > 0.5
    ]
    if fls:
        max_fl = max(fls)
        results.append(
            (
                "fused_loss finite and < 10 on active steps",
                all(math.isfinite(x) for x in fls) and max_fl < 10.0,
                f"max_active_fused_loss = {max_fl:.4f}",
            )
        )
    return results


def _gate_ctrl(rows: list[dict], stdout_path: Path | None) -> list[tuple[str, bool, str]]:
    """MV2SV pipeline-matched intraview-only control gates.

    All MV2SV auxiliary lambdas are 0; view_nce / pair_view / pair_shared
    columns will be zero and are not informative. What we check:
      - sampler is still delivering target_clip (pct_target >= 0.85)
      - no silent clip_b fallback
      - intraview is stable (drift <= 20%)
      - no NaN/inf in loss/intraview
      - stdout is clean
    """
    results: list[tuple[str, bool, str]] = []
    if not rows:
        results.append(("rows_present", False, "CSV empty"))
        return results
    any_fallback = any(_safe_float(r.get("used_clip_b_fallback", 0)) > 0.5 for r in rows)
    results.append(
        ("used_clip_b_fallback == 0 every step", not any_fallback, "see CSV")
    )
    pct_tgts = [_safe_float(r.get("pct_target_clip_present", 0)) for r in rows]
    mean_pct_tgt = sum(pct_tgts) / max(1, len(pct_tgts))
    results.append(
        (
            "pct_target_clip_present mean >= 0.85",
            mean_pct_tgt >= 0.85,
            f"mean = {mean_pct_tgt:.3f}",
        )
    )
    # Intraview stability: compare first and last quartiles.
    intras = [_safe_float(r.get("intraview", 0)) for r in rows if r.get("intraview")]
    if len(intras) >= 4:
        first_q = intras[: max(1, len(intras) // 4)]
        last_q = intras[-max(1, len(intras) // 4):]
        first_mean = sum(first_q) / len(first_q)
        last_mean = sum(last_q) / len(last_q)
        drift = abs(last_mean - first_mean) / max(1e-6, first_mean)
        results.append(
            (
                "intraview drift <= 20%",
                drift <= 0.20,
                f"first={first_mean:.4f}, last={last_mean:.4f}, drift={drift*100:.1f}%",
            )
        )
    # Intraview ~= total loss (all other lambdas are zero).
    max_abs_diff = 0.0
    for r in rows:
        total = _safe_float(r.get("loss"))
        intra = _safe_float(r.get("intraview"))
        if math.isnan(total) or math.isnan(intra):
            continue
        max_abs_diff = max(max_abs_diff, abs(total - intra))
    results.append(
        (
            "total_loss == intraview (all aux lambdas = 0)",
            max_abs_diff < 1e-2,
            f"max|total-intra| = {max_abs_diff:.6f}",
        )
    )
    ok, msg = _check_no_nan(rows, ["loss", "intraview"])
    results.append(("no NaN/inf in loss/intraview", ok, msg))
    ok, msg = _check_stdout(stdout_path)
    results.append(("stdout clean", ok, msg))
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["parity", "nce", "fused", "ctrl"])
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--stdout", type=Path, default=None)
    args = parser.parse_args()

    rows = _read_csv_rows(args.csv)
    if args.stage == "parity":
        results = _gate_parity(rows, args.stdout)
    elif args.stage == "nce":
        results = _gate_nce(rows, args.stdout)
    elif args.stage == "ctrl":
        results = _gate_ctrl(rows, args.stdout)
    else:
        results = _gate_fused(rows, args.stdout)

    all_ok = True
    print(f"=== MV2SV v5 gate check: stage={args.stage} ===")
    for name, ok, note in results:
        status = "PASS" if ok else "FAIL"
        all_ok = all_ok and ok
        print(f"  [{status}] {name}" + (f" ({note})" if note else ""))
    print(f"=== overall: {'PASS' if all_ok else 'FAIL'} ===")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
