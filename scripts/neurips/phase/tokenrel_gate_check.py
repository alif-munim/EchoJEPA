#!/usr/bin/env python3
"""EchoJEPA-TokenRel / -Motion smoke gate checker.

Reads a run's ``log_r0.csv`` and (optionally) its stdout log, reports
pass/fail per the smoke gates. Exit code 0 = pass; 1 = fail.

Usage:
    tokenrel_gate_check.py run1 --csv /opt/.../log_r0.csv --stdout /opt/.../job.log
    tokenrel_gate_check.py run2 --csv /opt/.../log_r0.csv --stdout /opt/.../job.log

The sbatch for each smoke invokes this as its final step. Non-zero exit
prevents the chained ``afterok:`` full-run from launching.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path


def _read_rows(path: Path) -> list[dict]:
    if not path.exists():
        print(f"[GATE FAIL] CSV missing: {path}", file=sys.stderr)
        return []
    with path.open() as fh:
        return list(csv.DictReader(fh))


def _safe_float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _no_nan(rows: list[dict], cols: list[str]) -> tuple[bool, str]:
    for i, r in enumerate(rows):
        for col in cols:
            if col not in r:
                continue
            v = _safe_float(r[col])
            if math.isnan(v) or math.isinf(v):
                return False, f"row {i}: {col}={v}"
    return True, "no NaN/inf"


def _intraview_drift(rows: list[dict]) -> tuple[bool, str]:
    vals = [_safe_float(r.get("intraview", "nan")) for r in rows]
    vals = [v for v in vals if not math.isnan(v)]
    if len(vals) < 10:
        return True, f"not enough intraview samples ({len(vals)})"
    first = sum(vals[:10]) / 10
    last = sum(vals[-10:]) / 10
    if first <= 0.0:
        return True, f"intraview first-window mean non-positive ({first})"
    drift = abs(last - first) / first
    if drift > 0.20:
        return False, f"intraview drift = {drift:.3f} > 0.20 (first={first:.3f}, last={last:.3f})"
    return True, f"intraview drift = {drift:.3f} (first={first:.3f}, last={last:.3f})"


def _top1_above_random(rows: list[dict]) -> tuple[bool, str]:
    """token_rel_top1_with_hard window (last 20) > 1/(B+2) approx.

    Candidate pool is 1 pos + 1 hard + (B-1) batch. With B=32 local and
    8 GPUs, global batch = 32. So 1/(32+2) ~ 0.029. We require last-
    window mean > 0.05 to have a safety margin over pure random.
    """
    vals = [_safe_float(r.get("token_rel_top1_with_hard", "nan")) for r in rows]
    vals = [v for v in vals if not math.isnan(v)]
    if len(vals) < 20:
        return True, f"not enough top1 samples ({len(vals)}); skip gate"
    last_mean = sum(vals[-20:]) / 20
    if last_mean < 0.05:
        return False, f"token_rel_top1_with_hard last-20 mean = {last_mean:.3f} < 0.05"
    return True, f"token_rel_top1_with_hard last-20 mean = {last_mean:.3f}"


def _gap_non_decreasing(rows: list[dict]) -> tuple[bool, str]:
    """token_rel_pos_minus_hard_gap: last 20 >= first 20 (mean).

    Allow a small epsilon tolerance for numerical noise on 100-step runs.
    """
    vals = [_safe_float(r.get("token_rel_pos_minus_hard_gap", "nan")) for r in rows]
    vals = [v for v in vals if not math.isnan(v)]
    if len(vals) < 40:
        return True, f"not enough gap samples ({len(vals)}); skip gate"
    first = sum(vals[:20]) / 20
    last = sum(vals[-20:]) / 20
    if last < first - 0.01:
        return False, f"token_rel_pos_minus_hard_gap regressed: first={first:.3f}, last={last:.3f}"
    return True, f"token_rel_pos_minus_hard_gap first={first:.3f}, last={last:.3f}"


def _q_var_nonzero(rows: list[dict]) -> tuple[bool, str]:
    vals = [_safe_float(r.get("token_rel_q_var", "nan")) for r in rows]
    vals = [v for v in vals if not math.isnan(v)]
    if not vals:
        return True, "no q_var samples; skip"
    last = sum(vals[-10:]) / min(10, len(vals))
    if last <= 1e-8:
        return False, f"token_rel_q_var last-window mean = {last:.3e} ≈ 0 (representation collapse)"
    return True, f"token_rel_q_var last-window mean = {last:.3e}"


def _pool_rel_finite(rows: list[dict]) -> tuple[bool, str]:
    ok, msg = _no_nan(rows, ["pool_rel_loss"])
    return ok, f"pool_rel: {msg}"


def _delta_valid_rows(rows: list[dict]) -> tuple[bool, str]:
    vals = [_safe_float(r.get("delta_valid_rows", "nan")) for r in rows]
    vals = [v for v in vals if not math.isnan(v)]
    if not vals:
        return False, "delta_valid_rows missing"
    mean = sum(vals) / len(vals)
    if mean <= 0.0:
        return False, f"delta_valid_rows mean = {mean:.3f} ≤ 0 (no same-view rows ever)"
    return True, f"delta_valid_rows mean = {mean:.3f}"


def _delta_loss_finite(rows: list[dict]) -> tuple[bool, str]:
    ok, msg = _no_nan(rows, ["delta_loss", "delta_pos_minus_hard_gap"])
    return ok, f"delta: {msg}"


def _stdout_clean(stdout_path: Path | None) -> tuple[bool, str]:
    if stdout_path is None or not stdout_path.exists():
        return True, "no stdout check"
    red = ["non-finite", "NaN detected", "Traceback (most recent call last):"]
    with stdout_path.open() as fh:
        for line in fh:
            for flag in red:
                if flag in line:
                    return False, f"stdout flag {flag!r}: {line.strip()[:120]}"
    return True, "stdout clean"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run1", "run2"])
    ap.add_argument("--csv", required=True)
    ap.add_argument("--stdout", required=False, default=None)
    args = ap.parse_args()

    rows = _read_rows(Path(args.csv))
    stdout = Path(args.stdout) if args.stdout else None

    checks = []
    checks.append(("no NaN/inf (core)", _no_nan(rows, [
        "loss", "intraview", "token_rel_loss", "token_rel_top1_with_hard",
        "token_rel_pos_minus_hard_gap",
    ])))
    checks.append(("intraview drift <= 20%", _intraview_drift(rows)))
    checks.append(("top1 > random", _top1_above_random(rows)))
    checks.append(("pos−hard gap non-decreasing", _gap_non_decreasing(rows)))
    checks.append(("q_var nonzero", _q_var_nonzero(rows)))
    checks.append(("pool_rel_loss finite", _pool_rel_finite(rows)))

    if args.stage == "run2":
        checks.append(("delta_valid_rows > 0 (mean)", _delta_valid_rows(rows)))
        checks.append(("delta loss finite", _delta_loss_finite(rows)))

    checks.append(("stdout clean", _stdout_clean(stdout)))

    any_fail = False
    print(f"[tokenrel_gate_check] stage={args.stage} csv={args.csv} rows={len(rows)}")
    for name, (ok, msg) in checks:
        tag = "PASS" if ok else "FAIL"
        if not ok:
            any_fail = True
        print(f"  [{tag}] {name}: {msg}")

    if any_fail:
        print("[GATE FAIL] one or more checks failed")
        sys.exit(1)
    print("[GATE PASS] all checks passed")


if __name__ == "__main__":
    main()
