"""Core evaluation orchestrator for EchoBench.

Runs a model through all noise conditions on a given task and collects results.
"""

import json
import time
from datetime import datetime, timezone
from itertools import product

from echobench.noise import NOISE_TYPES, SEVERITY_LEVELS


def evaluate(
    adapter,
    task,
    noise_types=None,
    severity_levels=None,
    device="cuda",
    max_cases=None,
    output_path=None,
):
    """
    Run the full EchoBench evaluation.

    Args:
        adapter: EncoderAdapter instance (frozen, on device)
        task: task instance with ``evaluate_condition()`` method (e.g., LVEFTask)
        noise_types: list of noise types to evaluate (default: all 4)
        severity_levels: list of severity levels (default: all 3)
        device: torch device string
        max_cases: limit number of test videos per condition
        output_path: path to write JSON results (optional)

    Returns:
        dict with "meta", "conditions", and "summary" keys
    """
    if noise_types is None:
        noise_types = NOISE_TYPES
    if severity_levels is None:
        severity_levels = SEVERITY_LEVELS

    # Build condition list: clean + noise × severity
    conditions = [("clean", None, None)]
    for nt, sev in product(noise_types, severity_levels):
        conditions.append((f"{nt}/{sev}", nt, sev))

    print(f"EchoBench: {len(conditions)} conditions "
          f"({len(noise_types)} noise types x {len(severity_levels)} severities + clean)")
    if max_cases:
        print(f"  Max cases per condition: {max_cases}")

    # Run all conditions
    results = []
    t_start = time.time()

    for i, (name, nt, sev) in enumerate(conditions):
        t0 = time.time()
        print(f"\n[{i + 1}/{len(conditions)}] {name}")

        result = task.evaluate_condition(
            adapter, noise_type=nt, severity=sev, device=device, max_cases=max_cases,
        )

        elapsed = time.time() - t0
        result["condition"] = name
        result["noise_type"] = nt
        result["severity"] = sev
        result["elapsed_seconds"] = round(elapsed, 1)
        results.append(result)

        # Print metrics inline
        m = result.get("metrics", {})
        metric_str = ", ".join(f"{k}={v:.4f}" for k, v in m.items())
        print(f"  {metric_str}  ({result['n_samples']} samples, {elapsed:.1f}s)")

    total_time = time.time() - t_start

    # Compute summary
    clean_metrics = results[0]["metrics"] if results else {}
    summary = _compute_summary(results, clean_metrics)

    output = {
        "meta": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "num_conditions": len(conditions),
            "noise_types": noise_types,
            "severity_levels": severity_levels,
            "max_cases": max_cases,
            "total_seconds": round(total_time, 1),
        },
        "conditions": results,
        "summary": summary,
    }

    # Print summary table
    _print_summary(results, summary)

    if output_path:
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nResults saved to {output_path}")

    return output


def _compute_summary(results, clean_metrics):
    """Compute average degradation across all noised conditions."""
    if not clean_metrics or not results:
        return {}

    # Pick the primary metric for degradation computation
    if "mae" in clean_metrics:
        primary_metric = "mae"
        higher_is_worse = True
    elif "r2" in clean_metrics:
        primary_metric = "r2"
        higher_is_worse = False
    else:
        return {}

    clean_val = clean_metrics[primary_metric]
    if clean_val == 0:
        return {"primary_metric": primary_metric, "avg_degradation_pct": float("nan")}

    degradations = []
    per_noise = {}

    for r in results:
        if r["noise_type"] is None:
            continue
        nt = r["noise_type"]
        m = r.get("metrics", {})
        if primary_metric not in m:
            continue

        val = m[primary_metric]
        if higher_is_worse:
            deg_pct = (val - clean_val) / abs(clean_val) * 100
        else:
            deg_pct = (clean_val - val) / abs(clean_val) * 100

        degradations.append(deg_pct)

        if nt not in per_noise:
            per_noise[nt] = []
        per_noise[nt].append(deg_pct)

    avg_deg = sum(degradations) / len(degradations) if degradations else 0.0

    per_noise_avg = {nt: sum(d) / len(d) for nt, d in per_noise.items()}

    return {
        "primary_metric": primary_metric,
        "clean_value": clean_val,
        "avg_degradation_pct": round(avg_deg, 2),
        "per_noise_degradation_pct": {k: round(v, 2) for k, v in per_noise_avg.items()},
    }


def _print_summary(results, summary):
    """Print a human-readable summary table."""
    print("\n" + "=" * 70)
    print("EchoBench Summary")
    print("=" * 70)

    if not results:
        print("No results.")
        return

    # Compact table
    clean = results[0]
    m = clean.get("metrics", {})
    metric_keys = list(m.keys())

    # Header
    header = f"{'Condition':<30}" + "".join(f"{k:>12}" for k in metric_keys)
    print(header)
    print("-" * len(header))

    for r in results:
        m = r.get("metrics", {})
        row = f"{r['condition']:<30}"
        for k in metric_keys:
            val = m.get(k, float("nan"))
            row += f"{val:>12.4f}"
        print(row)

    if summary:
        print("-" * len(header))
        print(f"Avg Degradation: {summary.get('avg_degradation_pct', 'N/A')}% "
              f"(metric: {summary.get('primary_metric', 'N/A')})")

        per_noise = summary.get("per_noise_degradation_pct", {})
        if per_noise:
            print("Per-noise breakdown:")
            for nt, deg in per_noise.items():
                print(f"  {nt}: {deg:+.2f}%")
