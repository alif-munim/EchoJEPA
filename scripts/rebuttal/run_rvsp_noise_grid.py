"""
RVSP noise robustness grid: run a probe config under all perturbation combinations.

Usage:
    TMPDIR=/tmp LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH \
        python scripts/rebuttal/run_rvsp_noise_grid.py \
        --config configs/inference/vitl/icml/echojepa_l_pt50_rvsp_a4c_test.yaml \
        --gpu 2 --tag a4c

Runs 9 combinations (3 types × 3 severities) sequentially on the given GPU,
extracts R²/Pearson from the log CSV after each run, and prints a summary table.
"""

import argparse
import csv
import os
import subprocess
import sys


PERTURBATION_TYPES = ["depth_attenuation", "gaussian_shadow", "haze_artifact"]
SEVERITY_LEVELS = ["mild", "moderate", "severe"]


def find_log_csv(config_path):
    """Parse config to find the log CSV path."""
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    folder = cfg.get("folder", "")
    eval_name = cfg.get("eval_name", "video_classification_frozen")
    tag = cfg.get("tag", "run")
    return os.path.join(folder, eval_name, tag, "log_r0.csv")


def run_inference(config_path, gpu, ptype, severity):
    """Run one inference with perturbation env vars, return (r2, pearson, mae)."""
    env = os.environ.copy()
    env["PERTURBATION_TYPE"] = ptype
    env["PERTURBATION_SEVERITY"] = severity
    env["TMPDIR"] = "/tmp"
    env["LD_LIBRARY_PATH"] = "/opt/conda/lib:" + env.get("LD_LIBRARY_PATH", "")

    cmd = [
        sys.executable, "-m", "evals.main",
        "--fname", config_path,
        "--devices", f"cuda:{gpu}",
        "--val_only",
    ]

    log_csv = find_log_csv(config_path)

    # Remove old log so we get fresh results
    if os.path.exists(log_csv):
        os.remove(log_csv)

    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1800)

    # Parse results from log CSV
    r2, pearson, mae = None, None, None
    if os.path.exists(log_csv):
        with open(log_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                mae = float(row.get("val_mae", 0))
                r2 = float(row.get("val_r2", 0))
                pearson = float(row.get("val_pearson", 0))

    # Also grep from stderr for the summary line
    for line in (proc.stderr + proc.stdout).split("\n"):
        if "val R" in line and "Pearson" in line:
            # [    1] val R²: 0.1809 (head 5)  val Pearson: 0.4482 (head 4)
            import re
            m_r2 = re.search(r"val R.?: ([-\d.]+)", line)
            m_p = re.search(r"val Pearson: ([-\d.]+)", line)
            if m_r2:
                r2 = float(m_r2.group(1))
            if m_p:
                pearson = float(m_p.group(1))

    return r2, pearson, mae


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Inference config YAML")
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--tag", default="probe", help="Label for output")
    parser.add_argument("--skip_clean", action="store_true", help="Skip clean baseline (already have it)")
    args = parser.parse_args()

    results = {}

    if not args.skip_clean:
        print(f"=== {args.tag} / CLEAN on cuda:{args.gpu} ===")
        # Run without perturbation env vars
        env = os.environ.copy()
        env.pop("PERTURBATION_TYPE", None)
        env.pop("PERTURBATION_SEVERITY", None)
        env["TMPDIR"] = "/tmp"
        env["LD_LIBRARY_PATH"] = "/opt/conda/lib:" + env.get("LD_LIBRARY_PATH", "")
        log_csv = find_log_csv(args.config)
        if os.path.exists(log_csv):
            os.remove(log_csv)
        cmd = [sys.executable, "-m", "evals.main", "--fname", args.config,
               "--devices", f"cuda:{args.gpu}", "--val_only"]
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1800)
        r2, pearson, mae = None, None, None
        for line in (proc.stderr + proc.stdout).split("\n"):
            if "val R" in line and "Pearson" in line:
                import re
                m_r2 = re.search(r"val R.?: ([-\d.]+)", line)
                m_p = re.search(r"val Pearson: ([-\d.]+)", line)
                if m_r2:
                    r2 = float(m_r2.group(1))
                if m_p:
                    pearson = float(m_p.group(1))
        results[("clean", "-")] = (r2, pearson, mae)
        print(f"  Clean: R²={r2}, Pearson={pearson}, MAE={mae}")

    for ptype in PERTURBATION_TYPES:
        for sev in SEVERITY_LEVELS:
            print(f"=== {args.tag} / {ptype} / {sev} on cuda:{args.gpu} ===")
            r2, pearson, mae = run_inference(args.config, args.gpu, ptype, sev)
            results[(ptype, sev)] = (r2, pearson, mae)
            print(f"  {ptype}/{sev}: R²={r2}, Pearson={pearson}, MAE={mae}")

    # Print summary table
    print(f"\n{'='*70}")
    print(f"SUMMARY: {args.tag}")
    print(f"{'='*70}")
    print(f"{'Perturbation':<25} {'Severity':<10} {'R²':>8} {'Pearson':>8} {'MAE':>8}")
    print("-" * 70)
    for (ptype, sev), (r2, pearson, mae) in sorted(results.items()):
        r2_s = f"{r2:.4f}" if r2 is not None else "N/A"
        p_s = f"{pearson:.4f}" if pearson is not None else "N/A"
        m_s = f"{mae:.3f}" if mae is not None else "N/A"
        print(f"{ptype:<25} {sev:<10} {r2_s:>8} {p_s:>8} {m_s:>8}")


if __name__ == "__main__":
    main()
