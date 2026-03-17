#!/usr/bin/env python3
"""Run linear probe training across all model/task combinations defined in a config file."""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def build_commands(config):
    """Build list of (description, command) tuples from config."""
    base_dir = Path(config["base_dir"])
    output_dir = Path(config["output_dir"])
    models = config["models"]
    tasks = config["tasks"]

    # Build task name -> type lookup
    task_lookup = {t["name"]: t["type"] for t in tasks}

    commands = []
    for model in models:
        splits_dir = base_dir / f"{model}_splits_allclips"
        for task in tasks:
            task_name = task["name"]
            task_type = task["type"]
            task_dir = splits_dir / task_name
            train_path = task_dir / "train.npz"
            val_path = task_dir / "val.npz"
            test_path = task_dir / "test.npz"

            if not train_path.exists():
                print(f"[SKIP] {model}/{task_name}: {train_path} not found")
                continue

            out_dir = output_dir / model / task_name
            cmd = [
                sys.executable, "-m", "evals.train_probe",
                "--train", str(train_path),
                "--val", str(val_path),
                "--task", task_type,
                "--save_model",
                "--output_dir", str(out_dir),
            ]
            if test_path.exists():
                cmd.extend(["--test", str(test_path)])

            commands.append((f"{model}/{task_name} ({task_type})", cmd))

    return commands


def main():
    parser = argparse.ArgumentParser(description="Run probe training from config.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config file")
    parser.add_argument("--models", nargs="*", help="Run only these models (overrides config)")
    parser.add_argument("--tasks", nargs="*", help="Run only these tasks (overrides config)")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without executing")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.models:
        config["models"] = args.models
    if args.tasks:
        config["tasks"] = [t for t in config["tasks"] if t["name"] in args.tasks]

    commands = build_commands(config)
    total = len(commands)
    print(f"Total jobs: {total}")

    for i, (desc, cmd) in enumerate(commands, 1):
        print(f"\n[{i}/{total}] {desc}")
        print(f"  CMD: {' '.join(cmd)}")

        if args.dry_run:
            continue

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"  [FAILED] exit code {result.returncode}")
        else:
            print(f"  [DONE]")


if __name__ == "__main__":
    main()
