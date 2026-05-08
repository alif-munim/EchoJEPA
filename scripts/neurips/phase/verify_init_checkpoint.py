"""Init-checkpoint preflight for phase-relational pretraining.

Verifies that the checkpoint path configured in a paper-ready YAML is
the exact MIMIC-standard JEPA e100 checkpoint used by the prior +25
continuation experiments (jobs 542/548). Runs BEFORE any GPU work so
the user sees the sha256 / metadata before committing compute.

Usage:
    python scripts/neurips/phase/verify_init_checkpoint.py \\
        --ckpt /opt/dlami/nvme/checkpoints/jepa_in21k_vitl_e100.pt \\
        --expected-tag mimic_standard_jepa_e100

Exit codes:
    0 — verified (sha256 matches allowlist OR allowlist entry is TBD and
        the computed prefix has been logged for the user to commit).
    2 — path missing / unreadable / top-level-keys don't look like a
        V-JEPA checkpoint.
    3 — sha256 prefix mismatches the allowlist for the given tag.

The allowlist lives in-file; populate ``expected_sha256_prefix`` after
the first invocation prints the computed prefix.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

import torch


# Tag → known identity. Populated after first invocation on the real
# production checkpoint. Until then the preflight prints the computed
# prefix and asks the user to commit it here.
KNOWN_INIT_CHECKPOINTS = {
    "mimic_standard_jepa_e100": {
        "description": (
            "IN21K → MIMIC standard V-JEPA e100. The exact checkpoint that "
            "jobs 542/548 started from. Lives at:\n"
            "  s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/"
            "vjepa2-artifacts/checkpoints/jepa_in21k/jepa_in21k_vitl_e100.pt\n"
            "  (or /opt/dlami/nvme/checkpoints/jepa_in21k_vitl_e100.pt on "
            "the HyperPod compute node after the sbatch download.)"
        ),
        # Committed 2026-05-01 from the local repo copy at
        # checkpoints/jepa_in21k_vitl_e100.pt (5,127,835,835 bytes,
        # mtime 2026-04-06). Full sha256:
        # cdf0fabefe83e21e8e0570919e81cc73a30c9b942e0b88d43ff845869a0ceefc
        "expected_sha256_prefix": "cdf0fabefe83e21e",
        "expected_epoch": 100,
    },
    "mimic_lk_jepa_e100": {
        "description": (
            "EchoJEPA-L-K e100 (Kinetics-400 → MIMIC, runs/11 lineage). "
            "Byte-identical to `runs/11/training_folder/e100.pt`; "
            "canonical copy at:\n"
            "  s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/"
            "vjepa2-artifacts/checkpoints/echojepa_l_k_pretrain_k400_to_mimic/e100.pt\n"
            "  (5,127,746,365 bytes, 2026-01-20 on S3). See "
            "claude/architecture/echojepa-l-k-pretrain-checkpoints.md for lineage proof."
        ),
        # SHA256 prefix pinned on first run via the sbatch preflight; the
        # initial invocation runs with --strict disabled, prints the prefix,
        # and the operator commits it here so subsequent jobs gate on it.
        "expected_sha256_prefix": "TBD",
        "expected_epoch": 100,
    },
}


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True, help="Path to the init checkpoint.")
    ap.add_argument(
        "--expected-tag", type=str, default="mimic_standard_jepa_e100",
        help="Which entry in KNOWN_INIT_CHECKPOINTS to validate against."
    )
    ap.add_argument(
        "--strict", action="store_true",
        help=(
            "If set, exit non-zero when the tag's expected_sha256_prefix "
            "is still 'TBD'. Useful for launch gates that must not run "
            "against an un-validated init."
        ),
    )
    args = ap.parse_args()

    print("INIT CHECKPOINT VERIFICATION")
    print("-" * 60)

    # 1. Path
    print(f"  path:                  {args.ckpt}")
    if not args.ckpt.exists():
        print("  exists:                False")
        print("\n[FAIL] checkpoint path does not exist.")
        return 2
    print("  exists:                True")
    size_bytes = args.ckpt.stat().st_size
    print(f"  file size:             {size_bytes:,} bytes ({size_bytes / 1e9:.2f} GB)")
    mtime = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(args.ckpt.stat().st_mtime))
    print(f"  mtime:                 {mtime}")

    # 2. Tag allowlist lookup
    tag = args.expected_tag
    if tag not in KNOWN_INIT_CHECKPOINTS:
        print(f"\n[FAIL] expected_init_checkpoint_tag={tag!r} is not in "
              f"KNOWN_INIT_CHECKPOINTS.")
        print("       Known tags:", sorted(KNOWN_INIT_CHECKPOINTS.keys()))
        return 3
    spec = KNOWN_INIT_CHECKPOINTS[tag]
    expected_prefix = spec["expected_sha256_prefix"]
    print(f"  expected_init_checkpoint_tag: {tag}")
    print(f"    description: {spec['description'].splitlines()[0]}")

    # 3. Hash
    print("  computing sha256 ...", end="", flush=True)
    t0 = time.time()
    full_sha = _sha256_file(args.ckpt)
    dt = time.time() - t0
    print(f" done in {dt:.1f}s")
    print(f"  sha256 (full):         {full_sha}")
    prefix = full_sha[:16]
    print(f"  sha256 (16-char prefix): {prefix}")

    # 4. Top-level keys + epoch + optimizer-state presence
    print("  loading checkpoint header ...")
    try:
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(args.ckpt, map_location="cpu")
    except Exception as e:
        print(f"\n[FAIL] torch.load raised: {type(e).__name__}: {e}")
        return 2
    if not isinstance(ckpt, dict):
        print(f"\n[FAIL] checkpoint root is not a dict (got {type(ckpt).__name__}).")
        return 2
    top_keys = sorted(ckpt.keys())
    print(f"  top-level keys:        {top_keys}")
    epoch_in_ckpt = ckpt.get("epoch", "n/a")
    print(f"  epoch in ckpt:         {epoch_in_ckpt}")
    print(f"  opt state present:     {'opt' in ckpt and ckpt['opt'] is not None}")
    has_relhead = "relational_head" in ckpt
    print(f"  relational_head keys present: {has_relhead}")

    # Sanity: this must look like a V-JEPA checkpoint (encoder + target_encoder
    # keys) — catches accidental model-only torch.save files.
    expected_top = {"encoder", "target_encoder"}
    missing = expected_top - set(top_keys)
    if missing:
        print(f"\n[FAIL] checkpoint missing expected V-JEPA top-level keys: {missing}")
        return 2

    # 5. Epoch sanity
    expected_epoch = spec.get("expected_epoch")
    if expected_epoch is not None and isinstance(epoch_in_ckpt, int):
        if epoch_in_ckpt != expected_epoch:
            print(f"\n[WARN] ckpt epoch={epoch_in_ckpt} but expected {expected_epoch} for "
                  f"tag={tag}.")

    # 6. Allowlist check
    if expected_prefix == "TBD":
        print()
        print("  [INFO] expected_sha256_prefix for this tag is TBD.")
        print(f"         Computed prefix: {prefix}")
        print("         After you confirm this is the right file, edit:")
        print(f"         {Path(__file__).resolve()}")
        print(f"         and set KNOWN_INIT_CHECKPOINTS[{tag!r}]['expected_sha256_prefix']")
        print(f"         = {prefix!r}")
        if args.strict:
            print("\n[FAIL] --strict: allowlist entry still TBD.")
            return 3
        print("\n[OK ] preflight complete (allowlist TBD; user must commit the prefix).")
        return 0

    if prefix != expected_prefix:
        print()
        print(f"  [FAIL] sha256 prefix mismatch:")
        print(f"         got      {prefix}")
        print(f"         expected {expected_prefix}")
        print("         The init checkpoint path does NOT match the "
              f"{tag} allowlist entry. Do not launch.")
        return 3

    print()
    print(f"  [OK ] sha256 prefix matches allowlist entry for tag={tag}.")
    print("\nINIT CHECKPOINT VERIFIED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
