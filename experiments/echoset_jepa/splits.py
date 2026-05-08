"""Patient-level train/val/test split for EchoSet-JEPA (plan §11.2).

A study cannot appear in two splits. A patient cannot appear in two splits.
Default split: 85% train / 7.5% val / 7.5% test by hashed patient_id. The hash
is deterministic so every invocation with the same seed yields identical
assignments — cluster-friendly, no pickle required.

Why patient-level: MIMIC has ~4.5k patients × ~1.6 studies/patient. Splitting
at the study level would leak patient-specific anatomy / vendor / site
characteristics into val/test. Every prior NeurIPS and Nature Medicine probe
used patient-level splits; this keeps EchoSet-JEPA comparable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

SplitName = str  # "train" | "val" | "test"


def _hash_to_unit(patient_id: str, seed: int) -> float:
    """Deterministic [0, 1) hash of a patient_id, parameterized by seed."""
    h = hashlib.sha256(f"{seed}:{patient_id}".encode("utf-8")).hexdigest()
    # Take the first 16 hex chars (64 bits) as the numerator.
    return int(h[:16], 16) / float(1 << 64)


def assign_split(
    patient_id: str,
    seed: int = 0,
    train_frac: float = 0.85,
    val_frac: float = 0.075,
) -> SplitName:
    """Return one of 'train', 'val', 'test' for a given patient_id."""
    u = _hash_to_unit(patient_id, seed)
    if u < train_frac:
        return "train"
    if u < train_frac + val_frac:
        return "val"
    return "test"


def build_split(
    manifest_path: str,
    out_path: str,
    seed: int = 0,
    train_frac: float = 0.85,
    val_frac: float = 0.075,
) -> None:
    """Read a manifest, assign each row's split by hashing patient_id, write an
    annotated manifest + a per-patient split parquet."""
    import pandas as pd

    if train_frac + val_frac > 1.0:
        raise ValueError("train_frac + val_frac must be <= 1.0")

    df = pd.read_parquet(manifest_path)
    if "patient_id" not in df.columns:
        raise ValueError("manifest must have a patient_id column")

    df["split"] = [assign_split(p, seed=seed, train_frac=train_frac, val_frac=val_frac)
                   for p in df["patient_id"].astype(str)]

    # Sanity: no patient in two splits.
    per_pt = df.groupby("patient_id")["split"].nunique()
    if (per_pt > 1).any():
        bad = per_pt[per_pt > 1]
        raise RuntimeError(f"{len(bad)} patients span multiple splits (hash collision?)")

    df.to_parquet(out_path, index=False)
    logger.info(
        "wrote split manifest: %d rows, splits=%s",
        len(df),
        df["split"].value_counts().to_dict(),
    )
    _emit_split_report(df, str(out_path).replace(".parquet", ".split.json"), seed, train_frac, val_frac)


def _emit_split_report(df, path: str, seed: int, train_frac: float, val_frac: float) -> None:
    report = {
        "seed": seed,
        "train_frac": train_frac,
        "val_frac": val_frac,
        "test_frac": 1.0 - train_frac - val_frac,
        "clip_counts": df["split"].value_counts().to_dict(),
        "study_counts": df.groupby("split")["study_id"].nunique().to_dict(),
        "patient_counts": df.groupby("split")["patient_id"].nunique().to_dict(),
    }
    Path(path).write_text(json.dumps(report, indent=2))
    logger.info("split report: %s", path)


def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train_frac", type=float, default=0.85)
    ap.add_argument("--val_frac", type=float, default=0.075)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build_split(args.manifest, args.out, args.seed, args.train_frac, args.val_frac)


if __name__ == "__main__":
    _main()


__all__ = ["assign_split", "build_split"]
