#!/usr/bin/env python
"""Standalone smoke test for phase-aware MaskCollator.

Builds a synthetic batch of 4 samples with known HR / FrameTime, runs the
collator in both vanilla and phase-aware modes, prints bucket counts and
dphi stats, and sanity-checks that shapes and invariants hold. Exits 0 on
success, 1 otherwise.

Run:
    cd /path/to/vjepa2
    python scripts/neurips/phase/smoke_test_phase_mask.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.masks.multiseq_multiblock3d import MaskCollator  # noqa: E402


MASK_CFGS = [
    # Phase-aware masking needs localized target blocks (temporal_scale<1.0)
    # so there is a non-trivial set of valid t_start positions to gate on phase.
    {
        "aspect_ratio": [0.75, 1.5],
        "full_complement": False,
        "max_keep": None,
        "max_temporal_keep": 1.0,
        "num_blocks": 2,
        "spatial_scale": [0.15, 0.15],
        "temporal_scale": [0.25, 0.25],
    },
]

PHASE_CFG = {
    "phase_aware": True,
    "phase_buckets": {
        "local": [0.05, 0.15],
        "mid_cycle": [0.20, 0.35],
        "opposite_phase": [0.45, 0.55],
        "same_phase_next_beat": True,
    },
    "phase_bucket_probs": {
        "local": 0.25,
        "mid_cycle": 0.35,
        "opposite_phase": 0.25,
        "same_phase_next_beat": 0.15,
    },
    "phase_fallback": "random",
    "require_valid_hr": True,
    "shuffled_hr": False,
    "phase_max_attempts": 20,
    "phase_seed": 42,
}


def make_sample(fpc=16, hr=70.0, ft=33.3, nf=60):
    buf = torch.zeros((3, fpc, 16, 16), dtype=torch.float32)
    label = torch.tensor(0, dtype=torch.long)
    clip_indices = [list(range(fpc))]
    meta = {"hr_bpm": float(hr), "frame_time_ms": float(ft), "num_frames": int(nf)}
    return (buf, label, clip_indices, "/tmp/dummy.mp4", meta)


def run_vanilla():
    coll = MaskCollator(
        cfgs_mask=MASK_CFGS,
        dataset_fpcs=[16],
        crop_size=(224, 224),
        patch_size=(16, 16),
        tubelet_size=2,
    )
    batch = [make_sample(hr=70.0 + i * 5) for i in range(4)]
    out = coll(batch)
    assert len(out) == 1, "vanilla: expected 1 fpc collation"
    _, masks_enc, masks_pred = out[0]
    assert masks_enc[0].shape[0] == 4
    assert masks_pred[0].shape[0] == 4
    print("[vanilla] enc shape", tuple(masks_enc[0].shape),
          "pred shape", tuple(masks_pred[0].shape))


def run_phase_aware(cfg, label):
    coll = MaskCollator(
        cfgs_mask=MASK_CFGS,
        dataset_fpcs=[16],
        crop_size=(224, 224),
        patch_size=(16, 16),
        tubelet_size=2,
        fps_sampled=8.0,
        phase_mask_cfg=cfg,
    )
    batch = [make_sample(hr=60.0 + i * 6) for i in range(4)]
    out = coll(batch)
    assert len(out) == 1, f"{label}: expected 1 fpc collation"
    _, masks_enc, masks_pred = out[0]
    s = coll.stats.summarize()
    print(f"[{label}] enc shape", tuple(masks_enc[0].shape),
          "pred shape", tuple(masks_pred[0].shape))
    for k, v in s.items():
        print(f"    {k}: {v}")
    # Minimal invariants.
    assert masks_enc[0].shape[0] == 4
    assert masks_pred[0].shape[0] == 4
    # At least some clips should have valid metadata.
    assert s["n_valid_meta"] >= 1


def run_phase_aware_nan_meta():
    coll = MaskCollator(
        cfgs_mask=MASK_CFGS,
        dataset_fpcs=[16],
        crop_size=(224, 224),
        patch_size=(16, 16),
        tubelet_size=2,
        fps_sampled=8.0,
        phase_mask_cfg=PHASE_CFG,
    )
    batch = [make_sample(hr=float("nan")) for _ in range(4)]
    out = coll(batch)
    _, masks_enc, masks_pred = out[0]
    s = coll.stats.summarize()
    print("[phase_aware nan-meta] fallback frac:",
          s["n_fallback_invalid_meta"] / max(1, s["n_clips"]))
    assert masks_enc[0].shape[0] == 4
    assert masks_pred[0].shape[0] == 4
    assert s["n_fallback_invalid_meta"] == 4


def main() -> int:
    try:
        run_vanilla()
        run_phase_aware(PHASE_CFG, "phase_aware")
        shuffled_cfg = dict(PHASE_CFG)
        shuffled_cfg["shuffled_hr"] = True
        run_phase_aware(shuffled_cfg, "phase_aware_shuffled_hr")
        run_phase_aware_nan_meta()
    except AssertionError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"FAIL: {e}", file=sys.stderr)
        return 1
    print("\nSMOKE PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
