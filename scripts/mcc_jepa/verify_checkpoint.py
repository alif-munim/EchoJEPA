"""Verify the canonical JEPA IN21K e100 checkpoint used to initialize all MCC-JEPA arms.

Expected path: checkpoints/jepa_in21k_vitl_e100.pt
Expected size: 5,127,835,835 bytes
Expected keys: encoder, predictor, target_encoder, epoch=100 (or close)
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

CKPT = Path(__file__).resolve().parents[2] / "checkpoints" / "jepa_in21k_vitl_e100.pt"
EXPECTED_SIZE = 5_127_835_835


def main() -> int:
    if not CKPT.exists():
        print(f"MISSING: {CKPT}")
        return 1
    size = CKPT.stat().st_size
    print(f"path:  {CKPT}")
    print(f"size:  {size} bytes (expected {EXPECTED_SIZE})")
    if size != EXPECTED_SIZE:
        print("WARNING: size mismatch; continuing")
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    keys = sorted(sd.keys()) if isinstance(sd, dict) else []
    print(f"top-level keys: {keys}")
    for req in ("encoder", "predictor", "target_encoder"):
        if req not in keys:
            print(f"FAIL: missing key '{req}'")
            return 1
    epoch = sd.get("epoch", "?")
    print(f"epoch: {epoch}")
    enc_params = sum(v.numel() for v in sd["encoder"].values())
    pred_params = sum(v.numel() for v in sd["predictor"].values())
    print(f"encoder params: {enc_params:,}")
    print(f"predictor params: {pred_params:,}")
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
