"""Utilities for cardiac-phase conditioning (phi-JEPA).

Provides:
  - parse_clip_id: derive a clip_id from a MIMIC-IV-Echo DICOM path or S3 URI.
  - load_phase_metadata: read the preprocessed per-clip metadata CSV.
  - compute_delta_phi: cycle-fraction offset from context mean frame to each
    target frame, using per-clip HR and the dataloader's sampled fps.

The Δφ convention matches the formula documented in
claude/neurips/experiments/phase-jepa.md:

    seconds_per_tubelet = tubelet_size / fps_sampled
    delta_phi = (target_t - ctx_t) * seconds_per_tubelet * hr_bpm / 60.0

Units: target_t / ctx_t are tubelet indices (0..D-1) where D = num_frames / tubelet_size.
hr_bpm is beats per minute. The formula returns cycle fractions.

NaN in hr_bpm (irregular rhythm or missing) propagates through multiplication
and the predictor routes those targets to a <no_phase> sentinel token.
"""

from __future__ import annotations

import os
import re
from typing import Optional

import numpy as np
import pandas as pd
import torch

_CLIP_RE = re.compile(r"/s\d+/([^/]+)\.dcm$")
_CLIP_RE_PATH = re.compile(r"/([^/]+?)(?:\.dcm|\.mp4|\.avi)?$")


def parse_clip_id(path: str) -> str:
    """Extract clip_id from a MIMIC S3 URI or filename.

    For MIMIC raw DICOM paths this returns the .dcm stem; for video mirrors
    (mp4/avi), it returns the file stem. Falls back to basename-without-ext.
    """
    m = _CLIP_RE.search(path)
    if m is not None:
        return m.group(1)
    base = os.path.basename(path)
    stem, _ = os.path.splitext(base)
    return stem


def load_phase_metadata(csv_path: str) -> dict[str, tuple[float, float, bool]]:
    """Load the preprocessed per-clip phase metadata CSV.

    Expected columns: clip_id, hr_bpm, frame_time_ms, is_irregular (bool).
    Returns dict clip_id -> (hr_bpm, frame_time_ms, is_irregular).
    """
    df = pd.read_csv(csv_path)
    required = {"clip_id", "hr_bpm", "is_irregular"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"phase metadata CSV {csv_path} missing columns {missing}; "
            f"has {list(df.columns)}"
        )

    ft_col = "frame_time_ms" if "frame_time_ms" in df.columns else None
    out: dict[str, tuple[float, float, bool]] = {}
    for row in df.itertuples(index=False):
        hr = float(row.hr_bpm) if row.hr_bpm is not None else float("nan")
        ft = float(getattr(row, ft_col)) if ft_col and getattr(row, ft_col) is not None else float("nan")
        irr = bool(row.is_irregular)
        out[str(row.clip_id)] = (hr, ft, irr)
    return out


def sanitize_hr(hr: Optional[float], is_irregular: bool, lo: float = 40.0, hi: float = 180.0) -> float:
    """Return hr_bpm if valid + regular, else nan."""
    if hr is None:
        return float("nan")
    try:
        f = float(hr)
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite(f):
        return float("nan")
    if is_irregular:
        return float("nan")
    if f < lo or f > hi:
        return float("nan")
    return f


def compute_delta_phi(
    target_t: torch.Tensor,
    ctx_t: torch.Tensor,
    hr_bpm: torch.Tensor,
    fps_sampled: float,
    tubelet_size: int,
) -> torch.Tensor:
    """Compute per-target Δφ (cycle fractions).

    Args:
        target_t: [B, N_target] target tubelet indices (float).
        ctx_t:    [B] or [B, 1] context-reference tubelet index (float, e.g. mean over context).
        hr_bpm:   [B] per-clip heart rate in bpm; may contain NaN (irregular/missing).
        fps_sampled:  dataloader sampled fps (e.g. 8).
        tubelet_size: temporal patch size in frames (e.g. 2).

    Returns:
        delta_phi: [B, N_target] tensor of cycle fractions. NaN where hr is NaN.
    """
    if ctx_t.dim() == 1:
        ctx_t = ctx_t.unsqueeze(-1)  # [B, 1]
    hr = hr_bpm.unsqueeze(-1).to(target_t.dtype)  # [B, 1]
    seconds_per_tubelet = float(tubelet_size) / float(fps_sampled)
    return (target_t - ctx_t) * seconds_per_tubelet * hr / 60.0
