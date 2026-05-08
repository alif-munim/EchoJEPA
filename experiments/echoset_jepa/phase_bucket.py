"""Per-clip phase_bucket derivation from ``per_frame_phase_json``.

Convention (standard echo physiology):
  phase ∈ [0, 0.35] → systole
  phase ∈ [0.35, 1.0] → diastole

``per_frame_phase_json`` (from
``classifier/phase/phase_annotations/phase_annotations.parquet``) is a list of
floats ∈ [0, 1) giving each frame's position within the current RR interval,
with ``None`` for frames outside any confident ECG window.

For EchoSet-JEPA pretraining we cache one c_clip per clip over its full
frame window. The clip-level phase_bucket is therefore a coarse descriptor
of what the clip contains:

  - ``full_cycle``: at least one complete cardiac cycle is covered
    (≥2 R-peaks, or phase range spans ≥0.8).
  - ``systolic`` / ``diastolic``: the clip covers a narrow phase window
    dominated by one phase (>60% of confident frames in that phase) and
    does not cover a full cycle.
  - ``unknown``: too few confident frames to classify.

In practice MIMIC clips are overwhelmingly ``full_cycle`` (median ~2.6s at
~30fps → 78 frames, almost always ≥2 R-peaks). Single-phase clips will be
rare but taxonomically distinct.
"""

from __future__ import annotations

import json
from typing import Iterable, List, Optional, Sequence


SYSTOLE_FRACTION = 0.35   # standard systolic/diastolic boundary


def _parse_phases(raw: Optional[str]) -> List[Optional[float]]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return list(raw)
    if not raw:
        return []
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []


def _parse_confident_mask(raw) -> List[int]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return list(raw)
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []


def _parse_rpeaks(raw) -> List[int]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return list(raw)
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []


def derive_clip_phase_bucket(
    per_frame_phase_json: Optional[str],
    confident_mask_json: Optional[str] = None,
    r_peaks_video_json: Optional[str] = None,
    min_confident_fraction: float = 0.3,
    full_cycle_phase_range: float = 0.8,
    dominant_phase_fraction: float = 0.6,
) -> str:
    """Return one of {systolic, diastolic, full_cycle, unknown}.

    Rules:
      1. If too few confident frames (< ``min_confident_fraction``), return ``unknown``.
      2. If ``r_peaks_video_json`` lists ≥2 R-peaks, return ``full_cycle``.
      3. If the phase span ≥ ``full_cycle_phase_range``, return ``full_cycle``.
      4. If > ``dominant_phase_fraction`` of confident frames are in systole,
         return ``systolic``. Same for diastole.
      5. Otherwise ``full_cycle`` (mixed-phase, partial coverage).
    """
    phases = _parse_phases(per_frame_phase_json)
    if not phases:
        return "unknown"

    # Confident mask: 1 for frames with trustworthy ECG, 0 otherwise.
    mask = _parse_confident_mask(confident_mask_json)
    if mask and len(mask) == len(phases):
        confident = [p for p, m in zip(phases, mask) if m and p is not None]
    else:
        confident = [p for p in phases if p is not None]

    total = len(phases)
    if total == 0 or len(confident) / total < min_confident_fraction:
        return "unknown"

    # Rule 2: explicit R-peak count, if available.
    rpeaks = _parse_rpeaks(r_peaks_video_json)
    if len(rpeaks) >= 2:
        return "full_cycle"

    # Rule 3: phase span.
    try:
        span = max(confident) - min(confident)
    except ValueError:
        return "unknown"
    if span >= full_cycle_phase_range:
        return "full_cycle"

    # Rule 4: dominant phase.
    n_sys = sum(1 for p in confident if p < SYSTOLE_FRACTION)
    n_dia = sum(1 for p in confident if p >= SYSTOLE_FRACTION)
    total_confident = n_sys + n_dia
    if total_confident == 0:
        return "unknown"
    if n_sys / total_confident > dominant_phase_fraction:
        return "systolic"
    if n_dia / total_confident > dominant_phase_fraction:
        return "diastolic"

    # Rule 5: mixed but partial — call it full_cycle.
    return "full_cycle"


def derive_phase_buckets_batch(
    per_frame_phase_jsons: Sequence[Optional[str]],
    confident_masks: Optional[Sequence] = None,
    r_peaks: Optional[Sequence] = None,
) -> List[str]:
    n = len(per_frame_phase_jsons)
    masks = confident_masks if confident_masks is not None else [None] * n
    rpks = r_peaks if r_peaks is not None else [None] * n
    return [
        derive_clip_phase_bucket(p, m, r)
        for p, m, r in zip(per_frame_phase_jsons, masks, rpks)
    ]


__all__ = [
    "SYSTOLE_FRACTION",
    "derive_clip_phase_bucket",
    "derive_phase_buckets_batch",
]
