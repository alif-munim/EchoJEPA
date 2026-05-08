"""Lightweight image-quality proxy for EchoSet-JEPA (v0.1).

The MIMIC view/color classifier CSVs do not emit an image-quality score. This
module builds a proxy from signals that *are* available per clip:

- ``view_confidence``         — low confidence = classifier is uncertain; often
                                correlates with noisy or atypical frames.
- ``video_duration_s``        — clips shorter than ~0.5s or longer than ~10s
                                are unlikely to be standard cardiac cines.
- ``fps_video``               — clips with degenerate frame rates are suspect.
- ``n_video_frames``          — very short/long clips are down-weighted.
- ``phase_annotations.quality_tier`` — an ECG-trace-based tier (NOT image
                                quality) used here only as a *reliability*
                                signal: if the ECG trace is uninterpretable,
                                we penalize the clip because phase-conditional
                                EchoSet losses will be noisier on it.

Output: ``quality_score ∈ [0, 1]``. Training-cohort tertiles become
``quality_bucket ∈ {low, med, high}``. The proxy version string is written
into the manifest so we can tell which proxy generated a given bucket.

This is explicitly a v0.1 proxy: deterministic, interpretable, and cheap. It
will be replaced (or augmented) in PR-N3 with c_clip-based signals once the
cache lands. Quality is context-only, never on the target-mask slot.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import math

QUALITY_PROXY_VERSION = "v0.1"

# ECG-trace quality tier → reliability weight. These are NOT image quality; we
# only use them to say "if phase labels are trustworthy, trust this clip more
# for phase-conditional masking."
_TIER_TO_WEIGHT = {
    "high": 1.00,
    "medium": 0.85,
    "low": 0.65,
    "reject": 0.30,    # clip kept but down-weighted; excluding reject happens upstream
}


def _clip01(x: float) -> float:
    if x != x:     # NaN
        return 0.5
    return max(0.0, min(1.0, x))


def _duration_score(duration_s: Optional[float]) -> float:
    """1.0 for clips near the cardiac-cine sweet spot (1–4 s), tapering off."""
    if duration_s is None or duration_s != duration_s:
        return 0.5
    if duration_s <= 0.0:
        return 0.0
    if 1.0 <= duration_s <= 4.0:
        return 1.0
    if duration_s < 1.0:
        return max(0.0, duration_s / 1.0)
    # >4s: taper
    return max(0.0, math.exp(-(duration_s - 4.0) / 8.0))


def _fps_score(fps: Optional[float]) -> float:
    """Penalize degenerate frame rates. 25–60 fps is the usual clinical range."""
    if fps is None or fps != fps:
        return 0.5
    if fps <= 0:
        return 0.0
    if 25.0 <= fps <= 60.0:
        return 1.0
    if fps < 25.0:
        return max(0.0, fps / 25.0)
    return max(0.0, math.exp(-(fps - 60.0) / 60.0))


def _frames_score(n_frames: Optional[int]) -> float:
    if n_frames is None:
        return 0.5
    if n_frames < 8:
        return max(0.0, n_frames / 8.0)
    if n_frames < 200:
        return 1.0
    return max(0.0, math.exp(-(n_frames - 200.0) / 200.0))


def quality_score_row(
    view_confidence: Optional[float],
    video_duration_s: Optional[float],
    fps_video: Optional[float],
    n_video_frames: Optional[int],
    ecg_quality_tier: Optional[str] = None,
) -> float:
    """Compute a single scalar quality score in [0, 1].

    Weights (must sum to 1.0):
      - 0.40 view_confidence
      - 0.15 duration
      - 0.10 fps
      - 0.10 n_frames
      - 0.25 ecg_tier (reliability, not image quality)

    Returns 0.5 when all signals are missing (neutral), not 0.0.
    """
    v = _clip01(view_confidence if view_confidence is not None else 0.5)
    d = _duration_score(video_duration_s)
    f = _fps_score(fps_video)
    n = _frames_score(n_video_frames)
    t = _TIER_TO_WEIGHT.get((ecg_quality_tier or "").lower(), 0.5)
    return 0.40 * v + 0.15 * d + 0.10 * f + 0.10 * n + 0.25 * t


def quality_scores(
    view_confidences: Sequence[Optional[float]],
    durations_s: Sequence[Optional[float]],
    fps: Sequence[Optional[float]],
    n_frames: Sequence[Optional[int]],
    ecg_tiers: Optional[Sequence[Optional[str]]] = None,
) -> list[float]:
    """Vectorized version of :func:`quality_score_row`."""
    if ecg_tiers is None:
        ecg_tiers = [None] * len(view_confidences)
    return [
        quality_score_row(vc, ds, f_, n, t)
        for vc, ds, f_, n, t in zip(view_confidences, durations_s, fps, n_frames, ecg_tiers)
    ]


def quality_buckets_from_train_tertiles(
    all_scores: Sequence[float],
    train_mask: Sequence[bool],
) -> list[str]:
    """Bucket every score into {low, med, high} using TRAIN-COHORT tertiles.

    Rationale: val/test cohorts must not leak their distribution into the
    bucket boundaries; compute tertile thresholds on train rows only, then
    apply globally.
    """
    if len(all_scores) != len(train_mask):
        raise ValueError("all_scores and train_mask must have the same length")
    train_scores = [s for s, m in zip(all_scores, train_mask) if m]
    if len(train_scores) < 10:
        # Not enough train rows to compute tertiles robustly — everyone 'unknown'.
        return ["unknown"] * len(all_scores)
    sorted_train = sorted(train_scores)
    n = len(sorted_train)
    lo = sorted_train[n // 3]
    hi = sorted_train[(2 * n) // 3]
    out: list[str] = []
    for s in all_scores:
        if s < lo:
            out.append("low")
        elif s < hi:
            out.append("med")
        else:
            out.append("high")
    return out


__all__ = [
    "QUALITY_PROXY_VERSION",
    "quality_score_row",
    "quality_scores",
    "quality_buckets_from_train_tertiles",
]
