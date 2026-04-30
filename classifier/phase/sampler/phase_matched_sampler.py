"""PhaseMatchedStudySampler — per-epoch generator of within-study clip
pairs for multi-view JEPA training.

Turn-1 rewrite extends the original sampler with:
  - ``sampling_mode`` dispatch: uniform_phase | ed_es_biased | wrong_phase
    | same_study_random
  - ``phase_tolerance`` — resample when no confident frame is within this
    tolerance of the requested phi
  - ``frames_per_clip`` / ``frame_step`` awareness — pairs know their
    temporal source span in frames, seconds, and cycles
  - rich ``MatchRecord`` with per-clip view, HR, fps, source-span metadata
  - ``build_anchor_table(side=...)`` emits the ``{row_idx: [{..}]}`` dict
    the updated ``VideoGroupDataset`` consumes via ``set_anchors_by_index``
  - CLI dry-run (``python phase_matched_sampler.py --dry-run ...``) that
    reports diagnostics without loading video

The sampler yields one "pair record" per iteration. Two consumption
patterns:
  1. ``__iter__`` yields a single row index (clip_b by default) for
     DataLoader-compatible sampling. Use ``side="b"`` (default) or
     ``side="a"``.
  2. ``last_records`` exposes the full ``MatchRecord`` list for a paired
     loader wrapper that needs both clip_a and clip_b (Turn 2).

The class is DDP-aware: ``num_replicas`` / ``rank`` auto-detected from
``torch.distributed`` if initialized.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from logging import getLogger
from pathlib import Path
from typing import Iterator, Optional, Sequence

import numpy as np
import pandas as pd

# Local import: standalone RR utility module (Turn 1).
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from rr_consistency import rr_stats, rr_consistent  # noqa: E402

logger = getLogger(__name__)


SAMPLING_MODES = (
    "uniform_phase", "ed_es_biased", "wrong_phase", "same_study_random",
    "phase_curriculum",
)
RR_FILTER_MODES = ("strict", "permissive_afib")


# --------------------------------------------------------------------------- #
# View-distance curriculum: canonical taxonomy + bucket lookup
# --------------------------------------------------------------------------- #
# Families used to decide "apical vs parasternal_long vs parasternal_short"
# distance. UNKNOWN is intentionally its own family so cross-family pairs
# with UNKNOWN land in the "hard" bucket unless the caller disables it.
VIEW_FAMILIES = {
    "A2C": "apical",
    "A3C": "apical",
    "A4C": "apical",
    "A5C": "apical",
    "PLAX": "parasternal_long",
    "PSAX-AV": "parasternal_short",
    "PSAX-MV": "parasternal_short",
    "PSAX-PM": "parasternal_short",
    "PSAX-AP": "parasternal_short",
    "Subcostal": "other",
    "SUBCOSTAL": "other",
    "IVC": "other",
    "SSN": "other",
    "TEE": "other",
    "UNKNOWN": "other",
}

# Pairs that count as "easy" despite not being identical views. Order-
# agnostic; normalized to a frozenset at lookup.
_VIEW_EASY_PAIRS = frozenset({
    frozenset(("A4C", "A5C")),
    frozenset(("A4C", "A3C")),
    frozenset(("A2C", "A3C")),
    frozenset(("PSAX-MV", "PSAX-PM")),
    frozenset(("PSAX-MV", "PSAX-AP")),
    frozenset(("PSAX-AV", "PSAX-MV")),
})

# "medium" = within-family but not nearest-neighbor, OR parasternal-long ↔
# parasternal-short. Anything else within a family is easy; anything else
# across families is hard.
_VIEW_MEDIUM_PAIRS = frozenset({
    frozenset(("A4C", "A2C")),
    frozenset(("A5C", "A2C")),
    frozenset(("PSAX-AV", "PSAX-PM")),
    frozenset(("PSAX-AV", "PSAX-AP")),
    frozenset(("PSAX-PM", "PSAX-AP")),
    # Cross-parasternal (long <-> short) is medium
    frozenset(("PLAX", "PSAX-AV")),
    frozenset(("PLAX", "PSAX-MV")),
    frozenset(("PLAX", "PSAX-PM")),
    frozenset(("PLAX", "PSAX-AP")),
})


def view_pair_class(view_a: Optional[str], view_b: Optional[str]) -> str:
    """Coarser-grained classification used by view_pair_policy.

    Returns one of:
    * ``same_view``: same canonical known view (UNKNOWN not allowed)
    * ``same_family``: near-view pair per _VIEW_EASY_PAIRS OR same family
      but not nearest-neighbor OR parasternal-long↔short (medium bucket)
    * ``cross_family``: different families, or UNKNOWN on either side

    Maps to a 3-way policy distribution (same_view / same_family / cross_family)
    instead of the 3-way distance-bucket (easy/medium/hard) used by curriculum.
    """
    a = (view_a or "UNKNOWN").upper()
    b = (view_b or "UNKNOWN").upper()
    if a == "SUBCOSTAL":
        a = "Subcostal"
    if b == "SUBCOSTAL":
        b = "Subcostal"
    if a == "UNKNOWN" or b == "UNKNOWN" or a not in VIEW_FAMILIES or b not in VIEW_FAMILIES:
        return "cross_family"
    if a == b:
        return "same_view"
    pair = frozenset((a, b))
    if pair in _VIEW_EASY_PAIRS or pair in _VIEW_MEDIUM_PAIRS:
        return "same_family"
    fam_a = VIEW_FAMILIES.get(a, "other")
    fam_b = VIEW_FAMILIES.get(b, "other")
    if "other" in (fam_a, fam_b):
        return "cross_family"
    if fam_a == fam_b:
        return "same_family"
    return "cross_family"


def view_distance_bucket(view_a: Optional[str], view_b: Optional[str]) -> str:
    """Return 'easy', 'medium', or 'hard'.

    Semantics:
    * ``easy``: same known view OR in _VIEW_EASY_PAIRS.
    * ``medium``: explicitly in _VIEW_MEDIUM_PAIRS, or same family but not
      nearest-neighbor.
    * ``hard``: different families, or at least one side is UNKNOWN / unmapped.

    If either input is None/UNKNOWN, the pair is "hard" — the two views
    being identical doesn't count as easy if we don't know what view it is.
    """
    a = (view_a or "UNKNOWN").upper()
    b = (view_b or "UNKNOWN").upper()
    def _canon(v: str) -> str:
        if v == "SUBCOSTAL":
            return "Subcostal"
        return v
    a = _canon(a)
    b = _canon(b)
    # Any UNKNOWN / unmapped side → hard (even if both UNKNOWN).
    if a == "UNKNOWN" or b == "UNKNOWN" or a not in VIEW_FAMILIES or b not in VIEW_FAMILIES:
        return "hard"
    if a == b:
        return "easy"
    pair = frozenset((a, b))
    if pair in _VIEW_EASY_PAIRS:
        return "easy"
    if pair in _VIEW_MEDIUM_PAIRS:
        return "medium"
    fam_a = VIEW_FAMILIES.get(a, "other")
    fam_b = VIEW_FAMILIES.get(b, "other")
    if "other" in (fam_a, fam_b):
        return "hard"
    if fam_a == fam_b:
        # same family but not nearest-neighbor → medium by default. Covers
        # e.g. A5C ↔ A3C which we didn't enumerate above.
        return "medium"
    return "hard"


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ClipAnchor:
    row_idx: int
    dicom_id: str
    n_frames: int
    anchor_frame: int
    phase_at_anchor: float
    phase_error: float          # circular distance to requested target_phi
    view: Optional[str] = None
    hr_metadata: Optional[float] = None
    fps_video: Optional[float] = None
    quality_tier: Optional[str] = None
    rr_consistent: Optional[bool] = None


@dataclass(frozen=True)
class MatchRecord:
    study_id: str
    subject_id: Optional[str]
    acquisition_datetime_a: Optional[str]
    acquisition_datetime_b: Optional[str]
    clip_a: ClipAnchor
    clip_b: ClipAnchor
    target_phi_a: float
    target_phi_b: float
    circular_phase_diff: float
    sampling_mode: str
    frame_step: int
    frames_per_clip: int
    source_span_frames: int
    source_span_seconds_a: float
    source_span_seconds_b: float
    source_span_cycles_a: float
    source_span_cycles_b: float
    # Curriculum diagnostics (filled in phase_curriculum mode; None otherwise)
    view_distance_bucket: Optional[str] = None
    view_distance_numeric: Optional[int] = None
    view_family_a: Optional[str] = None
    view_family_b: Optional[str] = None
    curriculum_epoch_frac: Optional[float] = None
    curriculum_bucket_probs: Optional[str] = None  # "easy=0.70,medium=0.25,hard=0.05"


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def circular_phase_distance(a: float, b: float) -> float:
    d = abs(float(a) - float(b))
    return min(d, 1.0 - d)


def _decode_phase(row) -> tuple[np.ndarray, np.ndarray]:
    """Return (phase[n_frames], confident[n_frames] as bool)."""
    phase = np.array(
        [np.nan if v is None else float(v) for v in json.loads(row.per_frame_phase_json)],
        dtype=np.float64,
    )
    confident = np.array(json.loads(row.confident_mask_json), dtype=bool)
    return phase, confident


def nearest_confident_frame(
    phase: np.ndarray,
    confident: np.ndarray,
    target_phi: float,
    tolerance: Optional[float] = None,
) -> Optional[tuple[int, float, float]]:
    """Pick the confident frame whose phase is closest to ``target_phi``
    under circular distance.

    Returns ``(frame_idx, phase_at_frame, phase_error)`` or ``None`` if no
    confident frame exists or (when tolerance is set) the nearest is too
    far.
    """
    if not confident.any():
        return None
    idx = np.where(confident)[0]
    ph = phase[idx]
    d = np.abs(ph - target_phi)
    d = np.minimum(d, 1.0 - d)
    pick = int(np.argmin(d))
    err = float(d[pick])
    if tolerance is not None and err > tolerance:
        return None
    return int(idx[pick]), float(ph[pick]), err


def _sample_from_windows(
    rng: np.random.Generator,
    windows: Sequence[tuple[float, float]],
) -> float:
    """Sample uniformly from a union of [lo, hi) intervals on [0, 1)."""
    spans = [max(0.0, hi - lo) for lo, hi in windows]
    total = float(sum(spans))
    if total <= 0.0:
        return float(rng.random())
    cumulative = np.cumsum(spans)
    x = float(rng.random()) * total
    i = int(np.searchsorted(cumulative, x, side="right"))
    i = min(i, len(windows) - 1)
    lo, _ = windows[i]
    offset = x - (cumulative[i - 1] if i > 0 else 0.0)
    return float((lo + offset) % 1.0)


# --------------------------------------------------------------------------- #
# Sampler
# --------------------------------------------------------------------------- #

class PhaseMatchedStudySampler:
    """Draw phase-matched within-study clip pairs.

    Parameters
    ----------
    parquet_path : path to phase_annotations.parquet.
    tiers : quality tiers to include. Default ("high",) for the first
        pilot; pass ("high", "medium") for later runs.
    require_rr_consistent : apply the median-RR-vs-metadata filter before
        grouping by study. Default True.
    rr_filter_mode : ``"strict"`` (default) rejects both every-Nth-beat
        errors (median-vs-metadata) AND missed-beat errors (max/min RR
        ratio > 1.40). ``"permissive_afib"`` disables the max/min check
        so AFib patients survive, at the cost of admitting missed-beat
        clips. First pilot configs must use ``"strict"``.
    rr_meta_ratio_range : tolerance for median(RR) / metadata_cycle.
    sampling_mode : see SAMPLING_MODES.
    phase_tolerance : max circular distance from the requested phi to the
        nearest confident frame; pair is skipped if exceeded.
    frames_per_clip, frame_step : temporal-window geometry used to compute
        source_span diagnostics and (optionally) exclude clips that can't
        fit the span.
    pairs_per_study : how many pair records to draw per study per epoch.
    same_session_only : if True and acquisition_datetime is available,
        pair only clips that share the exact acquisition_datetime.
    ed_probability, ed_windows, es_windows : ed_es_biased mode config.
    resample_attempts : number of times to retry within a study when the
        phase_tolerance check fails on either clip.
    seed, num_replicas, rank, drop_last : distributed-sampler plumbing.
    prefer_different_views : when True and view labels are available via
        ``view_labels`` map (dicom_id -> label), prefer cross-view pairs.
    view_labels : optional dict {dicom_id: view_label}. If None, pairs are
        not constrained by view.
    min_frames : drop clips with ``n_video_frames < min_frames``. Default
        None — no constraint.
    require_span_fits : if True, drop clips with
        ``n_video_frames < source_span_frames`` (padded-short clips stay
        out of the training set).
    """

    def __init__(
        self,
        parquet_path: str | Path,
        tiers: Sequence[str] = ("high",),
        require_rr_consistent: bool = True,
        rr_filter_mode: str = "strict",
        rr_meta_ratio_range: tuple[float, float] = (0.80, 1.25),
        rr_max_min_ratio: float = 1.40,
        sampling_mode: str = "uniform_phase",
        phase_tolerance: float = 0.15,
        frames_per_clip: int = 16,
        frame_step: int = 1,
        pairs_per_study: int = 1,
        same_session_only: bool = False,
        ed_probability: float = 0.5,
        ed_windows: Sequence[tuple[float, float]] = ((0.95, 1.0), (0.0, 0.05)),
        es_windows: Sequence[tuple[float, float]] = ((0.30, 0.45),),
        resample_attempts: int = 4,
        seed: int = 0,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        drop_last: bool = False,
        prefer_different_views: bool = True,
        view_labels: Optional[dict[str, str]] = None,
        view_confidences: Optional[dict[str, float]] = None,
        min_view_confidence: float = 0.0,
        min_frames: Optional[int] = None,
        require_span_fits: bool = False,
        # Curriculum config: a schedule of
        # {start_frac, end_frac, bucket_probs={easy:..., medium:..., hard:...}}
        curriculum: Optional[dict] = None,
        total_epochs: int = 1,
        # View-pair mixture policy. When enabled, each pair draw samples a
        # view-pair class (same_view / same_family / cross_family) from the
        # target probability vector and then picks two clips whose view
        # labels satisfy that class. Applies on top of uniform_phase,
        # same_study_random, wrong_phase. For phase_curriculum, a
        # same_view floor is applied to every stage of the schedule so the
        # encoder always sees a meaningful fraction of true-view positives.
        # Schema:
        #   {enabled: bool,
        #    same_view_prob: float,
        #    same_family_prob: float,
        #    cross_family_prob: float,
        #    require_different_dicom: bool (default True),
        #    allow_same_view: bool (default True),
        #    curriculum_same_view_floor: {early,middle,late: float},  # opt
        #    resample_attempts: int (default 8)}
        view_pair_policy: Optional[dict] = None,
    ) -> None:
        if sampling_mode not in SAMPLING_MODES:
            raise ValueError(f"unknown sampling_mode={sampling_mode}; want one of {SAMPLING_MODES}")
        if rr_filter_mode not in RR_FILTER_MODES:
            raise ValueError(f"unknown rr_filter_mode={rr_filter_mode}; want one of {RR_FILTER_MODES}")
        if sampling_mode == "phase_curriculum":
            if not view_labels:
                raise ValueError(
                    "sampling_mode='phase_curriculum' requires view_labels "
                    "(dict dicom_id->view). Pass them via data_manager config "
                    "field view_labels_path."
                )
            if not curriculum or not curriculum.get("enabled", False):
                raise ValueError(
                    "sampling_mode='phase_curriculum' requires "
                    "curriculum.enabled=True with a schedule."
                )
            if curriculum.get("type", "view_distance") != "view_distance":
                raise ValueError(
                    f"curriculum.type={curriculum.get('type')!r} not supported; "
                    f"only 'view_distance' is implemented."
                )

        self.parquet_path = Path(parquet_path)
        self.tiers = tuple(tiers)
        self.require_rr_consistent = bool(require_rr_consistent)
        self.rr_filter_mode = str(rr_filter_mode)
        self.rr_meta_ratio_range = tuple(rr_meta_ratio_range)
        self.rr_max_min_ratio = float(rr_max_min_ratio)
        self.sampling_mode = str(sampling_mode)
        self.phase_tolerance = float(phase_tolerance)
        self.frames_per_clip = int(frames_per_clip)
        self.frame_step = int(frame_step)
        self.pairs_per_study = int(pairs_per_study)
        self.same_session_only = bool(same_session_only)
        self.ed_probability = float(ed_probability)
        self.ed_windows = tuple(tuple(w) for w in ed_windows)
        self.es_windows = tuple(tuple(w) for w in es_windows)
        self.resample_attempts = int(resample_attempts)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.prefer_different_views = bool(prefer_different_views)
        self.view_labels = dict(view_labels) if view_labels else {}
        self.view_confidences = dict(view_confidences) if view_confidences else {}
        self.min_view_confidence = float(min_view_confidence)
        self.min_frames = None if min_frames is None else int(min_frames)
        self.require_span_fits = bool(require_span_fits)
        self.source_span_frames = (self.frames_per_clip - 1) * self.frame_step + 1
        self.epoch = 0
        # Curriculum setup (only used when sampling_mode='phase_curriculum').
        self.curriculum = dict(curriculum) if curriculum else None
        self.total_epochs = max(1, int(total_epochs))
        # Per-epoch bucket-probability cache (filled on each build_records())
        self._last_curriculum_probs: Optional[dict[str, float]] = None
        self._last_curriculum_epoch_frac: Optional[float] = None
        self._last_curriculum_skipped: dict[str, int] = {}

        # View-pair policy normalization.
        self.view_pair_policy = None
        if view_pair_policy and view_pair_policy.get("enabled", False):
            p = dict(view_pair_policy)
            probs = {
                "same_view": float(p.get("same_view_prob", 0.0)),
                "same_family": float(p.get("same_family_prob", 0.0)),
                "cross_family": float(p.get("cross_family_prob", 0.0)),
            }
            tot = sum(probs.values()) or 1.0
            probs = {k: v / tot for k, v in probs.items()}
            self.view_pair_policy = {
                "enabled": True,
                "probs": probs,
                "require_different_dicom": bool(p.get("require_different_dicom", True)),
                "allow_same_view": bool(p.get("allow_same_view", True)),
                "resample_attempts": int(p.get("resample_attempts", 8)),
                "curriculum_same_view_floor": dict(p.get("curriculum_same_view_floor", {})),
            }
        self._last_view_pair_skipped: dict[str, int] = {"same_view": 0, "same_family": 0, "cross_family": 0}

        # DDP auto-detect.
        if num_replicas is None or rank is None:
            try:
                import torch.distributed as dist

                if dist.is_available() and dist.is_initialized():
                    num_replicas = num_replicas or dist.get_world_size()
                    rank = rank if rank is not None else dist.get_rank()
                else:
                    num_replicas = num_replicas or 1
                    rank = rank if rank is not None else 0
            except ImportError:
                num_replicas = num_replicas or 1
                rank = rank if rank is not None else 0
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)

        self._load()

    # ---------- load / filter / group ------------------------------------ #

    def _load(self) -> None:
        need = [
            "dicom_id", "study_id", "s3_uri", "n_video_frames", "fps_video",
            "hr_metadata", "quality_tier", "n_rpeaks_in_video",
            "rpeak_ratio_dist", "coverage_frac", "r_peaks_video_json",
            "per_frame_phase_json", "confident_mask_json",
        ]
        extras = []
        # Optional columns: subject_id, acquisition_datetime, dicom_filepath.
        probe = pd.read_parquet(self.parquet_path, columns=[c for c in need if c in [c]])
        # We don't know the full set until we read — do a cheap schema check.
        schema = pd.read_parquet(self.parquet_path, columns=["dicom_id"]).columns  # noqa: F841
        full_cols = set(pd.read_parquet(self.parquet_path, engine="pyarrow").columns) if False else None
        # Safer: read once with needed cols, then try to add optional ones.
        df = pd.read_parquet(self.parquet_path, columns=need)
        for opt in ("subject_id", "acquisition_datetime", "dicom_filepath"):
            try:
                df[opt] = pd.read_parquet(self.parquet_path, columns=[opt])[opt].values
            except Exception:
                df[opt] = None

        n_all = len(df)
        df = df[df.quality_tier.isin(self.tiers)].reset_index(drop=True)
        n_tier = len(df)

        # RR consistency. Two-layer check; second layer (max/min RR ratio)
        # is gated by rr_filter_mode. Also compute both counts for the
        # dry-run summary regardless of which mode is active.
        if self.require_rr_consistent:
            permissive_mask = df.apply(
                lambda r: rr_consistent(
                    r, median_tol=self.rr_meta_ratio_range, max_min_rr_ratio=None
                ),
                axis=1,
            ).values
            strict_mask = df.apply(
                lambda r: rr_consistent(
                    r,
                    median_tol=self.rr_meta_ratio_range,
                    max_min_rr_ratio=self.rr_max_min_ratio,
                ),
                axis=1,
            ).values
            n_after_permissive = int(permissive_mask.sum())
            n_after_strict = int(strict_mask.sum())
            keep_mask = strict_mask if self.rr_filter_mode == "strict" else permissive_mask
            df = df[keep_mask].reset_index(drop=True)
        else:
            n_after_permissive = n_after_strict = len(df)
        n_rr = len(df)

        # Optional minimum-length filter.
        dropped_short = 0
        if self.min_frames is not None:
            short = df.n_video_frames < self.min_frames
            dropped_short = int(short.sum())
            df = df[~short].reset_index(drop=True)
        dropped_span = 0
        if self.require_span_fits:
            short = df.n_video_frames < self.source_span_frames
            dropped_span = int(short.sum())
            df = df[~short].reset_index(drop=True)
        n_pool = len(df)

        # Row index in the filtered frame is the stable pair-loader key.
        df["row_idx_"] = df.index

        # Attach view labels if provided. Rows with confidence below
        # min_view_confidence get view='UNKNOWN' so curriculum assigns them
        # to the hard bucket (or filters them out via the study gate below).
        def _view_lookup(d: str) -> Optional[str]:
            v = self.view_labels.get(str(d))
            if v is None:
                return None
            if self.min_view_confidence > 0.0:
                c = self.view_confidences.get(str(d))
                if c is None or c < self.min_view_confidence:
                    return "UNKNOWN"
            return v
        df["view"] = df.dicom_id.map(_view_lookup)
        df["view_confidence"] = df.dicom_id.map(
            lambda d: self.view_confidences.get(str(d), float("nan"))
        )

        # Group by study (or by study + acquisition_datetime if session-only).
        if self.same_session_only and df.acquisition_datetime.notna().any():
            df["_group_key"] = df[["study_id", "acquisition_datetime"]].astype(str).agg("|".join, axis=1)
        else:
            df["_group_key"] = df.study_id.astype(str)

        study_to_rows: dict[str, list[int]] = defaultdict(list)
        for _, r in df.iterrows():
            study_to_rows[str(r._group_key)].append(int(r.row_idx_))
        study_to_rows = {k: v for k, v in study_to_rows.items() if len(v) >= 2}
        self.study_to_rows = study_to_rows
        self.study_keys = sorted(study_to_rows.keys())
        self.n_studies = len(self.study_keys)

        self._df = df

        # Precomputed per-study pair indices. Built once here so
        # _pick_pair_rows_{viewpair,curriculum} can sample in O(1) instead
        # of reclassifying all C(N,2) pairs per draw. Uses numpy-backed
        # views array for fast scalar-free indexing.
        # Layout:
        #   self._pair_index_viewpair[key] = {
        #       "same_view":    list[(ra, rb)],
        #       "same_family":  list[(ra, rb)],
        #       "cross_family": list[(ra, rb)],
        #   }
        #   self._pair_index_curriculum[key] = {
        #       "easy":   list[(ra, rb)],
        #       "medium": list[(ra, rb)],
        #       "hard":   list[(ra, rb)],
        #   }
        self._pair_index_viewpair: dict[str, dict[str, list[tuple[int, int]]]] = {}
        self._pair_index_curriculum: dict[str, dict[str, list[tuple[int, int]]]] = {}
        # Numpy-pulled view column indexed by the filtered-df integer index.
        # After reset_index(drop=True) the row_idx_ we stored equals the
        # position in this array.
        view_arr = df["view"].to_numpy(dtype=object)
        t_cache0 = time.time()
        for key in self.study_keys:
            rows = study_to_rows[key]
            views = [view_arr[r] for r in rows]
            vp: dict[str, list[tuple[int, int]]] = {"same_view": [], "same_family": [], "cross_family": []}
            cb: dict[str, list[tuple[int, int]]] = {"easy": [], "medium": [], "hard": []}
            n = len(rows)
            for ii in range(n):
                ra = int(rows[ii])
                va = views[ii]
                for jj in range(ii + 1, n):
                    rb = int(rows[jj])
                    vb = views[jj]
                    vp[view_pair_class(va, vb)].append((ra, rb))
                    cb[view_distance_bucket(va, vb)].append((ra, rb))
            self._pair_index_viewpair[key] = vp
            self._pair_index_curriculum[key] = cb
        logger.info(
            "PhaseMatchedStudySampler: built per-study pair-index cache over "
            "%d studies in %.1fs",
            self.n_studies, time.time() - t_cache0,
        )
        self._filter_stats = {
            "n_all": int(n_all),
            "n_after_tier": int(n_tier),
            "n_after_rr": int(n_rr),
            "rr_filter_mode": self.rr_filter_mode,
            "n_after_rr_strict": int(n_after_strict) if self.require_rr_consistent else None,
            "n_after_rr_permissive": int(n_after_permissive) if self.require_rr_consistent else None,
            "rr_strict_drop_vs_permissive": (
                None if not self.require_rr_consistent
                else int(n_after_permissive - n_after_strict)
            ),
            "dropped_short_frames": dropped_short,
            "dropped_span_fits": dropped_span,
            "n_pool": int(n_pool),
            "n_studies_multi_clip": int(self.n_studies),
            "tiers": self.tiers,
            "require_rr_consistent": self.require_rr_consistent,
            "frames_per_clip": self.frames_per_clip,
            "frame_step": self.frame_step,
            "source_span_frames": int(self.source_span_frames),
        }

        logger.info(
            "PhaseMatchedStudySampler: %d -> %d (tier) -> %d (rr) -> %d (span/min) clips; "
            "%d multi-clip groups; mode=%s fpc=%d fs=%d source_span=%d",
            n_all, n_tier, n_rr, n_pool, self.n_studies,
            self.sampling_mode, self.frames_per_clip, self.frame_step, self.source_span_frames,
        )

        # Per-rank sizing (DistributedSampler style).
        n_records = self.n_studies * self.pairs_per_study
        if self.drop_last and n_records % self.num_replicas != 0:
            self.num_samples = math.ceil((n_records - self.num_replicas) / self.num_replicas)
        else:
            self.num_samples = math.ceil(n_records / max(1, self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    # ---------- per-epoch draw ------------------------------------------- #

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _current_bucket_probs(self, epoch_frac: float) -> dict[str, float]:
        """Select bucket-probs stage from curriculum schedule based on
        fractional epoch position. Returns a normalized {easy, medium, hard}
        dict summing to 1.0. Uses the last stage whose [start_frac, end_frac)
        contains epoch_frac, or the final stage if epoch_frac == 1.0.
        """
        if not self.curriculum or not self.curriculum.get("enabled", False):
            raise RuntimeError("_current_bucket_probs called without curriculum")
        schedule = self.curriculum["schedule"]
        selected = schedule[0]
        for stage in schedule:
            s, e = float(stage["start_frac"]), float(stage["end_frac"])
            if s <= epoch_frac < e or (epoch_frac >= 1.0 and e >= 1.0):
                selected = stage
        probs = dict(selected["bucket_probs"])
        total = sum(probs.values()) or 1.0
        return {k: probs.get(k, 0.0) / total for k in ("easy", "medium", "hard")}

    def _bucket_pairs_for_study(
        self, group_key: str
    ) -> dict[str, list[tuple[int, int]]]:
        """Return the precomputed {easy,medium,hard} pair lists for a study.
        Uses ``self._pair_index_curriculum`` built in ``_load``."""
        return self._pair_index_curriculum[group_key]

    def _pick_pair_rows(self, group_key: str, rng: np.random.Generator) -> tuple[int, int]:
        row_idxs = self.study_to_rows[group_key]
        if self.prefer_different_views and self.view_labels:
            views = [self._df.loc[r, "view"] for r in row_idxs]
            unique = set(v for v in views if v is not None)
            if len(unique) >= 2:
                # Pair across distinct view labels.
                view_groups: dict[str, list[int]] = defaultdict(list)
                for r, v in zip(row_idxs, views):
                    if v is not None:
                        view_groups[v].append(r)
                keys = list(view_groups.keys())
                rng.shuffle(keys)
                va, vb = keys[0], keys[1]
                a = int(rng.choice(view_groups[va]))
                b = int(rng.choice(view_groups[vb]))
                return a, b
        # Fallback: random pair within the group.
        i, j = rng.choice(len(row_idxs), size=2, replace=False)
        return int(row_idxs[int(i)]), int(row_idxs[int(j)])

    def _pick_pair_rows_curriculum(
        self,
        group_key: str,
        target_bucket: str,
        rng: np.random.Generator,
    ) -> Optional[tuple[int, int, str]]:
        """Try to draw a (row_a, row_b) pair from ``group_key`` whose
        view_distance_bucket matches ``target_bucket``. Returns
        ``(row_a, row_b, bucket)`` or None if no matching pair exists.
        """
        by_bucket = self._bucket_pairs_for_study(group_key)
        pool = by_bucket.get(target_bucket, [])
        if not pool:
            return None
        pick = int(rng.integers(0, len(pool)))
        ra, rb = pool[pick]
        # Random swap of a/b for symmetry; the phase sampler is symmetric.
        if rng.random() < 0.5:
            ra, rb = rb, ra
        return ra, rb, target_bucket

    def _viewpair_pairs_for_study(
        self, group_key: str
    ) -> dict[str, list[tuple[int, int]]]:
        """Return the precomputed {same_view,same_family,cross_family} pair
        lists for a study. Uses ``self._pair_index_viewpair`` from ``_load``."""
        return self._pair_index_viewpair[group_key]

    def _pick_pair_rows_viewpair(
        self,
        group_key: str,
        target_class: str,
        rng: np.random.Generator,
    ) -> Optional[tuple[int, int, str]]:
        """Draw (row_a, row_b) from ``group_key`` satisfying a view-pair
        class. Returns None if study has no eligible pair in that class.
        """
        by_cls = self._viewpair_pairs_for_study(group_key)
        pool = by_cls.get(target_class, [])
        if not pool:
            return None
        pick = int(rng.integers(0, len(pool)))
        ra, rb = pool[pick]
        if rng.random() < 0.5:
            ra, rb = rb, ra
        return ra, rb, target_class

    def _sample_phi_pair(self, rng: np.random.Generator) -> tuple[float, float]:
        """Return (target_phi_a, target_phi_b) depending on sampling mode."""
        if self.sampling_mode in ("uniform_phase", "phase_curriculum"):
            # phase_curriculum uses identical phase matching to uniform_phase;
            # only the row-pair selection differs.
            phi = float(rng.random())
            return phi, phi
        if self.sampling_mode == "wrong_phase":
            phi = float(rng.random())
            return phi, (phi + 0.5) % 1.0
        if self.sampling_mode == "ed_es_biased":
            windows = self.ed_windows if rng.random() < self.ed_probability else self.es_windows
            phi = _sample_from_windows(rng, windows)
            return phi, phi
        if self.sampling_mode == "same_study_random":
            # Both anchors drawn independently; phi values are unused for
            # anchor selection.
            return float(rng.random()), float(rng.random())
        raise ValueError(f"unknown sampling_mode {self.sampling_mode}")

    def _draw_anchor(
        self,
        phase: np.ndarray,
        confident: np.ndarray,
        target_phi: float,
        rng: np.random.Generator,
    ) -> Optional[tuple[int, float, float]]:
        if self.sampling_mode == "same_study_random":
            idxs = np.where(confident)[0]
            if not len(idxs):
                return None
            pick = int(rng.choice(idxs))
            return pick, float(phase[pick]) if np.isfinite(phase[pick]) else float("nan"), 0.0
        return nearest_confident_frame(phase, confident, target_phi, tolerance=self.phase_tolerance)

    def _draw_pair(
        self,
        group_key: str,
        rng: np.random.Generator,
        target_bucket: Optional[str] = None,
        target_viewpair_class: Optional[str] = None,
    ) -> Optional[MatchRecord]:
        for _ in range(self.resample_attempts):
            if target_bucket is not None:
                picked = self._pick_pair_rows_curriculum(group_key, target_bucket, rng)
                if picked is None:
                    return None  # caller will resample bucket/study
                row_a_idx, row_b_idx, _ = picked
            elif target_viewpair_class is not None:
                picked = self._pick_pair_rows_viewpair(group_key, target_viewpair_class, rng)
                if picked is None:
                    return None
                row_a_idx, row_b_idx, _ = picked
            else:
                row_a_idx, row_b_idx = self._pick_pair_rows(group_key, rng)
            row_a = self._df.loc[row_a_idx]
            row_b = self._df.loc[row_b_idx]
            ph_a, c_a = _decode_phase(row_a)
            ph_b, c_b = _decode_phase(row_b)
            phi_a, phi_b = self._sample_phi_pair(rng)
            hit_a = self._draw_anchor(ph_a, c_a, phi_a, rng)
            hit_b = self._draw_anchor(ph_b, c_b, phi_b, rng)
            if hit_a is None or hit_b is None:
                continue
            fa, pa, err_a = hit_a
            fb, pb, err_b = hit_b
            study_id_val = str(row_a.study_id)
            subj = row_a.subject_id if "subject_id" in row_a.index else None
            fps_a = float(row_a.fps_video) if row_a.fps_video and row_a.fps_video > 0 else float("nan")
            fps_b = float(row_b.fps_video) if row_b.fps_video and row_b.fps_video > 0 else float("nan")
            hr_a = float(row_a.hr_metadata) if row_a.hr_metadata and row_a.hr_metadata > 0 else float("nan")
            hr_b = float(row_b.hr_metadata) if row_b.hr_metadata and row_b.hr_metadata > 0 else float("nan")
            span_s_a = self.source_span_frames / fps_a if fps_a > 0 else float("nan")
            span_s_b = self.source_span_frames / fps_b if fps_b > 0 else float("nan")
            span_c_a = span_s_a * hr_a / 60.0 if fps_a > 0 and hr_a > 0 else float("nan")
            span_c_b = span_s_b * hr_b / 60.0 if fps_b > 0 and hr_b > 0 else float("nan")
            clip_a = ClipAnchor(
                row_idx=int(row_a_idx),
                dicom_id=str(row_a.dicom_id),
                n_frames=int(row_a.n_video_frames),
                anchor_frame=int(fa),
                phase_at_anchor=float(pa) if np.isfinite(pa) else float("nan"),
                phase_error=float(err_a),
                view=(row_a["view"] if isinstance(row_a["view"], str) else None),
                hr_metadata=hr_a if np.isfinite(hr_a) else None,
                fps_video=fps_a if np.isfinite(fps_a) else None,
                quality_tier=str(row_a.quality_tier),
                rr_consistent=(None if not self.require_rr_consistent else True),
            )
            clip_b = ClipAnchor(
                row_idx=int(row_b_idx),
                dicom_id=str(row_b.dicom_id),
                n_frames=int(row_b.n_video_frames),
                anchor_frame=int(fb),
                phase_at_anchor=float(pb) if np.isfinite(pb) else float("nan"),
                phase_error=float(err_b),
                view=(row_b["view"] if isinstance(row_b["view"], str) else None),
                hr_metadata=hr_b if np.isfinite(hr_b) else None,
                fps_video=fps_b if np.isfinite(fps_b) else None,
                quality_tier=str(row_b.quality_tier),
                rr_consistent=(None if not self.require_rr_consistent else True),
            )
            circ_diff = circular_phase_distance(pa if np.isfinite(pa) else 0.0,
                                                 pb if np.isfinite(pb) else 0.0)
            # Curriculum diagnostics (only populated in phase_curriculum mode)
            if self.sampling_mode == "phase_curriculum":
                bucket = view_distance_bucket(clip_a.view, clip_b.view)
                bucket_num = {"easy": 0, "medium": 1, "hard": 2}.get(bucket, -1)
                fam_a = VIEW_FAMILIES.get((clip_a.view or "UNKNOWN"), "other")
                fam_b = VIEW_FAMILIES.get((clip_b.view or "UNKNOWN"), "other")
                probs = self._last_curriculum_probs or {}
                probs_str = ",".join(f"{k}={probs.get(k, 0.0):.2f}" for k in ("easy", "medium", "hard"))
                cur_frac = self._last_curriculum_epoch_frac
            else:
                bucket = None
                bucket_num = None
                fam_a = None
                fam_b = None
                probs_str = None
                cur_frac = None
            return MatchRecord(
                study_id=study_id_val,
                subject_id=(None if subj is None or (isinstance(subj, float) and math.isnan(subj)) else str(subj)),
                acquisition_datetime_a=(None if pd.isna(row_a.acquisition_datetime) else str(row_a.acquisition_datetime)),
                acquisition_datetime_b=(None if pd.isna(row_b.acquisition_datetime) else str(row_b.acquisition_datetime)),
                clip_a=clip_a,
                clip_b=clip_b,
                target_phi_a=float(phi_a),
                target_phi_b=float(phi_b),
                circular_phase_diff=float(circ_diff),
                sampling_mode=self.sampling_mode,
                frame_step=int(self.frame_step),
                frames_per_clip=int(self.frames_per_clip),
                source_span_frames=int(self.source_span_frames),
                source_span_seconds_a=float(span_s_a),
                source_span_seconds_b=float(span_s_b),
                source_span_cycles_a=float(span_c_a),
                source_span_cycles_b=float(span_c_b),
                view_distance_bucket=bucket,
                view_distance_numeric=bucket_num,
                view_family_a=fam_a,
                view_family_b=fam_b,
                curriculum_epoch_frac=cur_frac,
                curriculum_bucket_probs=probs_str,
            )
        return None

    # ---------- public API: build records, anchor table, iteration -------- #

    def build_records(self) -> list[MatchRecord]:
        rng = np.random.default_rng(self.seed + self.epoch)
        records: list[MatchRecord] = []
        order = list(self.study_keys)
        perm = rng.permutation(len(order))
        order = [order[i] for i in perm]
        skipped_tol = 0
        # Curriculum: compute epoch_frac and active bucket probs once.
        if self.sampling_mode == "phase_curriculum":
            denom = max(1, self.total_epochs - 1)
            epoch_frac = float(min(self.epoch, denom)) / denom
            probs = self._current_bucket_probs(epoch_frac)
            self._last_curriculum_epoch_frac = epoch_frac
            self._last_curriculum_probs = probs
            self._last_curriculum_skipped = {"easy": 0, "medium": 0, "hard": 0}
            buckets = ("easy", "medium", "hard")
            p = np.array([probs[b] for b in buckets], dtype=np.float64)
            p = p / p.sum()
        # View-pair policy: set up target distribution if enabled.
        use_viewpair_policy = (
            self.view_pair_policy is not None
            and self.view_pair_policy.get("enabled", False)
        )
        if use_viewpair_policy:
            classes = ("same_view", "same_family", "cross_family")
            vp = self.view_pair_policy["probs"]
            vp_p = np.array([vp[c] for c in classes], dtype=np.float64)
            vp_p = vp_p / vp_p.sum()
            self._last_view_pair_skipped = {c: 0 for c in classes}
        for key in order:
            for _ in range(self.pairs_per_study):
                if self.sampling_mode == "phase_curriculum":
                    # Sample a bucket; if study has no eligible pair in that
                    # bucket, try each other bucket in probability order once
                    # more before giving up.
                    bucket_order = list(np.random.default_rng(
                        int(rng.integers(0, 2**31 - 1))
                    ).choice(len(buckets), size=len(buckets), replace=False, p=p))
                    r = None
                    for i in bucket_order:
                        tgt = buckets[int(i)]
                        r = self._draw_pair(key, rng, target_bucket=tgt)
                        if r is not None:
                            break
                        # Record the skipped attempt
                        self._last_curriculum_skipped[tgt] = (
                            self._last_curriculum_skipped.get(tgt, 0) + 1
                        )
                elif use_viewpair_policy:
                    # Sample a view-pair class; if study has no eligible
                    # pair in that class, try each other class in probability
                    # order.
                    cls_order = list(np.random.default_rng(
                        int(rng.integers(0, 2**31 - 1))
                    ).choice(len(classes), size=len(classes), replace=False, p=vp_p))
                    r = None
                    for i in cls_order:
                        tgt = classes[int(i)]
                        r = self._draw_pair(key, rng, target_viewpair_class=tgt)
                        if r is not None:
                            break
                        self._last_view_pair_skipped[tgt] = (
                            self._last_view_pair_skipped.get(tgt, 0) + 1
                        )
                else:
                    r = self._draw_pair(key, rng)
                if r is None:
                    skipped_tol += 1
                    continue
                records.append(r)
        if not records:
            raise RuntimeError(
                f"No phase-matched pairs drawn (skipped={skipped_tol}, "
                f"studies={self.n_studies}, mode={self.sampling_mode})"
            )
        # Pad to total_size and rank-slice.
        if len(records) < self.total_size:
            pad = self.total_size - len(records)
            records += records[:pad]
        else:
            records = records[: self.total_size]
        self._last_skipped_tol = skipped_tol
        return records[self.rank : self.total_size : self.num_replicas]

    def __iter__(self) -> Iterator[int]:
        """Yield pair-DataFrame positions, NOT source-parquet row indices.

        When used with ``PhaseMatchedEpochBuilder``, the dataset's
        ``self.df`` is swapped to a pair DataFrame whose index is
        ``[0..len(pair_df))``. The anchor table is keyed on those same
        positions. Yielding ``i`` aligns all three: DataLoader index ->
        pair-DF row -> per-clip anchor lookup.
        """
        self._last_records = self.build_records()
        return iter(range(len(self._last_records)))

    def __len__(self) -> int:
        return self.num_samples

    @property
    def last_records(self) -> list[MatchRecord]:
        return getattr(self, "_last_records", [])

    def build_anchor_table(
        self,
        records: Optional[list[MatchRecord]] = None,
        side: str = "b",
    ) -> dict[int, list[dict]]:
        """Legacy helper: ``{source_row_idx: [{anchor_frame, frame_step}]}``.

        Kept for callers that want to anchor-center a *single-clip* dataset
        whose ``self.df`` IS the filtered source parquet. For paired
        training (``PhaseMatchedEpochBuilder`` + ``set_pair_dataframe``)
        use ``phase_matched_pair_dataset._records_to_anchor_table`` which
        is keyed on pair-DataFrame positions.
        """
        if records is None:
            records = self.last_records
        table: dict[int, list[dict]] = {}
        anchor_entry = lambda a: [{"anchor_frame": int(a.anchor_frame), "frame_step": int(self.frame_step)}]
        for r in records:
            if side in ("a", "both"):
                table[int(r.clip_a.row_idx)] = anchor_entry(r.clip_a)
            if side in ("b", "both"):
                table[int(r.clip_b.row_idx)] = anchor_entry(r.clip_b)
        return table

    def build_pair_dataframe(self, records: Optional[list[MatchRecord]] = None) -> pd.DataFrame:
        """Flatten MatchRecords into a wide pair DataFrame.

        Schema is stable for the paired-loader wrapper (Turn 2). Every
        record becomes one row with separate ``clip_a_*`` / ``clip_b_*``
        columns for the URIs, anchor frames, phases, HR, fps, etc.
        """
        if records is None:
            records = self.last_records
        rows = []
        for r in records:
            row = {
                "study_id": r.study_id,
                "subject_id": r.subject_id,
                "acquisition_datetime_a": r.acquisition_datetime_a,
                "acquisition_datetime_b": r.acquisition_datetime_b,
                "target_phi_a": r.target_phi_a,
                "target_phi_b": r.target_phi_b,
                "circular_phase_diff": r.circular_phase_diff,
                "sampling_mode": r.sampling_mode,
                "frame_step": r.frame_step,
                "frames_per_clip": r.frames_per_clip,
                "source_span_frames": r.source_span_frames,
            }
            for side, ca in (("a", r.clip_a), ("b", r.clip_b)):
                row[f"clip_{side}_row_idx"] = ca.row_idx
                row[f"clip_{side}_dicom_id"] = ca.dicom_id
                row[f"clip_{side}_n_frames"] = ca.n_frames
                row[f"clip_{side}_anchor_frame"] = ca.anchor_frame
                row[f"clip_{side}_phase_at_anchor"] = ca.phase_at_anchor
                row[f"clip_{side}_phase_error"] = ca.phase_error
                row[f"clip_{side}_view"] = ca.view
                row[f"clip_{side}_hr_metadata"] = ca.hr_metadata
                row[f"clip_{side}_fps_video"] = ca.fps_video
                row[f"clip_{side}_quality_tier"] = ca.quality_tier
                # URI pulled from the underlying parquet for the loader.
                src = self._df.loc[ca.row_idx]
                row[f"clip_{side}_s3_uri"] = src.s3_uri
                if "dicom_filepath" in src.index:
                    row[f"clip_{side}_dicom_filepath"] = src.dicom_filepath
            row["source_span_seconds_a"] = r.source_span_seconds_a
            row["source_span_seconds_b"] = r.source_span_seconds_b
            row["source_span_cycles_a"] = r.source_span_cycles_a
            row["source_span_cycles_b"] = r.source_span_cycles_b
            rows.append(row)
        return pd.DataFrame(rows)

    # ---------- diagnostics ------------------------------------------------ #

    def summary(self, records: Optional[list[MatchRecord]] = None) -> dict:
        recs = records if records is not None else self.last_records
        if not recs:
            return {"n_records": 0, **self._filter_stats}
        err_a = np.array([r.clip_a.phase_error for r in recs])
        err_b = np.array([r.clip_b.phase_error for r in recs])
        circ = np.array([r.circular_phase_diff for r in recs])
        cyc_a = np.array([r.source_span_cycles_a for r in recs])
        cyc_b = np.array([r.source_span_cycles_b for r in recs])
        hr_a = np.array([r.clip_a.hr_metadata for r in recs if r.clip_a.hr_metadata is not None], dtype=float)
        hr_b = np.array([r.clip_b.hr_metadata for r in recs if r.clip_b.hr_metadata is not None], dtype=float)
        hr_diff = np.abs(hr_a - hr_b) if len(hr_a) == len(hr_b) and len(hr_a) > 0 else np.array([])
        views_a = [r.clip_a.view for r in recs]
        views_b = [r.clip_b.view for r in recs]
        same_view = sum(1 for a, b in zip(views_a, views_b) if a is not None and a == b)
        diff_view = sum(1 for a, b in zip(views_a, views_b) if a is not None and b is not None and a != b)
        d = {
            "n_records": len(recs),
            "sampling_mode": self.sampling_mode,
            "phase_tolerance": self.phase_tolerance,
            "frame_step": self.frame_step,
            "frames_per_clip": self.frames_per_clip,
            "source_span_frames": int(self.source_span_frames),
            "phase_error_a": {
                "median": float(np.median(err_a)),
                "p90": float(np.quantile(err_a, 0.9)),
                "max": float(err_a.max()),
            },
            "phase_error_b": {
                "median": float(np.median(err_b)),
                "p90": float(np.quantile(err_b, 0.9)),
                "max": float(err_b.max()),
            },
            "circular_phase_diff": {
                "median": float(np.median(circ)),
                "p90": float(np.quantile(circ, 0.9)),
                "max": float(circ.max()),
            },
            "source_span_cycles_a": {
                "median": float(np.nanmedian(cyc_a)),
                "p10": float(np.nanquantile(cyc_a, 0.10)),
                "p90": float(np.nanquantile(cyc_a, 0.90)),
            },
            "source_span_cycles_b": {
                "median": float(np.nanmedian(cyc_b)),
                "p10": float(np.nanquantile(cyc_b, 0.10)),
                "p90": float(np.nanquantile(cyc_b, 0.90)),
            },
            "hr_difference_bpm": {
                "median": (float(np.median(hr_diff)) if hr_diff.size else None),
                "p90": (float(np.quantile(hr_diff, 0.9)) if hr_diff.size else None),
                "max": (float(hr_diff.max()) if hr_diff.size else None),
            },
            "same_view_pairs": same_view,
            "diff_view_pairs": diff_view,
            "view_labels_present": bool(self.view_labels),
            "skipped_by_tolerance": int(getattr(self, "_last_skipped_tol", 0)),
            **self._filter_stats,
        }
        if self.sampling_mode in ("uniform_phase", "ed_es_biased"):
            p90 = d["source_span_cycles_a"]["p90"]
            if np.isfinite(p90) and p90 > 1.5:
                logger.warning(
                    "source_span_cycles p90 = %.2f > 1.5 for phase-matched mode — "
                    "same-shape tensors span different fractions of the cardiac cycle. "
                    "Consider frame_step=1 (source_span_frames=%d) if currently higher.",
                    p90, self.source_span_frames,
                )
        return d


# --------------------------------------------------------------------------- #
# CLI dry-run
# --------------------------------------------------------------------------- #

def _pretty(d: dict, indent: int = 2) -> None:
    def _json_default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        return str(o)
    print(json.dumps(d, indent=indent, default=_json_default))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", type=Path, required=True)
    ap.add_argument("--quality-tiers", nargs="+", default=["high"])
    ap.add_argument("--require-rr-consistent", action="store_true", default=True)
    ap.add_argument("--no-rr", dest="require_rr_consistent", action="store_false")
    ap.add_argument("--rr-filter-mode", choices=RR_FILTER_MODES, default="strict")
    ap.add_argument("--sampling-mode", default="uniform_phase", choices=SAMPLING_MODES)
    ap.add_argument("--phase-tolerance", type=float, default=0.15)
    ap.add_argument("--frames-per-clip", type=int, default=16)
    ap.add_argument("--frame-step", type=int, default=1)
    ap.add_argument("--pairs-per-study", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epoch", type=int, default=0)
    ap.add_argument("--view-predictions", type=Path, default=None,
                    help="CSV with dicom_id,view,view_confidence (optional).")
    ap.add_argument("--same-session-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true", default=True,
                    help="Default on; kept for call-site compatibility.")
    ap.add_argument("--require-span-fits", action="store_true")
    ap.add_argument("--min-frames", type=int, default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    view_labels = None
    if args.view_predictions is not None:
        vdf = pd.read_csv(args.view_predictions)
        # extract dicom_id from s3 path if not already a column
        if "dicom_id" not in vdf.columns and "s3_uri" in vdf.columns:
            vdf["dicom_id"] = vdf.s3_uri.str.extract(r"/([^/]+)\.mp4$")
        view_labels = dict(zip(vdf.dicom_id.astype(str), vdf.view.astype(str)))

    s = PhaseMatchedStudySampler(
        parquet_path=args.parquet,
        tiers=tuple(args.quality_tiers),
        require_rr_consistent=args.require_rr_consistent,
        rr_filter_mode=args.rr_filter_mode,
        sampling_mode=args.sampling_mode,
        phase_tolerance=args.phase_tolerance,
        frames_per_clip=args.frames_per_clip,
        frame_step=args.frame_step,
        pairs_per_study=args.pairs_per_study,
        same_session_only=args.same_session_only,
        seed=args.seed,
        view_labels=view_labels,
        require_span_fits=args.require_span_fits,
        min_frames=args.min_frames,
    )
    s.set_epoch(args.epoch)
    records = s.build_records()
    summary = s.summary(records)

    print("=" * 72)
    print(f"PhaseMatchedStudySampler DRY RUN  (seed={args.seed}, epoch={args.epoch}, rank=0)")
    print("=" * 72)
    _pretty(summary)

    # Show a few sample records for eyeballing.
    print("\n-- first 3 records --")
    for r in records[:3]:
        _pretty({
            "study": r.study_id,
            "phi_a": r.target_phi_a,
            "phi_b": r.target_phi_b,
            "circ_diff": r.circular_phase_diff,
            "a": asdict(r.clip_a),
            "b": asdict(r.clip_b),
            "source_span_cycles_a": r.source_span_cycles_a,
            "source_span_cycles_b": r.source_span_cycles_b,
        })


if __name__ == "__main__":
    main()
