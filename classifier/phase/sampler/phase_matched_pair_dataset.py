"""Paired-loader scaffolding for phase-matched multi-view JEPA training.

Wraps ``VideoGroupDataset(group_size=2)`` with an in-memory pair
DataFrame built from ``PhaseMatchedStudySampler`` records. Each dataset
``__getitem__(idx)`` returns the standard V-JEPA tuple
``(segs, label, clip_indices_out, slot_mask)`` from the underlying
loader PLUS the per-pair phase metadata for the training-loop consumer.

The design hinges on three points:

1. **Atomic epoch refresh.** ``refresh_epoch(epoch)`` rebuilds the
   sampler records, flattens them to a wide pair DataFrame, builds the
   anchor table keyed on pair-row index, and calls
   ``dataset.set_pair_dataframe(...)`` which swaps both fields in one
   call. Consumers MUST call this before ``iter(loader)`` each epoch.

2. **In-memory DataFrame.** No per-epoch disk I/O. A debug option
   (``debug_csv_path=Path(...)``) dumps the pair DataFrame once after
   refresh for inspection, but is not in the training hot path.

3. **View wiring.** ``view_0`` always = clip_a URI, ``view_1`` = clip_b.
   The anchor table's per-row list has length 2 matching this order.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from phase_matched_sampler import MatchRecord, PhaseMatchedStudySampler  # noqa: E402


def _rewrite_s3_uri_dicom_to_mp4(
    uri: str,
    raw_bucket_prefix: str,
    mp4_bucket_prefix: str,
) -> str:
    """Rewrite a raw DICOM S3 URI to the corresponding MP4 URI.

    Example:
      s3://echodata25/mimic-raw-staging/files/p10/.../94106955_0001.dcm
      ->
      s3://echodata25/mimic-echo-224px/files/p10/.../94106955_0001.mp4

    Both bucket-prefix strings should match the leading portion of the URI
    exactly (scheme + bucket + top-level key prefix). Validated via a
    probe: 400/400 MP4s have identical frame counts to their DICOM
    sources (see /claude/dev/ for the probe artifact).
    """
    if not isinstance(uri, str):
        return uri
    if not uri.startswith(raw_bucket_prefix):
        return uri
    rewritten = uri.replace(raw_bucket_prefix, mp4_bucket_prefix, 1)
    if rewritten.endswith(".dcm"):
        rewritten = rewritten[: -len(".dcm")] + ".mp4"
    return rewritten


def _records_to_pair_dataframe(
    records: list[MatchRecord],
    underlying_df: pd.DataFrame,
    video_uri_mode: str = "mp4",
    raw_bucket_prefix: str = "s3://echodata25/mimic-raw-staging",
    mp4_bucket_prefix: str = "s3://echodata25/mimic-echo-224px",
) -> pd.DataFrame:
    """Flatten MatchRecords into the wide pair DataFrame schema the pair
    dataset consumes. ``underlying_df`` is the sampler's filtered source
    table — used to pull the clip URIs back out by row_idx.

    Args:
      video_uri_mode: "mp4" (default, production; decord-compatible) or
        "dicom" (test/debug; requires a pydicom shim for decord).
      raw_bucket_prefix / mp4_bucket_prefix: source-parquet URI prefix
        and the MP4-bucket prefix to rewrite to, when video_uri_mode="mp4".
    """
    if video_uri_mode not in ("mp4", "dicom"):
        raise ValueError(
            f"video_uri_mode must be 'mp4' or 'dicom'; got {video_uri_mode!r}"
        )

    def _wire(uri: str) -> str:
        if video_uri_mode == "dicom":
            return uri
        return _rewrite_s3_uri_dicom_to_mp4(uri, raw_bucket_prefix, mp4_bucket_prefix)

    rows = []
    for r in records:
        src_a = underlying_df.loc[r.clip_a.row_idx]
        src_b = underlying_df.loc[r.clip_b.row_idx]
        v0 = _wire(src_a.s3_uri)
        v1 = _wire(src_b.s3_uri)
        if video_uri_mode == "mp4":
            # Fail loudly if the rewrite didn't land on an .mp4 URI.
            for tag, v in (("view_0", v0), ("view_1", v1)):
                if not (isinstance(v, str) and v.endswith(".mp4")):
                    raise ValueError(
                        f"video_uri_mode='mp4' but {tag} did not end in .mp4: {v!r} "
                        f"(source uri={src_a.s3_uri if tag=='view_0' else src_b.s3_uri!r}, "
                        f"raw_prefix={raw_bucket_prefix!r})"
                    )
        row = {
            # VideoGroupDataset-mandatory columns:
            "view_0": v0,
            "view_1": v1,
            "label": 0.0,                  # phase-matched pretraining has no label
            # Pair-level metadata:
            "study_id": r.study_id,
            "subject_id": r.subject_id,
            "sampling_mode": r.sampling_mode,
            "target_phi_a": r.target_phi_a,
            "target_phi_b": r.target_phi_b,
            "circular_phase_diff": r.circular_phase_diff,
            "frame_step": r.frame_step,
            "frames_per_clip": r.frames_per_clip,
            "source_span_frames": r.source_span_frames,
            "source_span_seconds_a": r.source_span_seconds_a,
            "source_span_seconds_b": r.source_span_seconds_b,
            "source_span_cycles_a": r.source_span_cycles_a,
            "source_span_cycles_b": r.source_span_cycles_b,
            # Per-clip metadata (dicom_id, anchor_frame, phase, view, HR):
            "clip_a_dicom_id": r.clip_a.dicom_id,
            "clip_b_dicom_id": r.clip_b.dicom_id,
            "clip_a_row_idx": r.clip_a.row_idx,
            "clip_b_row_idx": r.clip_b.row_idx,
            "clip_a_n_frames": r.clip_a.n_frames,
            "clip_b_n_frames": r.clip_b.n_frames,
            "clip_a_anchor_frame": r.clip_a.anchor_frame,
            "clip_b_anchor_frame": r.clip_b.anchor_frame,
            "clip_a_phase_at_anchor": r.clip_a.phase_at_anchor,
            "clip_b_phase_at_anchor": r.clip_b.phase_at_anchor,
            "clip_a_phase_error": r.clip_a.phase_error,
            "clip_b_phase_error": r.clip_b.phase_error,
            "clip_a_view": r.clip_a.view,
            "clip_b_view": r.clip_b.view,
            "clip_a_hr_metadata": r.clip_a.hr_metadata,
            "clip_b_hr_metadata": r.clip_b.hr_metadata,
            "clip_a_fps_video": r.clip_a.fps_video,
            "clip_b_fps_video": r.clip_b.fps_video,
            "clip_a_quality_tier": r.clip_a.quality_tier,
            "clip_b_quality_tier": r.clip_b.quality_tier,
            "acquisition_datetime_a": r.acquisition_datetime_a,
            "acquisition_datetime_b": r.acquisition_datetime_b,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _records_to_anchor_table(
    records: list[MatchRecord],
) -> dict[int, list]:
    """Return ``{pair_row_idx: [{anchor_frame, frame_step}, {...}]}``.

    The two per-row entries correspond to view_0 (clip_a) and view_1
    (clip_b) respectively, matching ``set_pair_dataframe`` column order.
    """
    table: dict[int, list] = {}
    for i, r in enumerate(records):
        table[int(i)] = [
            {"anchor_frame": int(r.clip_a.anchor_frame), "frame_step": int(r.frame_step)},
            {"anchor_frame": int(r.clip_b.anchor_frame), "frame_step": int(r.frame_step)},
        ]
    return table


class PhaseMatchedEpochBuilder:
    """Orchestrates per-epoch refresh of a ``VideoGroupDataset(group_size=2)``
    with phase-matched pair records.

    Typical usage inside the training loop::

        sampler = PhaseMatchedStudySampler(...)
        dataset = VideoGroupDataset(data_paths=..., group_size=2, ...)
        builder = PhaseMatchedEpochBuilder(sampler, dataset)

        for epoch in range(n_epochs):
            builder.refresh_epoch(epoch)  # atomic: sampler->pair_df->anchors
            for batch in loader:
                ...

    In DDP, each rank constructs its own sampler (rank/world_size handled
    inside the sampler's constructor) and its own builder; the
    ``refresh_epoch`` call on each rank loads only that rank's records
    into its own dataset instance. Verify disjointness with the DDP
    smoke-test script in ``check_ddp_disjoint.py``.
    """

    def __init__(
        self,
        sampler: PhaseMatchedStudySampler,
        dataset,
        debug_csv_path: Optional[Path] = None,
        video_uri_mode: str = "mp4",
        raw_bucket_prefix: str = "s3://echodata25/mimic-raw-staging",
        mp4_bucket_prefix: str = "s3://echodata25/mimic-echo-224px",
    ) -> None:
        self.sampler = sampler
        self.dataset = dataset
        self.debug_csv_path = debug_csv_path
        self.video_uri_mode = video_uri_mode
        self.raw_bucket_prefix = raw_bucket_prefix
        self.mp4_bucket_prefix = mp4_bucket_prefix
        self._last_pair_df: Optional[pd.DataFrame] = None
        self._last_anchors: Optional[dict] = None

    def refresh_epoch(self, epoch: int) -> int:
        """Build records for ``epoch`` and swap them onto the dataset
        atomically. Returns the number of pair rows for this rank."""
        self.sampler.set_epoch(epoch)
        records = self.sampler.build_records()
        pair_df = _records_to_pair_dataframe(
            records,
            self.sampler._df,
            video_uri_mode=self.video_uri_mode,
            raw_bucket_prefix=self.raw_bucket_prefix,
            mp4_bucket_prefix=self.mp4_bucket_prefix,
        )
        anchors = _records_to_anchor_table(records)
        # Atomic swap:
        self.dataset.set_pair_dataframe(pair_df, anchors_by_index=anchors)
        self._last_pair_df = pair_df
        self._last_anchors = anchors
        if self.debug_csv_path is not None:
            self.debug_csv_path.parent.mkdir(parents=True, exist_ok=True)
            pair_df.to_csv(self.debug_csv_path, index=False)
        return len(pair_df)

    @property
    def last_pair_df(self) -> Optional[pd.DataFrame]:
        return self._last_pair_df

    @property
    def last_anchors(self) -> Optional[dict]:
        return self._last_anchors


__all__ = [
    "PhaseMatchedEpochBuilder",
    "_records_to_pair_dataframe",
    "_records_to_anchor_table",
    "_rewrite_s3_uri_dicom_to_mp4",
]
