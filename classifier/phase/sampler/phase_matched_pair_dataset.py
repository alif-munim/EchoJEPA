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

# Token consumed by ``VideoGroupDataset`` to mark missing views; matches
# the constant defined in ``src/datasets/video_group_dataset.py``.
MISSING_TOKEN = "MISS"


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
        raise ValueError(f"video_uri_mode must be 'mp4' or 'dicom'; got {video_uri_mode!r}")

    def _wire(uri: str) -> str:
        if video_uri_mode == "dicom":
            return uri
        return _rewrite_s3_uri_dicom_to_mp4(uri, raw_bucket_prefix, mp4_bucket_prefix)

    # Detect whether any record carries a hard negative; if so, the emitted
    # DataFrame gains `view_2` + clip_b_neg_* metadata columns (same length
    # for every row; None/NaN when a particular record happens to lack one).
    any_hard_neg = any(r.clip_b_neg_phase is not None for r in records)
    # MV2SV target_clip / fused_clips schema detection.
    any_target_clip = any(r.target_clip is not None for r in records)
    actual_max_n_fused = 0
    for r in records:
        if r.fused_clips:
            actual_max_n_fused = max(actual_max_n_fused, max(0, len(r.fused_clips) - 1))

    rows = []
    for r in records:
        src_a = underlying_df.loc[r.clip_a.row_idx]
        src_b = underlying_df.loc[r.clip_b.row_idx]
        v0 = _wire(src_a.s3_uri)
        v1 = _wire(src_b.s3_uri)
        v2 = None
        src_neg = None
        if r.clip_b_neg_phase is not None:
            src_neg = underlying_df.loc[r.clip_b_neg_phase.row_idx]
            v2 = _wire(src_neg.s3_uri)
        if video_uri_mode == "mp4":
            # Fail loudly if the rewrite didn't land on an .mp4 URI.
            pairs_to_check = [("view_0", v0, src_a), ("view_1", v1, src_b)]
            if v2 is not None:
                pairs_to_check.append(("view_2", v2, src_neg))
            for tag, v, src in pairs_to_check:
                if not (isinstance(v, str) and v.endswith(".mp4")):
                    raise ValueError(
                        f"video_uri_mode='mp4' but {tag} did not end in .mp4: {v!r} "
                        f"(source uri={src.s3_uri!r}, "
                        f"raw_prefix={raw_bucket_prefix!r})"
                    )
        row = {
            # VideoGroupDataset-mandatory columns:
            "view_0": v0,
            "view_1": v1,
            "label": 0.0,  # phase-matched pretraining has no label
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
        # Optional: hard-negative triple-clip columns. Emitted on every row
        # (as NaN/empty when the specific record lacks a hard negative) when
        # ANY record in this batch carries one, so the pair DataFrame has
        # homogeneous schema for default_collate.
        if any_hard_neg:
            row["view_2"] = v2 if v2 is not None else MISSING_TOKEN
            if r.clip_b_neg_phase is not None:
                neg = r.clip_b_neg_phase
                row["clip_b_neg_dicom_id"] = neg.dicom_id
                row["clip_b_neg_row_idx"] = neg.row_idx
                row["clip_b_neg_n_frames"] = neg.n_frames
                row["clip_b_neg_anchor_frame"] = neg.anchor_frame
                row["clip_b_neg_phase_at_anchor"] = neg.phase_at_anchor
                row["clip_b_neg_phase_error"] = neg.phase_error
                row["clip_b_neg_view"] = neg.view if neg.view is not None else ""
                row["clip_b_neg_hr_metadata"] = neg.hr_metadata
                row["clip_b_neg_fps_video"] = neg.fps_video
                row["clip_b_neg_quality_tier"] = neg.quality_tier if neg.quality_tier is not None else ""
            else:
                # Placeholder NaN/empty for this row; the record lacks a hard neg
                # but the DataFrame schema must match.
                row["clip_b_neg_dicom_id"] = ""
                row["clip_b_neg_row_idx"] = -1
                row["clip_b_neg_n_frames"] = 0
                row["clip_b_neg_anchor_frame"] = 0
                row["clip_b_neg_phase_at_anchor"] = float("nan")
                row["clip_b_neg_phase_error"] = float("nan")
                row["clip_b_neg_view"] = ""
                row["clip_b_neg_hr_metadata"] = float("nan")
                row["clip_b_neg_fps_video"] = float("nan")
                row["clip_b_neg_quality_tier"] = ""
            row["target_phi_b_neg"] = float(r.target_phi_b_neg) if r.target_phi_b_neg is not None else float("nan")
            row["delta_phase_bucket_pos"] = (
                int(r.delta_phase_bucket_pos) if r.delta_phase_bucket_pos is not None else -1
            )
            row["delta_phase_bucket_neg"] = (
                int(r.delta_phase_bucket_neg) if r.delta_phase_bucket_neg is not None else -1
            )
            row["view_pair_class_pos"] = r.view_pair_class_pos if r.view_pair_class_pos is not None else ""
            row["view_pair_class_neg"] = r.view_pair_class_neg if r.view_pair_class_neg is not None else ""
            row["hard_neg_available"] = int(bool(r.hard_neg_available))
            row["hard_neg_resample_count"] = int(r.hard_neg_resample_count)

        # ---- MV2SV target_clip view + metadata ----
        if any_target_clip:
            if r.target_clip is not None:
                src_tgt = underlying_df.loc[r.target_clip.row_idx]
                v_tgt = _wire(src_tgt.s3_uri)
                if video_uri_mode == "mp4":
                    if not (isinstance(v_tgt, str) and v_tgt.endswith(".mp4")):
                        raise ValueError(
                            f"video_uri_mode='mp4' but view_3 (target_clip) did not end in .mp4: "
                            f"{v_tgt!r} (source uri={src_tgt.s3_uri!r})"
                        )
                row["view_3"] = v_tgt
                row["target_clip_dicom_id"] = r.target_clip.dicom_id
                row["target_clip_row_idx"] = int(r.target_clip.row_idx)
                row["target_clip_n_frames"] = int(r.target_clip.n_frames)
                row["target_clip_anchor_frame"] = int(r.target_clip.anchor_frame)
                row["target_clip_phase_at_anchor"] = float(r.target_clip.phase_at_anchor)
                row["target_clip_phase_error"] = float(r.target_clip.phase_error)
                row["target_clip_view"] = r.target_view if r.target_view is not None else ""
                row["target_delta_phase"] = (
                    float(r.target_delta_phase) if r.target_delta_phase is not None else float("nan")
                )
                row["target_clip_present"] = 1
            else:
                row["view_3"] = MISSING_TOKEN
                row["target_clip_dicom_id"] = ""
                row["target_clip_row_idx"] = -1
                row["target_clip_n_frames"] = 0
                row["target_clip_anchor_frame"] = 0
                row["target_clip_phase_at_anchor"] = float("nan")
                row["target_clip_phase_error"] = float("nan")
                row["target_clip_view"] = ""
                row["target_delta_phase"] = float("nan")
                row["target_clip_present"] = 0

        # ---- MV2SV fused_clips views + metadata ----
        # fused_clips[0] IS the target_clip (already loaded as view_3).
        # Additional fused views are view_4..view_{3+actual_max_n_fused-1}.
        if actual_max_n_fused > 0:
            fused_extra = list(r.fused_clips[1 : 1 + actual_max_n_fused]) if r.fused_clips else []
            fused_extra_views = list(r.fused_views[1 : 1 + actual_max_n_fused]) if r.fused_views else []
            fused_extra_phases = list(r.fused_phases[1 : 1 + actual_max_n_fused]) if r.fused_phases else []
            for k in range(actual_max_n_fused):
                col_idx = 4 + k  # view_4, view_5, ...
                if k < len(fused_extra):
                    fc = fused_extra[k]
                    src_fc = underlying_df.loc[fc.row_idx]
                    v_fc = _wire(src_fc.s3_uri)
                    if video_uri_mode == "mp4":
                        if not (isinstance(v_fc, str) and v_fc.endswith(".mp4")):
                            raise ValueError(
                                f"video_uri_mode='mp4' but view_{col_idx} "
                                f"(fused_clips[{k+1}]) did not end in .mp4: {v_fc!r}"
                            )
                    row[f"view_{col_idx}"] = v_fc
                    row[f"fused_clip_{k+1}_view"] = fused_extra_views[k] if k < len(fused_extra_views) else ""
                    row[f"fused_clip_{k+1}_delta_phase"] = (
                        float(fused_extra_phases[k]) if k < len(fused_extra_phases) else float("nan")
                    )
                    row[f"fused_clip_{k+1}_valid"] = 1
                else:
                    row[f"view_{col_idx}"] = MISSING_TOKEN
                    row[f"fused_clip_{k+1}_view"] = ""
                    row[f"fused_clip_{k+1}_delta_phase"] = float("nan")
                    row[f"fused_clip_{k+1}_valid"] = 0

        rows.append(row)
    return pd.DataFrame(rows)


def _records_to_anchor_table(
    records: list[MatchRecord],
    max_n_fused: int = 0,
) -> dict[int, list]:
    """Return ``{pair_row_idx: [{anchor_frame, frame_step}, ...]}``.

    Per-row entries match the column order emitted by
    ``_records_to_pair_dataframe``:

        view_0: clip_a
        view_1: clip_b_pos
        view_2: clip_b_neg_phase (if any record has a hard-neg)
        view_3: target_clip    (MV2SV — if any record has a target_clip)
        view_4..view_(3+max_n_fused-1): fused_clips[1..]
            (MV2SV — fused_clips[0] is the target_clip; we start at 1 to
            avoid double-loading the target. If fewer than max_n_fused
            fused clips exist for a record, trailing slots are placeholder.)
    """
    table: dict[int, list] = {}
    any_hard_neg = any(r.clip_b_neg_phase is not None for r in records)
    any_target_clip = any(r.target_clip is not None for r in records)
    # max fused pool actually produced across records (not including the
    # target itself, which is position 0 of MatchRecord.fused_clips).
    actual_max_n_fused = 0
    if max_n_fused > 0:
        for r in records:
            if r.fused_clips:
                actual_max_n_fused = max(actual_max_n_fused, max(0, len(r.fused_clips) - 1))
        actual_max_n_fused = min(actual_max_n_fused, max_n_fused)

    for i, r in enumerate(records):
        entries = [
            {"anchor_frame": int(r.clip_a.anchor_frame), "frame_step": int(r.frame_step)},
            {"anchor_frame": int(r.clip_b.anchor_frame), "frame_step": int(r.frame_step)},
        ]
        if any_hard_neg:
            if r.clip_b_neg_phase is not None:
                entries.append(
                    {
                        "anchor_frame": int(r.clip_b_neg_phase.anchor_frame),
                        "frame_step": int(r.frame_step),
                    }
                )
            else:
                entries.append({"anchor_frame": 0, "frame_step": int(r.frame_step)})
        if any_target_clip:
            if r.target_clip is not None:
                entries.append(
                    {
                        "anchor_frame": int(r.target_clip.anchor_frame),
                        "frame_step": int(r.frame_step),
                    }
                )
            else:
                entries.append({"anchor_frame": 0, "frame_step": int(r.frame_step)})
        # Fused pool (skipping index 0 which IS the target_clip).
        if actual_max_n_fused > 0:
            fused_extra = list(r.fused_clips[1 : 1 + actual_max_n_fused]) if r.fused_clips else []
            for k in range(actual_max_n_fused):
                if k < len(fused_extra):
                    entries.append(
                        {
                            "anchor_frame": int(fused_extra[k].anchor_frame),
                            "frame_step": int(r.frame_step),
                        }
                    )
                else:
                    entries.append({"anchor_frame": 0, "frame_step": int(r.frame_step)})
        table[int(i)] = entries
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
        # MV2SV: pass max fused-pool size so the anchor table reserves
        # enough view slots for the fused clips (fused[0] is the target,
        # so the extra slots are fused[1..]).
        mv2sv_max_n_fused = int(getattr(self.sampler, "mv2sv_fused_n_max", 0) or 0)
        extra_fused_slots = max(0, mv2sv_max_n_fused - 1)
        anchors = _records_to_anchor_table(records, max_n_fused=extra_fused_slots)
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
