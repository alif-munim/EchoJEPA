# src/datasets/video_group_dataset.py

import io
import math
import os
import pathlib
import warnings
from logging import getLogger

import boto3
from botocore.config import Config
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from decord import VideoReader, cpu

from src.datasets.utils.dataloader import MonitoredDataset, NondeterministicDataLoader
from src.datasets.utils.weighted_sampler import DistributedWeightedSampler

logger = getLogger()
MISSING_TOKEN = "MISS"


def _compute_clip_indices(
    num_frames: int,
    fpc: int,
    frame_step: int = 1,
    clip_idx: int = 0,
    num_clips: int = 1,
    anchor_frame: int | None = None,
) -> tuple[np.ndarray, dict]:
    """Compute source-frame indices for one clip.

    Two modes:
      * ``anchor_frame is None``: strided window starting at
        ``clip_idx * source_span`` (matches legacy behavior when
        ``frame_step == 1`` and ``num_clips == 1``).
      * ``anchor_frame`` set: the clip is centered on ``anchor_frame``.
        For ``num_clips == 1`` the window spans
        ``[start, start + source_span)`` where
        ``start = round(anchor_frame - ((fpc-1) * frame_step) / 2)``,
        clamped to ``[0, num_frames - source_span]``.
        For ``num_clips > 1`` the sampler passes anchor per clip, so
        ``clip_idx`` selects which window; callers that want anchor-centering
        should use ``num_clips == 1`` and issue separate calls per clip.

    Returns ``(indices, meta)`` where ``meta`` contains:
      - ``start_frame``: first raw-frame index before clamping edge-pad
      - ``anchor_frame``: echoed int or None
      - ``anchor_pos``: position in ``indices`` closest to ``anchor_frame``
        (None if anchor_frame is None)
      - ``frame_step``: stride used
      - ``source_span_frames``: (fpc-1)*frame_step + 1
      - ``was_clamped``: True if start clipped against video boundary
      - ``padded``: True if video was shorter than source_span

    If ``num_frames < source_span``, the window is clipped to
    ``[0, num_frames-1]`` and the last valid index is repeated to pad to
    length ``fpc``. ``anchor_pos`` is still computed against ``anchor_frame``
    when provided.
    """
    source_span = (fpc - 1) * frame_step + 1
    padded = False
    was_clamped = False

    if anchor_frame is None:
        start = int(clip_idx) * source_span
        if num_frames >= source_span:
            indices = start + np.arange(fpc, dtype=np.int64) * frame_step
        else:
            # Degenerate short video: emit what we can, pad with last index.
            raw = start + np.arange(fpc, dtype=np.int64) * frame_step
            raw = np.clip(raw, 0, max(0, num_frames - 1))
            indices = raw
            padded = True
    else:
        anchor = int(round(anchor_frame))
        if num_frames >= source_span:
            raw_start = int(round(anchor - ((fpc - 1) * frame_step) / 2.0))
            start = max(0, min(raw_start, num_frames - source_span))
            was_clamped = (start == 0 and raw_start < 0) or (
                start == num_frames - source_span and raw_start > num_frames - source_span
            )
            indices = start + np.arange(fpc, dtype=np.int64) * frame_step
        else:
            # Video shorter than requested source span: ideal window would
            # start at anchor - (fpc-1)*step/2; clamp to [0, num_frames-1]
            # on every sample and pad by repeating the last valid index.
            raw_start = int(round(anchor - ((fpc - 1) * frame_step) / 2.0))
            start = max(0, min(raw_start, max(0, num_frames - 1)))
            raw = start + np.arange(fpc, dtype=np.int64) * frame_step
            raw = np.clip(raw, 0, max(0, num_frames - 1))
            indices = raw
            padded = True
            was_clamped = True

    if indices.shape[0] < fpc:
        pad_val = int(indices[-1]) if indices.shape[0] > 0 else 0
        pad = np.full(fpc - indices.shape[0], pad_val, dtype=np.int64)
        indices = np.concatenate([indices, pad], axis=0)
        padded = True
    if indices.shape[0] > fpc:
        indices = indices[:fpc]

    anchor_pos = None
    if anchor_frame is not None:
        anchor_pos = int(np.argmin(np.abs(indices - int(round(anchor_frame)))))

    meta = {
        "start_frame": int(indices[0]) if indices.shape[0] > 0 else 0,
        "anchor_frame": None if anchor_frame is None else int(round(anchor_frame)),
        "anchor_pos": anchor_pos,
        "frame_step": int(frame_step),
        "source_span_frames": int(source_span),
        "was_clamped": bool(was_clamped),
        "padded": bool(padded),
    }
    return indices, meta


def _worker_init_fn(_):
    try:
        import torch as _torch, cv2, os as _os

        _torch.set_num_threads(1)
        try:
            cv2.setNumThreads(1)
        except Exception:
            pass
        _os.environ["OMP_NUM_THREADS"] = "1"
        _os.environ["MKL_NUM_THREADS"] = "1"
    except Exception:
        pass


def make_videogroupdataset(
    *,
    data_paths,  # str | list[str]  (CSV path(s))
    batch_size,
    group_size,  # maps to num_segments from config
    frames_per_clip,
    frame_step=None,
    duration=None,
    fps=None,
    num_clips_per_video=1,  # NEW in your pipeline: per-video temporal clips
    random_clip_sampling=True,
    allow_clip_overlap=False,
    filter_short_videos=False,
    filter_long_videos=int(10**9),
    transform=None,
    shared_transform=None,
    rank=0,
    world_size=1,
    datasets_weights=None,
    collator=None,
    drop_last=True,
    num_workers=10,
    pin_mem=True,
    persistent_workers=True,
    deterministic=True,
    log_dir=None,
    img_size=336,  # <<< NEW (pass resolution)
    training=False,  # <<< NEW
    miss_augment_prob=0.0,  # <<< NEW
    min_present=1,  # <<< NEW
    split_name="train",
):
    # Check for perturbation via environment variables (for noise robustness evaluation).
    # Mirrors VideoDataset's perturbation hook.
    from src.datasets.video_dataset import _PerturbationFn

    perturbation_fn = None
    ptype = os.environ.get("PERTURBATION_TYPE")
    psev = os.environ.get("PERTURBATION_SEVERITY")
    if ptype and psev:
        transducer_pos = tuple(float(x) for x in os.environ.get("TRANSDUCER_POS", "0.5,0.0").split(","))
        perturbation_fn = _PerturbationFn(ptype, psev, transducer_pos)
        logger.info(f"VideoGroupDataset perturbation enabled: {ptype}/{psev} (transducer_pos={transducer_pos})")

    ds = VideoGroupDataset(
        data_paths=data_paths,
        group_size=group_size,
        frames_per_clip=frames_per_clip,
        frame_step=frame_step,
        duration=duration,
        fps=fps,
        num_clips_per_video=num_clips_per_video,
        random_clip_sampling=random_clip_sampling,
        allow_clip_overlap=allow_clip_overlap,
        filter_short_videos=filter_short_videos,
        filter_long_videos=filter_long_videos,
        shared_transform=shared_transform,
        transform=transform,
        img_size=img_size,  # <<< NEW
        training=training,  # <<< pass through
        miss_augment_prob=miss_augment_prob,
        min_present=min_present,
        split_name=split_name,
        perturbation_fn=perturbation_fn,
    )

    # Mark the split (used by MISS augmentation)
    # ds._is_training = bool(training)

    # Optional per-worker resource logging, as in your other datasets
    log_dir = pathlib.Path(log_dir) if log_dir else None
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        resource_log_filename = log_dir / f"resource_file_{rank}_%w.csv"
        ds = MonitoredDataset(
            dataset=ds,
            log_filename=str(resource_log_filename),
            log_interval=10.0,
            monitor_interval=5.0,
        )

    logger.info("VideoGroupDataset created")

    if datasets_weights is not None:
        dist_sampler = DistributedWeightedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True)
    else:
        dist_sampler = torch.utils.data.distributed.DistributedSampler(
            ds, num_replicas=world_size, rank=rank, shuffle=True
        )

    dl_kwargs = dict(
        dataset=ds,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0) and persistent_workers,
        worker_init_fn=_worker_init_fn,
    )
    if num_workers > 0:
        dl_kwargs["prefetch_factor"] = 1

    if deterministic:
        data_loader = torch.utils.data.DataLoader(**dl_kwargs)
    else:
        data_loader = NondeterministicDataLoader(**dl_kwargs)

    logger.info("VideoGroupDataset data loader created")
    return ds, data_loader, dist_sampler


class VideoGroupDataset(Dataset):
    """
    One row per **study/group**. CSV must have:
      - a 'label' column (int)
      - N video columns for the group (any names). We will auto-detect them as
        all non-'label' columns in left-to-right order.
    Each of the N videos yields `num_clips_per_video` temporal clips.
    Total segments returned per sample = group_size * num_clips_per_video.

    S3 is supported via boto3; files are read into memory (no full local mirror).
    """

    def __init__(
        self,
        data_paths,
        group_size,
        frames_per_clip,
        frame_step=None,
        duration=None,
        fps=None,
        num_clips_per_video=1,
        random_clip_sampling=True,
        allow_clip_overlap=False,
        filter_short_videos=False,
        filter_long_videos=int(10**9),
        shared_transform=None,
        transform=None,
        img_size=336,
        training=False,
        miss_augment_prob=0.0,
        min_present=1,
        split_name="train",
        perturbation_fn=None,
    ):
        super().__init__()

        # --- load & normalize CSVs (supports headerless or headered formats) ---
        def _read_group_csv(path: str) -> pd.DataFrame:
            # Try headerless, whitespace-delimited first
            try:
                df = pd.read_csv(path, header=None, sep=r"\s+", engine="python")
                if df.shape[1] == 1:
                    # fallback to "::" or single-space if needed
                    try:
                        df = pd.read_csv(path, header=None, sep="::", engine="python")
                    except Exception:
                        df = pd.read_csv(path, header=None, sep=" ", engine="python")
                ncols = df.shape[1]
                if ncols < 2:
                    raise ValueError(f"CSV '{path}' must have at least 2 columns (>=1 view + label)")
                view_cols = [f"view_{i}" for i in range(ncols - 1)]
                df.columns = view_cols + ["label"]
                return df
            except Exception:
                # Fallback: assume the file already has a header (must include 'label')
                df = pd.read_csv(path)
                if "label" not in df.columns:
                    raise ValueError(f"CSV '{path}' must contain a 'label' column or be headerless.")
                return df

        if isinstance(data_paths, str):
            data_paths = [data_paths]
        dfs = [_read_group_csv(p) for p in data_paths]
        self.df = pd.concat(dfs, ignore_index=True)

        # Auto-detect video columns: everything except 'label'
        self.view_cols = [c for c in self.df.columns if c != "label"]
        if len(self.view_cols) == 0:
            raise ValueError("CSV must have at least one video column besides 'label'")

        # Enforce fixed group size deterministically
        self.view_cols = self.view_cols[:group_size]
        self.group_size = group_size

        # Core temporal / sampling configuration
        self.frames_per_clip = frames_per_clip
        self.frame_step = frame_step
        self.duration = duration
        self.fps = fps
        self.num_clips_per_video = num_clips_per_video
        self.random_clip_sampling = random_clip_sampling
        self.allow_clip_overlap = allow_clip_overlap
        self.filter_short_videos = filter_short_videos
        self.filter_long_videos = filter_long_videos
        self.shared_transform = shared_transform
        self.transform = transform

        self.img_size = int(img_size)
        self.miss_augment_prob = float(miss_augment_prob)
        self.min_present = int(min(min_present, self.group_size))
        self._is_training = bool(training)
        self.split_name = str(split_name)
        self.perturbation_fn = perturbation_fn

        logger.info(
            f"[{self.split_name}] MISS augmentation: p={self.miss_augment_prob} "
            f"min_present={self.min_present} (train={self._is_training})"
        )

        # One S3 client per worker (lazily created in _ensure_s3_client)
        self.s3_client = None

        # Optional per-row anchor-frame table for phase-matched sampling.
        # Keys: row indices (same as __getitem__ input). Values: list of
        # length ``group_size`` where each entry is None (default temporal
        # sampling) or an int raw-frame index to center the K-clip span on
        # for that view. Set by PhaseMatchedStudySampler via
        # ``set_anchors_by_index`` before each epoch.
        self.anchors_by_index: dict[int, list] | None = None

        # Temporal mode validation (match VideoDataset semantics)
        if sum(v is not None for v in (self.fps, self.duration, self.frame_step)) != 1:
            raise ValueError(
                f"Must specify exactly one of fps={self.fps}, duration={self.duration}, or frame_step={self.frame_step}."
            )

        logger.info(f"Loaded {len(self.df)} groups; using columns: {self.view_cols}")

    def __len__(self):
        return len(self.df)

    def set_anchors_by_index(self, table: dict | None) -> None:
        """Install or clear the per-row anchor-frame lookup.

        Call this before an epoch starts (e.g. from the training loop after
        the phase-matched sampler builds its records). Must be called on the
        dataset instance that was passed to the DataLoader; DataLoader
        workers see the installed value via pickle/copy semantics.

        ``table``: dict mapping row index -> list of length ``group_size``
        (each entry None, int anchor frame, or dict with keys
        ``anchor_frame`` + optionally ``frame_step`` for per-clip stride).
        Set to None to disable.
        """
        self.anchors_by_index = table

    def set_pair_dataframe(
        self,
        pair_df: "pd.DataFrame",
        anchors_by_index: dict | None = None,
    ) -> None:
        """Atomically swap in a new pair DataFrame + anchor table.

        The pair DataFrame must have columns ``view_0``, ``view_1``, and
        ``label`` (any additional metadata columns are ignored by the
        loader but kept on the DataFrame for downstream use).
        ``group_size`` must already be 2 on this dataset.

        Anchors are installed last, after ``self.df`` is replaced, so
        indexing cannot fall out of sync mid-update. Both fields are
        mutated together; readers inside ``__getitem__`` see a consistent
        view because dict/DataFrame assignment is atomic in CPython.
        """
        if self.group_size < 2:
            raise ValueError(f"set_pair_dataframe requires group_size >= 2; " f"got group_size={self.group_size}")
        # Required columns scale with group_size: view_0, view_1, ...,
        # view_{group_size-1}, plus ``label``. This supports MV2SV's
        # target_clip (view_3) + fused_clips (view_4+).
        required = {f"view_{i}" for i in range(self.group_size)} | {"label"}
        missing = required - set(pair_df.columns)
        if missing:
            raise KeyError(f"pair DataFrame missing required columns: {missing}")
        # Keep the index contiguous to match anchor-table keys.
        pair_df = pair_df.reset_index(drop=True)
        self.df = pair_df
        self.view_cols = [f"view_{i}" for i in range(self.group_size)]
        self.anchors_by_index = anchors_by_index

    # ---------- S3 helper ----------
    def _ensure_s3_client(self):
        if self.s3_client is None:
            self.s3_client = boto3.client(
                "s3",
                config=Config(max_pool_connections=32, retries={"max_attempts": 5, "mode": "standard"}),
            )

    def _make_dummy_clip(self, fpc: int, h: int = 336, w: int = 336):
        """
        Create a black video clip [fpc, H, W, 3] to stand in for missing/failed views.
        Transforms will resize/crop as usual.
        """
        return np.zeros((fpc, h, w, 3), dtype=np.uint8)

    # ---------- Dataset API ----------
    #
    # Pair-mode return shape: ``(segs, label, clip_indices_out, slot_mask, meta)``
    # where ``meta`` is a dict of per-sample pair metadata pulled from the
    # installed pair DataFrame. The meta dict is included iff the dataset
    # is in pair mode (``self.anchors_by_index`` installed by
    # ``set_pair_dataframe``). The mask collator's ``_phase_call`` path
    # already expects ``sample[4]`` to be a dict, so the shape is
    # compatible; base V-JEPA collate ignores the extra element.
    _PAIR_META_COLS = (
        "study_id",
        "subject_id",
        "sampling_mode",
        "target_phi_a",
        "target_phi_b",
        "circular_phase_diff",
        "frame_step",
        "frames_per_clip",
        "source_span_frames",
        "source_span_seconds_a",
        "source_span_seconds_b",
        "source_span_cycles_a",
        "source_span_cycles_b",
        "clip_a_dicom_id",
        "clip_b_dicom_id",
        "clip_a_anchor_frame",
        "clip_b_anchor_frame",
        "clip_a_phase_at_anchor",
        "clip_b_phase_at_anchor",
        "clip_a_phase_error",
        "clip_b_phase_error",
        "clip_a_view",
        "clip_b_view",
        "clip_a_hr_metadata",
        "clip_b_hr_metadata",
        "clip_a_fps_video",
        "clip_b_fps_video",
        "clip_a_quality_tier",
        "clip_b_quality_tier",
        # --- phase_relational triple-clip extensions (present when
        #     group_size=3 and sampler emits a same-study wrong-phase
        #     hard negative; NaN/empty placeholders otherwise) ---
        "clip_b_neg_dicom_id",
        "clip_b_neg_anchor_frame",
        "clip_b_neg_phase_at_anchor",
        "clip_b_neg_phase_error",
        "clip_b_neg_view",
        "clip_b_neg_hr_metadata",
        "clip_b_neg_fps_video",
        "clip_b_neg_quality_tier",
        "target_phi_b_neg",
        "delta_phase_bucket_pos",
        "delta_phase_bucket_neg",
        "view_pair_class_pos",
        "view_pair_class_neg",
        "hard_neg_available",
        "hard_neg_resample_count",
        # --- MV2SV target_clip + fused_clips columns (Fix 1e). Present
        #     when the sampler was built with mv2sv_config.enabled=True.
        #     target_clip_present is the per-sample 0/1 flag the forward
        #     uses to gate its fallback / fail-loud logic.
        "target_clip_dicom_id",
        "target_clip_view",
        "target_clip_anchor_frame",
        "target_clip_phase_at_anchor",
        "target_clip_phase_error",
        "target_delta_phase",
        "target_clip_present",
        # Fused pool: up to 4 extra clips (beyond target). Ones that the
        # sampler couldn't fill have fused_clip_K_valid=0. We enumerate
        # through K=4 statically so the meta collate schema is stable.
        "fused_clip_1_view",
        "fused_clip_1_delta_phase",
        "fused_clip_1_valid",
        "fused_clip_2_view",
        "fused_clip_2_delta_phase",
        "fused_clip_2_valid",
        "fused_clip_3_view",
        "fused_clip_3_delta_phase",
        "fused_clip_3_valid",
        "fused_clip_4_view",
        "fused_clip_4_delta_phase",
        "fused_clip_4_valid",
    )

    def _row_to_meta(self, row) -> dict:
        """Extract pair metadata, replacing None/NaN with collation-safe
        sentinels. ``default_collate`` can't stack heterogeneous-None
        columns, so we force:
          * str cols (view, dicom_id, subject_id, mode, tier): "" for None
          * numeric cols: float("nan") for None
        Call sites in ``app/vjepa_multiview/train.py::summarize_pair_metadata``
        filter out NaN/empty-string before aggregating.
        """
        out: dict = {}
        str_cols = {
            "study_id",
            "subject_id",
            "sampling_mode",
            "clip_a_dicom_id",
            "clip_b_dicom_id",
            "clip_b_neg_dicom_id",
            "clip_a_view",
            "clip_b_view",
            "clip_b_neg_view",
            "clip_a_quality_tier",
            "clip_b_quality_tier",
            "clip_b_neg_quality_tier",
            "view_pair_class_pos",
            "view_pair_class_neg",
            # MV2SV string metadata.
            "target_clip_dicom_id",
            "target_clip_view",
            "fused_clip_1_view",
            "fused_clip_2_view",
            "fused_clip_3_view",
            "fused_clip_4_view",
        }
        for col in self._PAIR_META_COLS:
            if col not in row.index:
                continue
            v = row[col]
            if v is None or (isinstance(v, float) and v != v):  # NaN check
                out[col] = "" if col in str_cols else float("nan")
            else:
                if col in str_cols and not isinstance(v, str):
                    out[col] = str(v)
                else:
                    out[col] = v
        return out

    def __getitem__(self, index):
        # retry semantics similar to VideoDataset
        while True:
            row = self.df.iloc[index]
            anchors = None
            if self.anchors_by_index is not None:
                anchors = self.anchors_by_index.get(int(index))
            try:
                loaded = self._get_item_row(row, anchors_per_view=anchors)
                if loaded:
                    # Pair mode: append per-sample metadata dict. Detected
                    # by presence of an installed anchor table (which is
                    # set together with pair DataFrame in
                    # ``set_pair_dataframe``).
                    if self.anchors_by_index is not None:
                        meta = self._row_to_meta(row)
                        return (*loaded, meta)
                    return loaded
            except Exception as e:
                warnings.warn(f"Retrying idx={index} due to error: {e}")
            # On retry, drop the anchor — the new random index has no
            # associated anchor, and forcing None prevents a stale anchor
            # from being applied to a different row.
            index = np.random.randint(len(self))

    def _get_item_row(self, row, anchors_per_view: list | None = None):
        """Load one CSV row into (segs, label, clip_indices, slot_mask).

        ``anchors_per_view`` (optional): a list of length ``group_size`` where
        each element is either ``None`` (default random/strided sampling) or an
        int specifying the raw-frame index to center the K-clip span on for
        that view. Used by ``PhaseMatchedStudySampler`` to request
        phase-matched temporal windows across views.
        """
        label = float(row["label"])

        # ---- collect URIs and initial presence flags from CSV ----
        uris, present = [], []
        for c in self.view_cols:
            v = row[c]
            if isinstance(v, str):
                v = v.strip()
            is_missing = (v is None) or (v == "") or (isinstance(v, float) and math.isnan(v)) or (v == MISSING_TOKEN)
            if is_missing:
                uris.append(None)
                present.append(0)
            else:
                uris.append(v)
                present.append(1)

        # ---- stochastic view-level MISS augmentation (training only) ----
        # Flip some PRESENT views to missing with prob self.miss_augment_prob,
        # while enforcing at least self.min_present survivors.
        if self._is_training and self.miss_augment_prob > 0.0:
            rng = np.random.default_rng()
            pres_idx = [i for i, p in enumerate(present) if p]
            if len(pres_idx) > 0:
                drops = rng.random(len(pres_idx)) < float(self.miss_augment_prob)

                # survivors if we applied the drops
                survivors = [i for i, d in zip(pres_idx, drops) if not d]
                need_min = max(1, int(getattr(self, "min_present", 1)))
                restore = set()
                if len(survivors) < need_min:
                    need = need_min - len(survivors)
                    dropped = [i for i, d in zip(pres_idx, drops) if d]
                    if len(dropped) > 0:
                        restore_sel = rng.choice(dropped, size=min(need, len(dropped)), replace=False)
                        restore = set(int(x) for x in np.atleast_1d(restore_sel))

                # apply the (possibly corrected) drops
                for i, d in zip(pres_idx, drops):
                    if d and (i not in restore):
                        uris[i] = None
                        present[i] = 0

        # ---- load/construct clips per slot ----
        segs, clip_indices_out, slot_mask = [], [], []
        for view_idx, (uri, p) in enumerate(zip(uris, present)):
            anchor = None
            if anchors_per_view is not None and view_idx < len(anchors_per_view):
                a = anchors_per_view[view_idx]
                if a is None:
                    anchor = None
                elif isinstance(a, dict):
                    # Dict form: passed through to _loadvideo_decord_multi,
                    # which handles the anchor_frame + optional frame_step
                    # keys.
                    anchor = a
                else:
                    anchor = int(a)
            if not p:
                # Missing view → dummy black clips (shape compatible with transforms)
                T = self.frames_per_clip
                H = W = self.img_size
                dummy = np.zeros((T, H, W, 3), dtype=np.uint8)
                clips = [dummy for _ in range(self.num_clips_per_video)]
                idxs = [np.arange(T, dtype=np.int64) for _ in range(self.num_clips_per_video)]
            else:
                # Contiguous multi-clip loader (K clips of length fpc, non-overlapping).
                # If an anchor frame was provided (phase-matched sampling), the
                # K-window span is centered on it.
                clips, idxs = self._loadvideo_decord_multi(
                    uri,
                    self.frames_per_clip,
                    self.num_clips_per_video,
                    anchor_frame=anchor,
                )
                if clips is None or len(clips) == 0:
                    # Fallback to dummy if load failed
                    T = self.frames_per_clip
                    H = W = self.img_size
                    dummy = np.zeros((T, H, W, 3), dtype=np.uint8)
                    clips = [dummy for _ in range(self.num_clips_per_video)]
                    idxs = [np.arange(T, dtype=np.int64) for _ in range(self.num_clips_per_video)]

            if self.transform is not None:
                clips = [self.transform(c) for c in clips]

            # Apply perturbation if set (for noise robustness evaluation).
            # Operates on normalized tensors: un-normalize → perturb in [0,1] → re-normalize.
            if self.perturbation_fn is not None and p:  # only perturb present views
                MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
                STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
                perturbed_clips = []
                for clip_or_list in clips:
                    if isinstance(clip_or_list, (list, tuple)):
                        clip = clip_or_list[0]
                        pixel = (clip * STD + MEAN).clamp(0, 1)
                        pixel = self.perturbation_fn(pixel, uri)
                        perturbed_clips.append([(pixel - MEAN) / STD])
                    else:
                        clip = clip_or_list
                        pixel = (clip * STD + MEAN).clamp(0, 1)
                        pixel = self.perturbation_fn(pixel, uri)
                        perturbed_clips.append((pixel - MEAN) / STD)
                clips = perturbed_clips

            segs.extend(clips)
            clip_indices_out.extend(idxs)
            # Per-clip presence flag (replicate the view's presence for its K clips)
            slot_mask.extend([bool(p)] * len(clips))

        return segs, label, clip_indices_out, torch.tensor(slot_mask, dtype=torch.bool)

    def _open_vr(self, sample_uri: str):
        # Local path
        if not (isinstance(sample_uri, str) and sample_uri.startswith("s3://")):
            if not os.path.exists(sample_uri):
                warnings.warn(f"video path not found fname='{sample_uri}'")
                return None
            if self.filter_long_videos:
                try:
                    _fsize = os.path.getsize(sample_uri)
                    if _fsize > self.filter_long_videos:
                        warnings.warn(f"skipping long video of size _fsize={_fsize} (bytes)")
                        return None
                except Exception:
                    pass
            try:
                return VideoReader(sample_uri, num_threads=-1, ctx=cpu(0))
            except Exception as e:
                logger.warning(f"VideoReader local fail: {e}")
                return None

        # S3 path
        try:
            bucket, key = sample_uri.replace("s3://", "").split("/", 1)
            self._ensure_s3_client()

            try:
                head = self.s3_client.head_object(Bucket=bucket, Key=key)
            except self.s3_client.exceptions.NoSuchKey:
                warnings.warn(f"video path not found fname='{sample_uri}'")
                return None
            except self.s3_client.exceptions.ClientError as e:
                logger.warning(f"S3 access error for {sample_uri}: {e}")
                return None

            fsize = head.get("ContentLength", 0)
            if self.filter_long_videos and fsize > self.filter_long_videos:
                warnings.warn(f"skipping long video of size _fsize={fsize} (bytes)")
                return None

            obj = self.s3_client.get_object(Bucket=bucket, Key=key)
            data = obj["Body"].read()
            if not data:
                logger.warning(f"Empty S3 object: {sample_uri}")
                return None

            bio = io.BytesIO(data)
            return VideoReader(bio, num_threads=-1, ctx=cpu(0))
        except Exception as e:
            logger.warning(f"Failed to open video: {sample_uri}\n{e}")
            return None

    def _loadvideo_decord_multi(self, sample_uri: str, fpc: int, k: int, anchor_frame=None):
        """
        Return K clips, each exactly fpc frames.

        - ``anchor_frame=None``: legacy behavior — K strided windows
          starting at ``i*clip_len`` at the dataset-configured ``frame_step``.
        - ``anchor_frame=<int>``: each of K windows is centered on
          ``anchor_frame`` (same window for all K clips). Use ``k=1`` for
          phase-matched training; ``k>1`` with a single int anchor is
          redundant and should be avoided.
        - ``anchor_frame=<dict>`` with keys ``anchor_frame`` and optionally
          ``frame_step``: window is anchor-centered at the specified
          per-clip frame_step, overriding the dataset default. This is the
          path used by ``PhaseMatchedStudySampler``.

        Uses ``_compute_clip_indices`` for the index math so every path is
        unit-testable.
        """
        vr = self._open_vr(sample_uri)
        if vr is None:
            return [], None

        # --- derive stride and window size ---
        fstp = self.frame_step
        if (self.duration is not None) or (self.fps is not None):
            try:
                video_fps = max(1, int(math.ceil(vr.get_avg_fps())))
            except Exception as e:
                logger.warning(f"fps read failed: {e}")
                video_fps = None

            if self.duration is not None:
                assert self.fps is None
                if video_fps is None:
                    return [], None
                fstp = max(1, int(self.duration * video_fps / fpc))
            else:
                assert self.duration is None
                if video_fps is None:
                    return [], None
                fstp = max(1, int(video_fps // max(1, self.fps)))

        assert fstp is not None and fstp > 0
        V = len(vr)  # total raw frames

        # Unpack the anchor argument:
        #   None -> default strided-from-0
        #   int  -> anchor-centered with default frame_step
        #   dict -> per-clip {'anchor_frame': int, 'frame_step': int | None}
        anchor_val = None
        per_clip_fstp = fstp
        if isinstance(anchor_frame, dict):
            anchor_val = anchor_frame.get("anchor_frame")
            fs_override = anchor_frame.get("frame_step")
            if fs_override is not None:
                per_clip_fstp = int(fs_override)
        elif anchor_frame is not None:
            anchor_val = int(anchor_frame)

        per_clip_inds = []
        per_clip_meta = []
        for i in range(k):
            inds, meta = _compute_clip_indices(
                num_frames=V if V > 0 else 1,
                fpc=fpc,
                frame_step=per_clip_fstp,
                clip_idx=i,
                num_clips=k,
                anchor_frame=anchor_val,
            )
            # If V == 0 we already zeroed indices; keep them inside [0, V-1].
            if V > 0:
                inds = np.clip(inds, 0, V - 1)
            else:
                inds = np.zeros_like(inds)
            per_clip_inds.append(inds)
            per_clip_meta.append(meta)

        all_inds = np.concatenate(per_clip_inds, axis=0)
        frames_all = vr.get_batch(all_inds).asnumpy()  # [sum_k fpc, H, W, 3]

        clips = []
        offset = 0
        for _ in range(k):
            clips.append(frames_all[offset : offset + fpc])
            offset += fpc

        # Surface per-clip metadata so the dataset can log / return it if
        # it wants. Stored on the instance for the most recent call.
        self._last_clip_meta = per_clip_meta
        return clips, per_clip_inds

    # ---------- Sampling (shared with single-video dataset) ----------
    def _sample_from_vr(self, vr, fpc):
        """
        Returns (buffer[T,H,W,3], indices[np.int64, shape=(fpc,)])
        Picks a random valid window only when there is room to slide it.
        """
        # Resolve effective frame step
        fstp = self.frame_step
        if (self.duration is not None) or (self.fps is not None):
            try:
                video_fps = max(1, int(math.ceil(vr.get_avg_fps())))
            except Exception as e:
                logger.warning(f"fps read failed: {e}")
                video_fps = None

            if self.duration is not None:
                assert self.fps is None
                if video_fps is None:
                    raise RuntimeError("duration mode requires readable FPS")
                fstp = max(1, int(self.duration * video_fps / fpc))
            else:
                assert self.duration is None
                if video_fps is None:
                    raise RuntimeError("fps mode requires readable FPS")
                fstp = max(1, int(video_fps // max(1, self.fps)))

        assert fstp is not None and fstp > 0
        clip_len = int(fpc * fstp)
        V = len(vr)

        if V < clip_len:
            # Too short → spread indices and pad to fpc if needed
            base = max(1, V // max(1, fstp))
            inds = np.linspace(0, max(0, V), num=base)
            if base < fpc:
                inds = np.concatenate((inds, np.ones(fpc - base) * max(0, V - 1)))
            inds = np.clip(inds, 0, max(0, V - 1)).astype(np.int64)
        else:
            # Enough frames; randomize only when there is slack
            if self.random_clip_sampling and (V > clip_len):
                # randint upper bound is EXCLUSIVE → use V+1 to allow end==V
                end_indx = np.random.randint(clip_len, V + 1)
            else:
                end_indx = clip_len  # single valid placement, no randomness
            start_indx = end_indx - clip_len
            inds = np.linspace(start_indx, end_indx, num=fpc)
            # Ensure [start, end) with integer frame indices
            inds = np.clip(inds, start_indx, max(start_indx, end_indx - 1)).astype(np.int64)

        buffer = vr.get_batch(inds).asnumpy()  # [T,H,W,3], uint8
        return buffer, inds

    def _split_into_clips(self, buffer, base_indices, fpc, num_clips):
        """
        Split a loaded video buffer into `num_clips` temporal clips of length fpc (frames),
        mirroring the main sampler’s semantics, and avoiding invalid randint ranges.
        Returns (clips: List[np.ndarray[T,H,W,3]], idx_slices: List[np.int64[fpc]]).
        """
        T = int(buffer.shape[0])
        if num_clips <= 1 or T <= fpc:
            # Not enough frames or no split requested
            inds = np.arange(min(fpc, T), dtype=np.int64)
            if len(inds) < fpc:
                pad = np.ones(fpc - len(inds), dtype=np.int64) * max(0, len(inds) - 1)
                inds = np.concatenate((inds, pad))
            return [buffer[inds]], [inds]

        partition_len = T // num_clips
        clips, idx_slices = [], []

        for i in range(num_clips):
            if partition_len > fpc:
                # Random window inside this partition only if there is slack
                end_indx = fpc
                if self.random_clip_sampling and (partition_len > fpc):
                    # EXCLUSIVE upper bound → +1 to allow end==partition_len
                    end_indx = np.random.randint(fpc, partition_len + 1)
                start_indx = end_indx - fpc
                inds = np.linspace(start_indx, end_indx, num=fpc)
                inds = np.clip(inds, start_indx, max(start_indx, end_indx - 1)).astype(np.int64)
                inds = inds + i * partition_len
            else:
                if not self.allow_clip_overlap:
                    # Evenly spread within the partition; pad if needed
                    # Use step-aware count if frame_step is defined
                    step = max(1, (self.frame_step or 1))
                    base = max(1, partition_len // step)
                    inds = np.linspace(0, partition_len, num=base)
                    if base < fpc:
                        inds = np.concatenate((inds, np.ones(fpc - base) * max(0, partition_len - 1)))
                    inds = np.clip(inds, 0, max(0, partition_len - 1)).astype(np.int64)
                    inds = inds + i * partition_len
                else:
                    # Allow overlap across the whole sequence; slide partitions
                    sample_len = max(1, min(fpc, T) - 1)
                    base = max(1, sample_len)
                    inds = np.linspace(0, sample_len, num=base)
                    if base < fpc:
                        inds = np.concatenate((inds, np.ones(fpc - base) * sample_len))
                    inds = np.clip(inds, 0, sample_len).astype(np.int64)
                    clip_step = 0
                    if T > fpc and num_clips > 1:
                        clip_step = (T - fpc) // (num_clips - 1)
                    inds = inds + i * clip_step

            clips.append(buffer[inds])
            idx_slices.append(inds)

        return clips, idx_slices
