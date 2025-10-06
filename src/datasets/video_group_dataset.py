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
    data_paths,                  # str | list[str]  (CSV path(s))
    batch_size,
    group_size,                  # maps to num_segments from config
    frames_per_clip,
    frame_step=None,
    duration=None,
    fps=None,
    num_clips_per_video=1,       # NEW in your pipeline: per-video temporal clips
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
):
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
    )

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
    
        # One S3 client per worker (lazily created in _ensure_s3_client)
        self.s3_client = None
    
        # Temporal mode validation (match VideoDataset semantics)
        if sum(v is not None for v in (self.fps, self.duration, self.frame_step)) != 1:
            raise ValueError(
                f"Must specify exactly one of fps={self.fps}, duration={self.duration}, or frame_step={self.frame_step}."
            )
    
        logger.info(f"Loaded {len(self.df)} groups; using columns: {self.view_cols}")


    def __len__(self):
        return len(self.df)

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
    def __getitem__(self, index):
        # retry semantics similar to VideoDataset
        while True:
            row = self.df.iloc[index]
            try:
                loaded = self._get_item_row(row)
                if loaded:
                    return loaded
            except Exception as e:
                warnings.warn(f"Retrying idx={index} due to error: {e}")
            index = np.random.randint(len(self))

    def _get_item_row(self, row):
        label = int(row["label"])
    
        # Collect URIs for this group (order preserved)
        uris = []
        for c in self.view_cols:
            v = row[c]
            if isinstance(v, str) and len(v.strip()) > 0:
                uris.append(v.strip())
            else:
                uris.append(None)
    
        segs = []
        clip_indices_out = []
    
        # For each video in the group, sample `num_clips_per_video` *independent* clips
        for uri in uris:
            if uri is None:
                # synthesize K dummy clips to preserve fixed shape
                for _ in range(self.num_clips_per_video):
                    dummy = self._make_dummy_clip(self.frames_per_clip)
                    if self.transform is not None:
                        dummy = self.transform(dummy)
                    segs.append(dummy)
                    clip_indices_out.append(np.arange(self.frames_per_clip, dtype=np.int64))
                continue
    
            clips_i, idxs_i = self._loadvideo_decord_multi(
                sample_uri=uri,
                fpc=self.frames_per_clip,
                k=self.num_clips_per_video,
            )
    
            # On failure, backfill with dummy clips
            if not clips_i:
                for _ in range(self.num_clips_per_video):
                    dummy = self._make_dummy_clip(self.frames_per_clip)
                    if self.transform is not None:
                        dummy = self.transform(dummy)
                    segs.append(dummy)
                    clip_indices_out.append(np.arange(self.frames_per_clip, dtype=np.int64))
                continue
    
            # Apply transforms per clip (each becomes a list of spatial views; with num_views_per_segment=1 → length 1)
            if self.shared_transform is not None:
                clips_i = [self.shared_transform(c) for c in clips_i]
            if self.transform is not None:
                clips_i = [self.transform(c) for c in clips_i]
    
            segs.extend(clips_i)
            clip_indices_out.extend(idxs_i)
    
        return segs, label, clip_indices_out

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


    # ---------- Core video loader (local or S3) ----------
    def _loadvideo_decord_multi(self, sample_uri: str, fpc: int, k: int):
        """
        Open (local or S3) and return K *independent* clips, each with exactly `fpc` frames.
        Returns:
          clips: List[np.ndarray [fpc, H, W, 3] (uint8)]
          per_clip_inds: List[np.int64[fpc]]
        """
        vr = self._open_vr(sample_uri)
        if vr is None:
            return [], None
    
        # Resolve effective frame step (same semantics as single-clip sampler)
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
        clip_len = int(fpc * fstp)
        V = len(vr)
    
        # Build K index arrays; partition the timeline and pick one window per partition when possible
        per_clip_inds = []
        all_inds = []
    
        if k <= 1:
            # Single-clip path (identical to previous semantics, but inlined)
            if V < clip_len:
                base = max(1, V // max(1, fstp))
                inds = np.linspace(0, max(0, V), num=base)
                if base < fpc:
                    inds = np.concatenate((inds, np.ones(fpc - base) * max(0, V - 1)))
                inds = np.clip(inds, 0, max(0, V - 1)).astype(np.int64)
            else:
                if self.random_clip_sampling and (V > clip_len):
                    end_idx = np.random.randint(clip_len, V + 1)  # high is exclusive
                else:
                    end_idx = clip_len
                start_idx = end_idx - clip_len
                inds = np.linspace(start_idx, end_idx, num=fpc)
                inds = np.clip(inds, start_idx, max(start_idx, end_idx - 1)).astype(np.int64)
    
            per_clip_inds.append(inds)
            frames = vr.get_batch(inds).asnumpy()
            return [frames], per_clip_inds
    
        # Multi-clip: slice the timeline into ~k partitions; last partition takes the remainder
        part = max(1, V // k)
        for i in range(k):
            seg_start = i * part
            seg_end = V if i == k - 1 else (i + 1) * part
            seg_len = max(0, seg_end - seg_start)
    
            if seg_len >= clip_len:
                # Enough slack: optional random position within partition
                if self.random_clip_sampling and (seg_len > clip_len):
                    end_idx = np.random.randint(seg_start + clip_len, seg_end + 1)  # exclusive high
                else:
                    end_idx = seg_start + clip_len
                start_idx = end_idx - clip_len
                inds = np.linspace(start_idx, end_idx, num=fpc)
                inds = np.clip(inds, start_idx, max(start_idx, end_idx - 1)).astype(np.int64)
            else:
                # Not enough frames in this partition: spread & pad to fpc
                step = max(1, fstp)
                base = max(1, seg_len // step)
                inds = np.linspace(seg_start, seg_end, num=base)
                if base < fpc:
                    inds = np.concatenate((inds, np.ones(fpc - base) * max(seg_start, seg_end - 1)))
                inds = np.clip(inds, seg_start, max(seg_start, seg_end - 1)).astype(np.int64)
    
            per_clip_inds.append(inds)
            all_inds.extend(list(inds))
    
        # Single batched fetch, then split back into K clips by lengths
        all_inds = np.asarray(all_inds, dtype=np.int64)
        frames_all = vr.get_batch(all_inds).asnumpy()
    
        clips = []
        offset = 0
        for inds in per_clip_inds:
            t = len(inds)
            clips.append(frames_all[offset:offset + t])
            offset += t
    
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
