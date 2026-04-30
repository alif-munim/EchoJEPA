# src/datasets/video_dataset.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import os
import pathlib
import warnings
from logging import getLogger

import io
import boto3
from botocore.config import Config

import numpy as np
import pandas as pd
import torch
import torchvision
from decord import VideoReader, cpu

from src.datasets.utils.dataloader import ConcatIndices, MonitoredDataset, NondeterministicDataLoader
from src.datasets.utils.weighted_sampler import DistributedWeightedSampler

_GLOBAL_SEED = 0
logger = getLogger()
  


def _worker_init_fn(_):
    # keep each worker to 1 CPU thread to avoid oversubscription
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


class _PerturbationFn:
    """Picklable perturbation callable for DataLoader workers."""
    def __init__(self, ptype, severity, transducer_pos=(0.5, 0.0)):
        self.ptype = ptype
        self.severity = severity
        self.transducer_pos = transducer_pos

    def __call__(self, clip, video_path):
        import hashlib
        from scripts.neurips.echo_perturbations import apply_perturbation, create_scan_mask
        seed = int(hashlib.md5(str(video_path).encode()).hexdigest()[:8], 16)
        mask = create_scan_mask(clip[:, 0, :, :])
        return apply_perturbation(clip, self.ptype, self.severity, scan_mask=mask,
                                  seed=seed, transducer_pos=self.transducer_pos)


def make_videodataset(
    data_paths,
    batch_size,
    frames_per_clip=8,
    dataset_fpcs=None,
    frame_step=4,
    duration=None,
    fps=None,
    num_clips=1,
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
    study_sampling=False,
    class_balance_ratio=None,
    phase_metadata_csv=None,
):
    # Check for perturbation via environment variables (for noise robustness evaluation).
    # Set PERTURBATION_TYPE and PERTURBATION_SEVERITY to enable.
    # Example: PERTURBATION_TYPE=depth_attenuation PERTURBATION_SEVERITY=moderate python -m evals.main ...
    perturbation_fn = None
    ptype = os.environ.get("PERTURBATION_TYPE")
    psev = os.environ.get("PERTURBATION_SEVERITY")
    if ptype and psev:
        transducer_pos = tuple(
            float(x) for x in os.environ.get("TRANSDUCER_POS", "0.5,0.0").split(",")
        )
        perturbation_fn = _PerturbationFn(ptype, psev, transducer_pos)
        logger.info(f"Perturbation enabled: {ptype}/{psev} (transducer_pos={transducer_pos})")

    # Check for temporal ablation via environment variables.
    # FRAME_SHUFFLE=<seed>  — base seed for stochastic shuffles (required for frame/tubelet, ignored for reverse)
    # FRAME_SHUFFLE_TYPE=frame|tubelet|reverse  — shuffle type (default: frame)
    #   frame:   permute all frames independently (destroys all temporal structure)
    #   tubelet: permute pairs of consecutive frames (preserves local motion, destroys phase)
    #   reverse: play video backwards (preserves local motion magnitude, reverses cardiac cycle direction)
    # Example: FRAME_SHUFFLE=100 python -m evals.main ...                         # frame shuffle
    # Example: FRAME_SHUFFLE=100 FRAME_SHUFFLE_TYPE=tubelet python -m evals.main ...  # tubelet shuffle
    # Example: FRAME_SHUFFLE=0 FRAME_SHUFFLE_TYPE=reverse python -m evals.main ...    # temporal reversal
    frame_shuffle_seed = None
    frame_shuffle_type = "frame"
    fs_env = os.environ.get("FRAME_SHUFFLE")
    if fs_env:
        frame_shuffle_seed = int(fs_env)
        frame_shuffle_type = os.environ.get("FRAME_SHUFFLE_TYPE", "frame")
        logger.info(f"Temporal ablation enabled: type={frame_shuffle_type}, base_seed={frame_shuffle_seed}")

    dataset = VideoDataset(
        data_paths=data_paths,
        datasets_weights=datasets_weights,
        frames_per_clip=frames_per_clip,
        dataset_fpcs=dataset_fpcs,
        duration=duration,
        fps=fps,
        frame_step=frame_step,
        num_clips=num_clips,
        random_clip_sampling=random_clip_sampling,
        allow_clip_overlap=allow_clip_overlap,
        filter_short_videos=filter_short_videos,
        filter_long_videos=filter_long_videos,
        shared_transform=shared_transform,
        transform=transform,
        perturbation_fn=perturbation_fn,
        frame_shuffle_seed=frame_shuffle_seed,
        frame_shuffle_type=frame_shuffle_type,
        phase_metadata_csv=phase_metadata_csv,
    )

    log_dir = pathlib.Path(log_dir) if log_dir else None
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        # Worker ID will replace '%w'
        resource_log_filename = log_dir / f"resource_file_{rank}_%w.csv"
        dataset = MonitoredDataset(
            dataset=dataset,
            log_filename=str(resource_log_filename),
            log_interval=10.0,
            monitor_interval=5.0,
        )

    logger.info("VideoDataset dataset created")
    if study_sampling:
        from src.datasets.study_sampler import DistributedStudySampler

        dist_sampler = DistributedStudySampler(
            dataset, num_replicas=world_size, rank=rank,
            class_balance_ratio=class_balance_ratio,
        )
        logger.info(f"Using DistributedStudySampler: {dist_sampler.num_studies} studies, "
                     f"{len(dataset)} total clips, 1 clip/study/epoch")
    elif datasets_weights is not None:
        dist_sampler = DistributedWeightedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    else:
        dist_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True
        )

    dl_kwargs = dict(
        dataset=dataset,
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
        dl_kwargs["prefetch_factor"] = 8  # increased from 4; buffers more S3 downloads ahead of GPU

    if deterministic:
        data_loader = torch.utils.data.DataLoader(**dl_kwargs)
    else:
        # custom loader variant with relaxed determinism
        data_loader = NondeterministicDataLoader(**dl_kwargs)

    logger.info("VideoDataset unsupervised data loader created")
    return dataset, data_loader, dist_sampler


class VideoDataset(torch.utils.data.Dataset):
    """
    Video classification dataset that supports both local filesystem and S3 paths.
    """

    def __init__(
        self,
        data_paths,
        datasets_weights=None,
        frames_per_clip=16,
        fps=None,
        dataset_fpcs=None,
        frame_step=4,
        num_clips=1,
        transform=None,
        shared_transform=None,
        random_clip_sampling=True,
        allow_clip_overlap=False,
        filter_short_videos=False,
        filter_long_videos=int(10**9),
        duration=None,  # duration in seconds
        perturbation_fn=None,  # Optional: callable(clip_tensor, video_path) -> clip_tensor
        frame_shuffle_seed=None,  # Optional: int seed for temporal ablation
        frame_shuffle_type="frame",  # "frame", "tubelet", or "reverse"
        phase_metadata_csv=None,  # Optional: path to per-clip HR/FrameTime CSV (phi-JEPA)
    ):
        self.data_paths = data_paths
        self.datasets_weights = datasets_weights
        self.frame_step = frame_step
        self.num_clips = num_clips
        self.transform = transform
        self.shared_transform = shared_transform
        self.random_clip_sampling = random_clip_sampling
        self.allow_clip_overlap = allow_clip_overlap
        self.filter_short_videos = filter_short_videos
        self.filter_long_videos = filter_long_videos
        self.duration = duration
        self.fps = fps
        self.perturbation_fn = perturbation_fn
        self.frame_shuffle_seed = frame_shuffle_seed
        self.frame_shuffle_type = frame_shuffle_type
        self.phase_metadata_csv = phase_metadata_csv
        self.phase_metadata = None
        if phase_metadata_csv is not None:
            from src.datasets.phase_utils import load_phase_metadata
            self.phase_metadata = load_phase_metadata(phase_metadata_csv)
            logger.info(
                f"Loaded phase metadata: {len(self.phase_metadata)} clips from {phase_metadata_csv}"
            )

        # Initialize S3 client lazily per worker (avoid pickling/FD sharing)
        self.s3_client = None

        if sum([v is not None for v in (fps, duration, frame_step)]) != 1:
            raise ValueError(f"Must specify exactly one of either {fps=}, {duration=}, or {frame_step=}.")

        if isinstance(data_paths, str):
            data_paths = [data_paths]

        if dataset_fpcs is None:
            self.dataset_fpcs = [frames_per_clip for _ in data_paths]
        else:
            if len(dataset_fpcs) != len(data_paths):
                raise ValueError("Frames per clip not properly specified for data paths")
            self.dataset_fpcs = dataset_fpcs

        if VideoReader is None:
            raise ImportError('Unable to import "decord" which is required to read videos.')

        # Load video paths and labels from the annotation file(s).
        # Supports two CSV formats (auto-detected by column count):
        #   2-col: uri label
        #   3-col: uri anchor_frame label  (anchor-aware sampling)
        # When 3-col is used, _sample_from_vr centers a single window on
        # anchor_frame; num_clips is forced to 1. Caller must ensure
        # num_segments/num_clips align.
        samples, labels, anchors = [], [], []
        self.num_samples_per_dataset = []
        for data_path in self.data_paths:
            if data_path.endswith(".csv"):
                try:
                    data = pd.read_csv(data_path, header=None, delimiter=" ")
                except pd.errors.ParserError:
                    data = pd.read_csv(data_path, header=None, delimiter="::")
                n_cols = data.shape[1]
                samples.extend(list(data.values[:, 0]))
                if n_cols >= 3:
                    anchors.extend(list(data.values[:, 1].astype(np.int64)))
                    labels.extend(list(data.values[:, 2]))
                else:
                    anchors.extend([None] * len(data))
                    labels.extend(list(data.values[:, 1]))
                self.num_samples_per_dataset.append(len(data))
            elif data_path.endswith(".npy"):
                data = np.load(data_path, allow_pickle=True)
                data = [repr(x)[1:-1] for x in data]
                samples.extend(data)
                anchors.extend([None] * len(data))
                labels.extend([0] * len(data))
                self.num_samples_per_dataset.append(len(data))

        self.per_dataset_indices = ConcatIndices(self.num_samples_per_dataset)

        self.sample_weights = None
        if self.datasets_weights is not None:
            self.sample_weights = []
            for dw, ns in zip(self.datasets_weights, self.num_samples_per_dataset):
                self.sample_weights.extend([dw / ns] * ns)

        self.samples = samples
        self.labels = labels
        # Anchor-aware sampling: if any sample has an anchor, anchor-mode is active.
        self.anchors = anchors if any(a is not None for a in anchors) else None
        self._current_anchor = None  # Set per-sample before loadvideo_decord
        if self.anchors is not None and self.num_clips != 1:
            logger.warning(
                f"Anchor sampling activated but num_clips={self.num_clips} "
                f"(!=1). Forcing num_clips=1 for anchor mode."
            )
            self.num_clips = 1

        # Track load-failure substitutions (per-worker counter; not shared across processes)
        self._substitution_count = 0

        logger.info(f"Loaded {len(self.samples)} samples")
        if len(self.samples) > 0:
            logger.info(f"First 5 samples: {self.samples[:5]}")
            logger.info(f"Sample types: {[type(s) for s in self.samples[:5]]}")

    # ---------- S3 helper ----------
    def _ensure_s3_client(self):
        if self.s3_client is None:
            self.s3_client = boto3.client(
                "s3",
                config=Config(
                    max_pool_connections=32,
                    retries={"max_attempts": 5, "mode": "standard"},
                ),
            )

    @property
    def substitution_count(self):
        """Number of times a failed load was replaced with a random sample."""
        return self._substitution_count

    def _lookup_phase_meta(self, sample_uri):
        """Return phase metadata dict for this clip. Empty dict if not configured.

        Fields: hr_bpm (float or nan), frame_time_ms (float or nan).
        Clips flagged as irregular-rhythm or with invalid HR emit nan for hr_bpm.
        """
        if self.phase_metadata is None:
            return {}
        from src.datasets.phase_utils import parse_clip_id, sanitize_hr
        clip_id = parse_clip_id(str(sample_uri))
        row = self.phase_metadata.get(clip_id)
        if row is None:
            return {"hr_bpm": float("nan"), "frame_time_ms": float("nan")}
        hr_raw, ft, irr = row
        return {"hr_bpm": sanitize_hr(hr_raw, irr), "frame_time_ms": float(ft)}

    # ---------- Dataset API ----------
    def __getitem__(self, index):
        original_index = index
        # Cap retries so a bad-data worker cannot stall the collective indefinitely.
        # Each miss picks a fresh random index; 32 attempts is ample for realistic
        # corruption rates (<<1%) while still surfacing systemic failures.
        MAX_RETRIES = 32
        for attempt in range(MAX_RETRIES):
            sample_path = self.samples[index]
            loaded = None
            try:
                if isinstance(sample_path, str):
                    is_image = sample_path.split(".")[-1].lower() in ("jpg", "jpeg", "png")
                    loaded = self.get_item_image(index) if is_image else self.get_item_video(index)
            except Exception as e:
                logger.warning(f"Unhandled load error for {sample_path}: {e}")
                loaded = None
            if loaded:
                if index != original_index:
                    self._substitution_count += 1
                    logger.warning(
                        f"Load-failure substitution (attempt {attempt}): "
                        f"requested {original_index} ({self.samples[original_index]}), "
                        f"served {index} ({self.samples[index]})"
                    )
                return loaded
            warnings.warn(f"Retrying with new sample, failed to load: {self.samples[index]}")
            index = np.random.randint(len(self))
        raise RuntimeError(
            f"Dataset failed to produce a valid sample after {MAX_RETRIES} retries "
            f"(original_index={original_index}, last_index={index})."
        )

    def get_item_video(self, index):
        sample_uri = self.samples[index]
        dataset_idx, _ = self.per_dataset_indices[index]
        frames_per_clip = self.dataset_fpcs[dataset_idx]

        # Anchor-aware mode: forward the per-sample anchor to _sample_from_vr.
        self._current_anchor = self.anchors[index] if self.anchors is not None else None
        buffer, clip_indices = self.loadvideo_decord(sample_uri, frames_per_clip)
        if buffer is None or len(buffer) == 0:
            return None

        label = self.labels[index]

        def split_into_clips(video):
            fpc = frames_per_clip
            nc = self.num_clips
            return [video[i * fpc : (i + 1) * fpc] for i in range(nc)]

        if self.shared_transform is not None:
            buffer = self.shared_transform(buffer)
        buffer = split_into_clips(buffer)
        if self.transform is not None:
            buffer = [self.transform(clip) for clip in buffer]

        # Apply perturbation if set (for noise robustness evaluation).
        # Operates on normalized tensors: un-normalize → perturb in [0,1] → re-normalize.
        # Transform returns list-wrapped tensors: [[tensor], [tensor], ...].
        if self.perturbation_fn is not None:
            MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
            STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
            perturbed = []
            for clip_or_list in buffer:
                # Unwrap list if transform wrapped it (eval mode returns [tensor])
                if isinstance(clip_or_list, (list, tuple)):
                    clip = clip_or_list[0]
                    pixel = (clip * STD + MEAN).clamp(0, 1)
                    pixel = self.perturbation_fn(pixel, sample_uri)
                    perturbed.append([(pixel - MEAN) / STD])
                else:
                    clip = clip_or_list
                    pixel = (clip * STD + MEAN).clamp(0, 1)
                    pixel = self.perturbation_fn(pixel, sample_uri)
                    perturbed.append((pixel - MEAN) / STD)
            buffer = perturbed

        # Apply temporal ablation if set.
        # Types: frame (permute all), tubelet (permute pairs), reverse (flip temporal axis).
        if self.frame_shuffle_seed is not None:
            import hashlib
            video_hash = int(hashlib.md5(str(sample_uri).encode()).hexdigest()[:8], 16)
            rng = np.random.RandomState(video_hash + self.frame_shuffle_seed)
            shuffled = []
            for clip_or_list in buffer:
                is_wrapped = isinstance(clip_or_list, (list, tuple))
                clip = clip_or_list[0] if is_wrapped else clip_or_list
                T = clip.shape[1]

                if self.frame_shuffle_type == "reverse":
                    clip = clip[:, torch.arange(T - 1, -1, -1), :, :]
                elif self.frame_shuffle_type == "tubelet":
                    n_tubelets = T // 2
                    tubelet_perm = rng.permutation(n_tubelets)
                    indices = []
                    for ti in tubelet_perm:
                        indices.extend([ti * 2, ti * 2 + 1])
                    if T % 2 == 1:
                        indices.append(T - 1)
                    clip = clip[:, indices, :, :]
                elif self.frame_shuffle_type == "matched":
                    # Tubelet-level shuffle with FIXED perm (same for all videos).
                    # Encoder remaps RoPE positions to match content's original temporal position.
                    # Uses base seed only (not per-video) so encoder can reconstruct the same perm.
                    n_tubelets = T // 2
                    fixed_rng = np.random.RandomState(self.frame_shuffle_seed)
                    tubelet_perm = fixed_rng.permutation(n_tubelets)
                    indices = []
                    for ti in tubelet_perm:
                        indices.extend([ti * 2, ti * 2 + 1])
                    if T % 2 == 1:
                        indices.append(T - 1)
                    clip = clip[:, indices, :, :]
                elif self.frame_shuffle_type == "matched_frame":
                    # Frame-level shuffle with FIXED perm + matched RoPE (Goodfire-style).
                    # Individual frames are shuffled (creating incoherent tubelets), but each
                    # tubelet gets the RoPE position of its first frame's original temporal position.
                    # This removes positional compensation, exposing full temporal reliance.
                    fixed_rng = np.random.RandomState(self.frame_shuffle_seed)
                    frame_perm = fixed_rng.permutation(T)
                    clip = clip[:, frame_perm, :, :]
                else:  # "frame"
                    perm = rng.permutation(T)
                    clip = clip[:, perm, :, :]

                shuffled.append([clip] if is_wrapped else clip)
            buffer = shuffled

        return buffer, label, clip_indices, sample_uri, self._lookup_phase_meta(sample_uri)

    def get_item_image(self, index):
        sample_uri = self.samples[index]
        dataset_idx, _ = self.per_dataset_indices[index]
        fpc = self.dataset_fpcs[dataset_idx]

        try:
            if isinstance(sample_uri, str) and sample_uri.startswith("s3://"):
                # S3 image
                self._ensure_s3_client()
                bucket_name, key = sample_uri.replace("s3://", "").split("/", 1)
                response = self.s3_client.get_object(Bucket=bucket_name, Key=key)
                image_bytes = response["Body"].read()
                image_tensor = torchvision.io.decode_image(
                    torch.from_numpy(np.frombuffer(image_bytes, np.uint8)),
                    mode=torchvision.io.ImageReadMode.RGB,
                )
            else:
                # Local image
                image_tensor = torchvision.io.read_image(path=sample_uri, mode=torchvision.io.ImageReadMode.RGB)
        except Exception as e:
            logger.warning(f"Failed to load image {sample_uri}: {e}")
            return None

        label = self.labels[index]
        clip_indices = [np.arange(start=0, stop=fpc, dtype=np.int32)]

        # Expand to [T, H, W, 3]
        buffer = image_tensor.unsqueeze(dim=0).repeat((fpc, 1, 1, 1))
        buffer = buffer.permute((0, 2, 3, 1))

        if self.shared_transform is not None:
            buffer = self.shared_transform(buffer)
        if self.transform is not None:
            buffer = [self.transform(buffer)]

        return buffer, label, clip_indices, sample_uri, self._lookup_phase_meta(sample_uri)

    def debug_sample_loading(self, index):
        sample_uri = self.samples[index]
        print(f"Attempting to load sample {index}: {sample_uri}")
        print(f"Sample type: {type(sample_uri)}")
        print(f"Is string: {isinstance(sample_uri, str)}")
        print(f"Starts with s3://: {sample_uri.startswith('s3://') if isinstance(sample_uri, str) else False}")

        if self.s3_client is None:
            print("S3 client not initialized")
            return

        try:
            bucket_name, key = sample_uri.replace("s3://", "").split("/", 1)
            print(f"Bucket: {bucket_name}, Key: {key}")
            response = self.s3_client.head_object(Bucket=bucket_name, Key=key)
            print(f"Object exists, size: {response['ContentLength']}")
        except Exception as e:
            print(f"S3 error: {e}")

    # ---------- Core video loader ----------
    def loadvideo_decord(self, sample_uri, fpc):
        """
        Unified loader:
          - Local path: matches the default filesystem logic exactly.
          - S3 path: mirrors the same semantics (size check, skip behavior, sampling math).
        Returns (buffer[T,H,W,3], clip_indices) or ([], None) on skip/failure.
        """
        # --- Local filesystem branch
        if not (isinstance(sample_uri, str) and sample_uri.startswith("s3://")):
            fname = sample_uri
            if not os.path.exists(fname):
                warnings.warn(f"video path not found fname='{fname}'")
                return [], None

            _fsize = os.path.getsize(fname)
            if self.filter_long_videos and _fsize > self.filter_long_videos:
                warnings.warn(f"skipping long video of size _fsize={_fsize} (bytes)")
                return [], None

            try:
                vr = VideoReader(fname, num_threads=-1, ctx=cpu(0))
            except Exception as e:
                logger.warning(f"decord VideoReader failed for {fname}: {e}")
                return [], None

            try:
                return self._sample_from_vr(vr, fpc)
            except Exception as e:
                logger.warning(f"decord decode failed for {fname}: {e}")
                return [], None

        # --- S3 branch
        try:
            bucket, key = sample_uri.replace("s3://", "").split("/", 1)
            self._ensure_s3_client()

            try:
                head = self.s3_client.head_object(Bucket=bucket, Key=key)
            except self.s3_client.exceptions.NoSuchKey:
                warnings.warn(f"video path not found fname='{sample_uri}'")
                return [], None
            except self.s3_client.exceptions.ClientError as e:
                # Could be NoSuchKey or perms; treat as skip like default
                logger.warning(f"S3 access error for {sample_uri}: {e}")
                return [], None

            fsize = head.get("ContentLength", 0)
            if self.filter_long_videos and fsize > self.filter_long_videos:
                warnings.warn(f"skipping long video of size _fsize={fsize} (bytes)")
                return [], None

            obj = self.s3_client.get_object(Bucket=bucket, Key=key)
            data = obj["Body"].read()
            if not data:
                logger.warning(f"Empty S3 object: {sample_uri}")
                return [], None

            bio = io.BytesIO(data)
            vr = VideoReader(bio, num_threads=-1, ctx=cpu(0))

        except Exception as e:
            logger.warning(f"Failed to load video: {sample_uri}\n{e}")
            return [], None

        try:
            return self._sample_from_vr(vr, fpc)
        except Exception as e:
            logger.warning(f"decord decode failed for {sample_uri}: {e}")
            return [], None

    # ---------- Sampling (shared by local & S3) ----------
    def _sample_from_vr(self, vr, fpc):
        fstp = self.frame_step
        try:
            n_frames = len(vr)
        except Exception as e:
            logger.warning(f"decord len(vr) failed: {e}")
            return [], None
        if n_frames <= 0:
            return [], None
        if self.duration is not None or self.fps is not None:
            try:
                video_fps = math.ceil(vr.get_avg_fps())
            except Exception as e:
                logger.warning(e)
                # keep parity with default (no fallback change)
            if self.duration is not None:
                assert self.fps is None
                fstp = int(self.duration * video_fps / fpc)
            else:
                assert self.duration is None
                fstp = video_fps // self.fps

        # Validate frame step, fps
        # if not hasattr(self, "_logged_mode"):
        #     mode = "fps" if self.fps is not None else ("duration" if self.duration is not None else "frame_step")
        #     logger.info(f"[Temporal mode={mode}] self.fps={self.fps} self.duration={self.duration} "
        #                 f"self.frame_step={self.frame_step} video_fps={video_fps if 'video_fps' in locals() else 'n/a'} "
        #                 f"fstp={fstp}")
        #     self._logged_mode = True

        assert fstp is not None and fstp > 0
        clip_len = int(fpc * fstp)

        if self.filter_short_videos and len(vr) < clip_len:
            warnings.warn(f"skipping video of length {len(vr)}")
            return [], None

        vr.seek(0)  # Go to start of video before sampling frames

        # ------------------------------------------------------------------
        # Anchor-aware sampling: single clip centered on self._current_anchor.
        # Bypass the partition-based sampler when an anchor is present.
        # Guarantees the anchor is in the returned indices (clamped to edges
        # if the anchor is too close to frame 0 or n_frames).
        # ------------------------------------------------------------------
        if getattr(self, "_current_anchor", None) is not None:
            anchor = int(self._current_anchor)
            # Ideal window: fpc frames spaced by fstp, anchor at the center.
            half = (fpc // 2) * fstp
            start = anchor - half
            # Clamp so the window fits in the video.
            max_start = max(0, len(vr) - clip_len)
            start = max(0, min(start, max_start))
            indices = (start + np.arange(fpc) * fstp).astype(np.int64)
            indices = np.clip(indices, 0, len(vr) - 1)
            # Anchor-coverage guard: if anchor is still outside the window
            # (pathological case where the video is shorter than clip_len),
            # place anchor as close to the center of the clamped window as
            # possible (already handled by clipping above; assert for safety).
            assert indices.min() >= 0 and indices.max() < len(vr), (
                f"Anchor-mode indices out of range: {indices.min()}..{indices.max()} "
                f"for video of length {len(vr)}"
            )
            try:
                buffer = vr.get_batch(indices.tolist()).asnumpy()
            except Exception as e:
                logger.warning(f"decord get_batch failed (anchor-mode, n_frames={n_frames}): {e}")
                return [], None
            return buffer, [indices]

        # Partition video into equal sized segments and sample each clip
        partition_len = len(vr) // self.num_clips

        all_indices, clip_indices = [], []
        for i in range(self.num_clips):
            if partition_len > clip_len:
                # sample a random window of clip_len frames within the segment
                end_indx = clip_len
                if self.random_clip_sampling:
                    end_indx = np.random.randint(clip_len, partition_len)
                start_indx = end_indx - clip_len
                indices = np.linspace(start_indx, end_indx, num=fpc)
                indices = np.clip(indices, start_indx, end_indx - 1).astype(np.int64)
                indices = indices + i * partition_len
            else:
                if not self.allow_clip_overlap:
                    base = partition_len // fstp
                    indices = np.linspace(0, partition_len, num=base)
                    if base < fpc:
                        indices = np.concatenate(
                            (indices, np.ones(fpc - base) * partition_len)
                        )
                    indices = np.clip(indices, 0, partition_len - 1).astype(np.int64)
                    indices = indices + i * partition_len
                else:
                    sample_len = min(clip_len, len(vr)) - 1
                    base = sample_len // fstp
                    indices = np.linspace(0, sample_len, num=base)
                    if base < fpc:
                        indices = np.concatenate(
                            (indices, np.ones(fpc - base) * sample_len)
                        )
                    indices = np.clip(indices, 0, sample_len - 1).astype(np.int64)
                    clip_step = 0
                    if len(vr) > clip_len and self.num_clips > 1:
                        clip_step = (len(vr) - clip_len) // (self.num_clips - 1)
                    indices = indices + i * clip_step

            clip_indices.append(indices)
            all_indices.extend(list(indices))

        try:
            buffer = vr.get_batch(all_indices).asnumpy()
        except Exception as e:
            logger.warning(f"decord get_batch failed (n_frames={n_frames}): {e}")
            return [], None
        return buffer, clip_indices

    def __len__(self):
        return len(self.samples)