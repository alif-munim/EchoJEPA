# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from logging import getLogger
from typing import Optional

_GLOBAL_SEED = 0
logger = getLogger()


def init_data(
    batch_size,
    transform=None,
    shared_transform=None,
    data="ImageNet",
    collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    root_path=None,
    image_folder=None,
    training=True,
    drop_last=True,
    subset_file=None,
    clip_len=None,
    dataset_fpcs=None,
    frame_sample_rate=None,
    duration=None,
    fps=None,
    num_clips=1,
    num_clips_per_video=1,  # NEW parameter
    random_clip_sampling=True,
    allow_clip_overlap=False,
    filter_short_videos=False,
    filter_long_videos=int(1e9),
    datasets_weights=None,
    persistent_workers=False,
    deterministic=True,
    log_dir=None,
    img_size=336,
    miss_augment_prob=0.0,  # <<< NEW
    min_present=1,  # <<< NEW
    split_name="train",
    study_sampling=False,
    class_balance_ratio=None,
    phase_metadata_csv=None,
    sampler_type=None,
    phase_matched_config=None,
):
    if data.lower() == "imagenet":
        from src.datasets.imagenet1k import make_imagenet1k

        dataset, data_loader, dist_sampler = make_imagenet1k(
            transform=transform,
            batch_size=batch_size,
            collator=collator,
            pin_mem=pin_mem,
            training=training,
            num_workers=num_workers,
            world_size=world_size,
            rank=rank,
            root_path=root_path,
            image_folder=image_folder,
            persistent_workers=persistent_workers,
            drop_last=drop_last,
            subset_file=subset_file,
        )

    elif data.lower() == "videodataset":
        from src.datasets.video_dataset import make_videodataset

        dataset, data_loader, dist_sampler = make_videodataset(
            data_paths=root_path,
            batch_size=batch_size,
            frames_per_clip=clip_len,
            dataset_fpcs=dataset_fpcs,
            frame_step=frame_sample_rate,
            duration=duration,
            fps=fps,
            num_clips=num_clips,
            random_clip_sampling=random_clip_sampling,
            allow_clip_overlap=allow_clip_overlap,
            filter_short_videos=filter_short_videos,
            filter_long_videos=filter_long_videos,
            shared_transform=shared_transform,
            transform=transform,
            datasets_weights=datasets_weights,
            collator=collator,
            num_workers=num_workers,
            pin_mem=pin_mem,
            persistent_workers=persistent_workers,
            world_size=world_size,
            rank=rank,
            deterministic=deterministic,
            log_dir=log_dir,
            drop_last=drop_last,
            study_sampling=study_sampling,
            class_balance_ratio=class_balance_ratio,
            phase_metadata_csv=phase_metadata_csv,
        )

    elif data.lower() == "videogroupdataset":
        from src.datasets.video_group_dataset import make_videogroupdataset

        dataset, data_loader, dist_sampler = make_videogroupdataset(
            data_paths=root_path,
            batch_size=batch_size,
            group_size=num_clips,  # num_segments from config
            frames_per_clip=clip_len,
            frame_step=frame_sample_rate,
            num_clips_per_video=num_clips_per_video,  # NEW
            random_clip_sampling=random_clip_sampling,
            allow_clip_overlap=allow_clip_overlap,
            shared_transform=shared_transform,
            transform=transform,
            collator=collator,
            num_workers=num_workers,
            pin_mem=pin_mem,
            persistent_workers=persistent_workers,
            world_size=world_size,
            rank=rank,
            deterministic=deterministic,
            log_dir=log_dir,
            drop_last=drop_last,
            img_size=img_size,
            training=training,
            miss_augment_prob=miss_augment_prob,
            min_present=min_present,
            split_name=split_name,
        )

    # sampler_type="phase_matched" post-processes the VideoGroupDataset
    # result to attach a PhaseMatchedStudySampler + PhaseMatchedEpochBuilder.
    # The training loop MUST call ``dist_sampler.builder.refresh_epoch(e)``
    # before ``iter(data_loader)`` each epoch.
    if sampler_type == "phase_matched":
        if data.lower() != "videogroupdataset":
            raise ValueError(f"sampler_type=phase_matched requires data=videogroupdataset; got {data}")
        cfg = dict(phase_matched_config or {})
        cfg.setdefault("parquet_path", None)
        if cfg["parquet_path"] is None:
            raise ValueError("phase_matched_config.parquet_path is required")
        # Late-import to avoid circular deps; the classifier/phase sampler
        # dir is optional at repo level.
        from pathlib import Path as _Path
        import sys as _sys

        sampler_dir = _Path(cfg.get("sampler_dir", "classifier/phase/sampler")).resolve()
        if str(sampler_dir) not in _sys.path:
            _sys.path.insert(0, str(sampler_dir))
        from phase_matched_sampler import PhaseMatchedStudySampler  # noqa: E402
        from phase_matched_pair_dataset import PhaseMatchedEpochBuilder  # noqa: E402

        # Load view labels from CSV if a path was provided. Columns are
        # configurable but default to the schema produced by our view
        # classifier pipeline: dicom_id,view,view_confidence (s3_uri optional).
        view_labels_map: Optional[dict] = cfg.get("view_labels")
        view_confidences_map: Optional[dict] = None
        view_path = cfg.get("view_labels_path")
        if view_path and not view_labels_map:
            import pandas as _pd

            view_label_column = cfg.get("view_label_column", "view")
            view_conf_column = cfg.get("view_confidence_column", "view_confidence")
            vdf = _pd.read_csv(view_path)
            # Derive dicom_id if the CSV only stores s3_uri.
            if "dicom_id" not in vdf.columns and "s3_uri" in vdf.columns:
                vdf["dicom_id"] = vdf.s3_uri.astype(str).str.extract(r"/([^/]+)\.(?:mp4|dcm)$", expand=False)
            vdf = vdf.dropna(subset=["dicom_id"]).copy()
            vdf["dicom_id"] = vdf["dicom_id"].astype(str)
            view_labels_map = dict(zip(vdf.dicom_id, vdf[view_label_column].astype(str)))
            if view_conf_column in vdf.columns:
                view_confidences_map = dict(zip(vdf.dicom_id, vdf[view_conf_column].astype(float)))
            logger.info(
                "data_manager: loaded %d view labels from %s (conf col %s present=%s)",
                len(view_labels_map),
                view_path,
                view_conf_column,
                view_confidences_map is not None,
            )

        pm_sampler = PhaseMatchedStudySampler(
            parquet_path=cfg["parquet_path"],
            tiers=tuple(cfg.get("quality_tiers", ("high",))),
            require_rr_consistent=bool(cfg.get("require_rr_consistent", True)),
            rr_filter_mode=cfg.get("rr_filter_mode", "strict"),
            sampling_mode=cfg.get("sampling_mode", "uniform_phase"),
            phase_tolerance=float(cfg.get("phase_tolerance", 0.15)),
            frames_per_clip=int(cfg.get("frames_per_clip", clip_len or 16)),
            frame_step=int(cfg.get("frame_step", 1)),
            pairs_per_study=int(cfg.get("pairs_per_study", 1)),
            same_session_only=bool(cfg.get("same_session_only", False)),
            seed=int(cfg.get("seed", _GLOBAL_SEED)),
            num_replicas=world_size,
            rank=rank,
            drop_last=drop_last,
            view_labels=view_labels_map,
            view_confidences=view_confidences_map,
            min_view_confidence=float(cfg.get("min_view_confidence", 0.0)),
            curriculum=cfg.get("curriculum"),
            total_epochs=int(cfg.get("total_epochs", 1)),
            view_pair_policy=cfg.get("view_pair_policy"),
            require_span_fits=bool(cfg.get("require_span_fits", False)),
            min_frames=cfg.get("min_frames"),
            # --- phase_relational triple-clip extensions (optional; all
            #     default to behavior that matches the existing smooth_l1
            #     pair path when unset). ---
            delta_phase_mode=cfg.get("delta_phase_mode", "same_phase"),
            delta_phase_buckets=cfg.get("delta_phase_buckets"),
            delta_phase_bucket_probs=cfg.get("delta_phase_bucket_probs"),
            require_same_study_wrong_phase_negative=bool(
                cfg.get("rel_require_same_study_wrong_phase_negative", False)
            ),
            wrong_phase_min_delta=float(cfg.get("rel_wrong_phase_min_delta", 0.25)),
            wrong_phase_strategy=cfg.get("rel_wrong_phase_strategy", "same_view_then_same_family"),
            allow_missing_hard_negative=bool(cfg.get("rel_allow_missing_hard_negative", False)),
            hard_negative_fallback=cfg.get("rel_hard_negative_fallback", "resample_anchor"),
            max_hard_neg_attempts=int(cfg.get("rel_max_hard_neg_attempts", 16)),
            # --- MV2SV extensions (Fix 1): target_clip + fused_clips
            #     sampling. Passed through when the training entrypoint
            #     sets mv2sv_config on the phase_matched_config block. ---
            mv2sv_config=cfg.get("mv2sv_config"),
        )
        # Guardrail: phase-matched pilot requires frame_step == 1 unless
        # the caller explicitly allows larger strides.
        if pm_sampler.frame_step > 1 and not bool(cfg.get("allow_frame_step_gt1", False)):
            raise ValueError(
                f"phase_matched frame_step={pm_sampler.frame_step} requires "
                f"allow_frame_step_gt1=true in phase_matched_config; source_span_frames="
                f"{pm_sampler.source_span_frames} may span >1 cardiac cycle."
            )
        # Wrap the existing dataset with a pair-builder. Access the inner
        # VideoGroupDataset (unwrap MonitoredDataset if present).
        inner = dataset.dataset if hasattr(dataset, "dataset") else dataset
        builder = PhaseMatchedEpochBuilder(
            pm_sampler,
            inner,
            debug_csv_path=(cfg["debug_csv_path"] if cfg.get("debug_csv_path") else None),
            video_uri_mode=str(cfg.get("video_uri_mode", "mp4")),
            raw_bucket_prefix=str(cfg.get("raw_bucket_prefix", "s3://echodata25/mimic-raw-staging")),
            mp4_bucket_prefix=str(cfg.get("mp4_bucket_prefix", "s3://echodata25/mimic-echo-224px")),
        )
        # Attach for the training-loop caller.
        pm_sampler.builder = builder
        pm_sampler.epoch_builder = builder  # alias

        # Rebuild the DataLoader with our phase-matched sampler installed.
        # The VideoGroupDataset default path bakes a standard
        # DistributedSampler into the DataLoader; leaving that in place
        # means DataLoader iterates the original 1-row placeholder dataset
        # via its own DistributedSampler length, hitting an assertion in
        # torch.utils.data.distributed.DistributedSampler.__iter__.
        import torch as _torch

        dl_kwargs = dict(
            dataset=inner,
            collate_fn=data_loader.collate_fn,
            sampler=pm_sampler,
            batch_size=data_loader.batch_size,
            drop_last=getattr(data_loader, "drop_last", False),
            pin_memory=data_loader.pin_memory,
            num_workers=data_loader.num_workers,
        )
        if data_loader.num_workers > 0:
            dl_kwargs["persistent_workers"] = getattr(data_loader, "persistent_workers", False)
            dl_kwargs["prefetch_factor"] = getattr(data_loader, "prefetch_factor", 2) or 2
        try:
            dl_kwargs["worker_init_fn"] = data_loader.worker_init_fn
        except AttributeError:
            pass
        data_loader = _torch.utils.data.DataLoader(**dl_kwargs)
        dist_sampler = pm_sampler
        logger.info(
            "data_manager: phase_matched sampler attached + DataLoader rebuilt; " "pair_rows/epoch=%d (rank=%d/%d)",
            pm_sampler.num_samples,
            rank,
            world_size,
        )

    return (data_loader, dist_sampler)
