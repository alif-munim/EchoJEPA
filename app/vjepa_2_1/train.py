# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_LOCALID"]
except Exception:
    pass

import copy
import gc
import glob
import io
import random
import time

import boto3
import numpy as np
import torch
import torch.multiprocessing as mp

try:
    mp.set_sharing_strategy("file_system")
except Exception:
    pass

import torch.distributed as dist
import torch.nn.functional as F
from app.vjepa_2_1.models.utils.masks_dist import compute_mask_distance
from app.vjepa_2_1.models.utils.modules import Lambda_LinearWarmupHold
from app.vjepa_2_1.transforms import make_transforms
from app.vjepa_2_1.utils import (
    init_opt,
    init_video_model,
    load_checkpoint,
    normalize_nested,
)
from src.datasets.data_manager import init_data
from src.masks.multiseq_multiblock3d import MaskCollator
from src.masks.utils import apply_masks
from src.utils.distributed import init_distributed
from src.utils.logging import AverageMeter, CSVLogger, get_logger, gpu_timer
from torch.nn.parallel import DistributedDataParallel


def _barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


log_timings = True
log_freq = 10
CHECKPOINT_FREQ = 1
GARBAGE_COLLECT_ITR_FREQ = 50
MAX_REPEAT_COUNTS = 10

_GLOBAL_SEED = 0
random.seed(_GLOBAL_SEED)
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True


logger = get_logger(__name__, force=True)


def prune_local_checkpoints(folder, max_to_keep=4):
    try:
        all_checkpoints = [f for f in os.listdir(folder) if f.startswith('e') and f.endswith('.pt')]
        if len(all_checkpoints) > max_to_keep:
            all_checkpoints.sort(key=lambda x: int(x[1:-3]))
            checkpoints_to_delete = all_checkpoints[:-max_to_keep]
            logger.info(
                f"Pruning local checkpoints. Keeping {max_to_keep}, deleting {len(checkpoints_to_delete)}."
            )
            for ckpt_name in checkpoints_to_delete:
                full_path = os.path.join(folder, ckpt_name)
                try:
                    os.remove(full_path)
                except Exception as e:
                    logger.error(f"Failed to delete old checkpoint {full_path}: {e}")
    except Exception as e:
        logger.error(f"Failed to prune checkpoints in folder {folder}: {e}")


def main(args, resume_preempt=False):
    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- META
    folder = args.get("folder")
    cfgs_meta = args.get("meta")
    s3_checkpoint_uri = cfgs_meta.get("s3_checkpoint_uri", None)
    save_every_freq = cfgs_meta.get("save_every_freq", -1)
    checkpoints_to_keep = cfgs_meta.get("checkpoints_to_keep", 3)
    max_epoch_checkpoints = cfgs_meta.get("max_epoch_checkpoints", checkpoints_to_keep)
    load_model = cfgs_meta.get("load_checkpoint") or resume_preempt
    r_file = cfgs_meta.get("read_checkpoint", None)
    seed = cfgs_meta.get("seed", _GLOBAL_SEED)
    skip_batches = cfgs_meta.get("skip_batches", -1)
    use_sdpa = cfgs_meta.get("use_sdpa", False)
    sync_gc = cfgs_meta.get("sync_gc", False)
    logger.info(f"LD_PRELOAD: {os.environ.get('LD_PRELOAD')}")
    which_dtype = cfgs_meta.get("dtype")
    logger.info(f"{which_dtype=}")
    if which_dtype.lower() == "bfloat16":
        dtype = torch.bfloat16
        mixed_precision = True
    elif which_dtype.lower() == "float16":
        dtype = torch.float16
        mixed_precision = True
    else:
        dtype = torch.float32
        mixed_precision = False

    # -- MASK
    cfgs_mask = args.get("mask")

    # -- MODEL
    cfgs_model = args.get("model")
    compile_model = cfgs_model.get("compile_model", False)
    use_activation_checkpointing = cfgs_model.get("use_activation_checkpointing", False)
    model_name = cfgs_model.get("model_name")
    pred_depth = cfgs_model.get("pred_depth")
    pred_num_heads = cfgs_model.get("pred_num_heads", None)
    pred_embed_dim = cfgs_model.get("pred_embed_dim")
    uniform_power = cfgs_model.get("uniform_power", False)
    use_mask_tokens = cfgs_model.get("use_mask_tokens", False)
    zero_init_mask_tokens = cfgs_model.get("zero_init_mask_tokens", True)
    use_rope = cfgs_model.get("use_rope", False)
    use_silu = cfgs_model.get("use_silu", False)
    use_pred_silu = cfgs_model.get("use_pred_silu", False)
    wide_silu = cfgs_model.get("wide_silu", True)
    is_causal = cfgs_model.get("is_causal", False)
    pred_is_causal = cfgs_model.get("pred_is_causal", False)
    init_type = cfgs_model.get("init_type", "default")
    img_temporal_dim_size = cfgs_model.get("img_temporal_dim_size", None)
    n_registers = cfgs_model.get("n_registers", 0)
    has_cls_first = cfgs_model.get("has_cls_first", False)
    interpolate_rope = cfgs_model.get("interpolate_rope", False)
    lambda_value_img = cfgs_model.get("lambda_value_img", 0.0)
    lambda_value_vid = cfgs_model.get("lambda_value_vid", 0.0)
    n_registers_predictor = cfgs_model.get("n_registers_predictor", 0)
    lambda_progressive = cfgs_model.get("lambda_progressive", True)
    normalize_predictor = cfgs_model.get("normalize_predictor", False)
    modality_embedding = cfgs_model.get("modality_embedding", False)
    levels_predictor = cfgs_model.get("levels_predictor", 4)
    n_output_distillation = cfgs_model.get("n_output_distillation", None)
    _embed_dims = {
        "vit_tiny": 192, "vit_small": 384, "vit_base": 768,
        "vit_large": 1024, "vit_huge": 1280,
        "vit_giant": 1408, "vit_giant_xformers": 1408,
        "vit_gigantic": 1664, "vit_gigantic_xformers": 1664,
    }
    if model_name in _embed_dims:
        embed_dim_encoder = _embed_dims[model_name]
    else:
        raise ValueError(f"Unknown model_name '{model_name}'. Expected one of: {list(_embed_dims.keys())}")

    # -- DATA
    cfgs_data = args.get("data")
    dataset_type = cfgs_data.get("dataset_type", "videodataset")
    dataset_paths = cfgs_data.get("datasets", [])
    datasets_weights = cfgs_data.get("datasets_weights")
    dataset_fpcs = cfgs_data.get("dataset_fpcs")
    max_num_frames = max(dataset_fpcs)
    batch_size = cfgs_data.get("batch_size")
    tubelet_size = cfgs_data.get("tubelet_size")
    fps = cfgs_data.get("fps")
    crop_size = cfgs_data.get("crop_size", 224)
    patch_size = cfgs_data.get("patch_size")
    grid_size = crop_size // patch_size
    pin_mem = cfgs_data.get("pin_mem", False)
    num_workers = cfgs_data.get("num_workers", 1)
    persistent_workers = cfgs_data.get("persistent_workers", True)

    # -- IMG DATA
    cfgs_img_data = args.get("img_data")
    img_rank_ratio = 0.25
    img_mask = None
    if cfgs_img_data is not None:
        img_dataset_type = cfgs_img_data.get("dataset_type", "imagenet")
        img_dataset_paths = cfgs_img_data.get("datasets", [])
        img_dataset_weights = cfgs_img_data.get("datasets_weights", [])
        img_dataset_fpcs = cfgs_img_data.get("dataset_fpcs")
        img_dataset_batch_size = cfgs_img_data.get("batch_size")
        img_rank_ratio = cfgs_img_data.get("rank_ratio", img_rank_ratio)
        img_num_workers = cfgs_img_data.get("num_workers", num_workers)

        img_mask = args.get("img_mask", img_mask)

    # -- DATA AUGS
    cfgs_data_aug = args.get("data_aug")
    ar_range = cfgs_data_aug.get("random_resize_aspect_ratio", [3 / 4, 4 / 3])
    rr_scale = cfgs_data_aug.get("random_resize_scale", [0.3, 1.0])
    motion_shift = cfgs_data_aug.get("motion_shift", False)
    reprob = cfgs_data_aug.get("reprob", 0.0)
    use_aa = cfgs_data_aug.get("auto_augment", False)

    # -- LOSS
    cfgs_loss = args.get("loss")
    loss_exp = cfgs_loss.get("loss_exp")
    shift_by_n = cfgs_loss.get("shift_by_n")
    predict_all = cfgs_loss.get("predict_all", True)
    weight_distance_loss = cfgs_loss.get("weight_distance_loss", False)
    offset_context_loss = cfgs_loss.get("offset_context_loss", False)

    # -- OPTIMIZATION
    cfgs_opt = args.get("optimization")
    is_anneal = cfgs_opt.get("is_anneal", False)
    force_load_pretrain = cfgs_opt.get("force_load_pretrain", False)
    anneal_ckpt = cfgs_opt.get("anneal_ckpt", None)
    if is_anneal and anneal_ckpt is None and not force_load_pretrain:
        raise ValueError("Must specify anneal_ckpt if is_anneal is True")
    resume_anneal = cfgs_opt.get("resume_anneal", False) or (
        is_anneal and resume_preempt
    )
    ipe = cfgs_opt.get("ipe", None)
    ipe_scale = cfgs_opt.get("ipe_scale", 1.0)
    wd = float(cfgs_opt.get("weight_decay"))
    final_wd = float(cfgs_opt.get("final_weight_decay"))
    num_epochs = cfgs_opt.get("epochs")
    warmup = cfgs_opt.get("warmup")
    start_lr = cfgs_opt.get("start_lr")
    lr = cfgs_opt.get("lr")
    final_lr = cfgs_opt.get("final_lr")
    ema = cfgs_opt.get("ema")
    use_radamw = cfgs_opt.get("use_radamw", False)
    betas = cfgs_opt.get("betas", (0.9, 0.999))
    eps = cfgs_opt.get("eps", 1.0e-8)
    loss_reg_std_mult = cfgs_opt.get("loss_reg_std_mult", None)
    loss_reg_num_tracking_steps = cfgs_opt.get("loss_reg_num_tracking_steps", 300)
    loss_reg_min_epoch = cfgs_opt.get("loss_reg_min_epoch", 50)
    if loss_reg_std_mult is not None:
        logger.info("Loss regulation activated")
    # ----------------------------------------------------------------------- #

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    try:
        mp.set_start_method("spawn")
    except Exception:
        pass

    # -- init torch distributed backend
    world_size, rank = init_distributed()
    data_world_size, data_rank = world_size, rank
    logger.info(f"Initialized (rank/world-size) {rank}/{world_size}")
    img_world_size = 0

    if rank == 0 and s3_checkpoint_uri:
        logger.info(f"Checkpoints will be uploaded to: {s3_checkpoint_uri}")

    # make adjustments to batch size for image data
    model_fpcs = dataset_fpcs
    model_cfgs_mask = cfgs_mask
    model_tubelet_size = tubelet_size
    if cfgs_img_data is not None:
        img_world_size = int(world_size * img_rank_ratio)
        num_video_ranks = world_size - img_world_size
        img_total_batch_size = img_dataset_batch_size * world_size
        video_total_batch_size = batch_size * world_size

        if img_total_batch_size % img_world_size != 0:
            raise ValueError(
                f"img_total_batch_size ({img_total_batch_size}) must be divisible by num_img_ranks ({img_world_size})"
            )
        if video_total_batch_size % num_video_ranks != 0:
            raise ValueError(
                f"video_total_batch_size ({video_total_batch_size}) must be divisible by num_video_ranks ({num_video_ranks})"
            )

        batch_size = video_total_batch_size // num_video_ranks

        if rank < int(world_size * img_rank_ratio):
            crop_size = cfgs_img_data.get("crop_size", 512)
            grid_size = crop_size // patch_size

        if rank < int(world_size * img_rank_ratio):
            logger.info(
                f"On rank {rank}, updating dataset with dataset type {img_dataset_type}"
            )
            if img_temporal_dim_size is not None:
                if img_dataset_fpcs[0] != 1:
                    raise NotImplementedError(
                        "Image loader only supports 1 frame per clip with img_temporal_dim_size=1"
                    )
                tubelet_size = 1
            else:
                tubelet_size = tubelet_size

            dataset_type = img_dataset_type
            dataset_paths = img_dataset_paths
            datasets_weights = img_dataset_weights
            dataset_fpcs = img_dataset_fpcs
            batch_size = img_dataset_batch_size
            num_workers = img_num_workers
            if img_mask is not None:
                logger.info("Using image mask")
                cfgs_mask = img_mask

            data_rank = rank
            data_world_size = img_world_size
            lambda_value = lambda_value_img
        else:
            data_rank = rank - img_world_size
            data_world_size = world_size - img_world_size
            lambda_value = lambda_value_vid

        logger.info(
            f"For rank {rank} with world size {world_size}, "
            f"we have total image batch size {img_total_batch_size}, total video batch size {video_total_batch_size}, "
            f"image ranks: {img_world_size}, video ranks: {num_video_ranks}, "
            f"using the following params: "
            f"dataset_type: {dataset_type}, "
            f"dataset_paths: {dataset_paths}, "
            f"datasets_weights: {datasets_weights}, "
            f"dataset_fpcs: {dataset_fpcs}, "
            f"batch_size: {batch_size}, "
            f"num_workers: {num_workers}, "
            f"data_rank: {data_rank}, "
            f"data_world_size: {data_world_size}"
            f"lambda_value for the context loss: {lambda_value}"
        )
    else:
        lambda_value = lambda_value_vid

    # -- set device
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

    # -- log/checkpointing paths
    log_file = os.path.join(folder, f"log_r{rank}.csv")
    latest_path = os.path.join(folder, "latest.pt")

    load_path = None
    if load_model:
        if is_anneal:
            if os.path.exists(latest_path) and resume_anneal:
                load_path = latest_path
            else:
                load_path = anneal_ckpt
                resume_anneal = False
        else:
            load_path = os.path.join(folder, r_file) if r_file is not None else latest_path
        if not os.path.exists(load_path):
            load_path = None
            load_model = False

    # -- make csv_logger
    csv_logger = CSVLogger(
        log_file,
        ("%d", "epoch"),
        ("%d", "itr"),
        ("%.5f", "loss"),
        ("%.5f", "loss_pred"),
        ("%.5f", "loss_context"),
        ("%d", "iter-time(ms)"),
        ("%d", "gpu-time(ms)"),
        ("%d", "dataload-time(ms)"),
    )

    # -- init model
    encoder, predictor = init_video_model(
        uniform_power=uniform_power,
        use_mask_tokens=use_mask_tokens,
        num_mask_tokens=int(len(model_cfgs_mask) * len(model_fpcs)),
        zero_init_mask_tokens=zero_init_mask_tokens,
        device=device,
        patch_size=patch_size,
        max_num_frames=max_num_frames,
        tubelet_size=model_tubelet_size,
        model_name=model_name,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_num_heads=pred_num_heads,
        pred_embed_dim=pred_embed_dim,
        is_causal=is_causal,
        pred_is_causal=pred_is_causal,
        use_sdpa=use_sdpa,
        use_silu=use_silu,
        use_pred_silu=use_pred_silu,
        wide_silu=wide_silu,
        use_rope=use_rope,
        use_activation_checkpointing=use_activation_checkpointing,
        return_all_tokens=predict_all,
        chop_last_n_tokens=shift_by_n,
        init_type=init_type,
        img_temporal_dim_size=img_temporal_dim_size,
        n_registers=n_registers,
        n_registers_predictor=n_registers_predictor,
        has_cls_first=has_cls_first,
        interpolate_rope=interpolate_rope,
        modality_embedding=modality_embedding,
        n_output_distillation=n_output_distillation,
    )
    target_encoder = copy.deepcopy(encoder)

    if compile_model:
        logger.info("Compiling encoder, target_encoder, and predictor.")
        torch._dynamo.config.optimize_ddp = False
        encoder.compile()
        target_encoder.compile()
        predictor.compile()

    mask_collator = MaskCollator(
        cfgs_mask=cfgs_mask,
        dataset_fpcs=dataset_fpcs,
        crop_size=crop_size,
        patch_size=patch_size,
        tubelet_size=tubelet_size,
    )

    transform = make_transforms(
        random_horizontal_flip=True,
        random_resize_aspect_ratio=ar_range,
        random_resize_scale=rr_scale,
        reprob=reprob,
        auto_augment=use_aa,
        motion_shift=motion_shift,
        crop_size=crop_size,
    )

    # -- init data-loaders/samplers
    (unsupervised_loader, unsupervised_sampler) = init_data(
        data=dataset_type,
        root_path=dataset_paths,
        batch_size=batch_size,
        training=True,
        dataset_fpcs=dataset_fpcs,
        fps=fps,
        transform=transform,
        rank=data_rank,
        world_size=data_world_size,
        datasets_weights=datasets_weights,
        persistent_workers=persistent_workers,
        collator=mask_collator,
        num_workers=num_workers,
        pin_mem=pin_mem,
        log_dir=None,
    )
    try:
        _dlen = len(unsupervised_loader)
    except Exception:
        try:
            _dlen = unsupervised_loader.num_batches
        except Exception:
            _dlen = -1
    if ipe is None:
        ipe = _dlen
    logger.info(f"Using batch size of {batch_size}, fpcs of {dataset_fpcs}")
    logger.info(f"iterations per epoch/dataset length: {ipe}/{_dlen}")

    # -- init optimizer and scheduler
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        is_anneal=is_anneal,
        encoder=encoder,
        predictor=predictor,
        use_radamw=use_radamw,
        wd=wd,
        final_wd=final_wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        iterations_per_epoch=ipe,
        warmup=warmup,
        num_epochs=num_epochs,
        ipe_scale=ipe_scale,
        mixed_precision=mixed_precision,
        betas=betas,
        eps=eps,
    )
    encoder = DistributedDataParallel(encoder, static_graph=True)
    predictor = DistributedDataParallel(
        predictor, static_graph=False, find_unused_parameters=True
    )
    target_encoder = DistributedDataParallel(target_encoder)
    for p in target_encoder.parameters():
        p.requires_grad = False

    # -- momentum schedule
    momentum_scheduler = (
        ema[0] + i * (ema[1] - ema[0]) / (ipe * num_epochs * ipe_scale)
        for i in range(int(ipe * num_epochs) + 1)
    )
    lambda_sched = Lambda_LinearWarmupHold(lambda_value=lambda_value)

    start_epoch = 0
    start_itr = 0

    if force_load_pretrain:
        anneal_ckpt_path = anneal_ckpt
        if anneal_ckpt_path and os.path.exists(anneal_ckpt_path):
            logger.info(f"FORCE-LOADING pretrained model from {anneal_ckpt_path}")
            from src.utils.checkpoint_loader import robust_checkpoint_loader
            checkpoint = robust_checkpoint_loader(anneal_ckpt_path, map_location=torch.device("cpu"))

            def _strip_prefix(state_dict):
                import re
                return {re.sub(r'^module\.(?:backbone\.)?', '', k): v for k, v in state_dict.items()}

            def _load_with_shape_filter(module, pretrained_dict, name="module"):
                """Load state dict, skipping keys with shape mismatches (e.g., distilled predictor)."""
                model_dict = module.state_dict()
                skipped = []
                for k, v in list(pretrained_dict.items()):
                    if k in model_dict and v.shape != model_dict[k].shape:
                        skipped.append(f"{k}: ckpt {list(v.shape)} vs model {list(model_dict[k].shape)}")
                        del pretrained_dict[k]
                if skipped:
                    logger.warning(f"Skipped {len(skipped)} shape-mismatched keys in {name}:\n  " + "\n  ".join(skipped))
                msg = module.load_state_dict(pretrained_dict, strict=False)
                return msg

            if "encoder" in checkpoint:
                pretrained_dict = _strip_prefix(checkpoint["encoder"])
                msg = _load_with_shape_filter(encoder.module.backbone, pretrained_dict, "encoder")
                logger.info(f"Loaded pretrained encoder with msg: {msg}")

            target_key = "target_encoder" if "target_encoder" in checkpoint else "ema_encoder"
            if target_key in checkpoint:
                pretrained_dict = _strip_prefix(checkpoint[target_key])
                msg = _load_with_shape_filter(target_encoder.module.backbone, pretrained_dict, f"target_encoder ({target_key})")
                logger.info(f"Loaded pretrained target encoder ({target_key}) with msg: {msg}")

            if "predictor" in checkpoint:
                pretrained_dict = _strip_prefix(checkpoint["predictor"])
                msg = _load_with_shape_filter(predictor.module.backbone, pretrained_dict, "predictor")
                logger.info(f"Loaded pretrained predictor with msg: {msg}")

            del checkpoint
            gc.collect()
            logger.info("Force-loading complete. Starting fresh from epoch 0.")
        else:
            raise FileNotFoundError(f"force_load_pretrain is set but checkpoint not found at: {anneal_ckpt_path}")
    else:
        # -- load training checkpoint (resume or anneal)
        logger.info(f"Loading checkpoint from: {load_path}")
        if load_model or os.path.exists(latest_path):
            (
                encoder,
                predictor,
                target_encoder,
                optimizer,
                scaler,
                start_epoch,
            ) = load_checkpoint(
                r_path=load_path,
                encoder=encoder,
                predictor=predictor,
                target_encoder=target_encoder,
                opt=optimizer,
                scaler=scaler,
                is_anneal=is_anneal and not resume_anneal,
            )
            if not is_anneal or resume_anneal:
                for _ in range(start_epoch * ipe):
                    scheduler.step()
                    wd_scheduler.step()
                    next(momentum_scheduler)
                    mask_collator.step()

    def save_checkpoint(epoch, itr, local_path, s3_uri_base=None, is_periodic=False):
        if rank != 0:
            return

        save_dict = {
            "encoder": encoder.state_dict(),
            "predictor": predictor.state_dict(),
            "opt": optimizer.state_dict(),
            "scaler": None if scaler is None else scaler.state_dict(),
            "target_encoder": target_encoder.state_dict(),
            "epoch": epoch,
            "itr": itr,
            "loss": loss_meter.avg,
            "batch_size": batch_size,
            "world_size": world_size,
            "lr": lr,
        }

        try:
            torch.save(save_dict, local_path)
        except Exception as e:
            logger.error(f"Encountered exception when saving local checkpoint: {e}")
            return

        if s3_uri_base:
            try:
                s3_client = boto3.client("s3")
                bucket, key_prefix = s3_uri_base.replace("s3://", "").split("/", 1)
                filename = os.path.basename(local_path)
                s3_key = os.path.join(key_prefix, filename)

                file_size = os.path.getsize(local_path)
                logger.info(f"Checkpoint size: {file_size / (1024**3):.2f} GB")

                if file_size > 5 * 1024**3:
                    logger.info("Using multipart upload for large checkpoint...")
                    s3_client.upload_file(local_path, bucket, s3_key)
                else:
                    with open(local_path, 'rb') as f:
                        s3_client.put_object(Bucket=bucket, Key=s3_key, Body=f.read())
                logger.info(f"Successfully uploaded checkpoint to s3://{bucket}/{s3_key}")
            except Exception as e:
                logger.error(f"Failed to upload checkpoint to S3. Error: {e}")

        if is_periodic and max_epoch_checkpoints > 0:
            prune_local_checkpoints(os.path.dirname(local_path), max_to_keep=max_epoch_checkpoints)

    logger.info("Initializing loader...")
    unsupervised_sampler.set_epoch(start_epoch)
    loader = iter(unsupervised_loader)

    if skip_batches > 0:
        logger.info(f"Skip {skip_batches} batches")

        for itr in range(skip_batches):
            if itr % 10 == 0:
                logger.info(f"Skip {itr}/{skip_batches} batches")
            try:
                _ = next(loader)
            except Exception:
                loader = iter(unsupervised_loader)
                _ = next(loader)

    if sync_gc:
        gc.disable()
        gc.collect()

    trailing_losses = []
    step_count = 0

    # -- TRAINING LOOP
    try:
        for epoch in range(start_epoch, num_epochs):
            unsupervised_sampler.set_epoch(epoch)
            logger.info("Epoch %d" % (epoch + 1))

            loss_meter = AverageMeter()
            mask_meters = {fpc: AverageMeter() for fpc in dataset_fpcs}
            iter_time_meter = AverageMeter()
            gpu_time_meter = AverageMeter()
            data_elapsed_time_meter = AverageMeter()

            itr_start = start_itr if epoch == start_epoch else 0

            for itr in range(itr_start, ipe):
                itr_start_time = time.time()

                iter_retries = 0
                iter_successful = False
                while not iter_successful:
                    try:
                        sample = next(loader)
                        iter_successful = True
                    except StopIteration:
                        logger.info("Exhausted data loaders. Refreshing...")
                        if "airstore" in dataset_type.lower():
                            unsupervised_sampler.increase_epoch()
                        else:
                            unsupervised_sampler.set_epoch(epoch)
                        loader = iter(unsupervised_loader)
                    except Exception as e:
                        NUM_RETRIES = 5
                        if iter_retries < NUM_RETRIES:
                            logger.warning(
                                f"Encountered exception when loading data (num retries {iter_retries}):\n{e}"
                            )
                            iter_retries += 1
                            time.sleep(5)
                            loader = iter(unsupervised_loader)
                        else:
                            logger.warning("Exceeded max retries; rebuilding DataLoader to respawn workers.")
                            (unsupervised_loader, unsupervised_sampler) = init_data(
                                data=dataset_type,
                                root_path=dataset_paths,
                                batch_size=batch_size,
                                training=True,
                                dataset_fpcs=dataset_fpcs,
                                fps=fps,
                                transform=transform,
                                rank=data_rank,
                                world_size=data_world_size,
                                datasets_weights=datasets_weights,
                                persistent_workers=persistent_workers,
                                collator=mask_collator,
                                num_workers=num_workers,
                                pin_mem=pin_mem,
                                log_dir=None,
                            )
                            unsupervised_sampler.set_epoch(epoch)
                            loader = iter(unsupervised_loader)
                            iter_retries = 0
                            continue

                for _fpc_sample in sample:
                    try:
                        udata = _fpc_sample[0]
                        # Find the buffer tensor (first tensor element in collated batch)
                        buf = udata[0] if torch.is_tensor(udata[0]) else udata[0][0]
                        bs = buf.shape[0]
                        fpc = buf.shape[2]  # temporal frames
                        mask_meters[fpc].update(bs / batch_size)
                    except (IndexError, AttributeError, TypeError):
                        pass  # logging only — skip if structure is unexpected

                def load_clips():
                    all_clips, all_masks_enc, all_masks_pred = [], [], []
                    for fpc_sample in sample:
                        udata, masks_enc, masks_pred = fpc_sample
                        all_clips += [udata[0][0].to(device, non_blocking=True)]
                        all_masks_enc += [
                            [m.to(device, non_blocking=True) for m in masks_enc]
                        ]
                        all_masks_pred += [
                            [m.to(device, non_blocking=True) for m in masks_pred]
                        ]
                    return all_clips, all_masks_enc, all_masks_pred

                clips, masks_enc, masks_pred = load_clips()
                data_elapsed_time_ms = (time.time() - itr_start_time) * 1000.0

                if sync_gc and (itr + 1) % GARBAGE_COLLECT_ITR_FREQ == 0:
                    logger.info("Running garbage collection...")
                    gc.collect()

                def train_step():
                    _new_lr = scheduler.step()
                    _new_wd = wd_scheduler.step()

                    def forward_target(c, embed_dim=embed_dim_encoder):
                        with torch.no_grad():
                            h = target_encoder(c, gram_mode=False, training_mode=True)
                            new_h = []
                            for hi in h:
                                if levels_predictor > 1:
                                    hi_0 = F.layer_norm(hi[:, :, :embed_dim], (embed_dim,))
                                    hi_1 = F.layer_norm(
                                        hi[:, :, embed_dim : embed_dim * 2],
                                        (embed_dim,),
                                    )
                                    hi_2 = F.layer_norm(
                                        hi[:, :, embed_dim * 2 : embed_dim * 3],
                                        (embed_dim,),
                                    )
                                    hi_3 = F.layer_norm(hi[:, :, -embed_dim:], (embed_dim,))
                                    hi_norm = torch.cat([hi_0, hi_1, hi_2, hi_3], dim=2)
                                    new_h.append(hi_norm)
                                else:
                                    new_h.append(F.layer_norm(hi, (hi.size(-1),)))
                            return new_h

                    def forward_context(clips, embed_dim=embed_dim_encoder):
                        modality = "video"
                        if img_temporal_dim_size is not None:
                            if clips[0].shape[2] == img_temporal_dim_size:
                                modality = "image"
                        z = encoder(clips, masks_enc, gram_mode=False, training_mode=True)
                        z_pred, z_context = predictor(
                            z, masks_enc, masks_pred, mod=modality
                        )
                        if normalize_predictor:
                            z_pred = normalize_nested(z_pred, embed_dim)

                            if predict_all:
                                z_context = normalize_nested(z_context, embed_dim)
                        return z_pred, z_context

                    def loss_fn(z, h, masks_to_apply, cls_loss, d_weights):
                        if cls_loss:
                            h_cls = [hi[:, 0].unsqueeze(1) for hi in h]
                            h = [
                                apply_masks(hi[:, 1:], mi, concat=False)
                                for hi, mi in zip(h, masks_to_apply)
                            ]
                            loss, n = 0, 0
                            for zi, hi, hi_cls in zip(z, h, h_cls):
                                for zij, hij in zip(zi, hi):
                                    h_term = torch.cat([hi_cls, hij], dim=1)
                                    loss += (
                                        torch.mean(torch.abs(zij - h_term) ** loss_exp)
                                        / loss_exp
                                    )
                                    n += 1

                            loss /= n
                            return loss
                        else:
                            h = [
                                apply_masks(hi, mi, concat=False)
                                for hi, mi in zip(h, masks_to_apply)
                            ]

                            if d_weights is not None:
                                loss, n = 0, 0
                                for zi, hi, d_i in zip(z, h, d_weights):
                                    for zij, hij, d_ij in zip(zi, hi, d_i):
                                        loss_n = torch.abs(zij - hij) ** loss_exp * (
                                            1 / d_ij.unsqueeze(2)
                                        )
                                        loss += torch.mean(loss_n) / loss_exp
                                        n += 1
                                loss /= n
                                return loss
                            else:
                                loss, n = 0, 0
                                for zi, hi in zip(z, h):
                                    for zij, hij in zip(zi, hi):
                                        loss += (
                                            torch.mean(torch.abs(zij - hij) ** loss_exp)
                                            / loss_exp
                                        )
                                        n += 1
                                loss /= n
                                return loss

                    # Step 1. Forward
                    with torch.cuda.amp.autocast(dtype=dtype, enabled=mixed_precision):
                        h = forward_target(clips)
                        z_pred, z_context = forward_context(clips)
                        loss = 0
                        loss_pred = loss_fn(
                            z_pred, h, masks_pred, cls_loss=has_cls_first, d_weights=None
                        )
                        loss += loss_pred

                        # Context loss
                        loss_context_val = 0.0
                        if predict_all:
                            d_weights = None
                            if weight_distance_loss:
                                distance_weights = compute_mask_distance(
                                    masks_pred, masks_enc, grid_size, offset_context_loss
                                )
                                d_weights = distance_weights
                            loss_context = loss_fn(
                                z_context, h, masks_enc, cls_loss=False, d_weights=d_weights
                            )
                            loss_context_val = float(loss_context)
                            if lambda_progressive:
                                lambda_value_step = lambda_sched.value(epoch * ipe + itr)
                            else:
                                lambda_value_step = lambda_value
                            loss += loss_context * lambda_value_step

                    # Step 2. Backward & step
                    run_step = True
                    if loss_reg_std_mult is not None:
                        meanval = np.mean(trailing_losses)
                        stdval = np.std(trailing_losses)
                        max_bound = meanval + loss_reg_std_mult * stdval
                        if (
                            loss > max_bound
                            and epoch > loss_reg_min_epoch
                            and len(trailing_losses)
                            > int(0.5 * loss_reg_num_tracking_steps)
                        ):
                            run_step = False
                            loss.backward()
                            logger.info(
                                f"Loss {loss} is above bound {meanval} + {loss_reg_std_mult} * {stdval}. Skipping step."
                            )

                    if run_step:
                        if mixed_precision:
                            scaler.scale(loss).backward()
                            scaler.unscale_(optimizer)
                        else:
                            loss.backward()
                        if mixed_precision:
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                    optimizer.zero_grad()

                    # Step 3. momentum update of target encoder
                    m = min(next(momentum_scheduler), ema[1])
                    with torch.no_grad():
                        params_k = []
                        params_q = []
                        for param_q, param_k in zip(
                            encoder.parameters(), target_encoder.parameters()
                        ):
                            params_k.append(param_k)
                            params_q.append(param_q)
                        torch._foreach_mul_(params_k, m)
                        torch._foreach_add_(params_k, params_q, alpha=1 - m)

                    return (
                        float(loss),
                        float(loss_pred),
                        loss_context_val,
                        _new_lr,
                        _new_wd,
                        run_step,
                    )

                (
                    loss,
                    loss_pred_val,
                    loss_context_val,
                    _new_lr,
                    _new_wd,
                    run_step,
                ), gpu_etime_ms = gpu_timer(train_step)
                iter_elapsed_time_ms = (time.time() - itr_start_time) * 1000.0
                loss_meter.update(loss)
                iter_time_meter.update(iter_elapsed_time_ms)
                gpu_time_meter.update(gpu_etime_ms)
                data_elapsed_time_meter.update(data_elapsed_time_ms)

                if loss_reg_std_mult is not None:
                    if run_step:
                        trailing_losses.append(loss)
                        if len(trailing_losses) > loss_reg_num_tracking_steps:
                            trailing_losses = trailing_losses[1:]
                    else:
                        step_count += 1
                        if step_count > MAX_REPEAT_COUNTS:
                            raise RuntimeError(
                                "Loss is above bound for too many tries. Exiting."
                            )

                # -- Logging
                def log_stats():
                    csv_logger.log(
                        epoch + 1,
                        itr,
                        loss,
                        loss_pred_val,
                        loss_context_val,
                        iter_elapsed_time_ms,
                        gpu_etime_ms,
                        data_elapsed_time_ms,
                    )
                    if (
                        (itr % log_freq == 0)
                        or (itr == ipe - 1)
                        or np.isnan(loss)
                        or np.isinf(loss)
                    ):
                        logger.info(
                            "[%d, %5d] loss: %.3f (pred: %.3f ctx: %.3f) "
                            "masks: %s "
                            "[wd: %.2e] [lr: %.2e] "
                            "[mem: %.2e] "
                            "[iter: %.1f ms] "
                            "[gpu: %.1f ms] "
                            "[data: %.1f ms]"
                            % (
                                epoch + 1,
                                itr,
                                loss_meter.avg,
                                loss_pred_val,
                                loss_context_val,
                                "["
                                + ", ".join(
                                    [
                                        f"{k}: " + "%.1f" % mask_meters[k].avg
                                        for k in mask_meters
                                    ]
                                )
                                + "]",
                                _new_wd,
                                _new_lr,
                                torch.cuda.max_memory_allocated() / 1024.0**2,
                                iter_time_meter.avg,
                                gpu_time_meter.avg,
                                data_elapsed_time_meter.avg,
                            )
                        )

                log_stats()
                assert not np.isnan(loss), "loss is nan"

            # -- Save Checkpoint
            logger.info("avg. loss %.3f" % loss_meter.avg)
            _barrier()

            if (epoch + 1) % CHECKPOINT_FREQ == 0 or epoch == (num_epochs - 1):
                save_checkpoint(epoch + 1, 0, latest_path, None, is_periodic=False)

            if save_every_freq > 0 and (epoch + 1) % save_every_freq == 0:
                save_every_file = f"e{epoch}.pt"
                save_every_path = os.path.join(folder, save_every_file)
                save_checkpoint(epoch + 1, 0, save_every_path, s3_checkpoint_uri, is_periodic=True)

            _barrier()
    finally:
        try:
            _barrier()
        except Exception:
            pass
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
