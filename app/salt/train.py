"""
SALT pretraining: Static-teacher Asymmetric Latent Training.

Stage 1 (V-Pixel): Train encoder + pixel decoder with MSE reconstruction loss.
Stage 2 (Student): Freeze Stage 1 encoder as teacher, train student encoder +
                    predictor to predict teacher latents with L1 loss.

Config field `model.stage` selects which stage to run (1 or 2).
"""

import gc
import os
import random
import time

import numpy as np
import torch
import torch.multiprocessing as mp

try:
    mp.set_sharing_strategy("file_descriptor")
except Exception:
    pass

import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from app.salt.utils import (
    init_opt,
    init_salt_student_model,
    init_vpixel_model,
    load_checkpoint,
    normalize_and_concat,
)
from app.vjepa_2_1.transforms import make_transforms
from src.datasets.data_manager import init_data
from src.masks.multiseq_multiblock3d import MaskCollator
from src.masks.utils import apply_masks
from src.utils.distributed import init_distributed
from src.utils.logging import AverageMeter, CSVLogger, get_logger, gpu_timer


def _clone_for_save(obj):
    """Recursively clone all tensors to break shared storage for serialization."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().clone()
    elif isinstance(obj, dict):
        return {k: _clone_for_save(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(_clone_for_save(x) for x in obj)
    return obj


def _barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


log_freq = 10
CHECKPOINT_FREQ = 1

_GLOBAL_SEED = 0
random.seed(_GLOBAL_SEED)
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

logger = get_logger(__name__, force=True)


def prune_local_checkpoints(folder, max_to_keep=4):
    try:
        all_checkpoints = [f for f in os.listdir(folder) if f.startswith("e") and f.endswith(".pt")]
        if len(all_checkpoints) > max_to_keep:
            all_checkpoints.sort(key=lambda x: int(x[1:-3]))
            for ckpt_name in all_checkpoints[:-max_to_keep]:
                try:
                    os.remove(os.path.join(folder, ckpt_name))
                except Exception as e:
                    logger.error(f"Failed to delete old checkpoint: {e}")
    except Exception as e:
        logger.error(f"Failed to prune checkpoints: {e}")


def compute_pixel_targets(clips, patch_size, tubelet_size):
    """
    Patchify video clips and apply per-patch normalization (VideoMAE style).

    Args:
        clips: list of [B, C, T, H, W] tensors (already ImageNet-normalized).
        patch_size: spatial patch size (e.g. 16).
        tubelet_size: temporal patch size (e.g. 2).

    Returns:
        list of [B, N, C*t*p*p] tensors — one per clip, per-patch normalized.
    """
    targets = []
    p = patch_size
    t = tubelet_size
    for clip in clips:
        B, C, T, H, W = clip.shape
        # Reshape into patches: [B, T/t, H/p, W/p, C, t, p, p]
        x = clip.reshape(B, C, T // t, t, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7)  # [B, T/t, H/p, W/p, C, t, p, p]
        x = x.reshape(B, -1, C * t * p * p)  # [B, N, C*t*p*p]
        # Per-patch normalization
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        x = (x - mean) / (var.sqrt() + 1e-6)
        targets.append(x)
    return targets


def main(args, resume_preempt=False):
    # --------------------------------------------------------------- #
    #  Config parsing
    # --------------------------------------------------------------- #

    # -- META
    cfgs_meta = args.get("meta")
    load_model = cfgs_meta.get("load_checkpoint", False)
    r_file = cfgs_meta.get("read_checkpoint", None)
    seed = cfgs_meta.get("seed", _GLOBAL_SEED)
    save_every_freq = cfgs_meta.get("save_every_freq", -1)
    max_epoch_checkpoints = cfgs_meta.get("max_epoch_checkpoints", 4)
    use_sdpa = cfgs_meta.get("use_sdpa", True)
    dtype_str = cfgs_meta.get("dtype", "float16")
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(dtype_str, torch.float16)
    mixed_precision = dtype != torch.float32

    # -- FOLDER
    folder = args.get("folder")
    os.makedirs(folder, exist_ok=True)

    # -- MASK
    cfgs_mask = args.get("mask")

    # -- MODEL
    cfgs_model = args.get("model")
    stage = cfgs_model.get("stage", 1)
    model_name = cfgs_model.get("model_name", "vit_large")
    uniform_power = cfgs_model.get("uniform_power", False)
    use_activation_checkpointing = cfgs_model.get("use_activation_checkpointing", False)
    use_rope = cfgs_model.get("use_rope", True)
    use_silu = cfgs_model.get("use_silu", False)
    wide_silu = cfgs_model.get("wide_silu", True)

    # Stage 1 decoder config
    decoder_embed_dim = cfgs_model.get("decoder_embed_dim", 512)
    decoder_depth = cfgs_model.get("decoder_depth", 8)
    decoder_num_heads = cfgs_model.get("decoder_num_heads", 8)

    # Stage 2 teacher/predictor config
    teacher_model_name = cfgs_model.get("teacher_model_name", None)
    teacher_checkpoint = cfgs_model.get("teacher_checkpoint", None)
    pred_depth = cfgs_model.get("pred_depth", 12)
    pred_embed_dim = cfgs_model.get("pred_embed_dim", 384)
    pred_num_heads = cfgs_model.get("pred_num_heads", 12)
    use_mask_tokens = cfgs_model.get("use_mask_tokens", True)
    zero_init_mask_tokens = cfgs_model.get("zero_init_mask_tokens", True)
    n_output_distillation = cfgs_model.get("n_output_distillation", None)

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
    pin_mem = cfgs_data.get("pin_mem", False)
    num_workers = cfgs_data.get("num_workers", 1)
    persistent_workers = cfgs_data.get("persistent_workers", True)

    # -- DATA AUGS
    cfgs_data_aug = args.get("data_aug")
    ar_range = cfgs_data_aug.get("random_resize_aspect_ratio", [3 / 4, 4 / 3])
    rr_scale = cfgs_data_aug.get("random_resize_scale", [0.3, 1.0])
    motion_shift = cfgs_data_aug.get("motion_shift", False)
    reprob = cfgs_data_aug.get("reprob", 0.0)
    use_aa = cfgs_data_aug.get("auto_augment", False)

    # -- LOSS
    cfgs_loss = args.get("loss", {})
    loss_exp = cfgs_loss.get("loss_exp", 1.0)

    # -- OPTIMIZATION
    cfgs_opt = args.get("optimization")
    ipe = cfgs_opt.get("ipe", None)
    ipe_scale = cfgs_opt.get("ipe_scale", 1.0)
    wd = float(cfgs_opt.get("weight_decay"))
    final_wd = float(cfgs_opt.get("final_weight_decay"))
    num_epochs = cfgs_opt.get("epochs")
    warmup = cfgs_opt.get("warmup")
    start_lr = cfgs_opt.get("start_lr")
    lr = cfgs_opt.get("lr")
    final_lr = cfgs_opt.get("final_lr")
    betas = cfgs_opt.get("betas", (0.9, 0.999))
    eps = cfgs_opt.get("eps", 1e-8)
    grad_clip = cfgs_opt.get("grad_clip", 0.0)
    force_load_pretrain = cfgs_opt.get("force_load_pretrain", False)
    anneal_ckpt = cfgs_opt.get("anneal_ckpt", None)

    # --------------------------------------------------------------- #
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    try:
        mp.set_start_method("spawn")
    except Exception:
        pass

    # -- init distributed
    world_size, rank = init_distributed()
    logger.info(f"Initialized (rank/world-size) {rank}/{world_size}")

    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

    # -- paths
    log_file = os.path.join(folder, f"log_r{rank}.csv")
    latest_path = os.path.join(folder, "latest.pt")

    load_path = None
    if load_model:
        load_path = os.path.join(folder, r_file) if r_file is not None else latest_path
        if not os.path.exists(load_path):
            load_path = None
            load_model = False

    # -- CSV logger
    csv_logger = CSVLogger(
        log_file,
        ("%d", "epoch"),
        ("%d", "itr"),
        ("%.5f", "loss"),
        ("%d", "iter-time(ms)"),
        ("%d", "gpu-time(ms)"),
        ("%d", "dataload-time(ms)"),
    )

    # --------------------------------------------------------------- #
    #  Model initialization
    # --------------------------------------------------------------- #

    if stage == 1:
        encoder, decoder = init_vpixel_model(
            device=device,
            model_name=model_name,
            crop_size=crop_size,
            patch_size=patch_size,
            max_num_frames=max_num_frames,
            tubelet_size=tubelet_size,
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads,
            uniform_power=uniform_power,
            use_sdpa=use_sdpa,
            use_silu=use_silu,
            wide_silu=wide_silu,
            use_activation_checkpointing=use_activation_checkpointing,
            use_rope=use_rope,
        )
        head = decoder  # generic name for optimizer param groups
    elif stage == 2:
        encoder, predictor, teacher_encoder = init_salt_student_model(
            device=device,
            student_model_name=model_name,
            crop_size=crop_size,
            patch_size=patch_size,
            max_num_frames=max_num_frames,
            tubelet_size=tubelet_size,
            uniform_power=uniform_power,
            use_sdpa=use_sdpa,
            use_silu=use_silu,
            wide_silu=wide_silu,
            use_activation_checkpointing=use_activation_checkpointing,
            use_rope=use_rope,
            teacher_model_name=teacher_model_name,
            teacher_checkpoint=teacher_checkpoint,
            pred_depth=pred_depth,
            pred_num_heads=pred_num_heads,
            pred_embed_dim=pred_embed_dim,
            use_mask_tokens=use_mask_tokens,
            num_mask_tokens=int(len(cfgs_mask) * len(dataset_fpcs)),
            zero_init_mask_tokens=zero_init_mask_tokens,
            n_output_distillation=n_output_distillation,
        )
        head = predictor
    else:
        raise ValueError(f"Invalid stage: {stage}. Must be 1 or 2.")

    # -- mask collator
    mask_collator = MaskCollator(
        cfgs_mask=cfgs_mask,
        dataset_fpcs=dataset_fpcs,
        crop_size=crop_size,
        patch_size=patch_size,
        tubelet_size=tubelet_size,
    )

    # -- transforms
    transform = make_transforms(
        random_horizontal_flip=True,
        random_resize_aspect_ratio=ar_range,
        random_resize_scale=rr_scale,
        reprob=reprob,
        auto_augment=use_aa,
        motion_shift=motion_shift,
        crop_size=crop_size,
    )

    # -- data loader
    (unsupervised_loader, unsupervised_sampler) = init_data(
        data=dataset_type,
        root_path=dataset_paths,
        batch_size=batch_size,
        training=True,
        dataset_fpcs=dataset_fpcs,
        fps=fps,
        transform=transform,
        rank=rank,
        world_size=world_size,
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
        _dlen = -1
    if ipe is None:
        ipe = _dlen
    logger.info(f"iterations per epoch/dataset length: {ipe}/{_dlen}")

    # -- optimizer
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        encoder=encoder,
        head=head,
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

    # BF16 does not need GradScaler
    if dtype == torch.bfloat16:
        scaler = None
        logger.info("BF16 detected — disabling GradScaler")

    # -- DDP
    encoder = DistributedDataParallel(encoder, static_graph=True)
    if stage == 1:
        decoder = DistributedDataParallel(decoder, static_graph=True)
        head = decoder
    elif stage == 2:
        predictor = DistributedDataParallel(predictor, static_graph=False, find_unused_parameters=True)
        teacher_encoder = DistributedDataParallel(teacher_encoder)
        for p in teacher_encoder.parameters():
            p.requires_grad = False
        head = predictor

    # -- Force-load pretrained encoder (e.g. ImageNet init for Stage 1)
    start_epoch = 0
    start_itr = 0

    if force_load_pretrain:
        if anneal_ckpt and os.path.exists(anneal_ckpt):
            logger.info(f"Force-loading pretrained encoder from {anneal_ckpt}")
            from src.utils.checkpoint_loader import robust_checkpoint_loader

            checkpoint = robust_checkpoint_loader(anneal_ckpt, map_location=torch.device("cpu"))

            # Handle both formats: V-JEPA dict (has 'encoder' key) or flat state dict (ImageNet)
            if "encoder" in checkpoint:
                pretrained_dict = checkpoint["encoder"]
            else:
                pretrained_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))

            # Strip DDP/wrapper prefixes and drop classifier heads
            pretrained_dict = {
                k.replace("module.", "").replace("backbone.", ""): v
                for k, v in pretrained_dict.items()
                if not k.startswith(("head.", "fc_norm.", "module.head.", "module.fc_norm."))
            }

            # Inflate 2D patch_embed to 3D if loading from an image checkpoint
            pe_key = "patch_embed.proj.weight"
            if pe_key in pretrained_dict and pretrained_dict[pe_key].ndim == 4:
                pe_2d = pretrained_dict[pe_key]
                t = tubelet_size
                pe_3d = pe_2d.unsqueeze(2).repeat(1, 1, t, 1, 1) / float(t)
                pretrained_dict[pe_key] = pe_3d
                logger.info(f"Inflated patch_embed.proj.weight: {pe_2d.shape} -> {pe_3d.shape}")

            # Skip shape mismatches
            model_dict = encoder.module.backbone.state_dict()
            for k in list(pretrained_dict.keys()):
                if k in model_dict and pretrained_dict[k].shape != model_dict[k].shape:
                    logger.warning(f"Skipping {k}: ckpt {pretrained_dict[k].shape} vs model {model_dict[k].shape}")
                    del pretrained_dict[k]

            msg = encoder.module.backbone.load_state_dict(pretrained_dict, strict=False)
            logger.info(f"Loaded pretrained encoder with msg: {msg}")

            del checkpoint
            gc.collect()
        else:
            raise FileNotFoundError(f"force_load_pretrain set but checkpoint not found: {anneal_ckpt}")
    elif load_model and load_path and os.path.exists(load_path):
        encoder, head, optimizer, scaler, start_epoch, start_itr = load_checkpoint(
            r_path=load_path,
            stage=stage,
            encoder=encoder,
            head=head,
            opt=optimizer,
            scaler=scaler,
        )
        if stage == 1:
            decoder = head
        else:
            predictor = head
        for _ in range(start_epoch * ipe):
            scheduler.step()
            wd_scheduler.step()
            mask_collator.step()

    # -- Checkpoint save
    def save_checkpoint(epoch, itr, local_path, is_periodic=False):
        if rank != 0:
            return
        save_dict = {
            "encoder": encoder.state_dict(),
            "opt": optimizer.state_dict(),
            "scaler": None if scaler is None else scaler.state_dict(),
            "epoch": epoch,
            "itr": itr,
            "loss": loss_meter.avg,
            "batch_size": batch_size,
            "world_size": world_size,
            "lr": lr,
            "stage": stage,
        }
        if stage == 1:
            save_dict["decoder"] = decoder.state_dict()
        else:
            save_dict["predictor"] = predictor.state_dict()
            save_dict["teacher_checkpoint"] = teacher_checkpoint

        save_dict = _clone_for_save(save_dict)
        tmp_path = local_path + ".tmp"
        try:
            torch.save(save_dict, tmp_path)
            os.replace(tmp_path, local_path)
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            return

        if is_periodic and max_epoch_checkpoints > 0:
            prune_local_checkpoints(os.path.dirname(local_path), max_to_keep=max_epoch_checkpoints)

    # -- Collect trainable params for grad clipping
    trainable_params = list(encoder.parameters()) + list(head.parameters())

    # -- Determine teacher embed dim for Stage 2 hierarchical norm
    teacher_embed_dim = None
    if stage == 2:
        teacher_embed_dim = teacher_encoder.module.backbone.embed_dim

    # --------------------------------------------------------------- #
    #  Training loop
    # --------------------------------------------------------------- #
    logger.info(f"Starting SALT Stage {stage} training...")
    unsupervised_sampler.set_epoch(start_epoch)
    loader = iter(unsupervised_loader)

    try:
        for epoch in range(start_epoch, num_epochs):
            unsupervised_sampler.set_epoch(epoch)
            logger.info("Epoch %d" % (epoch + 1))

            loss_meter = AverageMeter()
            iter_time_meter = AverageMeter()
            gpu_time_meter = AverageMeter()
            data_elapsed_time_meter = AverageMeter()

            itr_start_offset = start_itr if epoch == start_epoch else 0

            for itr in range(itr_start_offset, ipe):
                itr_start_time = time.time()

                # -- Load data with retry
                iter_retries = 0
                iter_successful = False
                while not iter_successful:
                    try:
                        sample = next(loader)
                        iter_successful = True
                    except StopIteration:
                        unsupervised_sampler.set_epoch(epoch)
                        loader = iter(unsupervised_loader)
                    except Exception as e:
                        if iter_retries < 5:
                            logger.warning(f"Data loading error (retry {iter_retries}): {e}")
                            iter_retries += 1
                            time.sleep(5)
                            loader = iter(unsupervised_loader)
                        else:
                            logger.warning("Exceeded max retries; rebuilding DataLoader.")
                            (unsupervised_loader, unsupervised_sampler) = init_data(
                                data=dataset_type,
                                root_path=dataset_paths,
                                batch_size=batch_size,
                                training=True,
                                dataset_fpcs=dataset_fpcs,
                                fps=fps,
                                transform=transform,
                                rank=rank,
                                world_size=world_size,
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

                def load_clips():
                    all_clips, all_masks_enc, all_masks_pred = [], [], []
                    for fpc_sample in sample:
                        udata, masks_enc, masks_pred = fpc_sample
                        all_clips += [udata[0][0].to(device, non_blocking=True)]
                        all_masks_enc += [[m.to(device, non_blocking=True) for m in masks_enc]]
                        all_masks_pred += [[m.to(device, non_blocking=True) for m in masks_pred]]
                    return all_clips, all_masks_enc, all_masks_pred

                clips, masks_enc, masks_pred = load_clips()
                data_elapsed_time_ms = (time.time() - itr_start_time) * 1000.0

                def train_step():
                    _new_lr = scheduler.step()
                    _new_wd = wd_scheduler.step()

                    with torch.cuda.amp.autocast(dtype=dtype, enabled=mixed_precision):

                        if stage == 1:
                            # ============================================
                            #  Stage 1: V-Pixel (pixel reconstruction)
                            # ============================================

                            # Encoder forward: visible patches only, single-level output
                            z = encoder(clips, masks_enc, training_mode=False)

                            # Compute pixel reconstruction targets
                            pixel_targets = compute_pixel_targets(clips, patch_size, tubelet_size)

                            # Decoder forward + MSE loss
                            loss, n = 0, 0
                            for fpc_idx, (z_fpc, me_fpc, mp_fpc, pt_fpc) in enumerate(
                                zip(z, masks_enc, masks_pred, pixel_targets)
                            ):
                                for z_ij, me_ij, mp_ij in zip(z_fpc, me_fpc, mp_fpc):
                                    pixel_pred = decoder(z_ij, me_ij, mp_ij)
                                    # Extract pixel targets at masked positions
                                    target = apply_masks(pt_fpc, [mp_ij])
                                    loss += F.mse_loss(pixel_pred, target)
                                    n += 1
                            loss /= n

                        elif stage == 2:
                            # ============================================
                            #  Stage 2: Frozen-teacher latent prediction
                            # ============================================

                            # Frozen teacher: FULL unmasked forward (hierarchical)
                            with torch.no_grad():
                                h = teacher_encoder(clips, training_mode=True)
                                # h: list of [B, N_total, 4*teacher_embed_dim]
                                h = [normalize_and_concat(hi, teacher_embed_dim) for hi in h]

                            # Student encoder: visible patches only (hierarchical)
                            z = encoder(clips, masks_enc, training_mode=True)

                            # Predictor: predict teacher latents at masked positions
                            z_pred, _ = predictor(z, masks_enc, masks_pred)

                            # L1 loss on masked patch predictions
                            loss, n = 0, 0
                            for fpc_idx, (zp_fpc, h_fpc, mp_fpc) in enumerate(
                                zip(z_pred, h, masks_pred)
                            ):
                                h_masked = apply_masks(h_fpc, mp_fpc, concat=False)
                                for z_ij, h_ij in zip(zp_fpc, h_masked):
                                    loss += torch.mean(torch.abs(z_ij - h_ij) ** loss_exp) / loss_exp
                                    n += 1
                            loss /= n

                    # -- Backward
                    if scaler is not None:
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                    else:
                        loss.backward()

                    # -- Gradient clipping
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=grad_clip)

                    # -- Optimizer step
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()

                    # No EMA update — this is the whole point of SALT

                    return float(loss), _new_lr, _new_wd

                (loss, _new_lr, _new_wd), gpu_etime_ms = gpu_timer(train_step)
                iter_elapsed_time_ms = (time.time() - itr_start_time) * 1000.0
                loss_meter.update(loss)
                iter_time_meter.update(iter_elapsed_time_ms)
                gpu_time_meter.update(gpu_etime_ms)
                data_elapsed_time_meter.update(data_elapsed_time_ms)

                # -- Logging
                csv_logger.log(
                    epoch + 1,
                    itr,
                    loss,
                    iter_elapsed_time_ms,
                    gpu_etime_ms,
                    data_elapsed_time_ms,
                )
                if (itr % log_freq == 0) or (itr == ipe - 1) or np.isnan(loss) or np.isinf(loss):
                    logger.info(
                        "[%d, %5d] loss: %.3f "
                        "[wd: %.2e] [lr: %.2e] "
                        "[mem: %.2e] "
                        "[iter: %.1f ms] [gpu: %.1f ms] [data: %.1f ms]"
                        % (
                            epoch + 1,
                            itr,
                            loss_meter.avg,
                            _new_wd,
                            _new_lr,
                            torch.cuda.max_memory_allocated() / 1024.0**2,
                            iter_time_meter.avg,
                            gpu_time_meter.avg,
                            data_elapsed_time_meter.avg,
                        )
                    )
                assert not np.isnan(loss), "loss is nan"

            # -- Save checkpoint at end of epoch
            logger.info("avg. loss %.3f" % loss_meter.avg)
            _barrier()

            if (epoch + 1) % CHECKPOINT_FREQ == 0 or epoch == (num_epochs - 1):
                save_checkpoint(epoch + 1, 0, latest_path, is_periodic=False)

            if save_every_freq > 0 and (epoch + 1) % save_every_freq == 0:
                save_every_path = os.path.join(folder, f"e{epoch}.pt")
                save_checkpoint(epoch + 1, 0, save_every_path, is_periodic=True)

            _barrier()
    finally:
        try:
            _barrier()
        except Exception:
            pass
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
