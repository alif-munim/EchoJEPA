# BYOL-Video training app for EchoJEPA ICML rebuttal.
# Implements self-distillation with momentum encoder (no masking, no patch-level prediction).
# Reference: Feichtenhofer et al., "A Large-Scale Study on Unsupervised Spatiotemporal
# Representation Learning" (CVPR 2021).

import os

try:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_LOCALID"]
except Exception:
    pass

import copy
import gc
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
from torch.nn.parallel import DistributedDataParallel

from app.byol_video.utils import init_byol_model, init_opt, load_checkpoint
from app.vjepa_2_1.transforms import make_transforms
from src.datasets.data_manager import init_data
from src.utils.checkpoint_loader import robust_checkpoint_loader
from src.utils.distributed import init_distributed
from src.utils.logging import AverageMeter, CSVLogger, get_logger, gpu_timer


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
            checkpoints_to_delete = all_checkpoints[:-max_to_keep]
            logger.info(f"Pruning checkpoints. Keeping {max_to_keep}, deleting {len(checkpoints_to_delete)}.")
            for ckpt_name in checkpoints_to_delete:
                full_path = os.path.join(folder, ckpt_name)
                try:
                    os.remove(full_path)
                except Exception as e:
                    logger.error(f"Failed to delete old checkpoint {full_path}: {e}")
    except Exception as e:
        logger.error(f"Failed to prune checkpoints in folder {folder}: {e}")


class BYOLCollator:
    """Collator for BYOL-Video: stacks multi-clip samples without generating masks."""

    def __init__(self):
        pass

    def step(self):
        pass

    def __call__(self, batch):
        # batch: list of (clips_list, label, clip_indices, uri) from VideoDataset
        # clips_list: list of num_clips tensors, each [C, T, H, W]
        num_clips = len(batch[0][0])
        clips = []
        for i in range(num_clips):
            clip_i = torch.stack([sample[0][i] for sample in batch])  # [B, C, T, H, W]
            clips.append(clip_i)
        return clips


def main(args, resume_preempt=False):
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
    use_sdpa = cfgs_meta.get("use_sdpa", False)
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

    # -- MODEL
    cfgs_model = args.get("model")
    model_name = cfgs_model.get("model_name")
    uniform_power = cfgs_model.get("uniform_power", False)
    use_rope = cfgs_model.get("use_rope", False)
    use_silu = cfgs_model.get("use_silu", False)
    wide_silu = cfgs_model.get("wide_silu", True)
    use_activation_checkpointing = cfgs_model.get("use_activation_checkpointing", False)
    proj_hidden_dim = cfgs_model.get("proj_hidden_dim", 4096)
    proj_dim = cfgs_model.get("proj_dim", 256)
    pred_hidden_dim = cfgs_model.get("pred_hidden_dim", 4096)

    # -- DATA
    cfgs_data = args.get("data")
    dataset_type = cfgs_data.get("dataset_type", "videodataset")
    dataset_paths = cfgs_data.get("datasets", [])
    datasets_weights = cfgs_data.get("datasets_weights")
    dataset_fpcs = cfgs_data.get("dataset_fpcs")
    max_num_frames = max(dataset_fpcs)
    num_temporal_clips = cfgs_data.get("num_clips", 2)
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

    # -- OPTIMIZATION
    cfgs_opt = args.get("optimization")
    is_anneal = cfgs_opt.get("is_anneal", False)
    force_load_pretrain = cfgs_opt.get("force_load_pretrain", False)
    anneal_ckpt_path = cfgs_opt.get("anneal_ckpt", None)
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
    betas = cfgs_opt.get("betas", (0.9, 0.999))
    eps = cfgs_opt.get("eps", 1.0e-8)

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True

    try:
        mp.set_start_method("spawn")
    except Exception:
        pass

    world_size, rank = init_distributed()
    logger.info(f"Initialized (rank/world-size) {rank}/{world_size}")

    if rank == 0 and s3_checkpoint_uri:
        logger.info(f"Checkpoints will be uploaded to: {s3_checkpoint_uri}")

    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    log_file = os.path.join(folder, f"log_r{rank}.csv")
    csv_logger = CSVLogger(
        log_file,
        ("%d", "epoch"),
        ("%d", "itr"),
        ("%.5f", "loss"),
        ("%d", "iter-time(ms)"),
        ("%d", "gpu-time(ms)"),
        ("%d", "dataload-time(ms)"),
    )

    # -- INIT MODEL
    encoder, online_projector, online_predictor = init_byol_model(
        device=device,
        patch_size=patch_size,
        max_num_frames=max_num_frames,
        tubelet_size=tubelet_size,
        model_name=model_name,
        crop_size=crop_size,
        uniform_power=uniform_power,
        use_sdpa=use_sdpa,
        use_rope=use_rope,
        use_silu=use_silu,
        wide_silu=wide_silu,
        use_activation_checkpointing=use_activation_checkpointing,
        proj_hidden_dim=proj_hidden_dim,
        proj_dim=proj_dim,
        pred_hidden_dim=pred_hidden_dim,
    )

    # Target branch: deepcopy encoder + projector (no predictor on target side)
    logger.info("Creating target_encoder and target_projector via deepcopy...")
    target_encoder = copy.deepcopy(encoder)
    target_projector = copy.deepcopy(online_projector)

    # -- DATA
    collator = BYOLCollator()

    transform = make_transforms(
        random_horizontal_flip=False,
        random_resize_aspect_ratio=ar_range,
        random_resize_scale=rr_scale,
        reprob=reprob,
        auto_augment=use_aa,
        motion_shift=motion_shift,
        crop_size=crop_size,
    )

    (unsupervised_loader, unsupervised_sampler) = init_data(
        data=dataset_type,
        root_path=dataset_paths,
        batch_size=batch_size,
        training=True,
        dataset_fpcs=dataset_fpcs,
        fps=fps,
        num_clips=num_temporal_clips,
        transform=transform,
        rank=rank,
        world_size=world_size,
        datasets_weights=datasets_weights,
        persistent_workers=persistent_workers,
        collator=collator,
        num_workers=num_workers,
        pin_mem=pin_mem,
        log_dir=None,
    )

    try:
        _dlen = len(unsupervised_loader)
    except Exception:
        _dlen = unsupervised_loader.num_batches

    if ipe is None:
        ipe = _dlen
    logger.info(f"iterations per epoch/dataset length: {ipe}/{_dlen}")

    # -- OPTIMIZER
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        is_anneal=is_anneal,
        encoder=encoder,
        online_projector=online_projector,
        online_predictor=online_predictor,
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

    def make_momentum_scheduler(start_step=0):
        total = int(ipe * num_epochs * ipe_scale)
        return (ema[0] + i * (ema[1] - ema[0]) / total for i in range(start_step, total + 1))

    start_epoch, start_itr = 0, 0
    completed_steps = 0

    # -- LOAD WEIGHTS
    if force_load_pretrain:
        if anneal_ckpt_path and os.path.exists(anneal_ckpt_path):
            logger.info(f"FORCE-LOADING pretrained model from {anneal_ckpt_path}")
            checkpoint = robust_checkpoint_loader(anneal_ckpt_path, map_location=torch.device("cpu"))

            # Handle both formats: V-JEPA dict (has 'encoder' key) or flat state dict (ImageNet)
            if "encoder" in checkpoint:
                pretrained_dict = checkpoint["encoder"]
                epoch_from_ckpt = checkpoint.get("epoch", 0)
            else:
                # Flat state dict (e.g. ImageNet ViT-L from vitl_raw.pth / vitl_in21k.pt)
                pretrained_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
                epoch_from_ckpt = 0

            # Strip DDP 'module.' and MultiSeqWrapper 'backbone.' prefixes, drop classifier heads
            pretrained_dict = {
                k.replace("module.", "").replace("backbone.", ""): v
                for k, v in pretrained_dict.items()
                if not k.startswith(("head.", "fc_norm.", "module.head.", "module.fc_norm."))
            }

            # Inflate 2D patch_embed to 3D if loading from an image checkpoint
            pe_key = "patch_embed.proj.weight"
            if pe_key in pretrained_dict and pretrained_dict[pe_key].ndim == 4:
                pe_2d = pretrained_dict[pe_key]  # [C_out, C_in, H, W]
                t = tubelet_size
                pe_3d = pe_2d.unsqueeze(2).repeat(1, 1, t, 1, 1) / float(t)
                pretrained_dict[pe_key] = pe_3d
                logger.info(f"Inflated patch_embed.proj.weight: {pe_2d.shape} -> {pe_3d.shape}")

            msg = encoder.load_state_dict(pretrained_dict, strict=False)
            logger.info(f"Loaded pretrained encoder from epoch {epoch_from_ckpt} with msg: {msg}")

            # Copy encoder weights to target_encoder
            target_encoder.load_state_dict(encoder.state_dict())
            logger.info("Copied encoder weights to target_encoder")

            # Projector + predictor train from scratch
            target_projector.load_state_dict(online_projector.state_dict())

            del checkpoint
            gc.collect()
            logger.info("Force-loading complete. Starting fresh from epoch 0.")
            completed_steps = 0
        else:
            logger.error(f"Configured to force-load but checkpoint not found: {anneal_ckpt_path}")
            raise FileNotFoundError(f"Anneal checkpoint not found: {anneal_ckpt_path}")
    else:
        latest_path = os.path.join(folder, "latest.pt")
        load_path = None

        if load_model or os.path.exists(latest_path):
            load_path = os.path.join(folder, r_file) if r_file is not None else latest_path

            if load_path and os.path.exists(load_path):
                logger.info(f"Resuming training from checkpoint: {load_path}")
                (
                    encoder,
                    online_projector,
                    online_predictor,
                    target_encoder,
                    target_projector,
                    optimizer,
                    scaler,
                    start_epoch,
                    start_itr,
                ) = load_checkpoint(
                    r_path=load_path,
                    encoder=encoder,
                    online_projector=online_projector,
                    online_predictor=online_predictor,
                    target_encoder=target_encoder,
                    target_projector=target_projector,
                    opt=optimizer,
                    scaler=scaler,
                )
                logger.info(f"SUCCESS: Loaded checkpoint from {load_path}")
                completed_steps = start_epoch * ipe + start_itr

                for _ in range(completed_steps):
                    scheduler.step()
                    wd_scheduler.step()

    # -- DDP: wrap online branch only (targets have no gradients)
    if dist.is_available() and dist.is_initialized():
        encoder = DistributedDataParallel(encoder, static_graph=True)
        online_projector = DistributedDataParallel(online_projector, static_graph=True)
        online_predictor = DistributedDataParallel(online_predictor, static_graph=True)

    for p in target_encoder.parameters():
        p.requires_grad = False
    for p in target_projector.parameters():
        p.requires_grad = False

    momentum_scheduler = make_momentum_scheduler(start_step=completed_steps)

    # -- CHECKPOINT SAVE
    loss_meter = AverageMeter()

    def save_checkpoint(epoch, itr, local_path, s3_uri_base=None, is_periodic=False):
        if rank != 0:
            return

        save_dict = {
            "encoder": encoder.state_dict(),
            "online_projector": online_projector.state_dict(),
            "online_predictor": online_predictor.state_dict(),
            "target_encoder": target_encoder.state_dict(),
            "target_projector": target_projector.state_dict(),
            "opt": optimizer.state_dict(),
            "scaler": None if scaler is None else scaler.state_dict(),
            "epoch": epoch,
            "loss": loss_meter.avg,
            "batch_size": batch_size,
            "world_size": world_size,
            "lr": lr,
            "itr": itr,
        }

        try:
            torch.save(save_dict, local_path)
        except Exception as e:
            logger.error(f"Encountered exception when saving checkpoint: {e}")
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
                    s3_client.upload_file(local_path, bucket, s3_key)
                else:
                    with open(local_path, "rb") as f:
                        s3_client.put_object(Bucket=bucket, Key=s3_key, Body=f.read())
                logger.info(f"Uploaded checkpoint to s3://{bucket}/{s3_key}")
            except Exception as e:
                logger.error(f"Failed to upload checkpoint to S3: {e}")

        if is_periodic and max_epoch_checkpoints > 0:
            prune_local_checkpoints(os.path.dirname(local_path), max_to_keep=max_epoch_checkpoints)

    # -- TRAINING LOOP
    logger.info("Initializing loader...")
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
                        unsupervised_sampler.set_epoch(epoch)
                        loader = iter(unsupervised_loader)
                    except Exception as e:
                        NUM_RETRIES = 5
                        if iter_retries < NUM_RETRIES:
                            logger.warning(
                                f"Encountered exception when loading data (retry {iter_retries}):\n{e}"
                            )
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
                                num_clips=num_temporal_clips,
                                transform=transform,
                                rank=rank,
                                world_size=world_size,
                                datasets_weights=datasets_weights,
                                persistent_workers=persistent_workers,
                                collator=collator,
                                num_workers=num_workers,
                                pin_mem=pin_mem,
                                log_dir=None,
                            )
                            unsupervised_sampler.set_epoch(epoch)
                            loader = iter(unsupervised_loader)
                            iter_retries = 0
                            continue

                # sample is a list of num_temporal_clips tensors, each [B, C, T, H, W]
                clips = [c.to(device, non_blocking=True) for c in sample]
                data_elapsed_time_ms = (time.time() - itr_start_time) * 1000.0

                def train_step():
                    _new_lr = scheduler.step()
                    _new_wd = wd_scheduler.step()

                    # Per-pair gradient accumulation: each pair does forward+backward
                    # independently, avoiding BN running-stat inplace version conflicts.
                    optimizer.zero_grad()
                    total_pairs = num_temporal_clips * (num_temporal_clips - 1)
                    loss_accum = 0.0

                    for i in range(num_temporal_clips):
                        for j in range(num_temporal_clips):
                            if i == j:
                                continue

                            # Online branch
                            with torch.amp.autocast("cuda", dtype=dtype, enabled=mixed_precision):
                                z = encoder(clips[i])       # [B, N, D]
                                z = z.mean(dim=1)            # [B, D]
                                z = online_projector(z)      # [B, proj_dim]
                                z = online_predictor(z)      # [B, proj_dim]

                            # Target branch (no grad)
                            with torch.no_grad():
                                h = target_encoder(clips[j])  # [B, N, D]
                                h = h.mean(dim=1)              # [B, D]
                                h = target_projector(h)        # [B, proj_dim]

                            # Cosine loss for this pair
                            with torch.amp.autocast("cuda", dtype=dtype, enabled=mixed_precision):
                                z_norm = F.normalize(z, dim=-1)
                                h_norm = F.normalize(h.detach(), dim=-1)
                                pair_loss = -2.0 * (z_norm * h_norm).sum(dim=-1).mean()

                            # Backward for this pair (gradients accumulate)
                            scaled = pair_loss / total_pairs
                            if mixed_precision:
                                scaler.scale(scaled).backward()
                            else:
                                scaled.backward()
                            loss_accum += pair_loss.item()

                    loss_accum /= total_pairs

                    if mixed_precision:
                        scaler.unscale_(optimizer)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()

                    # EMA update: both encoder AND projector
                    m = next(momentum_scheduler)
                    with torch.no_grad():
                        for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                            param_k.mul_(m).add_(param_q, alpha=1 - m)
                        for param_q, param_k in zip(
                            online_projector.parameters(), target_projector.parameters()
                        ):
                            param_k.mul_(m).add_(param_q, alpha=1 - m)

                    return loss_accum, _new_lr, _new_wd

                (loss, _new_lr, _new_wd), gpu_etime_ms = gpu_timer(train_step)
                iter_elapsed_time_ms = (time.time() - itr_start_time) * 1000.0

                loss_meter.update(loss)
                iter_time_meter.update(iter_elapsed_time_ms)
                gpu_time_meter.update(gpu_etime_ms)
                data_elapsed_time_meter.update(data_elapsed_time_ms)

                def log_stats():
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
                            "[iter: %.1f ms] "
                            "[gpu: %.1f ms] "
                            "[data: %.1f ms]"
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

                log_stats()
                assert not np.isnan(loss), "loss is nan"

            logger.info("avg. loss %.3f" % loss_meter.avg)
            _barrier()

            latest_path = os.path.join(folder, "latest.pt")
            if epoch % CHECKPOINT_FREQ == 0 or epoch == (num_epochs - 1):
                save_checkpoint(epoch + 1, 0, latest_path, None, is_periodic=False)

            if save_every_freq > 0 and epoch % save_every_freq == 0:
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
