# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    # -- WARNING: IF DOING DISTRIBUTED TRAINING ON A NON-SLURM CLUSTER, MAKE
    # --          SURE TO UPDATE THIS TO GET LOCAL-RANK ON NODE, OR ENSURE
    # --          THAT YOUR JOBS ARE LAUNCHED WITH ONLY 1 DEVICE VISIBLE
    # --          TO EACH PROCESS
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_LOCALID"]
except Exception:
    pass

import logging
import math
import pprint

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from evals.video_classification_frozen.models import init_module
from evals.video_classification_frozen.utils import make_transforms
from src.datasets.data_manager import init_data
from src.models.attentive_pooler import AttentiveClassifier
from src.utils.checkpoint_loader import robust_checkpoint_loader
from src.utils.distributed import AllReduce, init_distributed
from src.utils.logging import AverageMeter, CSVLogger

from evals.action_anticipation_frozen.losses import sigmoid_focal_loss

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

pp = pprint.PrettyPrinter(indent=4)


def main(args_eval, resume_preempt=False):

    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- VAL ONLY
    val_only = args_eval.get("val_only", False)
    if val_only:
        logger.info("VAL ONLY")
    predictions_save_path = args_eval.get("predictions_save_path", None)

    # -- EXPERIMENT
    pretrain_folder = args_eval.get("folder", None)
    resume_checkpoint = args_eval.get("resume_checkpoint", False) or resume_preempt
    eval_tag = args_eval.get("tag", None)
    num_workers = args_eval.get("num_workers", 12)

    # -- PRETRAIN
    args_pretrain = args_eval.get("model_kwargs")
    checkpoint = args_pretrain.get("checkpoint")
    module_name = args_pretrain.get("module_name")
    args_model = args_pretrain.get("pretrain_kwargs")
    args_wrapper = args_pretrain.get("wrapper_kwargs")

    args_exp = args_eval.get("experiment")

    # -- CLASSIFIER
    args_classifier = args_exp.get("classifier")
    num_probe_blocks = args_classifier.get("num_probe_blocks", 1)
    num_heads = args_classifier.get("num_heads", 16)
    probe_checkpoint = args_eval.get("probe_checkpoint", None)

    # -- DATA
    args_data = args_exp.get("data")
    dataset_type = args_data.get("dataset_type", "VideoDataset")
    num_classes = args_data.get("num_classes")
    train_data_path = [args_data.get("dataset_train")]
    val_data_path = [args_data.get("dataset_val")]
    resolution = args_data.get("resolution", 224)
    num_segments = args_data.get("num_segments", 1)
    frames_per_clip = args_data.get("frames_per_clip", 16)
    frame_step = args_data.get("frame_step", 4)
    duration = args_data.get("clip_duration", None)
    num_views_per_segment = args_data.get("num_views_per_segment", 1)
    normalization = args_data.get("normalization", None)

    # -- OPTIMIZATION
    args_opt = args_exp.get("optimization")
    use_focal_loss = args_opt.get("use_focal_loss", False)

    batch_size = args_opt.get("batch_size")
    num_epochs = args_opt.get("num_epochs")
    use_bfloat16 = args_opt.get("use_bfloat16")
    opt_kwargs = [
        dict(
            ref_wd=kwargs.get("weight_decay"),
            final_wd=kwargs.get("final_weight_decay"),
            start_lr=kwargs.get("start_lr"),
            ref_lr=kwargs.get("lr"),
            final_lr=kwargs.get("final_lr"),
            warmup=kwargs.get("warmup"),
        )
        for kwargs in args_opt.get("multihead_kwargs")
    ]
    # ----------------------------------------------------------------------- #

    try:
        mp.set_start_method("spawn")
    except Exception:
        pass

    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

    world_size, rank = init_distributed()
    logger.info(f"Initialized (rank/world-size) {rank}/{world_size}")

    # -- log/checkpointing paths  
    folder = os.path.join(pretrain_folder, "video_classification_frozen/")  
    if eval_tag is not None:  
        folder = os.path.join(folder, eval_tag)  
    if not os.path.exists(folder):  
        os.makedirs(folder, exist_ok=True)  
    log_file = os.path.join(folder, f"log_r{rank}.csv")  
      
    # Use custom probe checkpoint if specified, otherwise use default  
    if probe_checkpoint is not None:  
        latest_path = probe_checkpoint  
    else:  
        latest_path = os.path.join(folder, "latest.pt")

    # -- make csv_logger
    if rank == 0:
        csv_logger = CSVLogger(log_file, ("%d", "epoch"), ("%.5f", "loss"), ("%.5f", "acc"))

    # Initialize model

    # -- init models
    encoder = init_module(
        module_name=module_name,
        frames_per_clip=frames_per_clip,
        resolution=resolution,
        checkpoint=checkpoint,
        model_kwargs=args_model,
        wrapper_kwargs=args_wrapper,
        device=device,
    )

    # -- init classifier (disable activation checkpointing; you have headroom)
    classifiers = [
        AttentiveClassifier(
            embed_dim=encoder.embed_dim,
            num_heads=num_heads,
            depth=num_probe_blocks,
            num_classes=num_classes,
            use_activation_checkpointing=True,
        ).to(device)
        for _ in opt_kwargs
    ]
    # classifiers = [DistributedDataParallel(c, static_graph=True) for c in classifiers
    # Add distributed check to avoid DDP crash when running concurrent jobs  
    from torch import distributed as dist  
    use_ddp = dist.is_available() and dist.is_initialized() and world_size > 1  
    if use_ddp:  
        classifiers = [DistributedDataParallel(c, static_graph=True) for c in classifiers]  
    else:  
        logger.info(f"DDP disabled (world_size={world_size}); running single-process.")
        
    print(classifiers[0])

    train_loader, train_sampler = make_dataloader(
        dataset_type=dataset_type,
        root_path=train_data_path,
        img_size=resolution,
        frames_per_clip=frames_per_clip,
        frame_step=frame_step,
        eval_duration=duration,
        num_segments=num_segments,
        num_views_per_segment=1,
        allow_segment_overlap=True,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        training=True,
        num_workers=num_workers,
        normalization=normalization,
    )
    val_loader, _ = make_dataloader(
        dataset_type=dataset_type,
        root_path=val_data_path,
        img_size=resolution,
        frames_per_clip=frames_per_clip,
        frame_step=frame_step,
        num_segments=num_segments,
        eval_duration=duration,
        num_views_per_segment=num_views_per_segment,
        allow_segment_overlap=True,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        training=False,
        num_workers=num_workers,
        normalization=normalization,
    )
    ipe = len(train_loader)
    logger.info(f"Dataloader created... iterations per epoch: {ipe}")

    # -- optimizer and scheduler
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        classifiers=classifiers,
        opt_kwargs=opt_kwargs,
        iterations_per_epoch=ipe,
        num_epochs=num_epochs,
        use_bfloat16=use_bfloat16,
    )

    # -- load training checkpoint
    start_epoch = 0
    if resume_checkpoint and os.path.exists(latest_path):
        classifiers, optimizer, scaler, start_epoch = load_checkpoint(
            device=device,
            r_path=latest_path,
            classifiers=classifiers,
            opt=optimizer,
            scaler=scaler,
            val_only=val_only,
        )
        for _ in range(start_epoch * ipe):
            [s.step() for s in scheduler]
            [wds.step() for wds in wd_scheduler]

    # ---- per-head running stats ----
    best_per_head = None
    sum_per_head = None
    min_per_head = None
    best_epoch_per_head = None
    count_epochs = 0

    def save_checkpoint(epoch, mean_val_acc, best_val_acc,
                        val_heads, best_per_head, mean_per_head, min_per_head, best_epoch_per_head):
        all_classifier_dicts = [c.state_dict() for c in classifiers]
        all_opt_dicts = [o.state_dict() for o in optimizer]

        save_dict = {
            "classifiers": all_classifier_dicts,
            "opt": all_opt_dicts,
            "scaler": None if scaler is None else [s.state_dict() for s in scaler],
            "epoch": epoch,
            "batch_size": batch_size,
            "world_size": world_size,

            # ---- scalar metrics (max over heads, as before) ----
            "mean_val_acc": float(mean_val_acc),
            "best_val_acc": float(best_val_acc),

            # ---- per-head metrics ----
            "val_acc_per_head": np.asarray(val_heads, dtype=float).tolist(),
            "best_val_acc_per_head": np.asarray(best_per_head, dtype=float).tolist(),
            "mean_val_acc_per_head": np.asarray(mean_per_head, dtype=float).tolist(),
            "min_val_acc_per_head": np.asarray(min_per_head, dtype=float).tolist(),
            "best_epoch_per_head": np.asarray(best_epoch_per_head, dtype=int).tolist(),

            # ---- grid (LR/WD mapping for each head) ----
            "opt_grid": opt_kwargs,
        }
        if rank == 0:
            # keep rolling latest
            _latest_path = os.path.join(folder, "latest.pt")
            torch.save(save_dict, _latest_path)

            # save a per-epoch snapshot too
            epoch_path = os.path.join(folder, f"epoch_{epoch:03d}.pt")
            torch.save(save_dict, epoch_path)

    # TRAIN LOOP
    best_val_acc_scalar = 0.0
    val_cnt = 0
    val_sum_scalar = 0.0
    for epoch in range(start_epoch, num_epochs):
        logger.info("Epoch %d" % (epoch + 1))
        train_sampler.set_epoch(epoch)

        if val_only:  
            train_acc_scalar, _ = -1.0, None  
        else:  
            train_acc_scalar, _ = run_one_epoch(  
                device=device,  
                training=True,  
                encoder=encoder,  
                classifiers=classifiers,  
                scaler=scaler,  
                optimizer=optimizer,  
                scheduler=scheduler,  
                wd_scheduler=wd_scheduler,  
                data_loader=train_loader,  
                use_bfloat16=use_bfloat16,  
                use_focal_loss=use_focal_loss,  
                val_only=val_only,  
                predictions_save_path=predictions_save_path,  
            )  

        val_acc_scalar, val_heads = run_one_epoch(  
            device=device,  
            training=False,  
            encoder=encoder,  
            classifiers=classifiers,  
            scaler=scaler,  
            optimizer=optimizer,  
            scheduler=scheduler,  
            wd_scheduler=wd_scheduler,  
            data_loader=val_loader,  
            use_bfloat16=use_bfloat16,  
            use_focal_loss=use_focal_loss,  
            val_only=val_only,  
            predictions_save_path=predictions_save_path,  
        )  

        # ---- update scalar running stats (max over heads) ----
        val_cnt += 1
        val_sum_scalar += float(val_acc_scalar)
        mean_val_acc_scalar = val_sum_scalar / val_cnt
        best_val_acc_scalar = max(best_val_acc_scalar, float(val_acc_scalar))

        # ---- update per-head running stats ----
        count_epochs += 1
        if best_per_head is None:
            best_per_head = val_heads.copy()
            sum_per_head = val_heads.copy()
            min_per_head = val_heads.copy()
            best_epoch_per_head = np.full_like(val_heads, epoch + 1, dtype=int)
        else:
            improved = val_heads > best_per_head
            best_epoch_per_head[improved] = epoch + 1
            best_per_head = np.maximum(best_per_head, val_heads)
            sum_per_head += val_heads
            min_per_head = np.minimum(min_per_head, val_heads)
        mean_per_head = sum_per_head / count_epochs

        logger.info("[%5d] train: %.3f%%  val(max-head): %.3f%%" % (epoch + 1, train_acc_scalar, val_acc_scalar))
        if rank == 0:
            csv_logger.log(epoch + 1, train_acc_scalar, val_acc_scalar)

        if val_only:
            return

        save_checkpoint(
            epoch + 1,
            mean_val_acc_scalar,
            best_val_acc_scalar,
            val_heads,
            best_per_head,
            mean_per_head,
            min_per_head,
            best_epoch_per_head,
        )


def run_one_epoch(  
    device,  
    training,  
    encoder,  
    classifiers,  
    scaler,  
    optimizer,  
    scheduler,  
    wd_scheduler,  
    data_loader,  
    use_bfloat16,  
    use_focal_loss=False,  
    val_only=False,  # NEW  
    predictions_save_path=None,  # NEW  
):  
    for c in classifiers:  
        c.train(mode=training)  
  
    criterion = sigmoid_focal_loss if use_focal_loss else torch.nn.CrossEntropyLoss()  
    top1_meters = [AverageMeter() for _ in classifiers]  
      
    # NEW: Initialize prediction storage  
    all_predictions = []  
    all_video_paths = []  
    all_labels = []  
  
    for itr, data in enumerate(data_loader):  
        if training:  
            [s.step() for s in scheduler]  
            [wds.step() for wds in wd_scheduler]  
  
        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_bfloat16):  
            # Load data and put on GPU  
            clips = [  
                [dij.to(device, non_blocking=True) for dij in di]  # iterate over spatial views of clip  
                for di in data[0]  # iterate over temporal index of clip  
            ]  
            clip_indices = [d.to(device, non_blocking=True) for d in data[2]]  
            labels = data[1].to(device)  
            batch_size = len(labels)  
              
            # NEW: Extract video paths (data[3] if available, otherwise generate)  
            video_paths = data[3] if len(data) > 3 else [f"video_{itr}_{i}" for i in range(batch_size)]  
  
            # Forward and prediction  
            with torch.no_grad():  
                outputs = encoder(clips, clip_indices)  
                if not training:  
                    outputs = [[c(o) for o in outputs] for c in classifiers]  
            if training:  
                outputs = [[c(o) for o in outputs] for c in classifiers]  
  
        # Compute loss  
        losses = [[criterion(o, labels) for o in coutputs] for coutputs in outputs]  
        with torch.no_grad():  
            outputs = [sum([F.softmax(o, dim=1) for o in coutputs]) / len(coutputs) for coutputs in outputs]  
            top1_accs = [100.0 * coutputs.max(dim=1).indices.eq(labels).sum() / batch_size for coutputs in outputs]  
            top1_accs = [float(AllReduce.apply(t1a)) for t1a in top1_accs]  
            for t1m, t1a in zip(top1_meters, top1_accs):  
                t1m.update(t1a)  
                  
            # NEW: Save predictions and metadata when in val_only mode  
            if val_only and predictions_save_path is not None:  
                for i, pred in enumerate(outputs[0]):  # Use first classifier  
                    all_predictions.append(pred.cpu().numpy())  
                    all_video_paths.append(video_paths[i])  
                    all_labels.append(labels[i].cpu().numpy())  
  
        if training:  
            if use_bfloat16:  
                [[s.scale(lij).backward() for lij in li] for s, li in zip(scaler, losses)]  
                [s.step(o) for s, o in zip(scaler, optimizer)]  
                [s.update() for s in scaler]  
            else:  
                [[lij.backward() for lij in li] for li in losses]  
                [o.step() for o in optimizer]  
            [o.zero_grad() for o in optimizer]  
  
        _agg_top1 = np.array([t1m.avg for t1m in top1_meters])  
        if itr % 10 == 0:  
            logger.info(  
                "[%5d] %.3f%% [mean %.3f%% min %.3f%%] [mem: %.2e]"  
                % (  
                    itr,  
                    _agg_top1.max(),  
                    _agg_top1.mean(),  
                    _agg_top1.min(),  
                    torch.cuda.max_memory_allocated() / 1024.0**2,  
                )  
            )  
  
    # NEW: Save predictions to CSV after epoch  
    if val_only and predictions_save_path is not None and len(all_predictions) > 0:  
        import pandas as pd  
        import os  
          
        # Create directory if it doesn't exist  
        os.makedirs(os.path.dirname(predictions_save_path), exist_ok=True)  
          
        # Convert predictions to class labels and probabilities  
        pred_classes = [np.argmax(pred) for pred in all_predictions]  
        pred_probs = [pred.max() for pred in all_predictions]  
          
        # Create DataFrame  
        df = pd.DataFrame({  
            'video_path': all_video_paths,  
            'true_label': all_labels,  
            'predicted_class': pred_classes,  
            'prediction_confidence': pred_probs  
        })  
          
        # Save to CSV  
        df.to_csv(predictions_save_path, index=False)  
        logger.info(f"Saved {len(all_predictions)} predictions to {predictions_save_path}")  
  
    return float(_agg_top1.max()), _agg_top1


def load_checkpoint(device, r_path, classifiers, opt, scaler, val_only=False):
    checkpoint = robust_checkpoint_loader(r_path, map_location=torch.device("cpu"))
    logger.info(f"read-path: {r_path}")

    # -- loading classifier(s)
    pretrained_dict = checkpoint["classifiers"]
    msg = [c.load_state_dict(pd) for c, pd in zip(classifiers, pretrained_dict)]

    if val_only:
        # Log metrics if present (no change to return signature)
        if "best_val_acc" in checkpoint or "mean_val_acc" in checkpoint:
            logger.info(
                "loaded metrics: best_val_acc=%s mean_val_acc=%s",
                checkpoint.get("best_val_acc", "NA"),
                checkpoint.get("mean_val_acc", "NA"),
            )
        logger.info(f"loaded pretrained classifier (val_only) with msg: {msg}")
        return classifiers, opt, scaler, 0

    epoch = int(checkpoint["epoch"])
    logger.info(f"loaded pretrained classifier from epoch {epoch} with msg: {msg}")

    # -- optimizer
    [o.load_state_dict(pd) for o, pd in zip(opt, checkpoint["opt"])]

    # -- scaler (if used)
    if scaler is not None and "scaler" in checkpoint and checkpoint["scaler"] is not None:
        [s.load_state_dict(pd) for s, pd in zip(scaler, checkpoint["scaler"])]

    # Log metrics if present (keeps return arity identical)
    if "best_val_acc" in checkpoint or "mean_val_acc" in checkpoint:
        logger.info(
            "loaded metrics: best_val_acc=%s mean_val_acc=%s",
            checkpoint.get("best_val_acc", "NA"),
            checkpoint.get("mean_val_acc", "NA"),
        )

    logger.info(f"loaded optimizers from epoch {epoch}")
    return classifiers, opt, scaler, epoch


def load_pretrained(encoder, pretrained, checkpoint_key="target_encoder"):
    logger.info(f"Loading pretrained model from {pretrained}")
    checkpoint = robust_checkpoint_loader(pretrained, map_location="cpu")
    try:
        pretrained_dict = checkpoint[checkpoint_key]
    except Exception:
        pretrained_dict = checkpoint["encoder"]

    pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
    pretrained_dict = {k.replace("backbone.", ""): v for k, v in pretrained_dict.items()}
    for k, v in encoder.state_dict().items():
        if k not in pretrained_dict:
            logger.info(f"key '{k}' could not be found in loaded state dict")
        elif pretrained_dict[k].shape != v.shape:
            logger.info(f"{pretrained_dict[k].shape} | {v.shape}")
            logger.info(f"key '{k}' is of different shape in model and loaded state dict")
            exit(1)
            pretrained_dict[k] = v
    msg = encoder.load_state_dict(pretrained_dict, strict=False)
    print(encoder)
    logger.info(f"loaded pretrained model with msg: {msg}")
    logger.info(f"loaded pretrained encoder from epoch: {checkpoint['epoch']}\n path: {pretrained}")
    del checkpoint
    return encoder


DEFAULT_NORMALIZATION = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


def make_dataloader(
    root_path,
    batch_size,
    world_size,
    rank,
    dataset_type="VideoDataset",
    img_size=224,
    frames_per_clip=16,
    frame_step=4,
    num_segments=8,
    eval_duration=None,
    num_views_per_segment=1,
    allow_segment_overlap=True,
    training=False,
    num_workers=12,
    subset_file=None,
    normalization=None,
):
    if normalization is None:
        normalization = DEFAULT_NORMALIZATION

    # Make Video Transforms
    transform = make_transforms(
        training=training,
        num_views_per_clip=num_views_per_segment,
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(0.75, 4 / 3),
        random_resize_scale=(0.08, 1.0),
        reprob=0.25,
        auto_augment=True,
        motion_shift=False,
        crop_size=img_size,
        normalize=normalization,
    )

    data_loader, data_sampler = init_data(
        data=dataset_type,
        root_path=root_path,
        transform=transform,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        clip_len=frames_per_clip,
        frame_sample_rate=frame_step,
        duration=eval_duration,
        num_clips=num_segments,
        allow_clip_overlap=allow_segment_overlap,
        num_workers=num_workers,
        drop_last=False,
        subset_file=subset_file,
    )
    return data_loader, data_sampler


def init_opt(classifiers, iterations_per_epoch, opt_kwargs, num_epochs, use_bfloat16=False):
    optimizers, schedulers, wd_schedulers, scalers = [], [], [], []
    for c, kwargs in zip(classifiers, opt_kwargs):
        param_groups = [
            {
                "params": (p for n, p in c.named_parameters()),
                "mc_warmup_steps": int(kwargs.get("warmup") * iterations_per_epoch),
                "mc_start_lr": kwargs.get("start_lr"),
                "mc_ref_lr": kwargs.get("ref_lr"),
                "mc_final_lr": kwargs.get("final_lr"),
                "mc_ref_wd": kwargs.get("ref_wd"),
                "mc_final_wd": kwargs.get("final_wd"),
            }
        ]
        logger.info("Using AdamW")
        optimizers += [torch.optim.AdamW(param_groups)]
        schedulers += [WarmupCosineLRSchedule(optimizers[-1], T_max=int(num_epochs * iterations_per_epoch))]
        wd_schedulers += [CosineWDSchedule(optimizers[-1], T_max=int(num_epochs * iterations_per_epoch))]
        scalers += [torch.cuda.amp.GradScaler() if use_bfloat16 else None]
    return optimizers, scalers, schedulers, wd_schedulers


class WarmupCosineLRSchedule(object):
    def __init__(self, optimizer, T_max, last_epoch=-1):
        self.optimizer = optimizer
        self.T_max = T_max
        self._step = 0.0

    def step(self):
        self._step += 1
        for group in self.optimizer.param_groups:
            ref_lr = group.get("mc_ref_lr")
            final_lr = group.get("mc_final_lr")
            start_lr = group.get("mc_start_lr")
            warmup_steps = group.get("mc_warmup_steps")
            T_max = self.T_max - warmup_steps
            if self._step < warmup_steps:
                progress = float(self._step) / float(max(1, warmup_steps))
                new_lr = start_lr + progress * (ref_lr - start_lr)
            else:
                # -- progress after warmup
                progress = float(self._step - warmup_steps) / float(max(1, T_max))
                new_lr = max(
                    final_lr,
                    final_lr + (ref_lr - final_lr) * 0.5 * (1.0 + math.cos(math.pi * progress)),
                )
            group["lr"] = new_lr


class CosineWDSchedule(object):
    def __init__(self, optimizer, T_max):
        self.optimizer = optimizer
        self.T_max = T_max
        self._step = 0.0

    def step(self):
        self._step += 1
        progress = self._step / self.T_max

        for group in self.optimizer.param_groups:
            ref_wd = group.get("mc_ref_wd")
            final_wd = group.get("mc_final_wd")
            new_wd = final_wd + (ref_wd - final_wd) * 0.5 * (1.0 + math.cos(math.pi * progress))
            if final_wd <= ref_wd:
                new_wd = max(final_wd, new_wd)
            else:
                new_wd = min(final_wd, new_wd)
            group["weight_decay"] = new_wd
