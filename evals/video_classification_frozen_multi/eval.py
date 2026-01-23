# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
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
# --- NEW: Import Regressor ---
from src.models.attentive_pooler import AttentiveRegressor

from src.utils.checkpoint_loader import robust_checkpoint_loader
from src.utils.distributed import AllReduce, init_distributed
from src.utils.logging import AverageMeter, CSVLogger

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

pp = pprint.PrettyPrinter(indent=4)

# --- NEW: FocalLoss Class ---
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: [B, C] logits
        # targets: [B] class indices
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (self.alpha * (1 - pt) ** self.gamma * ce_loss)
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
# ----------------------------


def main(args_eval, resume_preempt=False):

    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    import os
    
    # --- NEW: Environment Variable Overrides ---
    def set_override(env_var, target_dict, key, type_func=str):
        val = os.environ.get(env_var)
        if val is not None:
            if type_func == bool:
                val = val.lower() in ('true', '1', 't', 'yes')
            else:
                val = type_func(val)
            print(f"!!! MANUAL OVERRIDE: {key} -> {val}")
            target_dict[key] = val

    # 1. Top-level parameters
    set_override("OVERRIDE_TAG", args_eval, "tag")
    set_override("OVERRIDE_VAL_ONLY", args_eval, "val_only", bool)
    set_override("OVERRIDE_PRED_PATH", args_eval, "predictions_save_path")
    set_override("OVERRIDE_CKPT", args_eval, "probe_checkpoint")

    # Ensure nested dictionaries exist
    exp = args_eval.setdefault("experiment", {})
    clf = exp.setdefault("classifier", {})
    data = exp.setdefault("data", {})
    opt = exp.setdefault("optimization", {})

    # 2. Classifier parameters
    set_override("OVERRIDE_NUM_HEADS", clf, "num_heads", int)
    set_override("OVERRIDE_NUM_BLOCKS", clf, "num_probe_blocks", int)

    # 3. Data parameters
    set_override("OVERRIDE_TRAIN_DATA", data, "dataset_train")
    set_override("OVERRIDE_VAL_DATA", data, "dataset_val")
    set_override("OVERRIDE_NUM_CLASSES", data, "num_classes", int)
    set_override("OVERRIDE_RES", data, "resolution", int)

    # 4. Optimization parameters
    set_override("OVERRIDE_EPOCHS", opt, "num_epochs", int)
    set_override("OVERRIDE_FOCAL_LOSS", opt, "use_focal_loss", bool)
    set_override("OVERRIDE_BATCH", opt, "batch_size", int)
    # -------------------------------------------

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
    
    # Check for manual checkpoint override
    probe_checkpoint = args_eval.get("probe_checkpoint", None)

    # -- REGRESSION SETTINGS
    task_type = args_classifier.get("task_type", "classification") 
    num_targets = args_classifier.get("num_targets", 1) # Default 1 for scalar regression
    
    use_slot_embeddings = args_classifier.get("use_slot_embeddings", False)
    use_factorized = args_classifier.get("use_factorized", True)

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
    
    num_clips_per_video = args_data.get("num_clips_per_video", 1)
    
    miss_augment_prob = args_data.get("miss_augment_prob", 0.0)
    min_present       = args_data.get("min_present", 1)
    
    num_views = args_data.get("num_segments", 1)
    clips_per_view = args_data.get("num_clips_per_video", 1)

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
    
    if probe_checkpoint is not None:
        latest_path = probe_checkpoint
    else:
        latest_path = os.path.join(folder, "latest.pt")

    # -- make csv_logger (Headers based on task)
    if rank == 0:
        if task_type == "regression":
            csv_logger = CSVLogger(log_file, ("%d", "epoch"), ("%.5f", "train_mae"), ("%.5f", "val_mae"))
        else:
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

    # -- init classifier (Switch between Regressor and Classifier)
    if task_type == "regression":
        classifiers = [
            AttentiveRegressor(
                embed_dim=encoder.embed_dim,
                num_heads=num_heads,
                depth=num_probe_blocks,
                num_targets=num_targets, # Usually 1
                use_activation_checkpointing=True,
                # use_slot_embeddings=use_slot_embeddings, # Verify if Regressor supports this, usually yes
            ).to(device)
            for _ in opt_kwargs
        ]
    else:
        classifiers = [
            AttentiveClassifier(
                embed_dim=encoder.embed_dim,
                num_heads=num_heads,
                depth=num_probe_blocks,
                num_classes=num_classes,
                use_activation_checkpointing=True,
                use_slot_embeddings=use_slot_embeddings,
                num_views=num_views,
                clips_per_view=clips_per_view,
                use_factorized=use_factorized,
            ).to(device)
            for _ in opt_kwargs
        ]
    
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
        num_clips_per_video=num_clips_per_video, 
        allow_segment_overlap=True,  
        batch_size=batch_size,  
        world_size=world_size,  
        rank=rank,  
        training=True,  
        num_workers=num_workers,  
        normalization=normalization,
        miss_augment_prob=miss_augment_prob,
        min_present=min_present,
        split_name="train"
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
        num_clips_per_video=num_clips_per_video, 
        allow_segment_overlap=True,  
        batch_size=batch_size,  
        world_size=world_size,  
        rank=rank,  
        training=False,  
        num_workers=num_workers,  
        normalization=normalization,
        miss_augment_prob=0.0,
        min_present=min_present,
        split_name="val"
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
                        val_heads, best_per_head, mean_per_head, min_per_head, best_epoch_per_head,
                        is_best=False):
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
            
            # --- NEW: Save Best ---
            if is_best:
                best_path = os.path.join(folder, "best.pt")
                torch.save(save_dict, best_path)
                logger.info(f"Generated new best model: {best_path}")

    # TRAIN LOOP
    # Initialize "Best" metric based on task type
    if task_type == "regression":
        best_val_acc_scalar = float('inf') # Lower is better for MAE
    else:
        best_val_acc_scalar = 0.0 # Higher is better for Acc

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
                task_type=task_type,
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
            task_type=task_type,
        )

        # ---- update scalar running stats (max/min over heads) ----
        val_cnt += 1
        val_sum_scalar += float(val_acc_scalar)
        mean_val_acc_scalar = val_sum_scalar / val_cnt
        
        # --- Best Model Logic ---
        is_best = False
        if task_type == "regression":
            if float(val_acc_scalar) < best_val_acc_scalar:
                best_val_acc_scalar = float(val_acc_scalar)
                is_best = True
        else:
            if float(val_acc_scalar) > best_val_acc_scalar:
                best_val_acc_scalar = float(val_acc_scalar)
                is_best = True

        # ---- update per-head running stats ----
        count_epochs += 1
        if best_per_head is None:
            best_per_head = val_heads.copy()
            sum_per_head = val_heads.copy()
            min_per_head = val_heads.copy()
            best_epoch_per_head = np.full_like(val_heads, epoch + 1, dtype=int)
        else:
            if task_type == "regression":
                improved = val_heads < best_per_head
                best_per_head = np.minimum(best_per_head, val_heads)
            else:
                improved = val_heads > best_per_head
                best_per_head = np.maximum(best_per_head, val_heads)
                
            best_epoch_per_head[improved] = epoch + 1
            sum_per_head += val_heads
            min_per_head = np.minimum(min_per_head, val_heads)
        
        mean_per_head = sum_per_head / count_epochs

        # Logging String
        metric_symbol = "" if task_type == "regression" else "%"
        val_prefix = "val(min-head)" if task_type == "regression" else "val(max-head)"
        
        logger.info("[%5d] train: %.3f%s  %s: %.3f%s (Best: %.3f%s)" % (
            epoch + 1, train_acc_scalar, metric_symbol, 
            val_prefix, val_acc_scalar, metric_symbol, 
            best_val_acc_scalar, metric_symbol
        ))
        
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
            is_best=is_best,
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
    task_type="classification",
    use_focal_loss=False,
    val_only=False,
    predictions_save_path=None,
):
    from tqdm import tqdm
    
    for c in classifiers:
        c.train(mode=training)

    # --- UPDATED: Loss Selection ---
    if task_type == "regression":
        # Using SmoothL1 (Huber) for robustness, or MSE
        criterion = torch.nn.SmoothL1Loss()
        metric_meters = [AverageMeter() for _ in classifiers]
    else:
        if use_focal_loss:
            # Use the local class we defined
            criterion = FocalLoss(alpha=1.0, gamma=2.0)
        else:
            criterion = torch.nn.CrossEntropyLoss()
        top1_meters = [AverageMeter() for _ in classifiers]

    all_predictions = []
    all_video_paths = []
    all_labels = []

    # Wrap iterator for validation to see progress
    if val_only:
        iterator = tqdm(data_loader, desc="Inference", unit="batch", dynamic_ncols=True)
    else:
        iterator = data_loader

    from torch.amp import autocast
    for itr, data in enumerate(iterator):
        if training:
            [s.step() for s in scheduler]
            [wds.step() for wds in wd_scheduler]

        with autocast("cuda", dtype=torch.bfloat16, enabled=use_bfloat16):
            # Load batch
            clips = [
                [dij.to(device, non_blocking=True) for dij in di]
                for di in data[0]
            ]
            clip_indices = [d.to(device, non_blocking=True) for d in data[2]]
            labels = data[1].to(device)
            batch_size = len(labels)
            
            # Paths handling (if available)
            video_paths = data[3] if len(data) > 3 else [f"vid_{itr}_{i}" for i in range(batch_size)]

            # Forward
            with torch.no_grad():
                outputs = encoder(clips, clip_indices)
            
            # Classifier Forward
            if training:
                outputs = [[c(o) for o in outputs] for c in classifiers]
            else:
                with torch.no_grad():
                    outputs = [[c(o) for o in outputs] for c in classifiers]

        # --- Loss Calculation ---
        if task_type == "regression":
            # Ensure Float and dimensions match
            labels = labels.float()
            if labels.dim() == 1:
                labels = labels.unsqueeze(-1)
            losses = [[criterion(o.float(), labels) for o in coutputs] for coutputs in outputs]
        else:
            losses = [[criterion(o, labels) for o in coutputs] for coutputs in outputs]

        # --- Metrics & Accumulation ---
        with torch.no_grad():
            if task_type == "regression":
                # Average across spatial views
                preds = [sum([o for o in coutputs]) / len(coutputs) for coutputs in outputs]
                
                # Metric: L1 Loss (MAE)
                mae_errors = [F.l1_loss(p.float(), labels.float()) for p in preds]
                mae_errors = [float(AllReduce.apply(mae)) for mae in mae_errors]
                
                # Convert normalized MAE to real scale (optional logging adjustment)
                # LVEF_TRAIN_STD = 11.33 
                # mae_errors = [mae * LVEF_TRAIN_STD for mae in mae_errors]

                for meter, mae in zip(metric_meters, mae_errors):
                    meter.update(mae)
            else:
                # Classification
                preds = [sum([F.softmax(o, dim=1) for o in coutputs]) / len(coutputs) for coutputs in outputs]
                top1_accs = [100.0 * p.max(dim=1).indices.eq(labels).float().mean() for p in preds]
                top1_accs = [float(AllReduce.apply(t1a)) for t1a in top1_accs]
                for meter, acc in zip(top1_meters, top1_accs):
                    meter.update(acc)
            
            # Store predictions for CSV saving (val_only)
            if val_only and predictions_save_path is not None:
                # Average spatial views for storage
                final_preds = [sum(coutputs)/len(coutputs) for coutputs in outputs]
                # Just take the first head's prediction for simplicity in CSV 
                # (or average heads if you prefer, but usually we analyze specific heads)
                best_head_pred = final_preds[0] 
                
                all_predictions.extend(best_head_pred.float().cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_video_paths.extend(video_paths)

        if training:
            # Scaler step
            # Note: Your reference removed scaler usage for bfloat16, using standard backward
            # but usually scaler is safe to keep. If reference explicitly removes it:
            if use_bfloat16 and scaler[0] is not None:
                 [[s.scale(lij).backward() for lij in li] for s, li in zip(scaler, losses)]
                 [s.step(o) for s, o in zip(scaler, optimizer)]
                 [s.update() for s in scaler]
            else:
                [[lij.backward() for lij in li] for li in losses]
                [o.step() for o in optimizer]
            
            [o.zero_grad() for o in optimizer]

        # --- Logging ---
        if task_type == "regression":
            _agg_metrics = np.array([m.avg for m in metric_meters])
            metric_symbol = ""
        else:
            _agg_metrics = np.array([m.avg for m in top1_meters])
            metric_symbol = "%"

        if itr % 10 == 0:
            if val_only:
                val_metric = _agg_metrics.min() if task_type == "regression" else _agg_metrics.max()
                desc_label = "MAE" if task_type == "regression" else "Acc"
                iterator.set_description(f"Inf {desc_label}: {val_metric:.4f}")
            else:
                best_scalar = float(_agg_metrics.min()) if task_type == "regression" else float(_agg_metrics.max())
                logger.info(
                    "[%5d] %.3f%s [mean %.3f%s] [mem: %.2e]"
                    % (
                        itr,
                        best_scalar, metric_symbol,
                        _agg_metrics.mean(), metric_symbol,
                        torch.cuda.max_memory_allocated() / 1024.0**2,
                    )
                )

    # --- Save CSV at end of val epoch ---
    if val_only and predictions_save_path is not None and len(all_predictions) > 0:
        import pandas as pd
        os.makedirs(os.path.dirname(predictions_save_path), exist_ok=True)
        
        #Flatten inputs
        flat_preds = [p.item() if hasattr(p, 'item') else p for p in all_predictions]
        flat_labels = [l.item() if hasattr(l, 'item') else l for l in all_labels]
        flat_paths = [str(p) for p in all_video_paths]

        df = pd.DataFrame({
            'video_path': flat_paths,
            'label': flat_labels,
            'prediction': flat_preds
        })
        df.to_csv(predictions_save_path, index=False)
        logger.info(f"Saved predictions to {predictions_save_path}")

    # Return scalar for checkpoint saving logic
    if task_type == "regression":
        return float(_agg_metrics.min()), _agg_metrics
    else:
        return float(_agg_metrics.max()), _agg_metrics



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
    num_clips_per_video=1,  # NEW parameter  
    eval_duration=None,
    num_views_per_segment=1,
    allow_segment_overlap=True,
    training=False,
    num_workers=12,
    subset_file=None,
    normalization=None,
    miss_augment_prob=0.0,
    min_present=1,
    split_name="train"
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

    # data_loader, dist_sampler = init_data(  
    #     data=dataset_type,  
    #     root_path=root_path,  
    #     batch_size=batch_size,  
    #     clip_len=frames_per_clip,  
    #     frame_sample_rate=frame_step,  
    #     duration=eval_duration,  
    #     num_clips=num_segments,  
    #     num_clips_per_video=num_clips_per_video,  # NEW: pass through  
    #     allow_clip_overlap=allow_segment_overlap,  
    #     transform=transform,  
    #     # shared_transform=shared_transform,  
    #     collator=collator,  
    #     num_workers=num_workers,  
    #     world_size=world_size,  
    #     rank=rank,  
    #     training=training,  
    #     pin_mem=True,  
    #     persistent_workers=True,  
    # )  

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
        num_clips_per_video=num_clips_per_video,  # NEW: pass through  
        allow_clip_overlap=allow_segment_overlap,
        num_workers=num_workers,
        drop_last=False,
        subset_file=subset_file,
        img_size=img_size,
        training=training,
        miss_augment_prob=miss_augment_prob,
        min_present=min_present,
        split_name=split_name
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
