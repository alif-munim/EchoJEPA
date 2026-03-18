# evals/video_classification_frozen/eval.py

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
import re

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from evals.video_classification_frozen.models import init_module
from evals.video_classification_frozen.utils import make_transforms
from src.datasets.data_manager import init_data
from src.models.attentive_pooler import AttentiveClassifier
from src.models.attentive_pooler import AttentiveRegressor
from src.models.linear_pooler import LinearClassifier, LinearRegressor
from src.models.linear_pooler import MLPClassifier, MLPRegressor

from src.utils.checkpoint_loader import robust_checkpoint_loader
from src.utils.distributed import AllReduce, init_distributed
from src.utils.logging import AverageMeter, CSVLogger

import os
import tempfile  # <-- ADD THIS

# Fix for "AF_UNIX path too long" error
short_tmp = "/tmp/vjepa_run"
os.makedirs(short_tmp, exist_ok=True)
tempfile.tempdir = short_tmp
os.environ["TMPDIR"] = short_tmp

# from evals.action_anticipation_frozen.losses import sigmoid_focal_loss

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

pp = pprint.PrettyPrinter(indent=4)

# --- INSERT THIS CLASS IN eval.py ---
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
# ------------------------------------


def _extract_study_id(path):
    """Extract study ID from video path (matches DistributedStudySampler logic)."""
    match = re.search(r"/s(\d+)/\d+_\d+\.mp4$", str(path))
    if match:
        return match.group(1)
    return os.path.basename(os.path.dirname(str(path)))


def _average_by_study(study_ids, predictions_np, labels_np):
    """Average predictions per study, return study-level arrays.

    Args:
        study_ids: list of str, one per clip
        predictions_np: ndarray [N, C] (classification probs) or [N] (regression)
        labels_np: ndarray [N] (all clips in a study share the same label)

    Returns:
        avg_preds, study_labels, num_studies
    """
    from collections import defaultdict

    study_groups = defaultdict(list)
    study_label_map = {}
    for i, sid in enumerate(study_ids):
        study_groups[sid].append(i)
        study_label_map[sid] = labels_np[i]
    sorted_studies = sorted(study_groups.keys())
    avg_preds = np.array([predictions_np[study_groups[sid]].mean(axis=0) for sid in sorted_studies])
    study_labels = np.array([study_label_map[sid] for sid in sorted_studies])
    clip_counts = np.array([len(study_groups[sid]) for sid in sorted_studies])
    return avg_preds, study_labels, len(sorted_studies), sorted_studies, clip_counts


def main(args_eval, resume_preempt=False):

    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    import os
    
    # Helper to safe-set nested keys with type conversion
    def set_override(env_var, target_dict, key, type_func=str):
        val = os.environ.get(env_var)
        if val is not None:
            # Handle boolean explicitly
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

    # --- NEW: Override Mean/Std for regression ---
    set_override("OVERRIDE_TARGET_MEAN", data, "target_mean", float)
    set_override("OVERRIDE_TARGET_STD", data, "target_std", float)

    # 4. Optimization parameters
    set_override("OVERRIDE_EPOCHS", opt, "num_epochs", int)
    set_override("OVERRIDE_FOCAL_LOSS", opt, "use_focal_loss", bool)
    set_override("OVERRIDE_BATCH", opt, "batch_size", int)  # <--- ADD THIS LINE
    # --- INSERT END ---

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
    num_probe_blocks = args_classifier.get("num_probe_blocks", 1)  # depth=1 default: cross-attention only (V-JEPA 1 protocol). depth=4 for V-JEPA 2 protocol (3 SA + 1 CA).
    num_heads = args_classifier.get("num_heads", 16)  # Must divide embed_dim of each model
    probe_checkpoint = args_eval.get("probe_checkpoint", None)

    # -- REGRESSION
    task_type = args_classifier.get("task_type", "classification")  # "classification" or "regression"  
    num_targets = args_classifier.get("num_targets", None)  # Only for regression
    probe_type = args_classifier.get("probe_type", "attentive")  # "attentive", "linear", or "mlp"
    use_layernorm = args_classifier.get("use_layernorm", True)
    probe_dropout = args_classifier.get("dropout", 0.0)

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
    study_sampling = args_data.get("study_sampling", False)
    class_balance_ratio = args_data.get("class_balance_ratio", None)
    prediction_averaging = args_data.get("prediction_averaging", False)

    # -- REGRESSION NORMALIZATION: auto-load from zscore_params.json, or compute from train CSV
    target_mean = args_data.get("target_mean", None)
    target_std = args_data.get("target_std", None)
    if task_type == "regression" and target_mean is None and target_std is None:
        import json as _json
        zscore_path = os.path.join(os.path.dirname(train_data_path[0]), "zscore_params.json")
        if os.path.exists(zscore_path):
            with open(zscore_path) as _f:
                _params = _json.load(_f)
            target_mean = _params["target_mean"]
            target_std = _params["target_std"]
            logger.info("Loaded zscore params from %s: mean=%.4f, std=%.4f", zscore_path, target_mean, target_std)
        elif val_only:
            raise RuntimeError(
                f"Regression inference requires zscore params but zscore_params.json not found at {zscore_path} "
                f"and target_mean/target_std not set in config. "
                f"Either place zscore_params.json alongside the train CSV or set target_mean/target_std in the YAML."
            )
        else:
            import pandas as _pd
            _train_df = _pd.read_csv(train_data_path[0], sep=" ", header=None)
            _labels = _train_df.iloc[:, -1].astype(float)
            target_mean = float(_labels.mean())
            target_std = float(_labels.std())
            logger.info(
                "Computed zscore params from train CSV: mean=%.4f, std=%.4f (n=%d)",
                target_mean, target_std, len(_labels),
            )
            # Save for reproducibility and future inference
            with open(zscore_path, "w") as _f:
                _json.dump({"target_mean": target_mean, "target_std": target_std}, _f)
            logger.info("Saved zscore params to %s", zscore_path)

    # -- OPTIMIZATION
    args_opt = args_exp.get("optimization")
    use_focal_loss = args_opt.get("use_focal_loss", False)

    batch_size = args_opt.get("batch_size")
    val_batch_size = args_opt.get("val_batch_size", batch_size)
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
        if task_type == "regression":
            csv_logger = CSVLogger(
                log_file, ("%d", "epoch"), ("%.5f", "train_mae"), ("%.5f", "val_mae"),
                ("%.5f", "val_r2"), ("%.5f", "val_pearson"),
            )
        else:  # classification
            csv_logger = CSVLogger(
                log_file, ("%d", "epoch"), ("%.5f", "train_acc"), ("%.5f", "val_acc"),
                ("%.5f", "val_auroc"), ("%.5f", "val_bal_acc"), ("%.5f", "val_kappa"),
            )

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

    # -- init classifier based on probe_type
    def make_probe(task, probe, embed_dim, num_classes, num_targets, num_heads, depth, use_ln, dropout):
        if task == "regression":
            if probe == "linear":
                return LinearRegressor(
                    embed_dim=embed_dim,
                    num_targets=num_targets,
                    use_layernorm=use_ln,
                    dropout=dropout,
                )
            elif probe == "mlp":
                return MLPRegressor(
                    embed_dim=embed_dim,
                    num_targets=num_targets,
                    use_layernorm=use_ln,
                    dropout=dropout,
                )
            else:  # attentive (default)
                return AttentiveRegressor(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    depth=depth,
                    num_targets=num_targets,
                    use_activation_checkpointing=True,
                )
        else:  # classification
            if probe == "linear":
                return LinearClassifier(
                    embed_dim=embed_dim,
                    num_classes=num_classes,
                    use_layernorm=use_ln,
                    dropout=dropout,
                )
            elif probe == "mlp":
                return MLPClassifier(
                    embed_dim=embed_dim,
                    num_classes=num_classes,
                    use_layernorm=use_ln,
                    dropout=dropout,
                )
            else:  # attentive (default)
                return AttentiveClassifier(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    depth=depth,
                    num_classes=num_classes,
                    use_activation_checkpointing=True,
                )

    classifiers = [
        make_probe(
            task=task_type,
            probe=probe_type,
            embed_dim=encoder.embed_dim,
            num_classes=num_classes,
            num_targets=num_targets,
            num_heads=num_heads,
            depth=num_probe_blocks,
            use_ln=use_layernorm,
            dropout=probe_dropout,
        ).to(device)
        for _ in opt_kwargs
    ]
    
    logger.info(f"Initialized {len(classifiers)} {probe_type} probes for {task_type}")

    
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
        study_sampling=study_sampling,
        class_balance_ratio=class_balance_ratio,
    )
    # Auto-enable prediction averaging for val_only with study-level tasks:
    # score ALL clips per study, then average predictions before computing metrics
    val_prediction_averaging = prediction_averaging or (val_only and study_sampling)
    val_study_sampling = False if val_prediction_averaging else study_sampling
    if val_prediction_averaging:
        logger.info("Prediction averaging enabled: scoring all clips per study, averaging at metric time")

    val_loader, val_sampler = make_dataloader(
        dataset_type=dataset_type,
        root_path=val_data_path,
        img_size=resolution,
        frames_per_clip=frames_per_clip,
        frame_step=frame_step,
        num_segments=num_segments,
        eval_duration=duration,
        num_views_per_segment=num_views_per_segment,
        allow_segment_overlap=True,
        batch_size=val_batch_size,
        world_size=world_size,
        rank=rank,
        training=False,
        num_workers=num_workers,
        normalization=normalization,
        study_sampling=val_study_sampling,
    )
    ipe = len(train_loader)
    logger.info(f"Dataloader created... iterations per epoch: {ipe}")
    if val_batch_size != batch_size:
        logger.info(f"Val batch size: {val_batch_size} (train: {batch_size})")

    # -- compute inverse-frequency class weights for classification
    class_weights = None
    if task_type == "classification" and not use_focal_loss:
        try:
            from collections import Counter
            label_counts = Counter()
            with open(train_data_path[0]) as _f:
                for line in _f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        label_counts[int(parts[-1])] += 1
            if label_counts:
                counts_arr = torch.tensor(
                    [label_counts.get(c, 1) for c in range(num_classes)], dtype=torch.float32
                )
                class_weights = 1.0 / counts_arr
                class_weights = class_weights / class_weights.sum() * num_classes
                logger.info(f"Class weights (inverse freq): {class_weights.tolist()}")
                logger.info(f"Class counts: {dict(sorted(label_counts.items()))}")
        except Exception as e:
            logger.warning(f"Failed to compute class weights: {e}. Using uniform weights.")
            class_weights = None

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
                        is_best=False):  # <--- ADD THIS PARAMETER
        
        all_classifier_dicts = [c.state_dict() for c in classifiers]
        all_opt_dicts = [o.state_dict() for o in optimizer]

        save_dict = {
            "classifiers": all_classifier_dicts,
            "opt": all_opt_dicts,
            "scaler": None if (scaler is None) else [None if s is None else s.state_dict() for s in scaler],
            "epoch": epoch,
            "batch_size": batch_size,
            "world_size": world_size,
            "mean_val_acc": float(mean_val_acc),
            "best_val_acc": float(best_val_acc),
            "val_acc_per_head": np.asarray(val_heads, dtype=float).tolist(),
            "best_val_acc_per_head": np.asarray(best_per_head, dtype=float).tolist(),
            "mean_val_acc_per_head": np.asarray(mean_per_head, dtype=float).tolist(),
            "min_val_acc_per_head": np.asarray(min_per_head, dtype=float).tolist(),
            "best_epoch_per_head": np.asarray(best_epoch_per_head, dtype=int).tolist(),
            "opt_grid": opt_kwargs,
        }
        
        if rank == 0:
            # 1. Always save latest
            _latest_path = os.path.join(folder, "latest.pt")
            torch.save(save_dict, _latest_path)

            # 2. Save per-epoch snapshot
            epoch_path = os.path.join(folder, f"epoch_{epoch:03d}.pt")
            torch.save(save_dict, epoch_path)

            # 3. Archive log_r0.csv to safe backup every epoch
            _archive_path = os.environ.get("CHECKPOINT_ARCHIVE_PATH")
            if _archive_path:
                try:
                    import shutil
                    os.makedirs(_archive_path, exist_ok=True)
                    _log_src = os.path.join(folder, "log_r0.csv")
                    if os.path.exists(_log_src):
                        shutil.copy2(_log_src, os.path.join(_archive_path, "log_r0.csv"))
                except Exception:
                    pass  # non-fatal

            # 4. Save BEST checkpoint
            if is_best:
                best_path = os.path.join(folder, "best.pt")
                torch.save(save_dict, best_path)
                logger.info(f"Generated new best model: {best_path}")

                # 5. Archive best.pt to safe backup directory
                if _archive_path:
                    try:
                        shutil.copy2(best_path, os.path.join(_archive_path, "best.pt"))
                        logger.info(f"Archived best.pt to: {_archive_path}/best.pt")
                    except Exception as _e:
                        logger.warning(f"Archive failed (non-fatal): {_e}")

    # ---- per-head running stats ----
    best_per_head = None
    sum_per_head = None
    min_per_head = None
    best_epoch_per_head = None
    count_epochs = 0
    
    # [FIX 3] Initialize Best Scalar based on Task
    if task_type == "regression":
        best_val_acc_scalar = float('inf')
    else:
        best_val_acc_scalar = 0.0

    # TRAIN LOOP
    val_cnt = 0
    val_sum_scalar = 0.0
    
    for epoch in range(start_epoch, num_epochs):
        logger.info("Epoch %d" % (epoch + 1))
        train_sampler.set_epoch(epoch)
        if val_sampler is not None and hasattr(val_sampler, 'set_epoch'):
            val_sampler.set_epoch(epoch)

        if val_only:
            train_acc_scalar, _ = -1.0, None
        else:
            train_acc_scalar, _, _ = run_one_epoch(
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
                task_type=task_type,
                target_mean=target_mean,
                target_std=target_std,
                class_weights=class_weights,
            )

        val_acc_scalar, val_heads, val_auc_metrics = run_one_epoch(
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
            target_mean=target_mean,
            target_std=target_std,
            class_weights=class_weights,
            prediction_averaging=val_prediction_averaging,
            output_dir=folder,
        )

        # ---- update scalar running stats ----
        val_cnt += 1
        val_sum_scalar += float(val_acc_scalar)
        mean_val_acc_scalar = val_sum_scalar / val_cnt
        
        # --- Logic for determining "Best" ---
        # Classification: select by AUROC (robust to class imbalance), fall back to accuracy
        # Regression: select by lowest MAE
        is_best = False
        if task_type == "regression":
            if float(val_acc_scalar) < best_val_acc_scalar:
                best_val_acc_scalar = float(val_acc_scalar)
                is_best = True
        else:
            # Use AUROC for best-model selection when available
            if val_auc_metrics is not None and "auroc" in val_auc_metrics:
                cur_auroc = float(np.nanmax(val_auc_metrics["auroc"]))
                if cur_auroc > best_val_acc_scalar:
                    best_val_acc_scalar = cur_auroc
                    is_best = True
            elif float(val_acc_scalar) > best_val_acc_scalar:
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
            # For per-head, we need similar conditional logic for "improved"
            if task_type == "regression":
                improved = val_heads < best_per_head # Lower is better
                best_per_head = np.minimum(best_per_head, val_heads)
            else:
                improved = val_heads > best_per_head # Higher is better
                best_per_head = np.maximum(best_per_head, val_heads)
                
            best_epoch_per_head[improved] = epoch + 1
            sum_per_head += val_heads
            min_per_head = np.minimum(min_per_head, val_heads)
            
        mean_per_head = sum_per_head / count_epochs

        # Log appropriate metric name
        if task_type == "regression":
            t_mean = target_mean if target_mean is not None else 0.0
            t_std = target_std if target_std is not None else 1.0
            train_rel = train_acc_scalar / t_mean * 100 if t_mean != 0 else 0.0
            val_rel = val_acc_scalar / t_mean * 100 if t_mean != 0 else 0.0
            best_rel = best_val_acc_scalar / t_mean * 100 if t_mean != 0 else 0.0
            # Baseline: predicting the mean gives MAE ≈ std * sqrt(2/pi) for normal data
            baseline_mae = t_std * 0.7979  # sqrt(2/pi)
            logger.info(
                "[%5d] train MAE: %.3f (%.1f%%)  val MAE: %.3f (%.1f%%)  best: %.3f (%.1f%%)  "
                "[mean=%.2f, std=%.2f, predict-mean baseline=%.3f]",
                epoch + 1,
                train_acc_scalar, train_rel,
                val_acc_scalar, val_rel,
                best_val_acc_scalar, best_rel,
                t_mean, t_std, baseline_mae,
            )
        else:
            logger.info("[%5d] train Acc: %.2f%%  val Acc(max-head): %.2f%%  Best AUROC: %.4f" % (
                epoch + 1, train_acc_scalar, val_acc_scalar, best_val_acc_scalar,
            ))

        # Log AUROC/AUPRC/balanced_acc/kappa or R²/Pearson
        if val_auc_metrics is not None and rank == 0:
            if "auroc" in val_auc_metrics:
                auroc = val_auc_metrics["auroc"]
                auprc = val_auc_metrics["auprc"]
                bal_acc = val_auc_metrics.get("balanced_acc")
                kappa = val_auc_metrics.get("kappa")
                if not np.all(np.isnan(auroc)):
                    best_auroc_idx = np.nanargmax(auroc)
                    auprc_str = ""
                    if not np.all(np.isnan(auprc)):
                        best_auprc_idx = np.nanargmax(auprc)
                        auprc_str = f"  val AUPRC: {auprc[best_auprc_idx]:.4f} (head {best_auprc_idx})"
                    logger.info(
                        "[%5d] val AUROC: %.4f (head %d)%s",
                        epoch + 1,
                        auroc[best_auroc_idx], best_auroc_idx, auprc_str,
                    )
                else:
                    logger.warning("[%5d] val AUROC: all NaN (all heads failed)", epoch + 1)
                if bal_acc is not None and not np.all(np.isnan(bal_acc)):
                    best_bal_idx = np.nanargmax(bal_acc)
                    best_kappa_idx = np.nanargmax(kappa) if not np.all(np.isnan(kappa)) else 0
                    logger.info(
                        "[%5d] val BalAcc: %.4f (head %d)  val Kappa: %.4f (head %d)",
                        epoch + 1,
                        bal_acc[best_bal_idx], best_bal_idx,
                        kappa[best_kappa_idx], best_kappa_idx,
                    )
            if "r2" in val_auc_metrics:
                r2 = val_auc_metrics["r2"]
                pearson = val_auc_metrics["pearson"]
                best_r2_idx = np.nanargmax(r2)
                best_pearson_idx = np.nanargmax(pearson)
                logger.info(
                    "[%5d] val R²: %.4f (head %d)  val Pearson: %.4f (head %d)",
                    epoch + 1,
                    r2[best_r2_idx], best_r2_idx,
                    pearson[best_pearson_idx], best_pearson_idx,
                )

        if rank == 0:
            if task_type == "regression" and val_auc_metrics is not None and "r2" in val_auc_metrics:
                best_r2 = float(np.nanmax(val_auc_metrics["r2"]))
                best_pearson = float(np.nanmax(val_auc_metrics["pearson"]))
                csv_logger.log(epoch + 1, train_acc_scalar, val_acc_scalar, best_r2, best_pearson)
            elif task_type == "regression":
                csv_logger.log(epoch + 1, train_acc_scalar, val_acc_scalar, float("nan"), float("nan"))
            elif val_auc_metrics is not None and "auroc" in val_auc_metrics:
                best_auroc = float(np.nanmax(val_auc_metrics["auroc"]))
                best_bal = float(np.nanmax(val_auc_metrics.get("balanced_acc", [np.nan])))
                best_kap = float(np.nanmax(val_auc_metrics.get("kappa", [np.nan])))
                csv_logger.log(epoch + 1, train_acc_scalar, val_acc_scalar, best_auroc, best_bal, best_kap)
            else:
                csv_logger.log(epoch + 1, train_acc_scalar, val_acc_scalar, float("nan"), float("nan"), float("nan"))

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
            is_best=is_best,  # <--- PASS THE FLAG HERE
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
    task_type="classification",  # "classification" or "regression"
    use_focal_loss=False,
    val_only=False,
    predictions_save_path=None,
    target_mean=None,
    target_std=None,
    class_weights=None,
    prediction_averaging=False,
    output_dir=None,
):
    # --- NEW: Import tqdm for progress bar ---
    from tqdm import tqdm
    
    for c in classifiers:
        c.train(mode=training)

    # Choose loss function based on task type
    if task_type == "regression":
        criterion = torch.nn.SmoothL1Loss()  # Huber Loss
        metric_meters = [AverageMeter() for _ in classifiers]
    else:  # classification
        if use_focal_loss:
            criterion = FocalLoss(alpha=1.0, gamma=2.0)
        elif class_weights is not None:
            criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device=device, dtype=torch.bfloat16 if use_bfloat16 else torch.float32))
        else:
            criterion = torch.nn.CrossEntropyLoss()
        top1_meters = [AverageMeter() for _ in classifiers]
    
    all_predictions = []
    all_video_paths = []
    all_labels = []

    # Per-head collection for AUROC/AUPRC (classification validation only)
    collect_for_auc = not training and task_type == "classification"
    all_head_probs = [[] for _ in classifiers] if collect_for_auc else None
    all_val_labels = [] if collect_for_auc else None

    # Per-head collection for R²/Pearson (regression validation only)
    collect_for_r2 = not training and task_type == "regression"
    all_head_preds = [[] for _ in classifiers] if collect_for_r2 else None
    all_reg_labels = [] if collect_for_r2 else None

    # Study ID collection for prediction averaging
    collect_study_ids = prediction_averaging and not training
    all_study_ids = [] if collect_study_ids else None
    study_ids_sorted = None
    study_clip_counts = None

    # --- NEW: Wrap loader in tqdm if val_only ---
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
        # with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_bfloat16):
            # Load data and put on GPU
            clips = [
                [dij.to(device, non_blocking=True) for dij in di]
                for di in data[0]
            ]
            clip_indices = [d.to(device, non_blocking=True) for d in data[2]]
            labels = data[1].to(device)
            batch_size = len(labels)
            
            video_paths = data[3] if len(data) > 3 else [f"video_{itr}_{i}" for i in range(batch_size)]

            # Forward and prediction
            with torch.no_grad():
                outputs = encoder(clips, clip_indices)
                if not training:
                    outputs = [[c(o) for o in outputs] for c in classifiers]
            if training:
                outputs = [[c(o) for o in outputs] for c in classifiers]

        # Compute loss with proper dtype handling
        if task_type == "regression":
            labels = labels.float()
            if labels.dim() == 1:
                labels = labels.unsqueeze(-1)
            # Z-score normalize labels at runtime (CSVs store raw values)
            t_mean = target_mean if target_mean is not None else 0.0
            t_std = target_std if target_std is not None else 1.0
            labels = (labels - t_mean) / t_std
            losses = [[criterion(o.float(), labels) for o in coutputs] for coutputs in outputs]
        else:
            losses = [[criterion(o, labels) for o in coutputs] for coutputs in outputs]
            
        # Compute metrics based on task type
        with torch.no_grad():
            if task_type == "regression":
                outputs = [sum([o for o in coutputs]) / len(coutputs) for coutputs in outputs]
                
                # 1. Calculate Normalized MAE (Standard Deviations)
                mae_errors = [F.l1_loss(o.squeeze().float(), labels.squeeze()) for o in outputs]
                mae_errors = [float(AllReduce.apply(mae)) for mae in mae_errors]
                
                # 2. Convert to Real MAE for LOGGING
                # --- CHANGE THIS BLOCK ---
                t_std = target_std if target_std is not None else 1.0
                mae_errors = [mae * t_std for mae in mae_errors]

                for meter, mae in zip(metric_meters, mae_errors):
                    meter.update(mae)

                # Collect per-head predictions for R²/Pearson
                if collect_for_r2:
                    for h, head_pred in enumerate(outputs):
                        all_head_preds[h].append(head_pred.reshape(-1).float().cpu())
                    all_reg_labels.append(labels.reshape(-1).cpu())

            else:  # classification
                outputs = [sum([F.softmax(o.float(), dim=1) for o in coutputs]) / len(coutputs) for coutputs in outputs]
                top1_accs = [100.0 * coutputs.max(dim=1).indices.eq(labels).sum() / batch_size for coutputs in outputs]
                top1_accs = [float(AllReduce.apply(t1a)) for t1a in top1_accs]
                for t1m, t1a in zip(top1_meters, top1_accs):
                    t1m.update(t1a)

                # Collect per-head softmax probs for AUROC/AUPRC
                if collect_for_auc:
                    for h, head_probs in enumerate(outputs):
                        all_head_probs[h].append(head_probs.float().cpu())
                    all_val_labels.append(labels.cpu())

            # Collect study IDs for prediction averaging
            if collect_study_ids:
                for vp in video_paths:
                    all_study_ids.append(_extract_study_id(str(vp)))

            if val_only and predictions_save_path is not None:
                for i, pred in enumerate(outputs[0]):
                    all_predictions.append(pred.float().cpu().numpy())  # Convert to float32 first
                    all_video_paths.append(video_paths[i])
                    all_labels.append(labels[i].float().cpu().numpy())  # Also convert labels

        if training:
            [[lij.backward() for lij in li] for li in losses]
            [o.step() for o in optimizer]
            [o.zero_grad() for o in optimizer]


        # Aggregate metrics for logging
        if task_type == "regression":
            _agg_metrics = np.array([m.avg for m in metric_meters])
            metric_name = "MAE" 
            metric_symbol = ""
        else:  # classification
            _agg_metrics = np.array([t1m.avg for t1m in top1_meters])
            metric_name = "Acc"
            metric_symbol = "%"
            
        # Only log to text log periodically (keep tqdm clean)
        if itr % 10 == 0:
            if val_only:
                # Update description dynamically with metrics
                if task_type == "regression":
                    iterator.set_description(f"Inf MAE: {_agg_metrics.min():.4f}")
                else:
                    iterator.set_description(f"Inf Acc: {_agg_metrics.max():.2f}%")
            else:
                best_scalar = float(_agg_metrics.min()) if task_type == "regression" else float(_agg_metrics.max())
                if task_type == "regression":
                    t_mean = target_mean if target_mean is not None else 0.0
                    rel_err = best_scalar / t_mean * 100 if t_mean != 0 else 0.0
                    phase = "val" if not training else "train"
                    msg = "[%5d] %s best-head MAE: %.3f (%.1f%% of mean=%.2f) [avg-head: %.3f] [mem: %.2e]" % (
                        itr, phase, best_scalar, rel_err, t_mean, _agg_metrics.mean(),
                        torch.cuda.max_memory_allocated() / 1024.0**2,
                    )
                else:
                    msg = "[%5d] %.3f%s [mean %.3f%s] [mem: %.2e]" % (
                        itr, best_scalar, metric_symbol, _agg_metrics.mean(), metric_symbol,
                        torch.cuda.max_memory_allocated() / 1024.0**2,
                    )
                logger.info(msg)


    # Save predictions (Un-normalized)
    if val_only and predictions_save_path is not None and len(all_predictions) > 0:
        import pandas as pd
        os.makedirs(os.path.dirname(predictions_save_path), exist_ok=True)
            
        if task_type == "regression":
            # For regression, save REAL values
            # 1. Get raw normalized predictions
            pred_values_norm = [pred[0] if len(pred.shape) > 0 else pred for pred in all_predictions]
            
            # 2. Un-normalize logic for CSV
            # --- CHANGE THIS BLOCK ---
            t_mean = target_mean if target_mean is not None else 0.0
            t_std = target_std if target_std is not None else 1.0
            
            # Convert arrays to scalars and un-normalize
            labels_real = []
            for l in all_labels:
                val = l[0] if isinstance(l, (np.ndarray, list)) else l
                labels_real.append((val * t_std) + t_mean)  # Use variables
                
            preds_real = []
            for p in pred_values_norm:
                val = p[0] if isinstance(p, (np.ndarray, list)) else p
                preds_real.append((val * t_std) + t_mean)   # Use variables

            df = pd.DataFrame({
                'video_path': all_video_paths,
                'label_real': labels_real,
                'pred_real': preds_real,
                'abs_error': [abs(a-b) for a,b in zip(labels_real, preds_real)]
            })
        else:  # classification
            pred_classes = [np.argmax(pred) for pred in all_predictions]
            pred_probs = [pred.max() for pred in all_predictions]
            df = pd.DataFrame({
                'video_path': all_video_paths,
                'true_label': all_labels,
                'predicted_class': pred_classes,
                'prediction_confidence': pred_probs
            })
            
        df.to_csv(predictions_save_path, index=False)
        logger.info(f"Saved {len(all_predictions)} predictions to {predictions_save_path}")

    scalar = float(_agg_metrics.min()) if task_type == "regression" else float(_agg_metrics.max())

    # Gather study IDs across ranks for prediction averaging
    gathered_study_ids = None
    if collect_study_ids and all_study_ids:
        import torch.distributed as dist
        if dist.is_initialized() and dist.get_world_size() > 1:
            all_study_ids_gathered = [None] * dist.get_world_size()
            dist.all_gather_object(all_study_ids_gathered, all_study_ids)
            gathered_study_ids = []
            for sids in all_study_ids_gathered:
                gathered_study_ids.extend(sids)
        else:
            gathered_study_ids = all_study_ids

    # Compute AUROC/AUPRC per head (classification validation only)
    auc_metrics = None
    if collect_for_auc and len(all_val_labels) > 0:
        try:
            import torch.distributed as dist
            from sklearn.metrics import average_precision_score, roc_auc_score

            local_labels = torch.cat(all_val_labels, dim=0)
            local_probs = [torch.cat(all_head_probs[h], dim=0) for h in range(len(classifiers))]

            # Gather across ranks (NCCL requires CUDA tensors)
            if dist.is_initialized() and dist.get_world_size() > 1:
                local_labels_gpu = local_labels.to(device)
                gathered_labels = [torch.zeros_like(local_labels_gpu) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_labels, local_labels_gpu)
                all_labels_t = torch.cat(gathered_labels, dim=0).cpu()
                gathered_probs = []
                for h in range(len(classifiers)):
                    local_h_gpu = local_probs[h].to(device)
                    gathered_h = [torch.zeros_like(local_h_gpu) for _ in range(dist.get_world_size())]
                    dist.all_gather(gathered_h, local_h_gpu)
                    gathered_probs.append(torch.cat(gathered_h, dim=0).cpu())
            else:
                all_labels_t = local_labels
                gathered_probs = local_probs

            from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score

            labels_np = all_labels_t.numpy().astype(int)

            # Prediction averaging: average probabilities per study
            if gathered_study_ids is not None:
                n_clips = len(gathered_study_ids)
                for h in range(len(classifiers)):
                    probs_np = gathered_probs[h].numpy()
                    avg_probs, avg_labels, n_studies, study_ids_sorted, study_clip_counts = _average_by_study(
                        gathered_study_ids, probs_np, labels_np)
                    gathered_probs[h] = torch.from_numpy(avg_probs)
                labels_np = avg_labels.astype(int)
                logger.info(f"Prediction averaging (classification): {n_clips} clips -> {n_studies} studies")

            num_classes_actual = gathered_probs[0].shape[1]
            auroc_arr = np.full(len(classifiers), np.nan)
            auprc_arr = np.full(len(classifiers), np.nan)
            bal_acc_arr = np.full(len(classifiers), np.nan)
            kappa_arr = np.full(len(classifiers), np.nan)

            unique_labels = np.unique(labels_np)
            expected_labels = list(range(num_classes_actual))
            if len(unique_labels) < num_classes_actual:
                logger.warning(
                    f"Val labels missing classes: got {unique_labels.tolist()}, expected {expected_labels}"
                )

            for h in range(len(classifiers)):
                probs_np = gathered_probs[h].numpy()
                preds_np = probs_np.argmax(axis=1)
                try:
                    if num_classes_actual == 2:
                        auroc_arr[h] = roc_auc_score(labels_np, probs_np[:, 1])
                        auprc_arr[h] = average_precision_score(labels_np, probs_np[:, 1])
                    else:
                        auroc_arr[h] = roc_auc_score(
                            labels_np, probs_np, multi_class="ovr", average="macro",
                            labels=expected_labels,
                        )
                except ValueError as e:
                    if h == 0:
                        logger.warning(f"AUROC failed for head {h}: {e}")
                bal_acc_arr[h] = balanced_accuracy_score(labels_np, preds_np)
                kappa_arr[h] = cohen_kappa_score(labels_np, preds_np)

            auc_metrics = {
                "auroc": auroc_arr,
                "auprc": auprc_arr,
                "balanced_acc": bal_acc_arr,
                "kappa": kappa_arr,
            }
        except Exception as e:
            logger.warning(f"AUROC/AUPRC computation failed: {e}")

    # Compute R²/Pearson per head (regression validation only)
    reg_metrics = None
    if collect_for_r2 and len(all_reg_labels) > 0:
        try:
            import torch.distributed as dist
            from scipy.stats import pearsonr

            local_labels = torch.cat(all_reg_labels, dim=0)
            local_preds = [torch.cat(all_head_preds[h], dim=0) for h in range(len(classifiers))]

            # Gather across ranks (NCCL requires CUDA tensors)
            if dist.is_initialized() and dist.get_world_size() > 1:
                local_labels_gpu = local_labels.to(device)
                gathered_labels = [torch.zeros_like(local_labels_gpu) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_labels, local_labels_gpu)
                all_labels_t = torch.cat(gathered_labels, dim=0).cpu()
                gathered_preds = []
                for h in range(len(classifiers)):
                    local_h_gpu = local_preds[h].to(device)
                    gathered_h = [torch.zeros_like(local_h_gpu) for _ in range(dist.get_world_size())]
                    dist.all_gather(gathered_h, local_h_gpu)
                    gathered_preds.append(torch.cat(gathered_h, dim=0).cpu())
            else:
                all_labels_t = local_labels
                gathered_preds = local_preds

            labels_np = all_labels_t.numpy()

            # Prediction averaging: average predictions per study
            if gathered_study_ids is not None:
                n_clips = len(gathered_study_ids)
                for h in range(len(classifiers)):
                    preds_np = gathered_preds[h].numpy()
                    avg_preds, avg_labels, n_studies, study_ids_sorted, study_clip_counts = _average_by_study(
                        gathered_study_ids, preds_np, labels_np)
                    gathered_preds[h] = torch.from_numpy(avg_preds)
                labels_np = avg_labels
                logger.info(f"Prediction averaging (regression): {n_clips} clips -> {n_studies} studies")

            r2_arr = np.full(len(classifiers), np.nan)
            pearson_arr = np.full(len(classifiers), np.nan)

            ss_tot = np.sum((labels_np - labels_np.mean()) ** 2)
            for h in range(len(classifiers)):
                preds_np = gathered_preds[h].numpy()
                ss_res = np.sum((labels_np - preds_np) ** 2)
                r2_arr[h] = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
                pearson_arr[h] = pearsonr(labels_np, preds_np)[0]

            reg_metrics = {"r2": r2_arr, "pearson": pearson_arr}

            # Save per-study predictions for best head (rank 0 only)
            if study_ids_sorted is not None and output_dir is not None:
                import torch.distributed as dist
                is_rank0 = not dist.is_initialized() or dist.get_rank() == 0
                if is_rank0:
                    best_h = int(np.nanargmax(r2_arr))
                    best_preds_z = gathered_preds[best_h].numpy()
                    t_mean = target_mean if target_mean is not None else 0.0
                    t_std = target_std if target_std is not None else 1.0
                    preds_real = best_preds_z * t_std + t_mean
                    labels_real = labels_np * t_std + t_mean
                    import pandas as pd
                    df = pd.DataFrame({
                        "study_id": study_ids_sorted,
                        "label": labels_real,
                        "prediction": preds_real,
                        "n_clips": study_clip_counts,
                    })
                    save_path = os.path.join(output_dir, "study_predictions.csv")
                    df.to_csv(save_path, index=False)
                    logger.info(f"Saved {len(df)} study-level predictions to {save_path} (best head {best_h}, R²={r2_arr[best_h]:.4f})")
        except Exception as e:
            logger.warning(f"R²/Pearson computation failed: {e}")

    # Save per-study predictions for classification (best head by AUROC, rank 0 only)
    if auc_metrics is not None and collect_study_ids and output_dir is not None:
        try:
            import torch.distributed as dist
            is_rank0 = not dist.is_initialized() or dist.get_rank() == 0
            if is_rank0 and study_ids_sorted is not None:
                best_h = int(np.nanargmax(auc_metrics["auroc"]))
                best_probs = gathered_probs[best_h].numpy()
                import pandas as pd
                df_data = {
                    "study_id": study_ids_sorted,
                    "label": labels_np,
                    "predicted_class": best_probs.argmax(axis=1),
                    "n_clips": study_clip_counts,
                }
                for c in range(best_probs.shape[1]):
                    df_data[f"prob_class_{c}"] = best_probs[:, c]
                df = pd.DataFrame(df_data)
                save_path = os.path.join(output_dir, "study_predictions.csv")
                df.to_csv(save_path, index=False)
                logger.info(f"Saved {len(df)} study-level predictions to {save_path} (best head {best_h}, AUROC={auc_metrics['auroc'][best_h]:.4f})")
        except Exception as e:
            logger.warning(f"Classification study prediction save failed: {e}")

    return scalar, _agg_metrics, auc_metrics or reg_metrics




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
        for s, sd in zip(scaler, checkpoint["scaler"]):
            if s is None or sd is None:
                continue
            s.load_state_dict(sd)

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
    study_sampling=False,
    class_balance_ratio=None,
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
        study_sampling=study_sampling,
        class_balance_ratio=class_balance_ratio,
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
        # scalers += [torch.cuda.amp.GradScaler() if use_bfloat16 else None]
        scalers += [None]
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
