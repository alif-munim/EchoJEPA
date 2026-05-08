"""Full-joint Global Study-Token EchoMV-JEPA training entry point.

Trains, end-to-end:
  * ``f_theta``    V-JEPA clip encoder (trainable, layer-wise LR decay)
  * ``fbar_theta`` EMA teacher of ``f_theta``
  * ``f0``         frozen e100 anchor encoder
  * ``F_psi``      student study transformer
  * ``Fbar_psi``   EMA teacher study transformer
  * ``p_study``    study projector (student + EMA teacher)
  * ``clip_pred``  V-JEPA predictor (trainable)

Loss:
  L = λ_clip · L_clip_vjepa
    + λ_study · L_global_study_jepa
    + λ_nce · L_study_InfoNCE
    + λ_cov · L_cov
    + λ_anchor · L_anchor_to_e100
    + λ_sv · L_single_view_to_study

This module owns the distributed training loop and checkpointing. Model
construction lives in ``src/models/echomv_jepa/full_joint_model.py`` and
loss math in ``src/models/echomv_jepa/full_joint_losses.py``.
"""

from __future__ import annotations

import math
import os
import time
from typing import Any, Dict, List

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from app.vjepa.transforms import make_transforms
from src.datasets.echomv_jepa_pixel_dataset import (
    EchoMVJEPAPixelDataset,
    echomv_pixel_collate,
)
from src.models.echomv_jepa.clip_ema import (
    assert_no_grad,
    clip_ema_schedule,
    ema_delta_norm,
    step_clip_ema,
)
from src.models.echomv_jepa.full_joint_clip_backbone import layerwise_param_groups
from src.models.echomv_jepa.cross_rank_study_nce import cross_rank_study_nce_loss
from src.models.echomv_jepa.clip_subset_sampler import sample_clip_subset
from src.models.echomv_jepa.full_joint_losses import (
    LossDecay,
    LossRamp,
    LossWeights,
    anchor_cosine_to_e100,
    anchor_loss,
    assemble_total_loss,
    clip_vjepa_true_loss,
)
from src.models.echomv_jepa.layerwise_drift import compute_clip_cov_var, compute_layerwise_cosine
from src.models.echomv_jepa.single_view_branch import sample_single_view_rows
from src.masks.multiseq_multiblock3d import MaskCollator
from src.models.echomv_jepa.full_joint_model import (
    FullJointConfig,
    build_full_joint_model,
)
from src.models.echomv_jepa.losses import (
    covariance_penalty,
    layernorm_cosine,
    matched_nce,
)
from src.models.echomv_jepa.study_corruption import apply_study_corruption
from src.utils.distributed import init_distributed
from src.utils.logging import CSVLogger, get_logger

log = get_logger(__name__, force=True)


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #


def _spatial_pool_tokens(tokens: torch.Tensor, spatial_hw: int, pool: int) -> torch.Tensor:
    """Pool the spatial axis of a (N, T_tubelets * H' * W', D) token grid.

    Assumes tokens are flat patch grid. For ViT-L/16 at 224×16f: 8 tubelets ×
    14×14 patches = 1568 tokens; pool=2 → 8 × 7 × 7 = 392 tokens.
    """
    N, total, D = tokens.shape
    temporal = total // (spatial_hw * spatial_hw)
    tokens = tokens.reshape(N, temporal, spatial_hw, spatial_hw, D)
    if pool > 1:
        tokens = tokens.reshape(N, temporal, spatial_hw // pool, pool, spatial_hw // pool, pool, D)
        tokens = tokens.mean(dim=(3, 5))
    return tokens.reshape(N, -1, D)


def _flatten_studies_to_clips(
    clips_bm: torch.Tensor,  # (B, M, 3, T, H, W)
    pad_mask: torch.Tensor,  # (B, M) bool
) -> tuple:
    """Flatten the (B, M) clip grid to a list of valid (N_valid, 3, T, H, W)
    clips + bookkeeping indices. Padded positions are dropped to avoid
    wasting encoder compute.
    """
    B, M, C, T, H, W = clips_bm.shape
    valid = (~pad_mask).reshape(B * M)
    clips_flat = clips_bm.reshape(B * M, C, T, H, W)
    clips_valid = clips_flat[valid]
    return clips_valid, valid, B, M


def _unflatten_tokens_to_studies(
    tokens_valid: torch.Tensor,  # (N_valid, T_tok, D)
    valid: torch.Tensor,  # (B*M,) bool
    B: int,
    M: int,
) -> torch.Tensor:
    """Inverse of ``_flatten_studies_to_clips``; pads invalid positions
    with zeros and returns ``(B, M, T_tok, D)``."""
    T_tok = tokens_valid.shape[1]
    D = tokens_valid.shape[2]
    out = torch.zeros(B * M, T_tok, D, device=tokens_valid.device, dtype=tokens_valid.dtype)
    out[valid] = tokens_valid
    return out.reshape(B, M, T_tok, D)


def _single_view_subset(full_pad: torch.Tensor, full_view: torch.Tensor) -> torch.Tensor:
    """Build a mask where only clips of one randomly-chosen view family per
    row remain unpadded. Returns a new (B, M_full) pad mask.
    """
    B, _ = full_pad.shape
    new_pad = torch.ones_like(full_pad)
    for b in range(B):
        unpad = (~full_pad[b]).nonzero(as_tuple=False).squeeze(-1)
        if unpad.numel() == 0:
            continue
        views = full_view[b, unpad].unique()
        if views.numel() == 0:
            continue
        k = int(torch.randint(0, views.numel(), (1,)).item())
        pick = views[k]
        keep = unpad[full_view[b, unpad] == pick]
        new_pad[b, keep] = False
    return new_pad


# ---------------------------------------------------------------------- #
# Main
# ---------------------------------------------------------------------- #


def main(args=None, resume_preempt: bool = False) -> None:
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_LOCALID"]
    except KeyError:
        pass

    cfg = args if isinstance(args, dict) else {}
    exp = cfg.get("experiment", {})
    clip_enc_cfg = exp.get("clip_encoder", {})
    data_cfg = exp.get("data", {})
    optim_cfg = exp.get("optim", {})
    ema_cfg = exp.get("ema", {})
    loss_cfg = exp.get("loss", {})
    lambdas_cfg = exp.get("lambdas", {})
    corruption_cfg = exp.get("study_corruption", {})
    sv_cfg = exp.get("single_view_branch", {})
    logging_cfg = exp.get("logging", {})

    # ---- Distributed init ----
    world_size, rank = init_distributed()
    device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    log.info(f"full_joint: rank={rank}/{world_size} device={device}")

    # ---- Model ----
    fj_cfg = FullJointConfig(
        ckpt_path=str(clip_enc_cfg.get("init_ckpt")),
        model_name=str(clip_enc_cfg.get("model_name", "vit_large")),
        crop_size=int(data_cfg.get("crop_size", 224)),
        max_num_frames=int(data_cfg.get("frames_per_clip", 16)),
        tubelet_size=int(data_cfg.get("tubelet_size", 2)),
        patch_size=int(data_cfg.get("patch_size", 16)),
        pred_depth=int(clip_enc_cfg.get("pred_depth", 12)),
        pred_embed_dim=int(clip_enc_cfg.get("pred_embed_dim", 384)),
        pred_num_heads=int(clip_enc_cfg.get("pred_num_heads", 12)),
        use_mask_tokens=bool(clip_enc_cfg.get("use_mask_tokens", True)),
        num_mask_tokens=int(clip_enc_cfg.get("num_mask_tokens", 10)),
        use_rope=bool(clip_enc_cfg.get("use_rope", True)),
        use_sdpa=bool(clip_enc_cfg.get("use_sdpa", True)),
        use_activation_checkpointing=bool(clip_enc_cfg.get("use_activation_checkpointing", True)),
        uniform_power=bool(clip_enc_cfg.get("uniform_power", True)),
        zero_init_mask_tokens=bool(clip_enc_cfg.get("zero_init_mask_tokens", True)),
        d_model=int(exp.get("study_transformer", {}).get("d_model", 512)),
        study_depth=int(exp.get("study_transformer", {}).get("depth", 4)),
        study_num_heads=int(exp.get("study_transformer", {}).get("num_heads", 8)),
        projector_hidden=int(exp.get("projector", {}).get("hidden", 1024)),
        projector_out=int(exp.get("projector", {}).get("out", 256)),
        token_spatial_pool=int(clip_enc_cfg.get("token_spatial_pool", 2)),
    )
    model = build_full_joint_model(fj_cfg, device)
    # Sanity assertions — teacher / anchor are frozen.
    assert_no_grad(model.target_encoder, "clip target_encoder")
    assert_no_grad(model.anchor, "clip anchor f0")
    assert_no_grad(model.teacher_st_ema, "study teacher")

    # Wrap trainable components in DDP.
    if world_size > 1:
        model.encoder = DistributedDataParallel(model.encoder, static_graph=True)
        model.predictor = DistributedDataParallel(model.predictor, static_graph=True)
        model.student_st = DistributedDataParallel(model.student_st, static_graph=True)
        model.projector = DistributedDataParallel(model.projector, static_graph=True)
    # The token wrapper holds a reference to the unwrapped student_st; update it.
    model.token_student.st = (
        model.student_st.module if isinstance(model.student_st, DistributedDataParallel) else model.student_st
    )

    # ---- Optimizer ----
    base_lr = float(optim_cfg.get("base_lr", 2e-4))
    clip_base_lr = float(optim_cfg.get("clip_base_lr", 3e-5))
    weight_decay = float(optim_cfg.get("weight_decay", 0.04))

    clip_groups = layerwise_param_groups(
        model.encoder.module if isinstance(model.encoder, DistributedDataParallel) else model.encoder,
        model.predictor.module if isinstance(model.predictor, DistributedDataParallel) else model.predictor,
        base_lr=clip_base_lr,
        weight_decay=weight_decay,
        min_scale=float(optim_cfg.get("clip_min_scale", 0.1)),
        mid_scale=float(optim_cfg.get("clip_mid_scale", 0.3)),
        top_scale=float(optim_cfg.get("clip_top_scale", 1.0)),
    )
    study_st_unwrapped = (
        model.student_st.module if isinstance(model.student_st, DistributedDataParallel) else model.student_st
    )
    proj_unwrapped = (
        model.projector.module if isinstance(model.projector, DistributedDataParallel) else model.projector
    )
    study_groups: List[Dict[str, Any]] = [
        {
            "params": [p for p in study_st_unwrapped.parameters() if p.requires_grad],
            "lr": base_lr,
            "weight_decay": weight_decay,
            "lr_scale": 1.0,
        },
        {
            "params": [p for p in proj_unwrapped.student.parameters() if p.requires_grad],
            "lr": base_lr,
            "weight_decay": weight_decay,
            "lr_scale": 1.0,
        },
        {
            "params": [p for p in model.meta.parameters() if p.requires_grad],
            "lr": base_lr,
            "weight_decay": 0.0,
            "lr_scale": 1.0,
        },
    ]
    optimizer = torch.optim.AdamW(clip_groups + study_groups, lr=base_lr, betas=(0.9, 0.95))
    log.info(
        f"optimizer: {len(clip_groups)} clip groups + {len(study_groups)} study groups; "
        f"base_lr={base_lr} clip_base_lr={clip_base_lr}"
    )

    # ---- Data ----
    manifest_path = str(data_cfg.get("k_sample_manifest"))
    from src.models.meta_embeddings import MetaEmbeddings  # type: ignore

    dataset = EchoMVJEPAPixelDataset(
        k_sample_manifest_path=manifest_path,
        meta=MetaEmbeddings(d_model=fj_cfg.d_model),
        frames_per_clip=fj_cfg.max_num_frames,
        frame_step=int(data_cfg.get("frame_step", 2)),
        resolution=fj_cfg.crop_size,
        transform=make_transforms(
            random_horizontal_flip=False,
            random_resize_aspect_ratio=(0.9, 1.1),
            random_resize_scale=(0.5, 1.0),
            reprob=0.0,
            auto_augment=False,
            motion_shift=False,
            crop_size=fj_cfg.crop_size,
        ),
        strategy_weights=exp.get("element_strategy_weights"),
        seed=int(exp.get("seed", 0)),
        permute_every_step=True,
    )

    batch_size = int(optim_cfg.get("batch_studies_per_gpu", 4))
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=int(data_cfg.get("num_workers", 4)),
        pin_memory=True,
        collate_fn=echomv_pixel_collate,
        drop_last=True,
        persistent_workers=False,
    )

    # ---- Training knobs ----
    total_steps = int(optim_cfg.get("total_steps", 225))
    save_steps = set(int(s) for s in optim_cfg.get("save_at_steps", []))
    log_every = int(logging_cfg.get("log_every", 10))
    anchor_subsample = int(loss_cfg.get("anchor_subsample", 4))
    # dtype
    dtype_str = str(optim_cfg.get("dtype", "bfloat16"))
    dtype = torch.bfloat16 if dtype_str == "bfloat16" else (torch.float16 if dtype_str == "float16" else torch.float32)
    mixed_precision = dtype != torch.float32

    weights = LossWeights(
        lambda_clip=float(lambdas_cfg.get("clip", 1.0)),
        lambda_clip_vjepa_true=float(lambdas_cfg.get("clip_vjepa_true", 1.0)),
        lambda_clip_consistency=float(lambdas_cfg.get("clip_consistency", 0.1)),
        lambda_study=float(lambdas_cfg.get("study", 0.1)),
        lambda_nce=float(lambdas_cfg.get("nce", 0.005)),
        lambda_cov=float(lambdas_cfg.get("cov", 0.001)),
        lambda_anchor=float(lambdas_cfg.get("anchor", 0.05)),
        lambda_sv=float(lambdas_cfg.get("sv", 0.02)),
    )
    study_ramp = LossRamp(
        target_weight=float(lambdas_cfg.get("study_target", weights.lambda_study)),
        warmup_steps=int(lambdas_cfg.get("study_warmup_steps", 0)),
    )
    sv_ramp = LossRamp(
        target_weight=float(lambdas_cfg.get("sv_target", weights.lambda_sv)),
        warmup_steps=int(lambdas_cfg.get("sv_warmup_steps", 0)),
    )
    anchor_decay = LossDecay(
        start_weight=float(lambdas_cfg.get("anchor_start", weights.lambda_anchor)),
        final_weight=float(lambdas_cfg.get("anchor_final", weights.lambda_anchor)),
        decay_steps=int(lambdas_cfg.get("anchor_decay_steps", 0)),
        schedule=str(lambdas_cfg.get("anchor_schedule", "cosine")),
    )

    # Single-view branch config
    sv_p_rows = float(sv_cfg.get("p_rows", 0.25))
    sv_prefer = list(
        sv_cfg.get("prefer_view_families", ["apical", "parasternal_long", "parasternal_short", "subcostal"])
    )
    sv_min_valid_rows = int(sv_cfg.get("min_valid_rows_per_batch", 1))

    # Clip V-JEPA branch config
    clip_vjepa_cfg = exp.get("clip_vjepa", {}) or {}
    clip_vjepa_enabled = bool(clip_vjepa_cfg.get("enabled", True))
    clip_vjepa_n_per_study = int(clip_vjepa_cfg.get("n_clips_per_study", 1))
    clip_vjepa_sample_policy = str(clip_vjepa_cfg.get("sample_policy", "random_valid"))

    # Clip consistency branch config (legacy lightweight path, now optional).
    clip_consistency_cfg = exp.get("clip_consistency", {}) or {}
    clip_consistency_enabled = bool(clip_consistency_cfg.get("enabled", True))

    # Study NCE config (cross-rank)
    nce_cfg = exp.get("study_nce", {}) or {}
    nce_enabled = bool(nce_cfg.get("enabled", True))
    nce_cross_rank = bool(nce_cfg.get("cross_rank", True)) and world_size > 1
    nce_tau = float(nce_cfg.get("tau", float(loss_cfg.get("tau_nce", 0.1))))
    nce_match_view = bool(nce_cfg.get("match_view_count_bucket", True))
    nce_match_modality = bool(nce_cfg.get("match_modality_count_bucket", True))
    nce_match_clip = bool(nce_cfg.get("match_clip_count_bucket", True))
    nce_exclude_patient = bool(nce_cfg.get("exclude_same_patient", True))

    tau_clip_sched = clip_ema_schedule(
        tau_start=float(ema_cfg.get("clip_tau_start", 0.999)),
        tau_end=float(ema_cfg.get("clip_tau_end", 0.99995)),
        total_steps=total_steps,
    )
    tau_study_sched = clip_ema_schedule(
        tau_start=float(ema_cfg.get("study_tau_start", 0.996)),
        tau_end=float(ema_cfg.get("study_tau_end", 0.9999)),
        total_steps=total_steps,
    )

    # ---- Checkpoint dir + CSV logger ----
    folder = str(cfg.get("folder", "/opt/dlami/nvme/checkpoints/full_joint_default"))
    os.makedirs(folder, exist_ok=True)
    csv_path = os.path.join(folder, f"log_r{rank}.csv")
    csv = CSVLogger(
        csv_path,
        *[
            ("%d", "step"),
            ("%.6f", "loss_total"),
            ("%.6f", "loss_clip_vjepa_true"),
            ("%.6f", "loss_clip_consistency"),
            ("%.6f", "loss_study"),
            ("%.6f", "loss_nce"),
            ("%.6f", "loss_cov"),
            ("%.6f", "loss_anchor_raw"),
            ("%.6f", "loss_anchor_weighted"),
            ("%.6f", "lambda_anchor_t"),
            ("%.6f", "loss_sv"),
            ("%.6f", "lambda_sv_t"),
            ("%.6f", "lambda_study_t"),
            ("%.4f", "sv_valid_fraction"),
            ("%d", "sv_num_rows"),
            ("%d", "a4c_sv_count"),
            ("%.4f", "var_t"),
            ("%.4f", "cov_off"),
            ("%.4f", "clip_var"),
            ("%.4f", "clip_cov_off"),
            ("%.4f", "study_matched_rank_top1_global"),
            ("%.4f", "study_matched_rank_top5_global"),
            ("%.4f", "pos_minus_hardneg_gap_global"),
            ("%.4f", "study_nce_pool_size"),
            ("%.4f", "study_nce_fallback_fraction"),
            ("%.4f", "metadata_only_study_gap"),
            ("%.4f", "anchor_cosine_to_e100"),
            ("%.4f", "cos_block_0"),
            ("%.4f", "cos_block_6"),
            ("%.4f", "cos_block_12"),
            ("%.4f", "cos_block_18"),
            ("%.4f", "cos_block_23"),
            ("%.4f", "cos_top_block"),
            ("%.4f", "cos_a4c_pooled"),
            ("%.4f", "K_actual_mean"),
            ("%.4f", "a4c_present_fraction"),
            ("%.4f", "color_present_fraction"),
            ("%.6f", "clip_grad_norm"),
            ("%.6f", "study_grad_norm"),
            ("%.6f", "ema_clip_delta"),
            ("%.6f", "ema_study_delta"),
            ("%d", "iter_ms"),
            ("%d", "gpu_mem_mb"),
        ],
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))

    # ---- MaskCollator for the true-V-JEPA clip branch ----
    mask_generator = None
    if clip_vjepa_enabled:
        collator = MaskCollator(
            cfgs_mask=exp.get("mask", []),
            dataset_fpcs=[fj_cfg.max_num_frames],
            crop_size=fj_cfg.crop_size,
            patch_size=fj_cfg.patch_size,
            tubelet_size=fj_cfg.tubelet_size,
            fps_sampled=int(data_cfg.get("fps", 8)),
        )
        # Use the first mask-generator for each FPC (index 0). This is fine
        # for a single-clip-per-study loss: we only need one (masks_enc,
        # masks_pred) pair per clip each step.
        mask_generator = collator.mask_generators[fj_cfg.max_num_frames][0]
        log.info(
            f"true V-JEPA clip branch enabled: n_per_study={clip_vjepa_n_per_study} "
            f"policy={clip_vjepa_sample_policy} mask_generator={type(mask_generator).__name__}"
        )

    # Vocab ids we filter on for diagnostics.
    from src.models.meta_embeddings import VIEW_FAMILY_VOCAB, MODALITY_VOCAB

    apical_id = VIEW_FAMILY_VOCAB.index("apical")
    color_id = MODALITY_VOCAB.index("color_doppler")

    # ---- Training loop ----
    log.info(f"full_joint: starting loop, total_steps={total_steps}, batch={batch_size}, dtype={dtype_str}")
    step = 0
    stop = False
    data_iter = iter(loader)
    t_loop0 = time.time()

    while not stop:
        if sampler is not None:
            sampler.set_epoch(step // max(1, len(loader)))

        for batch in data_iter:
            if step >= total_steps:
                stop = True
                break
            itr_t0 = time.time()
            # Move tensors to device.
            full_clips = batch["full_clips"].to(device, non_blocking=True)  # (B, M, 3, T, H, W)
            full_pad = batch["full_pad_mask"].to(device, non_blocking=True)  # (B, M)
            B, M = full_pad.shape

            meta_view = batch["full_meta_view"].to(device, non_blocking=True)
            meta_modality = batch["full_meta_modality"].to(device, non_blocking=True)
            meta_phase = batch["full_meta_phase"].to(device, non_blocking=True)
            meta_quality = batch["full_meta_quality"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=dtype, enabled=mixed_precision):
                # === Encode clips through TEACHER (no grad) and ANCHOR ===
                # Flatten (B, M) to N valid clips to avoid encoder compute on pads.
                clips_valid, valid_mask, B2, M2 = _flatten_studies_to_clips(full_clips, full_pad)
                # Anchor subsample indices (for speed): at most anchor_subsample per batch.
                n_valid = int(clips_valid.shape[0])
                if n_valid == 0:
                    # Edge case: no valid clips; skip this step
                    step += 1
                    continue

                with torch.no_grad():
                    teacher_tokens_valid = model.encode_clips_teacher(clips_valid)
                    teacher_tokens_valid = _spatial_pool_tokens(
                        teacher_tokens_valid,
                        spatial_hw=fj_cfg.crop_size // fj_cfg.patch_size,
                        pool=fj_cfg.token_spatial_pool,
                    )
                    # Anchor only on a subsample
                    anchor_idx = torch.randperm(n_valid, device=device)[: min(anchor_subsample, n_valid)]
                    anchor_clips = clips_valid[anchor_idx]
                    anchor_tokens = model.encode_clips_anchor(anchor_clips)
                    anchor_tokens = _spatial_pool_tokens(
                        anchor_tokens,
                        spatial_hw=fj_cfg.crop_size // fj_cfg.patch_size,
                        pool=fj_cfg.token_spatial_pool,
                    )

                # === STUDENT: encode full clips (online, trainable) ===
                student_tokens_valid = model.encode_clips_online(clips_valid)
                student_tokens_valid_pooled = _spatial_pool_tokens(
                    student_tokens_valid,
                    spatial_hw=fj_cfg.crop_size // fj_cfg.patch_size,
                    pool=fj_cfg.token_spatial_pool,
                )

                # Reshape tokens back to (B, M, T_tok, d_clip)
                student_tokens_bm = _unflatten_tokens_to_studies(student_tokens_valid_pooled, valid_mask, B, M)
                teacher_tokens_bm = _unflatten_tokens_to_studies(teacher_tokens_valid, valid_mask, B, M)

                # === Lightweight clip consistency loss (optional auxiliary) ===
                # Random 40% token mask; L1(student, teacher) on masked positions.
                if clip_consistency_enabled:
                    with torch.no_grad():
                        T_tok = student_tokens_valid_pooled.shape[1]
                        rnd = torch.rand(n_valid, T_tok, device=device)
                        n_mask = max(1, int(0.4 * T_tok))
                        _, mask_idx = rnd.topk(n_mask, dim=1, largest=False)
                    s_sel = torch.gather(
                        student_tokens_valid_pooled,
                        1,
                        mask_idx.unsqueeze(-1).expand(-1, -1, student_tokens_valid_pooled.shape[-1]),
                    )
                    t_sel = torch.gather(
                        teacher_tokens_valid,
                        1,
                        mask_idx.unsqueeze(-1).expand(-1, -1, teacher_tokens_valid.shape[-1]),
                    ).detach()
                    l_clip_consistency = torch.mean(torch.abs(s_sel - t_sel))
                else:
                    l_clip_consistency = torch.zeros((), device=device)

                # === True V-JEPA clip loss (primary clip self-supervision) ===
                # Sample 1 valid clip per study and run the standard V-JEPA
                # predictor pipeline: student on visible tokens -> predictor
                # output at mask_pred positions -> compare to teacher tokens
                # at the same positions (LN + L_p).
                if clip_vjepa_enabled and mask_generator is not None:
                    with torch.no_grad():
                        sv_gen = torch.Generator(device="cpu").manual_seed(step * 97 + 11)
                        sel_clips, _sel_idx, sel_valid = sample_clip_subset(
                            full_clips,
                            full_pad,
                            n_per_study=clip_vjepa_n_per_study,
                            policy=clip_vjepa_sample_policy,
                            generator=sv_gen,
                        )
                        sel_clips = sel_clips[sel_valid]
                    if sel_clips.size(0) > 0:
                        with torch.no_grad():
                            m_enc, m_pred = mask_generator(sel_clips.size(0))
                            m_enc = m_enc.to(device)
                            m_pred = m_pred.to(device)
                        enc_unwrapped = (
                            model.encoder.module
                            if isinstance(model.encoder, DistributedDataParallel)
                            else model.encoder
                        )
                        pred_unwrapped = (
                            model.predictor.module
                            if isinstance(model.predictor, DistributedDataParallel)
                            else model.predictor
                        )
                        l_clip_vjepa_true = clip_vjepa_true_loss(
                            sel_clips,
                            enc_unwrapped,
                            model.target_encoder,
                            pred_unwrapped,
                            m_enc,
                            m_pred,
                            loss_exp=1.0,
                        )
                    else:
                        l_clip_vjepa_true = torch.zeros((), device=device)
                else:
                    l_clip_vjepa_true = torch.zeros((), device=device)

                # === Anchor loss ===
                # Compare student tokens on the same subsample against anchor.
                anchor_student_tokens = student_tokens_valid_pooled[anchor_idx]
                l_anchor = anchor_loss(anchor_student_tokens, anchor_tokens)
                with torch.no_grad():
                    anchor_cos = anchor_cosine_to_e100(anchor_student_tokens, anchor_tokens)

                # === Study-level path ===
                # Encode per-element meta (shared across ctx/tgt in full study).
                meta_add = model.meta.encode_context(meta_view, meta_modality, meta_phase, meta_quality)
                # Corrupted student view
                g = torch.Generator().manual_seed(step * 7919 + 13)
                # Pool T tokens → per-element (B, M, d_clip); corruption zeros elements.
                per_element_tokens = student_tokens_bm.mean(dim=2)
                ctx_el_corrupt, ctx_pad_corrupt = apply_study_corruption(
                    per_element_tokens,
                    meta_add,
                    full_pad,
                    meta_view,
                    meta_modality,
                    corruption_cfg
                    or {
                        "random_element_dropout": 0.30,
                        "whole_view_family_dropout": 0.25,
                        "whole_modality_dropout": 0.15,
                        "no_dropout": 0.30,
                    },
                    g,
                )
                # Build token grid consistent with corrupted pad: zero out masked elements'
                # tokens so the token-level study transformer sees matching content.
                newly_padded = ctx_pad_corrupt & (~full_pad)
                student_tokens_corrupt = student_tokens_bm.clone()
                if newly_padded.any():
                    student_tokens_corrupt[newly_padded] = 0.0

                # Student: corrupted tokens, corrupted pad
                _, h_study = model.study_forward_student(student_tokens_corrupt, meta_add, ctx_pad_corrupt)
                # Teacher: full tokens, full pad
                _, z_study = model.study_forward_teacher(teacher_tokens_bm, meta_add, full_pad)

                # === Projector + study loss ===
                proj_student = (
                    model.projector.module if isinstance(model.projector, DistributedDataParallel) else model.projector
                )
                if hasattr(proj_student, "student"):
                    h_t = proj_student.student.forward(h_study)
                else:
                    h_t = proj_student.student_forward(h_study)
                with torch.no_grad():
                    if hasattr(proj_student, "teacher"):
                        z_t = proj_student.teacher.forward(z_study).detach()
                    else:
                        z_t = proj_student.teacher_forward(z_study).detach()
                l_study = 1.0 - layernorm_cosine(h_t, z_t).mean()

                # === NCE (cross-rank study-level if DDP, otherwise local) ===
                nce_diag: Dict[str, float] = {}
                if nce_enabled and B > 1:
                    if nce_cross_rank and dist.is_initialized():
                        # All-gather student + teacher proj across ranks.
                        gathered_h = [torch.zeros_like(h_t) for _ in range(world_size)]
                        gathered_z = [torch.zeros_like(z_t) for _ in range(world_size)]
                        dist.all_gather(gathered_h, h_t.detach())
                        dist.all_gather(gathered_z, z_t.detach())
                        z_global = torch.cat(gathered_z, dim=0)
                        # Build per-study bookkeeping. Use study_id_int as both study
                        # and patient proxy (patient_id may collide with itself).
                        sid_local = batch["study_id_int"].to(device)
                        sid_list = [torch.zeros_like(sid_local) for _ in range(world_size)]
                        dist.all_gather(sid_list, sid_local)
                        sid_global = torch.cat(sid_list, dim=0)
                        # n_unique view/modality/clip buckets for matched negatives.
                        n_elements_local = batch["n_elements"].to(device).long()
                        ne_list = [torch.zeros_like(n_elements_local) for _ in range(world_size)]
                        dist.all_gather(ne_list, n_elements_local)
                        ne_global = torch.cat(ne_list, dim=0)
                        # Bucket n_elements into 3 bins (low / mid / high) as a cheap proxy.
                        clip_bucket = (ne_global // 3).clamp(max=2)
                        # For view_count / modality_count we don't have per-study counts on
                        # the current batch; use a uniform bucket (all 0) so match_* is a no-op
                        # unless the user provided explicit buckets. This is a safe default.
                        view_bucket = torch.zeros_like(ne_global)
                        mod_bucket = torch.zeros_like(ne_global)
                        l_nce, nce_diag = cross_rank_study_nce_loss(
                            h_t,
                            z_global,
                            local_offset=rank * B,
                            study_id_global=sid_global,
                            patient_id_global=sid_global if nce_exclude_patient else None,
                            view_count_bucket_global=view_bucket,
                            modality_count_bucket_global=mod_bucket,
                            clip_count_bucket_global=clip_bucket,
                            tau=nce_tau,
                            match_view_bucket=nce_match_view,
                            match_modality_bucket=nce_match_modality,
                            match_clip_bucket=nce_match_clip,
                            exclude_same_patient=nce_exclude_patient,
                        )
                    else:
                        # Local-only NCE (fallback for single-GPU debug)
                        neg_mask = torch.ones(B, B, dtype=torch.bool, device=h_t.device)
                        l_nce = matched_nce(h_t, z_t, neg_mask, tau=nce_tau)
                else:
                    l_nce = torch.zeros((), device=h_t.device)

                # === Covariance penalty ===
                if B > 1:
                    l_cov_val, l_var = covariance_penalty(h_t, var_floor=float(loss_cfg.get("var_floor", 0.0)))
                    l_cov = l_cov_val + l_var
                else:
                    l_cov = torch.zeros((), device=h_t.device)

                # === Single-view-to-study branch (DDP-safe, per-row) ===
                sv_gen_cpu = torch.Generator(device="cpu").manual_seed(step * 131 + rank * 17 + 3)
                sv_pad_mask, sv_row_mask, sv_stats = sample_single_view_rows(
                    full_pad,
                    meta_view,
                    p_rows=sv_p_rows,
                    preference_order=sv_prefer,
                    generator=sv_gen_cpu,
                    min_rows=sv_min_valid_rows,
                )
                sv_pad_mask = sv_pad_mask.to(device)
                sv_row_mask = sv_row_mask.to(device)
                l_sv = torch.zeros((), device=h_t.device)
                if sv_row_mask.any() and sv_row_mask.sum().item() >= sv_min_valid_rows:
                    _, h_study_sv = model.study_forward_student(student_tokens_bm, meta_add, sv_pad_mask)
                    h_t_sv = proj_student.student_forward(h_study_sv)
                    # Only compute loss on rows where sv_row_mask is True.
                    valid_rows = sv_row_mask.nonzero(as_tuple=False).squeeze(-1)
                    if valid_rows.numel() > 0:
                        h_t_sv_v = h_t_sv[valid_rows]
                        z_t_v = z_t[valid_rows]
                        l_sv = 1.0 - layernorm_cosine(h_t_sv_v, z_t_v).mean()

                # === Assemble total ===
                losses_map = {
                    "clip_vjepa_true": l_clip_vjepa_true,
                    "clip_consistency": l_clip_consistency,
                    "study": l_study,
                    "nce": l_nce,
                    "cov": l_cov,
                    "anchor": l_anchor,
                    "sv": l_sv,
                }
                total, applied = assemble_total_loss(
                    losses_map,
                    weights,
                    lambda_study_ramp=study_ramp if study_ramp.warmup_steps > 0 else None,
                    lambda_sv_ramp=sv_ramp if sv_ramp.warmup_steps > 0 else None,
                    lambda_anchor_decay=anchor_decay if anchor_decay.decay_steps > 0 else None,
                    global_step=step,
                )

            # Backward + step
            optimizer.zero_grad(set_to_none=True)
            enc_for_grad = (
                model.encoder.module if isinstance(model.encoder, DistributedDataParallel) else model.encoder
            )
            enc_params = [p for p in enc_for_grad.parameters() if p.requires_grad]
            if scaler.is_enabled():
                scaler.scale(total).backward()
                scaler.unscale_(optimizer)
                clip_grad = torch.nn.utils.clip_grad_norm_(
                    enc_params,
                    max_norm=float(optim_cfg.get("grad_clip", 1.0)),
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                total.backward()
                clip_grad = torch.nn.utils.clip_grad_norm_(
                    enc_params,
                    max_norm=float(optim_cfg.get("grad_clip", 1.0)),
                )
                optimizer.step()

            # EMA updates
            with torch.no_grad():
                tau_c = next(tau_clip_sched)
                tau_s = next(tau_study_sched)
                step_clip_ema(model.target_encoder, enc_for_grad, tau_c)
                model.update_study_teacher(tau_s)
                model.update_projector_teacher(tau_s)

            # Diagnostics (every log_every)
            if step % log_every == 0:
                with torch.no_grad():
                    ema_clip_dn = ema_delta_norm(model.target_encoder, enc_for_grad)
                    study_unwrapped_m = (
                        model.student_st.module
                        if isinstance(model.student_st, DistributedDataParallel)
                        else model.student_st
                    )
                    ema_study_dn = ema_delta_norm(model.teacher_st_ema.teacher, study_unwrapped_m)

                    study_grad = 0.0
                    for p in study_unwrapped_m.parameters():
                        if p.grad is not None:
                            study_grad += float(p.grad.norm().item()) ** 2
                    study_grad = math.sqrt(study_grad)

                    if B > 1:
                        cov_m = h_t - h_t.mean(0, keepdim=True)
                        cov_mat = cov_m.t() @ cov_m / max(B - 1, 1)
                        cov_off = cov_mat.fill_diagonal_(0.0).abs().mean().item()
                        var_t = h_t.std(dim=0).mean().item()
                        # Metadata-only: all elements padded
                        all_padded = torch.ones_like(ctx_pad_corrupt)
                        _, h_study_meta = model.study_forward_student(student_tokens_bm, meta_add, all_padded)
                        h_t_meta = proj_student.student_forward(h_study_meta)
                        cos_actual = layernorm_cosine(h_t, z_t).mean().item()
                        cos_meta = layernorm_cosine(h_t_meta, z_t).mean().item()
                        meta_gap = cos_actual - cos_meta
                    else:
                        cov_off = 0.0
                        var_t = 0.0
                        meta_gap = float("nan")

                    # Layer-wise drift on the anchor subsample
                    drift = compute_layerwise_cosine(
                        enc_for_grad,
                        model.anchor,
                        anchor_clips,
                        block_indices=[0, 6, 12, 18, 23],
                        view_mask=None,
                    )
                    # A4C-only clip cosine: pick clips whose view-family is apical.
                    # Use the per-clip meta recovered from the flat valid index.
                    a4c_cos = float("nan")
                    with torch.no_grad():
                        flat_view = meta_view.reshape(-1)[valid_mask]
                        a4c_mask = flat_view == apical_id
                        if a4c_mask.any():
                            a4c_clips_v = clips_valid[a4c_mask]
                            if a4c_clips_v.size(0) > 0:
                                drift_a4c = compute_layerwise_cosine(
                                    enc_for_grad,
                                    model.anchor,
                                    a4c_clips_v,
                                    block_indices=[23],
                                )
                                a4c_cos = drift_a4c.get("pooled_final", float("nan"))

                    # Clip cov/var on pooled student tokens (anchor subsample)
                    clip_diag = compute_clip_cov_var(anchor_student_tokens)

                    # K diagnostics
                    k_actual = (~full_pad).float().sum(dim=1)
                    k_actual_mean = float(k_actual.mean().item())
                    with torch.no_grad():
                        a4c_present_mask = (meta_view == apical_id) & ~full_pad
                        a4c_present_per_row = a4c_present_mask.any(dim=1)
                        a4c_present_fraction = float(a4c_present_per_row.float().mean().item())
                        color_present_mask = (meta_modality == color_id) & ~full_pad
                        color_present_per_row = color_present_mask.any(dim=1)
                        color_present_fraction = float(color_present_per_row.float().mean().item())

                    mem_mb = int(torch.cuda.max_memory_allocated() / (1024 * 1024)) if torch.cuda.is_available() else 0
                itr_ms = int((time.time() - itr_t0) * 1000)

                l_anchor_weighted_v = float(applied.get("lambda_anchor_t", 0.0)) * float(l_anchor.item())

                if rank == 0:
                    log.info(
                        f"[step {step:5d}/{total_steps}] total={float(total.item()):.4f} "
                        f"clipV={float(l_clip_vjepa_true.item()):.4f} "
                        f"clipC={float(l_clip_consistency.item()):.4f} "
                        f"study={float(l_study.item()):.4f} "
                        f"nce={float(l_nce.item()):.4f} cov={float(l_cov.item()):.4f} "
                        f"anchor={float(l_anchor.item()):.4f} (λ={applied.get('lambda_anchor_t', 0):.3f}) "
                        f"sv={float(l_sv.item()):.4f} "
                        f"(n_rows={sv_stats['sv_num_rows']}/{B} a4c={sv_stats['a4c_sv_count']}) "
                        f"anchor_cos={anchor_cos:.3f} meta_gap={meta_gap:.3f} "
                        f"rank_top1={nce_diag.get('study_matched_rank_top1_global', float('nan')):.3f} "
                        f"grad_clip={float(clip_grad):.3f} grad_study={study_grad:.3f} "
                        f"ema_c={ema_clip_dn:.2f} ema_s={ema_study_dn:.2f} "
                        f"K={k_actual_mean:.2f} A4C_frac={a4c_present_fraction:.2f} "
                        f"mem={mem_mb}MiB itr={itr_ms}ms"
                    )
                csv.log(
                    step,
                    float(total.item()),
                    float(l_clip_vjepa_true.item()),
                    float(l_clip_consistency.item()),
                    float(l_study.item()),
                    float(l_nce.item()),
                    float(l_cov.item()),
                    float(l_anchor.item()),
                    l_anchor_weighted_v,
                    float(applied.get("lambda_anchor_t", 0.0)),
                    float(l_sv.item()),
                    float(applied.get("lambda_sv_t", 0.0)),
                    float(applied.get("lambda_study_t", 0.0)),
                    float(sv_stats["sv_valid_fraction"]),
                    int(sv_stats["sv_num_rows"]),
                    int(sv_stats["a4c_sv_count"]),
                    var_t,
                    cov_off,
                    float(clip_diag["clip_var"]),
                    float(clip_diag["clip_cov_off"]),
                    float(nce_diag.get("study_matched_rank_top1_global", float("nan"))),
                    float(nce_diag.get("study_matched_rank_top5_global", float("nan"))),
                    float(nce_diag.get("pos_minus_hardneg_gap_global", float("nan"))),
                    float(nce_diag.get("study_nce_pool_size", float("nan"))),
                    float(nce_diag.get("study_nce_fallback_fraction", float("nan"))),
                    meta_gap,
                    anchor_cos,
                    drift.get("block_0", float("nan")),
                    drift.get("block_6", float("nan")),
                    drift.get("block_12", float("nan")),
                    drift.get("block_18", float("nan")),
                    drift.get("block_23", float("nan")),
                    drift.get("top_block", float("nan")),
                    a4c_cos,
                    k_actual_mean,
                    a4c_present_fraction,
                    color_present_fraction,
                    float(clip_grad),
                    study_grad,
                    ema_clip_dn,
                    ema_study_dn,
                    itr_ms,
                    mem_mb,
                )

            if rank == 0 and step in save_steps:
                path = os.path.join(folder, f"step_{step}.pt")
                _save_checkpoint(path, model, optimizer, scaler, step, cfg)

            step += 1

        # Restart iterator for next epoch
        try:
            data_iter = iter(loader)
        except Exception:
            break

    if rank == 0:
        path = os.path.join(folder, "latest.pt")
        _save_checkpoint(path, model, optimizer, scaler, step, cfg)
        log.info(f"full_joint: saved latest.pt at step {step}; total wall {int(time.time() - t_loop0)}s")


def _save_checkpoint(
    path: str,
    model: Any,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    step: int,
    cfg: Dict[str, Any],
) -> None:
    def _sd(m):
        return m.module.state_dict() if isinstance(m, DistributedDataParallel) else m.state_dict()

    sd = {
        "step": step,
        "clip_encoder": _sd(model.encoder),
        "clip_target_encoder": model.target_encoder.state_dict(),
        "clip_anchor_e100": model.anchor.state_dict(),
        "clip_predictor": _sd(model.predictor),
        "study_encoder": _sd(model.student_st),
        "study_target_encoder": model.teacher_st_ema.state_dict(),
        "study_projector": _sd(model.projector),
        "meta_embeddings": model.meta.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler.is_enabled() else None,
        "config": cfg,
    }
    tmp = path + ".tmp"
    torch.save(sd, tmp)
    os.replace(tmp, path)


__all__ = ["main"]
