"""
SALT utility functions: model initialization, optimizer setup, checkpoint loading.
"""

import logging
import re
import sys

import torch
import torch.nn.functional as F

import app.vjepa_2_1.models.predictor as vit_pred
import app.vjepa_2_1.models.vision_transformer as video_vit
from app.salt.models.pixel_decoder import PixelDecoder
from app.vjepa_2_1.wrappers import MultiSeqWrapper, PredictorMultiSeqWrapper
from src.utils.checkpoint_loader import robust_checkpoint_loader
from src.utils.schedulers import CosineWDSchedule, WarmupCosineSchedule

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def _count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _strip_ddp_and_wrapper_prefix(state_dict):
    """Strip 'module.' (DDP) and 'backbone.' (MultiSeqWrapper) prefixes."""
    return {re.sub(r"^module\.(?:module\.)?(?:backbone\.)?", "", k): v for k, v in state_dict.items()}


def _make_encoder(
    model_name,
    crop_size,
    patch_size,
    max_num_frames,
    tubelet_size,
    uniform_power,
    use_sdpa,
    use_silu,
    wide_silu,
    use_activation_checkpointing,
    use_rope,
):
    """Create a ViT encoder wrapped in MultiSeqWrapper."""
    encoder = video_vit.__dict__[model_name](
        img_size=crop_size,
        patch_size=patch_size,
        num_frames=max_num_frames,
        tubelet_size=tubelet_size,
        uniform_power=uniform_power,
        use_sdpa=use_sdpa,
        use_silu=use_silu,
        wide_silu=wide_silu,
        use_activation_checkpointing=use_activation_checkpointing,
        use_rope=use_rope,
    )
    return MultiSeqWrapper(encoder)


# ------------------------------------------------------------------ #
#  Stage 1: V-Pixel (encoder + pixel decoder)
# ------------------------------------------------------------------ #


def init_vpixel_model(
    device,
    model_name="vit_large",
    crop_size=224,
    patch_size=16,
    max_num_frames=16,
    tubelet_size=2,
    decoder_embed_dim=512,
    decoder_depth=8,
    decoder_num_heads=8,
    uniform_power=False,
    use_sdpa=True,
    use_silu=False,
    wide_silu=True,
    use_activation_checkpointing=False,
    use_rope=True,
):
    """Initialize encoder + pixel decoder for SALT Stage 1."""
    encoder = _make_encoder(
        model_name=model_name,
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
    )

    if isinstance(crop_size, int):
        crop_size = (crop_size, crop_size)
    num_patches = (max_num_frames // tubelet_size) * (crop_size[0] // patch_size) * (crop_size[1] // patch_size)

    decoder = PixelDecoder(
        num_patches=num_patches,
        encoder_embed_dim=encoder.backbone.embed_dim,
        decoder_embed_dim=decoder_embed_dim,
        decoder_depth=decoder_depth,
        decoder_num_heads=decoder_num_heads,
        patch_size=patch_size,
        tubelet_size=tubelet_size,
        in_chans=3,
        img_size=crop_size,
        num_frames=max_num_frames,
        use_rope=use_rope,
        use_activation_checkpointing=use_activation_checkpointing,
    )

    encoder.to(device)
    decoder.to(device)
    logger.info(f"Encoder parameters: {_count_parameters(encoder)}")
    logger.info(f"Pixel decoder parameters: {_count_parameters(decoder)}")

    return encoder, decoder


# ------------------------------------------------------------------ #
#  Stage 2: Frozen-teacher student training
# ------------------------------------------------------------------ #


def init_salt_student_model(
    device,
    # Student encoder config
    student_model_name="vit_large",
    crop_size=224,
    patch_size=16,
    max_num_frames=16,
    tubelet_size=2,
    uniform_power=False,
    use_sdpa=True,
    use_silu=False,
    wide_silu=True,
    use_activation_checkpointing=False,
    use_rope=True,
    # Teacher config
    teacher_model_name=None,
    teacher_checkpoint=None,
    # Predictor config
    pred_depth=12,
    pred_num_heads=12,
    pred_embed_dim=384,
    use_mask_tokens=True,
    num_mask_tokens=10,
    zero_init_mask_tokens=True,
    n_output_distillation=None,
):
    """Initialize student encoder, predictor, and frozen teacher for SALT Stage 2."""
    if teacher_model_name is None:
        teacher_model_name = student_model_name

    # -- Student encoder
    student_encoder = _make_encoder(
        model_name=student_model_name,
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
    )

    # -- Teacher encoder (same architecture, loaded from Stage 1 checkpoint)
    teacher_encoder = _make_encoder(
        model_name=teacher_model_name,
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
    )

    # Load teacher weights from Stage 1
    assert teacher_checkpoint is not None, "Stage 2 requires teacher_checkpoint path"
    logger.info(f"Loading frozen teacher from {teacher_checkpoint}")
    ckpt = robust_checkpoint_loader(teacher_checkpoint, map_location=torch.device("cpu"))
    teacher_sd = ckpt["encoder"]
    # Strip DDP/wrapper prefixes if present
    teacher_sd = _strip_ddp_and_wrapper_prefix(teacher_sd)
    msg = teacher_encoder.backbone.load_state_dict(teacher_sd, strict=False)
    logger.info(f"Loaded teacher encoder with msg: {msg}")
    del ckpt

    # Freeze teacher
    for p in teacher_encoder.parameters():
        p.requires_grad = False

    # -- Predictor
    # Determine teacher embed dim for output projection
    teacher_embed_dim = teacher_encoder.backbone.embed_dim
    student_embed_dim = student_encoder.backbone.embed_dim
    # SALT paper (Eq 2.1) uses single-level patch-token prediction — the predictor
    # produces one token per masked position with dim = teacher_embed_dim.
    # n_output_distillation defaults to 1 (single-level) to match the paper.
    # Setting n_output_distillation > 1 enables V-JEPA 2.1's hierarchical feature
    # distillation extension, which is NOT part of the SALT paper recipe.
    n_hier = n_output_distillation if n_output_distillation is not None else 1
    teacher_hier_dim = teacher_embed_dim * n_hier

    predictor_kwargs = dict(
        img_size=crop_size,
        use_mask_tokens=use_mask_tokens,
        patch_size=patch_size,
        num_frames=max_num_frames,
        tubelet_size=tubelet_size,
        embed_dim=student_embed_dim,
        predictor_embed_dim=pred_embed_dim,
        depth=pred_depth,
        num_heads=pred_num_heads,
        uniform_power=uniform_power,
        num_mask_tokens=num_mask_tokens,
        zero_init_mask_tokens=zero_init_mask_tokens,
        use_rope=use_rope,
        use_sdpa=use_sdpa,
        use_activation_checkpointing=use_activation_checkpointing,
        return_all_tokens=False,
        teacher_embed_dim=teacher_hier_dim,
        n_output_distillation=n_hier,  # Always pass so predictor matches train.py hier mode
    )

    predictor = vit_pred.__dict__["vit_predictor"](**predictor_kwargs)
    predictor = PredictorMultiSeqWrapper(predictor)

    student_encoder.to(device)
    predictor.to(device)
    teacher_encoder.to(device)

    logger.info(f"Student encoder parameters: {_count_parameters(student_encoder)}")
    logger.info(f"Predictor parameters: {_count_parameters(predictor)}")
    logger.info(f"Teacher encoder parameters (frozen): {_count_parameters(teacher_encoder)}")

    return student_encoder, predictor, teacher_encoder


# ------------------------------------------------------------------ #
#  Optimizer
# ------------------------------------------------------------------ #


def init_opt(
    encoder,
    head,
    iterations_per_epoch,
    start_lr,
    ref_lr,
    warmup,
    num_epochs,
    wd=1e-6,
    final_wd=1e-6,
    final_lr=0.0,
    mixed_precision=False,
    ipe_scale=1.25,
    betas=(0.9, 0.999),
    eps=1e-8,
    zero_init_bias_wd=True,
):
    """
    Initialize optimizer and schedulers.

    Args:
        encoder: encoder (Stage 1 or Stage 2 student).
        head: decoder (Stage 1) or predictor (Stage 2).
    """
    param_groups = [
        {
            "params": (
                p for n, p in encoder.named_parameters() if ("bias" not in n) and (len(p.shape) != 1)
            ),
        },
        {
            "params": (
                p for n, p in head.named_parameters() if ("bias" not in n) and (len(p.shape) != 1)
            ),
        },
        {
            "params": (
                p for n, p in encoder.named_parameters() if ("bias" in n) or (len(p.shape) == 1)
            ),
            "WD_exclude": zero_init_bias_wd,
            "weight_decay": 0,
        },
        {
            "params": (
                p for n, p in head.named_parameters() if ("bias" in n) or (len(p.shape) == 1)
            ),
            "WD_exclude": zero_init_bias_wd,
            "weight_decay": 0,
        },
    ]

    optimizer = torch.optim.AdamW(param_groups, betas=betas, eps=eps)

    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=int(warmup * iterations_per_epoch),
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
    )
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
    )

    scaler = torch.cuda.amp.GradScaler() if mixed_precision else None
    return optimizer, scaler, scheduler, wd_scheduler


# ------------------------------------------------------------------ #
#  Checkpoint loading (resume)
# ------------------------------------------------------------------ #


def load_checkpoint(r_path, stage, encoder, head, opt, scaler):
    """
    Load checkpoint for resuming training.

    Args:
        stage: 1 or 2.
        encoder: encoder (Stage 1) or student_encoder (Stage 2).
        head: decoder (Stage 1) or predictor (Stage 2).
    """
    logger.info(f"Loading checkpoint from {r_path}")
    checkpoint = robust_checkpoint_loader(r_path, map_location=torch.device("cpu"))
    epoch = checkpoint["epoch"]
    itr = checkpoint.get("itr", 0)

    # Encoder
    msg = encoder.load_state_dict(checkpoint["encoder"], strict=False)
    logger.info(f"Loaded encoder from epoch {epoch} with msg: {msg}")

    # Head (decoder or predictor)
    head_key = "decoder" if stage == 1 else "predictor"
    msg = head.load_state_dict(checkpoint[head_key], strict=False)
    logger.info(f"Loaded {head_key} from epoch {epoch} with msg: {msg}")

    # Optimizer
    try:
        opt.load_state_dict(checkpoint["opt"])
    except ValueError:
        logger.warning("Optimizer groups mismatch; reinitializing optimizer.")
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])

    logger.info(f"Loaded optimizers from epoch {epoch}")
    del checkpoint
    return encoder, head, opt, scaler, epoch, itr


# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #


def normalize_and_concat(tensor, embed_dim):
    """Split tensor into 4 chunks of size embed_dim along the last axis,
    apply LayerNorm to each chunk, then concatenate back."""
    chunks = [
        F.layer_norm(tensor[:, :, i * embed_dim : (i + 1) * embed_dim], (embed_dim,))
        for i in range(4)
    ]
    return torch.cat(chunks, dim=2)
