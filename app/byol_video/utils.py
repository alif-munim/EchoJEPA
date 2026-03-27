import logging
import os
import re
import sys

import torch

import src.models.vision_transformer as video_vit
from src.models.byol_projector import BYOLPredictor, BYOLProjector
from src.utils.checkpoint_loader import robust_checkpoint_loader
from src.utils.schedulers import CosineWDSchedule, LinearDecaySchedule, WarmupCosineSchedule

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def _strip_ddp_and_wrapper_prefix(state_dict):
    """Strip 'module.' (DDP) and 'backbone.' (MultiSeqWrapper) prefixes."""
    return {re.sub(r"^module\.(?:module\.)?(?:backbone\.)?", "", k): v for k, v in state_dict.items()}


def init_byol_model(
    device,
    patch_size=16,
    max_num_frames=16,
    tubelet_size=2,
    model_name="vit_large",
    crop_size=224,
    uniform_power=False,
    use_sdpa=False,
    use_rope=False,
    use_silu=False,
    wide_silu=False,
    use_activation_checkpointing=False,
    proj_hidden_dim=4096,
    proj_dim=256,
    pred_hidden_dim=4096,
):
    # Raw ViT backbone (no MultiSeqWrapper — BYOL processes full unmasked clips)
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

    embed_dim = encoder.embed_dim
    online_projector = BYOLProjector(embed_dim, proj_hidden_dim, proj_dim)
    online_predictor = BYOLPredictor(proj_dim, pred_hidden_dim, proj_dim)

    encoder.to(device)
    online_projector.to(device)
    online_predictor.to(device)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Encoder parameters: {count_parameters(encoder)}")
    logger.info(f"Projector parameters: {count_parameters(online_projector)}")
    logger.info(f"Predictor parameters: {count_parameters(online_predictor)}")

    return encoder, online_projector, online_predictor


def init_opt(
    is_anneal,
    encoder,
    online_projector,
    online_predictor,
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
    from itertools import chain

    proj_pred_named = chain(online_projector.named_parameters(), online_predictor.named_parameters())

    param_groups = [
        {
            "params": (p for n, p in encoder.named_parameters() if ("bias" not in n) and (len(p.shape) != 1)),
        },
        {
            "params": (
                p for n, p in proj_pred_named if ("bias" not in n) and (len(p.shape) != 1)
            ),
        },
        {
            "params": (p for n, p in encoder.named_parameters() if ("bias" in n) or (len(p.shape) == 1)),
            "WD_exclude": zero_init_bias_wd,
            "weight_decay": 0,
        },
        {
            "params": (
                p
                for n, p in chain(
                    online_projector.named_parameters(), online_predictor.named_parameters()
                )
                if ("bias" in n) or (len(p.shape) == 1)
            ),
            "WD_exclude": zero_init_bias_wd,
            "weight_decay": 0,
        },
    ]

    optimizer = torch.optim.AdamW(param_groups, betas=betas, eps=eps)
    if not is_anneal:
        scheduler = WarmupCosineSchedule(
            optimizer,
            warmup_steps=int(warmup * iterations_per_epoch),
            start_lr=start_lr,
            ref_lr=ref_lr,
            final_lr=final_lr,
            T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
        )
    else:
        scheduler = LinearDecaySchedule(
            optimizer,
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


def load_checkpoint(
    r_path,
    encoder,
    online_projector,
    online_predictor,
    target_encoder,
    target_projector,
    opt,
    scaler,
):
    logger.info(f"Loading checkpoint from {r_path}")
    checkpoint = robust_checkpoint_loader(r_path, map_location=torch.device("cpu"))

    epoch = checkpoint["epoch"]
    itr = checkpoint.get("itr", 0)

    pretrained_dict = _strip_ddp_and_wrapper_prefix(checkpoint["encoder"])
    msg = encoder.load_state_dict(pretrained_dict)
    logger.info(f"loaded encoder from epoch {epoch} with msg: {msg}")

    pretrained_dict = _strip_ddp_and_wrapper_prefix(checkpoint["online_projector"])
    msg = online_projector.load_state_dict(pretrained_dict)
    logger.info(f"loaded online_projector from epoch {epoch} with msg: {msg}")

    pretrained_dict = _strip_ddp_and_wrapper_prefix(checkpoint["online_predictor"])
    msg = online_predictor.load_state_dict(pretrained_dict)
    logger.info(f"loaded online_predictor from epoch {epoch} with msg: {msg}")

    pretrained_dict = _strip_ddp_and_wrapper_prefix(checkpoint["target_encoder"])
    msg = target_encoder.load_state_dict(pretrained_dict)
    logger.info(f"loaded target_encoder from epoch {epoch} with msg: {msg}")

    pretrained_dict = _strip_ddp_and_wrapper_prefix(checkpoint["target_projector"])
    msg = target_projector.load_state_dict(pretrained_dict)
    logger.info(f"loaded target_projector from epoch {epoch} with msg: {msg}")

    # Optimizer may be in the main checkpoint or a separate _opt.pt file
    # (split to keep each file under 4 GB for PyTorch's zipfile serializer)
    if "opt" in checkpoint:
        opt.load_state_dict(checkpoint["opt"])
    else:
        opt_path = r_path.replace(".pt", "_opt.pt")
        if os.path.exists(opt_path):
            opt_ckpt = robust_checkpoint_loader(opt_path, map_location=torch.device("cpu"))
            opt.load_state_dict(opt_ckpt["opt"])
            del opt_ckpt
        else:
            logger.warning(f"No optimizer state found in checkpoint or {opt_path}")
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    logger.info(f"loaded optimizers from epoch {epoch}")
    del checkpoint

    return (
        encoder,
        online_projector,
        online_predictor,
        target_encoder,
        target_projector,
        opt,
        scaler,
        epoch,
        itr,
    )
