"""Top-level bundle of every trainable + frozen network in the full-joint
Global Study-Token EchoMV-JEPA experiment.

Holds:
  * ``f_theta``       trainable V-JEPA clip encoder
  * ``fbar_theta``    EMA teacher of ``f_theta`` (no grad)
  * ``f0``            frozen e100 anchor encoder (no grad, static)
  * ``predictor``     V-JEPA predictor (trainable, used for clip-level loss)
  * ``F_psi``         student study transformer (wrapped as TokenStudyTransformer)
  * ``Fbar_psi``      EMA teacher study transformer (wrapped, no grad)
  * ``p_study``       study projector (student head)
  * ``pbar_study``    EMA study projector (no grad)
  * ``meta``          MetaEmbeddings for (view, modality, phase, quality)

This module only owns construction + forward helpers. Optimizer setup,
EMA updates, and the loss assembly live in callers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from src.models.echomv_jepa.clip_ema import freeze
from src.models.echomv_jepa.full_joint_clip_backbone import (
    ClipBackbonePack,
    build_clip_encoder_from_e100,
)
from src.models.echomv_jepa.study_target_encoder import StudyTransformerEMA
from src.models.echomv_jepa.token_study_transformer import TokenStudyTransformer
from src.models.meta_embeddings import MetaEmbeddings
from src.models.study_projectors import EMAProjectorPair
from src.models.study_transformer import StudyTransformer, StudyTransformerConfig

logger = logging.getLogger(__name__)


@dataclass
class FullJointConfig:
    ckpt_path: str
    # clip encoder
    model_name: str = "vit_large"
    crop_size: int = 224
    max_num_frames: int = 16
    tubelet_size: int = 2
    patch_size: int = 16
    pred_depth: int = 12
    pred_embed_dim: int = 384
    pred_num_heads: int = 12
    use_mask_tokens: bool = True
    num_mask_tokens: int = 10
    use_rope: bool = True
    use_sdpa: bool = True
    use_activation_checkpointing: bool = True
    uniform_power: bool = True
    zero_init_mask_tokens: bool = True
    # study transformer
    d_model: int = 512
    study_depth: int = 4
    study_num_heads: int = 8
    # projector
    projector_hidden: int = 1024
    projector_out: int = 256
    # token spatial pool (passed through to the caller — FullJointModel does
    # not perform clip encoding itself; the trainer does, so this is a hint)
    token_spatial_pool: int = 2  # 2x2 → 1568/4=392 tokens per clip for ViT-L


class FullJointModel(nn.Module):
    """Aggregator; forward methods are deliberately small delegations."""

    def __init__(
        self,
        clip_pack: ClipBackbonePack,
        student_st: StudyTransformer,
        teacher_st_ema: StudyTransformerEMA,
        projector: EMAProjectorPair,
        meta: MetaEmbeddings,
    ) -> None:
        super().__init__()
        # clip-side
        self.encoder = clip_pack.encoder  # f_theta (trainable)
        self.target_encoder = clip_pack.target_encoder  # fbar_theta (EMA)
        self.anchor = clip_pack.anchor  # f0 (frozen e100)
        self.predictor = clip_pack.predictor
        self.embed_dim = clip_pack.embed_dim
        # study-side
        self.student_st = student_st
        self.token_student = TokenStudyTransformer(student_st)
        self.teacher_st_ema = teacher_st_ema  # wraps an EMA of student_st
        self.token_teacher = TokenStudyTransformer(teacher_st_ema.teacher)
        # projector
        self.projector = projector  # EMAProjectorPair
        # meta
        self.meta = meta

    # ------------------------------------------------------------------ #
    # Forward helpers
    # ------------------------------------------------------------------ #

    def encode_clips_teacher(self, clips: torch.Tensor) -> torch.Tensor:
        """No-grad clip forward through the EMA teacher (``fbar_theta``).

        ``clips``: ``(N, 3, T, H, W)``. Returns ``(N, T_tok, d_clip)``.
        """
        with torch.no_grad():
            x = self.target_encoder([clips])  # MultiSeqWrapper expects list
        return x[0] if isinstance(x, list) else x

    def encode_clips_anchor(self, clips: torch.Tensor) -> torch.Tensor:
        """No-grad clip forward through the frozen e100 anchor (``f0``)."""
        with torch.no_grad():
            x = self.anchor([clips])
        return x[0] if isinstance(x, list) else x

    def encode_clips_online(
        self,
        clips: torch.Tensor,
        masks: Optional[list] = None,
    ) -> torch.Tensor:
        """Trainable forward through ``f_theta``.

        With ``masks=None`` returns full token grid; with ``masks`` returns
        only the visible tokens indexed by the mask (V-JEPA convention).
        """
        out = self.encoder([clips], masks=masks) if masks is not None else self.encoder([clips])
        if isinstance(out, list):
            out = out[0]
            if isinstance(out, list):  # when masks: list-of-list
                out = out[0]
        return out

    def study_forward_student(
        self,
        element_tokens: torch.Tensor,  # (B, M, T, d_clip)
        element_meta_add: torch.Tensor,  # (B, M, d_model)
        elem_pad_mask: torch.Tensor,  # (B, M) bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the student study transformer; returns (h_per_elem, h_study)."""
        return self.token_student.forward_with_study_token(element_tokens, element_meta_add, elem_pad_mask)

    @torch.no_grad()
    def study_forward_teacher(
        self,
        element_tokens: torch.Tensor,
        element_meta_add: torch.Tensor,
        elem_pad_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Teacher path (no grad). Returns (z_per_elem, z_study)."""
        return self.token_teacher.forward_with_study_token(element_tokens, element_meta_add, elem_pad_mask)

    # ------------------------------------------------------------------ #
    # Step helpers
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def update_study_teacher(self, tau: float) -> None:
        """Step the study-transformer EMA toward the student.

        Handles DDP wrapping transparently: unwraps ``self.student_st`` if it
        was replaced with a ``DistributedDataParallel`` wrapper by the caller.
        """
        student = self.student_st.module if hasattr(self.student_st, "module") else self.student_st
        self.teacher_st_ema.update_teacher(student, tau)

    @torch.no_grad()
    def update_projector_teacher(self, tau: float) -> None:
        """Step the EMA projector toward the student projector."""
        proj = self.projector.module if hasattr(self.projector, "module") else self.projector
        proj.update_teacher(tau)


# ---------------------------------------------------------------------- #
# Factory
# ---------------------------------------------------------------------- #


def build_full_joint_model(
    cfg: FullJointConfig,
    device: torch.device,
) -> FullJointModel:
    """One-call construction from config. Loads e100 weights into the
    clip encoder + target + anchor and initializes the study
    transformer / projector fresh (no init ckpt)."""
    clip_pack = build_clip_encoder_from_e100(
        ckpt_path=cfg.ckpt_path,
        device=device,
        model_name=cfg.model_name,
        crop_size=cfg.crop_size,
        max_num_frames=cfg.max_num_frames,
        tubelet_size=cfg.tubelet_size,
        patch_size=cfg.patch_size,
        pred_depth=cfg.pred_depth,
        pred_embed_dim=cfg.pred_embed_dim,
        pred_num_heads=cfg.pred_num_heads,
        use_mask_tokens=cfg.use_mask_tokens,
        num_mask_tokens=cfg.num_mask_tokens,
        use_rope=cfg.use_rope,
        use_sdpa=cfg.use_sdpa,
        use_activation_checkpointing=cfg.use_activation_checkpointing,
        uniform_power=cfg.uniform_power,
        zero_init_mask_tokens=cfg.zero_init_mask_tokens,
    )
    d_clip = clip_pack.embed_dim
    meta = MetaEmbeddings(d_model=cfg.d_model).to(device)

    st_cfg = StudyTransformerConfig(
        d_clip=d_clip,
        d_model=cfg.d_model,
        n_layers=cfg.study_depth,
        n_heads=cfg.study_num_heads,
    )
    student_st = StudyTransformer(st_cfg).to(device)
    teacher_st_ema = StudyTransformerEMA(student_st).to(device)
    freeze(teacher_st_ema)

    projector = EMAProjectorPair(
        d_model=cfg.d_model,
        d_hidden=cfg.projector_hidden,
        d_proj=cfg.projector_out,
    ).to(device)
    for p in projector.teacher.parameters():
        p.requires_grad_(False)

    model = FullJointModel(
        clip_pack=clip_pack,
        student_st=student_st,
        teacher_st_ema=teacher_st_ema,
        projector=projector,
        meta=meta,
    )
    logger.info(
        f"FullJointModel built: d_clip={d_clip} d_model={cfg.d_model} "
        f"projector_out={cfg.projector_out} use_ckpt={cfg.use_activation_checkpointing}"
    )
    return model


__all__ = ["FullJointModel", "FullJointConfig", "build_full_joint_model"]
