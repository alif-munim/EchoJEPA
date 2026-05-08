"""Control B2 — Capacity-matched supervised study transformer (plan §7).

Identical architecture to EchoSet-JEPA (same d_model, n_layers, n_heads, same
element grouping, same meta tokens, same K=8) but randomly initialized and
trained end-to-end with downstream labels only.

Reuses ``src.models.study_transformer.StudyTransformer`` + ``MetaEmbeddings``;
the only thing that differs from the EchoSet-JEPA path is that there is no
masked-prediction pretraining — the model goes straight from random init to
the supervised ``StudyProbeHead`` on ``[STUDY]``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from evals.echoset_jepa_probe.probe import StudyProbeHead
from src.models.meta_embeddings import MetaEmbeddings
from src.models.study_transformer import StudyTransformer, StudyTransformerConfig


def build_control_b2(
    st_cfg: StudyTransformerConfig,
    n_targets: int,
    include_target_phase: bool = True,
) -> nn.Module:
    """Build the B2 supervised study transformer wrapper.

    Called with the exact same ``StudyTransformerConfig`` as EchoSet-JEPA
    (``d_model=512, n_layers=4, n_heads=8, ffn_mult=4``).
    """

    class B2(nn.Module):
        def __init__(self):
            super().__init__()
            self.meta = MetaEmbeddings(d_model=st_cfg.d_model)
            self.st = StudyTransformer(st_cfg)
            self.head = StudyProbeHead(st_cfg.d_model, n_targets)
            self.include_target_phase = include_target_phase

        def forward(
            self,
            ctx_elements: torch.Tensor,
            ctx_meta_view: torch.Tensor,
            ctx_meta_modality: torch.Tensor,
            ctx_meta_phase: torch.Tensor,
            ctx_meta_quality: torch.Tensor,
            ctx_pad_mask: torch.Tensor,
        ) -> torch.Tensor:
            # Supervised: no mask slots — only [STUDY] + context.
            B = ctx_elements.shape[0]
            device = ctx_elements.device
            empty_tgt_meta = torch.zeros(B, 0, st_cfg.d_model, device=device)
            empty_tgt_pad = torch.zeros(B, 0, dtype=torch.bool, device=device)
            ctx_meta_add = self.meta.encode_context(
                ctx_meta_view, ctx_meta_modality, ctx_meta_phase, ctx_meta_quality
            )
            h_study, _ = self.st(
                ctx_elements=ctx_elements,
                ctx_meta_add=ctx_meta_add,
                ctx_pad_mask=ctx_pad_mask,
                tgt_meta_add=empty_tgt_meta,
                tgt_pad_mask=empty_tgt_pad,
            )
            return self.head(h_study)

    return B2()


__all__ = ["build_control_b2"]
