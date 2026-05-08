"""Meta-token embedding lookups for EchoSet-JEPA Stage-2.

Vocabularies follow docs/echoset_jepa_plan.md §3.1 and §4.1. Each meta field
(view_family, modality, phase_bucket, quality_bucket) has an explicit <unknown>
id so metadata dropout (§4.2) can replace a token with <unknown> rather than
zeroing it out.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


VIEW_FAMILY_VOCAB: List[str] = [
    "apical",
    "parasternal_long",
    "parasternal_short",
    "subcostal",
    "suprasternal",
    "doppler_spectral",
    "m_mode",
    "tdi",
    "unknown",
]

MODALITY_VOCAB: List[str] = [
    "b_mode",
    "color_doppler",
    "pw_doppler",
    "cw_doppler",
    "m_mode",
    "tdi",
    "contrast",
    "unknown",
]

PHASE_BUCKET_VOCAB: List[str] = [
    "systolic",
    "diastolic",
    "full_cycle",
    "not_applicable",
    "unknown",
]

QUALITY_BUCKET_VOCAB: List[str] = ["high", "med", "low", "unknown"]


def _vocab_to_index(vocab: List[str]) -> Dict[str, int]:
    return {name: idx for idx, name in enumerate(vocab)}


@dataclass
class MetaDropout:
    """Per-field dropout probabilities applied on the context side only."""

    view: float = 0.15
    modality: float = 0.10
    phase: float = 0.30
    quality: float = 0.30


class MetaEmbeddings(nn.Module):
    """View / modality / phase / quality lookups with dropout-to-unknown policy.

    All embeddings are sized at ``d_model`` so they can be additively fused with
    the element representation (see plan §4.1).
    """

    def __init__(self, d_model: int, dropout: Optional[MetaDropout] = None) -> None:
        super().__init__()
        self.view = nn.Embedding(len(VIEW_FAMILY_VOCAB), d_model)
        self.modality = nn.Embedding(len(MODALITY_VOCAB), d_model)
        self.phase = nn.Embedding(len(PHASE_BUCKET_VOCAB), d_model)
        self.quality = nn.Embedding(len(QUALITY_BUCKET_VOCAB), d_model)

        self.view_idx = _vocab_to_index(VIEW_FAMILY_VOCAB)
        self.modality_idx = _vocab_to_index(MODALITY_VOCAB)
        self.phase_idx = _vocab_to_index(PHASE_BUCKET_VOCAB)
        self.quality_idx = _vocab_to_index(QUALITY_BUCKET_VOCAB)

        self.dropout_cfg = dropout or MetaDropout()

    def _apply_dropout(
        self,
        ids: torch.Tensor,
        unknown_id: int,
        p: float,
        training: bool,
    ) -> torch.Tensor:
        if not training or p <= 0.0:
            return ids
        mask = torch.rand_like(ids, dtype=torch.float32) < p
        return torch.where(mask, torch.full_like(ids, unknown_id), ids)

    def encode_context(
        self,
        view_ids: torch.Tensor,
        modality_ids: torch.Tensor,
        phase_ids: torch.Tensor,
        quality_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Return the additive meta-token contribution for context elements.

        Shapes: ``view_ids`` etc are ``(B, M_ctx)`` of long indices into each vocab.
        Returns ``(B, M_ctx, d_model)``.
        """
        view_ids = self._apply_dropout(view_ids, self.view_idx["unknown"], self.dropout_cfg.view, self.training)
        modality_ids = self._apply_dropout(
            modality_ids, self.modality_idx["unknown"], self.dropout_cfg.modality, self.training
        )
        phase_ids = self._apply_dropout(phase_ids, self.phase_idx["unknown"], self.dropout_cfg.phase, self.training)
        quality_ids = self._apply_dropout(
            quality_ids, self.quality_idx["unknown"], self.dropout_cfg.quality, self.training
        )
        return self.view(view_ids) + self.modality(modality_ids) + self.phase(phase_ids) + self.quality(quality_ids)

    def encode_target_slot(
        self,
        view_ids: torch.Tensor,
        modality_ids: torch.Tensor,
        phase_ids: Optional[torch.Tensor] = None,
        *,
        include_phase: bool = True,
        include_quality: bool = False,
        quality_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Meta tokens injected at the [MASK] slots.

        Target-side meta tokens are NEVER dropped (§3.4). Quality is excluded by
        default (`target_quality_token_ablation=false`); the ablation flag is
        wired through ``include_quality`` so tests can exercise the alternate
        path.
        """
        out = self.view(view_ids) + self.modality(modality_ids)
        if include_phase:
            if phase_ids is None:
                raise ValueError("phase_ids required when include_phase=True")
            out = out + self.phase(phase_ids)
        if include_quality:
            if quality_ids is None:
                raise ValueError("quality_ids required when include_quality=True")
            out = out + self.quality(quality_ids)
        return out

    # -- convenience lookups used by the dataset layer --------------------
    def view_id(self, name: str) -> int:
        return self.view_idx.get(name, self.view_idx["unknown"])

    def modality_id(self, name: str) -> int:
        return self.modality_idx.get(name, self.modality_idx["unknown"])

    def phase_id(self, name: str) -> int:
        return self.phase_idx.get(name, self.phase_idx["unknown"])

    def quality_id(self, name: str) -> int:
        return self.quality_idx.get(name, self.quality_idx["unknown"])


__all__ = [
    "VIEW_FAMILY_VOCAB",
    "MODALITY_VOCAB",
    "PHASE_BUCKET_VOCAB",
    "QUALITY_BUCKET_VOCAB",
    "MetaDropout",
    "MetaEmbeddings",
]
