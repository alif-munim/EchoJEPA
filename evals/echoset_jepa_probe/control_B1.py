"""Control B1 — V-JEPA + lightweight supervised late-fusion attention (plan §7).

2-layer self-attention pool over the K ``c_clip`` embeddings (with view tokens
added), trained end-to-end with downstream labels only.

Params: d_model=256, n_heads=4, ~0.3M. Probe head is the shared ``StudyProbeHead``
with d_in=256.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from evals.echoset_jepa_probe.probe import StudyProbeHead


@dataclass
class ControlB1Config:
    d_clip: int = 1024
    d_model: int = 256
    n_layers: int = 2
    n_heads: int = 4
    ffn_mult: int = 2
    dropout: float = 0.1
    max_K: int = 16


class ControlB1Pool(nn.Module):
    def __init__(self, cfg: ControlB1Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.clip_in = nn.Linear(cfg.d_clip, cfg.d_model)
        self.view_emb = nn.Embedding(9, cfg.d_model)     # 9 view families incl. unknown
        self.study_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        nn.init.trunc_normal_(self.study_token, std=0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_model * cfg.ffn_mult,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_layers)
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, c_clip: torch.Tensor, view_ids: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        B = c_clip.shape[0]
        x = self.clip_in(c_clip) + self.view_emb(view_ids)
        study = self.study_token.expand(B, 1, -1)
        x = torch.cat([study, x], dim=1)
        study_pad = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
        full_pad = torch.cat([study_pad, pad_mask], dim=1)
        x = self.encoder(x, src_key_padding_mask=full_pad)
        return self.norm(x[:, 0, :])


def build_control_b1(cfg: ControlB1Config, n_targets: int) -> nn.Module:
    class B1(nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = ControlB1Pool(cfg)
            self.head = StudyProbeHead(cfg.d_model, n_targets)

        def forward(self, c_clip, view_ids, pad_mask):
            return self.head(self.pool(c_clip, view_ids, pad_mask))

    return B1()


__all__ = ["ControlB1Config", "ControlB1Pool", "build_control_b1"]
