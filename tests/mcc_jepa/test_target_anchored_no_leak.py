"""With gamma=0, target-anchored MCC-JEPA must produce the same prediction
as plain B-visible V-JEPA — the adapter contributes exactly nothing at init.
"""

from __future__ import annotations

import torch

from app.vjepa_multiview.mcc_jepa_forward import _apply_adapter_to_predictor_out
from src.models.mcc_jepa import CrossClipAdapter


def test_adapter_is_noop_on_predictor_output_at_gamma_zero():
    torch.manual_seed(42)
    D = 64
    # Simulate two fpc entries, two mask-generators each.
    z_pred_base = [
        [torch.randn(2, 10, D), torch.randn(2, 8, D)],
        [torch.randn(2, 12, D)],
    ]
    # Source A tokens per fpc.
    a_src = [torch.randn(2, 30, D), torch.randn(2, 28, D)]

    adapter = CrossClipAdapter(embed_dim=D, num_heads=4)
    z_pred_anchored = _apply_adapter_to_predictor_out(z_pred_base, a_src, adapter)

    for base_fpc, anch_fpc in zip(z_pred_base, z_pred_anchored):
        for zb, za in zip(base_fpc, anch_fpc):
            assert torch.allclose(zb, za, atol=0.0)


def test_adapter_broadcasts_across_mask_generator_repeats():
    """When masks_x has k generators, the predictor's batch dim is B*k;
    the helper must repeat source A across mask-generator repetitions."""
    torch.manual_seed(0)
    D = 32
    B, k, N_tgt = 2, 3, 7
    # Predictor output shape: (B*k, N_tgt, D) for one fpc / one mask_i.
    z_pred_base = [[torch.randn(B * k, N_tgt, D)]]
    a_src = [torch.randn(B, 20, D)]
    adapter = CrossClipAdapter(embed_dim=D, num_heads=4)
    out = _apply_adapter_to_predictor_out(z_pred_base, a_src, adapter)
    assert out[0][0].shape == (B * k, N_tgt, D)
    # still identity at gamma=0
    assert torch.allclose(out[0][0], z_pred_base[0][0])
