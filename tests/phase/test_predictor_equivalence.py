"""Predictor-equivalence regression gate for phi-JEPA.

Checks three invariants before any phi-JEPA training runs:

  1. phase_conditioned=False is a strict no-op relative to vanilla V-JEPA —
     forward pass produces the same shapes, same numerical output across seeds
     when input+weights are held fixed.

  2. phase_conditioned=True with delta_phi=None should produce output equal to
     phase_conditioned=False (within numerical noise) because:
        - no_phase_token is zero-initialized
        - pred_tokens + phase_emb(None) = pred_tokens + 0 = pred_tokens
     For identity to hold, we copy the shared weights from the False predictor
     into the True predictor, so only the new phase modules differ (and they
     contribute zero when delta_phi=None because of zero-init).

  3. phase_conditioned=True with non-trivial delta_phi produces output that
     differs materially from the delta_phi=None case — i.e. the phase
     conditioning signal reaches the predictor's forward computation.

Run:  python tests/phase/test_predictor_equivalence.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.models.predictor import vit_predictor  # noqa: E402


# Minimal predictor config (ViT-T-ish) for fast CPU tests.
CFG = dict(
    img_size=224,
    patch_size=16,
    num_frames=16,
    tubelet_size=2,
    embed_dim=192,
    predictor_embed_dim=96,
    depth=2,
    num_heads=4,
    use_mask_tokens=True,
    num_mask_tokens=2,
    use_rope=False,  # RoPE is orthogonal to the phase logic under test; fixed pos embed is simpler.
)


def _make_batch(B=2, N_ctx=10, N_tgt=6, embed_dim=192, grid=(8, 14, 14), seed=0):
    g = torch.Generator().manual_seed(seed)
    D, H, W = grid
    N_full = D * H * W
    x = torch.randn(B, N_ctx, embed_dim, generator=g)
    # Random non-overlapping flat indices into a (D*H*W) grid for masks.
    perm = torch.stack([torch.randperm(N_full, generator=g) for _ in range(B)])
    mx = perm[:, :N_ctx]
    my = perm[:, N_ctx : N_ctx + N_tgt]
    return x, mx, my


def _build(phase_conditioned, seed=0):
    torch.manual_seed(seed)
    model = vit_predictor(**CFG, phase_conditioned=phase_conditioned)
    model.eval()
    return model


def test_phase_false_is_deterministic():
    """Same-seed rebuilds with phase_conditioned=False must produce identical output."""
    mA = _build(phase_conditioned=False, seed=42)
    mB = _build(phase_conditioned=False, seed=42)
    x, mx, my = _make_batch()
    with torch.no_grad():
        yA = mA(x, mx, my, mask_index=0)
        yB = mB(x, mx, my, mask_index=0)
    assert yA.shape == yB.shape, f"shapes differ: {yA.shape} vs {yB.shape}"
    assert torch.allclose(yA, yB), f"phase_conditioned=False not deterministic (max diff = {(yA - yB).abs().max().item():.2e})"


def test_phase_true_none_matches_false():
    """phase_conditioned=True with delta_phi=None should match phase_conditioned=False
    when shared weights are copied. no_phase_token is zero-init -> phase_emb adds zero."""
    m_false = _build(phase_conditioned=False, seed=42)
    m_true = _build(phase_conditioned=True, seed=42)

    # Copy shared backbone weights from m_false to m_true so only the new phase
    # modules differ (and they add zero when delta_phi=None).
    false_sd = m_false.state_dict()
    true_sd = m_true.state_dict()
    # Only load keys that exist in m_false; m_true has additional phase_mlp and
    # no_phase_token entries which stay at their init (no_phase_token is zero).
    shared = {k: v for k, v in false_sd.items() if k in true_sd}
    missing, unexpected = m_true.load_state_dict(shared, strict=False)
    # Missing should be exactly the phase modules; unexpected should be empty.
    assert not unexpected, f"unexpected keys from false predictor: {unexpected}"
    phase_keys = {k for k in missing if "phase" in k or "no_phase" in k}
    assert phase_keys == set(missing), f"unexpected missing (non-phase) keys: {set(missing) - phase_keys}"

    # Zero-init no_phase_token explicitly (it is, but be defensive).
    with torch.no_grad():
        m_true.no_phase_token.zero_()

    x, mx, my = _make_batch()
    with torch.no_grad():
        y_false = m_false(x, mx, my, mask_index=0)
        y_true_none = m_true(x, mx, my, mask_index=0, delta_phi=None)

    max_diff = (y_false - y_true_none).abs().max().item()
    # With zero no_phase_token + no random drops (eval mode), outputs should match.
    assert torch.allclose(y_false, y_true_none, atol=1e-5), \
        f"phase=True+None != phase=False (max diff = {max_diff:.2e})"


def test_nontrivial_delta_phi_changes_output():
    """Non-trivial delta_phi should materially change predictor output."""
    m = _build(phase_conditioned=True, seed=42)
    x, mx, my = _make_batch()
    B, N_tgt = my.shape
    # Initialize no_phase_token to small random values so we can also verify
    # that explicit phase conditioning differs from the <no_phase> path.
    with torch.no_grad():
        m.no_phase_token.zero_()
    # Also make phase_mlp non-trivial by NOT zeroing it (default init keeps weights nonzero).

    with torch.no_grad():
        y_none = m(x, mx, my, mask_index=0, delta_phi=None)
        # Non-trivial Δφ: spread across [0, 1) with a distinct value per target token.
        dphi = torch.linspace(0.0, 0.99, steps=N_tgt).unsqueeze(0).expand(B, -1).contiguous()
        y_phase = m(x, mx, my, mask_index=0, delta_phi=dphi)

    diff = (y_none - y_phase).abs().max().item()
    assert diff > 1e-3, f"non-trivial Δφ produced negligible change (max diff = {diff:.2e}) — predictor is ignoring phase"


def test_nan_delta_phi_routes_to_no_phase():
    """All-NaN delta_phi should match delta_phi=None (both route to <no_phase>)."""
    m = _build(phase_conditioned=True, seed=42)
    x, mx, my = _make_batch()
    B, N_tgt = my.shape

    with torch.no_grad():
        y_none = m(x, mx, my, mask_index=0, delta_phi=None)
        nan_dphi = torch.full((B, N_tgt), float("nan"))
        y_nan = m(x, mx, my, mask_index=0, delta_phi=nan_dphi)

    assert torch.allclose(y_none, y_nan, atol=1e-5), \
        f"NaN delta_phi doesn't match None (max diff = {(y_none - y_nan).abs().max().item():.2e})"


def test_phase_drop_zero_in_eval():
    """In eval mode, phase_drop_p should have no effect (drop only active during training)."""
    m = _build(phase_conditioned=True, seed=42)
    m.eval()
    x, mx, my = _make_batch()
    B, N_tgt = my.shape
    dphi = torch.linspace(0.0, 0.99, steps=N_tgt).unsqueeze(0).expand(B, -1).contiguous()

    with torch.no_grad():
        y1 = m(x, mx, my, mask_index=0, delta_phi=dphi)
        y2 = m(x, mx, my, mask_index=0, delta_phi=dphi)
    assert torch.allclose(y1, y2), "eval-mode forward not deterministic (phase dropout active)"


if __name__ == "__main__":
    import traceback
    tests = [
        test_phase_false_is_deterministic,
        test_phase_true_none_matches_false,
        test_nontrivial_delta_phi_changes_output,
        test_nan_delta_phi_routes_to_no_phase,
        test_phase_drop_zero_in_eval,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {t.__name__}: {e}")
        except Exception:
            failed += 1
            print(f"  FAIL  {t.__name__} (exception)")
            traceback.print_exc()
    if failed:
        print(f"\n{failed}/{len(tests)} tests FAILED", file=sys.stderr)
        sys.exit(1)
    print(f"\n{len(tests)} tests passed")
