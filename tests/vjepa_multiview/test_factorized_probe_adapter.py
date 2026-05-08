"""Fix 7: factorized probe adapter round-trip tests.

Build a synthetic MV2SV checkpoint (ViT-T backbone + FactorizedProjectionHead),
save + reload via ``vit_factorized_encoder.init_module``, and verify each
``feature_mode`` emits the right shape, uses the factorized head only when
appropriate, and keeps the encoder + head frozen.
"""

from __future__ import annotations

import copy
import sys
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from app.vjepa_multiview.factorized_head import FactorizedProjectionHead  # noqa: E402
import src.models.vision_transformer as vit  # noqa: E402
from evals.video_classification_frozen.modelcustom.vit_factorized_encoder import (  # noqa: E402
    FactorizedClipAggregation,
    init_module,
)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


# --- Fixtures -------------------------------------------------------------- #


def _build_synthetic_checkpoint(
    ckpt_path: Path,
    embed_dim: int = 192,
    shared_dim: int = 64,
    phase_dim: int = 64,
    view_dim: int = 64,
) -> dict:
    """Build a ViT-T target_encoder + FactorizedProjectionHead(ema) and
    save them in the MV2SV checkpoint format."""
    # Build a ViT-T encoder; we only care that state_dict keys match.
    encoder = vit.vit_tiny(img_size=64, num_frames=8, patch_size=16, tubelet_size=2)
    fh = FactorizedProjectionHead(
        embed_dim=embed_dim,
        hidden_dim=128,
        shared_dim=shared_dim,
        phase_dim=phase_dim,
        view_dim=view_dim,
    )
    fh_ema = copy.deepcopy(fh)
    ckpt = {
        "target_encoder": encoder.state_dict(),
        "encoder": encoder.state_dict(),
        "factorized_head": fh.state_dict(),
        "factorized_head_ema": fh_ema.state_dict(),
    }
    torch.save(ckpt, ckpt_path)
    return {
        "embed_dim": embed_dim,
        "shared_dim": shared_dim,
        "phase_dim": phase_dim,
        "view_dim": view_dim,
    }


def _model_kwargs(dims: dict, feature_mode: str, encoder_name: str = "vit_tiny") -> dict:
    enc_embed_dim = 192 if encoder_name == "vit_tiny" else 768
    return {
        "encoder": {
            "model_name": encoder_name,
            "checkpoint_key": "target_encoder",
            "patch_size": 16,
            "tubelet_size": 2,
            "uniform_power": False,
            "use_sdpa": False,
            "use_rope": False,
        },
        "factorized_head": {
            "feature_mode": feature_mode,
            "embed_dim": enc_embed_dim,
            "shared_dim": dims["shared_dim"],
            "phase_dim": dims["phase_dim"],
            "view_dim": dims["view_dim"],
            "head_hidden_dim": 128,
        },
    }


def _fake_input(B: int = 2, T: int = 8, H: int = 64, W: int = 64, num_clips: int = 2, num_views: int = 1):
    return [[torch.randn(B, 3, T, H, W) for _ in range(num_views)] for _ in range(num_clips)]


# --- Tests ----------------------------------------------------------------- #


def _load_adapter(tmp_path: Path, feature_mode: str):
    ckpt_path = tmp_path / "mv2sv.pt"
    # ViT-T has embed_dim 192 by default.
    dims = _build_synthetic_checkpoint(ckpt_path, embed_dim=192)
    mk = _model_kwargs(dims, feature_mode)
    wrapper = init_module(
        resolution=64,
        frames_per_clip=8,
        checkpoint=str(ckpt_path),
        model_kwargs=mk,
        wrapper_kwargs={"tubelet_size": 2, "use_pos_embed": False},
    )
    return wrapper, dims


def test_encoder_pool_mode_matches_legacy(tmp_path: Path):
    wrapper, _ = _load_adapter(tmp_path, "encoder_pool")
    assert wrapper.feature_mode == "encoder_pool"
    assert wrapper.factorized_head is None
    # encoder embed_dim is exposed.
    assert wrapper.embed_dim == wrapper.encoder_embed_dim
    # Forward through two clips, one spatial view.
    x = _fake_input(B=2, num_clips=2, num_views=1)
    out = wrapper(x)
    assert len(out) == 1
    # Shape [B, N_enc * num_clips, D_enc]. N_enc = (T/tubelet) * (H/p) * (W/p).
    # With T=8/2=4 temporal bins × 4×4 spatial patches = 64; × 2 clips = 128.
    assert out[0].shape == (2, 128, wrapper.encoder_embed_dim)


def test_z_shared_mode_shape_and_dim(tmp_path: Path):
    wrapper, dims = _load_adapter(tmp_path, "z_shared")
    assert wrapper.embed_dim == dims["shared_dim"]
    x = _fake_input(B=2, num_clips=3, num_views=1)
    out = wrapper(x)
    assert len(out) == 1
    # [B, num_clips, shared_dim]
    assert out[0].shape == (2, 3, dims["shared_dim"])
    assert torch.isfinite(out[0]).all()


def test_z_phase_mode_shape_and_dim(tmp_path: Path):
    wrapper, dims = _load_adapter(tmp_path, "z_phase")
    assert wrapper.embed_dim == dims["phase_dim"]
    x = _fake_input(B=2, num_clips=2)
    out = wrapper(x)
    assert out[0].shape == (2, 2, dims["phase_dim"])


def test_z_view_mode_shape_and_dim(tmp_path: Path):
    wrapper, dims = _load_adapter(tmp_path, "z_view")
    assert wrapper.embed_dim == dims["view_dim"]
    x = _fake_input(B=2, num_clips=2)
    out = wrapper(x)
    assert out[0].shape == (2, 2, dims["view_dim"])


def test_concat_shared_phase_mode(tmp_path: Path):
    wrapper, dims = _load_adapter(tmp_path, "concat_shared_phase")
    assert wrapper.embed_dim == dims["shared_dim"] + dims["phase_dim"]
    x = _fake_input(B=2, num_clips=2)
    out = wrapper(x)
    assert out[0].shape == (2, 2, dims["shared_dim"] + dims["phase_dim"])


def test_concat_all_mode(tmp_path: Path):
    wrapper, dims = _load_adapter(tmp_path, "concat_all")
    assert wrapper.embed_dim == dims["shared_dim"] + dims["phase_dim"] + dims["view_dim"]
    x = _fake_input(B=2, num_clips=2)
    out = wrapper(x)
    assert out[0].shape == (
        2,
        2,
        dims["shared_dim"] + dims["phase_dim"] + dims["view_dim"],
    )


def test_multiview_spatial_outputs(tmp_path: Path):
    """num_views_per_clip > 1 → list of that length."""
    wrapper, dims = _load_adapter(tmp_path, "z_shared")
    x = _fake_input(B=2, num_clips=2, num_views=3)
    out = wrapper(x)
    assert len(out) == 3
    for o in out:
        assert o.shape == (2, 2, dims["shared_dim"])


def test_factorized_head_params_frozen_after_init(tmp_path: Path):
    wrapper, _ = _load_adapter(tmp_path, "z_shared")
    for p in wrapper.factorized_head.parameters():
        assert p.requires_grad is False


def test_slot_output_is_detached_from_head(tmp_path: Path):
    """Slot path runs the frozen head under no_grad → output must be
    detached and have no grad_fn pointing back through the head.
    Downstream probes need this so any trainable probe head they place
    after the adapter doesn't accidentally route grads into the
    factorized head."""
    wrapper, _ = _load_adapter(tmp_path, "z_shared")
    x = _fake_input(B=2, num_clips=2)
    out = wrapper(x)
    # Output of the slot path is detached — no grad tracking.
    assert out[0].requires_grad is False
    assert out[0].grad_fn is None
    # Head params stay without grad either way.
    for name, p in wrapper.factorized_head.named_parameters():
        assert p.grad is None, f"factorized_head {name} got grad"


def test_unknown_feature_mode_rejected(tmp_path: Path):
    ckpt_path = tmp_path / "mv2sv.pt"
    dims = _build_synthetic_checkpoint(ckpt_path)
    mk = _model_kwargs(dims, "bogus_mode")
    with pytest.raises(ValueError, match="feature_mode"):
        init_module(
            resolution=64,
            frames_per_clip=8,
            checkpoint=str(ckpt_path),
            model_kwargs=mk,
            wrapper_kwargs={"tubelet_size": 2, "use_pos_embed": False},
        )


def test_missing_mv2sv_keys_rejected(tmp_path: Path):
    """Checkpoint without factorized_head keys must reject non-encoder_pool modes."""
    ckpt_path = tmp_path / "legacy.pt"
    encoder = vit.vit_tiny(img_size=64, num_frames=8, patch_size=16, tubelet_size=2)
    torch.save({"target_encoder": encoder.state_dict()}, ckpt_path)
    mk = _model_kwargs({"shared_dim": 64, "phase_dim": 64, "view_dim": 64}, "z_shared")
    with pytest.raises(KeyError, match="factorized_head"):
        init_module(
            resolution=64,
            frames_per_clip=8,
            checkpoint=str(ckpt_path),
            model_kwargs=mk,
            wrapper_kwargs={"tubelet_size": 2, "use_pos_embed": False},
        )


def test_missing_mv2sv_keys_ok_for_encoder_pool(tmp_path: Path):
    """Legacy checkpoint (no factorized_head) still works in encoder_pool mode."""
    ckpt_path = tmp_path / "legacy.pt"
    encoder = vit.vit_tiny(img_size=64, num_frames=8, patch_size=16, tubelet_size=2)
    torch.save({"target_encoder": encoder.state_dict()}, ckpt_path)
    mk = _model_kwargs({"shared_dim": 64, "phase_dim": 64, "view_dim": 64}, "encoder_pool")
    wrapper = init_module(
        resolution=64,
        frames_per_clip=8,
        checkpoint=str(ckpt_path),
        model_kwargs=mk,
        wrapper_kwargs={"tubelet_size": 2, "use_pos_embed": False},
    )
    assert wrapper.factorized_head is None
    assert wrapper.embed_dim == wrapper.encoder_embed_dim


def test_prefers_ema_over_online_head(tmp_path: Path):
    """When both factorized_head and factorized_head_ema are present,
    the adapter loads the EMA copy (the target-side weights the
    student was trained against)."""
    ckpt_path = tmp_path / "mv2sv.pt"
    dims = _build_synthetic_checkpoint(ckpt_path)

    # Load the ckpt, poison the online head with a recognizable pattern,
    # keep the EMA head at its normal init.
    ckpt = torch.load(ckpt_path, map_location="cpu")
    for k in ckpt["factorized_head"]:
        ckpt["factorized_head"][k] = (
            ckpt["factorized_head"][k] * 0.0 + 42.0
            if ckpt["factorized_head"][k].dtype.is_floating_point
            else ckpt["factorized_head"][k]
        )
    torch.save(ckpt, ckpt_path)

    mk = _model_kwargs(dims, "z_shared")
    wrapper = init_module(
        resolution=64,
        frames_per_clip=8,
        checkpoint=str(ckpt_path),
        model_kwargs=mk,
        wrapper_kwargs={"tubelet_size": 2, "use_pos_embed": False},
    )
    # The EMA was untouched → weights should NOT be 42.0 everywhere.
    for n, p in wrapper.factorized_head.named_parameters():
        if p.dtype.is_floating_point:
            assert not torch.allclose(p, torch.full_like(p, 42.0))
