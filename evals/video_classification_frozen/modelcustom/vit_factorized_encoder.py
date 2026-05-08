"""Factorized-slot probe adapter for MV2SV (v4) checkpoints.

Loads both ``target_encoder`` (standard ViT) and ``factorized_head``
(FactorizedProjectionHead) from an MV2SV checkpoint, and exposes a
probe feature mode that selects which latent to probe:

    feature_mode          | embed_dim               | token sequence
    --------------------- | ----------------------- | ------------------
    encoder_pool (legacy) | encoder.embed_dim       | [B, N_enc, D_enc]
    z_shared              | shared_dim              | [B, 1, shared_dim]
    z_phase               | phase_dim               | [B, 1, phase_dim]
    z_view                | view_dim                | [B, 1, view_dim]
    concat_shared_phase   | shared_dim + phase_dim  | [B, 1, D_cat]
    concat_all            | s + p + v               | [B, 1, D_cat]

The slot paths emit a length-1 sequence so the existing AttentivePooler
can consume them unchanged — at ``num_probe_blocks=1`` the pooler is a
single cross-attention over N≥1 tokens, which works for N=1.

The encoder_pool path delegates to the same ``ClipAggregation``
wrapper used by ``vit_encoder_multiclip``, preserving bit-identical
behaviour for legacy probes.
"""

import logging

import torch
import torch.nn as nn

import src.models.vision_transformer as vit
from src.masks.utils import apply_masks
from src.models.utils.pos_embs import get_1d_sincos_pos_embed

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


_VALID_FEATURE_MODES = (
    "encoder_pool",
    "z_shared",
    "z_phase",
    "z_view",
    "concat_shared_phase",
    "concat_all",
)


def init_module(
    resolution: int,
    frames_per_clip: int,
    checkpoint: str,
    # --
    model_kwargs: dict,
    wrapper_kwargs: dict,
):
    """Load a ViT encoder + optional FactorizedProjectionHead from an
    MV2SV checkpoint.

    ``model_kwargs["encoder"]`` mirrors the vit_encoder_multiclip
    contract. A new optional block is read:

        model_kwargs["factorized_head"] = {
            "feature_mode": "z_shared" | "z_phase" | ... | "encoder_pool",
            "embed_dim": 1024,
            "shared_dim": 256,
            "phase_dim": 256,
            "view_dim": 256,
            "head_hidden_dim": 1024,
        }

    If the feature_mode is ``encoder_pool`` or the factorized_head
    block is absent, the head is not constructed — behaviour is
    identical to vit_encoder_multiclip.
    """
    logger.info(f"Loading pretrained model from {checkpoint}")
    ckpt = torch.load(checkpoint, map_location="cpu")

    enc_kwargs = model_kwargs["encoder"]
    enc_ckp_key = enc_kwargs.get("checkpoint_key", "target_encoder")
    enc_model_name = enc_kwargs.get("model_name")

    model = vit.__dict__[enc_model_name](img_size=resolution, num_frames=frames_per_clip, **enc_kwargs)

    pretrained_dict = ckpt[enc_ckp_key]
    pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
    pretrained_dict = {k.replace("backbone.", ""): v for k, v in pretrained_dict.items()}
    for k, v in model.state_dict().items():
        if k not in pretrained_dict:
            logger.info(f'key "{k}" could not be found in loaded state dict')
        elif pretrained_dict[k].shape != v.shape:
            logger.info(f'key "{k}" is of different shape in model and loaded state dict')
            pretrained_dict[k] = v
    msg = model.load_state_dict(pretrained_dict, strict=False)
    logger.info(f"loaded encoder with msg: {msg}")

    fh_cfg = model_kwargs.get("factorized_head")
    feature_mode = "encoder_pool"
    factorized_head = None
    if fh_cfg is not None:
        feature_mode = str(fh_cfg.get("feature_mode", "encoder_pool"))
        if feature_mode not in _VALID_FEATURE_MODES:
            raise ValueError(f"feature_mode={feature_mode!r}; want one of {_VALID_FEATURE_MODES}")
        if feature_mode != "encoder_pool":
            # Defer import so the adapter is usable without MV2SV installed.
            from app.vjepa_multiview.factorized_head import FactorizedProjectionHead

            factorized_head = FactorizedProjectionHead(
                embed_dim=int(fh_cfg.get("embed_dim", 1024)),
                hidden_dim=int(fh_cfg.get("head_hidden_dim", 1024)),
                shared_dim=int(fh_cfg.get("shared_dim", 256)),
                phase_dim=int(fh_cfg.get("phase_dim", 256)),
                view_dim=int(fh_cfg.get("view_dim", 256)),
            )
            # Prefer the EMA head for probing — it's the target-side
            # weights the student was trained against, and matches what
            # a "teacher-oracle" readout would see. Fall back to the
            # online head if EMA wasn't saved (e.g. a v3 checkpoint).
            fh_key = "factorized_head_ema" if "factorized_head_ema" in ckpt else "factorized_head"
            if fh_key not in ckpt:
                raise KeyError(
                    f"checkpoint missing MV2SV head key; expected "
                    f"'factorized_head_ema' or 'factorized_head'. "
                    f"Use feature_mode='encoder_pool' for non-MV2SV ckpts."
                )
            fh_state = {k.replace("module.", ""): v for k, v in ckpt[fh_key].items()}
            msg = factorized_head.load_state_dict(fh_state, strict=False)
            logger.info(f"loaded {fh_key} with msg: {msg}")
            # Freeze the head — it is a probe feature extractor, not a
            # trainable module.
            for p in factorized_head.parameters():
                p.requires_grad = False
            factorized_head.eval()

    # wrapper_kwargs may supply tubelet_size; if it does, honor it.
    # Otherwise fall back to the model's own tubelet_size.
    wk = dict(wrapper_kwargs or {})
    wk.setdefault("tubelet_size", model.tubelet_size)
    wrapper = FactorizedClipAggregation(
        model,
        factorized_head=factorized_head,
        feature_mode=feature_mode,
        **wk,
    )
    del ckpt
    return wrapper


def _slot_dim(fh_cfg: dict | None, mode: str) -> int:
    if fh_cfg is None:
        raise ValueError("factorized_head config missing for slot mode")
    s = int(fh_cfg.get("shared_dim", 256))
    p = int(fh_cfg.get("phase_dim", 256))
    v = int(fh_cfg.get("view_dim", 256))
    if mode == "z_shared":
        return s
    if mode == "z_phase":
        return p
    if mode == "z_view":
        return v
    if mode == "concat_shared_phase":
        return s + p
    if mode == "concat_all":
        return s + p + v
    raise ValueError(f"no slot_dim for mode={mode!r}")


class FactorizedClipAggregation(nn.Module):
    """Same as the legacy ClipAggregation when ``feature_mode`` is
    ``encoder_pool``; otherwise mean-pools the encoder output and runs
    the frozen factorized head, emitting a ``[B, 1, slot_dim]`` token
    sequence per spatial view.
    """

    def __init__(
        self,
        model: nn.Module,
        factorized_head: nn.Module | None,
        feature_mode: str = "encoder_pool",
        tubelet_size: int = 2,
        max_frames: int = 128,
        use_pos_embed: bool = False,
    ):
        super().__init__()
        if feature_mode not in _VALID_FEATURE_MODES:
            raise ValueError(f"feature_mode={feature_mode!r}")
        if feature_mode != "encoder_pool" and factorized_head is None:
            raise ValueError(f"feature_mode={feature_mode!r} requires factorized_head")
        self.model = model
        self.factorized_head = factorized_head
        self.feature_mode = feature_mode
        self.tubelet_size = tubelet_size
        self.encoder_embed_dim = model.embed_dim
        self.num_heads = model.num_heads

        # embed_dim exposed to the downstream probe.
        if feature_mode == "encoder_pool":
            self.embed_dim = self.encoder_embed_dim
        elif feature_mode == "z_shared":
            self.embed_dim = factorized_head.shared_dim
        elif feature_mode == "z_phase":
            self.embed_dim = factorized_head.phase_dim
        elif feature_mode == "z_view":
            self.embed_dim = factorized_head.view_dim
        elif feature_mode == "concat_shared_phase":
            self.embed_dim = factorized_head.shared_dim + factorized_head.phase_dim
        elif feature_mode == "concat_all":
            self.embed_dim = factorized_head.shared_dim + factorized_head.phase_dim + factorized_head.view_dim

        self.pos_embed = None
        if use_pos_embed and feature_mode == "encoder_pool":
            max_T = max_frames // tubelet_size
            self.pos_embed = nn.Parameter(torch.zeros(1, max_T, self.embed_dim), requires_grad=False)
            sincos = get_1d_sincos_pos_embed(self.embed_dim, max_T)
            self.pos_embed.copy_(torch.from_numpy(sincos).float().unsqueeze(0))

    def _slot_from_pooled(self, pooled: torch.Tensor) -> torch.Tensor:
        """pooled: [B, encoder_embed_dim] → [B, embed_dim] slot vector."""
        with torch.no_grad():
            slots = self.factorized_head(pooled)
        if self.feature_mode == "z_shared":
            return slots["z_shared"]
        if self.feature_mode == "z_phase":
            return slots["z_phase"]
        if self.feature_mode == "z_view":
            return slots["z_view"]
        if self.feature_mode == "concat_shared_phase":
            return torch.cat([slots["z_shared"], slots["z_phase"]], dim=-1)
        if self.feature_mode == "concat_all":
            return torch.cat([slots["z_shared"], slots["z_phase"], slots["z_view"]], dim=-1)
        raise AssertionError(f"unreachable: mode={self.feature_mode}")

    def forward(self, x, clip_indices=None):
        """Same outer interface as ClipAggregation.

        x: list-over-clips of list-over-spatial-views of [B, C, F, H, W].
        Returns list-over-spatial-views of [B, N_tok, embed_dim].

        For slot modes, N_tok == num_clips (one pooled slot per clip,
        concatenated along the token dim). For encoder_pool this
        returns the same as the legacy adapter (length N_enc * num_clips).
        """
        num_clips = len(x)
        num_views_per_clip = len(x[0])
        B, C, F, H, W = x[0][0].size()

        x_cat = [torch.cat(xi, dim=0) for xi in x]
        x_cat = torch.cat(x_cat, dim=0)

        outputs = self.model(x_cat)

        if self.feature_mode == "encoder_pool":
            return self._encoder_pool_postprocess(
                outputs,
                num_clips,
                num_views_per_clip,
                B,
                F,
                clip_indices,
            )
        return self._slot_postprocess(outputs, num_clips, num_views_per_clip, B)

    def _encoder_pool_postprocess(self, outputs, num_clips, num_views_per_clip, B, F, clip_indices):
        _, N, D = outputs.size()
        T = F // self.tubelet_size
        S = N // T
        eff_B = B * num_views_per_clip
        all_outputs = [[] for _ in range(num_views_per_clip)]
        for i in range(num_clips):
            o = outputs[i * eff_B : (i + 1) * eff_B]
            for j in range(num_views_per_clip):
                all_outputs[j].append(o[j * B : (j + 1) * B])
        for i, outs in enumerate(all_outputs):
            outs = [o.reshape(B, T, S, D) for o in outs]
            outs = torch.cat(outs, dim=1).flatten(1, 2)
            if (self.pos_embed is not None) and (clip_indices is not None):
                _indices = [c[:, :: self.tubelet_size] for c in clip_indices]
                pos_embed = self.pos_embed.repeat(B, 1, 1)
                pos_embed = apply_masks(pos_embed, _indices, concat=False)
                pos_embed = torch.cat(pos_embed, dim=1)
                pos_embed = pos_embed.unsqueeze(2).repeat(1, 1, S, 1)
                pos_embed = pos_embed.flatten(1, 2)
                outs = outs + pos_embed
            all_outputs[i] = outs
        return all_outputs

    def _slot_postprocess(self, outputs, num_clips, num_views_per_clip, B):
        """Mean-pool encoder tokens per clip, run factorized head,
        emit per-spatial-view [B, num_clips, slot_dim] tensors."""
        # outputs: [num_clips * num_views_per_clip * B, N_enc, D_enc]
        # Mean-pool over N_enc to get [eff_B_total, D_enc].
        pooled = outputs.mean(dim=1)
        D_enc = pooled.size(-1)
        eff_B = B * num_views_per_clip
        all_outputs = [[] for _ in range(num_views_per_clip)]
        for i in range(num_clips):
            p_i = pooled[i * eff_B : (i + 1) * eff_B]  # [eff_B, D_enc]
            for j in range(num_views_per_clip):
                pj = p_i[j * B : (j + 1) * B]  # [B, D_enc]
                slot = self._slot_from_pooled(pj)  # [B, slot_dim]
                all_outputs[j].append(slot.unsqueeze(1))  # [B, 1, slot_dim]
        # Concatenate along the token dim → [B, num_clips, slot_dim].
        for j in range(num_views_per_clip):
            all_outputs[j] = torch.cat(all_outputs[j], dim=1)
        return all_outputs
