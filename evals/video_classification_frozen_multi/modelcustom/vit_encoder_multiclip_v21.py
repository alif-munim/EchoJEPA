# evals/video_classification_frozen_multi/modelcustom/vit_encoder_multiclip_v21.py
"""
V-JEPA 2.1 ViT encoder adapter for multi-view frozen probe evaluation.

Uses the V-JEPA 2.1 VisionTransformer (app.vjepa_2_1) which differs from V-JEPA 2:
  - norms_block (4 hierarchical norms) instead of final LayerNorm
  - img_mod_embed / video_mod_embed (modality embeddings)
  - patch_embed_img (separate image patch embed)
  - RoPE computed on-the-fly (no pos_embed parameter)

API: same as vit_encoder_multiclip.py — returns list of tensors [B, N, D] (one per clip)
when return_per_clip=True, or merged temporal outputs otherwise.
"""

import logging

import torch
import torch.nn as nn

import app.vjepa_2_1.models.vision_transformer as vit21
from src.models.utils.pos_embs import get_1d_sincos_pos_embed

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def init_module(
    resolution: int,
    frames_per_clip: int,
    checkpoint: str,
    model_kwargs: dict,
    wrapper_kwargs: dict,
    device=None,
):
    logger.info(f"Loading V-JEPA 2.1 pretrained model from {checkpoint}")
    checkpoint = torch.load(checkpoint, map_location="cpu")

    enc_kwargs = model_kwargs["encoder"]
    enc_ckp_key = enc_kwargs.get("checkpoint_key")
    enc_model_name = enc_kwargs.get("model_name")

    model = vit21.__dict__[enc_model_name](
        img_size=resolution, num_frames=frames_per_clip, **enc_kwargs
    )

    pretrained_dict = checkpoint[enc_ckp_key]
    pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
    pretrained_dict = {k.replace("backbone.", ""): v for k, v in pretrained_dict.items()}

    for k, v in model.state_dict().items():
        if k not in pretrained_dict:
            logger.info(f'key "{k}" could not be found in loaded state dict')
        elif pretrained_dict[k].shape != v.shape:
            logger.info(f'key "{k}" is of different shape in model and loaded state dict')
            pretrained_dict[k] = v

    msg = model.load_state_dict(pretrained_dict, strict=False)
    logger.info(f"loaded pretrained model with msg: {msg}")

    model = ClipAggregation(
        model,
        tubelet_size=model.tubelet_size,
        **wrapper_kwargs,
    )
    del checkpoint
    return model


class ClipAggregation(nn.Module):
    """
    Wrapper that processes multiple clips.
    If return_per_clip=True, returns a list of tensors [B, N, D] (one per clip).
    If return_per_clip=False, concatenates/merges them (legacy behavior).
    """

    def __init__(
        self,
        model,
        tubelet_size=2,
        max_frames=128,
        use_pos_embed=False,
        return_per_clip=False,
    ):
        super().__init__()
        self.model = model
        self.tubelet_size = tubelet_size
        self.embed_dim = model.embed_dim
        self.num_heads = model.num_heads
        self.return_per_clip = return_per_clip

        self.pos_embed = None
        if use_pos_embed:
            max_T = max_frames // tubelet_size
            self.pos_embed = nn.Parameter(
                torch.zeros(1, max_T, self.embed_dim), requires_grad=False
            )
            sincos = get_1d_sincos_pos_embed(self.embed_dim, max_T)
            self.pos_embed.copy_(torch.from_numpy(sincos).float().unsqueeze(0))

    def forward(self, x, clip_indices=None):
        flat_clips = []
        for segment in x:
            for view in segment:
                flat_clips.append(view)

        batched_input = torch.cat(flat_clips, dim=0)
        all_tokens = self.model(batched_input)

        if self.return_per_clip:
            num_slots = len(flat_clips)
            return list(torch.chunk(all_tokens, num_slots, dim=0))

        # Legacy merged behavior
        num_clips = len(x)
        num_views_per_clip = len(x[0])
        B = x[0][0].shape[0]

        def multiviews_postprocess(outputs):
            _, N, D = outputs.size()
            F = x[0][0].shape[2]
            T = F // self.tubelet_size
            S = N // T

            outputs = outputs.view(num_clips * num_views_per_clip, B, N, D)
            final_outputs = []

            for v in range(num_views_per_clip):
                view_parts = []
                for c in range(num_clips):
                    idx = c * num_views_per_clip + v
                    part = outputs[idx].view(B, T, S, D)
                    view_parts.append(part)
                merged = torch.cat(view_parts, dim=1).flatten(1, 2)
                final_outputs.append(merged)

            return final_outputs

        return multiviews_postprocess(all_tokens)
