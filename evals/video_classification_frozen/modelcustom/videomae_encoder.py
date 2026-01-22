# evals/video_classification_frozen/modelcustom/videomae_encoder.py

import logging
from typing import Any, List, Tuple, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _collect_leaf_tensors(x: Any) -> List[torch.Tensor]:
    """
    Flattens nested list/tuple structures into a list of leaf tensors.
    Expected leaf shape: [B, C, T, H, W].
    """
    leaves: List[torch.Tensor] = []

    def rec(z: Any):
        if torch.is_tensor(z):
            leaves.append(z)
        elif isinstance(z, (list, tuple)):
            for zz in z:
                rec(zz)
        else:
            raise TypeError(f"Unsupported clip container type: {type(z)}")

    rec(x)
    if len(leaves) == 0:
        raise ValueError("No tensors found in clip container.")
    return leaves


class VideoMAEWrapper(nn.Module):
    """
    Wraps VideoMAE to satisfy V-JEPA2 eval API:
      - embed_dim attribute
      - forward(clips, clip_indices) returns Iterable[Tensor[B, N_tokens, D]]
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        # Most VideoMAE ViT variants expose embed_dim; otherwise infer from norm weight
        self.embed_dim = getattr(model, "embed_dim", None)
        if self.embed_dim is None:
            # Fallback inference (works for many ViT-like models)
            self.embed_dim = int(model.norm.weight.shape[0])

    @torch.no_grad()
    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B*, C, T, H, W]
        returns:
          - ideally [B*, N_tokens, D]
          - if model returns [B*, D], we convert to [B*, 1, D]
        """
        # Common VideoMAE APIs:
        #  - model.forward_features(x)
        #  - model.get_intermediate_layers(x)
        if hasattr(self.model, "forward_features"):
            feats = self.model.forward_features(x)
        else:
            # Some implementations return logits in forward(); avoid that if possible.
            feats = self.model(x)

        if feats.dim() == 2:
            feats = feats.unsqueeze(1)  # [B*, 1, D]
        if feats.dim() != 3:
            raise ValueError(f"Expected 2D or 3D features, got shape={tuple(feats.shape)}")
        return feats  # [B*, N, D]

    def forward(
        self,
        clips: Union[torch.Tensor, List[Any]],
        clip_indices=None,
    ) -> List[torch.Tensor]:
        """
        clips is the nested list structure coming from your eval.py dataloader:
          leaves are tensors of shape [B, C, T, H, W].
        We treat each leaf as one "view/segment clip", run VideoMAE on all leaves
        in a single batch, then return a list of per-clip features:
          out[i] = Tensor[B, N_tokens, D]
        """
        if torch.is_tensor(clips):
            # Rare path: already a tensor (assume single clip)
            x_flat = clips
            if x_flat.dim() == 4:
                # [B, T, H, W] not expected
                raise ValueError(f"Unexpected tensor rank for clips: {x_flat.shape}")
            feats = self._forward_features(x_flat)  # [B, N, D]
            return [feats]

        leaves = _collect_leaf_tensors(clips)
        B = int(leaves[0].shape[0])
        for t in leaves:
            if int(t.shape[0]) != B:
                raise ValueError("All clip leaves must share the same batch dimension.")

        # Concatenate along batch: [num_leaves*B, C, T, H, W]
        x_flat = torch.cat(leaves, dim=0)

        # VideoMAE forward -> [num_leaves*B, N_tokens, D]
        feats_flat = self._forward_features(x_flat)

        num_leaves = len(leaves)
        N_tokens = int(feats_flat.shape[1])
        D = int(feats_flat.shape[2])

        # Reshape back to [num_leaves, B, N_tokens, D] then to [B, num_leaves, N_tokens, D]
        feats = feats_flat.view(num_leaves, B, N_tokens, D).permute(1, 0, 2, 3).contiguous()

        # Return list length = num_leaves, each item [B, N_tokens, D]
        return [feats[:, i, :, :] for i in range(num_leaves)]


def init_module(
    resolution: int,
    frames_per_clip: int,
    checkpoint: str,
    model_kwargs: dict,
    wrapper_kwargs: dict,
):
    """
    V-JEPA2 eval entrypoint.

    model_kwargs should describe how to construct VideoMAE.
    Suggested YAML format:
      pretrain_kwargs:
        encoder:
          model_name: vit_large_patch16_224
          # (optional) other ctor kwargs
    """
    logger.info(f"Loading VideoMAE checkpoint from: {checkpoint}")

    enc_cfg = model_kwargs.get("encoder", model_kwargs)
    model_name = enc_cfg.get("model_name", None)
    if model_name is None:
        raise ValueError("VideoMAE config must include pretrain_kwargs.encoder.model_name")

    # Import your VideoMAE implementation
    # Adjust this import to match where you vendored the code.
    from third_party.videomae import modeling_finetune  # Option A
    # from modeling_finetune import ...                # Option B if sys.path insert is used

    if not hasattr(modeling_finetune, model_name):
        raise ValueError(f"modeling_finetune has no attribute {model_name}")

    ctor = getattr(modeling_finetune, model_name)

    # Instantiate model. Different VideoMAE forks accept different kwargs.
    # Keep it conservative; pass only common args unless you know your fork supports more.
    # Many VideoMAE ViT constructors accept img_size and num_frames; if yours doesn't, remove them.
    try:
        model = ctor(img_size=resolution, num_frames=frames_per_clip, **{k: v for k, v in enc_cfg.items() if k != "model_name"})
    except TypeError:
        # Fallback if the ctor signature is simpler
        model = ctor(**{k: v for k, v in enc_cfg.items() if k != "model_name"})

    ckpt = torch.load(checkpoint, map_location="cpu")
    state = ckpt.get("model", ckpt.get("state_dict", ckpt))

    # Strip common prefixes
    cleaned = {}
    for k, v in state.items():
        kk = k
        if kk.startswith("module."):
            kk = kk[len("module."):]
        cleaned[kk] = v

    msg = model.load_state_dict(cleaned, strict=False)
    logger.info(f"Loaded VideoMAE weights (strict=False). Missing={len(msg.missing_keys)} Unexpected={len(msg.unexpected_keys)}")

    # Freeze encoder weights
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    return VideoMAEWrapper(model)
