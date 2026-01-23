# evals/video_classification_frozen/modelcustom/echo_prime_encoder.py

import os
import logging
from typing import Any, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _collect_leaf_tensors(x: Any) -> List[torch.Tensor]:
    """Flatten nested list/tuple structures into a list of leaf tensors [B, C, T, H, W]."""
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


class EchoPrimeEncoderOnlyWrapper(nn.Module):
    """
    EchoPrime encoder-only wrapper that returns per-clip embeddings compatible with V-JEPA2 eval.

    Output embedding per video = concat([mvit_512], [view_onehot_11]) => D=523.
    Returns a List[Tensor], one entry per leaf clip, each shaped [B, 1, 523].
    """

    def __init__(
        self,
        echo_encoder: nn.Module,
        view_classifier: nn.Module,
        force_fp32: bool = True,
        bin_size: int = 50,
    ):
        super().__init__()
        self.echo_encoder = echo_encoder
        self.view_classifier = view_classifier
        self.force_fp32 = force_fp32
        self.bin_size = int(bin_size)

        # EchoPrime normalization constants (0..255 pixel space), shaped for [C, T, H, W]
        # Source: echonet/EchoPrime echo_prime/model.py
        mean = torch.tensor([29.110628, 28.076836, 29.096405]).reshape(3, 1, 1, 1)
        std = torch.tensor([47.989223, 46.456997, 47.20083]).reshape(3, 1, 1, 1)
        self.register_buffer("mean_255", mean, persistent=False)
        self.register_buffer("std_255", std, persistent=False)

        self.embed_dim = 523  # required by V-JEPA2 probe init

        # Freeze explicitly (eval loop also uses no_grad, but keep it clean)
        self.eval()
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def _normalize_like_echoprime(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [N, C, T, H, W], typically in 0..1 after V-JEPA2 transforms.
        EchoPrime normalizes in 0..255 space: x.sub_(mean).div_(std).
        """
        # Heuristic: if already looks like 0..255, don't rescale.
        # If it looks like 0..1 (or ImageNet-ish), rescale to 0..255.
        if x.dtype.is_floating_point:
            xmax = float(x.detach().max().cpu())
        else:
            xmax = 255.0

        if xmax <= 5.0:
            x = x * 255.0

        x = x.float()
        x = (x - self.mean_255) / self.std_255
        return x

    @torch.no_grad()
    def _embed_videos_batched(self, vids: torch.Tensor) -> torch.Tensor:
        """
        vids: [N, 3, 16, 224, 224] normalized like EchoPrime
        returns: [N, 512]
        Uses binning like upstream EchoPrime to avoid OOM.
        """
        N = int(vids.shape[0])
        feats = []
        for start in range(0, N, self.bin_size):
            end = min(start + self.bin_size, N)
            chunk = vids[start:end]
            feats.append(self.echo_encoder(chunk))
        return torch.cat(feats, dim=0)

    def forward(self, clips: Union[torch.Tensor, List[Any]], clip_indices=None) -> List[torch.Tensor]:
        # Collect leaves
        if torch.is_tensor(clips):
            leaves = [clips]
        else:
            leaves = _collect_leaf_tensors(clips)

        B = int(leaves[0].shape[0])
        for t in leaves:
            if int(t.shape[0]) != B:
                raise ValueError("All clip leaves must share the same batch dimension.")

        # Flatten leaves along batch: [L*B, C, T, H, W]
        x_flat = torch.cat(leaves, dim=0)

        # Ensure exactly 16 frames (EchoPrime expects 16 after its internal stride scheme)
        # If you always set frames_per_clip=16 in YAML, this is a no-op.
        C, T = int(x_flat.shape[1]), int(x_flat.shape[2])
        if T != 16:
            if T > 16:
                idx = torch.linspace(0, T - 1, steps=16, device=x_flat.device).round().long()
                x_flat = x_flat.index_select(2, idx)
            else:
                pad = torch.zeros((x_flat.shape[0], C, 16 - T, x_flat.shape[3], x_flat.shape[4]),
                                  device=x_flat.device, dtype=x_flat.dtype)
                x_flat = torch.cat([x_flat, pad], dim=2)

        # Normalize to EchoPrime expected space
        x_flat = self._normalize_like_echoprime(x_flat)

        # Disable autocast for safety if requested
        if self.force_fp32 and x_flat.is_cuda:
            with torch.autocast(device_type="cuda", enabled=False):
                vid_feats = self._embed_videos_batched(x_flat)  # [L*B, 512]
                first_frames = x_flat[:, :, 0, :, :]            # [L*B, 3, H, W]
                view_logits = self.view_classifier(first_frames)
        else:
            vid_feats = self._embed_videos_batched(x_flat)
            first_frames = x_flat[:, :, 0, :, :]
            view_logits = self.view_classifier(first_frames)

        view_ids = view_logits.argmax(dim=1)
        view_oh = F.one_hot(view_ids, num_classes=11).float()  # [L*B, 11]

        emb = torch.cat([vid_feats.float(), view_oh], dim=1)   # [L*B, 523]
        emb = emb.unsqueeze(1)                                 # [L*B, 1, 523]

        # Unflatten back to per-leaf list: feats_per_leaf[i] = [B, 1, 523]
        L = len(leaves)
        emb = emb.view(L, B, 1, self.embed_dim).permute(1, 0, 2, 3).contiguous()  # [B, L, 1, 523]
        return [emb[:, i, :, :] for i in range(L)]


def init_module(
    resolution: int,
    frames_per_clip: int,
    checkpoint: str,
    model_kwargs: dict,
    wrapper_kwargs: dict,
):
    """
    V-JEPA2 modelcustom entrypoint.

    You must provide paths to the EchoPrime weights. Defaults try to find a vendored folder:
      evals/video_classification_frozen/modelcustom/EchoPrime/model_data/weights/...

    wrapper_kwargs supports:
      - echo_prime_root (str)
      - encoder_ckpt (str)
      - view_ckpt (str)
      - force_fp32 (bool, default True)
      - bin_size (int, default 50)
    """
    if resolution != 224:
        logger.warning(f"EchoPrime was trained on 224; got resolution={resolution}. Proceeding anyway.")
    if frames_per_clip != 16:
        logger.warning(f"EchoPrime expects 16 frames; got frames_per_clip={frames_per_clip}. Wrapper will resample/pad.")

    this_dir = os.path.dirname(os.path.abspath(__file__))
    default_root = os.path.join(this_dir, "EchoPrime")

    echo_root = wrapper_kwargs.get("echo_prime_root", None)
    if echo_root is None:
        echo_root = os.environ.get("ECHOPRIME_ROOT", None)
    if echo_root is None and os.path.isdir(default_root):
        echo_root = default_root

    # Resolve ckpt paths
    encoder_ckpt = wrapper_kwargs.get("encoder_ckpt", None)
    view_ckpt = wrapper_kwargs.get("view_ckpt", None)

    # Allow "checkpoint" to directly be the encoder weight path
    if encoder_ckpt is None and checkpoint and os.path.isfile(checkpoint):
        encoder_ckpt = checkpoint

    if encoder_ckpt is None:
        if echo_root is None:
            raise ValueError("Set wrapper_kwargs.echo_prime_root or wrapper_kwargs.encoder_ckpt (or model_kwargs.checkpoint).")
        encoder_ckpt = os.path.join(echo_root, "model_data", "weights", "echo_prime_encoder.pt")

    if view_ckpt is None:
        if echo_root is None:
            raise ValueError("Set wrapper_kwargs.view_ckpt or wrapper_kwargs.echo_prime_root.")
        view_ckpt = os.path.join(echo_root, "model_data", "weights", "view_classifier.pt")

    logger.info(f"Loading EchoPrime encoder weights: {encoder_ckpt}")
    logger.info(f"Loading EchoPrime view classifier weights: {view_ckpt}")

    # Build models exactly like upstream EchoPrime
    echo_sd = torch.load(encoder_ckpt, map_location="cpu")
    echo_encoder = torchvision.models.video.mvit_v2_s()
    echo_encoder.head[-1] = nn.Linear(echo_encoder.head[-1].in_features, 512)
    echo_encoder.load_state_dict(echo_sd)
    echo_encoder.eval()
    for p in echo_encoder.parameters():
        p.requires_grad = False

    vc_sd = torch.load(view_ckpt, map_location="cpu")
    view_classifier = torchvision.models.convnext_base()
    view_classifier.classifier[-1] = nn.Linear(view_classifier.classifier[-1].in_features, 11)
    view_classifier.load_state_dict(vc_sd)
    view_classifier.eval()
    for p in view_classifier.parameters():
        p.requires_grad = False

    return EchoPrimeEncoderOnlyWrapper(
        echo_encoder=echo_encoder,
        view_classifier=view_classifier,
        force_fp32=bool(wrapper_kwargs.get("force_fp32", True)),
        bin_size=int(wrapper_kwargs.get("bin_size", 50)),
    )
