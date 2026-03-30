"""Frozen encoder segmentation evaluation on CAMUS.

Trains a lightweight segmentation decoder on frozen encoder features from
CAMUS echocardiography sequences. Evaluates Dice score per structure (LV,
myocardium, left atrium) at end-diastole and end-systole.

Usage:
    # EchoJEPA-L (V-JEPA, MIMIC)
    python -m evals.segmentation_frozen.eval \
        --model_type vjepa \
        --checkpoint checkpoints/vitl-pt-210-an25.pt \
        --output_dir results/segmentation/echojepa_l \
        --device cuda:0

    # EchoMAE-L (VideoMAE, MIMIC)
    python -m evals.segmentation_frozen.eval \
        --model_type videomae \
        --checkpoint checkpoints/echomae_l_mimic_ep99.pth \
        --output_dir results/segmentation/echomae_l \
        --device cuda:0

    # EchoPrime (MViT-v2-S)
    python -m evals.segmentation_frozen.eval \
        --model_type echoprime \
        --checkpoint checkpoints/echo_prime_encoder.pt \
        --output_dir results/segmentation/echoprime \
        --device cuda:0

    # PanEcho (ConvNeXt-Tiny + Transformer)
    python -m evals.segmentation_frozen.eval \
        --model_type panecho \
        --checkpoint unused \
        --output_dir results/segmentation/panecho \
        --device cuda:0
"""

import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from evals.segmentation_frozen.camus_dataset import (
    CAMUSSegDataset,
    LABEL_NAMES,
    NUM_CLASSES,
    load_split,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Segmentation decoders
# ---------------------------------------------------------------------------


class LinearSegDecoder(nn.Module):
    """Minimal segmentation decoder: 1x1 conv on spatial token grid + upsample.

    This is the segmentation analogue of a linear probe — a single linear
    layer per spatial token, followed by bilinear upsampling to the target
    resolution. Any performance difference is attributable to the frozen
    encoder's representation quality, not the decoder's capacity.
    """

    def __init__(self, embed_dim, num_classes=4, target_size=224):
        super().__init__()
        self.head = nn.Conv2d(embed_dim, num_classes, kernel_size=1)
        self.target_size = target_size

    def forward(self, x):
        # x: [B, H', W', D] (spatial token grid)
        x = x.permute(0, 3, 1, 2)  # [B, D, H', W']
        x = self.head(x)  # [B, C, H', W']
        x = F.interpolate(x, size=self.target_size, mode="bilinear", align_corners=False)
        return x  # [B, C, H, W]


class SmallConvSegDecoder(nn.Module):
    """Small convolutional segmentation decoder for better spatial precision.

    Four transposed convolution stages upsample 14x14 → 224x224 (16x factor,
    matching patch_size=16). Still lightweight (~2M params) — decoder capacity
    is not a confound.
    """

    def __init__(self, embed_dim, num_classes=4):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 256, 4, stride=2, padding=1),  # 14→28
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 28→56
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 56→112
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.ConvTranspose2d(64, num_classes, 4, stride=2, padding=1),  # 112→224
        )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # [B, D, H', W']
        return self.decoder(x)  # [B, C, H, W]


# ---------------------------------------------------------------------------
# Encoder loading
# ---------------------------------------------------------------------------


def load_vjepa_encoder(checkpoint_path, model_name=None, device="cpu", resolution=224):
    """Load a V-JEPA / EchoJEPA encoder from checkpoint.

    Auto-detects model size from checkpoint filename if model_name is None:
      vitg/giant → vit_giant, vitl/large → vit_large
    """
    import src.models.vision_transformer as vit

    if model_name is None:
        basename = os.path.basename(checkpoint_path).lower()
        if "vitg" in basename or "giant" in basename:
            model_name = "vit_giant_xformers"
        else:
            model_name = "vit_large"

    # Create model
    model_fn = vit.__dict__[model_name]
    encoder = model_fn(
        img_size=resolution,
        num_frames=16,
        patch_size=16,
        tubelet_size=2,
        uniform_power=True,
        use_rope=True,
    )

    # Load checkpoint
    state = torch.load(checkpoint_path, map_location="cpu")
    encoder_key = "target_encoder"
    if encoder_key not in state:
        # Try other common keys
        for key in ["encoder", "model", "state_dict"]:
            if key in state:
                encoder_key = key
                break
        else:
            # Assume the state dict is the checkpoint itself
            encoder_key = None

    encoder_state = state[encoder_key] if encoder_key else state
    encoder_state = {k.replace("module.", "").replace("backbone.", ""): v for k, v in encoder_state.items()}

    # Filter shape mismatches
    model_state = encoder.state_dict()
    for k in list(encoder_state.keys()):
        if k in model_state and encoder_state[k].shape != model_state[k].shape:
            logger.warning(f"Shape mismatch for {k}: checkpoint {encoder_state[k].shape} vs model {model_state[k].shape}")
            del encoder_state[k]

    msg = encoder.load_state_dict(encoder_state, strict=False)
    logger.info(f"Loaded V-JEPA encoder: {msg}")

    encoder = encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False

    return encoder, encoder.embed_dim


class VideoMAESpatialWrapper(nn.Module):
    """Wraps a VideoMAE model to return spatial tokens instead of pooled features."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.embed_dim = model.embed_dim

    @torch.no_grad()
    def forward(self, x):
        """Return pre-pooling spatial tokens [B, N, D]."""
        m = self.model
        x = m.patch_embed(x)
        B = x.shape[0]
        if m.pos_embed is not None:
            x = x + m.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        x = m.pos_drop(x)
        for blk in m.blocks:
            x = blk(x)
        x = m.norm(x)
        return x  # [B, N, D] — spatial tokens, no pooling


def load_videomae_encoder(checkpoint_path, device="cpu"):
    """Load a VideoMAE / EchoMAE encoder from checkpoint."""
    # Use the existing VideoMAE adapter to load the model
    from evals.video_classification_frozen.modelcustom.videomae_encoder import init_module

    wrapper = init_module(
        resolution=224,
        frames_per_clip=16,
        checkpoint=checkpoint_path,
        model_kwargs={
            "encoder": {
                "model_name": "vit_large_patch16_224",
                "tubelet_size": 2,
            }
        },
        wrapper_kwargs={},
    )
    # Extract the raw model and wrap it to return spatial tokens
    raw_model = wrapper.model
    spatial_wrapper = VideoMAESpatialWrapper(raw_model)
    spatial_wrapper = spatial_wrapper.to(device).eval()
    for p in spatial_wrapper.parameters():
        p.requires_grad = False

    return spatial_wrapper, spatial_wrapper.embed_dim


class EchoPrimeSpatialWrapper(nn.Module):
    """Wraps EchoPrime MViT-v2-S to return pre-pooling spatial tokens.

    MViT-v2-S produces [B, 393, 768] after norm = 1 CLS + 8×7×7 spatial tokens.
    We strip the CLS and return the spatial token grid.

    Handles normalization: input is ImageNet-normalized (from CAMUS dataset),
    converted to EchoPrime's expected normalization (0-255 pixel space).
    """

    def __init__(self, mvit_model):
        super().__init__()
        self.model = mvit_model
        self.embed_dim = 768
        self._features = {}

        # EchoPrime normalization constants (0-255 pixel space)
        mean = torch.tensor([29.110628, 28.076836, 29.096405]).reshape(1, 3, 1, 1, 1)
        std = torch.tensor([47.989223, 46.456997, 47.20083]).reshape(1, 3, 1, 1, 1)
        self.register_buffer("ep_mean", mean, persistent=False)
        self.register_buffer("ep_std", std, persistent=False)

        # ImageNet normalization constants (to undo)
        inet_mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1, 1)
        inet_std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1, 1)
        self.register_buffer("inet_mean", inet_mean, persistent=False)
        self.register_buffer("inet_std", inet_std, persistent=False)

        # Hook after norm to capture spatial tokens before head pools them
        self.model.norm.register_forward_hook(self._norm_hook)

    def _norm_hook(self, module, input, output):
        self._features["post_norm"] = output

    def _renormalize(self, x):
        """Convert ImageNet-normalized input to EchoPrime normalization."""
        x = x.float()
        x = x * self.inet_std + self.inet_mean  # undo ImageNet → [0, 1]
        x = x * 255.0  # scale to [0, 255]
        x = (x - self.ep_mean) / self.ep_std  # apply EchoPrime norm
        return x

    @torch.no_grad()
    def forward(self, x):
        """x: [B, 3, T, H, W] (ImageNet-normalized). Returns: [B, N, D] spatial tokens."""
        x = self._renormalize(x)
        _ = self.model(x)
        tokens = self._features["post_norm"]  # [B, 393, 768]
        # Strip CLS token (first token)
        return tokens[:, 1:, :]  # [B, 392, 768] = 8×7×7


class PanEchoSpatialWrapper(nn.Module):
    """Wraps PanEcho to return ConvNeXt spatial feature maps per frame.

    PanEcho is a 2+1D model: ConvNeXt encodes each frame to [768, 7, 7],
    then a Transformer does temporal modeling on pooled per-frame vectors.
    We hook into the ConvNeXt backbone to extract the 7×7 spatial features
    BEFORE global pooling, preserving spatial structure for segmentation.

    Note: These are per-frame features without temporal Transformer context.
    """

    def __init__(self, panecho_model):
        super().__init__()
        self.model = panecho_model
        self.embed_dim = 768
        self._features = {}

        # Hook into the ConvNeXt features (backbone) before classifier pooling
        # PanEcho's encoder.encoder is the ImageEncoder, which wraps ConvNeXt
        convnext = self.model.encoder.model
        convnext.features.register_forward_hook(self._backbone_hook)

    def _backbone_hook(self, module, input, output):
        self._features["backbone"] = output

    @torch.no_grad()
    def forward(self, x):
        """x: [B, 3, T, H, W]. Returns: [B, T*49, 768] spatial tokens."""
        B, C, T, H, W = x.shape

        # PanEcho expects [B, C, T, H, W] input
        _ = self.model(x)

        # backbone hook captures [B*T, 768, 7, 7] from ConvNeXt
        feat_maps = self._features["backbone"]  # [B*T, 768, 7, 7]
        _, D, Hp, Wp = feat_maps.shape

        # Reshape to [B, T, 768, 7, 7] -> [B, T, 49, 768] -> [B, T*49, 768]
        feat_maps = feat_maps.reshape(B, T, D, Hp, Wp)
        feat_maps = feat_maps.permute(0, 1, 3, 4, 2)  # [B, T, 7, 7, 768]
        feat_maps = feat_maps.reshape(B, T * Hp * Wp, D)  # [B, T*49, 768]
        return feat_maps


def load_echoprime_encoder(checkpoint_path, device="cpu"):
    """Load EchoPrime MViT-v2-S and wrap for spatial token extraction."""
    import torchvision

    echo_sd = torch.load(checkpoint_path, map_location="cpu")
    echo_encoder = torchvision.models.video.mvit_v2_s()
    echo_encoder.head[-1] = nn.Linear(echo_encoder.head[-1].in_features, 512)
    echo_encoder.load_state_dict(echo_sd)
    echo_encoder.eval()
    for p in echo_encoder.parameters():
        p.requires_grad = False

    wrapper = EchoPrimeSpatialWrapper(echo_encoder)
    wrapper = wrapper.to(device).eval()
    return wrapper, wrapper.embed_dim


def load_panecho_encoder(checkpoint_path, device="cpu"):
    """Load PanEcho and wrap for spatial token extraction."""
    from evals.video_classification_frozen.modelcustom.panecho_encoder import init_module

    panecho_wrapper = init_module(
        resolution=224,
        frames_per_clip=16,
        checkpoint=checkpoint_path,
        model_kwargs={},
        wrapper_kwargs={},
        device=device,
    )
    # Extract the raw panecho model from the classification wrapper
    panecho_model = panecho_wrapper.panecho_model

    wrapper = PanEchoSpatialWrapper(panecho_model)
    wrapper = wrapper.to(device).eval()
    for p in wrapper.parameters():
        p.requires_grad = False
    return wrapper, wrapper.embed_dim


def load_encoder(model_type, checkpoint_path, device="cpu", model_name=None, resolution=224):
    """Load frozen encoder. Returns (encoder, embed_dim)."""
    if model_type in ("vjepa", "byol"):
        return load_vjepa_encoder(checkpoint_path, model_name=model_name, device=device, resolution=resolution)
    elif model_type == "videomae":
        return load_videomae_encoder(checkpoint_path, device=device)
    elif model_type == "echoprime":
        return load_echoprime_encoder(checkpoint_path, device=device)
    elif model_type == "panecho":
        return load_panecho_encoder(checkpoint_path, device=device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def extract_spatial_features(encoder, video, model_type, temporal_tokens, device):
    """Extract per-frame spatial features from frozen encoder.

    Args:
        encoder: frozen encoder module
        video: [B, 3, T, H, W] video tensor
        model_type: "vjepa", "videomae", "echoprime", or "panecho"
        temporal_tokens: [B] tensor of temporal token indices to extract
        device: torch device

    Returns:
        spatial_features: [B, H', W', D] spatial feature grid at the
            requested temporal position
    """
    video = video.to(device)

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        tokens = encoder(video)  # [B, N, D]

    B, N, D = tokens.shape

    if model_type in ("echoprime",):
        # MViT-v2: 392 tokens = 8 temporal × 7×7 spatial
        T_tokens = 8
        H_patch = W_patch = 7
    elif model_type in ("panecho",):
        # PanEcho: T*49 tokens = T temporal × 7×7 spatial (per-frame ConvNeXt features)
        T_tokens = 16  # One token-set per input frame (no tubelet merging)
        H_patch = W_patch = 7
        # Map tubelet-based temporal_tokens to frame indices (frame = token * 2)
        temporal_tokens = torch.clamp(temporal_tokens * 2, max=T_tokens - 1)
    else:
        # ViT-based (vjepa, videomae): 1568 tokens = 8 temporal × 14×14 spatial
        T_tokens = 16 // 2  # 8
        spatial_per_t = N // T_tokens
        H_patch = W_patch = int(spatial_per_t**0.5)

    # Reshape to [B, T', H', W', D]
    tokens = tokens.float().reshape(B, T_tokens, H_patch, W_patch, D)

    # Extract spatial features at the requested temporal position
    spatial = torch.stack([tokens[b, temporal_tokens[b]] for b in range(B)])  # [B, H', W', D]

    return spatial


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def dice_score(pred, target, num_classes=NUM_CLASSES):
    """Compute per-class Dice score (excluding background).

    Returns dict mapping class index → Dice value.
    """
    scores = {}
    for c in range(1, num_classes):
        pred_c = (pred == c).float()
        target_c = (target == c).float()
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        if union == 0:
            scores[c] = 1.0 if target_c.sum() == 0 else 0.0
        else:
            scores[c] = (2 * intersection / union).item()
    return scores


def hausdorff_distance(pred, target, spacing=1.0):
    """Approximate 95th percentile Hausdorff distance between binary masks."""
    pred_pts = torch.nonzero(pred).float()
    target_pts = torch.nonzero(target).float()

    if len(pred_pts) == 0 or len(target_pts) == 0:
        return float("inf") if len(pred_pts) != len(target_pts) else 0.0

    # Compute directed distances (subsample for speed if large)
    max_pts = 2000
    if len(pred_pts) > max_pts:
        idx = torch.randperm(len(pred_pts))[:max_pts]
        pred_pts = pred_pts[idx]
    if len(target_pts) > max_pts:
        idx = torch.randperm(len(target_pts))[:max_pts]
        target_pts = target_pts[idx]

    d_pred_to_target = torch.cdist(pred_pts, target_pts).min(dim=1).values
    d_target_to_pred = torch.cdist(target_pts, pred_pts).min(dim=1).values

    hd95_fwd = torch.quantile(d_pred_to_target, 0.95).item() * spacing
    hd95_bwd = torch.quantile(d_target_to_pred, 0.95).item() * spacing
    return max(hd95_fwd, hd95_bwd)


# ---------------------------------------------------------------------------
# Multihead training and evaluation (all HP configs in parallel)
# ---------------------------------------------------------------------------


def train_one_epoch_multihead(encoder, heads, dataloader, model_type, device):
    """Train all decoder heads for one epoch, sharing encoder forward pass."""
    for h in heads:
        h["decoder"].train()
    losses = [0.0] * len(heads)
    n_batches = 0

    for batch in dataloader:
        video = batch["video"]
        ed_mask = batch["ed_mask"].to(device)
        es_mask = batch["es_mask"].to(device)
        ed_t = batch["ed_temporal_token"]
        es_t = batch["es_temporal_token"]

        # Single encoder forward pass for ED and ES
        ed_features = extract_spatial_features(encoder, video, model_type, ed_t, device)
        es_features = extract_spatial_features(encoder, video, model_type, es_t, device)

        # Detach so decoder graphs are independent
        ed_feat = ed_features.detach()
        es_feat = es_features.detach()

        for i, h in enumerate(heads):
            loss = (F.cross_entropy(h["decoder"](ed_feat), ed_mask)
                    + F.cross_entropy(h["decoder"](es_feat), es_mask)) / 2
            h["optimizer"].zero_grad()
            loss.backward()
            h["optimizer"].step()
            losses[i] += loss.item()

        n_batches += 1

    return [l / max(n_batches, 1) for l in losses]


@torch.no_grad()
def evaluate_multihead(encoder, heads, dataloader, model_type, device):
    """Evaluate all decoder heads, sharing encoder forward pass."""
    for h in heads:
        h["decoder"].eval()

    # Per-head results accumulators
    all_results = []
    for _ in heads:
        all_results.append({
            phase: {c: {"dice": [], "hd95": []} for c in range(1, NUM_CLASSES)}
            for phase in ["ed", "es"]
        })

    for batch in dataloader:
        video = batch["video"]
        B = video.shape[0]

        for phase in ["ed", "es"]:
            mask = batch[f"{phase}_mask"].to(device)
            t_idx = batch[f"{phase}_temporal_token"]
            features = extract_spatial_features(encoder, video, model_type, t_idx, device)

            for hi, h in enumerate(heads):
                logits = h["decoder"](features)
                pred = logits.argmax(1)
                for b in range(B):
                    d = dice_score(pred[b], mask[b])
                    for c in d:
                        all_results[hi][phase][c]["dice"].append(d[c])
                        hd = hausdorff_distance(pred[b] == c, mask[b] == c)
                        all_results[hi][phase][c]["hd95"].append(hd)

    # Aggregate per head
    summaries = []
    for hi in range(len(heads)):
        summary = {}
        for phase in ["ed", "es"]:
            for c in range(1, NUM_CLASSES):
                name = LABEL_NAMES[c]
                dices = all_results[hi][phase][c]["dice"]
                hds = [x for x in all_results[hi][phase][c]["hd95"] if x != float("inf")]
                summary[f"{phase}_{name}_dice"] = float(np.mean(dices)) if dices else 0.0
                summary[f"{phase}_{name}_hd95"] = float(np.mean(hds)) if hds else float("inf")
        for c in range(1, NUM_CLASSES):
            name = LABEL_NAMES[c]
            summary[f"mean_{name}_dice"] = (summary[f"ed_{name}_dice"] + summary[f"es_{name}_dice"]) / 2
        summary["mean_dice"] = float(np.mean([summary[f"mean_{LABEL_NAMES[c]}_dice"] for c in range(1, NUM_CLASSES)]))
        summaries.append(summary)

    return summaries


# Default hyperparameter grid (matched to probe eval multihead_kwargs style)
DEFAULT_GRID = [
    {"lr": 5e-2, "weight_decay": 1e-4},
    {"lr": 5e-2, "weight_decay": 1e-2},
    {"lr": 2e-2, "weight_decay": 1e-4},
    {"lr": 1e-2, "weight_decay": 1e-4},
    {"lr": 5e-3, "weight_decay": 1e-4},
    {"lr": 1e-3, "weight_decay": 1e-4},
    {"lr": 1e-3, "weight_decay": 1e-2},
]


def main():
    parser = argparse.ArgumentParser(description="Frozen encoder segmentation on CAMUS")
    parser.add_argument("--model_type", choices=["vjepa", "byol", "videomae", "echoprime", "panecho"], required=True)
    parser.add_argument("--checkpoint", required=True, help="Path to encoder checkpoint")
    parser.add_argument("--camus_root", default="data/camus/CAMUS_public")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--views", nargs="+", default=["4CH", "2CH"], help="Echo views to use")
    parser.add_argument("--decoder_type", choices=["linear", "conv"], default="linear")
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=None, help="Single LR (skip grid search)")
    parser.add_argument("--weight_decay", type=float, default=None, help="Single WD (skip grid search)")
    parser.add_argument("--resolution", type=int, default=224, help="Input resolution (default 224, use 384 for G)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_name", default=None, help="Override vjepa model name (e.g. vit_giant)")
    parser.add_argument("--fix_orientation", action="store_true",
                        help="Rotate CAMUS NIfTI images to standard A4C orientation (rot90 CCW + flip-H).")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # File logging
    fh = logging.FileHandler(os.path.join(args.output_dir, "training.log"))
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(fh)

    device = torch.device(args.device)
    logger.info(f"Config: {vars(args)}")

    # --- Load splits ---
    train_ids = load_split(args.camus_root, "training")
    val_ids = load_split(args.camus_root, "validation")
    test_ids = load_split(args.camus_root, "testing")
    logger.info(f"Splits: {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test")

    # --- Create datasets ---
    views = tuple(args.views)
    train_ds = CAMUSSegDataset(train_ids, args.camus_root, views=views, resolution=args.resolution, augment=True, fix_orientation=args.fix_orientation)
    val_ds = CAMUSSegDataset(val_ids, args.camus_root, views=views, resolution=args.resolution, augment=False, fix_orientation=args.fix_orientation)
    test_ds = CAMUSSegDataset(test_ids, args.camus_root, views=views, resolution=args.resolution, augment=False, fix_orientation=args.fix_orientation)
    logger.info(f"Samples: {len(train_ds)} train, {len(val_ds)} val, {len(test_ds)} test")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # --- Load frozen encoder ---
    logger.info(f"Loading {args.model_type} encoder from {args.checkpoint}")
    encoder, embed_dim = load_encoder(args.model_type, args.checkpoint, device=device, model_name=args.model_name, resolution=args.resolution)
    logger.info(f"Encoder embed_dim: {embed_dim}")

    # --- Build hyperparameter grid ---
    if args.lr is not None and args.weight_decay is not None:
        grid = [{"lr": args.lr, "weight_decay": args.weight_decay}]
    else:
        grid = DEFAULT_GRID
    logger.info(f"Multihead grid: {len(grid)} configs (all trained in parallel)")

    # --- Create all decoder heads ---
    heads = []
    for hp in grid:
        tag = f"lr{hp['lr']:.0e}_wd{hp['weight_decay']:.0e}"
        run_dir = os.path.join(args.output_dir, tag)
        os.makedirs(run_dir, exist_ok=True)

        if args.decoder_type == "linear":
            decoder = LinearSegDecoder(embed_dim, NUM_CLASSES, target_size=args.resolution)
        else:
            decoder = SmallConvSegDecoder(embed_dim, NUM_CLASSES)
        decoder = decoder.to(device)

        optimizer = torch.optim.AdamW(decoder.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)

        heads.append({
            "tag": tag, "decoder": decoder, "optimizer": optimizer, "scheduler": scheduler,
            "run_dir": run_dir, "best_val_dice": 0.0, "best_epoch": 0, "log": [],
            "lr": hp["lr"], "wd": hp["weight_decay"],
        })

    n_params = sum(p.numel() for p in heads[0]["decoder"].parameters() if p.requires_grad)
    logger.info(f"Decoder ({args.decoder_type}): {n_params:,} params x {len(heads)} heads")

    # --- Training loop (multihead) ---
    for epoch in range(1, args.num_epochs + 1):
        t0 = time.time()
        torch.manual_seed(args.seed + epoch)

        train_losses = train_one_epoch_multihead(encoder, heads, train_loader, args.model_type, device)
        val_summaries = evaluate_multihead(encoder, heads, val_loader, args.model_type, device)

        for h in heads:
            h["scheduler"].step()
        elapsed = time.time() - t0

        # Log and checkpoint each head
        parts = []
        for i, h in enumerate(heads):
            val_dice = val_summaries[i]["mean_dice"]
            h["log"].append({"epoch": epoch, "train_loss": train_losses[i], "val_mean_dice": val_dice, **val_summaries[i]})

            if val_dice > h["best_val_dice"]:
                h["best_val_dice"] = val_dice
                h["best_epoch"] = epoch
                torch.save(h["decoder"].state_dict(), os.path.join(h["run_dir"], "best_decoder.pt"))

            torch.save(h["decoder"].state_dict(), os.path.join(h["run_dir"], "latest_decoder.pt"))
            parts.append(f"{h['tag']}={val_dice:.4f}")

        logger.info(
            f"Epoch {epoch}/{args.num_epochs} | {elapsed:.1f}s | "
            + " | ".join(parts)
        )

    # --- Test evaluation with best decoders ---
    logger.info("\nLoading best decoders for test evaluation...")
    for h in heads:
        h["decoder"].load_state_dict(
            torch.load(os.path.join(h["run_dir"], "best_decoder.pt"), map_location=device)
        )
    test_summaries = evaluate_multihead(encoder, heads, test_loader, args.model_type, device)

    # --- Save per-head results ---
    all_results = []
    for i, h in enumerate(heads):
        tr = test_summaries[i]
        output = {
            "config": {"lr": h["lr"], "weight_decay": h["wd"], "num_epochs": args.num_epochs,
                       "decoder_type": args.decoder_type},
            "best_epoch": h["best_epoch"],
            "best_val_dice": float(h["best_val_dice"]),
            "test_results": tr,
            "training_log": h["log"],
        }
        with open(os.path.join(h["run_dir"], "results.json"), "w") as f:
            json.dump(output, f, indent=2, default=str)

        all_results.append({
            "tag": h["tag"], "lr": h["lr"], "weight_decay": h["wd"],
            "best_val_dice": float(h["best_val_dice"]), "best_epoch": h["best_epoch"],
            "test_mean_dice": float(tr["mean_dice"]), "test_results": tr,
        })

    # --- Summary ---
    best_r = max(all_results, key=lambda x: x["best_val_dice"])
    logger.info("")
    logger.info("=" * 70)
    logger.info(f"GRID SEARCH COMPLETE — {args.model_type}")
    logger.info("=" * 70)
    logger.info(f"{'Config':<22} {'Val Dice':>10} {'Test Dice':>10} {'Best Ep':>8}")
    logger.info("-" * 70)
    for r in sorted(all_results, key=lambda x: x["best_val_dice"], reverse=True):
        marker = " <-- BEST" if r["tag"] == best_r["tag"] else ""
        logger.info(
            f"{r['tag']:<22} {r['best_val_dice']:>10.4f} {r['test_mean_dice']:>10.4f} "
            f"{r['best_epoch']:>8}{marker}"
        )
    logger.info("=" * 70)

    tr = best_r["test_results"]
    logger.info(f"\nBest config: {best_r['tag']}")
    logger.info(f"{'Structure':<12} {'ED Dice':>10} {'ES Dice':>10} {'Mean Dice':>10}")
    logger.info("-" * 50)
    for c in range(1, NUM_CLASSES):
        name = LABEL_NAMES[c]
        logger.info(f"{name:<12} {tr[f'ed_{name}_dice']:>10.4f} {tr[f'es_{name}_dice']:>10.4f} {tr[f'mean_{name}_dice']:>10.4f}")
    logger.info("-" * 50)
    logger.info(f"{'Overall':<12} {'':>10} {'':>10} {tr['mean_dice']:>10.4f}")

    # Save grid summary
    summary = {
        "model_type": args.model_type,
        "checkpoint": args.checkpoint,
        "best_config": best_r["tag"],
        "best_val_dice": float(best_r["best_val_dice"]),
        "grid_results": all_results,
    }
    with open(os.path.join(args.output_dir, "grid_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"\nGrid summary saved to {os.path.join(args.output_dir, 'grid_summary.json')}")


if __name__ == "__main__":
    main()
