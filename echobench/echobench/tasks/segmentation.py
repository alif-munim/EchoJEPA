"""CAMUS segmentation task for EchoBench.

Evaluates frozen encoder + lightweight decoder on CAMUS echocardiography
segmentation under acoustic noise perturbations.
"""

import hashlib

import numpy as np
import torch
from torch.utils.data import DataLoader

from echobench.adapters.base import IMAGENET_MEAN, IMAGENET_STD
from echobench.metrics.segmentation import compute_dice, compute_hausdorff_95
from echobench.noise import apply_perturbation, create_scan_mask
from echobench.tasks.camus_dataset import (
    LABEL_NAMES,
    NUM_CLASSES,
    CAMUSSegDataset,
    load_split,
)
from echobench.tasks.decoders import LinearSegDecoder, SmallConvSegDecoder


def _patient_seed(patient_id, view):
    """Deterministic seed from patient/view (reproducible across models)."""
    key = f"{patient_id}_{view}"
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16)


def _normalize_video(video_raw):
    """Apply ImageNet normalization to [C, T, H, W] or [B, C, T, H, W] tensor."""
    if video_raw.dim() == 4:
        mean = IMAGENET_MEAN.unsqueeze(1)  # [3, 1, 1, 1] -> broadcast over T, H, W
        std = IMAGENET_STD.unsqueeze(1)
    else:
        mean = IMAGENET_MEAN.unsqueeze(0).unsqueeze(2)  # [1, 3, 1, 1, 1]
        std = IMAGENET_STD.unsqueeze(0).unsqueeze(2)
    return (video_raw - mean.to(video_raw.device)) / std.to(video_raw.device)


def extract_spatial_features(tokens, model_type, temporal_tokens):
    """Reshape encoder tokens and extract spatial grid at temporal index.

    Args:
        tokens: [B, N, D] from encoder forward
        model_type: one of "vjepa", "videomae", "echoprime", "panecho"
        temporal_tokens: [B] tensor of temporal token indices

    Returns:
        [B, H', W', D] spatial feature grid
    """
    B, N, D = tokens.shape

    if model_type == "echoprime":
        T_tokens, H_patch, W_patch = 8, 7, 7
    elif model_type == "panecho":
        T_tokens, H_patch, W_patch = 16, 7, 7
        temporal_tokens = torch.clamp(temporal_tokens * 2, max=T_tokens - 1)
    else:
        # ViT-based (vjepa, videomae, echofm): 8 temporal x 14x14 spatial
        T_tokens = 8
        spatial_per_t = N // T_tokens
        H_patch = W_patch = int(spatial_per_t**0.5)

    tokens = tokens.float().reshape(B, T_tokens, H_patch, W_patch, D)
    spatial = torch.stack([tokens[b, temporal_tokens[b]] for b in range(B)])
    return spatial


def load_decoder(decoder_checkpoint, decoder_type, embed_dim, device):
    """Load a pretrained segmentation decoder.

    Args:
        decoder_checkpoint: path to decoder checkpoint (.pt)
        decoder_type: "linear" or "conv"
        embed_dim: encoder embedding dimension
        device: torch device

    Returns:
        decoder nn.Module (eval mode, on device)
    """
    if decoder_type == "linear":
        decoder = LinearSegDecoder(embed_dim, num_classes=NUM_CLASSES)
    elif decoder_type == "conv":
        decoder = SmallConvSegDecoder(embed_dim, num_classes=NUM_CLASSES)
    else:
        raise ValueError(f"Unknown decoder_type: {decoder_type}. Choose 'linear' or 'conv'.")

    ckpt = torch.load(decoder_checkpoint, map_location="cpu", weights_only=False)

    # Handle multi-head checkpoint format (select best head)
    if "classifiers" in ckpt:
        best_vals = ckpt.get("best_val_acc_per_head", [])
        if best_vals:
            best_idx = best_vals.index(max(best_vals))  # Higher Dice = better
        else:
            best_idx = 0
        state_dict = ckpt["classifiers"][best_idx]
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    decoder.load_state_dict(state_dict)
    decoder.eval().to(device)
    return decoder


class CAMUSSegTask:
    """CAMUS segmentation evaluation task.

    Args:
        camus_root: path to CAMUS_public directory
        decoder_checkpoint: path to trained decoder checkpoint
        decoder_type: "linear" or "conv"
        model_type: encoder type for spatial feature extraction
            ("vjepa", "videomae", "echoprime", "panecho", "echofm")
        resolution: input spatial resolution
        num_frames: frames per clip
        views: tuple of views to evaluate
        split: dataset split ("testing", "training")
    """

    def __init__(
        self,
        camus_root,
        decoder_checkpoint,
        decoder_type="linear",
        model_type="vjepa",
        resolution=224,
        num_frames=16,
        views=("4CH",),
        split="testing",
    ):
        self.camus_root = camus_root
        self.decoder_checkpoint = decoder_checkpoint
        self.decoder_type = decoder_type
        self.model_type = model_type
        self.resolution = resolution
        self.num_frames = num_frames
        self.views = views
        self.split = split
        self._decoder = None

        patient_ids = load_split(camus_root, split)
        self.dataset = CAMUSSegDataset(
            patient_ids, camus_root,
            views=views, resolution=resolution,
            num_frames=num_frames, normalize=False,
        )

    def _ensure_decoder(self, embed_dim, device):
        if self._decoder is None:
            self._decoder = load_decoder(
                self.decoder_checkpoint, self.decoder_type, embed_dim, device,
            )

    @torch.no_grad()
    def evaluate_condition(self, adapter, noise_type=None, severity=None, device="cuda", max_cases=None):
        """Run segmentation evaluation for one condition.

        Returns:
            dict with "metrics" (mean_dice, per-structure dice, hausdorff) and "n_samples"
        """
        self._ensure_decoder(adapter.embed_dim, device)

        n_samples = min(len(self.dataset), max_cases) if max_cases else len(self.dataset)
        all_dice = {phase: {c: [] for c in range(1, NUM_CLASSES)} for phase in ["ed", "es"]}
        all_hd = {phase: {c: [] for c in range(1, NUM_CLASSES)} for phase in ["ed", "es"]}

        for i in range(n_samples):
            sample = self.dataset[i]
            video_raw = sample["video_raw"]  # [C, T, H, W] in [0, 1]

            if noise_type is not None:
                seed = _patient_seed(sample["patient_id"], sample["view"])
                mask = create_scan_mask(video_raw[:, 0, :, :])
                video_raw = apply_perturbation(video_raw, noise_type, severity, scan_mask=mask, seed=seed)

            video = _normalize_video(video_raw)
            video = video.unsqueeze(0).to(device)  # [1, C, T, H, W]

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                tokens = adapter(video)  # [1, N, D]

            for phase in ["ed", "es"]:
                gt_mask = sample[f"{phase}_mask"].to(device)  # [H, W]
                t_idx = torch.tensor([sample[f"{phase}_temporal_token"]])

                spatial = extract_spatial_features(tokens, self.model_type, t_idx)  # [1, H', W', D]
                logits = self._decoder(spatial.float())  # [1, C, H, W]
                pred = logits.argmax(1).squeeze(0)  # [H, W]

                dice = compute_dice(pred, gt_mask, num_classes=NUM_CLASSES)
                for c in dice:
                    all_dice[phase][c].append(dice[c])
                    hd = compute_hausdorff_95(pred == c, gt_mask == c)
                    if hd != float("inf"):
                        all_hd[phase][c].append(hd)

            if (i + 1) % 50 == 0:
                print(f"    {i + 1}/{n_samples}")

        # Aggregate metrics
        metrics = {}
        for phase in ["ed", "es"]:
            for c in range(1, NUM_CLASSES):
                name = LABEL_NAMES[c]
                dices = all_dice[phase][c]
                hds = all_hd[phase][c]
                metrics[f"{phase}_{name}_dice"] = float(np.mean(dices)) if dices else 0.0
                metrics[f"{phase}_{name}_hd95"] = float(np.mean(hds)) if hds else float("inf")

        # Mean across phases per structure
        for c in range(1, NUM_CLASSES):
            name = LABEL_NAMES[c]
            metrics[f"mean_{name}_dice"] = (metrics[f"ed_{name}_dice"] + metrics[f"es_{name}_dice"]) / 2

        # Overall mean dice
        metrics["mean_dice"] = float(np.mean([
            metrics[f"mean_{LABEL_NAMES[c]}_dice"] for c in range(1, NUM_CLASSES)
        ]))

        return {
            "metrics": metrics,
            "n_samples": n_samples,
            "n_skipped": 0,
        }
