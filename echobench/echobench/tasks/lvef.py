"""LVEF regression task for EchoBench.

Supports EchoNet-Dynamic, EchoNet-Pediatric, or any dataset with
space-delimited CSVs in the format: ``video_path label``.
"""

import hashlib

import decord
import numpy as np
import torch
import torchvision.transforms.functional as TF

from echobench.adapters.base import IMAGENET_MEAN, IMAGENET_STD
from echobench.metrics.regression import compute_regression_metrics
from echobench.noise import apply_perturbation, create_scan_mask

decord.bridge.set_bridge("torch")


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def read_csv(csv_path):
    """Read space-delimited CSV: path label."""
    paths, labels = [], []
    with open(csv_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                paths.append(parts[0])
                labels.append(float(parts[1]))
    return paths, labels


def path_seed(video_path):
    """Deterministic seed from video path (reproducible across runs/models)."""
    return int(hashlib.md5(video_path.encode()).hexdigest()[:8], 16)


def load_clip(video_path, frames=16, frame_step=2, resolution=224):
    """Load a single clip as [C, T, H, W] float32 in [0, 1]."""
    vr = decord.VideoReader(video_path, num_threads=1)
    total = len(vr)
    needed = frames * frame_step
    start = max(0, (total - needed) // 2)
    indices = list(range(start, min(start + needed, total), frame_step))
    while len(indices) < frames:
        indices.append(indices[-1])
    indices = indices[:frames]
    clip = vr.get_batch(indices)  # [T, H, W, C]
    clip = clip.permute(3, 0, 1, 2).float() / 255.0  # [C, T, H, W]
    clip = TF.resize(clip, [resolution, resolution], antialias=True)
    return clip


def normalize_clip(clip):
    """Apply ImageNet normalization to [C, T, H, W] clip."""
    mean = IMAGENET_MEAN.squeeze(-1)  # [3, 1, 1]
    std = IMAGENET_STD.squeeze(-1)
    return (clip - mean) / std


# ---------------------------------------------------------------------------
# Probe loading
# ---------------------------------------------------------------------------


def load_probe(probe_checkpoint, task_type, embed_dim, device):
    """
    Load the best probe head from a multi-head checkpoint.

    Returns:
        (probe_model, metadata_dict)
    """
    from src.models.attentive_pooler import AttentiveClassifier, AttentiveRegressor

    ckpt = torch.load(probe_checkpoint, map_location="cpu", weights_only=False)

    # Find best head
    best_vals = ckpt["best_val_acc_per_head"]
    if task_type == "regression":
        best_idx = best_vals.index(min(best_vals))
    else:
        best_idx = best_vals.index(max(best_vals))

    state_dict = ckpt["classifiers"][best_idx]
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    linear_key = "regressor.weight" if task_type == "regression" else "linear.weight"
    num_outputs = state_dict[linear_key].shape[0]

    sa_keys = [k for k in state_dict if "self_attention_blocks" in k and "norm1.weight" in k]
    depth = len(sa_keys) + 1
    num_heads = embed_dim // 64

    if task_type == "regression":
        probe = AttentiveRegressor(
            embed_dim=embed_dim, num_heads=num_heads, depth=depth, num_targets=num_outputs,
        )
    else:
        probe = AttentiveClassifier(
            embed_dim=embed_dim, num_heads=num_heads, depth=depth, num_classes=num_outputs,
        )

    probe.load_state_dict(state_dict)
    probe.eval().to(device)

    metadata = {
        "target_mean": ckpt.get("target_mean"),
        "target_std": ckpt.get("target_std"),
        "task_type": ckpt.get("task_type", task_type),
        "best_head_idx": best_idx,
        "best_val": best_vals[best_idx],
    }
    return probe, metadata


# ---------------------------------------------------------------------------
# Task class
# ---------------------------------------------------------------------------


class LVEFTask:
    """LVEF regression evaluation task.

    Args:
        data_csv: path to test CSV (space-delimited: video_path label)
        probe_checkpoint: path to trained probe checkpoint (.pt)
        task_type: "regression" or "classification"
        resolution: input spatial resolution
        frames: number of frames per clip
        frame_step: frame sampling stride
    """

    def __init__(
        self,
        data_csv,
        probe_checkpoint,
        task_type="regression",
        resolution=224,
        frames=16,
        frame_step=2,
    ):
        self.paths, self.labels = read_csv(data_csv)
        self.task_type = task_type
        self.resolution = resolution
        self.frames = frames
        self.frame_step = frame_step
        self.probe_checkpoint = probe_checkpoint
        self._probe = None
        self._metadata = None

    def _ensure_probe(self, embed_dim, device):
        if self._probe is None:
            self._probe, self._metadata = load_probe(
                self.probe_checkpoint, self.task_type, embed_dim, device,
            )

    @torch.no_grad()
    def evaluate_condition(self, adapter, noise_type=None, severity=None, device="cuda", max_cases=None):
        """
        Run inference for one condition (clean or a specific perturbation).

        Args:
            adapter: EncoderAdapter instance (already on device, frozen)
            noise_type: None for clean, or one of NOISE_TYPES
            severity: None for clean, or one of SEVERITY_LEVELS
            device: torch device string
            max_cases: limit number of test videos (None = all)

        Returns:
            dict with "metrics" and "n_samples" keys
        """
        self._ensure_probe(adapter.embed_dim, device)

        paths = self.paths[:max_cases] if max_cases else self.paths
        labels = self.labels[:max_cases] if max_cases else self.labels

        all_preds = []
        all_labels = []
        n_skipped = 0

        for i, (video_path, label) in enumerate(zip(paths, labels)):
            try:
                clip = load_clip(video_path, self.frames, self.frame_step, self.resolution)
            except Exception:
                n_skipped += 1
                continue

            if noise_type is not None:
                seed = path_seed(video_path)
                mask = create_scan_mask(clip[:, 0, :, :])
                clip = apply_perturbation(clip, noise_type, severity, scan_mask=mask, seed=seed)

            clip = normalize_clip(clip)
            clip = clip.unsqueeze(0).to(device)  # [1, C, T, H, W]
            features = adapter(clip)  # [1, N, D]
            pred = self._probe(features)  # [1, num_targets]
            all_preds.append(pred.cpu().squeeze())
            all_labels.append(label)

            if (i + 1) % 200 == 0:
                print(f"    {i + 1}/{len(paths)}")

        if not all_preds:
            return {"metrics": {}, "n_samples": 0, "n_skipped": n_skipped}

        preds = torch.stack(all_preds).numpy()
        labels_arr = np.array(all_labels)

        metrics = compute_regression_metrics(
            preds, labels_arr,
            target_mean=self._metadata.get("target_mean"),
            target_std=self._metadata.get("target_std"),
        )

        return {
            "metrics": metrics,
            "n_samples": len(all_preds),
            "n_skipped": n_skipped,
        }
