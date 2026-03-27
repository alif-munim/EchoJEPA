"""
Frame shuffling temporal ablation for ICML rebuttal.

Measures how much encoder representations degrade when frame order is destroyed.
EchoJEPA should show larger degradation (encodes temporal dynamics) vs EchoMAE
(learns static appearance, invariant to frame order).

Reports per-model: mean +/- std cosine similarity between original and shuffled
representations, across 3 shuffle seeds.

Usage:
    python scripts/rebuttal/frame_shuffling.py \
        --csv data/csv/uhn_views_22k_val.csv \
        --n_videos 200 \
        --device cuda:0

Models are hardcoded: EchoJEPA-G, EchoJEPA-L, EchoMAE-L.
"""

import argparse
import random

import decord
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

import src.models.vision_transformer as vit

decord.bridge.set_bridge("torch")

# Model configs: (checkpoint, model_name, checkpoint_key, extra_kwargs)
MODELS = {
    "EchoJEPA-G": {
        "checkpoint": "checkpoints/anneal/keep/pt-280-an81.pt",
        "model_name": "vit_giant_xformers",
        "checkpoint_key": "target_encoder",
        "kwargs": {"uniform_power": True, "use_rope": True},
    },
    "EchoJEPA-L": {
        "checkpoint": "checkpoints/anneal/keep/vitl-pt-210-an25.pt",
        "model_name": "vit_large",
        "checkpoint_key": "target_encoder",
        "kwargs": {"uniform_power": True, "use_rope": True},
    },
    "EchoMAE-L": {
        "checkpoint": "checkpoints/videomae-ep163.pth",
        "model_name": None,  # VideoMAE uses timm model
        "checkpoint_key": None,
        "kwargs": {},
    },
}


def load_clip_frames(video_path, frames=16, frame_step=2, resolution=224):
    """Load individual frames (not yet assembled into clip tensor)."""
    vr = decord.VideoReader(video_path, num_threads=1)
    total = len(vr)
    needed = frames * frame_step
    start = max(0, (total - needed) // 2)
    indices = list(range(start, min(start + needed, total), frame_step))
    while len(indices) < frames:
        indices.append(indices[-1])
    indices = indices[:frames]
    batch = vr.get_batch(indices)  # [T, H, W, C]
    batch = batch.permute(0, 3, 1, 2).float() / 255.0  # [T, C, H, W]
    batch = torch.stack([TF.resize(f, [resolution, resolution], antialias=True) for f in batch])
    return batch  # [T, C, H, W]


def frames_to_clip(frames_tensor):
    """Convert [T, C, H, W] -> [1, C, T, H, W] for encoder input."""
    return frames_tensor.permute(1, 0, 2, 3).unsqueeze(0)


def load_vjepa_encoder(cfg, device, resolution=224, frames=16):
    """Load a V-JEPA encoder."""
    ckpt = torch.load(cfg["checkpoint"], map_location="cpu", weights_only=False)
    model = vit.__dict__[cfg["model_name"]](
        img_size=resolution,
        num_frames=frames,
        patch_size=16,
        tubelet_size=2,
        **cfg["kwargs"],
    )
    state = ckpt[cfg["checkpoint_key"]]
    state = {k.replace("module.", "").replace("backbone.", ""): v for k, v in state.items()}
    for k, v in model.state_dict().items():
        if k not in state:
            print(f"  Missing key: {k}")
        elif state[k].shape != v.shape:
            print(f"  Shape mismatch: {k}")
            state[k] = v
    model.load_state_dict(state, strict=False)
    model.eval().to(device)
    return model


def load_videomae_encoder(cfg, device, resolution=224, frames=16):
    """Load VideoMAE encoder using vendored modeling_finetune."""
    from evals.video_classification_frozen.modelcustom.videomae_encoder import (
        _convert_pretrain_to_finetune_state_dict,
        _import_modeling_finetune,
    )

    mf = _import_modeling_finetune()
    model = mf.vit_large_patch16_224(
        img_size=resolution, all_frames=frames, tubelet_size=2, num_classes=1000,
    )
    ckpt = torch.load(cfg["checkpoint"], map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)
    state = _convert_pretrain_to_finetune_state_dict(state, model.state_dict())
    msg = model.load_state_dict(state, strict=False)
    truly_missing = [k for k in msg.missing_keys if not k.startswith("head.")]
    print(f"  VideoMAE load: missing={len(truly_missing)} non-head, unexpected={len(msg.unexpected_keys)}")
    model.eval().to(device)
    return model


@torch.no_grad()
def extract_features(model, clip, is_videomae=False):
    """Extract mean-pooled features from a clip [1, C, T, H, W]."""
    if is_videomae:
        # VideoMAE expects [B, C, T, H, W]
        out = model.forward_features(clip)
        if out.dim() == 3:
            return out.mean(dim=1)  # [1, D]
        return out
    else:
        out = model(clip)  # [1, N, D]
        return out.mean(dim=1)  # [1, D]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--n_videos", type=int, default=200)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--frame_step", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_shuffle_seeds", type=int, default=3)
    args = parser.parse_args()

    random.seed(args.seed)
    device = torch.device(args.device)

    # Read CSV and sample videos
    paths = []
    with open(args.csv) as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                paths.append(parts[0])
    if len(paths) > args.n_videos:
        paths = random.sample(paths, args.n_videos)

    # Load all video frames
    print(f"Loading {len(paths)} videos...")
    all_frames = []
    valid_paths = []
    for i, p in enumerate(paths):
        try:
            frames = load_clip_frames(p, args.frames, args.frame_step, args.resolution)
            all_frames.append(frames)
            valid_paths.append(p)
        except Exception as e:
            print(f"  Skip {p}: {e}")
        if (i + 1) % 50 == 0:
            print(f"  Loaded {i + 1}/{len(paths)}")
    print(f"Loaded {len(valid_paths)} videos successfully")

    # Process each model
    results = {}
    for model_name, cfg in MODELS.items():
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        is_videomae = model_name == "EchoMAE-L"
        if is_videomae:
            model = load_videomae_encoder(cfg, device, args.resolution, args.frames)
        else:
            model = load_vjepa_encoder(cfg, device, args.resolution, args.frames)

        # Collect cosine similarities across shuffle seeds
        all_cos_sims = []
        for shuffle_seed in range(args.n_shuffle_seeds):
            rng = np.random.RandomState(shuffle_seed + 100)
            cos_sims = []

            for frames in all_frames:
                clip_orig = frames_to_clip(frames).to(device)
                feat_orig = extract_features(model, clip_orig, is_videomae)

                # Shuffle frame order
                perm = rng.permutation(args.frames)
                frames_shuf = frames[perm]
                clip_shuf = frames_to_clip(frames_shuf).to(device)
                feat_shuf = extract_features(model, clip_shuf, is_videomae)

                cos = F.cosine_similarity(feat_orig, feat_shuf, dim=-1).item()
                cos_sims.append(cos)

            all_cos_sims.append(cos_sims)

        # Aggregate: mean across videos, then mean +/- std across seeds
        per_seed_means = [np.mean(s) for s in all_cos_sims]
        mean_cos = np.mean(per_seed_means)
        std_cos = np.std(per_seed_means)

        results[model_name] = {"mean": mean_cos, "std": std_cos, "per_seed": per_seed_means}
        print(f"  cos(orig, shuffled) = {mean_cos:.4f} +/- {std_cos:.4f}")
        print(f"  Per-seed means: {[f'{m:.4f}' for m in per_seed_means]}")
        print(f"  Degradation: {(1 - mean_cos) * 100:.1f}%")

        del model
        torch.cuda.empty_cache()

    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY: Frame Shuffling Temporal Ablation")
    print(f"{'='*60}")
    print(f"{'Model':<20} {'cos(orig,shuf)':<20} {'Degradation':<15}")
    print("-" * 55)
    for name, r in results.items():
        print(f"{name:<20} {r['mean']:.4f} +/- {r['std']:.4f}    {(1-r['mean'])*100:.1f}%")


if __name__ == "__main__":
    main()
