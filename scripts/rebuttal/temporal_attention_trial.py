"""
Quick trial: cross-temporal vs within-frame attention analysis.

For each layer, compute the fraction of attention flowing between
tokens at different temporal positions (cross-temporal) vs same temporal
position (within-frame). Tokens are ordered as [T_patches * H_patches * W_patches]
where T_patches = num_frames // tubelet_size.

Approach: set use_sdpa=False so the manual attention path is used, then
hook into attn_drop (nn.Dropout) whose input is the post-softmax attention
weight matrix [B, H, N, N].

Usage:
  python scripts/rebuttal/temporal_attention_trial.py \
    --checkpoint checkpoints/pretrain/mimic/jepa_in21k_e100.pt \
    --checkpoint_key target_encoder \
    --label jepa_e100 \
    --test_csv /opt/dlami/nvme/echonet_dynamic/test_local.csv \
    --n_videos 10 --device cuda:0
"""

import argparse
import csv
import os

import sys
import numpy as np
import torch
import torchvision.transforms.functional as TF
import decord
decord.bridge.set_bridge("torch")

sys.path.insert(0, ".")
import src.models.vision_transformer as vit


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--checkpoint_key", default="target_encoder")
    p.add_argument("--encoder_type", default="vjepa", choices=["vjepa", "videomae"])
    p.add_argument("--model_name", default="vit_large")
    p.add_argument("--label", required=True)
    p.add_argument("--test_csv", required=True)
    p.add_argument("--n_videos", type=int, default=10)
    p.add_argument("--num_frames", type=int, default=16)
    p.add_argument("--frame_step", type=int, default=2)
    p.add_argument("--resolution", type=int, default=224)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--save_dir", default="scripts/rebuttal/temporal_attention")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def read_csv(csv_path):
    paths, labels = [], []
    with open(csv_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                paths.append(parts[0])
                labels.append(float(parts[1]))
    return paths, labels


def load_clip(video_path, num_frames=16, frame_step=2, resolution=224):
    vr = decord.VideoReader(video_path, num_threads=1)
    total = len(vr)
    needed = num_frames * frame_step
    start = max(0, (total - needed) // 2)
    indices = list(range(start, min(start + needed, total), frame_step))
    while len(indices) < num_frames:
        indices.append(indices[-1])
    indices = indices[:num_frames]
    clip = vr.get_batch(indices).permute(3, 0, 1, 2).float() / 255.0
    clip = TF.resize(clip, [resolution, resolution], antialias=True)
    return clip


def load_vjepa_encoder(checkpoint_path, checkpoint_key, model_name, device, resolution=224, num_frames=16):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Extract state dict
    if checkpoint_key in ckpt:
        state = ckpt[checkpoint_key]
    else:
        for k in ["encoder", "model", "state_dict"]:
            if k in ckpt:
                state = ckpt[k]
                break
        else:
            state = ckpt

    state = {k.replace("module.", "").replace("backbone.", ""): v for k, v in state.items()}

    # Create model with use_sdpa=False so attention weights flow through attn_drop
    model = vit.__dict__[model_name](
        img_size=resolution, num_frames=num_frames, patch_size=16, tubelet_size=2,
        uniform_power=True, use_rope=True, use_sdpa=False,
    )

    model_sd = model.state_dict()
    for k in list(state.keys()):
        if k in model_sd and state[k].shape != model_sd[k].shape:
            del state[k]
    model.load_state_dict(state, strict=False)
    model.eval().to(device)
    return model


def load_videomae_encoder(checkpoint_path, device, resolution=224, num_frames=16):
    from evals.video_classification_frozen.modelcustom.videomae_encoder import (
        _convert_pretrain_to_finetune_state_dict,
        _import_modeling_finetune,
    )
    mf = _import_modeling_finetune()
    # VideoMAE finetune ViT-L: always uses explicit attention (no SDPA)
    model = mf.vit_large_patch16_224(
        img_size=resolution, all_frames=num_frames, tubelet_size=2, num_classes=1000,
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)
    state = _convert_pretrain_to_finetune_state_dict(state, model.state_dict())
    model.load_state_dict(state, strict=False)
    model.eval().to(device)
    return model


def compute_temporal_attention_ratio(model, clip, device, num_frames=16):
    """
    Hook into attn_drop to capture attention weights from each layer.
    Compute cross-temporal attention ratio per layer.

    Tokens: N = T_patches * H_patches * W_patches = 8 * 14 * 14 = 1568
    Token i's temporal index: i // (14 * 14)
    """
    t_patches = num_frames // 2  # tubelet_size = 2
    h_patches = 14
    w_patches = 14
    tokens_per_frame = h_patches * w_patches  # 196
    n_tokens = t_patches * tokens_per_frame  # 1568

    # Temporal index for each token
    temporal_idx = torch.arange(n_tokens) // tokens_per_frame
    # Cross-temporal mask: True where tokens are from different frames
    cross_mask = (temporal_idx.unsqueeze(0) != temporal_idx.unsqueeze(1)).float()  # [N, N]

    # Storage for captured attention weights
    captured_attn = {}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            # input[0] is the attention weights (post-softmax, pre-dropout)
            captured_attn[layer_idx] = input[0].detach()
        return hook_fn

    # Register hooks on attn_drop in each block
    hooks = []
    for i, block in enumerate(model.blocks):
        h = block.attn.attn_drop.register_forward_hook(make_hook(i))
        hooks.append(h)

    # Forward pass
    x = clip.unsqueeze(0).to(device)
    with torch.no_grad():
        _ = model(x)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Compute cross-temporal ratios
    cm = cross_mask.to(device)
    layer_cross_ratios = []
    layer_per_head_ratios = []

    for layer_idx in sorted(captured_attn.keys()):
        attn_weights = captured_attn[layer_idx]  # [B, H, N, N]
        # Per-head cross-temporal ratio: fraction of attention to different-frame tokens
        cross_attn_per_head = (attn_weights * cm).sum(dim=-1).mean(dim=-1)  # [B, H]
        mean_ratio = cross_attn_per_head.mean().item()
        per_head = cross_attn_per_head[0].cpu().numpy().tolist()  # [H]

        layer_cross_ratios.append(mean_ratio)
        layer_per_head_ratios.append(per_head)

    return layer_cross_ratios, layer_per_head_ratios


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device)

    print(f"Loading encoder ({args.encoder_type}): {args.checkpoint}")
    if args.encoder_type == "videomae":
        model = load_videomae_encoder(args.checkpoint, device, args.resolution, args.num_frames)
    else:
        model = load_vjepa_encoder(
            args.checkpoint, args.checkpoint_key, args.model_name,
            device, args.resolution, args.num_frames,
        )
    n_layers = len(model.blocks)
    n_heads = model.blocks[0].attn.num_heads
    print(f"Model loaded: {n_layers} layers, {n_heads} heads")

    paths, labels = read_csv(args.test_csv)
    rng = np.random.RandomState(args.seed)
    indices = rng.permutation(len(paths))[:args.n_videos]

    all_ratios = []  # [n_videos, n_layers]
    all_per_head = []  # [n_videos, n_layers, n_heads]

    for i, idx in enumerate(indices):
        video_path = paths[idx]
        try:
            clip = load_clip(video_path, args.num_frames, args.frame_step, args.resolution)
        except Exception as e:
            print(f"  Skip {video_path}: {e}")
            continue

        ratios, per_head = compute_temporal_attention_ratio(model, clip, device, args.num_frames)
        all_ratios.append(ratios)
        all_per_head.append(per_head)

        if (i + 1) % 5 == 0 or i == 0:
            mean_ratio = np.mean(ratios)
            print(f"  [{i+1}/{len(indices)}] Mean cross-temporal ratio: {mean_ratio:.4f}")

    all_ratios = np.array(all_ratios)  # [n_videos, n_layers]
    mean_per_layer = all_ratios.mean(axis=0)
    overall_mean = mean_per_layer.mean()

    print(f"\n=== {args.label} ===")
    print(f"Overall cross-temporal attention ratio: {overall_mean:.4f}")
    print(f"Per-layer cross-temporal ratios:")
    for layer_idx, r in enumerate(mean_per_layer):
        print(f"  Layer {layer_idx:2d}: {r:.4f}")

    # Save per-layer CSV
    csv_path = os.path.join(args.save_dir, f"{args.label}_temporal_attention.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer"] + [f"video_{j}" for j in range(len(all_ratios))] + ["mean"])
        for layer_idx in range(n_layers):
            row = [layer_idx] + [f"{all_ratios[j, layer_idx]:.6f}" for j in range(len(all_ratios))]
            row.append(f"{mean_per_layer[layer_idx]:.6f}")
            w.writerow(row)
    print(f"Saved: {csv_path}")

    # Save per-head CSV
    all_per_head = np.array(all_per_head)  # [n_videos, n_layers, n_heads]
    head_csv = os.path.join(args.save_dir, f"{args.label}_temporal_attention_per_head.csv")
    with open(head_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "head", "mean_cross_temporal_ratio"])
        mean_per_head = all_per_head.mean(axis=0)  # [n_layers, n_heads]
        for layer_idx in range(n_layers):
            for head_idx in range(n_heads):
                w.writerow([layer_idx, head_idx, f"{mean_per_head[layer_idx, head_idx]:.6f}"])
    print(f"Saved: {head_csv}")


if __name__ == "__main__":
    main()
