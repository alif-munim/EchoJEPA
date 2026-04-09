"""
MAE temporal context sensitivity: reconstruction visualization.

For each VideoMAE checkpoint, feed the same video+mask under two conditions:
  A) Natural frame order (adjacent temporal context available)
  B) Shuffled frame order (temporal context disrupted)

Compare reconstructions. At early training (e50), reconstructions differ because
the model uses temporal context. At late training (e99), reconstructions converge
because the model reconstructs from spatial context alone.

Outputs:
  - Per-video pixel MSE between condition A and B reconstructions
  - Sample visualizations: original, masked, recon-A, recon-B, |A-B| difference
  - Summary CSV with per-checkpoint temporal context sensitivity

Usage:
  python scripts/rebuttal/mae_reconstruction_temporal.py \
    --checkpoint /path/to/checkpoint.pth \
    --test_csv /path/to/test_local.csv \
    --label mae_e50 \
    --save_dir scripts/rebuttal/mae_recon_vis \
    --n_videos 100 --n_vis 5 --device cuda:0
"""

import argparse
import os
import csv
import hashlib

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from einops import rearrange
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import ToPILImage
import torchvision.transforms.functional as TF
import decord


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="VideoMAE pretraining checkpoint (.pth)")
    p.add_argument("--test_csv", required=True, help="CSV: path label (one per line)")
    p.add_argument("--label", required=True, help="Label for outputs, e.g. mae_e50")
    p.add_argument("--save_dir", default="scripts/rebuttal/mae_recon_vis")
    p.add_argument("--n_videos", type=int, default=100, help="Number of videos for quantitative eval")
    p.add_argument("--n_vis", type=int, default=5, help="Number of videos to save visualizations for")
    p.add_argument("--mask_ratio", type=float, default=0.75)
    p.add_argument("--num_frames", type=int, default=16)
    p.add_argument("--frame_step", type=int, default=2)
    p.add_argument("--input_size", type=int, default=224)
    p.add_argument("--decoder_depth", type=int, default=4)
    p.add_argument("--device", default="cuda:0")
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


def load_clip_frames(video_path, num_frames=16, frame_step=2, input_size=224):
    """Load video as [C, T, H, W] tensor in [0, 1], plus PIL frames for vis."""
    vr = decord.VideoReader(video_path, num_threads=1)
    total = len(vr)
    needed = num_frames * frame_step
    start = max(0, (total - needed) // 2)
    indices = list(range(start, min(start + needed, total), frame_step))
    while len(indices) < num_frames:
        indices.append(indices[-1])
    indices = indices[:num_frames]
    raw = vr.get_batch(indices).asnumpy()  # [T, H, W, C]
    pil_frames = [Image.fromarray(raw[t]).convert("RGB") for t in range(num_frames)]

    # Resize + normalize (ImageNet stats)
    clip = torch.from_numpy(raw).permute(3, 0, 1, 2).float() / 255.0  # [C, T, H, W]
    clip = TF.resize(clip, [input_size, input_size], antialias=True)
    mean = torch.tensor(IMAGENET_DEFAULT_MEAN)[:, None, None, None]
    std = torch.tensor(IMAGENET_DEFAULT_STD)[:, None, None, None]
    clip = (clip - mean) / std
    return clip, pil_frames


def make_tube_mask(num_temporal, h_patches, w_patches, mask_ratio, seed):
    """Generate tube mask: same spatial mask applied across all temporal positions."""
    rng = np.random.RandomState(seed)
    num_patches_per_frame = h_patches * w_patches
    num_masks_per_frame = int(mask_ratio * num_patches_per_frame)

    mask_per_frame = np.zeros(num_patches_per_frame)
    mask_per_frame[:num_masks_per_frame] = 1
    rng.shuffle(mask_per_frame)
    mask = np.tile(mask_per_frame, (num_temporal, 1)).flatten()
    return torch.from_numpy(mask).unsqueeze(0).bool()  # [1, T*H*W]


def reconstruct(model, clip, bool_masked_pos, device, patch_size, num_frames):
    """Run encoder+decoder and return reconstructed pixel tensor [1, C, T, H, W]."""
    x = clip.unsqueeze(0).to(device)  # [1, C, T, H, W]
    mask = bool_masked_pos.to(device)  # [1, N]

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        outputs = model(x, mask)  # predicted patches for masked positions

    # Denormalize original for patch-level stats
    mean_img = torch.tensor(IMAGENET_DEFAULT_MEAN, device=device)[None, :, None, None, None]
    std_img = torch.tensor(IMAGENET_DEFAULT_STD, device=device)[None, :, None, None, None]
    ori = x * std_img + mean_img  # [1, C, T, H, W] in [0, 1]

    ps = patch_size[0]
    # Reshape to patches
    img_squeeze = rearrange(
        ori, "b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c",
        p0=2, p1=ps, p2=ps,
    )
    img_norm = (img_squeeze - img_squeeze.mean(dim=-2, keepdim=True)) / (
        img_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6
    )
    img_patch = rearrange(img_norm, "b n p c -> b n (p c)")

    # Insert predictions at masked positions
    img_patch[mask] = outputs.float()

    # Denormalize patches back to pixel space
    rec = rearrange(img_patch, "b n (p c) -> b n p c", c=3)
    rec = rec * (img_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6) + img_squeeze.mean(
        dim=-2, keepdim=True
    )
    rec = rearrange(
        rec, "b (t h w) (p0 p1 p2) c -> b c (t p0) (h p1) (w p2)",
        p0=2, p1=ps, p2=ps, h=14, w=14,
    )
    return rec.clamp(0, 1).cpu(), ori.cpu()


def save_vis(save_dir, video_idx, label, ori, rec_natural, rec_shuffled, mask,
             patch_size, target_frame=4):
    """Save visualization for one video: original, recon-A, recon-B, |A-B|."""
    out = os.path.join(save_dir, label, f"video_{video_idx}")
    os.makedirs(out, exist_ok=True)
    to_pil = ToPILImage()

    # Save a few frames
    for t in [0, target_frame, 7, 15]:
        if t >= ori.shape[2]:
            continue
        to_pil(ori[0, :, t]).save(os.path.join(out, f"f{t:02d}_original.png"))
        to_pil(rec_natural[0, :, t]).save(os.path.join(out, f"f{t:02d}_recon_natural.png"))
        to_pil(rec_shuffled[0, :, t]).save(os.path.join(out, f"f{t:02d}_recon_shuffled.png"))

        # Difference map (amplified for visibility)
        diff = (rec_natural[0, :, t] - rec_shuffled[0, :, t]).abs()
        diff_amplified = (diff * 5).clamp(0, 1)  # 5x amplification
        to_pil(diff_amplified).save(os.path.join(out, f"f{t:02d}_diff_5x.png"))

    # Save mask visualization for target frame
    ps = patch_size[0]
    h_patches, w_patches = 14, 14
    t_patches = 8  # num_frames // tubelet_size
    mask_flat = mask[0].cpu().numpy()
    # Reshape to (T_patches, H_patches, W_patches)
    mask_3d = mask_flat.reshape(t_patches, h_patches, w_patches)
    # Target frame's temporal patch index
    tf_patch = target_frame // 2  # tubelet_size = 2
    mask_frame = mask_3d[tf_patch]  # (14, 14)
    mask_img = np.kron(mask_frame, np.ones((ps, ps)))  # upsample to pixel res
    mask_img = (mask_img * 255).astype(np.uint8)
    Image.fromarray(mask_img, mode="L").save(os.path.join(out, f"f{target_frame:02d}_mask.png"))

    print(f"  Saved vis to {out}")


def main():
    args = parse_args()
    os.makedirs(os.path.join(args.save_dir, args.label), exist_ok=True)
    device = torch.device(args.device)
    cudnn.benchmark = True

    # --- Step 1: Check checkpoint for decoder keys ---
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    if "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt

    encoder_keys = [k for k in state if k.startswith("encoder.")]
    decoder_keys = [k for k in state if k.startswith("decoder.")]
    other_keys = [k for k in state if not k.startswith("encoder.") and not k.startswith("decoder.")]
    print(f"Checkpoint keys: {len(encoder_keys)} encoder, {len(decoder_keys)} decoder, "
          f"{len(other_keys)} other ({other_keys[:5]}...)")

    if len(decoder_keys) == 0:
        print("ERROR: No decoder keys found! This checkpoint was likely stripped for eval.")
        print("Need a pretraining checkpoint with encoder + decoder.")
        return

    # --- Step 2: Load VideoMAE model ---
    import sys
    sys.path.insert(0, "evals/video_classification_frozen/modelcustom/VideoMAE")
    from functools import partial
    from modeling_pretrain import PretrainVisionTransformer

    print("Creating PretrainVisionTransformer (ViT-L)")
    model = PretrainVisionTransformer(
        img_size=224,
        patch_size=16,
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_num_classes=0,
        decoder_num_classes=1536,  # patch_size * patch_size * 3 / tubelet_size * tubelet_size ... = 16*16*2*3=1536
        decoder_embed_dim=512,
        decoder_depth=args.decoder_depth,
        decoder_num_heads=8,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
    )
    msg = model.load_state_dict(state, strict=True)
    print(f"Load result: {msg}")
    model.eval().to(device)

    patch_size = model.encoder.patch_embed.patch_size
    print(f"Patch size: {patch_size}")
    h_patches = args.input_size // patch_size[0]
    w_patches = args.input_size // patch_size[1]
    t_patches = args.num_frames // 2  # tubelet_size = 2

    # --- Step 3: Load videos ---
    paths, labels = read_csv(args.test_csv)
    # Deterministic selection
    rng = np.random.RandomState(args.seed)
    indices = rng.permutation(len(paths))[:args.n_videos]
    print(f"Processing {len(indices)} videos (vis for first {args.n_vis})")

    # --- Step 4: Run experiment ---
    mse_diffs = []
    for i, idx in enumerate(indices):
        video_path = paths[idx]
        # Deterministic mask seed per video
        mask_seed = int(hashlib.md5(video_path.encode()).hexdigest()[:8], 16) % (2**31)
        mask = make_tube_mask(t_patches, h_patches, w_patches, args.mask_ratio, mask_seed)

        try:
            clip_natural, pil_frames = load_clip_frames(
                video_path, args.num_frames, args.frame_step, args.input_size
            )
        except Exception as e:
            print(f"  Skip {video_path}: {e}")
            continue

        # Condition A: natural frame order
        rec_natural, ori = reconstruct(model, clip_natural, mask, device, patch_size, args.num_frames)

        # Condition B: shuffled frame order
        perm = rng.permutation(args.num_frames)
        clip_shuffled = clip_natural[:, perm, :, :]  # shuffle temporal dim
        # Reconstruct with same mask
        rec_shuffled_perm, _ = reconstruct(model, clip_shuffled, mask, device, patch_size, args.num_frames)
        # Un-shuffle to align with natural order for comparison
        inv_perm = np.argsort(perm)
        rec_shuffled = rec_shuffled_perm[:, :, inv_perm, :, :]

        # Compute MSE between reconstructions (only at masked positions in pixel space)
        mse = ((rec_natural - rec_shuffled) ** 2).mean().item()
        mse_diffs.append(mse)

        if i < args.n_vis:
            save_vis(
                args.save_dir, i, args.label, ori, rec_natural, rec_shuffled,
                mask, patch_size, target_frame=4,
            )

        if (i + 1) % 20 == 0 or i == len(indices) - 1:
            print(f"  [{i+1}/{len(indices)}] Running mean MSE(A-B): {np.mean(mse_diffs):.6f}")

    # --- Step 5: Summary ---
    mean_mse = np.mean(mse_diffs)
    std_mse = np.std(mse_diffs)
    print(f"\n=== {args.label} ===")
    print(f"Temporal context sensitivity (MSE between natural/shuffled recons):")
    print(f"  Mean: {mean_mse:.6f}  Std: {std_mse:.6f}  N: {len(mse_diffs)}")

    # Save per-video CSV
    csv_path = os.path.join(args.save_dir, f"{args.label}_temporal_sensitivity.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_idx", "mse_diff"])
        for j, m in enumerate(mse_diffs):
            w.writerow([j, f"{m:.8f}"])
    print(f"Saved: {csv_path}")

    # Append to summary
    summary_path = os.path.join(args.save_dir, "temporal_sensitivity_summary.csv")
    write_header = not os.path.exists(summary_path)
    with open(summary_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["checkpoint", "mean_mse", "std_mse", "n_videos"])
        w.writerow([args.label, f"{mean_mse:.8f}", f"{std_mse:.8f}", len(mse_diffs)])
    print(f"Appended to: {summary_path}")


if __name__ == "__main__":
    main()
