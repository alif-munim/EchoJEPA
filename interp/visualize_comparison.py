"""Side-by-side comparison: VideoMAE (given) | Original | EchoJEPA (given).

Exact logic from visualize_zscore_attention.ipynb, adapted to run on a
directory of videos.  For each video it outputs:
  <output-dir>/<video_stem>/
      <video_stem>_comparison.mp4      — 3-panel video
      images/frame_XX_comparison.png   — per-frame PNGs
"""

import argparse
import re
import os
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import Normalize
from transformers import AutoModel, AutoVideoProcessor, VideoMAEImageProcessor, VideoMAEModel
import warnings

warnings.filterwarnings("ignore")
matplotlib.rcParams["font.family"] = "Liberation Sans"


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="3-panel attention comparison: VideoMAE | Original | EchoJEPA"
    )
    p.add_argument("--video-dirs", nargs="+",
                   default=["/large_storage/goodarzilab/adibvafa/mimic_p10_masked"],
                   help="Directories containing .mp4 videos")
    p.add_argument("--output-dir", type=str,
                   default="/large_storage/goodarzilab/adibvafa/echo_visualizations/visualizations",
                   help="Output root directory")
    p.add_argument("--vjepa-ckpt", type=str,
                   default="/large_storage/goodarzilab/adibvafa/pt-280-an81.pt",
                   help="Path to VJEPA2 checkpoint (.pt)")
    p.add_argument("--mae-ckpt", type=str,
                   default="/large_storage/goodarzilab/adibvafa/video_mae.pth",
                   help="Path to VideoMAE checkpoint (.pth)")
    p.add_argument("--vjepa-model", type=str,
                   default="facebook/vjepa2-vitg-fpc64-384")
    p.add_argument("--mae-model", type=str,
                   default="MCG-NJU/videomae-large")
    p.add_argument("--num-frames", type=int, default=64,
                   help="Frames to sample per video (must be divisible by --mae-chunk-frames)")
    p.add_argument("--mae-chunk-frames", type=int, default=16,
                   help="VideoMAE chunk size (VideoMAE expects exactly this many frames)")
    p.add_argument("--num-videos", type=int, default=0,
                   help="Max videos per directory (0 = all)")
    p.add_argument("--attn-layer", type=int, default=0,
                   help="Encoder layer to visualize")
    p.add_argument("--zscore-threshold-given", type=float, default=0.7)
    p.add_argument("--alpha", type=float, default=0.8)
    p.add_argument("--zscore-vmax", type=float, default=3.0)
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--fig-width", type=float, default=13.0,
                   help="Figure width in inches")
    p.add_argument("--fig-height", type=float, default=5.0,
                   help="Figure height in inches")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Checkpoint conversion (VJEPA2)
# ---------------------------------------------------------------------------
def convert_vjepa_checkpoint(ckpt):
    new_sd = {}
    for k, v in ckpt["encoder"].items():
        k = k.replace("module.backbone.", "")
        if k == "patch_embed.proj.weight":
            new_sd["encoder.embeddings.patch_embeddings.proj.weight"] = v
        elif k == "patch_embed.proj.bias":
            new_sd["encoder.embeddings.patch_embeddings.proj.bias"] = v
        elif k == "norm.weight":
            new_sd["encoder.layernorm.weight"] = v
        elif k == "norm.bias":
            new_sd["encoder.layernorm.bias"] = v
        elif "blocks." in k:
            m = re.match(r"blocks\.(\d+)\.(.+)", k)
            idx, rest = m.group(1), m.group(2)
            if rest == "attn.qkv.weight":
                q, k_, v_ = v.chunk(3, dim=0)
                new_sd[f"encoder.layer.{idx}.attention.query.weight"] = q
                new_sd[f"encoder.layer.{idx}.attention.key.weight"] = k_
                new_sd[f"encoder.layer.{idx}.attention.value.weight"] = v_
            elif rest == "attn.qkv.bias":
                q, k_, v_ = v.chunk(3, dim=0)
                new_sd[f"encoder.layer.{idx}.attention.query.bias"] = q
                new_sd[f"encoder.layer.{idx}.attention.key.bias"] = k_
                new_sd[f"encoder.layer.{idx}.attention.value.bias"] = v_
            elif rest == "attn.proj.weight":
                new_sd[f"encoder.layer.{idx}.attention.proj.weight"] = v
            elif rest == "attn.proj.bias":
                new_sd[f"encoder.layer.{idx}.attention.proj.bias"] = v
            else:
                new_sd[f"encoder.layer.{idx}.{rest}"] = v

    mask_tokens = []
    for k, v in ckpt["predictor"].items():
        k = k.replace("module.backbone.", "")
        if k == "predictor_embed.weight":
            new_sd["predictor.embeddings.predictor_embeddings.weight"] = v
        elif k == "predictor_embed.bias":
            new_sd["predictor.embeddings.predictor_embeddings.bias"] = v
        elif k == "predictor_norm.weight":
            new_sd["predictor.layernorm.weight"] = v
        elif k == "predictor_norm.bias":
            new_sd["predictor.layernorm.bias"] = v
        elif k == "predictor_proj.weight":
            new_sd["predictor.proj.weight"] = v
        elif k == "predictor_proj.bias":
            new_sd["predictor.proj.bias"] = v
        elif k.startswith("mask_tokens."):
            mask_tokens.append((int(k.split(".")[1]), v))
        elif "predictor_blocks." in k:
            m = re.match(r"predictor_blocks\.(\d+)\.(.+)", k)
            idx, rest = m.group(1), m.group(2)
            if rest == "attn.qkv.weight":
                q, k_, v_ = v.chunk(3, dim=0)
                new_sd[f"predictor.layer.{idx}.attention.query.weight"] = q
                new_sd[f"predictor.layer.{idx}.attention.key.weight"] = k_
                new_sd[f"predictor.layer.{idx}.attention.value.weight"] = v_
            elif rest == "attn.qkv.bias":
                q, k_, v_ = v.chunk(3, dim=0)
                new_sd[f"predictor.layer.{idx}.attention.query.bias"] = q
                new_sd[f"predictor.layer.{idx}.attention.key.bias"] = k_
                new_sd[f"predictor.layer.{idx}.attention.value.bias"] = v_
            elif rest == "attn.proj.weight":
                new_sd[f"predictor.layer.{idx}.attention.proj.weight"] = v
            elif rest == "attn.proj.bias":
                new_sd[f"predictor.layer.{idx}.attention.proj.bias"] = v
            else:
                new_sd[f"predictor.layer.{idx}.{rest}"] = v

    mask_tokens.sort(key=lambda x: x[0])
    new_sd["predictor.embeddings.mask_tokens"] = torch.stack([t for _, t in mask_tokens])
    return new_sd


# ---------------------------------------------------------------------------
# Checkpoint conversion (VideoMAE)
# ---------------------------------------------------------------------------
def load_mae_checkpoint(model, ckpt_path):
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")["model"]
    new_sd = {}

    new_sd["embeddings.patch_embeddings.projection.weight"] = ckpt["encoder.patch_embed.proj.weight"]
    new_sd["embeddings.patch_embeddings.projection.bias"] = ckpt["encoder.patch_embed.proj.bias"]

    for i in range(24):
        pc = f"encoder.blocks.{i}"
        pm = f"encoder.layer.{i}"
        new_sd[f"{pm}.layernorm_before.weight"] = ckpt[f"{pc}.norm1.weight"]
        new_sd[f"{pm}.layernorm_before.bias"] = ckpt[f"{pc}.norm1.bias"]
        new_sd[f"{pm}.layernorm_after.weight"] = ckpt[f"{pc}.norm2.weight"]
        new_sd[f"{pm}.layernorm_after.bias"] = ckpt[f"{pc}.norm2.bias"]

        q, k_, v_ = ckpt[f"{pc}.attn.qkv.weight"].chunk(3, dim=0)
        new_sd[f"{pm}.attention.attention.query.weight"] = q
        new_sd[f"{pm}.attention.attention.key.weight"] = k_
        new_sd[f"{pm}.attention.attention.value.weight"] = v_
        new_sd[f"{pm}.attention.attention.q_bias"] = ckpt[f"{pc}.attn.q_bias"]
        new_sd[f"{pm}.attention.attention.v_bias"] = ckpt[f"{pc}.attn.v_bias"]
        new_sd[f"{pm}.attention.output.dense.weight"] = ckpt[f"{pc}.attn.proj.weight"]
        new_sd[f"{pm}.attention.output.dense.bias"] = ckpt[f"{pc}.attn.proj.bias"]

        new_sd[f"{pm}.intermediate.dense.weight"] = ckpt[f"{pc}.mlp.fc1.weight"]
        new_sd[f"{pm}.intermediate.dense.bias"] = ckpt[f"{pc}.mlp.fc1.bias"]
        new_sd[f"{pm}.output.dense.weight"] = ckpt[f"{pc}.mlp.fc2.weight"]
        new_sd[f"{pm}.output.dense.bias"] = ckpt[f"{pc}.mlp.fc2.bias"]

    new_sd["layernorm.weight"] = ckpt["encoder.norm.weight"]
    new_sd["layernorm.bias"] = ckpt["encoder.norm.bias"]

    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    print(f"  MAE checkpoint — missing: {len(missing)}, unexpected: {len(unexpected)}")
    return model


# ---------------------------------------------------------------------------
# Attention helpers (exact notebook logic)
# ---------------------------------------------------------------------------
def resize_attn(attn_map, target_size):
    return F.interpolate(
        attn_map.unsqueeze(0).unsqueeze(0).float(),
        size=(target_size, target_size),
        mode="bilinear",
        align_corners=False,
    ).squeeze()


def maybe_invert_attention(attn_norm, frame):
    """Invert attention if the biggest image area has the highest average attention."""
    gray = np.mean(frame, axis=2) if frame.ndim == 3 else frame
    median_val = np.median(gray)
    region1_mask = gray > median_val
    region2_mask = ~region1_mask
    area1, area2 = region1_mask.sum(), region2_mask.sum()
    avg1 = attn_norm[region1_mask].mean() if area1 > 0 else 0
    avg2 = attn_norm[region2_mask].mean() if area2 > 0 else 0
    bigger_avg = avg1 if area1 > area2 else avg2
    smaller_avg = avg2 if area1 > area2 else avg1
    if bigger_avg > smaller_avg:
        return 1.0 - attn_norm, True
    return attn_norm, False


def compute_zscore_overlay(attn_map, target_size, threshold, zscore_vmax):
    z = (attn_map - attn_map.mean()) / (attn_map.std() + 1e-8)
    z_resized = resize_attn(z, target_size).numpy()
    z_norm = (np.clip(z_resized, -zscore_vmax, zscore_vmax) + zscore_vmax) / (2 * zscore_vmax)
    return z_norm, np.ma.masked_where(z_norm < threshold, z_norm)


def find_best_head(attn_tensor, num_heads):
    """Return index of the most peaked head."""
    peakiness = [
        (h, attn_tensor[h].max().item() / attn_tensor[h].mean().item())
        for h in range(num_heads)
    ]
    return max(peakiness, key=lambda x: x[1])[0]


def precompute_given_maps(attn_tensor, best_head, num_temporal, num_spatial, spatial_h, spatial_w):
    """Extract per-temporal-token attention-given maps for the best head."""
    best = attn_tensor[best_head]
    given = []
    for t in range(num_temporal):
        s, e = t * num_spatial, (t + 1) * num_spatial
        given.append(best[s:e, s:e].sum(dim=1).view(spatial_h, spatial_w))
    return given


# ---------------------------------------------------------------------------
# Collect videos
# ---------------------------------------------------------------------------
def collect_videos(video_dirs, num_videos):
    videos = []
    for d in video_dirs:
        found = sorted(f for f in os.listdir(d) if f.endswith(".mp4"))
        if num_videos > 0:
            found = found[:num_videos]
        videos.extend((d, f) for f in found)
    return videos


# ---------------------------------------------------------------------------
# VJEPA2: load model, run one video, return attention maps + metadata
# ---------------------------------------------------------------------------
def run_vjepa_on_video(video_frames, model, processor, attn_layer, device):
    video_tensor = torch.from_numpy(video_frames).permute(0, 3, 1, 2)
    inputs = processor(video_tensor, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Monkey-patch target layer to eager for attention capture
    target_attn = model.encoder.layer[attn_layer].attention
    orig_impl = target_attn.config._attn_implementation
    target_attn.config = type(target_attn.config)(
        **{**target_attn.config.to_dict(), "_attn_implementation": "eager"}
    )

    captured = {}
    orig_fwd = model.encoder.layer[attn_layer].forward

    def hook(hidden_states, position_mask=None, head_mask=None, output_attentions=False):
        result = orig_fwd(hidden_states, position_mask, head_mask, output_attentions=True)
        captured["w"] = result[1].detach().cpu()
        return (result[0],)

    model.encoder.layer[attn_layer].forward = hook

    with torch.no_grad():
        model(**inputs, output_attentions=False)

    # Restore
    model.encoder.layer[attn_layer].forward = orig_fwd
    target_attn.config = type(target_attn.config)(
        **{**target_attn.config.to_dict(), "_attn_implementation": orig_impl}
    )

    cfg = model.config
    B, T_in, C, H, W = inputs["pixel_values_videos"].shape
    spatial_h = H // cfg.patch_size
    spatial_w = W // cfg.patch_size
    num_spatial = spatial_h * spatial_w
    num_temporal = T_in // cfg.tubelet_size

    # Denormalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    frames = ((inputs["pixel_values_videos"].squeeze(0).cpu() * std + mean)
              .clamp(0, 1).permute(0, 2, 3, 1).numpy())

    attn = captured["w"].squeeze(0)  # [heads, seq, seq]
    best_head = find_best_head(attn, cfg.num_attention_heads)
    given = precompute_given_maps(attn, best_head, num_temporal, num_spatial, spatial_h, spatial_w)

    del inputs, captured
    torch.cuda.empty_cache()

    return {
        "frames": frames,
        "given": given,
        "T_in": T_in,
        "tubelet_size": cfg.tubelet_size,
        "num_temporal": num_temporal,
        "IMG_SIZE": H,
    }


# ---------------------------------------------------------------------------
# VideoMAE: run one video in N-frame chunks, return attention maps
# ---------------------------------------------------------------------------
def run_mae_on_video(video_frames, model, processor, attn_layer, device, mae_chunk_frames):
    cfg = model.config
    num_chunks = len(video_frames) // mae_chunk_frames

    mae_total_patches = (
        (cfg.image_size // cfg.patch_size) ** 2
        * (cfg.num_frames // cfg.tubelet_size)
    )
    spatial_h = cfg.image_size // cfg.patch_size
    num_spatial = spatial_h * spatial_h
    temporal_per_chunk = mae_chunk_frames // cfg.tubelet_size

    all_given = []
    best_head = None

    for ci in range(num_chunks):
        s, e = ci * mae_chunk_frames, (ci + 1) * mae_chunk_frames
        chunk = video_frames[s:e]

        pv = processor([f for f in chunk], return_tensors="pt").pixel_values.to(device)
        mask = torch.zeros(1, mae_total_patches, dtype=torch.bool).to(device)

        with torch.no_grad():
            out = model(pv, bool_masked_pos=mask, output_attentions=True, return_dict=True)

        attn = out.attentions[attn_layer].squeeze(0).cpu()  # [heads, seq, seq]

        if best_head is None:
            best_head = find_best_head(attn, cfg.num_attention_heads)

        for t in range(temporal_per_chunk):
            st, en = t * num_spatial, (t + 1) * num_spatial
            all_given.append(
                attn[best_head][st:en, st:en].sum(dim=1).view(spatial_h, spatial_h)
            )

        del out, pv, mask
        torch.cuda.empty_cache()

    return {
        "given": all_given,
        "tubelet_size": cfg.tubelet_size,
        "num_temporal": len(all_given),
    }


# ---------------------------------------------------------------------------
# Render one comparison frame → numpy BGR image
# ---------------------------------------------------------------------------
def render_comparison_frame(frame, mae_masked, vjepa_masked, args, sm):
    fig = plt.Figure(figsize=(args.fig_width, args.fig_height), dpi=args.dpi, facecolor="black")
    canvas = FigureCanvasAgg(fig)
    gs = fig.add_gridspec(1, 3, wspace=-0.1, left=0.02, right=0.92, top=0.88, bottom=0.08)

    # Left: VideoMAE
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(frame)
    ax0.imshow(mae_masked, cmap="jet", alpha=args.alpha,
               vmin=args.zscore_threshold_given, vmax=1)
    ax0.set_title("VideoMAE", color="white", fontsize=15, fontweight="bold", pad=8)
    ax0.axis("off")

    # Middle: Original
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(frame)
    ax1.set_title("Original", color="white", fontsize=15, fontweight="bold", pad=8)
    ax1.axis("off")

    # Right: EchoJEPA
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.imshow(frame)
    ax2.imshow(vjepa_masked, cmap="jet", alpha=args.alpha,
               vmin=args.zscore_threshold_given, vmax=1)
    ax2.set_title("EchoJEPA", color="white", fontsize=15, fontweight="bold", pad=8)
    ax2.axis("off")

    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.21, 0.012, 0.6])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.ax.tick_params(colors="white", labelsize=7)
    cbar.set_label("Attention Z-Score", color="white", fontsize=10, labelpad=6)
    cbar.outline.set_edgecolor("white")

    canvas.draw()
    buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    w, h = canvas.get_width_height()
    img_bgr = cv2.cvtColor(buf.reshape(h, w, 4)[:, :, :3], cv2.COLOR_RGB2BGR)
    plt.close(fig)
    return img_bgr, (w, h)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    assert args.num_frames % args.mae_chunk_frames == 0, \
        f"--num-frames ({args.num_frames}) must be divisible by --mae-chunk-frames ({args.mae_chunk_frames})"

    # ---- Load VJEPA2 model (SDPA + selective eager hook) ----
    print("Loading VJEPA2 model...")
    vjepa_processor = AutoVideoProcessor.from_pretrained(args.vjepa_model)
    vjepa_model = AutoModel.from_pretrained(
        args.vjepa_model, attn_implementation="sdpa", dtype=torch.float16
    )
    vjepa_model = vjepa_model.to(args.device).eval()

    print(f"  Loading checkpoint: {args.vjepa_ckpt}")
    ckpt = torch.load(args.vjepa_ckpt, map_location="cpu")
    vjepa_model.load_state_dict(convert_vjepa_checkpoint(ckpt), strict=False)
    del ckpt
    torch.cuda.empty_cache()
    print("  VJEPA2 ready")

    # ---- Load VideoMAE model ----
    print("Loading VideoMAE model...")
    mae_processor = VideoMAEImageProcessor.from_pretrained(args.mae_model)
    mae_model = VideoMAEModel.from_pretrained(args.mae_model, attn_implementation="eager")

    print(f"  Loading checkpoint: {args.mae_ckpt}")
    mae_model = load_mae_checkpoint(mae_model, args.mae_ckpt)

    mae_model = mae_model.to(args.device).eval()
    torch.cuda.empty_cache()
    print("  VideoMAE ready")

    # ---- Collect videos ----
    videos = collect_videos(args.video_dirs, args.num_videos)
    print(f"\nProcessing {len(videos)} videos\n")

    sm = plt.cm.ScalarMappable(
        cmap="jet", norm=Normalize(vmin=args.zscore_threshold_given, vmax=1)
    )

    for vid_idx, (video_dir, video_name) in enumerate(videos):
        print(f"[{vid_idx + 1}/{len(videos)}] {video_name} (from {video_dir})")
        video_path = os.path.join(video_dir, video_name)

        # Output paths
        stem = Path(video_name).stem
        case_dir = os.path.join(args.output_dir, stem)
        images_dir = os.path.join(case_dir, "images")
        os.makedirs(images_dir, exist_ok=True)

        # Load video frames
        vr = VideoReader(video_path, ctx=cpu(0))
        frame_indices = np.linspace(0, len(vr) - 1, args.num_frames, dtype=int)
        video_frames = vr.get_batch(frame_indices).asnumpy()

        # Run both models
        print("  Running VJEPA2...")
        vjepa_result = run_vjepa_on_video(
            video_frames, vjepa_model, vjepa_processor, args.attn_layer, args.device
        )

        print("  Running VideoMAE...")
        mae_result = run_mae_on_video(
            video_frames, mae_model, mae_processor, args.attn_layer, args.device,
            args.mae_chunk_frames
        )

        # Render comparison frames
        out_mp4 = os.path.join(case_dir, f"{stem}_comparison.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = None

        T_in = vjepa_result["T_in"]
        vjepa_ts = vjepa_result["tubelet_size"]
        mae_ts = mae_result["tubelet_size"]
        vjepa_nt = vjepa_result["num_temporal"]
        mae_nt = mae_result["num_temporal"]
        img_size = vjepa_result["IMG_SIZE"]

        print(f"  Rendering {T_in} frames...")
        for f_idx in range(T_in):
            vjepa_t = min(f_idx // vjepa_ts, vjepa_nt - 1)
            mae_t = min(f_idx // mae_ts, mae_nt - 1)
            frame = vjepa_result["frames"][f_idx]

            # VJEPA attention given
            vjepa_zg, _ = compute_zscore_overlay(
                vjepa_result["given"][vjepa_t], img_size,
                args.zscore_threshold_given, args.zscore_vmax
            )
            vjepa_zg, _ = maybe_invert_attention(vjepa_zg, frame)
            vjepa_masked = np.ma.masked_where(
                vjepa_zg < args.zscore_threshold_given, vjepa_zg
            )

            # MAE attention given (resized to vjepa img size)
            mae_zg, _ = compute_zscore_overlay(
                mae_result["given"][mae_t], img_size,
                args.zscore_threshold_given, args.zscore_vmax
            )
            mae_zg, _ = maybe_invert_attention(mae_zg, frame)
            mae_masked = np.ma.masked_where(
                mae_zg < args.zscore_threshold_given, mae_zg
            )

            img_bgr, (w, h) = render_comparison_frame(
                frame, mae_masked, vjepa_masked, args, sm
            )

            if writer is None:
                writer = cv2.VideoWriter(out_mp4, fourcc, args.fps, (w, h))
            writer.write(img_bgr)

            # Save PNG
            cv2.imwrite(
                os.path.join(images_dir, f"frame_{f_idx:03d}_comparison.png"),
                img_bgr,
            )

        writer.release()

        del vjepa_result, mae_result, video_frames
        torch.cuda.empty_cache()

        print(f"  Saved to: {case_dir}")

    print("\nDone!")


if __name__ == "__main__":
    main()
