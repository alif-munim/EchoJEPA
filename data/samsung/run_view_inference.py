"""
Run EchoJEPA-G view classification on Samsung masked videos.

Loads the frozen EchoJEPA-G encoder + best attentive probe head,
runs inference on 10 CV1 + 10 CV2 masked videos, and saves predictions.

Usage:
    python data/samsung/run_view_inference.py [--device cuda:0]

Output:
    data/samsung/view_predictions.csv
"""

import argparse
import glob
import json
import os
import sys

# Ensure repo root is on sys.path for src imports
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import decord
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm

# --- Config ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

MASKED_DIR = os.path.join(SCRIPT_DIR, "mp4_masked")
OUTPUT_CSV = os.path.join(SCRIPT_DIR, "view_predictions.csv")

ENCODER_CHECKPOINT = os.path.join(REPO_ROOT, "checkpoints", "anneal", "keep", "pt-280-an81.pt")
PROBE_CHECKPOINT = os.path.join(REPO_ROOT, "checkpoints", "eval_probes", "classification", "echojepa_224px.pt")

VIEW_MAP = {
    0: "A2C", 1: "A3C", 2: "A4C", 3: "A5C", 4: "Exclude", 5: "PLAX",
    6: "PSAX-AP", 7: "PSAX-AV", 8: "PSAX-MV", 9: "PSAX-PM", 10: "SSN",
    11: "Subcostal", 12: "TEE",
}

NUM_CLASSES = 13
RESOLUTION = 224
FRAMES_PER_CLIP = 16
FRAME_STEP = 2
NUM_SEGMENTS = 2
BEST_HEAD_IDX = 2  # head with 87.5% val acc


def select_videos(masked_dir, n_per_patient=10):
    """Select n videos from each patient (CV1, CV2)."""
    all_mp4s = sorted(glob.glob(os.path.join(masked_dir, "*.mp4")))
    cv1 = [f for f in all_mp4s if os.path.basename(f).startswith("CV1_")]
    cv2 = [f for f in all_mp4s if os.path.basename(f).startswith("CV2_")]
    selected = cv1[:n_per_patient] + cv2[:n_per_patient]
    print(f"Selected {len(selected)} videos ({min(n_per_patient, len(cv1))} CV1, {min(n_per_patient, len(cv2))} CV2)")
    return selected


def load_clips(video_path, frames_per_clip, frame_step, num_segments, resolution):
    """Load video clips matching the eval pipeline: num_segments temporal clips."""
    vr = decord.VideoReader(video_path, num_threads=1)
    total_frames = len(vr)

    clip_len = frames_per_clip * frame_step
    clips = []

    if total_frames >= clip_len:
        # Evenly space num_segments clips
        if num_segments == 1:
            starts = [0]
        else:
            max_start = total_frames - clip_len
            starts = np.linspace(0, max_start, num_segments).astype(int)
    else:
        starts = [0] * num_segments

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((resolution, resolution), antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    for start in starts:
        indices = list(range(start, min(start + clip_len, total_frames), frame_step))
        # Pad if not enough frames
        while len(indices) < frames_per_clip:
            indices.append(indices[-1])
        indices = indices[:frames_per_clip]

        frames = vr.get_batch(indices).asnumpy()  # (T, H, W, C)
        frame_tensors = []
        for f in frames:
            frame_tensors.append(transform(f))
        clip = torch.stack(frame_tensors, dim=1)  # (C, T, H, W)
        clips.append(clip)

    return clips


def load_encoder(checkpoint_path, device):
    """Load EchoJEPA-G encoder."""
    import src.models.vision_transformer as vit

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model = vit.vit_giant_xformers(
        img_size=RESOLUTION,
        num_frames=FRAMES_PER_CLIP,
        patch_size=16,
        tubelet_size=2,
        uniform_power=True,
        use_rope=True,
    )

    pretrained_dict = checkpoint["target_encoder"]
    pretrained_dict = {k.replace("module.", "").replace("backbone.", ""): v for k, v in pretrained_dict.items()}
    msg = model.load_state_dict(pretrained_dict, strict=False)
    print(f"Encoder loaded: {msg}")

    model = model.to(device).eval()
    return model


def load_probe(checkpoint_path, embed_dim, device):
    """Load best attentive classifier head."""
    from src.models.attentive_pooler import AttentiveClassifier

    probe = AttentiveClassifier(
        embed_dim=embed_dim,
        num_heads=16,
        depth=4,
        num_classes=NUM_CLASSES,
        use_activation_checkpointing=True,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["classifiers"][BEST_HEAD_IDX]
    # Strip module. prefix (from DDP)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    msg = probe.load_state_dict(state_dict, strict=True)
    print(f"Probe loaded (head {BEST_HEAD_IDX}): {msg}")

    probe = probe.eval()
    return probe


@torch.no_grad()
def run_inference(encoder, probe, video_paths, device):
    """Run view classification on a list of videos."""
    results = []

    for vpath in tqdm(video_paths, desc="Inference"):
        filename = os.path.basename(vpath)
        try:
            clips = load_clips(vpath, FRAMES_PER_CLIP, FRAME_STEP, NUM_SEGMENTS, RESOLUTION)

            # Each clip: (C, T, H, W) → batch them: (num_segments, C, T, H, W)
            clip_batch = torch.stack(clips, dim=0).to(device)

            # Encode each clip independently
            all_tokens = []
            for i in range(clip_batch.shape[0]):
                tokens = encoder(clip_batch[i : i + 1])  # (1, N, D)
                all_tokens.append(tokens)
            # Concatenate tokens from all clips: (1, N*num_segments, D)
            tokens = torch.cat(all_tokens, dim=1)

            # Probe
            logits = probe(tokens)  # (1, num_classes)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

            pred_class = int(np.argmax(probs))
            pred_view = VIEW_MAP[pred_class]
            confidence = float(probs[pred_class])

            results.append({
                "video_file": filename,
                "video_path": vpath,
                "predicted_class": pred_class,
                "predicted_view": pred_view,
                "confidence": confidence,
                **{f"prob_{VIEW_MAP[i]}": float(probs[i]) for i in range(NUM_CLASSES)},
            })

        except Exception as e:
            print(f"  Error on {filename}: {e}")
            results.append({
                "video_file": filename,
                "video_path": vpath,
                "predicted_class": -1,
                "predicted_view": "ERROR",
                "confidence": 0.0,
            })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--n_per_patient", type=int, default=10)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Select videos
    video_paths = select_videos(MASKED_DIR, n_per_patient=args.n_per_patient)

    # Load models
    print("Loading encoder...")
    encoder = load_encoder(ENCODER_CHECKPOINT, device)
    embed_dim = encoder.embed_dim
    print(f"Encoder embed_dim: {embed_dim}")

    print("Loading probe...")
    probe = load_probe(PROBE_CHECKPOINT, embed_dim, device)

    # Run inference
    print("Running inference...")
    results = run_inference(encoder, probe, video_paths, device)

    # Save
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved predictions to {OUTPUT_CSV}")
    print(df[["video_file", "predicted_view", "confidence"]].to_string(index=False))


if __name__ == "__main__":
    main()
