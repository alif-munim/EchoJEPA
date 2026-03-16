"""
Run view and color classification on echocardiogram MP4 files.

Uses ConvNeXt-Small (336px) classifiers with multi-frame voting. Each video
is sampled at N uniformly-spaced frames, each frame is classified independently,
and softmax probabilities are averaged for the final prediction.

Usage:
    python preprocessing/classify_views.py \
        --input_dir /path/to/mp4s_masked \
        --output_csv /path/to/classifications.csv \
        --view_checkpoint classifier/checkpoints/view_convnext_small_336px.pt \
        --color_checkpoint classifier/checkpoints/color_convnext_small_336px.pt \
        --num_frames 5 --batch_size 32

    # View only (skip color)
    python preprocessing/classify_views.py \
        --input_dir /path/to/mp4s_masked \
        --output_csv /path/to/classifications.csv \
        --view_checkpoint classifier/checkpoints/view_convnext_small_336px.pt \
        --num_frames 5

Output CSV columns:
    path, view, view_confidence, color, color_confidence
"""

import argparse
import csv
import os

import decord
import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from timm.data import create_transform
from torch.amp import autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Label maps (must match classifier training)
VIEW_LABELS = {
    0: "A2C", 1: "A3C", 2: "A4C", 3: "A5C", 4: "Exclude", 5: "PLAX",
    6: "PSAX-AP", 7: "PSAX-AV", 8: "PSAX-MV", 9: "PSAX-PM", 10: "SSN",
    11: "Subcostal", 12: "TEE",
}
COLOR_LABELS = {0: "No", 1: "Yes"}

IMG_SIZE = 336
MODEL_NAME = "convnext_small.fb_in1k"


class VideoFrameDataset(Dataset):
    """Dataset that loads N uniformly-spaced frames per video."""

    def __init__(self, paths, transform, num_frames):
        self.paths = paths
        self.transform = transform
        self.num_frames = num_frames

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]

        try:
            vr = decord.VideoReader(path, num_threads=1)
            total = len(vr)

            if total < 1:
                return torch.zeros(self.num_frames, 3, IMG_SIZE, IMG_SIZE), path, False

            if total < self.num_frames:
                indices = np.arange(total).astype(int)
            else:
                indices = np.linspace(0, total - 1, self.num_frames).astype(int)

            frames = vr.get_batch(indices).asnumpy()
            images = []
            for i in range(len(indices)):
                img = Image.fromarray(frames[i])
                if self.transform:
                    img = self.transform(img)
                images.append(img)

            stack = torch.stack(images)
            # Pad if fewer frames than requested
            if stack.shape[0] < self.num_frames:
                diff = self.num_frames - stack.shape[0]
                stack = torch.cat([stack, stack[-1:].repeat(diff, 1, 1, 1)], dim=0)

            return stack, path, True

        except Exception:
            return torch.zeros(self.num_frames, 3, IMG_SIZE, IMG_SIZE), path, False


def load_classifier(checkpoint_path, num_classes, device):
    """Load a ConvNeXt-Small classifier from checkpoint."""
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=num_classes)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model", ckpt)

    # Strip DDP "module." prefix if present
    clean = {}
    for k, v in state_dict.items():
        clean[k.replace("module.", "")] = v
    model.load_state_dict(clean)

    model.to(device)
    model.eval()
    return model


def classify_batch(model, inputs, num_frames, num_classes, device):
    """Run multi-frame voting on a batch. Returns (predictions, confidences)."""
    B, F, C, H, W = inputs.shape
    flat = inputs.view(-1, C, H, W).to(device)

    with torch.no_grad(), autocast("cuda"):
        logits = model(flat)

    probs = F.softmax(logits, dim=1).view(B, F, num_classes).mean(dim=1)
    confs, preds = torch.max(probs, dim=1)
    return preds.cpu().numpy(), confs.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="Classify echo views and color")
    parser.add_argument("--input_dir", required=True, help="Directory of MP4 files (recursive)")
    parser.add_argument("--output_csv", required=True, help="Output CSV path")
    parser.add_argument("--view_checkpoint", default=None, help="View classifier checkpoint")
    parser.add_argument("--color_checkpoint", default=None, help="Color classifier checkpoint")
    parser.add_argument("--num_frames", type=int, default=5, help="Frames per video for voting")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    args = parser.parse_args()

    if args.view_checkpoint is None and args.color_checkpoint is None:
        raise ValueError("Provide at least one of --view_checkpoint or --color_checkpoint")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Find MP4s
    paths = []
    for root, _, files in os.walk(args.input_dir):
        for f in files:
            if f.lower().endswith(".mp4"):
                paths.append(os.path.join(root, f))
    paths.sort()
    print(f"Found {len(paths)} MP4 files.")

    if not paths:
        return

    # Load models
    view_model = None
    color_model = None
    if args.view_checkpoint:
        print(f"Loading view classifier: {args.view_checkpoint}")
        view_model = load_classifier(args.view_checkpoint, len(VIEW_LABELS), device)
    if args.color_checkpoint:
        print(f"Loading color classifier: {args.color_checkpoint}")
        color_model = load_classifier(args.color_checkpoint, len(COLOR_LABELS), device)

    # Transform
    transform = create_transform(
        IMG_SIZE, is_training=False, interpolation="bicubic",
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
    )

    dataset = VideoFrameDataset(paths, transform, args.num_frames)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=True, shuffle=False,
    )

    # Run inference
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)

    with open(args.output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["path"]
        if view_model:
            header += ["view", "view_confidence"]
        if color_model:
            header += ["color", "color_confidence"]
        writer.writerow(header)

        for inputs, batch_paths, valids in tqdm(loader, desc="Classifying"):
            valid_mask = valids.bool()
            n = len(batch_paths)

            view_preds = [None] * n
            view_confs = [0.0] * n
            color_preds = [None] * n
            color_confs = [0.0] * n

            if valid_mask.sum() > 0:
                valid_inputs = inputs[valid_mask]

                if view_model:
                    preds, confs = classify_batch(
                        view_model, valid_inputs, args.num_frames, len(VIEW_LABELS), device
                    )
                    vi = 0
                    for j in range(n):
                        if valids[j]:
                            view_preds[j] = VIEW_LABELS[preds[vi]]
                            view_confs[j] = float(confs[vi])
                            vi += 1

                if color_model:
                    preds, confs = classify_batch(
                        color_model, valid_inputs, args.num_frames, len(COLOR_LABELS), device
                    )
                    vi = 0
                    for j in range(n):
                        if valids[j]:
                            color_preds[j] = COLOR_LABELS[preds[vi]]
                            color_confs[j] = float(confs[vi])
                            vi += 1

            for j in range(n):
                row = [batch_paths[j]]
                if view_model:
                    row += [view_preds[j] or "ERROR", view_confs[j]]
                if color_model:
                    row += [color_preds[j] or "ERROR", color_confs[j]]
                writer.writerow(row)

    print(f"\nClassifications written to {args.output_csv}")


if __name__ == "__main__":
    main()
