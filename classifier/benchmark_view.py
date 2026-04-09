"""Quick benchmark of view classifiers on UHN test split."""
import argparse
import os
import random
import tempfile

import boto3
import decord
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from timm.data import create_transform
from torch.amp import autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

LABEL_MAP = {
    0: "A2C", 1: "A3C", 2: "A4C", 3: "A5C", 4: "Exclude", 5: "PLAX",
    6: "PSAX-AP", 7: "PSAX-AV", 8: "PSAX-MV", 9: "PSAX-PM", 10: "SSN",
    11: "Subcostal", 12: "TEE",
}


class EchoS3Dataset(Dataset):
    def __init__(self, csv_path, transform, num_frames_voting=3):
        self.df = pd.read_csv(
            csv_path, header=None, names=["s3_uri", "label"], sep=r"\s+", engine="python"
        )
        self.df["label"] = self.df["label"].astype(int)
        self.data = self.df.to_records(index=False)
        self.transform = transform
        self.num_frames_voting = num_frames_voting

    def __len__(self):
        return len(self.data)

    def _get_s3_client(self):
        if not hasattr(self, "s3_client"):
            self.s3_client = boto3.client("s3")
        return self.s3_client

    def _download_video(self, s3_uri):
        clean_uri = str(s3_uri).strip().replace("s3://", "")
        bucket, key = clean_uri.split("/", 1)
        fd, temp_path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        self._get_s3_client().download_file(bucket, key, temp_path)
        return temp_path

    def __getitem__(self, idx):
        max_retries = 5
        for attempt in range(max_retries):
            uri, label = self.data[idx]
            temp_path = None
            try:
                temp_path = self._download_video(uri)
                vr = decord.VideoReader(temp_path, num_threads=1)
                total_frames = len(vr)
                if total_frames < 1:
                    raise ValueError("Empty video")
                if total_frames < self.num_frames_voting:
                    indices = np.arange(total_frames).astype(int)
                else:
                    indices = np.linspace(0, total_frames - 1, self.num_frames_voting).astype(int)
                video_data = vr.get_batch(indices).asnumpy()
                images = []
                for i in range(len(indices)):
                    img = Image.fromarray(video_data[i])
                    if self.transform:
                        img = self.transform(img)
                    images.append(img)
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                img_stack = torch.stack(images)
                current_count = img_stack.shape[0]
                if current_count < self.num_frames_voting:
                    diff = self.num_frames_voting - current_count
                    padding = img_stack[-1:].repeat(diff, 1, 1, 1)
                    img_stack = torch.cat([img_stack, padding], dim=0)
                return img_stack, label
            except Exception:
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)
                idx = random.randint(0, len(self.data) - 1)
        raise RuntimeError(f"Failed after {max_retries} retries")


def load_model(checkpoint_path, num_classes, img_size, device):
    model = timm.create_model("convnext_small.fb_in1k", pretrained=False, num_classes=num_classes)
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    # Extract model weights from training checkpoint if needed
    if isinstance(raw, dict) and "model" in raw:
        state = raw["model"]
    else:
        state = raw
    # Handle DDP 'module.' prefix
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model = model.to(device, memory_format=torch.channels_last)
    model.eval()
    return model


def evaluate(model, loader, device, num_classes):
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for inp, target in tqdm(loader, desc="Evaluating"):
            B, N, C, H, W = inp.shape
            inp = inp.view(B * N, C, H, W).to(device, memory_format=torch.channels_last)
            target = target.to(device)
            with autocast("cuda"):
                logits = model(inp)
            probs = F.softmax(logits, dim=1).view(B, N, num_classes).mean(dim=1)
            all_probs.append(probs.cpu())
            all_labels.append(target.cpu())
    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()
    return all_probs, all_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True, help="Checkpoint paths to benchmark")
    parser.add_argument("--names", nargs="+", required=True, help="Display names for each checkpoint")
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--img_size", type=int, default=336)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--num_frames", type=int, default=3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(LABEL_MAP)

    val_transform = create_transform(
        input_size=args.img_size, is_training=False,
        interpolation="bicubic", crop_pct=1.0,
    )

    print(f"Loading test set: {args.test_csv}")
    dataset = EchoS3Dataset(args.test_csv, val_transform, num_frames_voting=args.num_frames)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    print(f"Test samples: {len(dataset)}")

    for ckpt_path, name in zip(args.checkpoints, args.names):
        print(f"\n{'='*60}")
        print(f"Model: {name}")
        print(f"Checkpoint: {ckpt_path}")
        print(f"{'='*60}")

        model = load_model(ckpt_path, num_classes, args.img_size, device)
        probs, labels = evaluate(model, loader, device, num_classes)
        preds = probs.argmax(axis=1)

        # Metrics
        acc = (preds == labels).mean()
        f1_macro = f1_score(labels, preds, average="macro")
        f1_weighted = f1_score(labels, preds, average="weighted")

        # AUROC (one-vs-rest)
        labels_bin = label_binarize(labels, classes=list(range(num_classes)))
        try:
            auroc_macro = roc_auc_score(labels_bin, probs, average="macro", multi_class="ovr")
            auroc_weighted = roc_auc_score(labels_bin, probs, average="weighted", multi_class="ovr")
        except ValueError:
            auroc_macro = auroc_weighted = float("nan")

        print(f"\nAccuracy:        {acc:.4f}")
        print(f"F1 (macro):      {f1_macro:.4f}")
        print(f"F1 (weighted):   {f1_weighted:.4f}")
        print(f"AUROC (macro):   {auroc_macro:.4f}")
        print(f"AUROC (weighted):{auroc_weighted:.4f}")

        class_names = [LABEL_MAP[i] for i in range(num_classes)]
        print(f"\n{classification_report(labels, preds, target_names=class_names, digits=3)}")


if __name__ == "__main__":
    main()
