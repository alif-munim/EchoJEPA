"""
Unified distributed inference on the 18M echocardiogram dataset.

Supports multiple classification tasks (view, color, quality, zoom) via --task flag
or custom label maps via --mapping_json.

Usage:
    torchrun --nproc_per_node=8 inference_18m.py \
        --task view \
        --input_csv ../indices/master_index_18M_cleaned.csv \
        --output_dir ./output/view_inference_18m \
        --num_frames 5 --batch_size 128

    torchrun --nproc_per_node=8 inference_18m.py \
        --task color \
        --input_csv ../indices/master_index_18M_cleaned.csv \
        --output_dir ./output/color_inference_18m \
        --num_frames 5 --batch_size 128
"""

import argparse
import json
import logging
import os
import tempfile
import time

import boto3
import decord
import numpy as np
import pandas as pd
import timm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from timm.data import create_transform
from torch.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm

# -----------------------------------------------------------------------------
# BUILT-IN LABEL MAPS
# -----------------------------------------------------------------------------
BUILTIN_LABEL_MAPS = {
    "view": {
        0: "A2C", 1: "A3C", 2: "A4C", 3: "A5C", 4: "Exclude", 5: "PLAX",
        6: "PSAX-AP", 7: "PSAX-AV", 8: "PSAX-MV", 9: "PSAX-PM", 10: "SSN",
        11: "Subcostal", 12: "TEE",
    },
    "color": {0: "No", 1: "Yes"},
    "quality": {0: "discard", 1: "keep"},
    "zoom": {0: "Full", 1: "Large", 2: "Small"},
}

DEFAULT_CHECKPOINTS = {
    "view": "checkpoints/view_convnext_small_336px.pt",
    "color": "checkpoints/color_convnext_small_336px.pt",
}

IMG_SIZE = 336


def setup_logger(rank):
    logger = logging.getLogger(f"inference_rank{rank}")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(f"[Rank {rank}] %(asctime)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


# -----------------------------------------------------------------------------
# INFERENCE DATASET
# -----------------------------------------------------------------------------
class InferenceDataset(Dataset):
    def __init__(self, df, transform=None, num_frames=3, img_size=336):
        self.data = df.to_records(index=False)
        self.transform = transform
        self.num_frames = num_frames
        self.img_size = img_size

    def __len__(self):
        return len(self.data)

    def _get_s3_client(self):
        if not hasattr(self, "s3_client"):
            session = boto3.session.Session()
            self.s3_client = session.client("s3")
        return self.s3_client

    def _download_video(self, s3_uri):
        try:
            clean_uri = str(s3_uri).strip()
            if clean_uri.startswith("s3://"):
                clean_uri = clean_uri.replace("s3://", "")

            bucket, key = clean_uri.split("/", 1)
            fd, temp_path = tempfile.mkstemp(suffix=".mp4")
            os.close(fd)

            self._get_s3_client().download_file(bucket, key, temp_path)
            return temp_path
        except Exception:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return None

    def __getitem__(self, idx):
        row = self.data[idx]
        uri = row[0]

        temp_path = self._download_video(uri)

        if temp_path is None:
            return torch.zeros((self.num_frames, 3, self.img_size, self.img_size)), uri, False

        try:
            vr = decord.VideoReader(temp_path, num_threads=1)
            total_frames = len(vr)

            if total_frames < 1:
                raise ValueError("Empty video")

            if total_frames < self.num_frames:
                indices = np.arange(total_frames).astype(int)
            else:
                indices = np.linspace(0, total_frames - 1, self.num_frames).astype(int)

            images = []
            video_data = vr.get_batch(indices).asnumpy()

            for i in range(len(indices)):
                img = Image.fromarray(video_data[i])
                if self.transform:
                    img = self.transform(img)
                images.append(img)

            os.remove(temp_path)

            img_stack = torch.stack(images)
            if img_stack.shape[0] < self.num_frames:
                diff = self.num_frames - img_stack.shape[0]
                last_frame = img_stack[-1:]
                padding = last_frame.repeat(diff, 1, 1, 1)
                img_stack = torch.cat([img_stack, padding], dim=0)

            return img_stack, uri, True

        except Exception:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return torch.zeros((self.num_frames, 3, self.img_size, self.img_size)), uri, False


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Distributed inference on 18M echo dataset")
    parser.add_argument("--task", type=str, default="view", choices=["view", "color", "quality", "zoom", "custom"],
                        help="Classification task (determines label map and defaults)")
    parser.add_argument("--mapping_json", type=str, default=None,
                        help="Path to custom JSON label map (required if --task=custom)")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to CSV with s3_uri column")
    parser.add_argument("--output_dir", type=str, default=None, help="Dir to save results (default: ./output/<task>_inference_18m)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--model", type=str, default="convnext_small.fb_in1k", help="timm model name")
    parser.add_argument("--img_size", type=int, default=336, help="Input image size")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N rows (for testing)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size per GPU")
    parser.add_argument("--num_frames", type=int, default=3, help="Number of frames to sample per video")
    parser.add_argument("--local_rank", type=int, default=int(os.getenv("LOCAL_RANK", 0)))
    args = parser.parse_args()

    # Resolve label map
    if args.task == "custom":
        if args.mapping_json is None:
            raise ValueError("--mapping_json required when --task=custom")
        with open(args.mapping_json) as f:
            label_map = {int(k): v for k, v in json.load(f).items()}
    else:
        label_map = BUILTIN_LABEL_MAPS[args.task]

    num_classes = len(label_map)

    # Resolve output dir
    if args.output_dir is None:
        args.output_dir = f"./output/{args.task}_inference_18m"

    # Resolve checkpoint
    if args.checkpoint is None:
        args.checkpoint = DEFAULT_CHECKPOINTS.get(args.task)
        if args.checkpoint is None:
            raise ValueError(f"No default checkpoint for task '{args.task}'. Provide --checkpoint.")

    # Prediction column name
    pred_col = "prediction" if args.task == "view" else f"is_{args.task}"

    # DDP Setup
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    logger = setup_logger(rank)

    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        logger.info(f"Task: {args.task} | Classes: {num_classes}")
        logger.info(f"Starting inference on {world_size} GPUs.")
        logger.info(f"Checkpoint: {args.checkpoint}")
        logger.info(f"Frames per video: {args.num_frames}")

    # Load Model
    model = timm.create_model(args.model, pretrained=False, num_classes=num_classes)

    ckpt = torch.load(args.checkpoint, map_location=f"cuda:{args.local_rank}")
    state_dict = ckpt["model"] if "model" in ckpt else ckpt

    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("module.", "")
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

    model.to(device)
    model = model.to(memory_format=torch.channels_last)
    model = DDP(model, device_ids=[args.local_rank])
    model.eval()

    # Prepare Data
    if rank == 0:
        logger.info("Loading CSV...")

    df = pd.read_csv(args.input_csv, usecols=[0], names=["s3_uri"], header=0)

    if args.limit:
        df = df.head(args.limit)

    if rank == 0:
        logger.info(f"Total rows to process: {len(df)}")

    val_tf = create_transform(
        args.img_size, is_training=False, interpolation="bicubic",
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
    )

    dataset = InferenceDataset(df, transform=val_tf, num_frames=args.num_frames, img_size=args.img_size)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=sampler,
        num_workers=8, pin_memory=True, drop_last=False,
    )

    # Inference Loop
    results = []
    output_file = os.path.join(args.output_dir, f"predictions_rank{rank}.csv")
    buffer_size = 1000

    if rank == 0:
        iterator = tqdm(loader, desc=f"Inference ({args.task})", total=len(loader), unit="batch")
    else:
        iterator = loader

    start_time = time.time()

    with torch.no_grad():
        for i, (inputs, uris, valids) in enumerate(iterator):
            valid_mask = valids.bool()
            batch_results = []

            if valid_mask.sum() > 0:
                valid_inputs = inputs[valid_mask]
                B_valid, Frames, C, H, W = valid_inputs.shape
                valid_inputs = valid_inputs.view(-1, C, H, W).to(device, memory_format=torch.channels_last)

                with autocast("cuda"):
                    logits = model(valid_inputs)

                probs = F.softmax(logits, dim=1).view(B_valid, Frames, num_classes).mean(dim=1)
                confs, preds = torch.max(probs, dim=1)

                preds = preds.cpu().numpy()
                confs = confs.cpu().numpy()

                valid_idx_ptr = 0
                for j in range(len(uris)):
                    uri = uris[j]
                    if valids[j]:
                        pred_idx = preds[valid_idx_ptr]
                        pred_label = label_map[pred_idx]
                        confidence = confs[valid_idx_ptr]

                        batch_results.append({
                            "s3_uri": uri,
                            pred_col: pred_label,
                            "confidence": float(confidence),
                            "status": "OK",
                        })
                        valid_idx_ptr += 1
                    else:
                        batch_results.append({
                            "s3_uri": uri,
                            pred_col: None,
                            "confidence": 0.0,
                            "status": "ERROR_DOWNLOAD",
                        })
            else:
                for uri in uris:
                    batch_results.append({
                        "s3_uri": uri,
                        pred_col: None,
                        "confidence": 0.0,
                        "status": "ERROR_DOWNLOAD",
                    })

            results.extend(batch_results)

            # Incremental write
            if len(results) >= buffer_size:
                df_res = pd.DataFrame(results)
                header = not os.path.exists(output_file)
                df_res.to_csv(output_file, mode="a", header=header, index=False)
                results = []

            if rank == 0 and i % 5 == 0:
                elapsed = time.time() - start_time
                if elapsed > 0:
                    rate = ((i + 1) * args.batch_size * world_size) / elapsed
                    iterator.set_postfix({"Global Speed": f"{rate:.1f} vid/s"})

    # Write remaining
    if results:
        df_res = pd.DataFrame(results)
        header = not os.path.exists(output_file)
        df_res.to_csv(output_file, mode="a", header=header, index=False)

    logger.info("Rank complete.")
    dist.barrier()
    if rank == 0:
        logger.info(f"All ranks finished. Results saved to {args.output_dir}/predictions_rank*.csv")


if __name__ == "__main__":
    main()
