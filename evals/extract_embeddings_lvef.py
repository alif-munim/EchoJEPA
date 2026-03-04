#!/usr/bin/env python
# evals/extract_embeddings_lvef.py
"""
Multi-GPU embedding extraction for LVEF UMAP visualization.
Handles continuous LVEF labels instead of discrete class labels.

Usage:
    python -m evals.extract_embeddings_lvef \
        --config configs/inference/vitg-384/view/echojepa_224px.yaml \
        --data data/csv/a4c_b_lvef_test_224px_RAW.csv \
        --output embeddings/lvef/echojepa_g_lvef_embeddings.npz \
        --devices cuda:0 cuda:1 cuda:2 cuda:3
"""

import os
import argparse
import yaml
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from tqdm import tqdm
from pathlib import Path
import tempfile

# Fix for "AF_UNIX path too long" error
short_tmp = "/tmp/extract_emb_lvef"
os.makedirs(short_tmp, exist_ok=True)
tempfile.tempdir = short_tmp
os.environ["TMPDIR"] = short_tmp

from evals.video_classification_frozen.models import init_module
from evals.video_classification_frozen.utils import make_transforms

DEFAULT_NORMALIZATION = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


class LVEFVideoDataset(Dataset):
    """
    Dataset for LVEF videos with continuous labels.
    Expects CSV format: <video_path> <lvef_value>
    """
    
    def __init__(
        self,
        csv_path: str,
        transform=None,
        frames_per_clip: int = 16,
        frame_step: int = 4,
        num_segments: int = 1,
    ):
        self.transform = transform
        self.frames_per_clip = frames_per_clip
        self.frame_step = frame_step
        self.num_segments = num_segments
        
        # Parse CSV (space-separated: path lvef_value)
        self.samples = []
        with open(csv_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.rsplit(' ', 1)  # Split from right to handle paths with spaces
                if len(parts) == 2:
                    path, lvef = parts
                    try:
                        lvef = float(lvef)
                        self.samples.append((path, lvef))
                    except ValueError:
                        print(f"Warning: Could not parse LVEF value: {lvef}")
                        continue
        
        print(f"Loaded {len(self.samples)} samples from {csv_path}")
        lvef_values = [s[1] for s in self.samples]
        print(f"LVEF range: [{min(lvef_values):.1f}, {max(lvef_values):.1f}]")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, lvef = self.samples[idx]
        
        # Load video frames
        # This uses the same video loading logic as your existing codebase
        # You may need to adapt this based on your actual video loading implementation
        try:
            from src.datasets.video_dataset import load_video_frames
            frames = load_video_frames(
                path,
                num_frames=self.frames_per_clip * self.num_segments,
                frame_step=self.frame_step,
            )
        except ImportError:
            # Fallback: use decord or similar
            frames = self._load_video_decord(path)
        
        # Apply transforms
        if self.transform is not None:
            frames = self.transform(frames)
        
        # Return format matching your existing dataloader
        # clips: list of segments, each segment is list of views
        # For single segment, single view: [[frames]]
        clips = [[frames]]
        clip_indices = [torch.arange(self.frames_per_clip)]
        
        return clips, torch.tensor(lvef, dtype=torch.float32), clip_indices, path
    
    def _load_video_decord(self, path):
        """Fallback video loading using decord."""
        import decord
        decord.bridge.set_bridge('torch')
        
        vr = decord.VideoReader(path)
        total_frames = len(vr)
        
        # Sample frames
        num_frames_needed = self.frames_per_clip * self.frame_step
        if total_frames >= num_frames_needed:
            start = (total_frames - num_frames_needed) // 2
            indices = list(range(start, start + num_frames_needed, self.frame_step))
        else:
            # Repeat last frame if video is too short
            indices = list(range(0, total_frames, max(1, total_frames // self.frames_per_clip)))
            while len(indices) < self.frames_per_clip:
                indices.append(indices[-1])
            indices = indices[:self.frames_per_clip]
        
        frames = vr.get_batch(indices)  # [T, H, W, C]
        frames = frames.permute(0, 3, 1, 2)  # [T, C, H, W]
        
        return frames


def make_lvef_dataloader(
    csv_path: str,
    batch_size: int,
    world_size: int = 1,
    rank: int = 0,
    img_size: int = 224,
    frames_per_clip: int = 16,
    frame_step: int = 4,
    num_segments: int = 1,
    num_workers: int = 8,
    normalization=None,
):
    """Create dataloader for LVEF dataset with optional distributed sampling."""
    if normalization is None:
        normalization = DEFAULT_NORMALIZATION
    
    transform = make_transforms(
        training=False,
        num_views_per_clip=1,
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(0.75, 4 / 3),
        random_resize_scale=(0.08, 1.0),
        reprob=0.25,
        auto_augment=True,
        motion_shift=False,
        crop_size=img_size,
        normalize=normalization,
    )
    
    dataset = LVEFVideoDataset(
        csv_path=csv_path,
        transform=transform,
        frames_per_clip=frames_per_clip,
        frame_step=frame_step,
        num_segments=num_segments,
    )
    
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
    else:
        sampler = None
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    
    return loader, sampler


def extract_worker(
    rank: int,
    world_size: int,
    device: str,
    config: dict,
    data_path: str,
    output_dir: str,
    batch_size: int,
    num_segments: int,
    num_workers: int,
):
    """Worker function for each GPU."""
    import logging
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO if rank == 0 else logging.WARNING)
    
    # Set device
    device_id = device.split(":")[-1]
    os.environ["CUDA_VISIBLE_DEVICES"] = device_id
    device = torch.device("cuda:0")
    
    logger.info(f"[Rank {rank}/{world_size}] Using device {device} (GPU {device_id})")
    
    # Extract config
    model_kwargs = config["model_kwargs"]
    data_cfg = config["experiment"]["data"]
    
    # Initialize encoder
    encoder = init_module(
        module_name=model_kwargs["module_name"],
        frames_per_clip=data_cfg.get("frames_per_clip", 16),
        resolution=data_cfg.get("resolution", 224),
        checkpoint=model_kwargs.get("checkpoint"),
        model_kwargs=model_kwargs.get("pretrain_kwargs", {}),
        wrapper_kwargs=model_kwargs.get("wrapper_kwargs", {}),
        device=device,
    )
    encoder.eval()
    
    if rank == 0:
        logger.info(f"Encoder embed_dim: {encoder.embed_dim}")
    
    # Create dataloader
    loader, sampler = make_lvef_dataloader(
        csv_path=data_path,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        img_size=data_cfg.get("resolution", 224),
        frames_per_clip=data_cfg.get("frames_per_clip", 16),
        frame_step=data_cfg.get("frame_step", 2),
        num_segments=num_segments,
        num_workers=num_workers,
        normalization=data_cfg.get("normalization"),
    )
    
    logger.info(f"[Rank {rank}] Processing {len(loader)} batches")
    
    all_embeddings = []
    all_lvef = []
    all_paths = []
    all_indices = []
    
    iterator = tqdm(loader, desc=f"GPU {device_id}", disable=(rank != 0))
    
    with torch.no_grad():
        for batch_idx, data in enumerate(iterator):
            clips, lvef_values, clip_indices, paths = data
            
            # Move to device
            clips = [[dij.to(device) for dij in di] for di in clips]
            clip_indices = [d.to(device) for d in clip_indices]
            batch_size_actual = len(lvef_values)
            
            # Forward through encoder
            outputs = encoder(clips, clip_indices)
            
            # Pool over tokens: [B, N_tokens, D] -> [B, D]
            pooled_segments = [o.mean(dim=1) for o in outputs]
            
            # Pool over segments
            if len(pooled_segments) > 1:
                pooled = torch.stack(pooled_segments, dim=1).mean(dim=1)
            else:
                pooled = pooled_segments[0]
            
            # Store
            all_embeddings.append(pooled.cpu().numpy())
            all_lvef.append(lvef_values.numpy())
            all_paths.extend(paths)
            
            # Compute global indices
            for i in range(batch_size_actual):
                global_idx = batch_idx * batch_size * world_size + rank + i * world_size
                all_indices.append(global_idx)
    
    # Concatenate
    embeddings = np.concatenate(all_embeddings, axis=0)
    lvef = np.concatenate(all_lvef, axis=0)
    paths = np.array(all_paths)
    indices = np.array(all_indices)
    
    logger.info(f"[Rank {rank}] Extracted {len(embeddings)} embeddings")
    logger.info(f"[Rank {rank}] LVEF range: [{lvef.min():.1f}, {lvef.max():.1f}]")
    
    # Save partial results
    partial_path = os.path.join(output_dir, f"partial_rank{rank}.npz")
    np.savez(partial_path, embeddings=embeddings, lvef=lvef, paths=paths, indices=indices)
    logger.info(f"[Rank {rank}] Saved partial results to {partial_path}")


def merge_results(output_dir: str, output_path: str, world_size: int):
    """Merge partial results from all workers."""
    print(f"\nMerging results from {world_size} workers...")
    
    all_embeddings = []
    all_lvef = []
    all_paths = []
    all_indices = []
    
    for rank in range(world_size):
        partial_path = os.path.join(output_dir, f"partial_rank{rank}.npz")
        data = np.load(partial_path, allow_pickle=True)
        all_embeddings.append(data['embeddings'])
        all_lvef.append(data['lvef'])
        all_paths.append(data['paths'])
        all_indices.append(data['indices'])
        
        os.remove(partial_path)
    
    # Concatenate
    embeddings = np.concatenate(all_embeddings, axis=0)
    lvef = np.concatenate(all_lvef, axis=0)
    paths = np.concatenate(all_paths, axis=0)
    indices = np.concatenate(all_indices, axis=0)
    
    # Sort by original index
    sort_order = np.argsort(indices)
    embeddings = embeddings[sort_order]
    lvef = lvef[sort_order]
    paths = paths[sort_order]
    
    # Save final results (note: 'lvef' instead of 'labels')
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, embeddings=embeddings, lvef=lvef, paths=paths)
    
    print(f"Saved to: {output_path}")
    print(f"  Embeddings shape: {embeddings.shape}")
    print(f"  LVEF shape: {lvef.shape}")
    print(f"  LVEF range: [{lvef.min():.1f}, {lvef.max():.1f}]")
    print(f"  LVEF mean: {lvef.mean():.1f} ± {lvef.std():.1f}")


def process_main(rank, world_size, devices, args, config):
    """Main function for each process."""
    device = devices[rank]
    
    extract_worker(
        rank=rank,
        world_size=world_size,
        device=device,
        config=config,
        data_path=args.data,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_segments=args.num_segments,
        num_workers=args.num_workers,
    )


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU LVEF embedding extraction")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--data", required=True, help="Path to LVEF CSV file")
    parser.add_argument("--output", required=True, help="Output .npz path")
    parser.add_argument(
        "--devices",
        type=str,
        nargs="+",
        default=["cuda:0"],
        help="Devices to use (e.g., cuda:0 cuda:1 cuda:2 cuda:3)",
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per GPU")
    parser.add_argument("--num_segments", type=int, default=1, help="Clips per video")
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader workers per GPU")
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Setup
    world_size = len(args.devices)
    output_dir = os.path.dirname(args.output) or "."
    os.makedirs(output_dir, exist_ok=True)
    args.output_dir = output_dir
    
    print(f"=" * 60)
    print(f"Multi-GPU LVEF Embedding Extraction")
    print(f"=" * 60)
    print(f"Config: {args.config}")
    print(f"Data: {args.data}")
    print(f"Output: {args.output}")
    print(f"Devices: {args.devices}")
    print(f"Batch size per GPU: {args.batch_size}")
    print(f"Total batch size: {args.batch_size * world_size}")
    print(f"=" * 60)
    
    if world_size == 1:
        process_main(0, 1, args.devices, args, config)
    else:
        mp.set_start_method("spawn", force=True)
        processes = []
        
        for rank in range(world_size):
            p = mp.Process(
                target=process_main,
                args=(rank, world_size, args.devices, args, config),
            )
            p.start()
            processes.append(p)
        
        for p in processes:
            p.join()
    
    merge_results(output_dir, args.output, world_size)
    
    print("\nDone!")


if __name__ == "__main__":
    main()