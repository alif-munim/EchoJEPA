#!/usr/bin/env python
# evals/extract_embeddings.py
"""
Multi-GPU embedding extraction for UMAP visualization.

Usage:
    python -m evals.extract_embeddings \
        --config configs/inference/vitg-384/view/echojepa_224px.yaml \
        --data /path/to/test.csv \
        --output embeddings/echojepa_g_embeddings.npz \
        --devices cuda:0 cuda:1 cuda:2 cuda:3
"""

import os
import argparse
import yaml
import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from pathlib import Path
import tempfile

# Fix for "AF_UNIX path too long" error
short_tmp = "/tmp/extract_emb"
os.makedirs(short_tmp, exist_ok=True)
tempfile.tempdir = short_tmp
os.environ["TMPDIR"] = short_tmp

from evals.video_classification_frozen.models import init_module
from evals.video_classification_frozen.utils import make_transforms
from src.datasets.data_manager import init_data

DEFAULT_NORMALIZATION = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


def make_dataloader(
    root_path,
    batch_size,
    world_size=1,
    rank=0,
    dataset_type="VideoDataset",
    img_size=224,
    frames_per_clip=16,
    frame_step=4,
    num_segments=1,
    num_views_per_segment=1,
    num_workers=8,
    normalization=None,
):
    """Create dataloader with optional distributed sampling."""
    if normalization is None:
        normalization = DEFAULT_NORMALIZATION

    transform = make_transforms(
        training=False,
        num_views_per_clip=num_views_per_segment,
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(0.75, 4 / 3),
        random_resize_scale=(0.08, 1.0),
        reprob=0.25,
        auto_augment=True,
        motion_shift=False,
        crop_size=img_size,
        normalize=normalization,
    )

    data_loader, data_sampler = init_data(
        data=dataset_type,
        root_path=[root_path],
        transform=transform,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        clip_len=frames_per_clip,
        frame_sample_rate=frame_step,
        duration=None,
        num_clips=num_segments,
        allow_clip_overlap=True,
        num_workers=num_workers,
        drop_last=False,
    )
    return data_loader, data_sampler


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
    device = torch.device("cuda:0")  # After setting CUDA_VISIBLE_DEVICES, use cuda:0
    
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
    
    # Create distributed dataloader
    loader, sampler = make_dataloader(
        root_path=data_path,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        dataset_type="VideoDataset",
        img_size=data_cfg.get("resolution", 224),
        frames_per_clip=data_cfg.get("frames_per_clip", 16),
        frame_step=data_cfg.get("frame_step", 2),
        num_segments=num_segments,
        num_views_per_segment=1,
        num_workers=num_workers,
        normalization=data_cfg.get("normalization"),
    )
    
    logger.info(f"[Rank {rank}] Processing {len(loader)} batches")
    
    all_embeddings = []
    all_labels = []
    all_paths = []
    all_indices = []  # Track original indices for proper ordering
    
    # Progress bar only on rank 0
    iterator = tqdm(loader, desc=f"GPU {device_id}", disable=(rank != 0))
    
    with torch.no_grad():
        for batch_idx, data in enumerate(iterator):
            # Load data
            clips = [[dij.to(device) for dij in di] for di in data[0]]
            clip_indices = [d.to(device) for d in data[2]]
            labels = data[1]
            batch_size_actual = len(labels)
            
            # Get paths if available
            if len(data) > 3:
                paths = data[3]
            else:
                paths = [f"sample_{batch_idx * batch_size + i}" for i in range(batch_size_actual)]
            
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
            all_labels.append(labels.numpy())
            all_paths.extend(paths)
            
            # Compute global indices for this batch
            # With DistributedSampler, samples are interleaved: rank 0 gets 0, n, 2n, ...; rank 1 gets 1, n+1, 2n+1, ...
            for i in range(batch_size_actual):
                global_idx = batch_idx * batch_size * world_size + rank + i * world_size
                all_indices.append(global_idx)
    
    # Concatenate
    embeddings = np.concatenate(all_embeddings, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    paths = np.array(all_paths)
    indices = np.array(all_indices)
    
    logger.info(f"[Rank {rank}] Extracted {len(embeddings)} embeddings")
    
    # Save partial results
    partial_path = os.path.join(output_dir, f"partial_rank{rank}.npz")
    np.savez(partial_path, embeddings=embeddings, labels=labels, paths=paths, indices=indices)
    logger.info(f"[Rank {rank}] Saved partial results to {partial_path}")


def merge_results(output_dir: str, output_path: str, world_size: int):
    """Merge partial results from all workers."""
    print(f"\nMerging results from {world_size} workers...")
    
    all_embeddings = []
    all_labels = []
    all_paths = []
    all_indices = []
    
    for rank in range(world_size):
        partial_path = os.path.join(output_dir, f"partial_rank{rank}.npz")
        data = np.load(partial_path, allow_pickle=True)
        all_embeddings.append(data['embeddings'])
        all_labels.append(data['labels'])
        all_paths.append(data['paths'])
        all_indices.append(data['indices'])
        
        # Clean up partial file
        os.remove(partial_path)
    
    # Concatenate
    embeddings = np.concatenate(all_embeddings, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    paths = np.concatenate(all_paths, axis=0)
    indices = np.concatenate(all_indices, axis=0)
    
    # Sort by original index to restore order
    sort_order = np.argsort(indices)
    embeddings = embeddings[sort_order]
    labels = labels[sort_order]
    paths = paths[sort_order]
    
    # Save final results
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, embeddings=embeddings, labels=labels, paths=paths)
    
    print(f"Saved to: {output_path}")
    print(f"  Embeddings shape: {embeddings.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Unique labels: {np.unique(labels)}")


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
    parser = argparse.ArgumentParser(description="Multi-GPU embedding extraction")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--data", required=True, help="Path to CSV data file")
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
    print(f"Multi-GPU Embedding Extraction")
    print(f"=" * 60)
    print(f"Config: {args.config}")
    print(f"Data: {args.data}")
    print(f"Output: {args.output}")
    print(f"Devices: {args.devices}")
    print(f"Batch size per GPU: {args.batch_size}")
    print(f"Total batch size: {args.batch_size * world_size}")
    print(f"=" * 60)
    
    if world_size == 1:
        # Single GPU - run directly
        process_main(0, 1, args.devices, args, config)
    else:
        # Multi-GPU - spawn processes
        mp.set_start_method("spawn", force=True)
        processes = []
        
        for rank in range(world_size):
            p = mp.Process(
                target=process_main,
                args=(rank, world_size, args.devices, args, config),
            )
            p.start()
            processes.append(p)
        
        # Wait for all processes
        for p in processes:
            p.join()
    
    # Merge results
    merge_results(output_dir, args.output, world_size)
    
    print("\nDone!")


if __name__ == "__main__":
    main()