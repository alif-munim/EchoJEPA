#!/usr/bin/env python
"""
Chunked multi-GPU embedding extraction for UHN 18M dataset.

Optimized for large-scale S3-based extraction with:
- bf16 autocast for ~2x GPU throughput
- Larger batch sizes (ViT-G fits batch 32 per A100-80GB in bf16)
- Chunked saving every N batches (crash-safe resume)
- Study-level pooling built-in (no separate P5 step)

Usage:
    python -m evals.extract_uhn_embeddings \
        --config configs/inference/vitg-384/extract_uhn.yaml \
        --data experiments/nature_medicine/uhn/uhn_all_clips.csv \
        --clip_index experiments/nature_medicine/uhn/uhn_clip_index.npz \
        --output_dir experiments/nature_medicine/uhn/echojepa_g_embeddings \
        --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \
        --batch_size 32 \
        --save_every 500
"""

import argparse
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
import yaml
from tqdm import tqdm

# Fix for "AF_UNIX path too long" error
import tempfile

short_tmp = "/tmp/extract_uhn"
os.makedirs(short_tmp, exist_ok=True)
tempfile.tempdir = short_tmp
os.environ["TMPDIR"] = short_tmp

from evals.video_classification_frozen.models import init_module
from evals.video_classification_frozen.utils import make_transforms
from src.datasets.data_manager import init_data


DEFAULT_NORMALIZATION = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def extract_worker(
    rank: int,
    world_size: int,
    device_str: str,
    config: dict,
    data_path: str,
    output_dir: str,
    batch_size: int,
    save_every: int,
    num_workers: int,
):
    """Worker function for each GPU with chunked saving."""
    device_id = device_str.split(":")[-1]
    os.environ["CUDA_VISIBLE_DEVICES"] = device_id
    device = torch.device("cuda:0")

    log = logging.getLogger(f"rank{rank}")
    log.setLevel(logging.INFO if rank == 0 else logging.WARNING)

    log.info(f"[Rank {rank}/{world_size}] GPU {device_id}")

    # Load model
    model_kwargs = config["model_kwargs"]
    data_cfg = config["experiment"]["data"]

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
    encoder = encoder.to(dtype=torch.bfloat16)  # Cast model to bf16

    if rank == 0:
        log.info(f"Encoder embed_dim: {encoder.embed_dim}, dtype: bf16")

    # Create dataloader
    transform = make_transforms(
        training=False,
        num_views_per_clip=1,
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(0.75, 4 / 3),
        random_resize_scale=(0.08, 1.0),
        reprob=0.25,
        auto_augment=True,
        motion_shift=False,
        crop_size=data_cfg.get("resolution", 224),
        normalize=data_cfg.get("normalization", DEFAULT_NORMALIZATION),
    )

    data_loader, _ = init_data(
        data="VideoDataset",
        root_path=[data_path],
        transform=transform,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        clip_len=data_cfg.get("frames_per_clip", 16),
        frame_sample_rate=data_cfg.get("frame_step", 2),
        duration=None,
        num_clips=1,
        allow_clip_overlap=True,
        num_workers=num_workers,
        drop_last=False,
    )

    total_batches = len(data_loader)
    log.info(f"[Rank {rank}] {total_batches} batches, save every {save_every}")

    # Check for resume
    chunk_dir = os.path.join(output_dir, f"chunks_rank{rank}")
    os.makedirs(chunk_dir, exist_ok=True)

    existing_chunks = sorted(
        [f for f in os.listdir(chunk_dir) if f.startswith("chunk_") and f.endswith(".npz")]
    )
    start_batch = 0
    if existing_chunks:
        last_chunk = existing_chunks[-1]
        # chunk_00000500.npz means batches 0-499 are done
        start_batch = int(last_chunk.replace("chunk_", "").replace(".npz", ""))
        log.info(f"[Rank {rank}] Resuming from batch {start_batch} ({len(existing_chunks)} chunks)")

    # Extract
    chunk_embeddings = []
    chunk_indices = []
    batches_in_chunk = 0

    iterator = tqdm(data_loader, desc=f"GPU {device_id}", disable=(rank != 0))

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for batch_idx, data in enumerate(iterator):
            if batch_idx < start_batch:
                continue

            clips = [[dij.to(device) for dij in di] for di in data[0]]
            clip_indices_batch = [d.to(device) for d in data[2]]
            batch_size_actual = len(data[1])

            # Forward
            outputs = encoder(clips, clip_indices_batch)
            pooled_segments = [o.mean(dim=1) for o in outputs]
            if len(pooled_segments) > 1:
                pooled = torch.stack(pooled_segments, dim=1).mean(dim=1)
            else:
                pooled = pooled_segments[0]

            chunk_embeddings.append(pooled.float().cpu().numpy())

            # Track global indices for ordering
            for i in range(batch_size_actual):
                global_idx = batch_idx * batch_size * world_size + rank + i * world_size
                chunk_indices.append(global_idx)

            batches_in_chunk += 1

            # Save chunk
            if batches_in_chunk >= save_every or batch_idx == total_batches - 1:
                emb = np.concatenate(chunk_embeddings, axis=0)
                idx = np.array(chunk_indices)
                chunk_path = os.path.join(chunk_dir, f"chunk_{batch_idx + 1:08d}.npz")
                np.savez_compressed(chunk_path, embeddings=emb, indices=idx)

                if rank == 0:
                    log.info(
                        f"Saved chunk {chunk_path} ({emb.shape[0]} clips, "
                        f"batch {batch_idx + 1}/{total_batches})"
                    )

                chunk_embeddings = []
                chunk_indices = []
                batches_in_chunk = 0

    log.info(f"[Rank {rank}] Done. Chunks saved to {chunk_dir}/")


def merge_and_pool(output_dir: str, clip_index_path: str, world_size: int):
    """Merge all chunk files from all ranks, then pool to study level."""
    logger.info("Merging chunks from all ranks...")

    all_embeddings = []
    all_indices = []

    for rank in range(world_size):
        chunk_dir = os.path.join(output_dir, f"chunks_rank{rank}")
        chunks = sorted(
            [f for f in os.listdir(chunk_dir) if f.startswith("chunk_") and f.endswith(".npz")]
        )
        for chunk_file in chunks:
            d = np.load(os.path.join(chunk_dir, chunk_file))
            all_embeddings.append(d["embeddings"])
            all_indices.append(d["indices"])

    embeddings = np.concatenate(all_embeddings, axis=0)
    indices = np.concatenate(all_indices, axis=0)

    # Sort by global index
    sort_order = np.argsort(indices)
    embeddings = embeddings[sort_order]

    logger.info(f"Total clip embeddings: {embeddings.shape}")

    # Save clip-level master
    clip_output = os.path.join(output_dir, "clip_embeddings.npz")
    np.savez(clip_output, embeddings=embeddings)
    logger.info(f"Clip-level embeddings saved to {clip_output}")

    # Pool to study level using clip index
    logger.info("Pooling to study level...")
    clip_index = np.load(clip_index_path, allow_pickle=True)
    study_uids = clip_index["study_uids"]

    if len(study_uids) != len(embeddings):
        logger.warning(
            f"Clip index ({len(study_uids)}) != embeddings ({len(embeddings)}). "
            f"Truncating to min."
        )
        n = min(len(study_uids), len(embeddings))
        study_uids = study_uids[:n]
        embeddings = embeddings[:n]

    unique_studies, inverse = np.unique(study_uids, return_inverse=True)
    n_studies = len(unique_studies)
    embed_dim = embeddings.shape[1]

    study_embeddings = np.zeros((n_studies, embed_dim), dtype=np.float64)
    clips_per_study = np.zeros(n_studies, dtype=np.int32)

    for i in range(len(embeddings)):
        study_embeddings[inverse[i]] += embeddings[i]
        clips_per_study[inverse[i]] += 1

    study_embeddings = (study_embeddings / clips_per_study[:, None]).astype(np.float32)

    study_output = os.path.join(output_dir, "study_embeddings.npz")
    np.savez(
        study_output,
        embeddings=study_embeddings,
        study_ids=unique_studies,
        clips_per_study=clips_per_study,
    )
    logger.info(
        f"Study-level embeddings saved to {study_output}: "
        f"{study_embeddings.shape} ({n_studies} studies)"
    )


def main():
    parser = argparse.ArgumentParser(description="UHN 18M embedding extraction")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True, help="CSV with s3_path 0 per line")
    parser.add_argument("--clip_index", required=True, help="uhn_clip_index.npz")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--devices", nargs="+", default=["cuda:0"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--save_every", type=int, default=500, help="Save chunk every N batches")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--merge_only", action="store_true", help="Skip extraction, just merge")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    world_size = len(args.devices)
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("UHN 18M Embedding Extraction")
    logger.info("=" * 60)
    logger.info(f"Config: {args.config}")
    logger.info(f"Data: {args.data}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Devices: {args.devices} (world_size={world_size})")
    logger.info(f"Batch size per GPU: {args.batch_size}")
    logger.info(f"Total batch size: {args.batch_size * world_size}")
    logger.info(f"Save every: {args.save_every} batches")
    logger.info("=" * 60)

    if not args.merge_only:
        if world_size == 1:
            extract_worker(0, 1, args.devices[0], config, args.data, args.output_dir,
                           args.batch_size, args.save_every, args.num_workers)
        else:
            mp.set_start_method("spawn", force=True)
            processes = []
            for rank in range(world_size):
                p = mp.Process(
                    target=extract_worker,
                    args=(rank, world_size, args.devices[rank], config, args.data,
                          args.output_dir, args.batch_size, args.save_every, args.num_workers),
                )
                p.start()
                processes.append(p)
            for p in processes:
                p.join()

    # Merge and pool
    merge_and_pool(args.output_dir, args.clip_index, world_size)
    logger.info("Done!")


if __name__ == "__main__":
    main()
