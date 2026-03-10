#!/usr/bin/env python
"""
Chunked multi-GPU embedding extraction for UHN 18M dataset.

Optimized for large-scale S3-based extraction with:
- bf16 autocast for ~2x GPU throughput
- Larger batch sizes (ViT-G fits batch 32 per A100-80GB in bf16)
- Chunked saving every N batches (crash-safe resume)
- Study-level pooling built-in (no separate P5 step)
- Sequential (non-shuffled) sampling for correct clip_index alignment and efficient resume

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
import gc
import logging
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
import yaml
from torch.utils.data.sampler import BatchSampler
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


class ListSampler:
    """Sampler that yields a fixed list of indices in order."""

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


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

    # Enable TF32 for fp32 models (up to 8x matmul throughput on A100)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # NOTE: cudnn.benchmark disabled — it caches workspace per unique layer config,
    # growing GPU memory from ~43GB to ~77GB over time, causing OOM on MViT.
    torch.backends.cudnn.benchmark = False

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
    use_bf16 = not model_kwargs.get("wrapper_kwargs", {}).get("force_fp32", False)
    if use_bf16:
        encoder = encoder.to(dtype=torch.bfloat16)

    if rank == 0:
        log.info(f"Encoder embed_dim: {encoder.embed_dim}, dtype: {'bf16' if use_bf16 else 'fp32'}")

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

    # Build dataset (via init_data) then construct DataLoader manually for resume support.
    # init_data creates a DistributedSampler which we need to bypass for resume.
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

    # CRITICAL: Disable shuffle so embeddings align with clip_index.npz order.
    data_loader.sampler.shuffle = False
    dataset = data_loader.dataset

    # Check for resume
    chunk_dir = os.path.join(output_dir, f"chunks_rank{rank}")
    os.makedirs(chunk_dir, exist_ok=True)

    existing_chunks = sorted(
        [f for f in os.listdir(chunk_dir) if f.startswith("chunk_") and f.endswith(".npz")]
    )
    start_batch = 0
    if existing_chunks:
        last_chunk = existing_chunks[-1]
        start_batch = int(last_chunk.replace("chunk_", "").replace(".npz", ""))
        log.info(f"[Rank {rank}] Resuming from batch {start_batch} ({len(existing_chunks)} chunks)")

    # Compute this rank's sequential indices (matches DistributedSampler with shuffle=False)
    n_dataset = len(dataset)
    total_size = math.ceil(n_dataset / world_size) * world_size
    all_indices = list(range(n_dataset))
    # Pad for even distribution (same as DistributedSampler)
    all_indices += all_indices[: (total_size - len(all_indices))]
    rank_indices = all_indices[rank:total_size:world_size]

    if start_batch > 0:
        # Efficient resume: create a NEW DataLoader with a sampler that skips
        # already-done clips. PyTorch doesn't allow reassigning batch_sampler
        # after DataLoader initialization.
        samples_done = start_batch * batch_size
        remaining_indices = rank_indices[samples_done:]
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=BatchSampler(
                ListSampler(remaining_indices), batch_size, drop_last=False
            ),
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=4,
        )
        total_batches = len(data_loader)
        log.info(
            f"[Rank {rank}] Efficient resume: skipping {samples_done} clips, "
            f"{total_batches} batches remaining"
        )
    else:
        total_batches = len(data_loader)

    log.info(f"[Rank {rank}] {total_batches} batches, save every {save_every}")

    # Extract
    chunk_embeddings = []
    chunk_indices = []
    chunk_paths = []
    batches_in_chunk = 0

    iterator = tqdm(data_loader, desc=f"GPU {device_id}", total=total_batches, disable=(rank != 0))

    autocast_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else torch.amp.autocast("cuda", enabled=False)
    with torch.no_grad(), autocast_ctx:
        for batch_idx_local, data in enumerate(iterator):
            batch_idx = batch_idx_local + start_batch

            clips = [[dij.to(device) for dij in di] for di in data[0]]
            clip_indices_batch = [d.to(device) for d in data[2]]
            batch_size_actual = len(data[1])

            # Get paths if available
            if len(data) > 3:
                batch_paths = list(data[3])
            else:
                batch_paths = [f"sample_{batch_idx * batch_size * world_size + rank + i * world_size}" for i in range(batch_size_actual)]

            # Forward
            outputs = encoder(clips, clip_indices_batch)
            pooled_segments = [o.mean(dim=1) for o in outputs]
            if len(pooled_segments) > 1:
                pooled = torch.stack(pooled_segments, dim=1).mean(dim=1)
            else:
                pooled = pooled_segments[0]

            chunk_embeddings.append(pooled.float().cpu().numpy())
            chunk_paths.extend(batch_paths)

            # Track global indices for ordering
            for i in range(batch_size_actual):
                global_idx = batch_idx * batch_size * world_size + rank + i * world_size
                chunk_indices.append(global_idx)

            # Explicitly free GPU tensors to prevent memory growth.
            # MViT's _add_rel_pos allocates 3.66 GiB spikes; without explicit del,
            # deferred Python GC lets GPU memory grow ~25 MB/batch → OOM after ~1000 batches.
            del clips, clip_indices_batch, outputs, pooled_segments, pooled, data

            batches_in_chunk += 1

            # Periodic GPU memory cleanup. CUDA allocator holds freed blocks;
            # gc.collect() + empty_cache() reclaims them (66 GB → 33 GB observed).
            # Every 100 batches keeps peak growth under ~2.5 GB between cleanups.
            if batches_in_chunk % 100 == 0:
                gc.collect()
                torch.cuda.empty_cache()

            # Save chunk
            if batches_in_chunk >= save_every or batch_idx_local == total_batches - 1:
                emb = np.concatenate(chunk_embeddings, axis=0)
                idx = np.array(chunk_indices)
                paths_arr = np.array(chunk_paths, dtype=object)
                chunk_path = os.path.join(chunk_dir, f"chunk_{batch_idx + 1:08d}.npz")
                np.savez(chunk_path, embeddings=emb, indices=idx, paths=paths_arr)

                if rank == 0:
                    log.info(
                        f"Saved chunk {chunk_path} ({emb.shape[0]} clips, "
                        f"batch {batch_idx + 1}/{start_batch + total_batches})"
                    )

                chunk_embeddings = []
                chunk_indices = []
                chunk_paths = []
                batches_in_chunk = 0
                gc.collect()
                torch.cuda.empty_cache()

    log.info(f"[Rank {rank}] Done. Chunks saved to {chunk_dir}/")


def merge_and_pool(output_dir: str, clip_index_path: str, world_size: int):
    """Merge all chunk files from all ranks, then pool to study level."""
    logger.info("Merging chunks from all ranks...")

    all_embeddings = []
    all_indices = []
    all_paths = []
    has_paths = True

    for rank in range(world_size):
        chunk_dir = os.path.join(output_dir, f"chunks_rank{rank}")
        chunks = sorted(
            [f for f in os.listdir(chunk_dir) if f.startswith("chunk_") and f.endswith(".npz")]
        )
        for chunk_file in chunks:
            d = np.load(os.path.join(chunk_dir, chunk_file), allow_pickle=True)
            all_embeddings.append(d["embeddings"])
            all_indices.append(d["indices"])
            if "paths" in d:
                all_paths.append(d["paths"])
            else:
                has_paths = False

    embeddings = np.concatenate(all_embeddings, axis=0)
    indices = np.concatenate(all_indices, axis=0)

    # Sort by global index to restore CSV order
    sort_order = np.argsort(indices)
    embeddings = embeddings[sort_order]

    logger.info(f"Total clip embeddings: {embeddings.shape}")

    # Save clip-level master
    clip_output = os.path.join(output_dir, "clip_embeddings.npz")
    if has_paths and all_paths:
        paths = np.concatenate(all_paths, axis=0)[sort_order]
        np.savez(clip_output, embeddings=embeddings, paths=paths)
        logger.info(f"Clip-level embeddings + paths saved to {clip_output}")
    else:
        np.savez(clip_output, embeddings=embeddings)
        logger.info(f"Clip-level embeddings saved to {clip_output} (no paths available)")

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
