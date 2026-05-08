"""Cache frozen V-JEPA per-clip embeddings (plan §11, PR-N3).

Runs the frozen V-JEPA encoder over every clip in ``study_clip_manifest``
and writes one ``.npy`` per clip to
``{cache_prefix}/{study_id}/{clip_id}.npy`` (local dir or s3://).

Outputs a ``cache_index.parquet`` alongside the cache prefix with one row
per clip:

  clip_id, study_id, cached_path, checkpoint_id, checksum, embedding_dim,
  dtype, created_at

Resume is automatic — clips whose ``.npy`` already exists at the target
path are skipped. The worker mirrors ``evals/extract_embeddings.py`` but
writes per-clip rather than a monolithic NPZ so the K-sampler can fetch
sparsely later.

Usage:
    # Dry run over 8 clips
    python -m experiments.echoset_jepa.cache_cclip \\
        --manifest /tmp/echoset_pr_n2/study_clip_manifest_dedup.parquet \\
        --config configs/inference/vitl/lvef.yaml \\
        --cache_prefix /opt/dlami/nvme/echoset_jepa/cclip \\
        --dry_run 8

    # Full cache, 8 GPUs
    python -m experiments.echoset_jepa.cache_cclip \\
        --manifest ... --config ... --cache_prefix ... \\
        --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \\
        --batch_size 16
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import tempfile
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# Fix for "AF_UNIX path too long" error in torch.multiprocessing (matches
# the pattern used in evals/extract_embeddings.py).
_SHORT_TMP = "/tmp/echoset_cache"
os.makedirs(_SHORT_TMP, exist_ok=True)
tempfile.tempdir = _SHORT_TMP
os.environ.setdefault("TMPDIR", _SHORT_TMP)


# ---------------------------------------------------------------------------
# Config + checkpoint metadata
# ---------------------------------------------------------------------------


@dataclass
class CacheConfig:
    # Path to a V-JEPA video_classification_frozen config (same format as
    # evals/extract_embeddings.py --config). The encoder + transform come
    # from this config; we don't need the probe head.
    config_path: str
    cache_prefix: str                # local dir or s3:// prefix
    batch_size: int = 16
    num_workers: int = 8
    num_segments: int = 1
    dry_run: int = 0                 # if >0, cache only this many clips


def _checkpoint_identity(checkpoint_path: str) -> Tuple[str, str]:
    """Return (checkpoint_id, checksum) pair.

    checkpoint_id = the path, normalized. Useful human identifier.
    checksum      = first 16 bytes of SHA-256 over the file header (first
                    1MB). Full-file hash would be too slow for 5GB
                    checkpoints; header hash catches replacement without
                    reading the whole thing.
    """
    p = Path(checkpoint_path)
    try:
        with open(p, "rb") as f:
            head = f.read(1 << 20)
        checksum = hashlib.sha256(head).hexdigest()[:16]
    except Exception:
        checksum = "unreadable"
    return str(p), checksum


# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------


def _cache_path(prefix: str, study_id: str, clip_id: str) -> str:
    if prefix.startswith("s3://"):
        return f"{prefix.rstrip('/')}/{study_id}/{clip_id}.npy"
    return str(Path(prefix) / str(study_id) / f"{clip_id}.npy")


def _write_npy(path: str, arr: np.ndarray) -> None:
    if path.startswith("s3://"):
        # Write to tmp file then aws-cp (keeps boto3 optional)
        import subprocess
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
            np.save(tmp.name, arr)
            tmp_path = tmp.name
        try:
            r = subprocess.run(
                ["aws", "s3", "cp", tmp_path, path, "--quiet"],
                capture_output=True, text=True, timeout=60,
            )
            if r.returncode != 0:
                raise RuntimeError(f"s3 cp failed: {r.stderr.strip()[:200]}")
        finally:
            os.unlink(tmp_path)
    else:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.save(path, arr)


def _exists(path: str) -> bool:
    if path.startswith("s3://"):
        import subprocess
        r = subprocess.run(
            ["aws", "s3", "ls", path],
            capture_output=True, text=True, timeout=30,
        )
        return r.returncode == 0 and bool(r.stdout.strip())
    return Path(path).exists()


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


def _filter_to_cache(
    manifest_rows: Sequence[dict],
    cache_prefix: str,
    force: bool = False,
) -> List[dict]:
    """Drop rows whose cache target already exists (resume support)."""
    if force:
        return list(manifest_rows)
    out = []
    for r in manifest_rows:
        target = _cache_path(cache_prefix, r["study_id"], r["clip_id"])
        if not _exists(target):
            out.append(r)
    return out


def extract_worker(
    rank: int,
    world_size: int,
    device_str: str,
    cfg: CacheConfig,
    manifest_path: str,
    index_dir: str,
    force: bool,
) -> None:
    import yaml
    import pandas as pd
    from tqdm import tqdm
    import torch

    import logging as _logging
    _logging.basicConfig(level=_logging.INFO if rank == 0 else _logging.WARNING)
    logger.info("[rank %d/%d] starting on %s", rank, world_size, device_str)

    # Local GPU isolation (same pattern as evals/extract_embeddings.py)
    device_id = device_str.split(":")[-1]
    os.environ["CUDA_VISIBLE_DEVICES"] = device_id
    device = torch.device("cuda:0")

    with open(cfg.config_path) as f:
        config = yaml.safe_load(f)
    model_kwargs = config["model_kwargs"]

    # Lazy imports so test-time import doesn't pull V-JEPA modules
    from evals.video_classification_frozen.models import init_module
    from evals.video_classification_frozen.utils import make_transforms
    from src.datasets.data_manager import init_data

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

    ckpt_id, checksum = _checkpoint_identity(model_kwargs.get("checkpoint", "unknown"))

    # Load manifest, split by rank, resume-filter
    df = pd.read_parquet(manifest_path)
    df = df.reset_index(drop=True)
    if cfg.dry_run > 0:
        df = df.head(cfg.dry_run)
    all_rows = df.to_dict("records")
    # Simple rank partition (non-distributed): rank i takes rows with i%world_size==rank
    my_rows = [r for i, r in enumerate(all_rows) if i % world_size == rank]
    my_rows = _filter_to_cache(my_rows, cfg.cache_prefix, force=force)
    logger.info("[rank %d] %d clips to cache after resume filter", rank, len(my_rows))

    # Write a per-clip CSV (s3_uri + label) for the dataloader
    rank_csv = Path(index_dir) / f"rank_{rank}.csv"
    Path(index_dir).mkdir(parents=True, exist_ok=True)
    with open(rank_csv, "w") as f:
        for r in my_rows:
            f.write(f"{r['s3_uri']} 0\n")     # label is a dummy 0

    if not my_rows:
        logger.info("[rank %d] nothing to do", rank)
        return

    # Dataloader
    DEFAULT_NORMALIZATION = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    normalization = data_cfg.get("normalization") or DEFAULT_NORMALIZATION
    transform = make_transforms(
        training=False,
        num_views_per_clip=1,
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(0.75, 4 / 3),
        random_resize_scale=(0.08, 1.0),
        reprob=0.25,
        auto_augment=False,
        motion_shift=False,
        crop_size=data_cfg.get("resolution", 224),
        normalize=normalization,
    )
    loader, _ = init_data(
        data="VideoDataset",
        root_path=[str(rank_csv)],
        transform=transform,
        batch_size=cfg.batch_size,
        world_size=1,
        rank=0,
        clip_len=data_cfg.get("frames_per_clip", 16),
        frame_sample_rate=data_cfg.get("frame_step", 2),
        duration=None,
        num_clips=cfg.num_segments,
        allow_clip_overlap=True,
        num_workers=cfg.num_workers,
        drop_last=False,
    )
    loader.sampler.shuffle = False

    # Iterate + write per-clip .npy + build index rows
    index_rows: List[dict] = []
    row_iter = iter(my_rows)
    pbar = tqdm(loader, desc=f"rank{rank}", disable=(rank != 0))
    with torch.no_grad():
        for batch_idx, data in enumerate(pbar):
            clips = [[dij.to(device) for dij in di] for di in data[0]]
            clip_indices = [d.to(device) for d in data[2]]
            outputs = encoder(clips, clip_indices)
            pooled_segments = [o.mean(dim=1) for o in outputs]   # (B, d) per segment
            if len(pooled_segments) > 1:
                pooled = torch.stack(pooled_segments, dim=1).mean(dim=1)
            else:
                pooled = pooled_segments[0]
            arr = pooled.float().cpu().numpy()                    # (B, d)

            for i in range(arr.shape[0]):
                try:
                    r = next(row_iter)
                except StopIteration:
                    break
                target = _cache_path(cfg.cache_prefix, r["study_id"], r["clip_id"])
                _write_npy(target, arr[i].astype(np.float32))
                index_rows.append(
                    {
                        "clip_id": r["clip_id"],
                        "study_id": r["study_id"],
                        "cached_path": target,
                        "checkpoint_id": ckpt_id,
                        "checksum": checksum,
                        "embedding_dim": int(arr.shape[1]),
                        "dtype": "float32",
                        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    }
                )

    # Dump per-rank index shard
    shard_path = Path(index_dir) / f"cache_index_rank{rank}.parquet"
    pd.DataFrame(index_rows).to_parquet(shard_path, index=False)
    logger.info("[rank %d] wrote %d rows → %s", rank, len(index_rows), shard_path)


def merge_index(index_dir: str, out_path: str, world_size: int) -> None:
    """Merge per-rank shards into a single cache_index.parquet."""
    import pandas as pd

    shards = []
    for rank in range(world_size):
        p = Path(index_dir) / f"cache_index_rank{rank}.parquet"
        if p.exists():
            shards.append(pd.read_parquet(p))
    if not shards:
        logger.warning("no shards found in %s", index_dir)
        return
    full = pd.concat(shards, ignore_index=True).drop_duplicates("clip_id")
    full.to_parquet(out_path, index=False)
    logger.info("merged index: %d rows → %s", len(full), out_path)


def _run_multiprocess(cfg: CacheConfig, manifest_path: str, index_dir: str,
                      devices: List[str], force: bool) -> None:
    import torch.multiprocessing as mp
    world_size = len(devices)
    if world_size == 1:
        extract_worker(0, 1, devices[0], cfg, manifest_path, index_dir, force)
        return
    mp.set_start_method("spawn", force=True)
    procs = []
    for rank in range(world_size):
        p = mp.Process(
            target=extract_worker,
            args=(rank, world_size, devices[rank], cfg, manifest_path, index_dir, force),
        )
        p.start()
        procs.append(p)
    for p in procs:
        p.join()


def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="study_clip_manifest parquet")
    ap.add_argument("--config", required=True, help="V-JEPA inference YAML (encoder kwargs + data cfg)")
    ap.add_argument("--cache_prefix", required=True,
                    help="local dir or s3:// prefix for per-clip .npy files")
    ap.add_argument("--index_dir", default=None,
                    help="where to write per-rank shards + merged cache_index.parquet "
                         "(default: {cache_prefix}/_index)")
    ap.add_argument("--devices", nargs="+", default=["cuda:0"])
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--num_segments", type=int, default=1)
    ap.add_argument("--dry_run", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    index_dir = args.index_dir
    if index_dir is None:
        # Default to a local scratch dir when cache_prefix is s3://
        if args.cache_prefix.startswith("s3://"):
            index_dir = "/tmp/echoset_cache_index"
        else:
            index_dir = os.path.join(args.cache_prefix, "_index")
    Path(index_dir).mkdir(parents=True, exist_ok=True)

    cfg = CacheConfig(
        config_path=args.config,
        cache_prefix=args.cache_prefix,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_segments=args.num_segments,
        dry_run=args.dry_run,
    )
    _run_multiprocess(cfg, args.manifest, index_dir, args.devices, args.force)
    merge_index(index_dir, os.path.join(index_dir, "cache_index.parquet"), len(args.devices))


if __name__ == "__main__":
    _main()


__all__ = ["CacheConfig", "extract_worker", "merge_index", "_cache_path", "_filter_to_cache"]
