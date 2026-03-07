# UHN 18M Embedding Extraction

Operational guide for extracting embeddings from the 18M UHN echocardiogram dataset. Covers infrastructure, performance tuning, failure modes, and lessons learned.

## Infrastructure

- **Machine**: AWS SageMaker p4de.24xlarge — 8x A100-80GB, 96 vCPUs, 1.2TB RAM
- **Storage**: EFS (shared filesystem), S3 for video data (`s3://echodata25/`)
- **Conda env**: `vjepa2-312` (Python 3.12) — **must be used for all extraction/inference**
- **Dataset**: 18,111,412 clips across 319,815 studies, stored as MP4s on S3
- **Script**: `evals/extract_uhn_embeddings.py` — chunked multi-GPU extraction with crash-safe resume

## Launch Command

```bash
cd /path/to/vjepa2

PYTHONUNBUFFERED=1 /home/sagemaker-user/.conda/envs/vjepa2-312/bin/python \
    -m evals.extract_uhn_embeddings \
    --config configs/inference/vitl/extract_uhn_kinetics.yaml \
    --data experiments/nature_medicine/uhn/uhn_all_clips.csv \
    --clip_index experiments/nature_medicine/uhn/uhn_clip_index.npz \
    --output_dir experiments/nature_medicine/uhn/echojepa_l_kinetics_embeddings \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \
    --batch_size 64 \
    --num_workers 12 \
    --save_every 300 \
    2>&1 | tee experiments/nature_medicine/uhn/echojepa_l_kinetics_extraction.log
```

### Critical: Use Python binary directly, NOT `conda run`

`conda run -n vjepa2-312 python ...` buffers ALL stdout/stderr until the process finishes. For a 10+ hour extraction, this means zero real-time visibility. Always invoke the conda env's Python binary directly:

```bash
# BAD — no real-time output
conda run -n vjepa2-312 python -m evals.extract_uhn_embeddings ...

# GOOD — real-time output
PYTHONUNBUFFERED=1 /home/sagemaker-user/.conda/envs/vjepa2-312/bin/python \
    -m evals.extract_uhn_embeddings ...
```

`PYTHONUNBUFFERED=1` ensures Python itself doesn't buffer. Combined with `tee`, you get real-time log streaming.

## Performance Tuning

### The Bottleneck: S3 Latency

GPU compute is fast (milliseconds per batch for ViT-L in bf16). The dominant bottleneck is **S3 video download + decode**. Each clip requires an HTTP GET from S3, MP4 demux, and frame decode via decord. The GPU idles waiting for the next batch.

### DataLoader Settings (src/datasets/video_dataset.py)

Three interacting settings control the data pipeline:

| Setting | Description | Effect |
|---------|-------------|--------|
| `batch_size` | Clips per forward pass | Larger = fewer round-trips, more GPU memory |
| `num_workers` | DataLoader worker processes per GPU | More = more parallel S3 connections |
| `prefetch_factor` | Batches pre-fetched per worker | Higher = deeper buffer against S3 stalls |

Total in-flight clips per GPU = `num_workers × prefetch_factor × batch_size`.
Total S3 connections = `num_GPUs × num_workers` (each worker streams one clip at a time).

### What Was Tried

| Config | batch_size | num_workers | prefetch_factor | Result |
|--------|-----------|-------------|-----------------|--------|
| Original | 32 | 8 | **1** | ~42 clips/s/GPU. Heavy GPU idle time between batches. |
| Aggressive | 128 | 16 | 4 | **Workers crashed.** 8×16 = 128 S3 connections, each prefetching 4×128 = 512 clips. Total in-flight: 65K clips. S3 connection storm + RAM pressure caused `terminate called without an active exception` on 6/8 GPUs. |
| Balanced (recommended) | **64** | **12** | **4** | ~67 clips/s/GPU. **~60% faster** than original. Stable, all 8 GPUs active. ~9-10h for full 18M dataset with ViT-L. |

### Recommended Settings by Model

| Model | batch_size | num_workers | GPU mem/device | Notes |
|-------|-----------|-------------|----------------|-------|
| ViT-G (1.1B params) | 32 | 8-12 | ~20GB | Larger model, less room for batch size |
| ViT-L (304M params) | 64 | 12 | ~8GB | Sweet spot for A100-80GB |
| ViT-L (304M params) | 128 | 16 | ~15GB | May work but risks S3 connection storm |

### The `prefetch_factor` Fix

The single most impactful change. In `src/datasets/video_dataset.py`, line 121:

```python
# Before (default, extremely conservative):
dl_kwargs["prefetch_factor"] = 1

# After (optimized for S3 streaming):
dl_kwargs["prefetch_factor"] = 4
```

With `prefetch_factor=1`, only 1 batch per worker is buffered. When a batch completes, the GPU waits for the next S3 download. With `prefetch_factor=4`, workers continuously fetch ahead, filling a deeper buffer that absorbs S3 latency spikes.

**Trade-off**: Higher prefetch = more RAM per worker (each holds `prefetch_factor × batch_size` decoded clips in memory). With 12 workers × 4 prefetch × 64 batch × 8 GPUs = 24,576 clips in RAM across all workers. At ~10MB per decoded clip, that's ~240GB — feasible on the 1.2TB p4de but would OOM on smaller machines.

### Why Not Larger Batch Size?

Doubling batch size from 64→128 halves the number of batches but doesn't improve S3 throughput — the same number of workers fetch the same number of clips. The benefit is reduced per-batch overhead (fewer CUDA kernel launches, less Python loop overhead), but this overhead is already negligible vs S3 latency.

At bs=128, the risk is that a single slow S3 download blocks an entire 128-clip batch, causing longer GPU stalls. At bs=64, a slow download only blocks a 64-clip batch, and the deeper prefetch buffer is more likely to have another batch ready.

## Completed Extractions

| Model | Checkpoint | Output dir | Studies | Embed dim | Time | Config |
|-------|-----------|------------|---------|-----------|------|--------|
| EchoJEPA-G | `pt-280-an81.pt` | `echojepa_g_embeddings/` | 319,815 | 1408 | ~25.5h | bs=32, w=8, pf=1 |
| EchoJEPA-L-K | `vitl-kinetics-pt220-an55.pt` | `echojepa_l_kinetics_embeddings/` | (in progress) | 1024 | ~9-10h est | bs=64, w=12, pf=4 |

### Remaining Models

| Model | Checkpoint | Status |
|-------|-----------|--------|
| EchoJEPA-L | `vitl-pt-210-an25.pt` | Died at 75% (13,183/17,686 batches at bs=32). Chunks saved, can resume. |
| EchoMAE-L | (VideoMAE checkpoint) | TODO |
| Random Init | (untrained ViT-L) | TODO |

## Crash Recovery & Resume

The script saves embedding chunks every `save_every` batches per GPU rank. On crash:

1. Existing chunks in `chunks_rank{0..7}/` are preserved
2. On restart, each rank scans its chunk directory and skips completed batches
3. No need to re-extract already-saved work

**Caveat**: Resume only works with the **same batch_size and world_size**. Changing batch_size changes the total batch count and global index calculation, making old chunks incompatible. When changing settings, delete `chunks_rank*/` and start fresh.

## Process Management

### Killing a Running Extraction

The script spawns 8 GPU workers + N×8 dataloader workers. Killing the parent process doesn't always propagate:

```bash
# Step 1: Kill the main process
pkill -9 -f "extract_uhn_embeddings"

# Step 2: Kill orphaned GPU workers and dataloader workers
ps aux | grep "vjepa2-312/bin/python" | grep -v grep | awk '{print $2}' | xargs kill -9

# Step 3: Wait for GPU memory to free (may take 5-10s)
sleep 10 && nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
```

**Zombie dataloader workers**: Workers stuck in `D` state (uninterruptible disk sleep, waiting on S3 IO) cannot be killed even with `kill -9`. They'll eventually timeout on their own (usually 30-60s). They don't hold GPU memory and won't interfere with a new extraction, but they do consume CPU/RAM until they die.

### Monitoring

```bash
# Real-time log
tail -f experiments/nature_medicine/uhn/{model}_extraction.log

# GPU utilization (should show >0% on all devices during active extraction)
watch -n 5 nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader

# Worker count (should be 8 + num_workers*8 = 104 for workers=12)
ps aux | grep "vjepa2-312/bin/python" | grep -v grep | wc -l

# Chunk progress (each chunk = save_every batches)
ls experiments/nature_medicine/uhn/{model}_embeddings/chunks_rank0/ | wc -l
```

## Output Structure

```
experiments/nature_medicine/uhn/{model}_embeddings/
├── chunks_rank0/          # Intermediate chunk files (deleted after merge)
│   ├── chunk_00000300.npz
│   ├── chunk_00000600.npz
│   └── ...
├── chunks_rank1/
│   └── ...
├── ...
├── chunks_rank7/
│   └── ...
├── clip_embeddings.npz    # Final merged clip-level (18M × embed_dim) — written by merge step
└── study_embeddings.npz   # Final study-level mean-pooled (320K × embed_dim) — written by merge step
```

The merge + pool step runs automatically after extraction completes. It:
1. Loads all chunks from all ranks
2. Sorts by global index to restore original CSV order
3. Saves `clip_embeddings.npz`
4. Pools clips to study-level using `uhn_clip_index.npz`
5. Saves `study_embeddings.npz` with `embeddings`, `study_ids`, `clips_per_study`

## Extraction Configs

All configs live in `configs/inference/`:

| Config | Model | Checkpoint |
|--------|-------|-----------|
| `vitg-384/extract_uhn.yaml` | EchoJEPA-G (ViT-G) | `pt-280-an81.pt` |
| `vitl/extract_uhn.yaml` | EchoJEPA-L (ViT-L) | `vitl-pt-210-an25.pt` |
| `vitl/extract_uhn_kinetics.yaml` | EchoJEPA-L-K (ViT-L, Kinetics init) | `vitl-kinetics-pt220-an55.pt` |

Config anatomy (all share the same structure):
```yaml
experiment:
  data:
    dataset_type: VideoDataset
    resolution: 224          # center crop size
    frames_per_clip: 16      # frames per clip
    frame_step: 2            # temporal stride

model_kwargs:
  checkpoint: "/path/to/checkpoint.pt"
  module_name: evals.video_classification_frozen.modelcustom.vit_encoder_multiclip
  pretrain_kwargs:
    encoder:
      checkpoint_key: target_encoder
      model_name: vit_large   # or vit_giant_xformers
      patch_size: 16
      tubelet_size: 2
      uniform_power: true
      use_rope: true
  wrapper_kwargs:
    max_frames: 128
    use_pos_embed: false
```

## Common Failure Modes

| Symptom | Cause | Fix |
|---------|-------|-----|
| `terminate called without an active exception` | Too many S3 connections or worker OOM | Reduce `batch_size`, `num_workers`, or `prefetch_factor` |
| Workers die silently (GPU memory drops to model-only) | Dataloader crash from bad video or S3 timeout | Script handles via retry; if persistent, check specific S3 path |
| Progress stalls at batch N forever | Deadlock in multiprocessing queue | Kill and resume (chunks saved) |
| `AF_UNIX path too long` | Default /tmp path exceeds socket limit | Already handled in script via `tempfile.tempdir = "/tmp/extract_uhn"` |
| 0% GPU utilization despite running | S3 latency starving GPU | Increase `prefetch_factor` (the most impactful fix) |
| No real-time log output | Using `conda run` which buffers stdout | Use conda env Python binary directly with `PYTHONUNBUFFERED=1` |
| EchoJEPA-G at 25h but ViT-L should be faster | ViT-G extraction was done before prefetch fix | ViT-G would be ~15h with optimized settings |

## S3 Video Load Failures

A small number of S3 videos (~180 out of 18M, <0.001%) fail to load due to:
- Corrupted MP4 headers (`moov atom not found`)
- Truncated files (`Error reading 48 bytes`)
- S3 transient errors

The `VideoDataset` handles these via retry with a random replacement sample. This is logged as a warning but doesn't affect extraction quality. The failed clip is skipped and another clip from the dataset is substituted. Study-level pooling is unaffected since each study has many clips.

## Timing Reference

For a full 18M clip extraction on 8x A100-80GB:

| Model size | Optimized (bs=64, w=12, pf=4) | Original (bs=32, w=8, pf=1) |
|-----------|-------------------------------|------------------------------|
| ViT-L (304M) | ~9-10h | ~15h |
| ViT-G (1.1B) | ~15h (estimated) | ~25.5h (actual) |

The speedup comes entirely from better data pipeline saturation, not faster GPU compute.
