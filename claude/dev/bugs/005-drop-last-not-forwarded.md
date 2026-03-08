# Bug 005: `drop_last` not forwarded to DataLoader

**Severity**: MEDIUM
**Discovered**: 2026-03-07
**Status**: FIXED (2026-03-07)

## Root Cause

`src/datasets/data_manager.py` lines 69-93. The `init_data()` function accepts a `drop_last` parameter (line 25, default `True`), and `extract_uhn_embeddings.py` passes `drop_last=False` (line 135). However, the `VideoDataset` branch never forwards this to `make_videodataset()`, which defaults to `drop_last=True` in the DataLoader constructor.

This means the DataLoader silently drops the last incomplete batch on each GPU rank, even when the caller explicitly requests `drop_last=False`.

## Impact

For 8 GPUs with batch_size=32, up to 8 x 31 = 248 clips silently dropped per extraction. Those clips have no embeddings, causing a mismatch with `clip_index.npz` during study-level pooling. The `merge_and_pool` truncation logic masks this (truncates to the shorter of clip_index vs embeddings), but studies containing dropped clips will have incomplete averages.

For MIMIC (525,312 clips), the dataset size happens to be evenly divisible by common GPU/batch configurations, so the bug is not triggered in practice. For UHN (18,111,412 clips) with 8 GPUs, up to 248 tail clips may be affected.

## Fix Applied

Added `drop_last=drop_last` to the `make_videodataset(...)` call in the `VideoDataset` branch of `init_data()` in `data_manager.py`. The parameter was already accepted by `make_videodataset()` (with default `True`); it just wasn't being forwarded from `init_data()`.
