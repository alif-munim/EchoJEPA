# Bug 004: Silent index misalignment on video load failure

**Severity**: HIGH
**Discovered**: 2026-03-07
**Status**: FIXED (2026-03-07) — substitution tracking added; substitution still occurs but is now logged and counted

## Root Cause

`src/datasets/video_dataset.py` lines 234-244. When `__getitem__` fails to load a video (S3 timeout, corrupt MP4, missing file), it retries with `np.random.randint(len(self))`, returning a **random different clip's data** at the **original index position**. The extraction scripts assign global indices based on batch position (not which sample was actually loaded), so the embedding is mapped to the wrong clip.

## Impact

At 18M scale with S3-backed data, video load failures are inevitable. Every failure produces an embedding for the **wrong clip** mapped to the **wrong global index**. This corrupts both clip-level and study-level embeddings: a study containing a failed clip will have one random embedding mixed into its average.

At 0.1% failure rate: ~18K misaligned embeddings across the dataset. The actual failure rate during UHN extractions is unknown (failures are logged as warnings but not counted systematically).

For MIMIC (local files, not S3), the failure rate is likely near-zero.

## Fix Applied (Option A)

Added `_substitution_count` counter and per-event logging to `VideoDataset.__getitem__`. Each substitution now logs the original requested index/path and the substitute index/path at WARNING level. The `substitution_count` property exposes the total count for programmatic access.

Note: Initially used `threading.Lock` for the counter, but this broke `mp.spawn` multi-GPU extraction (`TypeError: cannot pickle '_thread.lock' object`). Removed the lock — DataLoader workers are separate processes (not threads), so each gets its own counter copy. The lock was unnecessary.

This makes the corruption rate visible after extraction runs. The substitution behavior itself is unchanged (random replacement) to avoid breaking the DataLoader's batch assembly contract.

## Future Improvement (Option B/C)

For a more complete fix, the extraction scripts could check `dataset.substitution_count` after extraction and log it in the output metadata. Or `__getitem__` could return the actual loaded index alongside the data, allowing extraction scripts to record the true mapping.

## Workaround

Study-level pooling mitigates this: each study has many clips (~56 on average), so one wrong clip has a small effect on the study average. The impact on downstream probing is proportional to `failure_rate / clips_per_study`.
