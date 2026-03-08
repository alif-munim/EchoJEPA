# Code Review Findings (2026-03-07)

Comprehensive review of the embedding extraction, post-processing, and evaluation pipeline. Focus: correctness bugs that affect downstream frozen linear probing results for the Nature Medicine 9-model comparison.

## Encoder Adapters (`evals/video_classification_frozen/modelcustom/`)

All 5 adapters wrap external model checkpoints to conform to the V-JEPA2 eval API.

| Adapter | Model | Normalization | Output | Status |
|---------|-------|--------------|--------|--------|
| `vit_encoder_multiclip.py` | EchoJEPA-G/L | ImageNet (native) | `[B, T*S*clips, D]` | Correct |
| `panecho_encoder.py` | PanEcho | ImageNet (native, post-fix) | `[B, 1, 768]` | Fixed (bug 002) |
| `echo_prime_encoder.py` | EchoPrime | Undo ImageNet → [0,255] → EchoPrime norm | `[B, 1, 512]` | Fixed (bug 002) |
| `echofm_encoder.py` | EchoFM | Undo ImageNet → [0,1] | `[B, T*L, 1024]` | Fixed (bugs 002, 003) |
| `videomae_encoder.py` | VideoMAE/EchoMAE | ImageNet (native) | `[B, tokens, D]` | Correct |

### Minor notes

- **EchoPrime `force_fp32`**: The adapter calls `.float()` but `extract_uhn_embeddings.py` runs under `torch.amp.autocast(bf16)`, which overrides it. MViT-v2-S is stable in bf16 so this is harmless. If fp32 were needed: wrap forward in `torch.amp.autocast(enabled=False)`.
- **Random Init**: No dedicated adapter needed. `vit_encoder_multiclip.py` works with a random-weight checkpoint.

## Extraction Scripts

### `extract_embeddings.py` (MIMIC)

- **Global index formula** (line 195): Correct. Matches DistributedSampler's interleaved pattern.
- **Shuffle fix**: Applied (line 149). See bug 001.
- **No autocast**: Embeddings computed in fp32, saved as fp32. No precision issues.
- **No process exit code checking** (lines 325-327): After `p.join()`, proceeds to merge without checking `p.exitcode`. Operational annoyance, not a correctness bug.

### `extract_uhn_embeddings.py` (UHN 18M)

- **Global index formula** (line 210): Correct, including resume case.
- **Resume logic**: Correct. `batch_sampler` replacement skips already-extracted clips.
- **DistributedSampler padding**: Padded duplicates get global indices beyond `n_dataset`, sorted to end, truncated by `merge_and_pool`. Study-level output is correct. `clip_embeddings.npz` has extra rows at the end (cosmetic).
- **bf16 autocast**: Embeddings converted to fp32 before saving (line 206). Correct.

### `video_dataset.py` (shared)

- **BUG 004 — Silent load-failure substitution** (lines 234-244): On video load failure, retries with `np.random.randint(len(self))`. Returns a random clip's data at the original index position. Extraction scripts cannot detect this. At 18M scale with S3, even 0.1% failure = ~18K misaligned embeddings. See `bugs/004-video-load-substitution.md`.
- **`video_fps` unbound** (lines 391-402): If `vr.get_avg_fps()` raises and `self.duration is not None`, `video_fps` is used unbound. Not triggered by UHN extraction (uses `frame_step`).

### `data_manager.py`

- **BUG 005 — `drop_last` not forwarded** (lines 69-93): `init_data()` accepts `drop_last` but the `VideoDataset` branch doesn't pass it to `make_videodataset()`, which defaults to `drop_last=True`. Up to `world_size × (batch_size - 1)` clips silently dropped. See `bugs/005-drop-last-not-forwarded.md`.

## Post-Processing Scripts

### `pool_embeddings.py`

- **Core pooling logic** (lines 26-58): Correct. `np.unique`/`inverse` accumulation is sound. Float64 accumulator prevents precision loss.
- **Fallback labels** (line 92): When `--labels` not provided, falls back to master NPZ `labels` which are dummy zeros. Not a crash, but semantically misleading. Documented workflow always uses `--labels`.

### `remap_embeddings.py`

- **Path-based matching** (lines 52-81): Correct. `rsplit(" ", 1)` parses CSV correctly. `path_to_idx` dict maps S3 paths to master NPZ row indices.
- No bugs found.

### `train_probe.py`

- **BUG 006 — `--labels` with `--train`/`--val`** (lines 510-511): Same label-only NPZ applied to both train and val files. Indices are positions into the master file, not split files. Would silently select wrong rows. Not triggered by documented workflows (README uses pre-split NPZs with embedded labels). See `bugs/006-labels-trainval-mode.md`.
- **Task auto-detection** (lines 96-104): Label-only NPZ labels are always float64, so binary classification labels detected as regression. Must always pass `--task classification` explicitly.
- **k-fold patient grouping**: `StratifiedKFold` splits on samples, not patients. 30.7% of MIMIC patients have multiple studies. Mitigated: final results use pre-built patient-level splits via `--train`/`--val`.
- **HP search on val set**: In train/val mode, hyperparameter grid search and final evaluation share the same val set. Standard practice for linear probes with low HP sensitivity.

## Eval Scaffold (`evals/video_classification_frozen/eval.py`)

- **Encoder freeze**: Triple protection (`model.eval()` + `requires_grad=False` + `torch.no_grad()`). No gradient leakage.
- **Per-segment backward** (line 689): Accumulates sum of losses, not mean. Effective LR scaled by `num_segments`. Only matters when `num_segments > 1` (not used in Nature Medicine configs).
- **AllReduce accuracy** (lines 677-678): Incorrect weighting when last batch is smaller. Metric reporting only, doesn't affect model weights.
- **Heavy augmentation for probes** (lines 869-880): `auto_augment=True`, `reprob=0.25` applied during probe training. Designed for pretraining, potentially suboptimal for frozen linear probes. Not a correctness bug.

## UHN Label NPZs

- All 47 regression/classification + 9 rare disease + 6 trajectory NPZs verified.
- Zero duplicate study IDs, patient splits correct, labels in physiological ranges.
- Trajectory pairs: zero self-pairs, all `days_between` in [30, 365], deltas exact.
- MIMIC shuffle fix verified: 100% label match for all 7 models.
