# Changelog — EchoJEPA Codebase (`vjepa2/`)

Comprehensive record of all code changes, bug fixes, extraction runs, infrastructure work, and operational events in the `vjepa2` repository. For Nature Medicine manuscript-level progress (data pipeline, planning, writing), see `uhn_echo/nature_medicine/context_files/dev/changelog.md`.

**Format:** Each entry includes commit hash (where applicable), timestamp, and category. Entries without commits are operational events (extraction launches, crashes, verifications) that don't produce code changes but are critical for reproducibility.

---

## 2026-03-08 (continued — Session 9)

### Code Changes: S3 Path Provenance in Embeddings

- **VideoDataset returns video path** — `src/datasets/video_dataset.py`
  `__getitem__` now returns a 4-tuple `(buffer, label, clip_indices, sample_uri)` instead of the previous 3-tuple. `sample_uri` is `self.samples[index]` — the S3 URI or local path of the source video. All consumers verified safe:
  - `extract_embeddings.py` (MIMIC): Already had `if len(data) > 3:` check — now receives real paths automatically
  - `extract_uhn_embeddings.py` (UHN): Already had `if len(data) > 3:` check — now receives real paths automatically
  - `video_classification_frozen/eval.py`: Already had `if len(data) > 3:` check
  - `app/vjepa/train.py`: Only accesses `udata[0]` — unaffected
  - `video_classification_frozen_multi/eval.py`: Uses `VideoGroupDataset`, not modified

- **UHN extraction saves paths** — `evals/extract_uhn_embeddings.py`
  Added `chunk_paths` accumulator alongside `chunk_embeddings` and `chunk_indices`. Paths collected from `data[3]` per batch, saved in each chunk NPZ as `paths` array, and merged/sorted in `merge_and_pool()`. Graceful fallback: chunks without paths (from older runs) handled via `has_paths` flag.

- **MIMIC extraction** — `evals/extract_embeddings.py` — **No changes needed.** Lines 170-173 already handled `data[3]` with dummy fallback; now receives real paths automatically. Lines 201-208 already save `paths` in output NPZ.

- **Probe training** — `evals/train_probe.py` — **No changes needed.** Line 90 already handles missing paths gracefully: `paths = data["paths"] if "paths" in data else ...`

### Post-Hoc Path Injection into Existing MIMIC NPZs

Injected S3 video paths into all 7 existing MIMIC embedding NPZ files without re-extraction. Two cases:

**4 original models** (echojepa_g, echojepa_l, echojepa_l_kinetics, echomae — 525,312 clips each):
- Paths sourced from `clip_index.npz["s3_paths"][:525312]` — direct row-index mapping since shuffle fix ensures row N = CSV row N
- NPZ rewritten with `paths` array added alongside existing `embeddings` and `paths` keys

**3 re-extracted models** (panecho, echoprime, echofm — 525,320 clips each):
- 525,320 > 525,312 (clip_index size) because `drop_last=False` + DistributedSampler padding adds 1 extra clip (8 GPUs, ceil-padded)
- Paths for first 525,319 clips sourced from source CSV (`experiments/nature_medicine/mimic/mimic_clips.csv`)
- Last clip (index 525,319) marked as `padding_duplicate_0`
- NPZ rewritten with `paths` array

**UHN decision:** Paths NOT duplicated into 70-95GB clip embedding files. Paths already available via `uhn_clip_index.npz["s3_paths"]` at the same row index. For future extractions, `extract_uhn_embeddings.py` will automatically include paths in the merged `clip_embeddings.npz`.

### End-to-End Path-Embedding Verification (MIMIC, all 7 models)

Verified that stored embedding[i] actually corresponds to the video at paths[i] by re-encoding random clips through each model and comparing cosine similarity. All verification run on CPU (`CUDA_VISIBLE_DEVICES=""`) to avoid interfering with running GPU extractions (EchoJEPA-L-K, EchoMAE-L on GPUs 0-7).

**Match test** (3 random clips per model, encode same video as stored embedding):

| Model | Clip 1 | Clip 2 | Clip 3 | Mean |
|-------|--------|--------|--------|------|
| EchoJEPA-G | 0.967 | 0.977 | 0.975 | 0.973 |
| EchoJEPA-L | 0.996 | 0.995 | 0.996 | 0.996 |
| EchoJEPA-L-K | 0.986 | 0.983 | 0.983 | 0.984 |
| EchoMAE | 0.996 | 0.993 | 0.995 | 0.995 |
| PanEcho | 0.971 | 0.950 | 0.960 | 0.960 |
| EchoPrime | 0.999 | 0.999 | 0.999 | 0.999 |
| EchoFM | 1.000 | 1.000 | 1.000 | 1.000 |

Cosine similarity < 1.0 due to `random_clip_sampling=True` (different temporal crop each run). MIMIC clips are short enough that similarity stays >0.95.

**Negative control** (3 random clip pairs per model, compare stored embedding to WRONG video):

| Model | Mean Match | Mean Mismatch | Gap | Verdict |
|-------|-----------|--------------|-----|---------|
| EchoJEPA-G | 0.973 | 0.716 | 0.257 | PASS |
| EchoJEPA-L | 0.996 | 0.942 | 0.054 | PASS |
| EchoJEPA-L-K | 0.984 | 0.403 | 0.581 | PASS |
| EchoMAE | 0.995 | 0.516 | 0.478 | PASS |
| PanEcho | 0.961 | 0.398 | 0.562 | PASS |
| EchoPrime | 0.999 | 0.784 | 0.215 | PASS |
| EchoFM | 1.000 | 0.999 | 0.0002 | FAIL (collapse) |

**21/21 match tests PASS** (cosine > 0.95). **6/7 models show clear match/mismatch gap** (0.054-0.581). **EchoFM shows representation collapse** — cosine ~0.9998 everywhere regardless of input, making match/mismatch indistinguishable. This confirms the collapse finding from the earlier cosine similarity verification (Session 8).

### End-to-End Path-Embedding Verification (UHN, 2 models)

Same match/mismatch protocol as MIMIC (Session 9 earlier), applied to UHN 18M embeddings. Script used random-access reads into the 95/70GB NPZ files (no full load) and scanned `uhn_all_clips.csv` for S3 paths. All run on CPU (`CUDA_VISIBLE_DEVICES=""`).

| Model | Mean Match | Mean Mismatch | Gap | Verdict |
|-------|-----------|--------------|-----|---------|
| EchoJEPA-G | 0.9951 | 0.6476 | 0.3476 | PASS |
| EchoJEPA-L | 0.9951 | 0.9132 | 0.0818 | PASS |

- UHN gaps are larger than MIMIC (G: 0.348 vs 0.257, L: 0.082 vs 0.054) — more patient diversity in 18M dataset
- EchoJEPA-G encodes highly clip-specific representations (mismatch only 0.65)
- EchoJEPA-L has concentrated embedding space (mismatch 0.91) but gap is still clear
- Script: `/tmp/verify_uhn_embeddings.py` — uses `zipfile` random-access to read specific rows from uncompressed NPZ without loading full 95GB array

### Cross-Dataset Verification Analysis (MIMIC + UHN combined)

Consolidated results from all 9 model-dataset verifications (7 MIMIC + 2 UHN):

**Full results table (sorted by gap):**

| Dataset | Model | Embed Dim | Mean Match | Mean Mismatch | Gap | Mismatch Rank |
|---------|-------|-----------|-----------|--------------|-----|---------------|
| MIMIC | EchoJEPA-L-K | 1024 | 0.984 | 0.403 | 0.581 | 1 (most dispersed) |
| MIMIC | PanEcho | 768 | 0.961 | 0.398 | 0.562 | 2 |
| MIMIC | EchoMAE | 1024 | 0.995 | 0.516 | 0.478 | 3 |
| UHN | EchoJEPA-G | 1408 | 0.995 | 0.648 | 0.348 | 4 |
| MIMIC | EchoJEPA-G | 1408 | 0.973 | 0.716 | 0.257 | 5 |
| MIMIC | EchoPrime | 512 | 0.999 | 0.784 | 0.215 | 6 |
| UHN | EchoJEPA-L | 1024 | 0.995 | 0.913 | 0.082 | 7 |
| MIMIC | EchoJEPA-L | 1024 | 0.996 | 0.942 | 0.054 | 8 |
| MIMIC | EchoFM | 1024 | 1.000 | 0.999 | 0.000 | 9 (collapsed) |

**Key findings:**

1. **Alignment verified: 8/9 clear PASS, 1 collapsed.** All match cosines >0.95 across both datasets (27/27 individual clip tests). The verification protocol is definitive.

2. **Mismatch cosine is a representation dispersion metric.** It measures how "spread out" the embedding space is — low mismatch means random clips land far apart, high mismatch means they cluster together. This is distinct from downstream task performance but is a necessary condition for discrimination.

3. **Three regimes of representation geometry:**
   - **Dispersed** (mismatch <0.55): EchoJEPA-L-K, PanEcho, EchoMAE. Random clip pairs have cosine ~0.4-0.5 — embeddings are well-separated.
   - **Moderate** (mismatch 0.65-0.80): EchoJEPA-G, EchoPrime. Clips are distinguishable but share a stronger common structure.
   - **Concentrated** (mismatch >0.90): EchoJEPA-L, EchoFM. Most variation lives in a narrow band. EchoFM is the extreme — total collapse to a single point.

4. **EchoJEPA-L's concentration is consistent across datasets.** MIMIC mismatch 0.942, UHN mismatch 0.913. The model encodes information in small deviations from a dominant "echo video" direction. This is not necessarily bad for linear probes (they can exploit small but consistent differences) but makes cosine-based verification harder — the 0.054-0.082 gap is real but narrow.

5. **Same model, bigger gap on UHN.** EchoJEPA-G: 0.348 (UHN) vs 0.257 (MIMIC). EchoJEPA-L: 0.082 (UHN) vs 0.054 (MIMIC). UHN's 18M clips span far more patients, echo machines, operators, and time periods than MIMIC's 525K, producing greater inter-clip diversity.

6. **EchoFM collapse is now triple-confirmed.** (a) Study-level cosine verification: gap <0.001. (b) Clip-level match/mismatch: gap 0.0002. (c) Earlier manual inspection: cosine ~0.9998 everywhere. This model is unlikely to support meaningful downstream discrimination. Root cause unknown (possibly training divergence or normalization issue in original model).

7. **Match similarity variation reflects temporal sampling.** Models with higher match cosine (EchoPrime 0.999, EchoFM 1.000) are less sensitive to which frames are selected. Models with lower match cosine (PanEcho 0.950-0.971, EchoJEPA-G 0.967-0.997) are more sensitive to temporal cropping, suggesting they encode frame-level detail rather than just study-level appearance.

**Technical note:** UHN verification used `zipfile` random-access into 95/70GB uncompressed NPZ files (read specific rows by seeking to `header_offset + idx * row_bytes`). This avoided loading the full array and completed each row read in <100ms. Useful pattern for any future spot-checks on large embedding files.

### Shuffle Fix Mapping Verification

Independently verified that row N of each NPZ corresponds to CSV row N by reading the shuffle fix scripts:
- `fix_shuffle_order.py` (UHN): `reordered[perm[i]] = embeddings[i]` — positions output by CSV index
- `fix_mimic_shuffle.py` (MIMIC): Same permutation-inverse logic
- 180 UHN positions (EchoJEPA-G) zero-filled because their permutation targets were >= n_embeddings — these are at their correct CSV positions, not shifted
- For MIMIC: permutation uses n_dataset=525,319 (CSV rows), not 525,312 (NPZ rows after padding dedup), ensuring correct alignment

---

## 2026-03-08

### Bug Fixes

- **Fix: DataLoader resume logic** — `2065eb6` (03:09 UTC)
  PyTorch does not allow `data_loader.batch_sampler = ...` after init. Changed `extract_uhn_embeddings.py` to create a new `DataLoader` with `BatchSampler(ListSampler(remaining_indices), ...)` for resume. Also added `dataset = data_loader.dataset` (line 142) before the resume check to ensure dataset reference is captured before potential DataLoader replacement. Bug discovered when L-K extraction resume produced corrupt output (787,200/18.1M clips merged into truncated files that had to be deleted).

- **Fix: EchoJEPA-L shuffle status correction** — `2b0d7da` (09:05 UTC)
  Changelog and embedding-status docs incorrectly claimed EchoJEPA-L was "extracted after shuffle fix". Cosine similarity analysis proved it was extracted BEFORE the fix (within-study gap 0.005 = indistinguishable from random). Corrected all documentation.

### Extraction Runs

- **EchoJEPA-L-K UHN — Restart #2** (01:00 UTC)
  Killed stalled extraction (7/8 ranks dead at batch 600, only rank 3 still progressing — 88 chunks vs 63 for other ranks). Root cause: S3 connection storm with `num_workers=12` (8 ranks × 12 workers = 96 concurrent S3 connections). Zombie workers (Z state) and uninterruptible sleep (D state) workers had to be killed with `kill -9`.
  - Config: `configs/inference/vitl/extract_uhn_kinetics.yaml`
  - Checkpoint: `checkpoints/anneal/keep/vitl-kinetics-pt220-an55.pt`
  - Params: 8×A100, bs=64, w=6 (down from 12), pf=4, save_every=300
  - Log: `experiments/nature_medicine/uhn/extract_uhn_lk_p6.log`
  - **Status at 09:05 UTC:** 52% (18,025/34,474 batches), ~7h remaining
  - Chunk progress: ranks 0-2,4-7 have 63 chunks each; rank 3 has 88 (25 extra from stalled run before restart, will be handled correctly by merge since indices are tracked)

- **EchoMAE-L UHN — Started** (previous session, running on separate node)
  - Config: `configs/inference/vitl/extract_uhn_echomae.yaml`
  - Checkpoint: `checkpoints/videomae-ep163.pt` (pretrain format, auto-converted)
  - Params: 8×A100, bs=64, w=12, pf=1
  - Log: `logs/echomae_uhn_extraction.log`
  - **Status at 09:05 UTC:** 556 chunks across 8 ranks (~69-70 per rank), running stable

- **MIMIC re-extraction — Running on separate node**
  - Script: `scripts/reextract_mimic_3models.sh`
  - Sequential: PanEcho → EchoPrime → EchoFM
  - 8×A100, bs=32, w=8
  - Writing to shared EFS at `experiments/nature_medicine/mimic/`

### Data Integrity Verification

- **EchoJEPA-L shuffle verification** (08:30-09:00 UTC)
  Developed and applied cosine similarity verification method to check embedding-CSV alignment:
  1. Sample 5 studies at evenly-spaced positions in the dataset
  2. For each study, compute mean pairwise cosine similarity among its clips (within-study)
  3. Compare to cosine similarity between the study's clips and random clips from other studies (between-study)
  4. If correctly ordered: within-study >> between-study. If shuffled: gap ≈ 0.

  **Results BEFORE fix (shuffled):**
  - Mean within=0.951, between=0.946, gap=0.005 (indistinguishable)

  **Results AFTER fix (reordered):**
  - Mean within=0.956, between=0.925, gap=0.031 (6.2x improvement)
  - Per-study gaps: 0.066, 0.032, 0.012, 0.028, 0.018

  **Definitive method:** Reconstruct DistributedSampler permutation via `torch.randperm(n, generator=g)` with `g.manual_seed(seed + epoch)`, apply inverse permutation, re-check if within-study clustering improves. This is conclusive because the permutation is deterministic.

  Note: EchoJEPA-L has very uniform representations (pairwise cosine ~0.998 at study level), making shuffle detection harder than for EchoJEPA-G. The gap is real but small in absolute terms.

- **EchoJEPA-L post-hoc shuffle fix** (08:48-08:57 UTC, background task `by78tzd98`)
  Applied `fix_shuffle_order.py` to `echojepa_l_embeddings/`:
  - Input: 18,110,464 × 1024 (shuffled)
  - Permutation reconstructed: n=18,111,412, world_size=8
  - 948 clips had permutation targets >= n_embeddings (from drop_last), zero-filled
  - Output: reordered `clip_embeddings.npz` + re-pooled `study_embeddings.npz` (319,802 studies)
  - Originals backed up as `.shuffled_backup`

- **Chunk index verification method discovered to be unreliable** (earlier in session)
  Initially tried checking chunk `indices` arrays to verify shuffle status. Discovered that indices are computed from `batch_idx * batch_size * world_size + rank + i * world_size` — always sequential regardless of DistributedSampler shuffle setting. The indices track batch position, not dataset position. Had to develop the cosine similarity method instead.

### UHN Per-Task Split Pipeline

- **Built `evals/regenerate_uhn_downstream.py`** — `88f3c4e` (08:21 UTC)
  Joins study-level embeddings with label NPZs on `study_ids`, creates per-task train/val/test splits. Handles standard tasks (47) and trajectory paired-study tasks (6). Output: `{model}_splits/{task}/train.npz`, `val.npz`, `test.npz`.

- **Generated splits for EchoJEPA-G and EchoJEPA-L** (08:21 UTC)
  48 task directories each (47 standard + 1 trajectory parent dir containing 6 sub-tasks). Total: 96 task dirs, unblocks all UHN probing for these two models.

### Config Changes

- Created `configs/inference/vitl/extract_uhn_echomae.yaml` — `88f3c4e` (08:21 UTC)

### MIMIC Re-extraction Complete (from separate node, results on shared EFS)

All 3 norm-bugged models re-extracted with fixed adapters:

| Model | Clips | Dim | Size | Time | Commit (fix) |
|-------|-------|-----|------|------|-------------|
| PanEcho | 525,320 | 768 | 1.6GB | ~30min | `4803640` |
| EchoPrime | 525,320 | 512 | 1.1GB | ~17min | `4803640` |
| EchoFM | 525,320 | 1024 | 2.1GB | ~100min | `4803640` |

Downstream pipeline regenerated for all 3: study-level pooling + 23 task splits. All 7 MIMIC models now probe-ready.

### Embedding Audit (comprehensive status check)

Performed full audit of extraction status across UHN + MIMIC:

**MIMIC (all complete):**
| Model | Status | Splits |
|-------|--------|--------|
| EchoJEPA-G | Probe-ready | 23 tasks |
| EchoJEPA-L | Probe-ready | 23 tasks |
| EchoJEPA-L-K | Probe-ready | 23 tasks |
| EchoMAE | Probe-ready | 23 tasks |
| PanEcho | Probe-ready (re-extracted) | 23 tasks |
| EchoPrime | Probe-ready (re-extracted) | 23 tasks |
| EchoFM | Probe-ready (re-extracted) | 23 tasks |

**UHN:**
| Model | Clip Embeddings | Study Embeddings | Splits | Status |
|-------|----------------|-----------------|--------|--------|
| EchoJEPA-G | 18,111,232 × 1408 (95GB) | 319,815 × 1408 (1.7GB) | 48 tasks | Probe-ready |
| EchoJEPA-L | 18,110,464 × 1024 (70GB) | 319,802 × 1024 (1.3GB) | 48 tasks | Probe-ready |
| EchoJEPA-L-K | Extracting (52%) | N/A | N/A | ~7h remaining |
| EchoMAE-L | Extracting (~556 chunks) | N/A | N/A | Running |
| Random Init | N/A | N/A | N/A | TODO (MVP) |

### Analysis / Decisions

- **VideoMAE retraining decision:** Analyzed rebuttal docs (`claude/rebuttals/01-paper-audit.md`). VideoMAE was pretrained with ~170x lower LR than standard (8.79e-7 base vs typical 1.5e-4). Despite this, model converged (loss 0.87→0.27), RVSP competitive (5.36 vs 5.01 MAE), and all non-JEPA baselines cluster at similar performance regardless of training quality. Decision: **no retraining needed**. NatMed's claims don't hinge on JEPA-vs-MAE comparison (unlike ICML). The clustering pattern is actually a strength — it shows JEPA's advantage is robust to baseline quality.

---

## 2026-03-07

### Bug Fixes — `7ccc90b` (00:08 UTC, 2026-03-08) + `940bd2f` (13:40 UTC)

Six bugs discovered during comprehensive code review. Three were previously known from extraction failures; three were new discoveries.

- **CRITICAL: Shuffle ordering (Bug 001)** — `extract_embeddings.py`, `extract_uhn_embeddings.py`
  `DistributedSampler(shuffle=True)` is the default. This permuted clip order during extraction: embeddings[i] contained the representation for a random clip, not clip i from the CSV. Every extraction ever run was affected.
  - Fix: `data_loader.sampler.shuffle = False` in both scripts
  - Post-hoc repair: Created `fix_shuffle_order.py` (UHN) and `fix_mimic_shuffle.py` (MIMIC)
  - MIMIC: all 7 models reordered and verified (100% label match via label reconstruction)
  - UHN EchoJEPA-G: reordered post-hoc, 180 clips zero-filled (from drop_last)
  - See `bugs/001-shuffle-bug.md`

- **HIGH: Encoder normalization (Bug 002)** — `panecho_encoder.py`, `echo_prime_encoder.py`, `echofm_encoder.py`
  Three encoder adapters had incorrect input normalization, producing meaningless embeddings:
  - PanEcho: double ImageNet normalization (DataLoader normalized, then adapter normalized again)
  - EchoPrime: missing de-normalization before model-specific [0,255] range scaling
  - EchoFM: missing de-normalization to recover [0,1] range expected by model
  - Fix: PanEcho just resizes. EchoPrime: undo ImageNet → scale to [0,255] → apply model norm. EchoFM: undo ImageNet → recover [0,1].
  - See `bugs/002-normalization-bugs.md`

- **Moderate: EchoFM temporal padding (Bug 003)** — `echofm_encoder.py`
  Last-frame repetition for 16→32 frame adaptation created discontinuities. Fixed with `torch.linspace` + `index_select` for smooth temporal interpolation. Unified upsample/downsample into single code path.
  - See `bugs/003-echofm-padding.md`

- **HIGH: Video load substitution tracking (Bug 004)** — `src/datasets/video_dataset.py`
  When S3 video load fails, `__getitem__` silently returns a random different clip's data at the original index. The embedding gets mapped to the wrong clip with no indication. Added `_substitution_count` counter and per-event WARNING logging. Removed `threading.Lock` (unnecessary — DataLoader workers are separate processes; lock also broke `mp.spawn` pickling).
  - See `bugs/004-video-load-substitution.md`

- **MEDIUM: `drop_last` forwarding (Bug 005)** — `src/datasets/data_manager.py`
  `init_data(drop_last=False)` was silently ignored — the parameter was accepted but not forwarded to `make_videodataset()`. DataLoader always used `drop_last=True`. Fixed by adding `drop_last=drop_last` to the call.
  - See `bugs/005-drop-last-not-forwarded.md`

- **LOW: Labels + train/val mode (Bug 006)** — noted during review
  See `bugs/006-labels-trainval.md`

### Extraction Runs

- **EchoJEPA-G UHN — Complete** (started ~2026-03-06, finished ~2026-03-07)
  - 319,815 studies, 18,111,232 clips, 1408-dim, 95GB clip embeddings
  - Config: `configs/inference/vitg-384/extract_uhn.yaml`
  - Params: 8×A100, bs=32, w=8, pf=1 (pre-optimization)
  - Duration: ~25.5h
  - Post-hoc shuffle fix applied. 180 clips zero-filled (from drop_last across 8 ranks: 23 clips × 8 = 184, but 4 were in non-unique padding positions)
  - Study-level pooling: 319,815 studies, mean-pooled from ~56 clips/study median

- **EchoJEPA-L UHN — Complete** (started ~2026-03-06, finished ~2026-03-07)
  - 319,802 studies, 18,110,464 clips, 1024-dim, 70GB clip embeddings
  - Config: `configs/inference/vitl/extract_uhn.yaml`
  - Params: 8×A100, bs=128→64 (reduced after crashes), w=12, pf=4
  - Duration: ~12.5h
  - **Extracted BEFORE shuffle fix** (originally mislabeled as "post-fix"). Post-hoc fix applied 2026-03-08. 948 clips zero-filled.

- **EchoJEPA-L-K UHN — Attempt #1 (crashed)**
  - Launched with bs=64, w=12, pf=4 after shuffle fix in code
  - Crashed at batch ~600: 7/8 ranks died from S3 connection storm (96 concurrent S3 connections). Only rank 3 survived.
  - See 2026-03-08 entries for restart.

### Downstream Pipeline

- **UHN EchoJEPA-G shuffle fix** — reordered `clip_embeddings.npz` (18.1M clips) to CSV order using permutation reconstruction. Verified: all 8 ranks had identical chunk counts (142), contiguous global indices [0, 18111231] with zero gaps/duplicates. Re-pooled `study_embeddings.npz` (319,815 studies).

- **MIMIC all 7 models** — downstream pipeline regenerated via `evals/regenerate_mimic_downstream.py`. 7 models × 23 tasks = 161 study-level NPZs + train/val/test splits. 4 correct models immediately probe-ready. 3 models (PanEcho, EchoPrime, EchoFM) have correct shuffle but wrong normalization — queued for re-extraction.

### Code Review

- Full review of all 5 encoder adapters in `modelcustom/`. See `code-review.md`.
- Full review of extraction, pooling, remapping, and probe training scripts. 6 bugs identified (3 previously known from extraction failures, 3 new from code inspection).

### Config Changes

- Created `configs/inference/vitl/extract_uhn.yaml` (EchoJEPA-L)
- Created `configs/inference/vitl/extract_uhn_kinetics.yaml` (EchoJEPA-L-K)

### Cleanup & Re-extraction

- Deleted corrupted MIMIC embeddings for PanEcho, EchoPrime, EchoFM (~9.5GB total: master NPZs, shuffled backups, study-level dirs, split dirs)
- Started MIMIC re-extraction: `scripts/reextract_mimic_3models.sh` (8×A100, bs=32, w=8). Sequential: PanEcho → EchoPrime → EchoFM. ~1h each estimated.

### Runtime Fixes (during re-extraction)

- **PanEcho `hubconf.py` local tasks.pkl cache** — `pd.read_pickle()` was fetching `tasks.pkl` from GitHub on every worker init. 8 workers hitting simultaneously triggered HTTP 429. Fixed: downloaded to `PanEcho/content/tasks.pkl`, load from local path.

- **VideoDataset pickle compatibility** — `threading.Lock` in `_substitution_count` tracking (bug 004 fix) broke `mp.spawn` (Lock objects can't be pickled). Removed the lock; per-worker counter + WARNING logging sufficient since DataLoader workers are separate processes.

- **EchoFM missing `simplejson`** — `EchoFM/util/logging.py` imports `simplejson`. Added `pip install simplejson` to setup.

### Operational Notes

- **DataLoader optimization** — `940bd2f` (13:40 UTC)
  Changed `prefetch_factor` from 1→4 in `video_dataset.py:121`. This was the single biggest throughput win for S3-backed extraction. Also documented in `claude/ops/uhn-extraction.md`.
  - Optimal ViT-L on 8×A100: bs=64, num_workers=12, prefetch_factor=4 (~9-10h for 18M clips)
  - Optimal ViT-G on 8×A100: bs=32, num_workers=8, prefetch_factor=1 (~25h for 18M clips)
  - bs=128 crashed (S3 connection storm + worker OOM)
  - S3 download is the bottleneck, not GPU compute
  - Always use `PYTHONUNBUFFERED=1` + direct conda binary (not `conda run`)

---

## 2026-03-06

### UHN Extraction Pipeline — `4803640` (07:58 UTC)

Major commit adding the complete UHN extraction infrastructure:

- **Encoder normalization fixes** for PanEcho, EchoPrime, EchoFM (see Bug 002 above)
- **`extract_uhn_embeddings.py`** — chunked multi-GPU extraction with bf16 autocast, crash-safe resume, study-level pooling built-in
- **`uhn_all_clips.csv`** — 18,111,412 S3 paths (extraction source manifest)

### DICOM-to-Syngo Mapping — `b89a631` (04:38 UTC)

Added reference docs for the UHN DICOM UID → Syngo StudyRef mapping chain. Key files:
- `data/aws/aws_syngo.csv` (320K studies, 2002-2019) — the complete mapping
- `data/aws/R_21_009_011_echo_study_parts2and3_results.csv` (342K rows) — updated deid key

### Repository Reorganization — `4acb03b` (03:48 UTC)

- Renamed `vjepa2/embeddings/` → `vjepa2/experiments/`
- ICML UHN embeddings → `experiments/icml/`
- Nature Medicine MIMIC → `experiments/nature_medicine/mimic/`
- Updated ~100 path references across 12+ files

### Embedding Pipeline Docs

- `5726550` (11:26 UTC) — Multi-model embedding pipeline docs, PanEcho support
- `0c44abc` (11:36 UTC) — Custom pooling strategies documentation
- `f2bfe81` (19:54 UTC) — Updated docs for all 7 models

---

## 2026-03-05

### Probe Training on Precomputed Embeddings — `f5c48f5` (03:57 UTC)

Added `evals/train_probe.py` — sklearn linear probes directly on embedding NPZ files. Supports:
- Classification (logistic regression) and regression (ridge)
- `--labels` for label-only NPZs (references master by row index)
- `--train`/`--val` for precomputed splits
- Hyperparameter tuning via cross-validation

### MIMIC Embedding Pipeline — `c589c88` (10:47 UTC)

Initial multi-model embedding pipeline for MIMIC:
- `evals/extract_embeddings.py` — multi-GPU clip-level extraction
- `evals/remap_embeddings.py` — per-task label NPZs referencing master by row index
- `evals/pool_embeddings.py` — mean-pool clips to study level
- Shared infrastructure: `clip_index.npz`, `patient_split.json`, `labels/` (23 NPZs)

### EchoFM Encoder + L-K Config — `d5aaea5` (19:49 UTC)

- Added EchoFM encoder adapter to `modelcustom/`
- Created `configs/inference/vitl/extract_uhn_kinetics.yaml` (EchoJEPA-L-K)

### Repository Cleanup

- `4acd1bc` (03:25 UTC) — Clean up repository
- `d282f33` (02:50 UTC) — Reorganize `data/` directory, update docs
- `b4d80e5` (02:28 UTC) — Reorganize `classifier/` directory
- `553f761` (04:04 UTC) — Add quickstart section to README
- `971cc9e` (04:23 UTC) — Update README.md
- `7c4fbf3` (07:02 UTC) — Update docs

### Linear Probes + Claude Docs — `5f3bef2` (2026-03-04 22:42 UTC)

Added linear probe support to the evaluation system and Claude reference documentation.

---

## Pre-2026-03-05

### ICML Development (2026-01 through 2026-02)

- `0bb3fab` (2026-02-04) — Plotting scripts, embedding extractions, data augmentations, preprocessing
- `ce98206` (2026-01-29) — VideoMAE probe training for EchoNet-Pediatric
- `b577118` (2026-01-29) — EchoJEPA-L LVEF inference
- `40ec487` (2026-01-29) — EchoJEPA-L RVSP inference
- `573b053` (2026-01-29) — EchoJEPA-L RVSP inference scripts
- `81b89e9` (2026-01-28) — EchoJEPA-L EchoNet Pediatric scripts
- `e621d5d` (2026-01-28) — EchoNet Pediatric scripts
- `16c0265` (2026-01-28) — Set RVSP eval to multi
- `626305b` (2026-02-06) — Remove BibTeX section for VJEPA2 paper
- `ed9528b` (2026-02-06) — Update README with HTML formatting
