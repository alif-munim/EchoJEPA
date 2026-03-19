# Changelog — EchoJEPA Codebase (`vjepa2/`)

Comprehensive record of all code changes, bug fixes, extraction runs, infrastructure work, and operational events in the `vjepa2` repository. For Nature Medicine manuscript-level progress (data pipeline, planning, writing), see `uhn_echo/nature_medicine/context_files/dev/changelog.md`.

**Format:** Each entry includes commit hash (where applicable), timestamp, and category. Entries without commits are operational events (extraction launches, crashes, verifications) that don't produce code changes but are critical for reproducibility.

---

## 2026-03-19 (Session 21)

### Manuscript: B-mode vs All-Views Distinction Clarified

Systematic edit to `sn-article.tex` clarifying which tasks use B-mode-only input (hemodynamic/cross-modal claims) vs all available echo views (RV mechanics, standard benchmarks, trajectory, outcomes). Six edits across abstract, introduction, results sections 2.2 and 2.3, methods, and discussion.

**Key changes:**
- **Section 2.3 (RV mechanics):** Removed RVSP from this section (was duplicated from §2.2). Added explicit paragraph stating RV probes use all available echo views. Updated TAPSE to pred-avg R²=0.633 (from single-clip 0.537). Cross-references §2.2 for RVSP.
- **Section 2.2 (Hemodynamics):** Updated RVSP to pred-avg R²=0.504 (from 0.463). Added all 5 model comparisons.
- **Abstract + Introduction:** Added "from all available echo views" qualifier to RV mechanics mentions. Added TAPSE R²=0.633 to abstract.
- **Methods:** Added new `\paragraph{B-mode input restriction for hemodynamic tasks}` with clinical motivation (POCUS accessibility), literature citations (Akkus 2021 review, Zhang 2018 multi-pathology, plus existing Holste/EchoNet-Dynamic/Hughes refs), and explicit task-level inventory of B-mode-only vs all-views tasks.
- **Discussion:** Separated B-mode hemodynamic claims (RVSP R²=0.504) from all-views RV function (TAPSE R²=0.633).
- **Bibliography:** Added Akkus et al. 2021 (`akkus_ai_echo`) and Zhang et al. 2018 (`zhang_multipathology`).

---

## 2026-03-18 (Session 20)

### Bugs 008, 009, 010: Inference Pipeline Debugging & Fixes

**Bug 008 — Probe checkpoint never loaded during inference (CRITICAL):**
- `run_lvef_pred_avg.sh` generated YAML with `probe_checkpoint:` but omitted `resume_checkpoint: true`
- eval.py line 493 gates loading on `resume_checkpoint` flag; without it, probes run with random Xavier weights
- Symptom: R²=0.006, Z-score MAE=8.73 (8+ standard deviations from truth = random)
- Fix: Added `resume_checkpoint: true` to YAML template
- See `claude/dev/bugs/008-inference-probe-not-loaded.md`

**Bug 009 — /dev/shm exhaustion causes silent DDP worker death (HIGH):**
- 143GB/144GB of `/dev/shm` consumed by 40+ orphaned `multiprocessing.spawn` workers from previous crashed runs
- New DDP workers failed to allocate shm, died silently; surviving workers finished, parent exited 0
- Symptom: inference "completing" in 60-90 seconds (impossible for 266K clips), only 2-3 of 4 workers visible
- Fix: orphan cleanup between models, reduced num_workers 8→4, G batch_size 192→128
- See `claude/dev/bugs/009-shm-exhaustion-silent-ddp-death.md`

**Bug 010 — pkill orphan cleanup kills concurrent DDP jobs (HIGH):**
- Bug 009's fix used `pkill -f "multiprocessing.spawn"` which kills ALL spawn workers machine-wide
- LVEF pred avg cleanup between models killed TAPSE retrain's DDP workers on separate GPUs
- TAPSE G died at epoch 13/15, process exited with "56 leaked semaphore objects"
- Timing proof: LVEF G log_r0.csv written at 01:44, TAPSE died at 01:45:15
- Fix: replaced `pkill` with ppid=1 filtering — only kills orphaned workers (adopted by init)
- Applied to all 3 scripts: `run_lvef_pred_avg.sh`, `run_pred_avg.sh`, `run_uhn_probe.sh`
- See `claude/dev/bugs/010-pkill-kills-concurrent-jobs.md`

**Bug 012 — Resume logic skips inference on stale output dir (HIGH):**
- With `val_only: true` + `resume_checkpoint: true`, eval.py checks output dir for existing logs
- Stale header-only `log_r0.csv` from a previous failed run causes eval.py to think inference is already done
- Exits 0 silently — no warning, no results produced
- Impact: LVEF pred avg skipped L-K and PanEcho entirely; script reported success
- Fix: pred avg scripts now clear stale output dirs before each inference run (safe — no training state to preserve)
- See `claude/dev/bugs/012-resume-skips-inference-on-stale-output.md`

**Bug 011 — `rm -f /dev/shm/torch_*` cleanup kills concurrent jobs (HIGH):**
- Bug 009's fix included `rm -f /dev/shm/torch_*` between model runs to clean orphaned shm files
- This indiscriminately deletes ALL torch shm files, including those backing live DataLoader workers in concurrent jobs
- TAPSE G retrain (GPUs 4-7) crashed 3 times while LVEF pred avg (GPUs 0-3) transitioned between models
- Each crash correlated exactly with an LVEF model transition: G→L at 01:44, L→EP at 02:40
- Fix: removed `rm -f /dev/shm/torch_*` from all 3 scripts; ppid=1 process kill is sufficient
- Also fixed residual Bug 010 regression: `run_uhn_probe.sh` error paths still had unfiltered `pkill`
- See `claude/dev/bugs/011-shm-file-cleanup-kills-concurrent-jobs.md`

### Bug 013: Local `import os` Shadows Module Scope, Breaks study_predictions Save

- `run_one_epoch()` in `eval.py` had a conditional `import os` at line 980 (inside `if predictions_save_path`)
- Python treats `os` as local to the entire function → the study_predictions save block at line 1188 gets `UnboundLocalError` when the conditional import doesn't execute
- Symptom: `R²/Pearson computation failed: cannot access local variable 'os'` warning, but R²/Pearson values are actually correct — the error is in the study_predictions save, caught by a broad `except`
- Impact: LVEF G pred avg completed with correct metrics but no `study_predictions.csv` saved
- Fix: removed redundant `import os` at line 980; module-level import at line 8 is sufficient
- Required restarting the full 5-model pred avg pipeline
- See `claude/dev/bugs/013-os-import-shadows-module-scope.md`

### Generic Prediction Averaging Script

**Updated `scripts/run_pred_avg.sh`** (existed but was missing critical fixes):
- Added `resume_checkpoint: true` (Bug 008 fix)
- Added ppid=1-filtered orphan cleanup between models (Bug 009+010 fix)
- Fixed `NUM_TARGETS=0` → `NUM_CLASSES` for classification tasks
- Fixed EchoPrime batch size 64 → 256
- Added `cd $REPO`, `LD_LIBRARY_PATH`, `MASTER_PORT` export
- Auto-detects task type (regression/classification), view filtering, z-score params
- Usage: `bash scripts/run_pred_avg.sh <task>` (same interface as `run_uhn_probe.sh`)

### AV Mean Gradient Pred Avg — Invalid Results Discovered

All 5 `aov_mean_grad-*-predavg` output dirs contain results from BEFORE Bug 008 fix:
- G: val_mae=8.18, R²=-0.044 (random); L: mae=8.24, R²=-0.001; L-K: mae=8.33, R²=-0.009
- EchoPrime: R²=0.10 (noise from Xavier init); PanEcho: crashed (duplicate header, no data)
- **Must re-run** with fixed `run_pred_avg.sh`

### LVEF Prediction Averaging — In Progress

Running on GPUs 0-3. EchoJEPA-G complete: R²=0.778, Pearson r=0.889, Z-score MAE=4.78.
EchoJEPA-L in progress. 3 more models queued.

### TAPSE Retrain — Restarted

Crashed due to Bug 010 (killed by LVEF pred avg's orphan cleanup). Restarted on GPUs 4-7 with
fixed `run_uhn_probe.sh`. G resuming from epoch 13/15.

### Checkpoint Inventory

Complete (5/5 best.pt): `lvef`, `tr_severity`, `aov_mean_grad`, `trajectory_lvef_onset`, `trajectory_lvef_v1`
Partial: `rvsp` (3/5), `aov_vmax` (2/5), `tapse` (1/5 retraining), `ar_severity` (1/5), `trajectory_lvef` (3/5)

---

## 2026-03-16 (Session 19)

### Bug 007: Checkpoint Loss Fix + Retroactive Archive

**Incident:** All probe checkpoints for LVEF (5), TAPSE (5), MR severity (5), AS severity (5) discovered missing from eval output dirs. AV Vmax G/L/L-K logs also overwritten. 20 complete runs lost, 3 partially lost. Root cause of deletion unknown (no `rm` in any script version, bash history, or Claude session logs). Training logs in `logs/` confirm runs completed successfully. Key issue: no backup mechanism in `run_uhn_probe.sh` meant single point of failure.

**Fix (`scripts/run_uhn_probe.sh`):**
- Added `archive_model()` function: copies best.pt + log_r0.csv + latest.pt to `checkpoints/probes/{task}/{model}/`, pushes best.pt + log_r0.csv to S3
- Archive runs after every model (on completion and on skip)
- Added `best.pt` existence verification after training
- Fixed `is_complete()` to respect `FRESH=true` mode
- Added `NO_S3` env var, removed dead `echomae` case
- `ARCHIVE_DIR=checkpoints/probes`, `S3_PREFIX=s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts/checkpoints/probes`

**Retroactive archive:** 19 surviving checkpoints (tr_severity 4, aov_vmax 2, trajectory_lvef 3, trajectory_lvef_onset 5, trajectory_lvef_v1 5) backed up to local + S3.

**Retraining needed:** LVEF (5), TAPSE (5), MR sev (5), AS sev (5), AV Vmax G/L/L-K (3) = 23 runs.

See `claude/dev/bugs/007-checkpoint-loss.md` for full details.

---

## 2026-03-12 (Session 17)

### UHN Probe CSVs, Trajectory CSVs, MIMIC Fix, and Phase 1 Run Scripts

**UHN all-clip probe CSVs built (47 tasks):**
- `experiments/nature_medicine/uhn/build_probe_csvs.py --all` — builds train/val/test CSVs for every label NPZ
- Loads `study_to_clips_index.pkl` (cached study → clips mapping from 18.1M S3 paths)
- ALL clips per study emitted. DistributedStudySampler handles 1-per-study selection at training time
- Regression targets Z-scored using training-set mean/std; `zscore_params.json` saved per task
- 47 tasks × 3 splits = 141 CSVs at `experiments/nature_medicine/uhn/probe_csvs/{task}/`

**UHN view-filtered CSVs built (41 tasks):**
- `experiments/nature_medicine/uhn/build_viewfiltered_csvs.py --all` — joins all-clip CSVs with view/color classifier predictions
- 41 view-filtered tasks + 6 unfiltered tasks (cardiac_rhythm, gls, disease_dcm/endocarditis/stemi/takotsubo)
- View filters: task-specific echo views (e.g., A4C for TAPSE). B-mode filter for hemodynamics (MR/AS/TR severity).
- 41 tasks × 3 splits = 123 filtered CSVs (`train_vf.csv`, `val_vf.csv`, `test_vf.csv`)

**UHN trajectory CSVs built (5 tasks):**
- New script: `experiments/nature_medicine/uhn/build_trajectory_csvs.py`
- Builds train/val/test CSVs for 5 delta-prediction tasks: trajectory_lvef, trajectory_tapse, trajectory_lv_mass, trajectory_rv_sp, trajectory_mr_severity
- 1 random clip from study_1 per pair; view-filtered using same view definitions as base measurement
- Standard DataLoader (no study_sampling) — each pair is one training example
- `pairs_metadata.json` saved per task for future multi-clip aggregation

**MIMIC probe CSV regression bug fix:**
- `experiments/nature_medicine/mimic/build_probe_csvs.py` — `int(lbl)` was destroying float regression labels (creatinine 0.7→0, troponin 0.01→0)
- Fixed: auto-detect regression vs classification, Z-score normalize regression labels, save `zscore_params.json`
- All 23 MIMIC CSVs rebuilt with correct labels

**Phase 1 run scripts built:**
- `scripts/run_uhn_probe.sh` — Generic single-task probe runner for any UHN task
  - Auto-detects: task_type (regression/classification), num_classes, target_mean/std, view filtering (train_vf.csv vs train.csv), study_sampling (false for trajectory_* tasks)
  - Runs 5 models sequentially: echojepa-g, echojepa-l, echomae, echoprime, panecho
  - HP grid: 5 LRs × 4 WDs = 20 heads, 20 epochs, d=1 attentive probe
  - Supports `--models` and `--epochs` overrides
- `scripts/run_phase1.sh` — Phase 1 orchestrator: 18 tasks organized by group (rv, hemodynamics, standard, disease)
  - Supports `--group` to run subsets and `--models` for model subsets
  - Usage: `nohup bash scripts/run_phase1.sh 2>&1 | tee logs/phase1_*.log &`

---

## 2026-03-11 (Session 16)

### View-Filtered Training Pipeline and Val Sampler Fix

**Val loader fix:** `study_sampling` was only passed to the train loader in `eval.py`. Val iterated all 815K clips/epoch instead of 13K. Fixed by adding `study_sampling=study_sampling` to the val `make_dataloader()` call and `val_sampler.set_epoch(epoch)` in the epoch loop.

**View-filtered training (Decision 03 resolved):** For view-specific tasks, pre-filter training CSVs to contain only task-relevant views. Eliminates wasted gradient steps on uninformative clips (81% non-A4C for TAPSE). DistributedStudySampler still picks 1 clip/study/epoch from the filtered set.

**New files:**
- `experiments/nature_medicine/uhn/build_viewfiltered_csvs.py` — General-purpose script to join all-clip CSVs with view/color classifier predictions and produce filtered CSVs. Defines `TASK_FILTERS` dict mapping each task to (allowed_views, bmode_only). Supports `--task`, `--all`, `--views`, `--bmode_only`, `--list`.
- `experiments/nature_medicine/uhn/probe_csvs/{task}/train_vf.csv` — View-filtered training CSVs (5 RV tasks built)
- `experiments/nature_medicine/uhn/probe_csvs/{task}/val_vf.csv` — View-filtered validation CSVs
- `experiments/nature_medicine/uhn/probe_csvs/{task}/test_vf.csv` — View-filtered test CSVs
- `experiments/nature_medicine/uhn/probe_csvs/{task}/viewfilter_meta.json` — Filter metadata (views, bmode flag, source predictions)

**Modified files:**
- `evals/video_classification_frozen/eval.py` — Added `study_sampling=study_sampling` to val make_dataloader; added `val_sampler.set_epoch(epoch)` in epoch loop
- `scripts/run_uhn_tapse.sh` — Updated to use `train_vf.csv`/`val_vf.csv` (A4C-filtered) instead of unfiltered CSVs

**View-filtered CSVs built (5 RV tasks):**

| Task | Filter | Train clips | % kept | Studies kept |
|------|--------|-------------|--------|--------------|
| tapse | A4C | 281K | 18.4% | 25,337/25,737 |
| rv_fac | A4C | 80K | 19.3% | 6,398/6,714 |
| rv_sp | A4C+Subcostal | 392K | 26.2% | 24,852/25,174 |
| rv_function | A4C+Subcostal+PLAX | 2.1M | 39.4% | 86,113/91,872 |
| rv_size | A4C+Subcostal+PLAX+PSAX | 2.2M | 60.1% | 57,230/61,422 |

**Decision docs updated:**
- `decisions/03-training-sampling.md` — OPEN → DECIDED (view-filtered training for view-specific tasks)
- `decisions/04-view-task-mapping.md` — OPEN → DECIDED (per-task filter definitions implemented)
- `decisions/README.md` — Status updated

**Run launched:** TAPSE 5-model d=1 attentive probe with A4C-filtered CSVs. Log: `logs/tapse_5model_vf_*.log`

---

## 2026-03-11 (Session 15)

### DistributedStudySampler and MIMIC Probe CSV Pipeline

Implemented per-epoch random clip selection for study-level tasks and built the MIMIC probe CSV pipeline for all 23 tasks.

**New files:**
- `src/datasets/study_sampler.py` — `DistributedStudySampler`: groups CSV rows by study_id, picks 1 random clip per study per epoch, distributes across ranks. Study ID extracted from MIMIC S3 paths via regex `/s(\d+)/\d+_\d+\.mp4`, with parent-directory fallback for UHN.
- `experiments/nature_medicine/mimic/build_probe_csvs.py` — builds train/val/test CSVs for all 23 MIMIC tasks from `clip_index.npz` + `labels/*.npz` + `patient_split.json`. All splits contain ALL clips per study (sampler handles 1-per-study selection at training time).
- `configs/eval/vitg-384/nature_medicine/echojepa_g_mortality_1yr.yaml` — first Nature Medicine probe config: EchoJEPA-G, d=1 attentive probe, 35 epochs, 20-HP grid, `study_sampling: true`.

**Pipeline integration (4 files modified):**
- `src/datasets/video_dataset.py` — added `study_sampling` param to `make_videodataset()`, uses `DistributedStudySampler` when True
- `src/datasets/data_manager.py` — added `study_sampling` param to `init_data()`, passes through to `make_videodataset()`
- `evals/video_classification_frozen/eval.py` — parses `study_sampling` from data config, applies only to training dataloader

**MIMIC CSVs generated:** 23 tasks × 3 splits. All CSVs contain all clips per study (~72 clips/study average). Example: mortality_1yr train has 5,145 studies / 372,678 clips.

**Old pipeline artifacts archived (not deleted):**
- `experiments/nature_medicine/mimic/archived/` — 56 GB (master NPZs, study-level, splits, zips)
- `experiments/nature_medicine/uhn/archived/` — 1.2 TB (embedding dirs, split dirs)

**Documentation updated:** CLAUDE.md, probe-system.md, plus 13 other docs to mark old NPZ pipeline as superseded.

---

## 2026-03-09 (Session 14)

### UHN Linear Probe Training — EchoJEPA-G and EchoJEPA-L

Trained frozen linear probes on all available UHN study-level embeddings: 26 classification + 21 regression + 5 trajectory tasks × 2 models = 104 jobs. Script: `scripts/run_uhn_probes.py`. Protocol: StandardScaler → HP grid search on val split → evaluate on held-out test split.

- **Classification**: LogisticRegression(C, max_iter=2000, solver='lbfgs'), C ∈ {1e-4, 1e-3, 1e-2, 0.1, 1, 10}
- **Regression**: Ridge(alpha), alpha ∈ {1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100}
- **Trajectory**: Ridge on concat(emb_1, emb_2) → predict delta between paired studies

**BLAS thread contention fix:** Initial run with `--workers 4` stalled — 4 workers × ~24 BLAS threads = 96 threads saturating the 96-core machine. Fixed with `OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8` and `--workers 8` (8 × 8 = 64 threads, well under limit).

**EchoJEPA-G results (in-domain):**
- Classification mean AUC: 0.874 (26 tasks). Top: AS severity 0.947, pericardial effusion 0.940, LV systolic function 0.936, LV cavity size 0.934, diastolic function 0.918
- Regression mean R²: 0.625 (21 tasks). Top: ESV 0.853, EDV 0.798, LVEF 0.775, LV mass 0.738, LA vol 0.726
- Trajectory: LVEF delta R²=0.456, MR severity R²=0.234, TAPSE R²=0.129
- Rare diseases detectable from frozen representations: amyloidosis 0.835, HCM 0.877, STEMI 0.828, takotsubo 0.815

**EchoJEPA-L results (out-of-domain — complete failure):**
- All 26 classification tasks: AUC ~0.50 (chance)
- All 21 regression tasks: R² ≤ 0 (worse than mean predictor)
- All 5 trajectory tasks: R² < 0
- Root cause: L pretrained on ~7K MIMIC studies → extreme embedding concentration on UHN (mean pairwise cosine 0.998, var/dim ratio 0.0005 vs G's 0.293). Confirmed L works in-domain on MIMIC (HF AUC 0.761).

**Output:** `results/probes/nature_medicine/uhn/all_results.json`, per-task metrics in `results/probes/nature_medicine/uhn/{model}/{task}/test_metrics.json`. Runtime: 58.3 min on 96-core CPU instance.

---

## 2026-03-09 (Session 13)

### MIMIC Zip Regeneration and S3 Upload

Regenerated all 8 MIMIC zip files with current verified embeddings (now including S3 path provenance columns) and uploaded to `s3://echodata25/nature_medicine/mimic/`.

**Why re-zipped:** Previous zips (from March 5) contained pre-shuffle-fix, pre-normalization-fix embeddings without `paths` arrays. After the shuffle fix (Bug 001), normalization fix (Bug 002, PanEcho/EchoPrime/EchoFM re-extracted March 8), and path injection (Session 9), the master NPZ files now contain `['embeddings', 'labels', 'paths']` keys with verified end-to-end alignment.

**Zip contents** (each model, ~141 files per zip):
- Master embedding NPZ with S3 path column
- `clip_index.npz`, `patient_split.json` (shared)
- `labels/*.npz` (23 task label files)
- `{model}_study_level/*.npz` (23 study-level files)
- `{model}_splits/{task}/train|val|test.npz` (23 × 3 split files)
- `data/csv/nature_medicine/mimic/*.csv` (23 source CSVs)

**Script:** `scripts/rezip_mimic.py` — ZIP_STORED (no compression), atomic writes via `.tmp` + `os.replace()`.

**Files on S3** (`s3://echodata25/nature_medicine/mimic/`):

| File | Size |
|------|------|
| `echojepa_g_mimic_all.zip` | 5.0 GiB |
| `echojepa_l_mimic_all.zip` | 3.9 GiB |
| `echojepa_l_kinetics_mimic_all.zip` | 3.9 GiB |
| `echomae_mimic_all.zip` | 3.9 GiB |
| `panecho_mimic_all.zip` | 3.2 GiB |
| `echoprime_mimic_all.zip` | 2.5 GiB |
| `echofm_mimic_all.zip` | 3.9 GiB |
| `mimic_covariates.zip` | 730.5 KiB |

Presigned URLs (7-day TTL, expire ~2026-03-16) generated and saved to `uhn_echo/nature_medicine/context_files/dev/embedding-status.md`.

### UHN Path Injection into Existing NPZs

Injected S3 paths from `uhn_clip_index.npz` into 4 UHN clip embedding files (previously contained only `embeddings` array). Required loading full arrays into RAM (70-105 GB each) sequentially on 1.1 TiB system. Originals backed up as `.no_paths_backup`.

| Model | Shape | Paths injected | New size |
|-------|-------|---------------|----------|
| EchoJEPA-G | 18,111,232 × 1408 | paths[:18111232] | 105.8 GB |
| EchoJEPA-L | 18,110,464 × 1024 | paths[:18110464] | 78.0 GB |
| EchoJEPA-L-K | 18,111,416 × 1024 | 18,111,412 real + 4 padding | 78.0 GB |
| EchoMAE | 18,111,416 × 1024 | 18,111,412 real + 4 padding | 78.0 GB |

### UHN Path-Embedding Verification (4 models)

Extended verification from Session 9 (which covered G and L only) to include L-K and EchoMAE on UHN. Script: `scripts/verify_uhn_paths.py` — uses `MmapNpzReader` (memory-mapped random access into 70-105GB NPZ files).

| Model | Mean Match | Mean Mismatch | Gap | Verdict |
|-------|-----------|--------------|-----|---------|
| EchoJEPA-G | >0.95 | ~0.65 | ~0.35 | PASS |
| EchoJEPA-L | >0.99 | ~0.91 | ~0.08 | PASS |
| EchoJEPA-L-K | >0.98 | ~0.40 | ~0.58 | PASS |
| EchoMAE | >0.99 | ~0.52 | ~0.48 | PASS |

All 4 models show clear match > mismatch discrimination. Alignment chain verified end-to-end.

---

## 2026-03-08 (Session 12)

### MViT GPU Memory Leak Fix

Progressive GPU memory growth caused repeated OOMs during EchoPrime (MViT-v2-S) extraction — memory grew ~25 MB/batch (55→75 GB over 1000 batches) even with `cudnn.benchmark=False` and `expandable_segments:True`. Root cause: CUDA allocator retains freed blocks but doesn't return them to the pool without explicit prompting. Not a true tensor leak — `gc.collect()` + `torch.cuda.empty_cache()` at chunk saves dropped memory from 66 GB → 33 GB instantly.

**Changes to `evals/extract_uhn_embeddings.py`:**
- Added `import gc`
- Added `del clips, clip_indices_batch, outputs, pooled_segments, pooled, data` after each batch (explicit GPU tensor release)
- Added `gc.collect()` + `torch.cuda.empty_cache()` every 100 batches (periodic cleanup, ~2.5 GB max growth between cleanups) and at chunk saves
- Comment documents the 25 MB/batch growth rate and the 66→33 GB cleanup effect

**DataLoader Bus error fix:** Reduced `num_workers` from 12 → 8 for EchoPrime. With w=12 × 8 ranks = 96 workers, a transient Bus error (SIGBUS) killed a DataLoader worker mid-forward-pass. Shared memory (144 GB) and system RAM (1.1 TiB) were not exhausted — likely a sporadic S3/decord issue amplified by high worker count. w=8 (64 total workers) is more stable with minimal throughput loss.

**Final stable EchoPrime settings:** bs=64, w=8, pf=8, fp32+TF32, `cudnn.benchmark=False`, `expandable_segments:True`, gc every 100 batches. Memory stable at 38-43 GB (well under 80 GB limit). Throughput ~1.0-1.3 it/s. ETA ~8h.

## 2026-03-08 (Session 11)

### Extraction Performance Optimizations

Systematic optimization of `extract_uhn_embeddings.py` for fp32 models (EchoPrime) and S3-bottlenecked workloads. Combined changes yield ~2x wall-clock speedup on UHN 18M extraction.

**TF32 matmul was disabled** — the single biggest finding. PyTorch 2.6 defaults `torch.backends.cuda.matmul.allow_tf32 = False`, meaning all fp32 matmuls on A100 ran at 19.5 TFLOPS instead of ~156 TFLOPS. EchoPrime (MViT-v2-S) runs entirely in fp32 (adapter disables autocast at `echo_prime_encoder.py:147-149`), so enabling TF32 gives up to 8x throughput on matmul operations.

**Changes to `evals/extract_uhn_embeddings.py`:**
- Added `torch.backends.cuda.matmul.allow_tf32 = True` in each worker (TF32 for fp32 matmuls)
- Added `torch.backends.cudnn.allow_tf32 = True` (TF32 for cuDNN conv ops)
- `torch.backends.cudnn.benchmark = False` (explicitly disabled — see note below)
- Changed `np.savez_compressed` → `np.savez` for chunk saves (avoid compression overhead)

**Changes to `src/datasets/video_dataset.py`:**
- `prefetch_factor` increased from 4 → 8. With TF32, the GPU outruns the S3 download pipeline; deeper prefetch buffer (64 batches vs 32 per GPU) smooths S3 latency spikes. RAM cost: ~45 GB per GPU process (1.1 TiB system has headroom).

**EchoPrime UHN extraction launched** with optimized settings:
- Config: `configs/inference/vitl/extract_uhn_echoprime.yaml` (new file)
- Settings: bs=128, num_workers=8, prefetch_factor=8, fp32+TF32, 8×A100
- Output: `experiments/nature_medicine/uhn/echoprime_embeddings/`

**Measured impact** (EchoPrime, 8×A100-80GB, 18.1M clips):

| Setting | bs=32, pf=4, no TF32 | bs=64, pf=8, TF32, bench ON | bs=64, pf=8, TF32, bench OFF, w=12 |
|---------|----------------------|-----------------------------|--------------------------------------|
| Non-stall rate | ~1.1-1.5 s/batch | ~0.83 s/batch | ~0.71 s/batch (1.41 it/s) |
| S3 stall peaks | ~2.3 s | 2.5-3.0 s | 2.0-2.8 s |
| GPU util | intermittent | 94-98% sustained | 87-98% sustained |
| GPU memory | ~40/80 GB | 47→80 GB (OOM!) | ~55-58/80 GB (stable) |
| Total batches | 70,748 | 35,374 | 35,374 |
| Est. wall time | ~20-25h | OOM at batch 504 | ~7-8h |

**bs=128 OOMs for MViT-v2-S** — MViT `_add_rel_pos` requires a 3.66 GiB intermediate tensor at bs=128, exceeding 80 GB. Max safe batch size for EchoPrime is bs=64.

**cuDNN benchmark is HARMFUL for MViT** — `cudnn.benchmark=True` caches workspace memory for every unique (layer, input_size) combination. MViT's multi-scale pooling attention has many unique configurations, causing GPU memory to grow from ~43 GB → 73 GB → 80 GB over ~500 batches, eventually OOMing on the same `_add_rel_pos` 3.66 GiB allocation. Disabling it keeps memory stable at ~55-58 GB with no throughput loss (1.41 it/s without vs 1.35 it/s with). **Do not enable cuDNN benchmark for MViT/EchoPrime.**

**`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** added as env var for the extraction launch. Reduces CUDA memory fragmentation ("reserved but unallocated" pool). Recommended for all long-running extraction jobs.

Key insight: S3 download latency is the true bottleneck. GPU-side optimizations (TF32) reduce per-batch compute time, but the gains are partially eaten by S3 stalls when the prefetch buffer drains. Increasing prefetch_factor and num_workers was critical for sustaining GPU utilization.

**Note:** TF32 benefits all models (bf16 included) by accelerating any remaining fp32 ops. cuDNN benchmark may be safe for ViT-based models (fixed attention sizes) but should be tested carefully — the MViT OOM was silent and progressive.

### EchoPrime fp32 Compatibility Fix

`extract_uhn_embeddings.py` was hardcoded to cast all models to bf16 and use bf16 autocast. EchoPrime requires fp32 (adapter disables autocast internally, normalization calls `x.float()` explicitly).

- Added `use_bf16 = not model_kwargs.get("wrapper_kwargs", {}).get("force_fp32", False)` check
- bf16 cast and autocast now conditional on `use_bf16` flag
- Config `extract_uhn_echoprime.yaml` sets `force_fp32: true` in wrapper_kwargs

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
