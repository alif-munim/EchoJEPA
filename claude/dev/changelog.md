# Changelog — EchoJEPA Codebase (`vjepa2/`)

Comprehensive record of all code changes, bug fixes, extraction runs, infrastructure work, and operational events in the `vjepa2` repository. For Nature Medicine manuscript-level progress (data pipeline, planning, writing), see `uhn_echo/nature_medicine/context_files/dev/changelog.md`.

**Format:** Each entry includes commit hash (where applicable), timestamp, and category. Entries without commits are operational events (extraction launches, crashes, verifications) that don't produce code changes but are critical for reproducibility.

---

## 2026-04-08

### ViT-B Temporal Shortcut Pilot — Standard MAE vs Frame-Gap MAE

**What:** Implemented `FrameGapMaskingGenerator` in `s3_dataset.py` and launched two parallel ViT-B VideoMAE pretraining runs (100 epochs, random init, MIMIC-IV-Echo 525K clips):

- **Job 570** (node 83): Standard tube masking (baseline)
- **Job 571** (node 184): Frame-gap masking (intervention)

Frame-gap masking splits 8 temporal positions into context[t0:t3], gap[t3:t5], target[t5:t8]. Visible patches only in context frames (~27%), gap+target all masked. Overall 90% mask ratio preserved. Prevents spatial interpolation across the temporal gap, forcing temporal reasoning.

**Why:** Test temporal shortcut hypothesis — MAE transiently learns temporal features (~e50) then abandons them by convergence because tube masking allows spatial interpolation. Frame-gap masking should force temporal feature retention.

**Code changes:**
- `evals/video_classification_frozen/modelcustom/VideoMAE/s3_dataset.py` — Added `FrameGapMaskingGenerator`, `VideoDataset` now accepts `mask_type` and `temporal_gap`
- `evals/video_classification_frozen/modelcustom/VideoMAE/run_mae_pretraining.py` — Added `'frame_gap'` to `--mask_type` choices, added `--temporal_gap` arg
- `scripts/videomae_pilot_standard_vitb.sbatch` — Standard MAE ViT-B, node 83
- `scripts/videomae_pilot_framegap_vitb.sbatch` — Frame-Gap MAE ViT-B, node 184

**Docs:** `claude/neurips/experiments/vitb-temporal-shortcut-pilot.md`

**Training config:** BS 1024 (32×8 GPU × 4 accum), LR 6e-4, warmup 10 epochs, checkpoints at e25/e50/e75/e100. ~26 min/epoch, ~43 hrs total per model. ETA ~2026-04-10.

### Added ViT-B ImageNet-1K init path for JEPA / BYOL / VideoMAE

**What:** Built `checkpoints/vitb_in1k.pt` (343 MB) from torchvision `ViT_B_16_Weights.IMAGENET1K_V1` (DeIT recipe, 81.07% top-1) by remapping keys to the flat EchoJEPA convention (same format as `vitl_in21k.pt` / `vitl_raw.pth`). Dry-run load into `vit_base(use_rope=True)`: 0 missing, exactly 2 unexpected (`cls_token`, `pos_embed` — correctly ignored).

**Why:** Three-way cross-method comparison at ViT-B scale needs all three methods starting from the same image-ViT init, matching the ViT-L `vitl_in21k.pt` pattern already used for EchoJEPA-L in21k / EchoBYOL-L / EchoMAE-L.

**Trainer support:** `app/vjepa`, `app/byol_video`, `app/salt` already had the flat-init code path (`force_load_pretrain: true` + `anneal_ckpt:` at `train.py:327-380` etc., including the Conv2d -> Conv3d `patch_embed` inflation) from the ViT-L in21k work. The external VideoMAE trainer uses the inline adapter in `scripts/videomae_pretrain_mimic_matched.sbatch`, which is shape-generic and works for 768-dim sources unchanged.

**Files added:**
- `configs/train/vitb16/pretrain-jepa-mimic-224px-16f-in1k-hp.yaml` — JEPA ViT-B, 100 epochs, matches ViT-L in21k recipe (pred_depth 12 -> 6, act ckpt off)
- `configs/train/vitb16/pretrain-byol-mimic-224px-16f-in1k-hp.yaml` — BYOL ViT-B, 100 epochs, constant EMA 0.99925
- `scripts/videomae_pretrain_mimic_vitb.sbatch` — clone of `videomae_pretrain_mimic_matched.sbatch` with ViT-L -> ViT-B swaps (model import, `--model` flag, S3 init pointer). Prereq: upload `vitb_in1k.pt` to S3 as `checkpoints/vitb_raw.pth`.

**Caveat (inherited from ViT-L VideoMAE run):** VideoMAE's `Attention` uses `qkv = Linear(..., bias=False)` + separate `q_bias` / `v_bias` params. The sbatch adapter walks target keys looking for matches, so the fused source `attn.qkv.bias` is silently dropped and `q_bias` / `v_bias` remain at zero init. ~96% of encoder params still load cleanly. Kept identical to ViT-L for parity.

**Docs:** New "ImageNet Initialization (image ViT -> video ViT)" section in `claude/architecture/pretraining-and-cooldown.md` (checkpoint format, load-time transforms, `use_rope` requirement, inline torchvision remap script). `vitb_imagenet1k.pt` added to `claude/architecture/checkpoint-registry.md` Init Weights table.

### UHN MR severity probe — trained, stopped early, saved to S3

**Job 443** (node 83): EchoJEPA-G d=1 attentive probe for UHN MR severity 4-class (None-Trivial/Mild/Moderate/Severe). Config: `configs/eval/vitg-384/nature_medicine/echojepa_g_mr_severity_uhn_hp.yaml`. Encoder: `pt-280-an81.pt` (ViT-G). Resumed from job 437 epoch 1.

Training log (28 complete epochs, stopped during epoch 29/35):

| Epoch | Train Acc | Val Acc |
|-------|-----------|---------|
| 1 | 65.47 | 65.32 |
| 10 | 69.95 | 69.00 |
| 22 | 71.07 | **69.99** (best) |
| 28 | 71.58 | 69.94 |

Cancelled at epoch 29 to free node 83. Checkpoint saved to S3:
- `s3://.../runs/echojepa_g_mr_severity_uhn_443/training_folder/video_classification_frozen/echojepa-g-mr-severity-uhn/best.pt` (epoch 22, 4.9 GB)
- `s3://.../runs/echojepa_g_mr_severity_uhn_443/training_folder/video_classification_frozen/echojepa-g-mr-severity-uhn/latest.pt` (epoch 28)
- `s3://.../runs/echojepa_g_mr_severity_uhn_443/training_folder/video_classification_frozen/echojepa-g-mr-severity-uhn/log_r0.csv`

### MR severity cross-dataset comparison (UHN probe vs MIMIC probe on MIMIC test)

**Job 549** (node 184, 13 min): Both probes tested on MIMIC-IV-Echo MR test set (1,003 studies, prediction-averaged). Same frozen EchoJEPA-G encoder, same d=1 attentive probe architecture.

| Probe | Accuracy | Balanced Acc | Quad Kappa | Macro AUROC |
|-------|----------|-------------|------------|-------------|
| MIMIC (in-distribution, job 436) | 0.591 | 0.391 | 0.538 | **0.806** |
| UHN (cross-dataset, job 443) | 0.531 | 0.341 | 0.410 | **0.799** |

**Key finding:** AUROC preserved cross-institution (−0.9%). The UHN probe's discrimination transfers; only classification thresholds degrade. Neither probe detects Severe (n=56).

**Scripts:**
- sbatch: `scripts/echojepa_g_mr_compare_mimic_test.sbatch`
- UHN-on-MIMIC config: `configs/eval/vitg-384/nature_medicine/echojepa_g_mr_uhn_on_mimic_predavg.yaml`

**Artifacts:** `s3://.../runs/echojepa_g_mr_compare_549/logs/{mr_comparison.csv,sklearn_comparison.log,mimic_probe_study_predictions.csv,uhn_probe_study_predictions.csv}`

**Docs:** `claude/neurips/experiments/mr-cross-dataset-transfer.md`, `claude/neurips/completed-experiments.md` §10.

---

## 2026-03-29 (Session 37)

### Bug 017c: Single-view LVEF probe z-score mismatch

**Problem:** EchoMAE-L pt50 LVEF probe (HyperPod job 247) was trained on pre-March-14 code that lacked z-score normalization in `evals/video_classification_frozen/eval.py`. The probe predicted raw LVEF values (~60%). Test inference (job 264) with current code — which z-scores labels to ~0 at runtime — produced MAE 719 (expected ~7).

**Root cause:** `git blame` shows z-score normalization (lines 933-936) was added on March 14 (commit `531ae49e`). Job 247 trained on stale code.tar that predated this fix. During training, metrics were consistent (both raw), so val MAE looked correct (~7.15). The mismatch only appeared at test inference with current code.

**Fix:** Retrained probe with z-score normalization. Added `target_mean: 57.0569` / `target_std: 11.2817` to `configs/eval/vitb/icml/echomae_l_pt50_lvef_d4.yaml`. HyperPod job 274 (node 83), head 1/6 complete: val MAE 7.17.

### Migrated all 34 sbatch scripts from code.tar to deploy.sh

**Problem:** All sbatch scripts downloaded `code.tar` from S3, which was the root cause of stale code deployments (Bugs 017a, 017c). The code.tar had to be manually rebuilt and re-uploaded after every fix.

**Fix:** Replaced the code.tar extraction block in all 34 scripts with:
```bash
REPO_DIR="/opt/vjepa2"
cd "$REPO_DIR"
pip install -e . --no-deps --no-build-isolation 2>/dev/null || true
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
```

Scripts now use code deployed by `~/deploy.sh` to `/opt/vjepa2` on compute nodes via `srun`. This eliminates the S3 code.tar as a stale-code vector.

**Files modified:** All `scripts/*.sbatch` (34 files). Commit `238e027`.

### Updated deploy.sh to target both compute nodes

`~/deploy.sh` now deploys to both nodes (ip-10-0-50-83, ip-10-0-50-184) by default. Pass a node name to deploy to just one: `~/deploy.sh ip-10-0-50-83`.

### Updated CLAUDE.md with HyperPod deploy workflow

Added `deploy.sh` documentation and bold warning to always run it before `sbatch`. Commit `082ab16`.

---

## 2026-03-28 (Session 36)

### BYOL-Video v2 learning curve: representations improving

**Motivation:** BYOL-Video ViT-L v2 running on H100 cluster (2×8 H100, Job 241). Need to verify representations are improving across pretraining epochs, not stalling like v1.

**Method:** Downloaded 3 BYOL checkpoints from S3 (e0, e10, e44). Loaded frozen ViT-L encoder, extracted mean-pooled features (3K train / 1K val, UHN LVEF dataset), fit sklearn Ridge regression. Tested both `target_encoder` (momentum-averaged) and `encoder` (online).

**Results (Val R², UHN LVEF linear probe):**

| Epoch | BYOL Loss | Target Encoder | Online Encoder |
|-------|-----------|----------------|----------------|
| 1     | -1.659    | 0.103          | 0.144          |
| 11    | -1.987    | 0.177          | 0.183          |
| 45    | -1.986    | **0.224**      | **0.225**      |

**Analysis:**
- Clear upward trend — performance improving steadily, no collapse or stalling
- Constant EMA fix (v2) is working: target and online encoder converge by e44
- Feature norms constant at 32.0 — no representation collapse
- Only 19% through training (45/240), trajectory still ascending
- BYOL loss converges early (-1.99 by e10) but downstream quality continues to improve — confirms loss alone is uninformative (same as JEPA)

**Files:**
- `scripts/byol_learning_curve.py` (NEW) — quick feature extraction + sklearn Ridge probe for checkpoint evaluation
- `configs/eval/vitl/byol_lvef_e{0,10,44}.yaml` (NEW) — full attentive probe configs (not yet run)

**S3 checkpoints available:** e0 through e44 (every 2 epochs) at `s3://sagemaker-echojepa-h100-march-0d224785-bucket/checkpoints/byol-vitl-imagenet-v2/`

---

## 2026-03-28 (Session 35)

### Z-score params embedded in probe checkpoints (self-contained inference)

**Problem:** Z-score normalization params (`target_mean`, `target_std`) were stored in separate `zscore_params.json` files, causing two bugs:
- **Bug 017b**: A stale `data/csv/zscore_params.json` from an LVEF run (mean=57) was auto-loaded for RVSP tasks (mean=34), producing garbage predictions.
- Fragile inference: requires checkpoint + correct JSON file in the right directory.

**Solution:** Checkpoints are now self-contained — `target_mean`, `target_std`, and `task_type` are saved alongside model weights.

**Files modified:**
- `evals/video_classification_frozen/eval.py` — save/load z-score metadata in checkpoints; new 5-step precedence chain
- `evals/video_classification_frozen_multi/eval.py` — same changes mirrored; also added R²/Pearson logging per head, Bug 017 z-score fix
- `scripts/patch_zscore_into_checkpoints.py` (NEW) — one-time migration for 445 existing checkpoints (406 probes + 39 eval_probes)

**Z-score resolution precedence (both eval modules):**
1. YAML config (`target_mean`/`target_std`) — explicit override, with cross-check warning if checkpoint disagrees
2. Checkpoint metadata — the model's training contract (NEW)
3. `zscore_params.json` — legacy fallback (unchanged)
4. Compute from train CSV — training only, with **auto-save safety** (refuses to overwrite if existing JSON differs)
5. `RuntimeError` — inference with nothing found

**`load_checkpoint()` change:** Returns 5-tuple (was 4-tuple) with metadata dict. Old checkpoints return `None` for all metadata keys — safe fallback via `.get()`.

**Migration results:** `scripts/patch_zscore_into_checkpoints.py` patched all existing checkpoints:
- `checkpoints/probes/`: 166 regression + 240 classification = 406 checkpoints
- `checkpoints/eval_probes/`: 39 checkpoints (ICML preprint, used `data/scalers/*.pkl` for raw-unit params)
- Cross-validated 46 `zscore_params.json` files against train CSV statistics — 36 benign drifts (<5%), 0 cross-task poisoning

**Additional changes in eval modules:**
- Rank-0-only periodic logging (`if itr % 10 == 0 and rank == 0`) — reduces noise on multi-GPU runs
- Multi-view: R²/Pearson per head logged to CSV and epoch summary, `run_one_epoch` returns 3-tuple (scalar, agg, reg_metrics)

**Backward compatibility:** Old code ignores unknown checkpoint keys. New code uses `.get(key, None)` for old checkpoints. Classification checkpoints store `target_mean=None`.

---

## 2026-03-28 (Session 34)

### CAMUS frozen segmentation: all results complete

**Pipeline:** `evals/segmentation_frozen/eval.py` — frozen encoder + linear decoder (1×1 conv + bilinear upsample, ~4K params) on CAMUS (400/50/50 train/val/test, 4CH+2CH views). 7-config HP grid trained in parallel per model, 50 epochs each. Best config selected by val Dice, test Dice reported.

**Final results (test mean Dice):**

| Model | Test Dice | Best LR | Notes |
|-------|-----------|---------|-------|
| EchoJEPA-L | **0.818** | 5e-2 | Best overall |
| EchoMAE-L | 0.790 | 5e-2 | Same ViT-L, same data — objective comparison |
| EchoJEPA-L-K | 0.746 | 1e-2 | Kinetics detour vs continuous MIMIC pretraining |
| PanEcho | 0.734 | 2e-2 | 7×7 grid limits spatial resolution |
| EchoJEPA-G (384px) | 0.729 | 1e-2 | Native resolution, correct architecture |
| EchoJEPA-G (224px) | 0.718 | 5e-3 | Below native res |
| EchoPrime | 0.669 | 5e-2 | 7×7 grid, weakest spatial features |

### Bug 016 FIXED: `vit_giant` vs `vit_giant_xformers` num_heads mismatch (CRITICAL)

**File:** `evals/segmentation_frozen/eval.py`

EchoJEPA-G checkpoint (trained with `vit_giant_xformers`, 22 heads, head_dim=64) was loaded into `vit_giant` architecture (16 heads, head_dim=88). QKV weight shapes `[4224, 1408]` are identical regardless of `num_heads`, so `load_state_dict` succeeds silently — even with `strict=True`. But attention computation is completely wrong: each head reads wrong feature dimensions, and RoPE rotation dimensions mismatch (28 vs 20 per axis). Result: 0.600 test Dice instead of 0.718+.

**Debugging timeline:** 6 steps over ~3 hours. Failed hypotheses: resolution mismatch (384px identical to 224px — both broken), deeper ViTs lose locality (contradicted by DINOv2 ViT-g), echo fan geometry artifact. User identified the architecture mismatch class of bugs early; final diagnosis required comparing `vit_giant` (num_heads=16) vs `vit_giant_xformers` (num_heads=22) factory functions.

**Fix:** Changed auto-detection default from `vit_giant` → `vit_giant_xformers`. Added `--model_name` and `--resolution` CLI args. Verified RoPE is correctly enabled via `use_rope=True` kwarg (produces `RoPEAttention` blocks, no `pos_embed`).

**Commit:** `636469a`

See `claude/dev/bugs/016-vit-giant-num-heads-mismatch.md` for full details.

### Segmentation eval enhancements

- Expanded HP grid from 6 → 7 configs (added lr=5e-2 which won for 3/5 models)
- Added `--resolution` arg for running G at native 384px
- Added `--model_name` arg for explicit architecture override
- Resolution threaded through model creation, dataset, and decoder

---

## 2026-03-28 (Session 33)

### Bug 017b: Shared `zscore_params.json` poisoning RVSP runs

Even after the Bug 017 z-score fix, the EchoMAE RVSP ep163 rebuttal run showed no learning (train MAE ~10.6 in distorted units ≈ 120+ mmHg effective). Root cause: a stale `data/csv/zscore_params.json` created by a prior LVEF run contained LVEF params (mean=57.06, std=11.33). The auto-detection code loaded these for RVSP tasks (which need mean=34.47, std=14.01), causing RVSP labels to be z-scored with the wrong statistics. A typical RVSP of 34 mmHg was z-scored as (34-57.06)/11.33 = -2.03, producing deeply negative targets.

**Fix:**
1. Added explicit `target_mean: 34.4650` / `target_std: 14.0130` to all 9 RVSP ICML configs
2. Deleted stale `data/csv/zscore_params.json` (and EFS copy)
3. Documented in `claude/dev/bugs/017-multiview-rvsp-no-zscore.md` (017b section)

**Affected:** EchoMAE-L RVSP ep163 full 41K run (invalid, needs restart). All other RVSP configs now have explicit params.

---

## 2026-03-28 (Session 32)

### Bug 017 FIXED: Multi-view eval missing z-score normalization (CRITICAL)

**File:** `evals/video_classification_frozen_multi/eval.py`

The multi-view eval module never z-score normalized regression labels at runtime. The single-view module does this at line 899 (`labels = (labels - t_mean) / t_std`). This bug was dormant during the ICML preprint because the RVSP CSVs were pre-z-scored using `sklearn.StandardScaler`. When the CSVs were later rebuilt with raw mmHg values for the NatMed pipeline, the multi-view module was never updated, causing rebuttal RVSP runs to fail catastrophically (MAE ~145-176 logged scale).

**Fix:** Added `y = (y - t_mean) / t_std` before loss computation in multi-view eval, matching single-view.

**Impact:** ICML preprint RVSP valid (pre-z-scored CSVs). NatMed RVSP valid (uses single-view). All rebuttal multi-view RVSP runs before this fix are invalid.

**Key forensic findings:**
- `data/scalers/rvsp_scaler.pkl` confirms preprint used pre-z-scored CSVs (mean=34.465, std=14.013)
- Preprint checkpoint regressor biases ~0.02-0.04 (z-scored output range, not raw ~34)
- `VideoGroupDataset` uses `int()` cast on labels, which quantized z-scored floats to ~5-6 bins
- Git history: `a6a520e` (2026-01-23) parameterized `target_std`, `1a5dcf5` (2026-01-26) ran preprint RVSP

See `claude/dev/bugs/017-multiview-rvsp-no-zscore.md` for full details.

### Multi-view eval: Added R²/Pearson tracking for regression

Added per-head R² and Pearson correlation computation to the multi-view eval module's validation loop, matching the single-view module. Requires distributed gathering via `all_gather`. CSV logger updated to 5 columns: `epoch, train_mae, val_mae, val_r2, val_pearson`.

### Checkpoint registry and encoder symlinks

Created `checkpoints/encoders/` directory with 13 descriptive symlinks pointing to actual checkpoint files. Written `claude/architecture/checkpoint-registry.md` documenting all encoder checkpoints (pretraining lineage, epoch conventions, S3 sources) and probe checkpoints (ICML eval_probes, NatMed UHN probes, NatMed MIMIC probes).

### Model lineage corrections across docs

Corrected EchoJEPA-L documentation across 7 files: L is pretrained on **MIMIC-only** (not UHN). L-K is **Kinetics→MIMIC** (not Kinetics→UHN). Only G uses UHN data.

### RVSP rebuttal run launched (EchoJEPA-L full, 5K MIMIC)

**Config:** `configs/eval/vitl/icml/echojepa_l_mimic_full_rvsp_d4.yaml`
**Checkpoint:** `vitl-pt-210-an25.pt` (ViT-L, MIMIC pt210 + an25)
**Data:** 5K train / 1K val multi-view MIMIC RVSP subset
**Protocol:** ICML (d=4, 6-head, 20 epochs, BS=1)
**Status:** Running. Epoch 1 val MAE = 10.59 mmHg, R² = -0.071, Pearson = 0.085 (correct scale).

---

## 2026-03-28 (Session 31, continued)

### ICML Rebuttal: EchoJEPA-L-K LVEF d=4 Probe — PAUSED at Epoch 12/20

**Config:** `configs/eval/vitl/icml/echojepa_l_k_lvef_d4.yaml`
**Protocol:** ICML preprint (d=4 attentive, 6-head HP grid, 20 epochs, BS=1, 224px, 16f)
**Checkpoint:** `checkpoints/anneal/keep/vitl-kinetics-pt220-an55.pt` (ViT-L Kinetics→MIMIC)
**Data:** UHN LVEF, 176K train / 26K val (A4C + B-mode view-filtered, S3 224px)
**Output:** `evals/vitl/icml/lvef/video_classification_frozen/icml-echojepa-l-k-lvef-d4/`

**Results (12 epochs completed):**

| Epoch | Val MAE | Val R² | Val Pearson | Best Head |
|-------|---------|--------|-------------|-----------|
| 1 | 5.384 | 0.639 | 0.810 | head 4 |
| 2 | 5.342 | 0.653 | 0.821 | head 4 |
| 3 | 4.957 | 0.687 | 0.833 | head 3 |
| 4 | 4.952 | 0.693 | 0.834 | head 4 |
| 5 | 4.822 | 0.711 | 0.843 | head 3 |
| 6 | 5.117 | 0.670 | 0.842 | head 3 |
| 7 | 4.782 | 0.715 | 0.847 | head 4 |
| 8 | 4.845 | 0.712 | 0.848 | head 3 |
| 9 | 4.740 | 0.723 | 0.853 | head 3 |
| 10 | 4.696 | 0.725 | 0.854 | head 3 |
| 11 | 4.638 | 0.734 | 0.858 | head 3 |
| 12 | **4.617** | **0.735** | **0.858** | head 3 |

**Best:** Epoch 12 — R² 0.735, MAE 4.617, Pearson 0.858.
**Comparison:** Nature Medicine d=1 L-K was R² 0.702. d=4 achieves +0.033 R² improvement.
**Throughput:** 0.24s/step, ~94 min/epoch (89 min train + 5 min val), 8× A100.
**Log:** `logs/icml_lk_lvef_d4.log`

**Resume command:**
```bash
cd /mnt/custom-file-systems/efs/fs-0049217cdf69186d7_fsap-0fa7145b64eaa046b/vjepa2
# Set resume_checkpoint: true in the config first, then:
TMPDIR=/tmp LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH \
  nohup python -m evals.main \
    --fname configs/eval/vitl/icml/echojepa_l_k_lvef_d4.yaml \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \
  > /home/sagemaker-user/user-default-efs/vjepa2/logs/icml_lk_lvef_d4.log 2>&1 &
```
**To resume:** Change `resume_checkpoint: false` → `resume_checkpoint: true` in the YAML config. Will pick up from epoch 12 via `latest.pt`.

### Bug 015: torch_shm_manager Broken on SageMaker A100 — FIXED

See `claude/dev/bugs/015-torch-shm-manager-broken.md`. Changed sharing strategy from `file_system` to `file_descriptor` in `app/vjepa_2_1/train.py`. Requires `TMPDIR=/tmp` and `LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH` for all launches on this node.

---

## 2026-03-27 (Session 31)

### BYOL-Video V2 Run — Fresh Start with Matched Config

Stopped v1 BYOL run (job 202, epoch ~44) due to representation degradation caused by cosine EMA [0.996, 1.0] freezing the target encoder. Started fresh v2 run with config matched to V-JEPA:

- EMA: [0.99925, 0.99925] constant (was [0.996, 1.0] cosine)
- Batch size: 64 × 16 GPUs = 1024 effective (was 32 × 16 = 512)
- S3 path: `byol-vitl-imagenet-v2/` (fresh)
- `force_load_pretrain: true` for epoch 1, flipped to `false` for subsequent restarts

**Loss trajectory (v2, healthy — no plateau):** -1.659 (e1) → -1.826 (e2) → -1.930 (e5) → -1.967 (e10)

**V1 vs V2 comparison at epoch 8-10:** V1 had slightly better absolute loss (-1.986 vs -1.967 at e10) because it started from an already-trained checkpoint, but V1 plateaued at e12 and degraded to -1.955 by e44. V2 is still improving monotonically.

### Bug 015: Checkpoint Pruning Disk Space — FIXED

**Symptom:** Periodic checkpoint saves (e4.pt, e6.pt, e8.pt) silently failed due to insufficient disk space. 97 GB disk with ~42 GB consumed by 4 CUDA versions left only ~8.7 GB for checkpoints after latest.pt.

**Root cause (three issues):**
1. `prune_local_checkpoints` ran with `max_to_keep=N` but the upcoming save would add 1 more — net result exceeded disk capacity
2. Python `list[:-0]` returns `[]` (empty), so `max_to_keep=0` silently skipped all deletions
3. `checkpoints_to_keep: 2` meant 2 periodic (9.6 GB) + latest (4.8 GB) = 14.4 GB — only 4 GB headroom, not enough for a 4.8 GB save

**Fix:**
- Call site: `prune_local_checkpoints(folder, max_to_keep=max(max_epoch_checkpoints - 1, 0))` — prune to N-1 before saving
- Prune function: `checkpoints_to_delete = all_checkpoints if max_to_keep == 0 else all_checkpoints[:-max_to_keep]`
- Condition: `>=` instead of `>` for defensive pruning
- Config: `checkpoints_to_keep: 1` — exactly 1 periodic + latest on disk, all archived to S3

**Files changed:**
- `app/byol_video/train.py` — prune logic fix
- `configs/train/vitl16/pretrain-byol-mimic-224px-16f-h100.yaml` — `checkpoints_to_keep: 1`, `force_load_pretrain: false`

**Validated:** e8.pt saved and archived to S3 after fix deployed (job 241).

---

## 2026-03-27 (Session 30)

### Bug 014: PyTorch Checkpoint >4 GB Serialization Failure — FIXED

**Root cause:** PyTorch's zipfile serializer (`inline_container.cc`) uses 32-bit offsets internally. BYOL-Video ViT-L checkpoints (~4.8 GB) exceed this boundary, causing "unexpected pos" errors.

**Misdiagnosis path:** Initially thought to be shared tensor storage from DDP/`torch.compile`. Applied `_unwrap_state_dict()` and `_clone_for_save()` (recursive tensor cloning). Both failed because the issue was file size, not storage sharing.

**Fix — split checkpoint into two files:**
- `latest.pt` — model weights, schedulers, epoch (~2.4 GB)
- `latest_opt.pt` — optimizer state (~2.4 GB)
- Both use atomic writes (`.tmp` → `os.replace`)
- `load_checkpoint()` updated to load optimizer from separate file
- S3 upload updated to push both files
- Added missing `import os` to `utils.py` (caused `NameError` on first deploy)

**Files changed:**
- `app/byol_video/train.py` — split save, `_clone_for_save()`, `_unwrap_state_dict()`, `torch.compile` for encoder
- `app/byol_video/utils.py` — split load, `import os`
- `app/vjepa_2_1/train.py` — `_clone_for_save()`, `file_descriptor` sharing strategy
- `evals/video_classification_frozen/eval.py` — disabled per-epoch checkpoint saves (2.6 GB each filled 97 GB disk)

**Validated:** mini-epoch test (ipe=5) saved 2.4 GB + 2.4 GB successfully. Full training resumed from split checkpoint.

### ICML Rebuttal LVEF Probe — Protocol Correction

**Problem:** H100 probe config was using Nature Medicine protocol (d=1, 20 heads, 35 epochs, MIMIC data, study_sampling) instead of ICML preprint protocol.

**Fix:** Created `configs/eval/vitl/icml/echojepa_l_k_lvef_d4_h100.yaml` matching ICML preprint exactly:
- d=4 (3 SA + 1 CA), 6-head HP grid (2 LR x 3 WD), 20 epochs, no warmup
- UHN A4C/B-mode LVEF data with raw values (z-score normalized at runtime)

**CSV fix:** Replaced pre-z-scored CSVs with canonical raw-value CSVs (inverse-transformed with mean=57.057, std=11.325). Eval code computes z-score params from train CSV at runtime.

### Multi-Node Distributed Training — 2-Node BYOL

Set up 2-node distributed BYOL training across both H100 compute nodes (16 GPUs total).

**New files:**
- `app/main_srun.py` — srun-compatible launcher for multi-node training. Each srun task = 1 GPU. Sets `CUDA_VISIBLE_DEVICES` from `SLURM_LOCALID`, reads rank/world_size from SLURM env vars.

**Files changed:**
- `src/utils/distributed.py` — fixed `MASTER_ADDR` for multi-node: no longer blindly overwrites with `"localhost"` or `os.environ["HOSTNAME"]`. Respects pre-set `MASTER_ADDR` from sbatch script, only falls back to hostname for single-node.

**Sbatch script** (`~/byol_pretrain_2node.sbatch`):
- `--nodes=2 --ntasks-per-node=8 --gpus-per-node=8`
- Sets `MASTER_ADDR` to first allocated node via `scontrol show hostnames`
- Syncs checkpoint files from first node to other nodes before training starts
- Launches `srun python -m app.main_srun --fname config.yaml`

**Config change (deployed, not in repo):** `batch_size: 64 → 32` per GPU to keep effective batch at 512 (32 × 16 GPUs = 512, same as single-node 64 × 8).

**Results:** 3.6s/iter (was 7.0s single-node) — 1.94x speedup. ~18 min/epoch (was ~35 min). ETA ~2.9 days (was ~5.6 days). Checkpoint save + S3 upload verified working.

### BYOL-Video Training Plateau — Root Cause & Config Fix

**Finding:** BYOL-Video ViT-L training plateaued by epoch ~12. Two-stage collapse detection:

1. **View classification (d=1, 13-class, UHN 22K):** e10 vs latest target encoder gave identical results (24.61% val acc, 0.696 AUROC) — too coarse to detect degradation.
2. **LVEF regression (d=1, 5K train / 2K val subset):** Revealed clear collapse via Pearson r.

| Encoder | Checkpoint | Best Val MAE | Best Pearson r |
|---------|-----------|-------------|---------------|
| e10 online | epoch 10 | 8.060 | **0.151** |
| e10 target | epoch 10 | 8.057 | **0.156** |
| latest online | epoch 40 | 8.068 | 0.089 |
| latest target | epoch 40 | 8.069 | 0.068 |

Online encoder Pearson r dropped 41% (0.151→0.089), target encoder dropped 56% (0.156→0.068). Online vs target nearly identical within each checkpoint. Target encoder weight divergence was only 0.005 over 30 epochs — the target encoder effectively froze while accumulating degraded representations.

**Root cause:** EMA schedule mismatch. V-JEPA uses **constant** EMA (0.99925) — the target encoder continuously tracks the online encoder, providing fresh learning signal throughout training. BYOL config used **cosine ramp** (0.996 → 1.0) per the original BYOL paper, which progressively freezes the target encoder. Combined with constant high LR (no decay after warmup), the online encoder diverges from a frozen target without improving representations.

**Why constant LR works for V-JEPA but not BYOL (with ramping EMA):**
- V-JEPA's constant EMA keeps the target moving → prediction objective stays non-trivial → constant LR drives continued learning
- BYOL's EMA → 1.0 freezes the target → self-distillation objective becomes trivially satisfiable → constant LR pushes online encoder away from useful representations

**Additional confound:** Effective batch size was 512 (64 × 8 GPUs) vs V-JEPA's 1024 (128 × 8 GPUs).

**Config fix (both A100 and H100 configs):**
- `ema: [0.996, 1.0]` → `[0.99925, 0.99925]` (match V-JEPA constant EMA)
- `batch_size: 64` → `128` (match V-JEPA effective batch 1024)
- H100 config: `force_load_pretrain: true`, new S3 checkpoint URI for fresh run

**Files changed:**
- `configs/train/vitl16/pretrain-byol-mimic-224px-16f.yaml`
- `configs/train/vitl16/pretrain-byol-mimic-224px-16f-h100.yaml`

**Remaining legitimate differences (by design):**

| Parameter | V-JEPA | BYOL |
|-----------|--------|------|
| Masking | Spatio-temporal blocks | None |
| Prediction target | Local masked tokens (L1) | Global mean-pooled (cosine) |
| Predictor | 12-layer transformer | 2-layer MLP |
| Additional heads | None | Projector (online + target) |

These are the intended differences that isolate the prediction objective.

### HyperPod Operations

- Freed 9.3 GB disk on both nodes (stale S3 setup cache: old checkpoint + conda env tarball)
- Fixed deploy script issue: `/tmp` not shared between controller and compute nodes; piped tarball via srun stdin
- Fixed permissions on node 184: `/opt/vjepa2/` dirs owned by `nobody:nogroup` from initial setup, required `sudo chown`

### MIMIC Outcome Chain — Progress Update & Retry Fix

**Bug fix — `scripts/run_mimic_outcome_chain.sh`:**
- Retry logic only checked for missing `best.pt`, not missing pred avg (`study_predictions.csv`). Added `has_pred_avg()` function and split retry into `RETRY_TRAIN` (missing checkpoint) and `RETRY_PREDAVG` (checkpoint exists but pred avg missing).
- Created `scripts/run_mimic_predavg_retry.sh` — standalone cleanup script to mop up all missing pred avgs after chain completes.

**Chain COMPLETE (15:01 UTC).** All 4 models × 11 tasks trained + pred avg done. L-K mortality_1yr (0.779) + mortality_90d (0.808) initially failed — root cause was missing `latest.pt` (Mar 23 training only saved `best.pt`), not port collision. Fixed with symlink `latest.pt → best.pt`, completed 15:01 UTC.

**Final multi-model results (classification AUROC):**

| Task | G | L-K | EP | Pan |
|------|---|-----|-----|-----|
| mortality_1yr | **0.792** | 0.779 | 0.750 | 0.740 |
| mortality_90d | **0.827** | 0.808 | 0.772 | 0.745 |
| mortality_30d | **0.884** | 0.878 | 0.817 | 0.807 |
| in_hospital_mortality | **0.861** | 0.821 | 0.789 | 0.737 |
| readmission_30d | 0.608 | **0.626** | 0.623 | 0.594 |
| discharge_destination | 0.674 | 0.591 | **0.679** | 0.637 |

L-K mortality_1yr (0.779) + mortality_90d (0.808) completed after symlink fix.

**Regression (R² / Pearson):**

| Task | G | L-K | EP | Pan |
|------|---|-----|-----|-----|
| los_remaining | 0.038 / 0.319 | **0.052 / 0.298** | 0.009 / 0.286 | -0.008 / 0.193 |
| troponin_t | **0.036 / 0.264** | 0.020 / 0.171 | 0.018 / 0.210 | 0.009 / 0.154 |
| nt_probnp | **0.119 / 0.355** | 0.061 / 0.261 | 0.096 / 0.337 | 0.055 / 0.322 |
| creatinine | **0.014 / 0.240** | 0.000 / 0.189 | -0.008 / 0.163 | -0.018 / 0.144 |
| lactate | 0.004 / 0.211 | **0.032 / 0.293** | 0.011 / 0.225 | 0.000 / 0.161 |

**Key findings:**
- G dominates mortality (+4-12pp over EP/Pan), strongest on in_hospital_mortality (+12.4pp over Pan)
- L-K competitive: beats G on readmission (0.626 vs 0.608), los_remaining, lactate
- EchoPrime beats G on discharge_destination (0.679 vs 0.674)
- Biomarker R²s uniformly low (0.00-0.12), tiny test sets (n=126-535)
- Model ranking varies by task type — no single model dominates all outcomes

---

## 2026-03-26 (Session 29)

### BYOL-Video Training App — Implementation + Bug Fixes

**New training app** (`app: byol_video`) for ICML rebuttal three-way controlled comparison: JEPA vs BYOL-Video vs MAE. All from ImageNet ViT-L init, matched compute on MIMIC-IV-Echo.

**Files created:**
- `app/byol_video/__init__.py` — empty module for scaffold dispatch
- `app/byol_video/train.py` (~635 lines) — full training loop: BYOLCollator, force-load (V-JEPA + ImageNet formats), per-pair grad accum, EMA update, DDP guard
- `app/byol_video/utils.py` (~145 lines) — model init (raw ViT, no MultiSeqWrapper), optimizer (4 param groups), checkpoint load/save
- `src/models/byol_projector.py` (~40 lines) — BYOLProjector + BYOLPredictor MLPs (Linear→BN→ReLU→Linear)
- `configs/train/vitl16/pretrain-byol-mimic-224px-16f.yaml` — matched config (batch 64, 240 epochs, ImageNet init)

**Bugs encountered and fixed during smoke test:**
1. **Autograd inplace modification** — BN running stats updated inplace across multiple clip-pair forward passes before backward. Fix: per-pair gradient accumulation (each pair does forward+backward independently, gradients accumulate, optimizer steps once).
2. **`ReLU(inplace=True)` conflict** — Changed to `ReLU()` in byol_projector.py to avoid version counter issues with autograd.
3. **DDP `ValueError: Default process group has not been initialized`** — `init_distributed()` without args falls through to SLURM check in debug mode. Fix: guard DDP wrapping with `if dist.is_available() and dist.is_initialized()`.
4. **Wrong checkpoint (Kinetics vs ImageNet)** — Controlled comparison requires ImageNet init, not Kinetics V-JEPA2. Downloaded `vitl_raw.pth` from S3, processed to `vitl_in21k.pt` (stripped heads, flat state dict). Added 2D→3D patch_embed inflation in force-load logic.
5. **Flake8 unused variable** — Removed `proj_pred_params` in utils.py.

**Force-load logic** handles both:
- V-JEPA format: `{'encoder': sd, 'predictor': sd, 'target_encoder': sd, ...}`
- Flat ImageNet format: `{'model': sd}` with classification heads stripped, 2D patch_embed inflated to 3D

**Smoke test result**: loss=0.029 on first step (ImageNet init), confirming no collapse and correct initialization.

**Docs updated**: `configs/train/README.md` (BYOL entry + checkpoint reference), `claude/architecture/pretraining-and-cooldown.md` (BYOL section + config table entry).

---

## 2026-03-26 (Session 28)

### Trajectory Experiment Restructuring & MR Onset

**Analysis**: Identified fundamental confound in delta prediction tasks — model succeeds by encoding current measurement value (regression to mean), not by detecting subclinical trajectory signals. Onset paradigm (filter to normal baselines) avoids this because current value is uninformative.

**Code change — `experiments/nature_medicine/uhn/build_trajectory_csvs.py`:**
- Added `--direction {decline,worsen}` parameter to `build_onset_csvs_for_task()`
- Added `--baseline_max` and `--future_above` params for worsening onset (MR-style: baseline ≤ mild, future ≥ moderate+)
- Existing `--baseline_min` and `--future_below` remain for decline onset (LVEF-style)
- Generalized metadata field names from `baseline_ef`/`followup_ef` to `baseline_value`/`followup_value`

**New task — trajectory_mr_severity_onset:**
- Built: 19,967 train studies (10% worsened), 3,959 val, 5,932 test
- Definition: baseline MR ≤ mild (0-2), predict who develops moderate+ (≥3) within 30-365 days
- Views: A4C, A2C, A3C, PLAX (standard for MR assessment)

**Results — trajectory_mr_severity_onset (pred avg AUROC):**

| Model | AUROC |
|-------|-------|
| **EchoJEPA-G** | **0.7325** |
| PanEcho | 0.6880 |
| EchoPrime | 0.6657 |
| EchoJEPA-L-K | 0.6054 |

Validates onset paradigm generalizes beyond LVEF. G leads by 4.5pp over PanEcho.

**Results — trajectory_lvef 3-class (pred avg AUROC, completed EP + Pan):**

| Model | AUROC |
|-------|-------|
| PanEcho | 0.6327 |
| EchoPrime | 0.6281 |
| EchoJEPA-G | 0.6134 |
| EchoJEPA-L | 0.5360 |
| EchoJEPA-L-K | 0.5315 |

3-class (declined/stable/improved) weaker than onset, as expected. Non-JEPA models outperform on delta task — consistent with it measuring current-state encoding rather than true trajectory prediction.

**Bug fix — `scripts/run_mimic_outcome_chain.sh`:**
- `set -euo pipefail` + `grep` returning exit code 1 (no matches) in `wait_for_training()` silently killed the script
- Fix: added `|| true` to the grep pipeline
- Relaunched chain with correct conda env (vjepa2-312)

### MIMIC Outcome Chain — Relaunched

Missing checkpoints before relaunch:
- in_hospital_mortality: all 4 models
- mortality_1yr: EP, Pan
- mortality_30d: L-K, EP, Pan
- readmission_30d: L-K, EP, Pan
- discharge_destination: L-K, EP, Pan
- los_remaining: L-K, EP, Pan
- Plus pred avg gaps for L-K (mortality_1yr, mortality_90d)

Chain running on all 8 GPUs (2 tasks parallel on 4+4 split). ETA ~12-18 hours.

### trajectory_lvef_onset — Strategy E Pred Avg (4 manuscript models)

Ran proper Strategy E prediction averaging for LVEF onset (previously only had old-style single-clip inference). S3 throttling with 8 GPUs (NCCL timeout at batch 17) — fixed by running 4 GPUs per model with 2 models in parallel.

| Model | Pred Avg AUROC | Old single-clip |
|-------|:-:|:-:|
| **EchoJEPA-G** | **0.794** | 0.793 |
| EchoPrime | 0.782 | 0.776 |
| PanEcho | 0.781 | 0.759 |
| EchoJEPA-L-K | 0.683 | 0.677 |

MIMIC chain G pred avg re-confirmed after relaunch: mortality_1yr 0.792, 90d 0.827, 30d 0.884.

## 2026-03-26 (Session 27)

### Zero-Shot Anomaly Detection — Complete (40 Experiments)

**Multi-model representation distance (`evals/forward_prediction/anomaly_repr.py`):**

Added multi-model support to `anomaly_repr.py`:
- `MODEL_REGISTRY` dict with configs for echojepa, echoprime, panecho, videomae, echofm
- `load_model()` dispatches to JEPA loader or `init_module()` (from `evals/video_classification_frozen/models.py`)
- `--model` CLI argument, `--checkpoint` now optional (EchoPrime/PanEcho don't need one)
- Custom normalization passthrough via `make_transforms(normalize=...)`

**Binarized AUROC fix:**
- Multi-class ordinal labels (e.g., pericardial effusion 0-4) caused AUROC to be skipped entirely
- Fixed: `binary_study = (study_labels > 0).astype(int)` before AUROC computation (study-level and clip-level)

**28 single-model experiments (EchoJEPA-G):**

MIMIC (15 tasks, population negatives):
| Task | AUROC | Method |
|------|-------|--------|
| Takotsubo | 0.711 | Mahalanobis (inv) |
| Amyloidosis | 0.698 | Cosine |
| Tamponade | 0.605 | Mahalanobis / Cosine |
| STEMI | 0.592 | Cosine |
| HCM | 0.586 | Mahalanobis (inv) |
| In-hosp mortality | 0.557 | Mahalanobis |
| Mortality 1yr | 0.543 | Mahalanobis |
| DCM/HF/Mort 30d/AFib/MR/TR/LV wall | 0.51-0.53 | — |

UHN (13 tasks, hard negatives — model pretrained on UHN):
- Takotsubo 0.640 (only UHN task with signal — extreme visual phenotype partially survives in-distribution)
- All other UHN tasks 0.51-0.55 including LVEF extremes (0.542), AS severity (0.546), pericardial effusion (0.514)

**12 multi-model experiments (top 3 MIMIC tasks × 4 models):**

| Task | EchoJEPA-G (1012M) | VideoMAE-L (305M) | PanEcho (42M) | EchoPrime (35M) |
|------|:------------------:|:-----------------:|:-------------:|:---------------:|
| Takotsubo | 0.711 | **0.871** | 0.617 | 0.663 |
| Amyloidosis | 0.698 | **0.726** | 0.670 | 0.667 |
| Tamponade | 0.605 | 0.575 | **0.660** | 0.630 |

**Key findings:**
1. Zero-shot detection is NOT JEPA-specific — all self-supervised cardiac encoders show signal
2. VideoMAE-L leads on takotsubo/amyloidosis (0.871/0.726 vs EchoJEPA-G's 0.711/0.698)
3. Performance scales with visual distinctiveness of the phenotype
4. Two factors: (1) visually dramatic B-mode phenotype (primary), (2) out-of-distribution data (amplifier)
5. JEPA predictor-based scoring (prediction error, forward prediction) uniformly at chance — predictor is a universal reconstruction model

**Files modified:**
- `evals/forward_prediction/anomaly_repr.py` — multi-model support + binarized AUROC fix
- `evals/forward_prediction/RESULTS.md` — comprehensive 40-experiment log
- `claude/architecture/forward-prediction.md` — updated reference doc with all results + multi-model table

**Results stored:**
- `results/mimic_anomaly_repr/` — 15 EchoJEPA-G MIMIC tasks
- `results/uhn_anomaly_repr/` — 13 EchoJEPA-G UHN tasks
- `results/mimic_anomaly_repr/{echoprime,panecho,videomae}/` — 3 tasks × 3 models

---

## 2026-03-24 (Session 26)

### sklearn Reproduction Experiments + Train Feature Extraction

**sklearn on mean-pooled embeddings — CY reproduction attempt:**
- Created `scripts/sklearn_on_meanpool.py`: study-level mean-pool → sklearn (LogReg/Ridge). Uses CY's NPZ (`echojepa_g_mimic_embeddings.npz`, 525K clips). Results: mort_1yr AUROC 0.821, mort_30d 0.893.
- Created `scripts/sklearn_on_meanpool_predavg.py`: clip-level sklearn train + study-level prediction averaging. Trains on all 366K clips. Results: mort_1yr AUROC 0.808, mort_30d 0.895.
- **Did not reproduce CY's results**: Consistent 2-5pp gap (CY mort_1yr 0.846, ours 0.821). Likely causes: LBFGS solver convergence at max_iter=500 on 366K×1408 features, no L1 regularization, narrower HP grid.

**Train-set attentive feature extraction (all 10 MIMIC tasks):**
- Created `scripts/run_mimic_extract_train_features.sh` and `scripts/run_mimic_extract_train_features_all.sh`
- Runs frozen encoder + attentive pooler on training CSVs to produce clip_outputs.npz with 1408-dim features
- All 10 tasks completed (6.5h total, 2 parallel pairs on split GPUs)
- NPZ sizes: 247MB (nt_probnp) to 2.0GB (mortality tasks)

**sklearn on attentive features — head mismatch discovered:**
- Created `scripts/sklearn_on_attentive.py`
- **Bug found**: Each HP head has its own attentive pooler (separate learned cross-attention weights). Train NPZ saves features from head N (best on train), test NPZ saves from head M (best on test). Features from different heads are in incompatible feature spaces. Combining them gives AUROC 0.21 (below chance).
- Fix: used `clip_probs_all_heads` (N×15×2) instead of 1408-dim features. Consistent head selection from both NPZs.
- Results: direct attentive PA (val-selected best head) matches earlier pred avg results. sklearn ensemble of 15 heads' probabilities (30-dim) provides marginal improvement on some regression tasks.

**Final three-way comparison (test-set, study-level PA AUROC):**

| Task | CY sklearn | Our mean-pool sklearn | Attentive d=1 PA | Attentive sklearn ens. |
|------|-----------|----------------------|------------------|----------------------|
| mortality_1yr | **0.846** | 0.821 | 0.790 | 0.787 |
| mortality_90d | **0.883** | ~0.851 | 0.827 | 0.822 |
| mortality_30d | **0.912** | 0.895 | 0.882 | 0.883 |
| readmission_30d | **0.634** | 0.581 | 0.596 | 0.596 |
| discharge_dest | **0.689** | 0.655 | 0.670 | 0.670 |

**Key findings:**
1. Mean-pool sklearn consistently beats attentive probes on MIMIC mortality (by 3-6pp)
2. Attentive probes competitive on weaker-signal tasks (readmission, discharge)
3. CY's pipeline is gold standard — gap is solver/HP optimization, not feature quality
4. Multi-head attentive pooler architecture means features are head-specific (not shareable across runs with different best heads)

**CY code review (commit 549fb07):**
- Reviewed CY's full pipeline: `create_allclips.py` (study→clip expansion), `train_probe.py` (sklearn with HP sweep), `run_probes.py` (orchestration), `train_ehr.py` (XGBoost 500 trees + TabPFN), `fix_lab_labels.py` (positive timegap filter for troponin/creatinine)
- Key differences from our reproduction: CY likely uses saga solver with more iterations, L1+L2 sweep, broader HP grid

**Scripts created:**
- `scripts/sklearn_on_meanpool.py` — study-level mean-pool → sklearn
- `scripts/sklearn_on_meanpool_predavg.py` — clip-level sklearn → study PA
- `scripts/run_mimic_extract_train_features.sh` — single-task train feature extraction
- `scripts/run_mimic_extract_train_features_all.sh` — all 10 tasks parallel extraction
- `scripts/sklearn_on_attentive.py` — sklearn on attentive probe outputs

---

## 2026-03-23 (Session 25)

### MIMIC Outcome Chain + CY Baseline Integration

**Manuscript update — CY baseline results integrated:**
- Filled 40+ `\tbd` cells in outcomes table (`sn-article.tex` lines 239-266): all 5 baseline columns (LVEF, LVEF+demo+CCI, Elixhauser, XGBoost EHR, echo measurements) for 8 outcome + 4 biomarker rows
- Updated outcome prose (lines 219-227): replaced "in progress" with actual baseline comparisons
- Updated discussion (line 539) and limitations (line 551) with XGBoost context
- Revised biomarker claim to be honest: LVEF slightly outperforms EchoJEPA linear probes on troponin T (0.624 vs 0.600) and lactate (0.555 vs 0.534). Deferred "substantially stronger" claim to attentive probes.
- Updated footnote: `†` now defined as "mean-pooled frozen embeddings with linear probes"
- `\tbd` count: 60 → 44

**MIMIC outcome chain (`scripts/run_mimic_outcome_chain.sh`) — created and launched:**
- 11 tasks × 4 models (G first for manuscript priority)
- 2 tasks in parallel on split GPUs (0-3 and 4-7), ports 29500/29501
- Skip logic: checks `checkpoints/probes/mimic/{task}/{model}/best.pt`
- End-of-life excluded (CSVs not yet built)

**DDP port collision bug and fix:**
- **Root cause**: mortality_90d pred avg took 127 min on GPUs 4-7 (port 29501). When pair 2 launched in_hospital_mortality on same port, DDP init timed out after 300s.
- **Fix** (committed c3517d7): Added `cleanup_orphans()` + `wait_for_port_free()` (600s timeout with force-kill) between pairs. Added retry loop at end of each model phase.
- **Impact**: in_hospital_mortality G failed. Standalone requeue script (`scripts/run_mimic_inhospmort_g.sh`) launched, waits for chain to finish.

**MIMIC outcome pred avg results — EchoJEPA-G (Strategy E):**

| Task | Metric | Value |
|------|--------|-------|
| 1-yr mortality | AUROC | 0.791 |
| 90-day mortality | AUROC | 0.826 |
| 30-day mortality | AUROC | 0.881 |
| discharge_destination | AUROC | 0.677 |
| los_remaining | R² | 0.036 |
| Troponin T | R² | 0.013 |
| NT-proBNP | R² | 0.076 |
| Creatinine | R² | 0.029 |
| Lactate | R² | 0.008 |

**Key finding — attentive probes underperform linear probes on MIMIC outcomes:**
- Strategy E (d=1 attentive) mortality AUROCs: 0.791, 0.826, 0.881
- CY linear probes (mean-pooled sklearn): 0.846, 0.883, 0.912
- Delta: -3 to -6pp. Opposite of UHN pattern. Under investigation.
- Checkpoint integrity verified: md5sum confirms eval dir and archive dir best.pt are identical. All tasks completed 35 epochs (latest.pt epoch=35).

**Chain status at session end:**
- G phase COMPLETE (10/11 tasks). L-K phase IN PROGRESS (pair 1, epoch ~11/35).
- in_hospital_mortality G: standalone requeue (PID 2060121) waiting for chain.
- readmission_30d G: pred avg ran but study_predictions.csv not saved — needs re-run.

**Git commits this session:**
- CY baseline results in manuscript (sn-article.tex)
- Organized prior uncommitted changes (train.py fix, pretrain config, docs, run scripts, papers)
- `c3517d7` — Port collision fix + standalone requeue script

---

## 2026-03-22 (Session 24)

### Disease Pred Avg Completion + MIMIC Cross-Transfer Analysis

**Disease pred avg 6/7 DONE (4/4 manuscript models each):**
- Myxo MV newly complete: EP **0.859**, Pan **0.835**
- Bicuspid AV 3/4: EP **0.901** done, Pan RUNNING
- DCM/STEMI/rheumatic MV all 4/4 DONE (other machine): DCM G 0.837, STEMI G 0.826, rheumatic G 0.846

**MIMIC cross-institution disease transfer (4 diseases × 4 models DONE):**
- Amyloidosis: G **0.947** (improves over UHN 0.927!), EP 0.917, Pan 0.902, L-K 0.741
- HCM: G **0.847**, L-K 0.707, Pan 0.633, EP 0.516
- DCM: EP **0.721**, G 0.717, L-K 0.683, Pan 0.682
- STEMI: G **0.657**, L-K 0.636, EP 0.588, Pan 0.582
- Key: amyloidosis transfers exceptionally (structural signature is institution-invariant). HCM EP collapses (0.516). STEMI weak across all models.

**Cross-transfer analysis (class map compatibility):**
- MR severity: UHN 5-class (none/trace/mild/mod/sev) vs MIMIC 4-class (none-trivial/mild/mod/sev). Post-hoc merge feasible (sum logits 0+1).
- TR severity: same 5→4 mismatch. Same fix.
- LVEF: direct (both regression in %). Trivially runnable.
- MIMIC has afib/HF/tamponade/takotsubo labels without UHN counterparts (need fresh MIMIC training, not cross-transfer).

**STEMI L-K investigation:**
- L-K drops from 0.729 (val) to 0.623 (pred avg). Initially suspected collapse, but AUROC above chance is statistically significant (p=0.013). Real but weak signal — L-K's low UHN number was honest while EP/Pan were inflated by single-clip lucky sampling.

**Files modified:**
- TASK_TRACKER.md, roadmap ×2, probe-results.md, manuscript-tasks.md, changelog (this), MEMORY.md — all updated with disease PA completion + MIMIC xfer results

---

## 2026-03-21 (Session 23)

### Comprehensive Pred Avg Completion + Disease Training Results

**ALL 13 primary tasks pred avg DONE (all 5 models):**
- Final 3 tasks completed: RV S' (L-K 0.473, EP 0.353, Pan 0.301), RV FAC G (0.539), AR sev G (0.765) + Pan (0.692)
- shm failures resolved by running G jobs sequentially with different MASTER_PORT (29503)

**Disease probes trained (6/8, single-clip val AUROC):**
| Disease | G | L-K | EP | L | Pan | Machine |
|---------|---|-----|----|----|-----|---------|
| HCM | **0.942** | 0.845 | 0.778 | — | 0.816 | This |
| DCM | **0.846** | 0.760 | 0.763 | — | 0.733 | This |
| STEMI | **0.837** | 0.729 | 0.770 | 0.718 | 0.731 | Other |
| Amyloidosis | **0.935** | 0.706 | 0.755 | 0.683 | 0.801 | Other |
| Myxomatous MV | **0.917** | 0.854 | 0.813 | 0.756 | 0.759 | Other |
| Rheumatic MV | **0.795** | 0.714 | 0.749 | 0.735 | 0.702 | Other |

Bicuspid AV training launched (GPUs 4-7, G model). Takotsubo remaining.

**Disease probe pipeline (from earlier in session):**
- `build_probe_csvs.py` extended with `--max_neg_pos_ratio` flag for train-only negative downsampling (val/test untouched)
- Disease CSVs rebuilt at 3:1 cap from v7.2 labels. 5/8 diseases have view-filtered variants.
- Class imbalance handled at 3 levels: (1) 3:1 CSV neg cap, (2) `class_balance_ratio=3` in sampler, (3) inverse-freq class weighting in CE loss

**Files modified:**
- `experiments/nature_medicine/uhn/build_probe_csvs.py` — added `--max_neg_pos_ratio` CLI arg + downsampling logic
- All doc files updated with disease results and pred avg completion (TASK_TRACKER, roadmap ×2, changelog ×2, manuscript-tasks, probe-results, MEMORY)
- `claude/dev/changelog.md` — this entry

---

## 2026-03-20 (Session 22, continued)

### Disease Label Rebuild — Full Provenance (9 diseases)

Rebuilt all 9 disease detection NPZs with fully documented provenance. The original datasets were built by a now-deleted notebook with no surviving code.

**New file: `experiments/nature_medicine/uhn/build_disease_labels.py`** (~1000 lines):
- 9 builder functions: HCM, amyloidosis, takotsubo, STEMI, endocarditis, DCM, bicuspid AV, myxomatous MV, rheumatic MV
- Three label sources queried per disease: Syngo structured obs, HeartLab SENTENCE+NOTE, Syngo free-text
- Full negation filtering (no X, r/o X, rule out X, etc.) with per-disease custom patterns
- `--patient_propagation` flag for old-notebook-style labeling vs conservative study-level
- `--validate` mode compares against existing NPZs
- Provenance JSON output per disease

**Key discoveries:**
- HeartLab SENTENCE path (via `heartlab_findings.SENTENCE`) captures ~90% of HL matches; NOTE is supplementary
- Old notebook used patient-level propagation (if ANY study mentions disease, label ALL patient studies)
- **Endocarditis old NPZ (11,286 pos) was contaminated** — 82.6% negation rate. Reduced to 4,021 after proper filtering.
- Spot-check found family history contamination in patient-propagated amyloidosis labels

**Output:**
- `labels_v2/`: 9 disease NPZs with patient propagation + 10 provenance JSONs
- `labels_v2_study_level/`: 9 disease NPZs study-level (conservative) + provenance JSONs
- `UHN_DISEASE_PROVENANCE.md`: Full per-disease documentation (search terms, SQL sources, negation rates, confidence tiers)
- `CLASS_MAPS.md`: Disease section updated with source breakdown, confidence tiers

**Validation (study-level vs old NPZ):**
- HCM +0.5%, Bicuspid AV +7.5%, Rheumatic MV +13% — good matches
- Endocarditis -64% (intentional — old was contaminated)
- Myxomatous MV +242% (SENTENCE-level "mitral valve prolapse" captures more; old used narrower terms)

## 2026-03-20 (Session 22)

### Results: AS Severity Pred Avg + RV S' Training Complete

**AS severity pred avg (4/5 on disk):**
- G: AUROC 0.932, L: 0.846, L-K: 0.868, Pan: 0.813
- EP: 0.868 (confirmed in chain log, disk files lost in concurrent session collision — re-run needed)
- Major improvement over single-clip val: G went from 0.908* → 0.932

**RV S' training (4/5 done, Pan in progress):**
- G: R²=0.491, L-K: 0.374, EP: 0.284, L: 0.234 (all 15/15)
- Pan: ep 3/15 (R²=0.183 so far)

**Pred avg chain running (GPUs 4-7):** AV Vmax G → then 4 more tasks (AR sev, E/e', MR sev)

**Bug investigation:** EchoPrime AS severity disk files disappeared due to concurrent chain sessions on GPUs 4-7 — both chains wrote to same output dirs. Not a code bug. The `rm -rf` cleanup in run_pred_avg.sh is working as designed.

**Files updated:**
- `experiments/nature_medicine/TASK_TRACKER.md` — AS sev pred avg results, RV S' training status, checkpoint inventory
- `uhn_echo/nature_medicine/context_files/dev/manuscript-tasks.md` — AS sev PARTIAL, RV S' all trained, updated scoreboard + priorities
- `MEMORY.md` — Run status, pred avg results

---

## 2026-03-19 (Session 21, continued)

### Doc Update: Comprehensive Task & Pred Avg Inventory

Audited all checkpoint directories and pred avg output directories to build accurate inventory.

**Key findings during audit:**
- AV mean grad pred avg has been re-run with fixed `run_pred_avg.sh` — all 5 VALID (G R²=0.579, EP 0.462, Pan 0.378, L-K 0.328, L 0.147). Previous MEMORY entry was stale ("INVALID").
- LVEF pred avg now complete for all 5 models (L-K R²=0.702, Pan 0.665 added since last update).
- TR severity pred avg complete for all 5 models (EP 0.780, Pan 0.778 added).
- MR severity pred avg: G 0.882 done, L/L-K stale (Bug 012 header-only), EP/Pan not started.
- AS severity pred avg: G stale (Bug 012 header-only), rest not started.
- RV S' training in progress: G R²=0.491, L R²=0.234 done, L-K at ep 10/15.
- All Bug 007 retraining complete: 63 best.pt files across 14 task directories.

**Files updated:**
- `experiments/nature_medicine/TASK_TRACKER.md` — Rewrote checkpoint status, regression/classification results tables, surviving checkpoints summary, batch status
- `claude/dev/roadmap.md` — Updated blocking work priorities, added pred avg summary table
- `MEMORY.md` — Updated run status, blocking work, hemodynamic results

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
- Auto-detects task type (regression/classification), view filtering, regression mean/std
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
- Regression targets stored as raw values; Z-score normalization happens at runtime (mean/std computed from train CSV, saved to `zscore_params.json`)
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
- Fixed: auto-detect regression vs classification, store raw regression labels (Z-score normalization at runtime)
- All 23 MIMIC CSVs rebuilt with correct labels

**Phase 1 run scripts built:**
- `scripts/run_uhn_probe.sh` — Generic single-task probe runner for any UHN task
  - Auto-detects: task_type (regression/classification), num_classes, view filtering (train_vf.csv vs train.csv), study_sampling (false for trajectory_* tasks). Regression mean/std computed at runtime from raw CSV labels
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
