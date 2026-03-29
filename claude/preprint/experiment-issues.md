# ICML Preprint: Experiment Issues & Lessons Learned

Issues discovered during early experimental testing (Aug-Sep 2025) and subsequent code review (Feb-Mar 2026). These were exploratory runs done while developing the probe pipeline. Documented here for reference so the same pitfalls are avoided in Nature Medicine.

## 1. Batch Size Scaling Failure (Aug 2025)

### Setup

Early probe training (view classification / RV function, d=4 attentive, ViT-G) used `batch_size: 8` per GPU (eff_bs=64 on 8 GPUs). Attempted to speed up training by increasing to `batch_size: 48` (eff_bs=384), with and without linear LR scaling.

### Dataset

`a4c_rvfx_labels_train_tiny.csv`: **2,000 training clips** (binary RV function classification). This was a deliberately small development split used during probe pipeline iteration.

### Results

| Run | BS | LR | Best Test Acc | Outcome |
|-----|----|----|--------------|---------|
| BS8 baseline | 8 | 0.02 | 83.0% | Stable convergence |
| BS48 / no LR scale | 48 | 0.03 | 79.4% | Under-converged |
| BS48 / scaled LR v1 | 48 | 0.06 | 83.3% | Lucky convergence |
| BS48 / scaled LR v2 | 48 | 0.06 | 58.1% | Diverged/trapped |
| BS48 / scaled LR v3 | 48 | 0.05 | 62.0% | Diverged/trapped |

Three runs with the same BS48 + scaled LR config produced wildly different results (58-83%). The BS8 baseline was reproducible.

**Log files:** `logs/rvfx_cooldown_h16_b4_0824_keepe96_b8.log` (BS8), `logs/rvfx_kinetics_h16_b4_bs48_ep144_0826_scaledLR.log` (xLR v1), `logs/rvfx_kinetics_h16_b4_bs48_ep144_0827_scaledLR_v2.log` (v2), `logs/rvfx_kinetics_h16_b4_bs48_ep144_0827_scaledLR_v3.log` (v3).

**Plots:** `notebooks/probe_hps.png` (all HP variants), `notebooks/probe_scaledLR.png` (scaled LR train+test curves). Generated in `notebooks/plot.ipynb` (cells 4, 9).

### Root Cause: Dataset too small for batch size

| Config | Effective BS | Iters/Epoch | Fraction of dataset/step | Total grad updates (170 ep) |
|--------|-------------|-------------|--------------------------|---------------------------|
| BS8 | 64 | ~31 | 3.2% | ~5,300 |
| BS48 | 384 | ~5 | **19.2%** | ~850 |

With BS48, each gradient step sampled nearly 1/5 of the entire 2,000-clip dataset. Each epoch had only ~5 gradient updates. Three compounding factors:

1. **LR schedule compressed to ~5 steps/epoch.** Cosine decay with warmup needs hundreds of steps to function. With 5 steps/epoch, the warmup alone barely completes before the LR starts decaying. The schedule essentially collapses to a near-constant LR.

2. **Each step's gradient dominates the trajectory.** When one step sees 19% of the dataset, the random composition of that mini-batch determines the gradient direction. With so few total steps (~850), there's no law-of-large-numbers averaging. Different random seeds → different gradient trajectories → different basins.

3. **d=4 attentive probe (16H, 4B) was too large for ~850 updates.** The probe had ~10-20M trainable parameters. 850 gradient updates on 2,000 samples is massively under-determined. BS8's 5,300 updates gave 6x more signal.

### Lesson for Nature Medicine

The current Nature Medicine probes use BS1 (eff_bs=8), d=1 attentive probes, and much larger datasets (25K-100K+ studies). Even at BS8 (eff_bs=64), TAPSE's 25K studies would give ~390 iters/epoch — far above the danger zone. The ICML batch size failure was specific to the pathological combination of tiny dataset + large batch + deep probe.

The HP grid analysis (see `claude/dev/efficiency.md`) suggests the Nature Medicine probes could safely use BS2-4 with these larger datasets, though BS1 remains the conservative default.

---

## 2. Attentive Probe Inversion (d=4 on non-ViT models)

### Issue

The ICML preprint used d=4 attentive probes (16 heads, 4 self-attention blocks) for all models. Non-ViT baselines showed inverted results: linear probes outperformed attentive probes on view classification.

| Model | Tokens/clip | Attentive (d=4) | Linear | Status |
|-------|-------------|-----------------|--------|--------|
| EchoJEPA-G | 1568 | 87.4% | 80.9% | Expected |
| EchoJEPA-L | 1568 | 85.5% | 70.8% | Expected |
| EchoMAE-L | 1568 | 40.4% | 59.2% | **Inverted** |
| EchoPrime | 1 | 42.1% | 57.7% | **Inverted** |
| PanEcho | 1 | 41.9% | 52.8% | **Inverted** |

### Root Cause: Token starvation + normalization bugs

Two issues compounded:

1. **Token starvation**: EchoPrime and PanEcho output 1 pre-pooled token per clip. Cross-attention over a single token is degenerate (softmax trivially 1.0). The d=4 probe's 3 self-attention blocks became pure optimization overhead, causing overfitting. See `probe-architecture-analysis.md`.

2. **Normalization bugs** (bug 002): PanEcho was double-normalized (ImageNet applied twice), EchoPrime received ImageNet-normed input instead of [0,255], EchoFM received ImageNet-normed instead of [0,1]. All three baselines were running on corrupted inputs, further depressing their results. See `claude/dev/bugs/002-normalization-bugs.md`.

### Resolution for Nature Medicine

Nature Medicine adopted **Strategy E: d=1 attentive probes for all models**. At depth=1, there are no self-attention blocks — only a single cross-attention layer. Verification (UHN view classification, 22K clips) confirmed d=1 helps ALL models including EchoPrime (+9.3pp) and PanEcho (+7.1pp). The ICML inversion was an artifact of d=4 + normalization bugs + token starvation. Normalization bugs were fixed and affected models re-extracted.

---

## 3. Encoder Normalization Bugs (discovered Mar 2026)

Three of five baseline encoder adapters had incorrect input normalization. See `claude/dev/bugs/002-normalization-bugs.md` for full details.

| Model | Bug | Impact |
|-------|-----|--------|
| PanEcho | Double ImageNet normalization | Shifted input distribution |
| EchoPrime | Expected [0,255], received [-2,2] | Wrong input range entirely |
| EchoFM | Expected [0,1], received [-2,2] | Wrong input range entirely |

All ICML preprint baseline numbers for PanEcho, EchoPrime, and EchoFM were affected. The bugs were fixed and MIMIC embeddings re-extracted (2026-03-08). EchoJEPA and EchoMAE were unaffected (they use ImageNet normalization natively).

**Note:** The controlled comparison (EchoJEPA-L vs EchoMAE-L) is unaffected since both use the same ViT-L encoder adapter with correct normalization.

---

## 4. Shuffle Bug in Embedding Extraction (discovered Feb 2026)

`DistributedSampler(shuffle=True)` was used during embedding extraction, but the extracted embeddings were stored in shuffled order rather than the original CSV order. This meant embeddings were misaligned with their labels. See `claude/dev/bugs/001-shuffle-bug.md`.

**Impact:** All 7 models' MIMIC clip-level embeddings were in wrong order. Post-hoc reordering was applied using the saved shuffle permutation. UHN extractions fixed at the code level before extraction.

---

## 5. Pretraining Loss Divergence (ViT-L)

Visible in `notebooks/plot.ipynb` cell 16: the ViT-L pretraining loss drops from ~0.53 to ~0.455 by epoch 10, then **rises continuously** back to ~0.52 by epoch 200. This is classic overfitting on the pretraining objective.

The loss divergence is why the anneal (LR cooldown) stage was critical — it recovers representation quality by reducing the LR to allow the model to settle into a better basin. The anneal checkpoint (`vitl-pt-210-an25.pt`, 210 pretrain + 25 anneal epochs) is what's used for downstream probes, not the raw pretrain checkpoint.

The Kinetics-initialized ViT-L (`vitl-kinetics-pt220-an55.pt`) and ViT-G (`pt-280-an81.pt`) both went through similar anneal stages. Pretraining loss curves for these are in cells 47 and 49 of `plot.ipynb`.

---

## 6. Video Decode Failures (ongoing)

The S3-streaming video pipeline (decord) has a non-zero decode failure rate. Failed videos are silently substituted with the next valid video in the dataset. In the BS48 experiments, decode failure counts scaled with batch size:

| Config | Decode failures |
|--------|----------------|
| BS8 / 96 Ep (178 actual) | 358 |
| BS48 / xLR (170 ep) | 170 |
| BS48 / xLR v2 (64 ep) | 64 |

Roughly ~1 failure per epoch regardless of batch size. The substitution doesn't affect label alignment (bug 004 was fixed to track substitutions), but it does introduce subtle noise in training — the model sees a different video than intended. With BS48's few gradient updates per epoch, each substitution has proportionally more impact.

---

## 7. EchoMAE-L Pretraining LR ~170x Too Low (confirmed Mar 2026)

### The Claim

EchoMAE-L (VideoMAE ViT-L pretrained on MIMIC echos) used a learning rate ~170x below the standard VideoMAE LR. This was identified in the pre-review audit (`claude/archive/rebuttals/01-paper-audit.md`) and is the reason EchoMAE was dropped from the Nature Medicine manuscript.

### Evidence: Commit History

The LR was reduced across 4 commits, compounding from 6x to 171x below standard:

| Commit | Base `--lr` | Eff BS | Scaled LR | Standard | Ratio |
|--------|-------------|--------|-----------|----------|-------|
| `938fc40` Initial | 2.50e-5 | 512 | 5.00e-5 | 3.00e-4 | 6x low |
| `c95382c` Revised | 1.87e-5 | 384 | 2.81e-5 | 2.25e-4 | 8x low |
| `fbadf2b` Compute-matched | 7.03e-6 | 1024 | 2.81e-5 | 6.00e-4 | 21x low |
| `d91b4d4` Final (ep163 ckpt) | 8.79e-7 | 1024 | 3.52e-6 | 6.00e-4 | **171x low** |

Standard VideoMAE base LR is `1.5e-4` for BS=256 (the default in `run_mae_pretraining.py:77`). Linear scaling (`lr * eff_bs / 256`) is applied in the pretraining script at line 251.

The commit `d91b4d4` ("Fix videomae recursive calls, s3 credential expiry") set `LR_VAL="8.7890625e-7"` with the comment "Half of previous" — but `7.03125e-6 / 8 = 8.7890625e-7` is actually /8, not /2. The progressive reductions appear to have been reactive fixes for NaN losses during training (commit `85e7c89`: "Make VideoMAE training robust to NaN losses / corrupted videos"), rather than principled LR tuning.

### Worse Than 170x: Inverted Cosine Schedule

The checkpoint (`videomae-ep163.pth`) records the post-scaling training args:

```
peak_lr:   3.52e-6
min_lr:    4.00e-6   ← HIGHER than peak
warmup_lr: 4.00e-6
```

The `--min_lr 1.0e-6` from the sbatch was also linearly scaled: `1e-6 * 1024/256 = 4e-6`. Since `min_lr > peak_lr`, the cosine schedule was **inverted** — the LR was effectively clamped at a constant `4e-6` for the entire run. The true ratio vs standard peak (`6e-4`) is **150x**.

### How This Was Checked

1. **Sbatch file** (`scripts/videomae_pretrain_mimic.sbatch`): `LR_VAL="8.7890625e-7"`, `TARGET_EFF_BS=1024`
2. **Pretraining script** (`run_mae_pretraining.py:251`): `args.lr = args.lr * effective_global_batch / 256`
3. **Checkpoint metadata** (`videomae-ep163.pth`): `args.lr=3.515625e-06`, `args.min_lr=4e-06`
4. **Git history**: `git show {commit}:scripts/videomae_pretrain_mimic.sbatch` for each LR change

### Did the Model Learn?

Despite the severely low LR, EchoMAE-L converged (loss 0.87 → 0.27) and produced non-degenerate representations:
- RVSP MAE: 5.36 mmHg (vs EchoJEPA-L's 5.01) — **but see §8: both numbers are artifacts of label quantization, not real RVSP estimation**
- View classification: 40.4% attentive / 59.2% linear (inverted due to d=4 probe issue, see §2)

The model learned *something*, but almost certainly far below its potential given a proper LR.

### Impact on Preprint

The JEPA-vs-MAE controlled comparison (same ViT-L, same data, same probes) is the cleanest comparison in the preprint. However, the 171x LR gap means EchoMAE was handicapped at pretraining time, not just at probe time. The comparison shows JEPA > undertrained-MAE, which is not the same as JEPA > properly-trained-MAE.

This is why EchoMAE was dropped from Nature Medicine. A fair JEPA-vs-MAE comparison would require retraining with the standard `1.5e-4` base LR and proper schedule.

### Files

- `scripts/videomae_pretrain_mimic.sbatch` — final sbatch with `LR_VAL="8.7890625e-7"`
- `evals/video_classification_frozen/modelcustom/VideoMAE/run_mae_pretraining.py` — pretraining script with linear LR scaling
- `checkpoints/videomae-ep163.pth` — the checkpoint used for all ICML probes (epoch 163, `args.lr=3.52e-6`)

---

## 8. Multi-View RVSP: Missing Z-Score Normalization (discovered Mar 2026)

### Issue

The multi-view eval module (`video_classification_frozen_multi`) never z-score normalized regression labels at runtime, unlike the single-view module. See `claude/dev/bugs/017-multiview-rvsp-no-zscore.md` for full details.

### How the Preprint Appeared to Work (But Didn't)

The ICML preprint CSVs were **pre-z-scored** using `sklearn.StandardScaler` (saved as `data/scalers/rvsp_scaler.pkl`, mean=34.465, std=14.013). With pre-normalized labels:
- `VideoGroupDataset` read z-scored floats but cast to `int()`, quantizing to ~5-6 bins (-2 through 3)
- **78% of z-scored labels rounded to 0** (raw RVSP ~20-48 mmHg → z ∈ [-1.0, 1.0] → `int()` = 0)
- Model learned to output ~0 in z-space (≈35 mmHg after un-z-scoring)
- Reported MAE looked reasonable (4.54-5.65) only because **most labels were the mean**

### Forensic Analysis: ICML RVSP Numbers Were Artifacts (confirmed Mar 2026)

Examining the saved prediction CSV (`predictions/echojepa_L_rvsp_test_predictions.csv`) reveals the EchoJEPA-L RVSP "5.01 MAE" was meaningless:

| Statistic | Value |
|-----------|-------|
| N (test predictions) | 636 |
| Label range | [20.45, 132.56] mmHg |
| **Prediction range** | **[34.89, 35.09] mmHg** |
| Prediction std | **0.03 mmHg** |
| Labels = 34.465 (target mean) | **494/636 (77.7%)** |
| Unique label values | **9** |
| MAE | 5.22 |
| R² | **-0.047** (worse than predicting mean) |
| Pearson | 0.196 |

**The model learned nothing about RVSP.** It predicted ~35 mmHg for every input. The MAE of 5.22 is simply the average distance from the mean to the 22% of labels that weren't quantized to the mean.

The `int()` cast was more destructive than previously understood. For RVSP (mean=34.465, std=14.013):
- Raw 20 mmHg → z = (20-34.465)/14.013 = -1.03 → `int()` = **-1**
- Raw 30 mmHg → z = (30-34.465)/14.013 = -0.32 → `int()` = **0**
- Raw 35 mmHg → z = (35-34.465)/14.013 = +0.04 → `int()` = **0**
- Raw 48 mmHg → z = (48-34.465)/14.013 = +0.97 → `int()` = **0**
- Raw 62 mmHg → z = (62-34.465)/14.013 = +1.97 → `int()` = **1**

The entire clinically relevant range (20-48 mmHg, covering normal through moderate pulmonary hypertension) maps to just two bins: 0 and -1. The model learned "predict 0" and got rewarded for it because 78% of labels were 0.

**This affects ALL models in ICML Table 4** (RVSP), not just EchoJEPA-L. The preprint's EchoJEPA-G 4.54, EchoPrime 5.65, PanEcho 5.49, and EchoMAE-L 5.36 are all artifacts of the same quantization. The tight spread (4.54-5.65) across models with very different representations is itself a red flag — they all learned the same trivial solution.

### What Broke Later (Pre-Fix Runs)

Post-preprint, the RVSP CSVs were rebuilt with raw mmHg values (19, 31, 30...) for the NatMed pipeline, which uses the single-view module with proper runtime z-scoring. The multi-view module was never updated. ICML rebuttal runs on raw CSVs produced catastrophic MAE (~145-176 logged scale) because:
1. Labels were raw (~34 mmHg mean)
2. Model outputs started near zero (no z-score target to learn toward efficiently)
3. `SmoothL1Loss` on the raw offset dominated all gradients

### Fix (Bug 017a — missing z-score normalization)

Added runtime z-scoring to the multi-view eval module (`y = (y - t_mean) / t_std`) before loss computation, matching the single-view module. Fixed run shows epoch 1 val MAE ~10.6 mmHg (correct scale, baseline ~11.2).

### Secondary Bug (Bug 017b — shared `zscore_params.json` poisoning)

Even after the z-score fix, the EchoMAE RVSP ep163 rebuttal run still failed. Root cause: the auto-detection code loaded z-score params from a **stale `zscore_params.json`** in the shared `data/csv/` directory. This file had been created by a prior LVEF run and contained LVEF parameters (mean=57.06, std=11.33) — wrong for RVSP (mean=34.47, std=14.01).

RVSP labels (~34 mmHg) were z-scored as `(34 - 57.06) / 11.33 = -2.03`, producing deeply negative targets. The model could never converge because the z-scored label distribution was centered around -2.0 instead of 0.0, and the un-normalization used the wrong std (11.33 instead of 14.01).

**Fix**: Added explicit `target_mean: 34.4650` / `target_std: 14.0130` to all 9 RVSP ICML configs. Deleted the stale `zscore_params.json`. **Lesson**: never rely on auto-detection when multiple tasks share a CSV directory — always specify z-score params in the YAML.

### Current RVSP Run: First Real Multi-View RVSP Probe (started 2026-03-28)

With all bugs fixed (017a: runtime z-scoring, 017b: explicit params in YAML), EchoJEPA-L is being retrained on the full UHN 41K multi-view RVSP dataset. This is the first run that produces genuine RVSP estimation.

**Config:** `configs/eval/vitl/icml/echojepa_l_mimic_full_rvsp_d4_uhn.yaml`
- Checkpoint: `vitl-pt-210-an25.pt` (MIMIC pt210 + 25ep anneal)
- Multi-view: 2 views, 2 clips/view, VideoGroupDataset
- d=4 attentive probe, 6-head HP grid, 20 epochs, BS=1
- Z-score: target_mean=34.465, target_std=14.013 (explicit in YAML)
- 8× A100, confirmed running `video_classification_frozen_multi`

**Training progress (epoch 1-8, as of 2026-03-29):**

| Epoch | Train MAE | Val MAE | Val R² | Val Pearson |
|-------|-----------|---------|--------|-------------|
| 1 | 9.825 | 10.525 | -0.034 | 0.167 |
| 2 | 9.540 | 9.788 | 0.082 | 0.347 |
| 3 | 9.199 | 9.782 | 0.160 | 0.401 |
| 4 | 9.010 | 9.542 | 0.195 | 0.442 |
| 5 | 8.927 | 9.284 | 0.199 | 0.459 |
| 6 | 8.861 | 9.148 | 0.187 | 0.477 |
| 7 | 8.766 | 9.343 | 0.236 | 0.486 |
| 8 | 8.729 | 9.163 | 0.196 | 0.497 |

Key observations:
- **Genuine learning**: Pearson climbing steadily (0.167 → 0.497), R² positive and increasing
- **Val MAE ~9.2 mmHg** at epoch 8 — higher than preprint's "5.01" but that was fake
- **R² ~0.2** vs preprint's effective R²=-0.05 — the current run actually predicts RVSP, not just the mean
- 12 epochs remaining, ~8-9 hours to completion
- For context: Nature Medicine d=1 single-view RVSP was R²=0.168 for L. Current d=4 multi-view already exceeds this.

### Impact

| Context | Affected? | Notes |
|---------|-----------|-------|
| ICML preprint RVSP (Table 4) | **YES — ALL NUMBERS INVALID** | `int()` cast quantized 78% of z-scored labels to 0. All models predicted the mean (~35 mmHg). Reported MAEs (4.54-5.65) are artifacts of label quantization, not RVSP estimation. See forensic analysis above. |
| ICML rebuttal EchoJEPA-L RVSP (41K UHN) | **Fixed & running** | Bug 017a + 017b fixed. First genuine RVSP probe. Epoch 8: R²=0.196, Pearson=0.497. |
| ICML rebuttal EchoMAE-L RVSP (all runs) | **Yes** | Bug 017a (pre-fix runs on 5K) + Bug 017b (post-fix ep163 run on 41K). All invalid. |
| ICML rebuttal EchoJEPA-B/L-K/BYOL RVSP | **Fixed** | Configs now have explicit params. Not yet run. |
| Nature Medicine RVSP | **No** | Uses single-view module with correct z-scoring + raw labels |

---

## Summary: What Survived vs What Didn't

| Finding | Survived to Nature Medicine? | Notes |
|---------|------|-------|
| JEPA > MAE controlled comparison | **Partially** | EchoMAE pretrained at 170x-low LR with inverted schedule — comparison is JEPA vs undertrained-MAE, not a fair objective comparison. Dropped from NatMed. |
| EchoJEPA-G dominance over baselines | Partially | Magnitudes inflated by d=4 inversion + normalization bugs. Re-evaluated with d=1 |
| Attentive probe as primary eval | No | Replaced by d=1 attentive (Strategy E) after discovering d=4 degeneration |
| Batch size scaling | No | Specific to tiny dev dataset. NM uses BS1 on large datasets |
| Sample efficiency (1% labels) | Yes | Controlled comparison, probe mismatch constant across fractions |
| Pediatric transfer | Yes | Tests representation directly, not probe design |
| Multi-view RVSP (4.54 MAE) | **No — artifact** | All RVSP MAEs in Table 4 were artifacts of `int()` quantization of z-scored labels. 78% of labels mapped to the mean; all models predicted ~35 mmHg. First real run in progress (epoch 8: R²=0.196). |

## Related Documents

- `probe-architecture-analysis.md` — attentive vs linear inversion root cause
- `encoder-fairness.md` — confound analysis (dim, scale, data, tokens)
- `claim-validity.md` — which claims survive the fairness analysis
- `hindsight-recommendations.md` — what we'd change if redoing the preprint
- `claude/dev/efficiency.md` — HP grid analysis for Nature Medicine probes
- `claude/dev/bugs/` — all 6 bugs with severity, status, and fixes
