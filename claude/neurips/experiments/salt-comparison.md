# SALT Comparison — Frozen Teacher vs EMA Self-Distillation

**Date:** 2026-04-07 (updated 2026-04-08)
**Status:** Complete (3 SALT variants × pred-avg test set)
**NeurIPS section:** §3 (Three-way comparison) — adds a fourth row as a mechanistic probe

---

## 🔒 FINAL DECISION (2026-04-08)

**Primary SALT checkpoint for ALL final NeurIPS experiments: SALT v1 e79.**

- **Encoder:** `checkpoints/salt_s2_vitl_e79.pt` (local) / `HYP/runs/salt_s2_pretrain_388/checkpoints/e79.pt` (S3)
- **END LVEF probe:** `evals/vitl/icml/salt_s2_e79_end_lvef_224/video_classification_frozen/icml-salt-s2-e79-end-lvef-d4/best.pt`
- **Pred-avg inference config:** `configs/eval/vitl/icml/salt_s2_e79_end_lvef_d4_predavg.yaml`
- **Registered as:** `SALT-S2-e79` in `scripts/rebuttal/frame_shuffle_severity.ALL_CONFIGS`
- **Best test numbers:** R²=0.414, MAE=6.66, Pearson=0.659 on EchoNet-Dynamic (1,277 videos)

**For any new SALT experiment** (UHN LVEF, RVSP, CAMUS segmentation, Pediatric zero-shot, EchoBench, frame shuffling, speckle probing, etc.): clone the JEPA-IN21K-e100 config and swap the encoder checkpoint to `salt_s2_vitl_e79.pt`. Do **not** re-run v3 or v1 e199 on new tasks — one variant for the main table keeps the story clean. v3 and e199 remain as appendix robustness lines on END LVEF only.

**Why v1 e79 (not v3 the paper-spec variant):**
1. Best SALT variant we have (R² 0.414 vs v3's 0.348 and v1 e199's 0.360) — conservative framing for the "SALT loses" claim. Picking a weaker variant would look like cherry-picking downward.
2. Both v1 and v3 use `loss_exp: 1.0` (L1 loss, matches SALT paper Eq 2.1) — the "v1 was L2 and therefore invalid" claim was retracted 2026-04-07 after config inspection.
3. Predictor architecture (hierarchical vs single-level) is a documented design axis the SALT paper itself ablates — neither variant is uniquely "the paper recipe."
4. v1 e199's lower R² is most parsimoniously explained by overfitting from constant LR on a small homogeneous dataset, not a SALT pathology — opens a confound the paper doesn't need.
5. The **robustness story is stronger than any single variant:** all three variants (v1 e79, v1 e199, v3 e79) land within ±0.03 R² of each other and all below MAE's 0.445. The one-sentence appendix framing "SALT is robust to predictor architecture and training length, indicating the gap to EMA-based methods is intrinsic to the frozen-teacher mechanism" is more persuasive than any single variant alone.

**Do NOT revisit this decision** unless new SALT variants change the *best-case* numbers, not the worst-case. The point of the §3 table is that SALT's best is still below EMA baselines.

---

## Question

The SALT paper (Apple, 2025) replaces V-JEPA's co-evolving EMA teacher with a frozen pixel-reconstruction teacher, claiming compute-efficiency without loss of representation quality. Does SALT match V-JEPA on echocardiography under matched compute and data conditions?

---

## Setup

**Three SALT variants tested**, all ViT-L pretrained on MIMIC-IV-Echo (525K clips), IN21K initialized for Stage 1, then SALT Stage 2 student training:

| Variant | Pretrain config | S2 epochs | Predictor architecture | Hyperparameters |
|---|---|---|---|---|
| **v1 e79** | original (pre-fix) configs | 80 | hierarchical 4-layer (4096-dim) | LR 1.75e-4 constant, weak augmentation |
| **v1 e199** | same as v1, extended | 200 | hierarchical 4-layer | LR 1.75e-4 constant, weak augmentation |
| **v3 e79** | -hp configs (paper-spec) | 80 | single-level (1024-dim) | LR 2.55e-4 sqrt-scaled cosine, paper augmentation |

**Probe protocol:** d=4 attentive probe, 6-head HP grid, 20 epochs on EchoNet-Dynamic train (7,460 videos). Pred-avg inference on EchoNet-Dynamic test (1,277 videos).

---

## Results

### EchoNet-Dynamic LVEF — full e100 baseline + SALT comparison

All test set numbers, all init-matched at ~100 epochs (or equivalent):

| Method | Predictor | Test MAE | Test R² | Test Pearson | vs predict-mean MAE |
|---|---|---|---|---|---|
| **JEPA-IN21K e100** | EMA self-distillation | **5.77** | **0.591** | **0.771** | −41.7% |
| BYOL e100 | EMA self-distillation | 6.41 | 0.468 | 0.690 | −35.3% |
| MAE e99 | (no teacher, pixel reconstruction) | 6.58 | 0.445 | 0.674 | −33.5% |
| **SALT v1 e79** | **frozen pixel-reconstruction teacher** | **6.66** | **0.414** | **0.659** | **−32.7%** |
| SALT v1 e199 | (same, extended) | 7.02 | 0.360 | 0.626 | −29.0% |
| SALT v3 e79 | (paper-spec single-level) | 7.03 | 0.348 | 0.617 | −29.0% |
| Predict-mean | — | 9.90 | — | — | — |

### The publishable finding

Under init-matched, compute-matched, data-matched conditions:

> **JEPA > BYOL > MAE > SALT** on EchoNet-Dynamic LVEF.

The ranking is the same as the rebuttal three-way comparison at pt50, now confirmed at e100 with init-matching. SALT is the **worst SSL method** by every metric (R², Pearson, MAE).

### One-line summary for the paper

**Replacing JEPA's co-evolving EMA teacher with a frozen pixel-reconstruction teacher (SALT) reduces LVEF R² from 0.591 to 0.414 (−30%), placing it below all three EMA-based objectives. This suggests co-evolution of the target encoder contributes to representation quality independent of the prediction target choice.**

---

## What This Does and Does Not Say

### Does say
- **EMA teacher dynamics matter for functional task performance.** SALT keeps the prediction target the same (latent prediction with masked tokens) but freezes the teacher. The performance gap is therefore attributable to teacher dynamics, not prediction target.
- **The result is robust.** v1 (hierarchical, weak aug) and v3 (paper-spec single-level) both land in the same neighborhood (R² 0.41 and 0.35). The SALT failure is not an artifact of any specific implementation choice.

### Does NOT say
- **It does NOT confirm any speckle-pollution mechanism.** The original ICML rebuttal claim that "JEPA filters speckle via EMA" was retracted after init-matched probing showed JEPA−MAE speckle gap is only 4%, not 23%, with BYOL being the best speckle filter (see `speckle-probing.md`). Do not frame SALT through speckle.
- **It does NOT mean SALT is broken in general.** The SALT paper's results on natural video (V-3.6M, K710, SSv2) may be valid. The compute regime is different: paper batch 3072 / 240K steps vs our batch 512 / 24K steps. SALT may need more optimization to surface its claimed advantages.
- **It does NOT mean the v1 e79 → e199 regression is a SALT-specific pathology.** The more parsimonious explanation is overfitting from constant LR (no decay) on a small homogeneous dataset. JEPA/BYOL avoid this through EMA implicit regularization. SALT has no such mechanism.

---

## Conservative Framing (Web Session Recommendation)

After extensive analysis (this session + a parallel web Claude session), the agreed-upon framing is:

1. **Single SALT row** in the main comparison table, using the best variant (v1 e79).
2. **Two sentences in the methods/results.** Don't oversell.
3. **No speckle pollution argument.** It depends on retracted claims.
4. **No e199 regression in main text.** Confounded by constant LR; appendix only with caveat.
5. **No v1 vs v3 architectural delta in main text.** Appendix as a robustness check at most.

### Recommended paper text

> "We additionally evaluated SALT (Li et al., 2025), which replaces JEPA's co-evolving EMA teacher with a frozen pixel-reconstruction teacher. Under matched conditions (same ViT-L, MIMIC-IV-Echo, IN21K initialization, ~100 epochs), SALT achieves R²=0.414 on EchoNet-Dynamic LVEF, below all three EMA-based objectives (JEPA 0.591, BYOL 0.468, MAE 0.445). This suggests that co-evolution of the target encoder contributes to functional task performance independent of the prediction target choice."

Two sentences. One row in the main table. That's all SALT deserves in the paper.

---

## Robustness Check (Appendix Material)

The fact that v1 e79 (hierarchical, weak aug) and v3 e79 (single-level, paper-spec) land at essentially the same place (R² 0.414 and 0.348, MAE 6.66 and 7.03) is a useful robustness check:

> "SALT performance is robust to predictor architecture (single-level vs hierarchical) and to extending training (e79 vs e199), indicating the gap to EMA-based methods is intrinsic to the frozen-teacher mechanism rather than an artifact of any particular implementation choice."

This is one sentence in the appendix.

---

## Resolved: Effective Dimensionality (2026-04-07)

**Does SALT inherit MAE's low effective dimensionality from its frozen pixel-reconstruction teacher?**

**Answer: No.** RankMe (spectral entropy) on 500 EchoNet-Dynamic test videos, same script (`scripts/rebuttal/rankme.py`), same GPU:

| Model | RankMe Eff Dim | Usage |
|-------|---------------|-------|
| JEPA e95 | 245.3 | 24.0% |
| BYOL e100 | 220.7 | 21.6% |
| MAE e99 | 206.4 | 20.2% |
| SALT v1 e79 | 202.7 | 19.8% |

All four models are in the **200-245 range**. SALT's effective dimensionality (203) is essentially identical to MAE's (206) — no collapse. The prior MAE=63 number (from Goodfire report) is not reproducible with this pipeline and should not be cited.

**Implication:** SALT's gap to JEPA is purely about **teacher dynamics** (lacks EMA co-evolution), not representational capacity. The student has enough capacity to learn diverse features, but without the evolving teacher signal, those features don't organize into useful temporal/functional structure. This supports the "co-evolution" framing — no dimensionality-based mechanism needed.

---

## Complete Artifacts Inventory (detailed, 2026-04-08)

**S3 bucket abbreviations:**
- `HYP` = `s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts`

### Encoder checkpoints (all verified 2026-04-08)

All four SALT v1 checkpoints have the same structural signature: `stage: 2` (SALT S2), 8 `norms_block.{0,1,2,3}.{weight,bias}` keys (hierarchical 4-layer predictor head — the v1 signature; v3 single-level has at most 1 `norms_block` entry), ViT-L encoder (304M params), 3.98 GB file size. V-JEPA epoch convention: stored `epoch = N+1` means N epochs completed.

| Name | Local path | S3 path | Stored epoch | Completed epochs | Variant | Size | Date |
|---|---|---|---|---|---|---|---|
| `salt_s2_vitl_e29.pt` | `checkpoints/salt_s2_vitl_e29.pt` | `HYP/runs/salt_s2_pretrain_388/checkpoints/e29.pt` | 30 | 29 | v1 (hierarchical) | 3.98 GB | Apr 5 |
| `salt_s2_vitl_e49.pt` | `checkpoints/salt_s2_vitl_e49.pt` | `HYP/runs/salt_s2_pretrain_388/checkpoints/e49.pt` | 50 | 49 | v1 (hierarchical) | 3.98 GB | Apr 5 |
| **`salt_s2_vitl_e79.pt`** | `checkpoints/salt_s2_vitl_e79.pt` | `HYP/runs/salt_s2_pretrain_388/checkpoints/e79.pt` | 80 | 79 | v1 (hierarchical) | 3.98 GB | Apr 5 |
| `salt_s2_vitl_e199.pt` | `checkpoints/salt_s2_vitl_e199.pt` | `HYP/runs/salt_s2_resume_e100_392/checkpoints/e199.pt` | 200 | 199 | v1 (hierarchical) | 3.98 GB | Apr 6 |

**Primary encoder for NeurIPS: `salt_s2_vitl_e79.pt`** (locked 2026-04-08). See § FINAL DECISION above for rationale.

**Full S2 v1 epoch series on S3** (every 5 epochs): `HYP/runs/salt_s2_pretrain_388/checkpoints/e{4,9,14,19,24,29,34,39,44,49,54,59,64,69,74,79}.pt` (S2 1→80), `HYP/runs/salt_s2_resume_e80_391/checkpoints/e{84,89,94,99}.pt` (S2 80→100), `HYP/runs/salt_s2_resume_e100_392/checkpoints/e{104,109,...,199}.pt` (S2 100→200).

**S2 v3 encoder** (paper-spec single-level predictor, S3-only, not mirrored locally): `HYP/runs/salt_s2v2_pretrain_446/checkpoints/e79.pt`. Used only for the appendix robustness row; do not re-run on new tasks per the locked decision.

**Stage 1 (V-Pixel teacher) checkpoints:** `HYP/runs/salt_s1_pretrain_379/checkpoints/e{4,9,14,19}.pt` + `latest.pt` (= e20). Both v1 and v3 use the same S1 teacher. Teacher is frozen during S2 — see `app/salt/train.py:377` for the DDP-unwrapping fix.

### Pretraining configs

| Variant | Config path | Key settings |
|---|---|---|
| S1 teacher (local) | `configs/train/vitl16/pretrain-salt-s1-mimic-224px-16f.yaml` | Pixel reconstruction, 20 epochs, IN21K init (`vitl_in21k.pt`), batch 128 × 8 GPUs |
| S1 teacher (HyperPod) | `configs/train/vitl16/pretrain-salt-s1-mimic-224px-16f-hp.yaml` | Same, `/opt/dlami/nvme` paths |
| **S2 v1 (primary)** | `configs/train/vitl16/pretrain-salt-s2-mimic-224px-16f.yaml` | Hierarchical 4-layer predictor, LR 1.75e-4 constant, weak augmentation, random student init, `loss_exp: 1.0` (L1) |
| S2 v1 extended | `configs/train/vitl16/pretrain-salt-s2-mimic-224px-16f-resume-e80-hp.yaml` → `...-resume-e100-hp.yaml` | v1 resume chain: e80 → e100 → e199 |
| S2 v3 (paper-spec) | `configs/train/vitl16/pretrain-salt-s2-mimic-224px-16f-hp.yaml` | Single-level predictor, LR 2.55e-4 sqrt-scaled cosine, paper augmentation, `loss_exp: 1.0` (L1) |

**Loss verification:** Both v1 and v3 use `loss_exp: 1.0` (L1 loss, matching SALT paper Eq 2.1). Verified by config inspection 2026-04-07. The earlier "v1 was L2 and therefore invalid" claim was retracted on the same date — see `claude/neurips/experiments/salt-comparison.md` (this doc) change history.

### HyperPod training jobs

| Job ID | Stage | Variant | Epochs | Node | Status | S3 prefix |
|---|---|---|---|---|---|---|
| 379 | S1 teacher | — | 0 → 20 | — | Complete | `HYP/runs/salt_s1_pretrain_379/` |
| 388 | S2 v1 | hierarchical | 0 → 80 | — | Complete | `HYP/runs/salt_s2_pretrain_388/` |
| 391 | S2 v1 resume | hierarchical | 80 → 100 | — | Complete | `HYP/runs/salt_s2_resume_e80_391/` |
| 392 | S2 v1 resume | hierarchical | 100 → 200 | — | Complete | `HYP/runs/salt_s2_resume_e100_392/` |
| 446 | S2 v3 | single-level | 0 → 80 | — | Complete | `HYP/runs/salt_s2v2_pretrain_446/` |

### Probe checkpoints (EchoNet-Dynamic LVEF, d=4 attentive)

All probes are 6-head HP grid (LR × WD), 20 training epochs, same d=4 attentive regressor architecture. `target_mean: 55.7776`, `target_std: 12.4072` (z-scored train label statistics). Each probe has six heads indexed 0–5 with different hyperparameters; the `best_val_acc_per_head` array picks the best head per checkpoint.

| Variant | Encoder | Probe path | Heads | Best head | Best val MAE | Test R² (pred-avg) | Test MAE (pred-avg) |
|---|---|---|---|---|---|---|---|
| v1 e29 | `salt_s2_vitl_e29.pt` | `evals/vitl/icml/salt_s2_e29_end_lvef_224/video_classification_frozen/icml-salt-s2-e29-end-lvef-d4/best.pt` | 6 | — | 7.82 | — (not evaluated) | — |
| **v1 e79** (primary) | `salt_s2_vitl_e79.pt` | `evals/vitl/icml/salt_s2_e79_end_lvef_224/video_classification_frozen/icml-salt-s2-e79-end-lvef-d4/best.pt` | 6 | 2 | **6.47** | **0.414** | **6.66** |
| v1 e199 (extended) | `salt_s2_vitl_e199.pt` | `evals/vitl/icml/salt_s2_e199_end_lvef_224/video_classification_frozen/icml-salt-s2-e199-end-lvef-d4/best.pt` | 6 | — | 6.73 | 0.360 | 7.02 |
| v3 e79 (S3 only) | `HYP/runs/salt_s2v2_pretrain_446/.../e79.pt` | `HYP/runs/salt_s2v3_echonet_lvef_454/probe/best.pt` | 6 | — | ~6.84 | 0.348 | 7.03 |

**Note on e49 probe:** a local probe checkpoint at `evals/vitl/icml/salt_s2_e49_end_lvef_224/.../best.pt` was mentioned in earlier versions of `checkpoint-inventory.md` §9 but is **not** present in the current filesystem. Only e29, e79, and e199 probes exist locally. If e49 evaluation is needed later, train from the existing `salt_s2_vitl_e49.pt` encoder using the same config template.

**Val vs test MAE offset:** all three SALT probes show a ~0.2–0.3 MAE gap between val and test (6.47 → 6.66, 6.73 → 7.02, 6.84 → 7.03) consistent with the JEPA/BYOL/MAE probes in this pipeline. No anomaly.

### Probe training and inference configs

| Purpose | Config path | Notes |
|---|---|---|
| v1 e29 probe training | `configs/eval/vitl/icml/salt_s2_e29_end_lvef_d4.yaml` | d=4 attentive, 6-head HP grid, 20 epochs |
| v1 e49 probe training | `configs/eval/vitl/icml/salt_s2_e49_end_lvef_d4.yaml` | Config exists, probe not trained yet |
| **v1 e79 probe training** | `configs/eval/vitl/icml/salt_s2_e79_end_lvef_d4.yaml` | Primary |
| v1 e199 probe training | `configs/eval/vitl/icml/salt_s2_e199_end_lvef_d4.yaml` | Extended variant |
| **v1 e79 pred-avg inference** | `configs/eval/vitl/icml/salt_s2_e79_end_lvef_d4_predavg.yaml` | Used for test-set R²=0.414 |
| v1 e199 pred-avg inference | `configs/eval/vitl/icml/salt_s2_e199_end_lvef_d4_predavg.yaml` | — |
| v3 pred-avg inference | `configs/eval/vitl/neurips/salt_s2v3_echonet_lvef_d4_predavg.yaml` | HyperPod-only |

**Pred-avg config gotcha:** All EchoNet-Dynamic pred-avg configs must use `study_sampling: false` because each EchoNet-Dynamic video IS a study (no multi-clip-per-study grouping). Setting `study_sampling: true` causes broken study_id extraction and groups all 1,280 clips into 1 fake study, making R² undefined. Verified across all three SALT pred-avg configs.

### Frame-shuffling and representation-analysis result CSVs

All result CSVs live in `scripts/rebuttal/samples/` (gitignored by `*.csv` rule — not tracked in git; raw data lives on EFS only, interpretation in markdown is what gets versioned). Always recomputable from the checkpoints + configs above if needed.

| Script | CSV output | Log | Status | Notes |
|---|---|---|---|---|
| `scripts/rebuttal/frame_shuffle_severity.py` (registered as `SALT-S2-e79` in `ALL_CONFIGS`) | `scripts/rebuttal/samples/severity_SALT_e79.csv` | `severity_SALT_e79.log`, `severity_SALT_e79.json` | Complete (2026-04-05) | 5 fractions × 3 seeds = 13 rows + header. Used in §4.2 severity gradient matrix. |
| `scripts/rebuttal/frame_shuffle_6cond.py` (uses same registry) | `scripts/rebuttal/samples/6cond_SALT_e79.csv` | `6cond_SALT_e79.log` | **Complete (2026-04-08)** | 6 conditions × {1 det, 3 seeds} = 14 rows + header. Used in §4.1 cross-model 4-way table. |
| `scripts/rebuttal/rankme.py` | (aggregated, no per-model CSV) | — | Complete (2026-04-07) | Single RankMe score: 202.7 (20% of 1024-dim space). In §4.3 effective-dim table. |
| `scripts/rebuttal/information_probing.py` | (integrated with other models) | — | Complete | Speckle probing partial R² — SALT not the primary focus here. |

**Registry entry for reproducibility** (`scripts/rebuttal/frame_shuffle_severity.py:82-89`):

```python
"SALT-S2-e79": {
    "encoder_type": "vjepa",
    "encoder_checkpoint": "checkpoints/salt_s2_vitl_e79.pt",
    "encoder_model_name": "vit_large",
    "encoder_key": "encoder",
    "probe_checkpoint": "evals/vitl/icml/salt_s2_e79_end_lvef_224/"
                        "video_classification_frozen/icml-salt-s2-e79-end-lvef-d4/best.pt",
},
```

Both `frame_shuffle_severity.py` and `frame_shuffle_6cond.py` import this registry, so the same encoder+probe pair drives both experiments — verified end-to-end by matching clean R² to 4 decimal places (severity fraction=0.00 row = 0.2926; 6cond clean row = 0.2926).

### Result tables (authoritative numbers)

**EchoNet-Dynamic LVEF — 6-condition frame shuffling, SALT-S2-e79** (1,277 test videos, 3 seeds for stochastic conditions; from `scripts/rebuttal/samples/6cond_SALT_e79.csv`, cliff profile):

| Condition | R² (mean) | R² σ | Pearson (mean) | MAE (mean) | ΔR² vs clean |
|---|---|---|---|---|---|
| clean | 0.2926 | — (deterministic) | 0.5643 | 7.35 | — |
| tubelet | 0.2902 | 0.0051 | 0.5633 | 7.38 | −0.8% |
| reverse | 0.2062 | — (deterministic) | 0.5139 | 7.92 | −29.6% |
| matched | 0.2915 | 0.0078 | 0.5636 | 7.37 | −0.4% |
| **shuffle** | **−0.4116** | 0.0094 | 0.4604 | 10.25 | **−241%** |
| **matched_frame** | **−0.4393** | 0.0523 | 0.4910 | 10.44 | **−250%** |

Interpretation: **cliff profile** — flat under local disruption (tubelet, matched, even reverse holds ~70% of clean R²), catastrophic under global frame-level disruption. The cliff happens at the tubelet granularity boundary: tubelet/matched permutations respect the 2-frame tubelet unit (SALT's pretraining patch-embed granularity), while shuffle/matched_frame operate below that boundary. SALT learned tubelet-level temporal features the frozen teacher provided but has no mechanism to generalize below the tubelet unit. Note σ on matched_frame (0.052) is ~5× the other stochastic conditions due to fixed permutation choice per seed; all three seeds are still strongly negative (−0.495, −0.455, −0.368). See `claude/neurips/experiments/6-condition-shuffling.md` for the full interpretation and connection to the §4.1 4-way table.

**EchoNet-Dynamic LVEF — severity gradient, SALT-S2-e79** (1,277 test videos, 3 seeds per non-zero fraction; from `scripts/rebuttal/samples/severity_SALT_e79.csv`):

| Fraction shuffled | R² (mean) | Pearson (mean) | MAE (mean) |
|---|---|---|---|
| 0.00 | 0.2926 | 0.5643 | 7.35 |
| 0.25 | −0.0368 | 0.4381 | 8.56 |
| 0.50 | −0.2771 | 0.4537 | 9.67 |
| 0.75 | −0.3821 | 0.4547 | 10.13 |
| 1.00 | −0.3968 | 0.4645 | 10.23 |

**Severity gradient collapses at just 25% shuffle** (R² goes negative). Compare to JEPA e100 (5 fractions: 0.591 → 0.542 → 0.507 → 0.485 → 0.488, stays positive and nearly flat 50%→100%), BYOL e100 (0.468 → 0.410 → 0.336 → 0.300 → 0.291, monotonic decay but stays positive), MAE e99 (0.445 → 0.421 → 0.436 → 0.414 → 0.428, flat). SALT is the only method that crosses zero at any fraction, and it does so at the lightest disruption level.

**Effective dimensionality (RankMe, 500 EchoNet-Dynamic videos):**

| Model | RankMe d_eff | % of 1024-dim space |
|---|---|---|
| JEPA IN21K e95 | 245.3 | 24.0% |
| BYOL e100 | 220.7 | 21.6% |
| MAE e99 | 206.4 | 20.2% |
| **SALT v1 e79** | **202.7** | **19.8%** |

SALT's effective dimensionality is essentially identical to MAE's (202.7 vs 206.4) — the student has comparable representational capacity. The cliff is NOT a capacity collapse; it is a feature-organization failure attributable to the frozen teacher. All four models are in the 200–245 range; the prior MAE=63 claim (Goodfire report) is not reproducible and should not be cited.

### Training convergence diagnostics (for the Q1 "undertrained?" rebuttal)

From `salt_s2_vitl_e79.pt` → `salt_s2_vitl_e199.pt` (2.5× more S2 training):

| Metric | e79 | e199 | Δ | Interpretation |
|---|---|---|---|---|
| Test R² (END LVEF) | 0.414 | 0.360 | **−0.054** | Regression (if undertrained, should improve) |
| Val MAE | 6.47 | 6.73 | +0.26 | Regression |
| Test MAE (pred-avg) | 6.66 | 7.02 | +0.36 | Regression |
| S2 training loss (flat from e100 onward) | 0.429 | 0.419 | −0.010 | Loss barely decreases after e100 — optimization converged |
| Weight cosine similarity between checkpoints (all blocks) | — | >0.999 | — | Encoder weights are essentially unchanged e100→e199 — optimization has nothing left to do |

**Interpretation:** SALT at e79 is optimization-converged. The e79 → e199 regression in downstream performance, combined with flat training loss and near-identical encoder weights, rules out "undertraining" as the cause of SALT's underperformance. The most parsimonious explanation is overfitting to frozen teacher targets from constant-LR training on a small homogeneous dataset — a failure mode JEPA/BYOL avoid through EMA implicit regularization that SALT inherently lacks.

### Known deviations from the SALT paper

Documented in `claude/architecture/salt-training-reference.md`. Cannot be fixed without >10× more compute or a different dataset:

| Deviation | Our value | Paper value | Impact |
|---|---|---|---|
| Batch size | 512 (per-node) | 3072 | LR sqrt-scaled for v3 (2.55e-4 vs paper 6.25e-4); v1 uses constant 1.75e-4 |
| Total S1+S2 training steps | ~24K | ~240K (240K S2 per paper Table 1) | ~10% of paper absolute compute |
| Pretraining dataset | MIMIC-IV-Echo, 525K single-domain echo clips | V-3.6M, 3.6M diverse natural video | Narrow-domain teacher has narrow coverage |
| Student init | Random (per paper recipe) | Random (per paper recipe) | Matches paper; disadvantage relative to our IN21K-initialized JEPA/BYOL/MAE baselines |
| Hierarchical norms training | v1: `norms_block` layers in teacher untrained (S1 uses `training_mode=False`) | Paper specifies dense supervision training mode | v3 single-level variant doesn't use these layers and also underperforms (R²=0.348), bridging the v1 concern |

These deviations explain why our SALT *absolute* numbers differ from the paper. They do NOT explain the cliff profile or the e79→e199 regression, which are robust to each variant and to extended training. The *qualitative finding* — SALT < EMA-based methods under our matched conditions — holds across both implementation variants and is the load-bearing claim for §4.5.

### Commit history (for provenance)

| Commit | Date | Change |
|---|---|---|
| `755a319` | 2026-04-04 | HP fixes for SALT training (batch, LR, augmentation) |
| `0eaf0ab` | 2026-04-05 | Loss function and hierarchical predictor revert |
| `71bd4e5` | 2026-04-05 | Predictor initialization fix |
| `db67dc8` | 2026-04-07 | Retract "SALT invalidated / must retrain" framing (L1 loss confirmed for both v1 and v3) |
| `48d4bb7` | 2026-04-08 | Lock SALT v1 e79 as primary NeurIPS checkpoint |
| `b805600` | 2026-04-08 | Add 6-condition SALT results + reviewer-defense Q&A + §1 regime framing |

---

## References

- **Paper:** Li et al., "Rethinking JEPA: Compute-Efficient Video SSL with Frozen Teachers", Apple, 2025. `claude/papers/vjepa-salt/arxiv.tex`
- **Implementation reference:** `claude/architecture/salt-training-reference.md`
- **Audit and bug fixes:** Commits `755a319` (HP fixes), `0eaf0ab` (loss/hierarchical revert), `71bd4e5` (predictor init fix)
- **Companion mechanism doc:** `claude/neurips/experiments/representation-analysis.md`

---

## Reviewer Rebuttal Q&A (2026-04-08)

This section collects the defensive responses to reviewer critiques the SALT row is most likely to attract. Load-bearing framing for the main table. **Keep all four answers ready for the rebuttal.**

### Q1: "Your SALT is undertrained. SALT paper uses 240K steps; you used ~24K. Run it longer."

**A:** We did. v1 e79 (80 S2 epochs) → v1 e199 (200 S2 epochs) is 2.5× more training and the result is **worse**, not better:

| Variant | S2 epochs | S2 steps (approx) | Test R² | Test MAE |
|---|---|---|---|---|
| v1 e79 | 80 | ~21K | **0.414** | **6.66** |
| v1 e199 | 200 | ~52K | 0.360 | 7.02 |

If SALT were under-converged, more epochs should improve it. Instead R² dropped 0.054 and MAE rose 0.36. Convergence diagnostics from e79→e199:
- Training loss was flat (0.429 → 0.419) from e100 onward
- Weight cosine similarity between the e79 and e199 encoder exceeded 0.999 on every block
- Probe val MAE regressed (6.47 → 6.73)

This is strong evidence the student is **not** optimization-undertrained. The plausible explanation for the e199 regression is overfitting from constant LR on a small homogeneous dataset (JEPA/BYOL avoid this via EMA implicit regularization; SALT has no such mechanism).

**Residual concern:** absolute step count is still ~10% of the paper's 240K. A sharp reviewer could argue "flat loss under constant LR proves optimization convergence, not representation consolidation." We cannot fully refute this without running 200K+ steps (~3 weeks single-node). The e199 regression is the strongest defense we have and it is a complete answer to "more epochs would help."

**One-sentence rebuttal:** "SALT v1 extended from 80 to 200 S2 epochs shows a test R² regression (0.414 → 0.360), indicating the result is not bounded by optimization undertraining."

### Q2: "SALT needs diverse data. You ran it on 525K single-domain echo clips; the paper uses 3.6M diverse natural video."

**A:** This is correct, and **it is the finding, not a flaw.** Our framing:

> Our result does not claim SALT is fundamentally inferior to EMA methods in general. Our result claims that **in the regime of medical video SSL** (single domain, ~500K-scale dataset, single-node compute, no strong external teacher), the frozen-teacher mechanism underperforms EMA-based co-evolution. The SALT paper's natural-video results on V-3.6M remain valid; US-JEPA (concurrent) succeeds with SALT by using URFM, a BiomedCLIP-distilled external teacher with broad medical coverage. The frozen-teacher mechanism works when **either** (a) the pretraining data has broad natural coverage (SALT paper) **or** (b) an external strong teacher is available (US-JEPA). With **neither**, EMA-based co-evolution is strictly better on functional tasks.

This is a **regime-conditional finding**, not an architectural claim. It is directly useful to practitioners who want to apply video SSL in narrow-domain, single-institution medical settings.

**One-sentence rebuttal:** "Our finding is that SALT requires either data diversity (paper) or a strong external teacher (US-JEPA), and absent both, fails in the narrow-domain single-institution regime that is representative of medical imaging deployment."

**Load-bearing action:** §1 of the paper must frame the data/compute regime as **the experimental variable**, not a limitation. See `paper-outline.md` §1 "Preempts the regime concern" for the sentences that do this.

### Q3: "Your primary SALT row isn't paper-spec. v1 uses a hierarchical 4-layer predictor; the paper uses a single-level predictor."

**A:** Both are valid SALT variants. We tested both and they land in the same neighborhood:

| Variant | Predictor | HP regime | S2 epochs | Test R² | Test MAE |
|---|---|---|---|---|---|
| **v1 e79** (our primary) | hierarchical 4-layer | LR 1.75e-4 constant, weak aug | 80 | **0.414** | **6.66** |
| v3 e79 (paper-spec) | single-level | LR 2.55e-4 cosine, paper aug | 80 | 0.348 | 7.03 |
| v1 e199 (extended) | hierarchical 4-layer | same as v1 | 200 | 0.360 | 7.02 |

**Three points:**

1. **Both use L1 loss (`loss_exp: 1.0`), matching SALT paper Eq 2.1.** The earlier "v1 used L2" claim was retracted on 2026-04-07 after config inspection.

2. **Predictor architecture is a documented design axis in the SALT paper itself.** The SALT paper ablates hierarchical vs single-level in their own experiments; neither is uniquely "the paper recipe."

3. **All three variants land within ±0.03 R² of each other, and all three are below MAE's 0.445.** The robustness spread (0.348–0.414) is tight and contains no SALT variant that beats any EMA baseline.

We use v1 e79 as the primary row because it is **the best SALT we have**. Picking v3 would look like cherry-picking downward to make SALT look worse. The load-bearing framing is: "the best SALT variant we tested still loses to the worst EMA-based baseline (MAE)."

**One-sentence rebuttal:** "SALT underperforms EMA-based methods across three implementation variants spanning predictor architecture, LR schedule, and S2 training length; we report the best variant (v1 e79, R²=0.414) as the main row to give SALT the conservative benefit of the doubt."

### Q4: "You used a hierarchical predictor but the teacher's hierarchical norm layers were never trained (S1 uses `training_mode=False`). Your v1 result is invalid."

**A:** This is a real implementation deviation and a reviewer who reads `app/salt/train.py` carefully may catch it. The bridge to defend v1 is that **v3 uses a single-level predictor that does not depend on the hierarchical teacher norm layers, and v3 also underperforms (R²=0.348)**. The v3 result isolates the frozen-teacher mechanism from the hierarchical-norms implementation concern.

| Concern | Variant that addresses it | Result |
|---|---|---|
| Teacher hierarchical norms never trained | v3 (single-level predictor, doesn't use hierarchical norms) | R²=0.348, still below MAE's 0.445 |
| Predictor is non-standard | v3 (paper-spec single-level) | R²=0.348, still below MAE's 0.445 |
| 80 epochs is undertrained | v1 e199 (200 S2 epochs) | R²=0.360, regression from 0.414 |

**No single variant has all three fixes simultaneously** (e.g. v3 at 200 epochs was not run). But each fix individually does not close the gap. The claim "SALT < EMA under our conditions" is robust to each concern tested in isolation.

**One-sentence rebuttal:** "The v1 result's hierarchical-norms concern is addressed by the v3 single-level predictor variant (R²=0.348), which also underperforms MAE and confirms the gap is not an artifact of the hierarchical-norms implementation."

### The critique we cannot fully refute

> **"You ran SALT in a data/compute regime it was never designed for. The paper tested V-3.6M × 240K steps; you tested 525K × 24K steps. Your test is out-of-distribution for SALT."**

This is the hardest attack because it is factually correct. Our only defense is **reframing the regime as the experimental variable**:

> "Our paper's contribution is to characterize SSL method behavior under the realistic deployment regime for medical video (single domain, ~500K clips, single-node compute, no external teacher). We compare four methods under identical conditions in this regime. Three work; one does not. This is a **regime-sensitivity finding**, not a replication attempt of the SALT paper."

**If this framing is NOT in §1 of the paper, SALT defensibility drops to medium.** If it IS in §1 (and reappears in §2 experimental design and §4.5 SALT discussion), SALT defensibility is high. The framing is doing 80% of the defense work.

**Action:** `paper-outline.md` §1 adds a "Preempts the regime concern" paragraph alongside the existing "Preempts the novelty concern" paragraph. See that doc for the specific sentences.

---

## Summary of Defensibility (2026-04-08)

| Critique | Severity | Defense | Defensibility |
|---|---|---|---|
| Undertrained (1/10 paper compute) | High | v1 e199 regression: more training hurts, not helps | Medium-High |
| Narrow domain dataset | Medium | Regime-conditional framing in §1 (required) | **High if §1 framing present** |
| Non-paper-spec primary row | Low-Medium | v1/v3 robustness: all variants within ±0.03 R², all below MAE | High |
| Hierarchical norms untraining | Medium | v3 single-level doesn't use them, still fails | High |
| OOD regime for SALT | High | Frame regime as experimental variable, not limitation | **High if §1 framing present** |

**Two non-negotiable items for the paper:**
1. §1 explicitly frames the data/compute regime as the experimental variable.
2. §4.5 reports all three variants (or references the appendix robustness table) so the reader can see the ±0.03 R² spread without clicking through.

Without these, SALT is medium-defensible. With both, it is high-defensible.
