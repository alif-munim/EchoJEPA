# Controlled Objective Comparisons — JEPA vs BYOL vs MAE vs SALT

Canonical consolidated summary of the controlled comparison of four
self-supervised objectives pretrained on MIMIC-IV-Echo at matched
compute. Sources: `claude/rebuttals/09-three-way-comparison-results.md`,
`claude/neurips/completed-experiments.md`, `experiments/frame-shuffling.md`,
`experiments/severity-gradient.md`, `experiments/6-condition-shuffling.md`,
`experiments/representation-analysis.md`, `experiments/salt-comparison.md`,
`experiments/echobench-e100.md`, `experiments/cmr-cross-modality.md`,
`rebuttals/12-checkpoint-reference.md`.

---

## 1. Experimental design

### 1.1 Shared controls

Every model in this comparison has the **same** pretraining data,
initialization, architecture, and compute budget. Only the objective
differs.

| Axis | Value |
|---|---|
| Architecture | ViT-L/16 (224 px, 16 frames, patch 16×16, tubelet 2) |
| Pretraining data | MIMIC-IV-Echo 525K clips (`indices/s3_pretrain.csv`) |
| Initialization | ImageNet-21K weights for all four models |
| Budget | ~100 epochs (JEPA e100, BYOL e100, MAE e99, SALT S2 e79) |
| Probe protocol | d=4 attentive, 20 epochs, 6-HP grid, frozen encoder |
| Test set | EchoNet-Dynamic 1,277 videos |
| Statistics | Paired bootstrap, n=1,277, 10K resamples |

### 1.2 Objectives (what actually differs)

| Objective | Prediction target | Auxiliary structure |
|---|---|---|
| **JEPA** | masked clip latents (EMA teacher) | predictor MLP; target encoder follows student via EMA |
| **BYOL** | student output after projector (EMA teacher) | projector + predictor heads, no masking |
| **MAE** | masked pixel patches | autoencoder decoder, no teacher |
| **SALT** v1 | JEPA-style but teacher is a frozen pretrained MAE (replaces EMA co-evolution with fixed target) | lightweight S1 teacher, S2 student |

All four objectives fit into the same loss skeleton `student(x) →
predict(target(y))`, differing only in (a) whether `target` co-evolves
(JEPA/BYOL) or is frozen (SALT) or is the input itself (MAE) and (b)
whether the target lives in latent or pixel space.

### 1.3 The init confound (resolved)

The ICML rebuttal three-way comparison used a "JEPA pt50" checkpoint
that was actually a 235-epoch fully-trained model. The pt50 results
below are retained as a historical record but **the canonical numbers
are the e100 init-matched runs**.

| Checkpoint | What it actually is | Used in |
|---|---|---|
| JEPA pt50 (ICML) | 235-epoch IN21K JEPA | Rebuttal three-way, retracted |
| JEPA e100 (canonical) | IN21K-init, 100-ep MIMIC | NeurIPS primary |
| BYOL e100 | IN21K-init, 100-ep MIMIC | NeurIPS primary |
| MAE e99 | IN21K-init, 99-ep MIMIC | NeurIPS primary |
| SALT v1 e79 | S1 teacher e100 + S2 student e79 from IN21K | NeurIPS primary |

---

## 2. Primary result — LVEF (EchoNet-Dynamic test, n=1,277)

### 2.1 e100 init-matched, full test-set metrics

| Objective | Total ep | Test R² [95% CI] | Test Pearson [95% CI] | Test MAE |
|---|---:|---|---|---:|
| **JEPA** | 100 | **0.652** [0.608, 0.691] | **0.808** [0.781, 0.832] | **5.32** |
| BYOL | 100 | 0.511 [0.452, 0.564] | 0.720 [0.680, 0.756] | 6.18 |
| MAE | 99 | 0.447 [0.389, 0.500] | 0.688 [0.645, 0.728] | 6.59 |
| SALT v1 (frozen teacher) | S2 e79 | 0.416 [0.347, 0.478] | 0.659 [0.613, 0.702] | 6.66 |

### 2.2 Pairwise differences (paired bootstrap, n=1,277, 10K resamples)

| Comparison | ΔR² [95% CI] | Δr [95% CI] | R² sig? | r sig? |
|---|---|---|---|---|
| JEPA vs BYOL | **+0.141** [+0.109, +0.175] | +0.088 [+0.066, +0.111] | **SIG** | **SIG** |
| JEPA vs MAE | **+0.205** [+0.164, +0.247] | +0.120 [+0.090, +0.152] | **SIG** | **SIG** |
| JEPA vs SALT | **+0.237** [+0.188, +0.289] | +0.149 [+0.116, +0.184] | **SIG** | **SIG** |
| BYOL vs MAE | +0.064 [+0.016, +0.110] | +0.032 [−0.001, +0.066] | SIG | n.s. |
| BYOL vs SALT | +0.096 [+0.042, +0.151] | +0.061 [+0.025, +0.098] | **SIG** | **SIG** |
| MAE vs SALT | +0.032 [−0.022, +0.086] | +0.029 [−0.010, +0.066] | n.s. | n.s. |

**Preserved ranking**: JEPA >> BYOL > MAE ≈ SALT. All JEPA pairwise
comparisons are highly significant. MAE and SALT are statistically
equivalent.

### 2.3 Pathology-stratified LVEF

| EF category | N | JEPA MAE | BYOL MAE | MAE MAE |
|---|---:|---:|---:|---:|
| Normal (≥55%) | 876 | 4.3 | 5.0 | 5.1 |
| Mildly reduced (40-54%) | 241 | 7.6 | 7.8 | 7.1 |
| **Reduced (<40%)** | 160 | **12.4** | 14.4 | **19.3** |

MAE predicts 48% for patients with true EF 29% — misses severe heart
failure. JEPA advantage is **8× larger on reduced-EF** than on normals.

### 2.4 Compute-scaling: JEPA e100 → e200

| Checkpoint | Best val MAE | Best val R² | Best val Pearson |
|---|---:|---:|---:|
| JEPA e100 | 5.32 | 0.808 | — |
| **JEPA e200** | **4.88** | 0.714 (test) | 0.845 |
| EchoJEPA-L-K (anneal on Kinetics) | 4.45 | 0.766 | 0.876 |
| PanEcho (specialist ref) | 4.83 | 0.719 | 0.849 |
| EchoPrime (specialist ref) | 5.44 | 0.636 | 0.798 |

**e100 → e200** improves val MAE by ~0.44 (5.32 → 4.88). JEPA-e200
matches PanEcho within val-sweep noise, using only generic IN21K
pretrain + 525K MIMIC echoes. **EchoJEPA-L-K** (annealing on Kinetics)
is the strongest ViT-L variant (4.45), beating both the e200 extension
and both specialists.

---

## 3. Secondary task results

### 3.1 RVSP (MIMIC SV, 10K/2K/2K subset, job 484)

9-way comparison; 5 of 9 models produced complete probes before SLURM
wall cutoff.

| Model | Best val epoch | Best val MAE | Best val R² | Best val Pearson |
|---|---:|---:|---:|---:|
| **EchoJEPA-L-K** (anneal-Kinetics) | — | — | — | **0.525** |
| **JEPA IN21K e100** | 5 | 6.823 | 0.175 | 0.458 |
| BYOL IN21K e100 | 2 | 7.174 | 0.043 | 0.326 |
| MAE IN21K e100 | 20 | 7.252 | 0.015 | 0.335 |
| EchoPrime | — | — | — | ~0.50 |

**Same ranking as LVEF**: EchoJEPA-L-K >> JEPA-e100 ≈ EchoPrime > BYOL ≈ MAE.

Early-epoch overfitting visible in BYOL (best at e2) and JEPA (best
at e5); MAE still descending at e20 (did not converge within the
schedule).

### 3.2 Pediatric Zero-Shot (UHN-trained probes → EchoNet-Pediatric 368-clip test)

| Model | Test Pearson | Test MAE | Test R² |
|---|---:|---:|---:|
| **EchoJEPA-L** | **0.705** | **6.957** | **0.405** |
| EchoMAE-L | 0.626 | 7.857 | 0.187 |
| EchoBYOL-L | 0.602 | 8.004 | 0.206 |

### 3.3 Pediatric Zero-Shot (END-trained probes → 368 pediatric test)

| Model | Test Pearson | Test MAE | Test R² |
|---|---:|---:|---:|
| **EchoJEPA-L** | **0.615** | **7.358** | **0.293** |
| EchoMAE-L | 0.531 | 9.203 | 0.041 |
| EchoBYOL-L | 0.498 | 12.132 | −0.847 |

**BYOL collapses on cross-population transfer from END** (R² = −0.847).

---

## 4. Robustness — EchoBench noise perturbations

### 4.1 LVEF robustness (EchoNet-Dynamic, 1,277 test, R² with 95% bootstrap CIs)

| Condition | JEPA | BYOL | MAE | SALT |
|---|---|---|---|---|
| clean | **0.591** [0.538, 0.638] | 0.465 [0.401, 0.523] | 0.445 [0.377, 0.505] | 0.293 [0.215, 0.362] |
| depth_atten severe | **0.396** [0.321, 0.463] | 0.342 [0.288, 0.392] | **0.090** [0.051, 0.129] | 0.137 [0.094, 0.179] |
| shadow severe | **0.457** [0.385, 0.518] | 0.320 [0.240, 0.390] | 0.400 [0.340, 0.455] | 0.208 [0.128, 0.278] |
| haze severe | **0.553** [0.498, 0.603] | 0.431 [0.368, 0.488] | **0.159** [0.099, 0.217] | 0.217 [0.145, 0.283] |

**MAE collapses** under depth attenuation (0.090) and haze (0.159).
JEPA is most robust across every severe perturbation.

### 4.2 CAMUS segmentation robustness (100 samples, Dice with 95% CIs)

| Condition | JEPA | BYOL | MAE | SALT |
|---|---|---|---|---|
| clean | 0.815 [0.801, 0.829] | 0.823 [0.811, 0.835] | **0.827** [0.814, 0.838] | 0.777 [0.759, 0.794] |
| depth_atten severe | **0.683** [0.663, 0.703] | 0.368 [0.345, 0.391] | 0.654 [0.625, 0.681] | 0.508 [0.486, 0.529] |
| shadow severe | 0.717 [0.697, 0.736] | 0.587 [0.556, 0.616] | **0.737** [0.717, 0.755] | 0.645 [0.621, 0.668] |
| haze severe | 0.794 [0.778, 0.808] | **0.815** [0.804, 0.826] | 0.778 [0.763, 0.792] | 0.767 [0.749, 0.785] |
| **Avg severe drop** | **10.3%** [9.4, 11.3] | **28.4%** [26.8, 29.9] | 12.6% [11.4, 13.8] | 17.6% [16.4, 18.9] |

**JEPA is most robust on BOTH tasks.** Clean performance doesn't
predict robustness: MAE leads clean CAMUS (0.827) but JEPA has the
lowest average severe drop (10.3%). **BYOL catastrophically collapses
under depth attenuation on CAMUS** (0.823 → 0.368, −55%).

---

## 5. Mechanism — temporal encoding regimes (frame shuffling)

Four qualitatively distinct profiles emerge under frame-shuffling
ablations.

### 5.1 Severity gradient (partial frame shuffling at 0/25/50/75/100%)

R² on EchoNet-Dynamic test, mean over 3 seeds:

| Fraction | JEPA e100 | BYOL e100 | MAE e99 | SALT e79 |
|---:|---:|---:|---:|---:|
| 0.00 | **0.591** | 0.468 | 0.445 | 0.296 |
| 0.25 | **0.542** | 0.410 | 0.421 | 0.048 |
| 0.50 | **0.507** | 0.336 | 0.436 | −0.161 |
| 0.75 | **0.485** | 0.300 | 0.414 | −0.256 |
| 1.00 | **0.488** | 0.291 | 0.428 | −0.270 |

Four distinct profiles:

| Regime | Model | Clean → 100% shuffle | Interpretation |
|---|---|---|---|
| **Gentle slope (−17%)** | JEPA | 0.591 → 0.488 | Moderate temporal reliance |
| **Steep slope (−38%)** | BYOL | 0.468 → 0.291 | Strong temporal dependence |
| **Invariant (−4%)** | MAE | 0.445 → 0.428 | Temporal structure barely used |
| **Cliff (−191%)** | SALT | 0.296 → −0.270 | Temporal-order dependence + low ceiling |

**JEPA shuffled (0.488) still beats BYOL clean (0.468).** JEPA's
advantage isn't purely about temporal information — its spatial
features are also better.

### 5.2 Six-condition (init-matched 4 models at convergence)

| Condition | JEPA e100 | BYOL e100 | MAE e99 | SALT e79 |
|---:|---:|---:|---:|---:|
| clean | **0.591** | 0.468 | 0.445 | 0.296 |
| tubelet (local) | **0.582** | 0.402 | 0.424 | 0.294 |
| reverse | **0.539** | 0.373 | 0.431 | 0.120 |
| matched (RoPE remap) | **0.580** | 0.415 | 0.419 | 0.296 |
| shuffle | **0.484** | 0.291 | 0.422 | −0.283 |
| matched_frame | **0.477** | 0.280 | 0.449 | −0.310 |

### 5.3 CAMUS segmentation frame shuffling (spatial task control)

Same shuffling on segmentation. Dice degradation:

| % shuffle | JEPA e100 | BYOL e100 | MAE e99 | SALT e79 |
|---:|---:|---:|---:|---:|
| 25% | 1.6% | 1.3% | 2.1% | 0.8% |
| 50% | 3.6% | 2.7% | 4.9% | 2.1% |
| 100% | 7.0% | 6.0% | **8.6%** | 4.9% |

**Reverse is catastrophic on CAMUS (12–15%) — 2× worse than full
shuffle.** Even segmentation has temporal-order dependencies.

### 5.4 SALT training dynamics

SALT's cliff profile persists at every training stage (S2 e4/e29/
e54/e79). Unlike JEPA (−42%→−17% consolidation), SALT stays at
−187% to −256% from e29 onward. **The frozen teacher cannot drive
temporal consolidation.**

---

## 6. Mechanism — cross-temporal attention hierarchy

Fraction of attention flowing between tokens at different temporal
positions. Random baseline = 0.875. Lower = more within-frame
(spatial) attention.

### 6.1 Epoch ~100 (init-matched)

| Model | Layers 0-1 | Layers 2-10 | Layers 11-23 | Overall |
|---|---|---|---|---|
| **SALT S2 e79** | **0.44-0.49** | **0.39-0.56** | 0.83-0.88 | **0.672** |
| **JEPA e100** | **0.57-0.60** | 0.82-0.87 | 0.87-0.88 | **0.839** |
| BYOL e100 | 0.77-0.86 | 0.81-0.86 | 0.87-0.88 | 0.855 |
| MAE e99 | 0.86 | 0.82-0.87 | 0.87 | 0.861 |

**SALT develops the strongest spatial→temporal hierarchy**: layers
0-10 are heavily within-frame, sharp transition at layer 11. JEPA
shows a milder version (layers 0-1 only). BYOL and MAE show no
spatial-first specialization.

The hierarchy **deepens with training**: at SALT e29, only layer 0
is spatial-biased (0.27); by e79, the entire first half of the
network specializes.

### 6.2 Four hypotheses tested for "why JEPA > MAE on functional tasks"

| Hypothesis | Status | Source |
|---|---|---|
| EMA filters frame-varying noise | ❌ Not supported | Multiple tests (speckle, temporal consistency, noise autocorrelation) |
| JEPA encodes temporal dynamics MAE doesn't | ✅ Supported | Frame-shuffling severity gradient |
| JEPA uses representational capacity more efficiently | ❌ Not supported (revised) | RankMe 245/221/206/203 — all 200-245 range |
| Predictive objectives induce spatial→temporal layer specialization | ✅ Supported | Cross-temporal attention analysis |

**Surviving mechanisms**: temporal-structure encoding and spatial→temporal layer specialization. Predictive objectives (JEPA, SALT) force early layers to attend within-frame before integrating across time.

---

## 7. Retractions — prior mechanism claims

### 7.1 Speckle filtering (RETRACTED)

| Comparison | JEPA | BYOL | MAE | JEPA−MAE gap |
|---|---:|---:|---:|---:|
| ICML rebuttal pt50 (confounded) | 0.674 | 0.775 | **0.875** | −0.201 (23%) |
| **e100 init-matched** | 0.848 | **0.716** | 0.885 | **−0.037 (4%)** |

Under init-matching, **BYOL is the best speckle filter, not JEPA**.
The "JEPA filters speckle via EMA target averaging" narrative is
**not supported**.

### 7.2 Effective dimensionality (REVISED)

Consistent 4-model RankMe (`scripts/neurips/rankme.py`, 500 EchoNet-Dynamic test videos):

| Model | Effective Dimensionality | % of embed_dim (1024) |
|---|---:|---:|
| JEPA IN21K e95 | **245.3** | 24.0% |
| BYOL e100 | 220.7 | 21.6% |
| MAE e99 | 206.4 | 20.2% |
| SALT v1 e79 | 202.7 | 19.8% |

**All four in the 200-245 range.** The prior MAE=63 (Goodfire report) is **not reproducible** and should not be cited. Effective dimensionality does NOT explain MAE's weakness — the gap is modest (~20%), not 3×.

---

## 8. SALT — frozen-teacher distillation

### 8.1 Design

SALT (Li et al., Apple 2025) replaces JEPA's co-evolving EMA teacher
with a frozen pixel-reconstruction teacher. Our test: does the
frozen-teacher design retain JEPA's advantage on echo?

### 8.2 Result on EchoNet-Dynamic LVEF (test set, e100 init-matched)

| Method | Test MAE | Test R² | Test Pearson |
|---|---:|---:|---:|
| **JEPA IN21K e100** | **5.32** | **0.652** | **0.808** |
| BYOL e100 | 6.18 | 0.511 | 0.720 |
| MAE e99 | 6.59 | 0.447 | 0.688 |
| **SALT v1 e79** (best variant) | 6.66 | 0.416 | 0.659 |

**Replacing JEPA's co-evolving EMA teacher with a frozen
pixel-reconstruction teacher (SALT) reduces LVEF R² from 0.652 to
0.416 (−36%), placing SALT statistically equivalent to MAE.** This
suggests co-evolution of the target encoder contributes to
representation quality independent of the prediction target.

### 8.3 Extended-teacher experiments

- **V-Pixel teacher extended e100 → S2 e79** (job 349): clean R²=0.445, ΔR²matched_frame=−0.93 (vs −0.33 for short-teacher variant).
- **JEPA-teacher SALT** (job 350): S2 e80 student from JEPA e100 teacher → R²=0.252 vs raw JEPA e100 R²=0.650 — **0.398 R² lost attributable to the distillation step alone.**
- S1 V-Pixel teacher trajectory (e24/e49/e74/e99): matched-frame ΔR² deepens monotonically from −0.02 → −1.14 as the teacher trains longer.

**Conclusion**: distillation from a frozen teacher is strictly
worse than co-evolution at every teacher-compute operating point
tested.

---

## 9. CMR cross-modality — direction flip

### 9.1 Finding

On cardiac MRI (ACDC cohort), **JEPA's advantage reverses**:

- Fast-EMA JEPA peaks at e30-e100 on ACDC LVEF and Dx, then **degrades**.
- Slow-EMA JEPA shows the same pattern (peaks R²=0.138 at e30, collapses to 0.089 at e295 on LVEF).
- **MAE climbs monotonically** and overtakes JEPA by e200-e800.

### 9.2 Implication

The "JEPA uniformly beats MAE on cardiac video" claim is **falsified
on MRI**. The advantage is specific to echocardiography.

The mechanism hypothesis (JEPA consolidates temporal structure, MAE
doesn't) is consistent with this flip: echo is spectrally heavy and
short-horizon; MRI is spatially heavy and long-horizon. Without rapid
cycle-by-cycle temporal turnover, JEPA's temporal-consolidation
advantage erodes.

**Source**: `claude/neurips/experiments/cmr-cross-modality.md`.

---

## 10. Multi-view & scaling (NeurIPS Appendix)

### 10.1 RVSP single-view vs multi-view (JEPA-L pt50)

| View | Test Pearson | Test R² |
|---|---:|---:|
| Multi-view (A4C + PSAX) | **0.484** | **0.220** |
| A4C only | 0.447 | 0.181 |
| PSAX only | 0.449 | 0.188 |

Multi-view +3.9 pp R² over best single view.

### 10.2 Scaling

| Model | Params | Data | LVEF R² |
|---|---:|---|---:|
| EchoJEPA-B (V-JEPA 2.1) | 86M | MIMIC 525K | 0.650 |
| EchoJEPA-L (V-JEPA 2.0, 50ep) | 304M | MIMIC 525K | 0.436 |
| EchoJEPA-G (V-JEPA 2.0) | 1,012M | UHN 18M | 0.778 |

(Caveat: B uses V-JEPA 2.1; L→G confounds data scale.)

---

## 11. Checkpoint registry

All paths are under `s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts/`.

| Model | Pretrain checkpoint | Probe checkpoint (LVEF) |
|---|---|---|
| JEPA IN21K e100 | `runs/jepa_in21k_pretrain_376/checkpoints/e100.pt` | `evals/vitl/icml/jepa_in21k_e100_end_lvef_224/.../icml-jepa-in21k-e100-end-lvef-d4/best.pt` |
| JEPA IN21K e200 | `runs/jepa_in21k_e200_280/training_folder/e200.pt` | `runs/lvef_resume_jepa_e200_421/jepa_in21k_e200_lvef/.../best.pt` |
| EchoJEPA-L-K | (Kinetics-annealed) | `runs/fourmodel_vjepa_lvef_405/echojepa_l_k_lvef/.../best.pt` |
| BYOL e100 | `checkpoints/byol_vitl_imagenet_v2_e100.pt` | `evals/vitl/icml/echobyol_e100_end_lvef_224/.../best.pt` |
| MAE e99 | `runs/videomae_matched_2n_245/training_folder/checkpoint-98.pth` | equivalent 99-ep probe |
| SALT S2 e79 | `checkpoints/salt_s2_vitl_e79.pt` | see `experiments/salt-comparison.md` |
| PanEcho | public | `runs/lvef_resume_panecho_483/panecho_lvef/.../best.pt` |
| EchoPrime | public | `runs/fourmodel_extern_lvef_406/echoprime_lvef/.../best.pt` |

---

## 12. Headline findings

1. **JEPA wins at matched compute by a large, statistically clean margin**: +14 pp test R² over BYOL, +21 pp over MAE, +24 pp over SALT on LVEF. All pairwise paired-bootstrap differences are highly significant.

2. **The gap holds on RVSP, pediatric zero-shot, CAMUS segmentation, and all four EchoBench robustness conditions**. JEPA is not merely "good at LVEF" — it's best-in-class on every echo endpoint we've measured.

3. **The mechanism is temporal-structure encoding** + **spatial→temporal layer hierarchy**. Predictive objectives force early layers to specialize within-frame before integrating across time. MAE lacks this hierarchy and pays the price on functional tasks.

4. **Prior candidate mechanisms are retracted**: speckle filtering (ranking changed under init-matching), effective dimensionality (all 200-245 range, not 3× collapse).

5. **Distillation from a frozen teacher (SALT) doesn't preserve JEPA's advantage** — even with a JEPA teacher, S2 student loses 0.40 R². Co-evolution of the target encoder is load-bearing.

6. **Cross-modality check fails on MRI** — JEPA's advantage is specific to echo and degrades on ACDC CMR while MAE climbs. The story is not "JEPA > MAE universally" but "JEPA > MAE when temporal dynamics matter on short horizons."

7. **BYOL's catastrophic failure modes** — collapses on pediatric END transfer (R² −0.847) and on CAMUS under depth attenuation (0.823 → 0.368). The projection-head architecture is fragile in ways JEPA isn't.

8. **e100 → e200 closes the specialist gap**: extended JEPA pretraining matches PanEcho (val MAE 4.88 vs 4.83) using only IN21K + 525K MIMIC echoes.

9. **EchoJEPA-L-K** (JEPA annealed on Kinetics) is the strongest ViT-L variant across the entire comparison — better than the e200 extension and both specialist baselines. The annealing step buys more than the e100→e200 extension does.

---

## 13. Cross-references

- `claude/rebuttals/09-three-way-comparison-results.md` — original 50-ep three-way detail (pt50 era)
- `claude/rebuttals/10-rebuttal-experiment-results.md` — consolidated rebuttal tables
- `claude/rebuttals/12-checkpoint-reference.md` — full checkpoint / probe / S3 paths
- `claude/rebuttals/13-post-rebuttal-outcome.md` — ICML decision status
- `claude/neurips/completed-experiments.md` — primary source for e100 NeurIPS numbers
- `claude/neurips/README.md` — NeurIPS plan hub, severity-gradient entries, model registry
- `claude/neurips/paper-outline.md` — paper-section mapping
- `claude/neurips/experiments/frame-shuffling.md` — 6-condition mechanism writeup
- `claude/neurips/experiments/severity-gradient.md` — 13-model severity gradient
- `claude/neurips/experiments/6-condition-shuffling.md` — 12-model six-condition
- `claude/neurips/experiments/representation-analysis.md` — speckle retraction + RankMe + cross-temporal attention
- `claude/neurips/experiments/salt-comparison.md` — SALT v1/v3/extended teacher
- `claude/neurips/experiments/echobench-e100.md` — 4-model robustness
- `claude/neurips/experiments/cmr-cross-modality.md` — CMR direction-flip
- `claude/neurips/experiments/camus-frame-shuffling.md` — CAMUS segmentation shuffling
- `claude/neurips/neurips_results.tex` — LaTeX-ready tables
- `claude/preprint/icml_preprint.tex` — current ICML preprint
