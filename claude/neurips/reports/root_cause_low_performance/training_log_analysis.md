# Training-log root-cause analysis — MCC 762 & FJ 776

Both training CSVs pulled from S3. The MCC CSV does **not** log γ or `pred_delta_from_A` directly; the FJ CSV does log 42 diagnostic columns. Key findings from trajectories:

## MCC 762

### The CSV

`runs/mcc_target_anchored_25of100_762/checkpoints/log_r0.csv` (16,250 rows, ~650 iter/epoch, 25 epochs).

Columns: `epoch, itr, loss, intraview, crossview, iter-time(ms), data-time(ms)`.

**γ is not in the CSV.** `pred_delta_from_A` is not in the CSV either. The only diagnostic is the gap between `intraview` (= `L_vjepa_self` = L_p(pred_base, h_B)) and `crossview` (= `L_mcc` = L_p(pred_anchored, h_B)). `crossview < intraview` only if the adapter reduces loss → evidence γ > 0 and the adapter is doing something.

### Intraview vs crossview gap

| Epoch | mean intraview | mean crossview | gap (iv − cv) | gap % |
|---:|---:|---:|---:|---:|
| 1 | 0.491 | 0.491 | 0.000 | 0.03% |
| 5 | 0.498 | 0.495 | 0.004 | 0.73% |
| 10 | 0.500 | 0.492 | 0.008 | 1.67% |
| 15 | 0.546 | 0.532 | 0.015 | 2.71% |
| 20 | 0.546 | 0.526 | 0.020 | 3.71% |
| 25 | 0.539 | 0.509 | 0.030 | 5.49% |

**Findings:**

1. **The adapter IS active (γ > 0 empirically).** The crossview loss drops ~5.5% below intraview by ep 25. At γ=0 the gap would be exactly 0 (by construction). So the adapter learned to route useful information from clip_a — the MCC objective was not dead.

2. **The vanilla V-JEPA loss `L_vjepa_self` DID NOT DECREASE across training.** It starts at 0.491 (ep 1), drifts UP to 0.546 (ep 15) and stays there. Base e125 continuation (pretrain_21 e101–e130) has mean loss ~0.47 and slowly decreasing. **MCC's clip_b encoder is worse at vanilla V-JEPA prediction after 25 epochs than base at the same compute.**

3. **The encoder is not learning useful representations for clip-level prediction.** The only improvement in total loss comes from the adapter learning to use clip_a; without clip_a the encoder is objectively worse than vanilla continuation at the same compute.

### Why `L_vjepa_self` stays high

The encoder receives gradients from both terms:
- `∂L_vjepa_self / ∂encoder` — trains encoder to make `pred_base` good at predicting `h_B`
- `∂L_mcc / ∂encoder` — trains encoder to emit tokens at clip_a that are useful keys/values for the adapter

These are pulling in orthogonal directions: the encoder is being asked to (a) make clip_b visible tokens + predictor output match teacher clip_b, AND (b) make clip_a encoder output a useful source for cross-attending to predict clip_b. Objective (b) does not require the encoder to produce "good" clip representations — just tokens that cross-attend well with another clip's predictor output.

**The adapter's success at lowering MCC loss does not help the encoder represent individual clips better.** This is consistent with the 40% weight drift from e100 → the encoder moved a lot, but not toward "better clip V-JEPA representations."

### Why single-clip A4C LVEF probes bad

The probe sees only one clip at a time — no adapter, no clip_a. It uses the encoder's representation of clip_b only. That representation has been trained under a split objective where clip-level quality was only one of two goals, and the second goal (be a useful source for clip_a cross-attention) pulled the encoder away from clip-level quality. Result: the probe sees a worse single-clip encoder than vanilla continuation.

### Missing diagnostics

The MCC CSV should have logged γ, pred_delta_from_A, and cross-modality / cross-family pair fractions. It didn't. This is fixable for future MCC runs but can't be recovered for 762. Only indirect evidence (crossview − intraview gap) is available.

## FJ 776

### The CSV

`echomv_jepa/full_joint_restart_v2_30k_runs/776/log_r0.csv` (600 rows at step intervals of ~50 steps, 30,000 total steps).

Columns: 42 diagnostics including `loss_clip_vjepa_true`, `loss_clip_consistency`, `loss_study`, `loss_nce`, `loss_cov`, `loss_anchor_*`, `loss_sv`, all lambdas, layerwise cosines, grad norms, EMA deltas.

### Trajectory (step-bin-averaged, every 1500 steps)

| step | clip_vj | clip_cons | study | sv | anchor | λ_anchor | total |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.456 | 0.0209 | 0.038 | 0.052 | 0.003 | 0.0497 | 0.474 |
| 1500 | 0.438 | 0.0191 | 0.007 | 0.013 | 0.005 | 0.0475 | 0.448 |
| 6000 | 0.430 | 0.0162 | 0.004 | 0.007 | 0.007 | 0.0311 | 0.434 |
| 10500 | 0.442 | 0.0150 | 0.004 | 0.007 | 0.009 | 0.0117 | 0.446 |
| 15000 | 0.428 | 0.0146 | 0.003 | 0.007 | 0.010 | 0.0050 | 0.439 |
| 20000 | 0.406 | 0.0145 | 0.004 | 0.008 | 0.013 | 0.0050 | 0.412 |
| 25500 | 0.413 | 0.0148 | 0.004 | 0.007 | 0.013 | 0.0050 | 0.419 |
| 28500 | 0.407 | 0.0151 | 0.003 | 0.009 | 0.013 | 0.0050 | 0.411 |

### Findings

**1. `loss_clip_vjepa_true` is the dominant loss term** (0.43 vs 0.03 study vs 0.008 SV) **and it does decrease** — from 0.456 → 0.407. That's about the same improvement as vanilla V-JEPA continuation gets in the same compute range. The clip encoder is training as a clip V-JEPA learner.

**2. Study loss collapsed to ~0 by step 1500** (0.038 → 0.004). Study teacher/student cosine became essentially 1.0 within 1 epoch. The study-level objective is not providing useful gradient signal to the clip encoder after the first few hundred steps.

**3. NCE loss: 64% of steps had NCE = 0.0.** The `study_nce_fallback_fraction` mean is 0.016 but **382/600 steps report NCE loss = 0.0**. This means either the NCE path was almost always returning trivially-zero (e.g., because all positives are closer than all negatives and margin is 0) or the valid-negative pool often couldn't form. This objective contributed nothing most of the time.

**4. SV branch loss: 0.008, stable.** `sv_valid_fraction = 0.5` always, `a4c_sv_count = 1` always, `K_actual_mean = 8`. SV branch is firing but loss is small and stable — not adding or removing much signal.

**5. Anchor cosine drops from 0.998 (step 50) to 0.985 (step 30k).** The clip encoder *did* drift from e100, but slowly:

| step | cos_b0 | cos_b6 | cos_b12 | cos_b18 | cos_b23 |
|---:|---:|---:|---:|---:|---:|
| 50 | 0.9997 | 0.9946 | 0.9960 | 0.9979 | 0.9999 |
| 10k | 0.940 | 0.799 | 0.910 | 0.963 | 0.999 |
| 20k | 0.890 | 0.740 | 0.883 | 0.948 | 0.999 |
| 30k | 0.867 | 0.720 | 0.871 | 0.939 | 0.998 |

Block 6 moved the most (cosine 0.999 → 0.720, ~27° direction shift). Block 0 is also mobile (~30° shift). Block 23 barely moved. This is a healthier drift pattern than MCC's uniform 40% — middle layers adapted while output head stayed near e100.

**6. Study-matched-rank-top-1 is mostly 1.0 ("perfect" study matching) but `metadata_only_study_gap ≈ 0.006`.** The "metadata-only" baseline (predicting study identity from view/modality/phase counts alone) is almost as good as the full model. **The study transformer is mostly learning metadata shortcuts, not content.**

**7. Clip grad norm ~0.3, study grad norm 0.001–0.3 (erratic).** Study-side gradients are sparse and noisy. The clip encoder gets most of its gradient from the clip V-JEPA term.

### Interpretation

FJ is effectively running **vanilla V-JEPA on ~1 of K=8 clips per step, with negligible study signal, small anchor retention, and an SV branch that fires but provides small signal**. The primary compute engine is the clip V-JEPA path.

Per-step compute: 128 clips/step (16 studies × K=8). Only ~1 of K=8 clips is used for clip V-JEPA loss → **effective V-JEPA throughput is 16 clips/step**, vs 1024 clips/step for base e125. With 30k steps, FJ did 30k × 16 = **480k effective V-JEPA clip forwards**, vs base e125's 7.5k × 1024 = 7.68M. FJ got **~16× less V-JEPA gradient** than base e125 despite using similar wall-clock compute. Most of FJ's compute went into: (a) full-clip encoder forwards through 8 clips per step to feed a study transformer that collapsed to triviality, (b) anchor loss forwards, (c) EMA teacher forwards.

**The FJ clip encoder is effectively undertrained on V-JEPA by a large factor.** The healthy layerwise drift pattern and 0.41 final V-JEPA loss suggest the encoder trained well *on the reduced signal budget it got*, but that budget is much smaller than base e125.

### Why single-clip A4C LVEF probes will look bad

For single-clip downstream tasks, FJ's clip encoder is approximately "base e100 + very small V-JEPA continuation." Expected R² somewhere between e100 and e125, closer to e100.

## Summary

| Aspect | MCC 762 | FJ 776 | base e125 |
|---|---|---|---|
| L_vjepa_self final | **0.54 (rose)** | 0.41 (fell) | 0.47 (fell) |
| Effective V-JEPA clip-forwards | ~8M (1.2× loss on clip_b) | ~480k (K=8 subset) | 7.68M |
| Weight drift b6 (vs e100) | ~38% (L2) | ~28% (1 − cos 0.720) | ~23% (L2) |
| Adapter / auxiliary signal | γ > 0 (gap grew to 5.5%) | Study loss collapsed; NCE mostly 0 | N/A |
| Primary gradient | Clip V-JEPA + cross-clip adapter | Clip V-JEPA dominant | Clip V-JEPA only |

### MCC diagnosis

- **Adapter active** (γ grew): proven by crossview < intraview gap widening.
- **But encoder degraded on standalone clip V-JEPA**: `L_vjepa_self` went UP over training (0.49 → 0.54), while base e125 continuation goes DOWN (0.47 → 0.47).
- **Single-clip probe sees degraded encoder** because the probe uses the standalone clip representation, not the adapter-augmented prediction.
- **Most likely root cause**: the clip_a encoder forward produces gradients that train the encoder to be a good "key/value source" for cross-clip attention, which conflicts with being a good "standalone representation of clip_b." The 0.2·L_mcc weighting wasn't enough to contain this conflict because the adapter's cross-attention makes the clip_a path's gradient flow non-trivially into the encoder.

### FJ diagnosis

- **Clip V-JEPA path is active and works** — loss falls from 0.46 → 0.41.
- **But only ~1/8 of each step's clip forwards get V-JEPA gradient** → ~16× less V-JEPA signal than base e125 at the same wall-clock compute.
- **Study-level objective collapsed** to triviality (loss ~0.003) and **mostly just learned metadata shortcuts** (`metadata_only_gap ≈ 0.006`). Doesn't provide meaningful gradient to the clip encoder.
- **NCE mostly zero** (382/600 steps). Not contributing.
- **SV branch** fires but at small loss magnitude (~0.008). Likely not a major gradient contributor.
- **Healthy layerwise drift pattern**: middle blocks moved, output block stayed near e100. Not destructive.

### Core problem shared by both

Both MCC and FJ **added secondary objectives that consumed compute without producing useful signal, leaving the primary V-JEPA objective either damaged (MCC) or starved (FJ).**

- MCC: the secondary objective (cross-clip adapter) became active but degraded the standalone encoder representation that single-clip probes use.
- FJ: the secondary objectives (study, NCE, SV) collapsed or starved, leaving the clip encoder undertrained on V-JEPA by ~16×.

Neither objective, as implemented, is a better clip representation learner than vanilla V-JEPA continuation. Both are architecturally sound; both fail for different compute-allocation reasons.

## Recommendations for next diagnostic step

1. **Wait for EchoNet-Dynamic LVEF probes 794 (MCC) and 795 (FJ) to complete.** These use the **same dataset** as MV-PhaseRel (0.699), MV-PairedIntra (0.670), TokenRel-Motion (0.667). The trajectories will directly indicate relative ranking.

2. If MCC 794 final R² ~0.40–0.55 and FJ 795 final R² ~0.55–0.65, the hypothesis ("MCC degraded the encoder, FJ undertrained the encoder") is confirmed.

3. If either exceeds 0.65, revisit — maybe the secondary objectives did help in ways that don't show up in the training CSV.

4. **Do not rerun either pretraining** without changing: (MCC) explicit γ schedule, or no-grad on clip_a, or lower λ_mcc; (FJ) raise clip V-JEPA subset to K-out-of-K or remove study/NCE if they collapse to 0.
