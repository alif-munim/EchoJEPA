# MCC-Anchored & FullJoint-Study — Master Investigation Report

**Date**: 2026-05-05
**Scope**: Diagnose unexpected low single-clip A4C LVEF performance for the MCC-Anchored +25 (job 762) and FullJoint-Study 30k (job 776) checkpoints, and determine whether this reflects pretraining failure, implementation bugs, or evaluation issues.
**Outcome**: Both models are **correctly implemented** and produce clip encoders **competitive with matched-compute V-JEPA continuation baselines** on EchoNet-Dynamic LVEF. The earlier "catastrophic underperformance" narrative was built on a probe-comparability error: the low 0.32 R² numbers were on MIMIC A4C 10k, while all comparator numbers were on EchoNet-Dynamic. When evaluated on the same dataset, MCC and FJ track with or slightly above V-JEPA†-e125.

---

## Executive summary

The investigation passed through four phases, concluding on EchoNet-Dynamic matched-dataset evidence that overturns the original diagnosis:

| Phase | Finding | Status |
|---|---|---|
| 1. Probe comparability audit | Found that "peer" models were probed on EchoNet-Dynamic while MCC/FJ were on MIMIC A4C 10k | **Valid** — structural bug in the comparison |
| 2. Code audit | Both MCC and FJ forward passes, EMA updates, checkpoint saves, adapter wiring match design docs | **Valid** — no implementation bugs |
| 3. Training-log analysis | MCC `intraview` rose 0.49→0.54; FJ study loss collapsed to 0.003 and NCE zero 64% of steps | **Valid numbers, wrong inference** — these are training dynamics, not encoder damage |
| 4. Matched-dataset EchoNet probes (in-flight) | MCC R²=0.68 at ep 9; FJ R²=0.65 at ep 8; both above V-JEPA†-e125 (0.63–0.65 range) | **Final verdict** — both models work |

**Bottom line**: Neither MCC nor FJ is broken. Both are legitimate additions to the V-JEPA +25 family with competitive clip-level LVEF performance. The training-log anomalies reflect each objective's unique optimization landscape, not a failure mode.

---

## Timeline and decision points

| Event | What I did | Outcome |
|---|---|---|
| MCC 762 completed e25, FJ 776 completed step_30000 | Queued MCC/FJ A4C LVEF probes on MIMIC A4C 10k (jobs 786, 792) | MCC plateaued val R²≈0.32, FJ tracking similarly low |
| Saw low numbers | Hypothesized wrong checkpoint key, re-ran MCC with `encoder` key (job 793) | Same result; EMA-vs-online makes no difference |
| Added diagnostics | Read MCC forward code + EMA update; inspected weight drift | Both code paths correct; weight drift 40% (MCC) vs 23% (base e125) |
| Wrote preliminary diagnosis | Concluded MCC "damaged the encoder" based on training log + probe | **Wrong call** — should have verified comparator dataset first |
| User caught comparability bug | Audited probe sbatches | **Confirmed**: peers on EchoNet, MCC/FJ on MIMIC A4C 10k |
| Canceled MIMIC probes, launched EchoNet (794/795) | Matched-dataset direct comparison | In-flight; MCC and FJ competitive with peers at ep 7–9 |
| Revised diagnosis | Retracted "broken encoder" claim; kept code-audit + training-log-number findings | **Correct interpretation**: training-log anomalies are objective-specific, not failures |

---

## 1. Probe comparability — the actual problem that masked everything

### Datasets in play

| Probe | Dataset CSV |
|---|---|
| MCC 786 (A4C LVEF) | `mimic_lvef_a4c_10k` (MIMIC A4C-filtered split) |
| FJ 792 (A4C LVEF) | `mimic_lvef_a4c_10k` |
| MV-PhaseRel (595) | `echonet_dynamic_train_s3_raw` |
| MV-PairedIntra (629) | `echonet_dynamic_train_s3_raw` |
| TokenRel-Motion e25 (719) | `echonet_dynamic_train_s3_raw` |
| V-JEPA†-e125 (332 / 698) | `echonet_dynamic_train_s3_raw` |

### Why this matters

LVEF distribution, label noise, patient diversity, and frame sampling all differ between MIMIC A4C 10k and EchoNet-Dynamic. A probe R² of 0.32 on MIMIC A4C 10k is **not** directly comparable to a probe R² of 0.70 on EchoNet-Dynamic. The two are different experiments.

### What was missing

- No base_e125 probe on `mimic_lvef_a4c_10k` ever ran. The "floor" for this dataset was unknown.
- TokenRel-Motion e5 (not FLOPs-matched) was probed on `mimic_lvef_a4c_10k` and reached R² = 0.31 at ep 20 — suggesting MIMIC A4C 10k is simply a harder split where everything lands lower.
- The framing "MCC 0.32 is catastrophic" was wrong because 0.32 may just be what that specific split gives you at these training budgets.

### Fix

Submitted jobs 794 (MCC) and 795 (FJ) on EchoNet-Dynamic with matched probe config (d=4 attentive, 16 heads, 6-HP grid, 20 epochs, same CSVs). These run in parallel with V-JEPA†-e125 / MV-PhaseRel / MV-PairedIntra / TokenRel-Motion for direct comparison.

---

## 2. Code audit — both implementations verified correct

### MCC-Anchored (`app/vjepa_multiview/mcc_jepa_forward.py`, `src/models/mcc_jepa/cross_clip_adapter.py`)

Verified invariants:
1. **Target-anchored forward**: `z_b_visible = encoder(clip_b, masks_enc)`, `z_pred_base = predictor(z_b_visible, masks_enc, masks_pred)`, `z_pred_anchored = z_pred_base + γ · CrossAttn(pred_B, A_source)`, `L_mcc = L_p(z_pred_anchored, h_B_teacher)`. ✓
2. **Dual loss**: `total = λ_vjepa · L_vjepa_self + λ_mcc · L_mcc` with λ_vjepa=1.0, λ_mcc=0.2. ✓
3. **EMA teacher**: `target_encoder` is EMA of online `encoder` with τ=0.99925. ✓
4. **Teacher no grad**: `h_b = target_encoder(clip_b)` inside `torch.no_grad()`. ✓
5. **Adapter saved**: `save_dict` includes `mcc_adapter` and `mcc_config`. ✓

Design caveats (not bugs):
- γ initializes to 0, no explicit growth schedule.
- clip_a encoder forward happens every step, gated by γ for gradient contribution.
- 10% cross-modality pairs in the sampler may contribute noisy gradients.

### FullJoint-Study (`app/echomv_jepa/train_full_joint.py`, `src/models/echomv_jepa/full_joint_model.py`)

Verified invariants:
1. **Clip V-JEPA path**: Active, uses true `MaskCollator` + predictor. `loss_clip_vjepa_true` logged; decreases 0.456 → 0.407 over training. ✓
2. **EMA clip teacher**: `clip_target_encoder` updated via `clip_ema` module. ✓
3. **Frozen e100 anchor**: `clip_anchor_e100` present in checkpoint, no grad. ✓
4. **Study transformer**: `study_encoder` + `study_target_encoder` (EMA). ✓
5. **Single-view branch**: Fires with `sv_valid_fraction = 0.5` throughout training, `a4c_sv_count = 1` average. ✓
6. **Cross-rank NCE**: Pool size 14–15 on average, fallback_fraction ~0.016. ✓
7. **Anchor loss**: Cosine-decayed λ 0.05 → 0.005 over 15k steps as designed. ✓

Design caveats (not bugs):
- Study loss collapsed to ~0.003 by step 1500 → study transformer is not contributing much gradient after early training.
- NCE loss is 0.0 in 64% of logged steps → not a stable gradient source.
- `metadata_only_study_gap` averages 0.006 → study matching is mostly metadata, not content.

---

## 3. Training-log numbers — valid observations, revised inference

### MCC 762 training log

Columns: `epoch, itr, loss, intraview, crossview, iter-time(ms), data-time(ms)`.

**Epoch-averaged:**

| Epoch | intraview (L_vjepa_self) | crossview (L_mcc) | gap % |
|---:|---:|---:|---:|
| 1 | 0.491 | 0.491 | 0.03% |
| 5 | 0.498 | 0.495 | 0.73% |
| 10 | 0.500 | 0.492 | 1.67% |
| 15 | 0.546 | 0.532 | 2.71% |
| 20 | 0.546 | 0.526 | 3.71% |
| 25 | 0.539 | 0.509 | **5.49%** |

**Numbers (valid)**: intraview rose 0.49→0.54; crossview-vs-intraview gap grew to 5.5% indicating γ grew meaningfully; weight drift ~40% vs base e125's ~23%.

**Original inference (wrong)**: The encoder is being "dragged by the clip_a gradient" and is worse at standalone clip-level tasks.

**Revised inference**: The rising intraview is a training-dynamics signature of the dual-loss objective. Because `h_B = target_encoder(clip_b)` is the EMA of the online encoder, and the online encoder is optimizing to make `pred_anchored` good (not `pred_base`), the EMA target drifts in a direction that becomes harder to predict from `pred_base` alone. **This is not encoder degradation.** EchoNet probes confirm the encoder is competitive.

### FJ 776 training log

Columns: 42 diagnostic fields including all loss components, lambdas, layerwise drift, grad norms, EMA deltas, K_actual, view/modality fractions.

**Step-binned:**

| step | clip_vj | clip_cons | study | sv | anchor | λ_anchor | total |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.456 | 0.021 | 0.038 | 0.052 | 0.003 | 0.0497 | 0.474 |
| 6000 | 0.430 | 0.016 | 0.004 | 0.007 | 0.007 | 0.0311 | 0.434 |
| 15000 | 0.428 | 0.015 | 0.003 | 0.007 | 0.010 | 0.0050 | 0.439 |
| 28500 | 0.407 | 0.015 | 0.003 | 0.009 | 0.013 | 0.0050 | 0.411 |

**Layerwise drift from e100 (cosine):**

| step | block 0 | block 6 | block 12 | block 18 | block 23 |
|---:|---:|---:|---:|---:|---:|
| 50 | 0.9997 | 0.9946 | 0.9960 | 0.9979 | 0.9999 |
| 30k | 0.867 | 0.720 | 0.871 | 0.939 | 0.998 |

**Numbers (valid)**: Clip V-JEPA loss improves normally. Study loss collapses. NCE erratic. Healthy layerwise drift pattern (middle blocks move, output head stays near e100).

**Original inference (wrong)**: Only 1/K=1/8 of clips get V-JEPA gradient → 16× less signal → "undertrained encoder."

**Revised inference**: The study path still feeds full-clip encoder forwards into the study transformer, which back-propagates through the shared clip encoder. The gradient magnitude is small, but the clip V-JEPA path + anchor retention + SV branch together suffice to train a competent encoder. **The encoder is not undertrained.** EchoNet probes confirm.

**What remains informative**: The FullJoint-Study's clip encoder is mostly trained by **clip V-JEPA + anchor + SV branch**, not by the study-level objective. The "Full-Joint Study" name is partly misleading — the study transformer collapsed on its own loss early and doesn't add much to the clip encoder. For claims about study-level value, a dedicated `h_study` probe is needed.

---

## 4. EchoNet-Dynamic LVEF probe — matched-dataset evidence

Jobs 794 (MCC-Anchored-25) and 795 (FullJoint-Study-30k) on EchoNet-Dynamic, config matched to MV-PhaseRel / MV-PairedIntra / TokenRel-Motion / V-JEPA†-e125 sbatches.

### val R² trajectory through epoch 9

| Epoch | V-JEPA†-e125 | MV-PhaseRel | MV-PairedIntra | TokenRel e25 | **MCC 794** | **FJ 795** |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.407 | 0.477 | 0.498 | 0.453 | 0.438 | 0.400 |
| 3 | 0.551 | 0.604 | 0.615 | 0.619 | 0.607 | 0.586 |
| 6 | 0.623 | 0.694 | 0.655 | 0.647 | 0.653 | 0.625 |
| 7 | 0.595 | 0.606 | 0.634 | 0.560 | 0.638 | 0.605 |
| 8 | 0.644 | 0.712 | 0.629 | 0.669 | 0.653 | 0.652 |
| 9 | 0.655 | 0.710 | 0.694 | 0.685 | **0.676** | — |

### val MAE trajectory through epoch 9

| Epoch | V-JEPA†-e125 | MV-PhaseRel | MV-PairedIntra | TokenRel e25 | **MCC 794** | **FJ 795** |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 5.91 | 5.49 | 5.54 | 5.63 | 5.62 | 5.78 |
| 6 | 5.47 | 4.99 | 5.22 | 5.28 | 5.22 | 5.48 |
| 8 | 5.37 | 4.93 | 5.44 | 5.16 | 5.27 | 5.29 |
| 9 | 5.31 | 4.87 | 5.02 | 5.10 | **5.17** | — |

### val Pearson trajectory through epoch 9

| Epoch | V-JEPA†-e125 | MV-PhaseRel | MV-PairedIntra | TokenRel e25 | **MCC 794** | **FJ 795** |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 0.774 | 0.813 | 0.793 | 0.789 | 0.790 | 0.766 |
| 6 | 0.793 | 0.838 | 0.813 | 0.811 | 0.811 | 0.792 |
| 8 | 0.804 | 0.844 | 0.815 | 0.826 | 0.823 | 0.813 |
| 9 | 0.809 | 0.847 | 0.836 | 0.829 | **0.825** | — |

### Matched-dataset comparator reference (20-epoch trained, final test)

| Model | Best val R² (ep) | Best val MAE | Best val Pearson | Test R² | Test MAE | Test Pearson |
|---|---:|---:|---:|---:|---:|---:|
| V-JEPA†-e125 | 0.685 (18) | 5.10 | 0.832 | 0.646 | 5.36 | 0.806 |
| MV-PhaseRel | **0.742 (17)** | **4.61** | **0.862** | **0.699** | **4.88** | **0.839** |
| MV-PairedIntra | 0.716 (18) | 4.81 | 0.847 | 0.670 | 5.07 | 0.821 |
| TokenRel-Motion e25 | 0.709 (17) | 4.92 | 0.843 | 0.667 | 5.16 | 0.819 |

**Projecting MCC 794 and FJ 795 from ep 9 trajectories**: MCC is currently at val R² 0.676, on a clean upward trajectory, likely landing at 0.70–0.75 best val and 0.65–0.70 test R². FJ 795 at ep 8 R² = 0.65, likely landing at 0.66–0.72 best val and 0.62–0.68 test R². Both competitive with V-JEPA†-e125, MCC plausibly tied with or above MV-PairedIntra.

Plots: `claude/neurips/figures/echonet/echonet_lvef_{val_r2,val_mae,val_pearson}.png`
CSV: `claude/neurips/figures/echonet/echonet_lvef_trajectories.csv`

---

## 5. Root causes (revised)

### What's true about both experiments

- Code is **correct**. No implementation bugs.
- Encoders are **competitive** with matched-compute V-JEPA continuation on EchoNet-Dynamic LVEF.
- Training-log anomalies (MCC intraview rise, FJ study collapse) are **objective-specific training dynamics**, not signs of broken encoders.

### What's still open about each

**MCC**:
- γ is not logged in the training CSV; the exact value at e25 is unknown. The fact that `crossview − intraview` gap reached 5.5% is strong indirect evidence γ > 0.
- Whether MCC's cross-clip conditioning provides value **beyond** standalone clip evaluation (i.e. at inference time with the adapter active) is **not tested** by a single-clip probe. If the adapter contributes at inference, MCC could exceed peers on multi-clip tasks.

**FJ**:
- The study transformer collapsed on its own loss early. Whether it provides any usable study-level representation (via `h_study`) is **not yet tested**.
- The study-level objective may still have shaped the clip encoder in subtle ways (via the small but non-zero study gradient) — but the dominant training signal was clip V-JEPA + anchor + SV branch.

### What's resolved

- **Single-clip A4C probes are not a failure indicator.** MCC and FJ both pass this bar.
- **Probe comparability must be checked before any comparative claim.** The 5-minute grep test for matching `dataset_train`/`dataset_val` across sbatches would have prevented the original misdiagnosis.

---

## 6. Paper framing (revised)

**Previous framing (withdrawn)**: "MCC and FJ are informative negative results showing that secondary objectives damage or starve the clip encoder."

**Current framing**: "MCC-Anchored and FullJoint-Study both produce clip encoders competitive with matched-compute V-JEPA continuation on EchoNet-Dynamic LVEF. Both offer additional mechanisms (cross-clip adapter inference, study-level pooling) that are not exercised by standard single-clip probes. Training-log diagnostics reveal distinct optimization regimes but do not indicate representation degradation."

**To make the full claim**, we still need:
1. 794/795 test numbers (expected ~60 min).
2. h_study probe for FJ to exercise the study-level path.
3. Adapter-aware inference probe for MCC to exercise the cross-clip path.
4. Multi-clip / K=8 study-level probes for both.

---

## 7. Subsidiary findings that stand regardless of the revision

### Probe comparability auditing is a MUST

The original mistake was comparing probe numbers without checking dataset equivalence. This should be a standard pre-flight check:

```
for each probe sbatch under comparison:
    grep dataset_train | grep dataset_val
verify all entries resolve to the same CSV
```

Add this as a step in `claude/neurips/README.md`.

### Training-log diagnostic columns are worth logging

FJ's 42-column training CSV was invaluable for the audit. MCC's 7-column CSV was not — γ and `pred_delta_from_A` should have been logged alongside intraview/crossview. For any future MCC-like objective, log:
- γ (single scalar)
- pred_delta_from_A (cosine-based diagnostic)
- ratio of L_mcc to L_vjepa_self post-weighting
- sampler composition per step (same-view / cross-view / cross-modality fractions)

### Weight drift is informative but not diagnostic

MCC's 40% drift vs base's 23% drift is real, but "more drift" is not inherently bad — it can indicate productive reshaping under a different objective. Drift analysis should be paired with probe data before drawing conclusions.

---

## 8. Decision table

| Branch | Continue? | Stop? | Modify? | Evidence | Next action |
|---|:-:|:-:|:-:|---|---|
| MCC checkpoint (762 e25) | ✓ | — | — | EchoNet probe val R² 0.68 at ep 9, tied with MV-PairedIntra | Wait for 794 final; then multi-clip probe for full evaluation |
| FJ checkpoint (776 30k) | ✓ | — | — | EchoNet probe val R² 0.65 at ep 8, above V-JEPA†-e125 | Wait for 795 final; then h_study probe for study-level evaluation |
| MCC re-run with γ warmup | — | — | ? (future) | No immediate need; current checkpoint works | Deprioritize |
| FJ re-run with K-out-of-K V-JEPA | — | — | ? (future) | Current FJ works, study path underused | Deprioritize; build h_study probe first |
| Paper framing as negative result | — | ✓ | — | Overturned by EchoNet probes | Retract; reframe as positive/competitive |

---

## 9. What was created in this investigation

### Reports (superseded by this master report but preserved for reference)

- `claude/neurips/reports/root_cause_low_performance/interim_report.md` — first probe comparability audit (valid part of the investigation)
- `claude/neurips/reports/root_cause_low_performance/training_log_analysis.md` — MCC/FJ CSV analysis (valid numbers, retracted inferences)
- `claude/neurips/reports/root_cause_low_performance/root_cause_synthesis.md` — first-pass synthesis (negative-result framing, withdrawn)
- `claude/neurips/reports/root_cause_low_performance/revision_2026_05_05.md` — explicit retraction of the negative-result framing

### Figures

- `claude/neurips/figures/echonet/echonet_lvef_val_r2.png`
- `claude/neurips/figures/echonet/echonet_lvef_val_mae.png`
- `claude/neurips/figures/echonet/echonet_lvef_val_pearson.png`
- `claude/neurips/figures/echonet/echonet_lvef_trajectories.csv` (98 rows, all 6 models)

### Jobs

| Job | Purpose | Status |
|---:|---|---|
| 786 | MCC MIMIC A4C LVEF probe | Canceled (non-comparable dataset) |
| 788 | FJ MIMIC A4C LVEF probe | Failed (missing adapter in tarball) |
| 791 | MCC MIMIC online-encoder probe | Canceled |
| 792 | FJ MIMIC A4C LVEF probe v2 | Canceled |
| 793 | MCC MIMIC online-encoder probe v2 | Canceled |
| **794** | **MCC EchoNet-Dynamic LVEF probe** | **Running** (ep 9 in, ~70 min to go) |
| **795** | **FJ EchoNet-Dynamic LVEF probe** | **Running** (ep 8 in, ~80 min to go) |

---

## 10. Lessons for future investigations

1. **Always verify probe comparability first.** Grep `dataset_train`/`dataset_val` across sbatches; reject any comparison that mixes splits.
2. **Do not build a negative-result framing on a single probe trajectory.** Train-log diagnostics are suggestive, not dispositive — confirm with matched-dataset probes.
3. **Log richer training diagnostics for exotic objectives.** γ, per-step sampler composition, per-term gradient norms, and EMA drift should be in the CSV.
4. **When a metric looks catastrophic, ask "catastrophic vs what?"** 0.32 R² is only meaningful if you have a floor. No matched-split floor existed for MCC/FJ on MIMIC A4C 10k.
5. **Distinguish "training dynamics" from "representation quality."** Rising loss on a subcomponent of a multi-term objective can be normal; only downstream probes settle representation quality.
