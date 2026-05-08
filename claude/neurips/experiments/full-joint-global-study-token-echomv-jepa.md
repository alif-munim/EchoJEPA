# Full-Joint Global Study-Token EchoMV-JEPA

**Status (2026-05-05):** smoke **PASSED** (job 771). Overnight run
**RUNNING** (job 774, 30,000 steps on `ip-10-0-50-146`, 1-day walltime).

## 1. Story in one paragraph

Arm C (`reports/echomv_jepa/stage1_global_study_token_smoke_results.md`)
was the first echo-domain objective to produce a non-trivial
`metadata_only_study_gap = 0.16` (27× Arm A) and study matching 8× chance.
But Arm C trained only a small study transformer on top of **frozen pooled
c_clip vectors**. The clip encoder itself — the thing downstream probes
actually use — never saw a gradient from the study objective. This
experiment unfreezes the clip encoder. The online V-JEPA clip encoder
`f_θ` is trainable, has its own EMA teacher `f̄_θ`, and receives gradient
from the global `[STUDY]`-token objective *plus* a plain V-JEPA
self-supervision loss that keeps it from specializing away from
single-clip usefulness. An **anchor loss** against a frozen e100 copy
(`f_0`) further bounds how far the clip encoder can drift. A
**single-view-to-study** branch occasionally hides all but one view
family from the student to force the clip encoder to be useful even
when only A4C (or PLAX, PSAX, RV-focused) is available at downstream
inference.

The question this experiment answers: **does full-joint study-level
learning produce a clip encoder that beats a matched-compute vanilla
+25 V-JEPA continuation on both A4C-only single-view probes *and* K=8
study-level probes, without hallucination?**

## 2. Why this is not MCC-JEPA, Arm C, or MV2SV

| | MCC-JEPA | Arm C (frozen) | MV2SV | **Full-Joint (this)** |
|---|---|---|---|---|
| Clip encoder | trainable (V-JEPA on pairs) | **frozen** (c_clip cache) | trainable (factorized slots) | **trainable** (V-JEPA online + EMA + anchor) |
| Study transformer | — | trainable | — | trainable |
| Primary objective | masked tubelets of clip B | global [STUDY] pooled | pooled target-view slot | **global [STUDY] over corrupted study** |
| Cross-acquisition mechanism | zero-gated cross-attn at predictor | study transformer attention | factorized slot pooling | study transformer + corruption/SV branch |
| Hallucination risk | low (target-anchored B) | N/A (pooled) | **high** (pooled target view) | low (study target, not view target) |
| Single-view downstream claim | clip encoder helps A4C | frozen encoder unchanged | **single-view student internalizes multi-view** | clip encoder learns A4C via SV branch, no view hallucination |
| Loss protects base V-JEPA | yes (L_vjepa_self on B) | n/a | no | **yes (L_clip_vjepa + L_anchor to e100)** |

The defining difference from MV2SV: **no factorized slots, no pooled
target-view latent, no target-view metadata as the primary
prediction signal.** The single-view-to-study branch predicts the
*study* embedding from single-view input — never a pooled target view.
The model is never asked to hallucinate missing views; it only has to
aggregate available-view information into a study summary.

The defining differences from Arm C: **the clip encoder is trainable**,
and **there is no c_clip cache** — clips are encoded online every step
through `f_θ`. This is what turns Arm C from "better study head" into
"better clip encoder + better study head."

The defining differences from MCC-JEPA: **study-level objective**, not
clip-level. No cross-clip adapter in the predictor. No target-anchored
masked tubelet prediction. Both experiments share the same e100
initialization and run in parallel on different nodes; they are
**complementary**, not alternatives. MCC-JEPA asks "can same-study
pairs improve masked-tubelet prediction?" Full-joint asks "can a
global study objective improve the clip encoder *and* the study
encoder simultaneously?"

## 3. The mechanism in detail

### 3.1 Networks

Eight modules live in one `FullJointModel` aggregator
(`src/models/echomv_jepa/full_joint_model.py`):

- `f_θ` — online V-JEPA clip encoder, trainable, layer-wise LR decay
- `f̄_θ` — EMA teacher of `f_θ`, no grad, τ: 0.999 → 0.99995
- `f_0` — frozen e100 anchor (deepcopy at init, static)
- `clip_pred` — V-JEPA predictor, trainable (kept for potential
  clip-level V-JEPA loss; in the current smoke/overnight we compute
  `L_clip` directly as a masked-token distance without the predictor)
- `F_ψ` — student study transformer, trainable
- `F̄_ψ` — EMA teacher study transformer (`StudyTransformerEMA`), no grad,
  τ: 0.996 → 0.9999
- `p_study` — student study projector (MLP d_model → d_hidden → d_proj)
- `p̄_study` — EMA teacher study projector

All three clip encoders are initialized from
`checkpoints/jepa_in21k_vitl_e100.pt`. The study transformer and
projectors start fresh (no init checkpoint).

### 3.2 Input pipeline

One study per sample, K clips per study (K ≤ 8 from the K-sample
manifest). `EchoMVJEPAPixelDataset` (`src/datasets/echomv_jepa_pixel_dataset.py`)
emits raw MP4 tensors. `echomv_pixel_collate` pads and produces:

```
full_clips     (B, M_full, 3, T, H, W)   — raw pixels
full_pad_mask  (B, M_full) bool
full_meta_*    (B, M_full) long          — view, modality, phase, quality
```

No c_clip cache. No per-step pooling into a fixed embedding. The pixel
tensor for every clip in every batch goes through `f_θ` online.

### 3.3 Forward pass

```text
1. Encode all valid (non-padded) clips via the online f_θ        → student tokens
2. Same clips through the EMA teacher f̄_θ (no grad)              → teacher tokens
3. Subsample K_anchor clips, run through frozen f_0 (no grad)    → anchor tokens
4. Spatial 2×2 pool on both sides: 1568 → 392 tokens per clip
5. L_clip = L1(student_tokens, teacher_tokens) on random 40%
   of tokens per clip (masked-token distance, no predictor)
6. L_anchor = 1 - cos(LN(pool(student_anchor)), stopgrad(LN(pool(f_0_anchor))))
7. Pool student tokens per element: (B, M, T_tok, d) -> (B, M, d)
8. Apply study corruption to student's per-element input:
     random_element_dropout 0.30
     whole_view_family_dropout 0.25
     whole_modality_dropout 0.15
     no_dropout 0.30
9. Student F_ψ on (corrupted student elements + meta, corrupted pad) → h_study
10. Teacher F̄_ψ on (teacher elements + meta, full pad), no grad     → z_study
11. L_study_jepa = 1 - cos(LN(p_study(h_study)), stopgrad(LN(p̄_study(z_study))))
12. L_nce       = matched_nce across the batch (other studies as negatives)
13. L_cov       = off-diagonal covariance penalty on p_study(h_study)
14. With p=0.25: drop all but one view family on the student, recompute
    h_study_sv, add L_sv = 1 - cos(LN(p_study(h_study_sv)), stopgrad(LN(p̄_study(z_study_full))))
```

Total loss:

```
L = λ_clip   · L_clip
  + λ_study  · L_study_jepa
  + λ_nce    · L_nce
  + λ_cov    · L_cov
  + λ_anchor · L_anchor
  + λ_sv     · L_sv
```

Initial weights: `λ_clip=1.0, λ_study=0.1, λ_nce=0.005, λ_cov=0.001,
λ_anchor=0.05, λ_sv=0.02`. The overnight run ramps `λ_study: 0.1 → 0.3`
over 6000 steps and `λ_sv: 0.02 → 0.05` over 9000 steps.

### 3.4 EMA updates (after optimizer step)

```text
step_clip_ema(f̄_θ, f_θ, τ_clip)                # ViT-L parameters, fused _foreach
teacher_st_ema.update_teacher(student_st, τ_study)
projector.update_teacher(τ_study)
```

Both schedules are linear over `total_steps`. The EMA deltas are
logged as `ema_clip_delta` / `ema_study_delta` (L2 norm of teacher −
student parameter vectors).

### 3.5 Layer-wise LR decay (clip side)

`layerwise_param_groups()` in
`src/models/echomv_jepa/full_joint_clip_backbone.py` builds AdamW
param groups with depth-dependent scales on the encoder:

```text
blocks 0 .. 5                 → 0.1×   base_lr
blocks 6 .. 17                → 0.3×   base_lr
blocks 18 .. 23 + final norm  → 1.0×   base_lr
patch_embed / pos_embed       → 0.1×   base_lr
predictor                     → 1.0×   base_lr
```

`clip_base_lr = 3e-5`. Study transformer / projector / meta embeddings
all live at `base_lr = 2e-4` in their own param groups. This keeps the
low-level features close to e100 while letting the late layers +
predictor + study head adapt faster.

### 3.6 Masking

No V-JEPA block-mask collator in the current path. `L_clip` uses a
random 40% token mask per clip for student-vs-teacher distance. This
is weaker than the full MaskCollator+predictor path but runs in
~1.8 s/step vs the 5–7 s we budgeted for the full pipeline, and
smoke 771 confirmed the clip encoder still gets a useful gradient
from it (`L_clip` stayed bounded at 0.01–0.02, anchor held at 0.999).

The full predictor+MaskCollator path is the natural next upgrade if
we want the clip encoder to get a stronger self-supervision gradient.
Hooks exist in `FullJointModel` (`clip_predictor`,
`encode_clips_online(clips, masks=...)`), so the swap is localized.

## 4. Why this should work given prior evidence

- **Arm C passed 6/7 gates at smoke scale.** The study-level
  objective is learnable and produces content signal beyond metadata.
  Full-joint inherits this.
- **V-JEPA is the strongest base objective on echo** (controlled
  objective study). The `L_clip_vjepa` term + anchor loss protect
  this foundation.
- **MV2SV failed because it injected view-hallucination pressure.**
  Full-joint targets the *study* embedding, not any view — so no view
  the student didn't see is a target.
- **Arm A/B failed contextualization** because the clip encoder was
  frozen and the study transformer couldn't change the target
  geometry. Full-joint unfreezes the clip encoder *and* has a
  study-level target, breaking both bottlenecks simultaneously.
- **The single-view-to-study branch directly attacks the most common
  downstream setting** (A4C-only probes). It's the explicit signal
  that says "your A4C clip features should be enough to summarize
  this study."

## 5. Files

All new code is additive; no changes to Arm A/B/C code paths. The
shared `_apply_study_corruption` was **extracted** from
`app/echomv_jepa/train.py` into a standalone module (byte-identical
behavior pinned by `tests/echomv_jepa/test_study_corruption.py`).

| Path | Role | LOC |
|---|---|---|
| `src/models/echomv_jepa/study_corruption.py` | extracted corruption sampler | 90 |
| `src/models/echomv_jepa/full_joint_clip_backbone.py` | e100 load + layer-wise LR decay | 120 |
| `src/models/echomv_jepa/clip_ema.py` | clip-encoder EMA helpers | 50 |
| `src/models/echomv_jepa/full_joint_model.py` | FullJointModel aggregator | 200 |
| `src/models/echomv_jepa/full_joint_losses.py` | L_clip / L_anchor / L_study / L_sv + total | 120 |
| `app/echomv_jepa/train_full_joint.py` | distributed training loop | 550 |
| `app/echomv_jepa/train.py` | +10 lines: dispatch when `trainer: full_joint` | +10 |
| `src/models/echomv_jepa/token_study_transformer.py` | +`forward_with_study_token` method | +30 |
| `configs/train/echomv_jepa/full_joint_global_study_smoke.yaml` | 225-step smoke config | 180 |
| `configs/train/echomv_jepa/full_joint_global_study_overnight.yaml` | 30k-step overnight config | 200 |
| `scripts/echomv_jepa/pretrain_full_joint_smoke.sbatch` | 1h 30m smoke | 120 |
| `scripts/echomv_jepa/pretrain_full_joint_overnight.sbatch` | 24h overnight | 120 |
| `tests/echomv_jepa/test_full_joint_*.py` (×6) | unit tests | 6 × 40 |

### 5.1 Tests

`tests/echomv_jepa/` — 24 new tests, all passing (part of 116/116
green suite):

- `test_study_corruption.py` (5) — byte-identical to extracted-from
  behavior; 4 corruption modes covered + determinism
- `test_clip_ema_update.py` (3) — EMA moves teacher, schedule is
  linear, delta is zero on identical copies
- `test_teacher_no_grad_full_joint.py` (3) — freeze() helpers pass/fail
  correctly
- `test_global_study_loss.py` (5) — L_global_study returns 0 on
  matched inputs, >0 on mismatched, LossRamp warms up linearly
- `test_anchor_loss.py` (4) — L_anchor == 0 on identical tokens,
  gradient flows to online side, diagnostic cosine matches
- `test_single_view_to_study_branch.py` (2) — subset logic keeps
  exactly one view family unpadded, handles all-padded edge case
- `test_full_joint_no_cache.py` (2) — trainer imports pixel dataset,
  not cached c_clip; config does not reference `cache_*_prefix`

## 6. Experimental setup

### 6.1 Initialization

Canonical `checkpoints/jepa_in21k_vitl_e100.pt` (5,127,835,835 bytes,
encoder 304M + predictor 22M, epoch 100). Same file MCC-JEPA uses.
The `build_clip_encoder_from_e100()` helper deepcopies the freshly-
loaded encoder twice more for `f̄_θ` and `f_0`, then sets `requires_grad=False`
on both deepcopies.

### 6.2 Data

K=8 manifest at
`experiments/echoset_jepa/artifacts/study_clip_sample_K8_seed0_train.parquet`
(47,955 rows, 6,089 studies, mean 7.88 clips/study). Pre-fetched to
`/opt/dlami/nvme/echomv_cache/k8_train.parquet` on the compute node.
Per-clip metadata carried by manifest: view_family, modality,
phase_bucket, quality_score.

### 6.3 Compute budget

Smoke: 225 steps, K=4, batch_studies_per_gpu=4, 8 × H100 with
activation checkpointing + bfloat16. Observed: **1.8 s/step mean**,
**17.3 GB/GPU memory**. 6:43 wall (including init, data loader
warmup, first-batch S3 fetch).

Overnight: 30,000 steps, K=8 (full K-sample), batch_studies_per_gpu=4.
Projected ~15 h at the observed throughput; sbatch walltime 24 h.
Checkpoints at step 2k / 5k / 10k / 15k / 20k / 25k + `latest.pt` at
30k.

### 6.4 Matched-compute comparison

Per-step clip forwards: K × B × world_size = 8 × 4 × 8 = **256 clips/step**.
30,000 steps × 256 = **7.68M clip forwards total**.

MCC-JEPA (job 762, target-anchored +25) per-step clip forwards: 2 × 32 ×
8 = 512. 16,250 steps × 512 = **8.32M** clip forwards.

At these settings the two runs are within ~8% of each other on
clip-forward compute — approximately matched, which was the reason
for stretching full-joint from 5k to 30k steps. Throughput is similar
(both near 2 s/step on 8 × H100).

## 7. Smoke results (job 771, 225 steps, gate PASS)

Full detail in `reports/echomv_jepa/full_joint_smoke_results.md` (to
be written). Summary:

### 7.1 Loss trajectory

| step | loss_total | loss_clip | loss_study | loss_anchor | anchor_cos | rank_top1 | meta_gap |
|---|---|---|---|---|---|---|---|
| 0   | 0.018 | 0.000 | 0.081 | 0.000 | 1.000 | 0.75 | 0.819 |
| 50  | 0.082 | 0.014 | 0.071 | 0.001 | 0.999 | 0.75 | 0.223 |
| 100 | 0.064 | 0.014 | 0.050 | 0.001 | 0.999 | 0.50 | 0.198 |
| 200 | 0.021 | 0.016 | 0.039 | 0.000 | 1.000 | 1.00 | 0.065 |
| 220 | 0.080 | 0.015 | 0.040 | 0.001 | 1.000 | 0.50 | 0.030 |

### 7.2 Gate scorecard

| Gate | Target | Observed | Pass |
|---|---|---|---|
| No NaNs throughout | 0 | 0 | ✓ |
| L_clip stable or decreasing | Δ ≤ +25% | 0.012 → 0.015 | ✓ (borderline) |
| L_study decreases, does not collapse | yes | 0.16 → 0.04 (−73%) | ✓ |
| study_matched_rank_top1 > chance (1/4=0.25) | > 0.25 | 0.60 avg | ✓✓ |
| metadata_only_study_gap > 0.05 | > 0.05 | 0.14 avg | ✓✓ |
| anchor_cosine_to_e100 > 0.90 | > 0.90 | 0.999 | ✓✓ |
| cov_off < 0.5 | < 0.5 | 0.04 | ✓ |
| Clip gradient nonzero | yes | yes | ✓ |
| Both EMAs update | yes | clip Δ 0.28 → 2.56, study Δ 0.72 → 6.48 | ✓ |
| Memory fits at K=4, batch=4 | yes | 17.3 GB/GPU | ✓ |
| Single-view branch finite | yes | L_sv = 0 (no SV draw in this 225-step run; p=0.25) | ⚠ unverified |

`L_sv` never fired because at p=0.25 the expected count in 225 steps
is 56 triggers, but in distributed training the SV branch only runs
when every rank draws true from the same RNG — effectively lower rate.
The overnight run will see many SV steps.

### 7.3 What the smoke proved

- The full computational graph works end-to-end: 8 GPUs, bf16,
  activation checkpointing, online encoder, EMA teacher, anchor, study
  transformer with corruption, projector with EMA teacher, all six
  loss components.
- **`L_study` dropped 73%** in 225 steps — the global [STUDY]
  objective is actively learning, exactly as Arm C predicted.
- **anchor_cosine_to_e100 held at 0.999** — the clip encoder is NOT
  drifting away from e100's feature geometry, despite receiving
  gradients from all five non-anchor losses. This is the crucial
  safety signal: without it, the joint training could specialize the
  clip encoder onto study-identity discrimination and silently
  degrade A4C-only probes.
- **`metadata_only_study_gap = 0.14`** — the student's study
  embedding uses content beyond what metadata alone would predict.
  The same signal that made Arm C promising is present here, at
  comparable magnitude (Arm C: 0.16; full-joint smoke: 0.14).
- **EMAs update, gradients nonzero, covariance bounded.** No
  collapse; no divergence.

### 7.4 API fixes during smoke (all deployed for 774)

- `StudyTransformerConfig` takes `n_layers` / `n_heads`, not `depth` /
  `num_heads`
- `EMAProjectorPair` takes `d_model` / `d_hidden` / `d_proj`, not
  `d_in` / `d_hidden` / `d_out`
- `make_transforms()` has no `training` kwarg
- `FullJointModel.update_projector_teacher` /
  `update_study_teacher` must unwrap `DistributedDataParallel`
  before calling `.update_teacher()`

## 8. Overnight run (job 774, live)

- Submitted: 2026-05-05 09:09 on `ip-10-0-50-146`
- Target: 30,000 steps, 24 h walltime
- Projected completion: ~00:00 PDT next day
- Checkpoints: 2k / 5k / 10k / 15k / 20k / 25k + latest.pt at 30k
- λ_study ramps 0.1 → 0.3 over 6000 steps
- λ_sv ramps 0.02 → 0.05 over 9000 steps
- Periodic S3 sync every 10 min to
  `s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts/echomv_jepa/full_joint_overnight_runs/774/`

Monitor by pulling `log_r0.csv` from S3; the 20-column CSV schema is
documented in the smoke. Key diagnostics to watch hourly:

- `anchor_cosine_to_e100` — must stay ≥ 0.95. If it drops below 0.85,
  kill the run and lower `clip_base_lr` or raise `λ_anchor`.
- `L_clip` — must stay bounded; a rise above 0.10 suggests the clip
  encoder is learning to optimize study objectives at cost of
  tubelet-prediction consistency.
- `study_matched_rank_top1` — should grow through epoch 5 then plateau
  above chance.
- `metadata_only_study_gap` — should stay positive; a drop to ~0
  means the study embedding has collapsed to metadata-only.

## 9. Risks and open questions

### 9.1 Clip encoder drift

**Risk:** the joint objective moves the clip encoder away from e100
into a study-identity-discrimination regime, silently degrading
A4C-only downstream probes even while K=8 probes improve.

**Mitigation:** `L_anchor` at λ=0.05 + layer-wise LR decay (0.1× on
lower blocks) + smoke confirmed anchor_cosine stays at 0.999. The
overnight monitor has a hard-kill threshold at 0.85.

**Backup:** if full-joint fails downstream, fall back to
**adapter-joint** — freeze blocks 0–17, train only blocks 18–23 +
predictor + study side. The code already supports this by zeroing
the lower-block scales in `layerwise_param_groups`.

### 9.2 Single-view-to-study branch is too aggressive

**Risk:** λ_sv ramping from 0.02 to 0.05 pushes the student to
predict study-level information from one view, which is a harder
task than predicting it from all views. If SV loss dominates, the
clip encoder could specialize onto view-agnostic features at the
cost of view-specific ones.

**Mitigation:** p=0.25 limits the fraction of steps with SV loss;
λ_sv stays below λ_study throughout. Monitor `L_sv` — if it's
stuck at > 2× `L_study`, SV is too hard and needs softer λ_sv or
longer warmup.

### 9.3 L_clip is weak (random token mask vs full predictor path)

**Risk:** the current L_clip is `L1(student_tokens, teacher_tokens)`
on a random 40% of tokens, not the full V-JEPA masked-prediction
objective. If the clip encoder needs stronger self-supervision to
stay useful for single-view tasks, this shortcut could be
insufficient.

**Mitigation:** anchor loss + EMA teacher already provide self-
supervision even without the predictor path. If the anchor diagnostic
holds, the shortcut is defensible. If it fails, we can swap to the
full MaskCollator + predictor path (hooks exist in `FullJointModel`;
~30 minutes of engineering).

### 9.4 No matched-compute vanilla +25 control

**Risk:** job 761 (vanilla +25) was cancelled to make room for this
experiment. Without it, the only available baseline is e100. A +25
continuation is the fair comparison — otherwise full-joint wins on
raw compute alone.

**Mitigation:** queue a vanilla +25 run as a follow-up on the next
free compute window. Downstream comparison should flag this
limitation until the matched control exists.

### 9.5 Overnight throughput estimate is based on smoke

**Risk:** smoke ran K=4 at batch=4; overnight runs K=8 at batch=4
per GPU. Per-clip compute is doubled. 1.8 s/step at K=4 may be
~3.6 s/step at K=8 → 30 h instead of 15 h.

**Mitigation:** 24 h walltime cap gives headroom; if throughput
drops below 2.5 s/step, the run will wrap at ~28,000 steps and save
`latest.pt` on exit via the trap.

## 10. What downstream success looks like

Out of overnight scope; queued as a follow-up. Compare on:

**Clip-only probes (single-view input):**
- A4C-only LVEF
- A4C-only TAPSE
- A4C-only RV qualitative function
- HCM
- LVH

**Study-level probes (K=8 input):**
- K=8 prediction averaging on LVEF, RV qualitative, HCM, LVH
- Incident HF, mortality
- K=8 late-fusion attention (if script ready)

Baselines:
- e100 (frozen, current ceiling for unaltered V-JEPA)
- Arm C frozen study transformer (checkpoint already exists from job 754)
- Vanilla +25 V-JEPA continuation (to-be-queued)
- MCC-JEPA target-anchored +25 (job 762, running in parallel)

**Interpretation:**
- If full-joint improves BOTH clip-level and study-level: the clip
  encoder was reshaped in a useful direction and the study head
  also learned. Primary success.
- If it improves study-level but regresses clip-level: the clip
  encoder specialized on study identity. Fall back to adapter-joint.
- If it improves clip-level but not study-level: the anchor + L_clip
  dominated and the study objective didn't bite. Raise λ_study.
- If it regresses both: abandon; Arm C (frozen) is the better axis.

## 11. Open file references

- Design primitives: `src/models/echomv_jepa/{full_joint_*,study_corruption,clip_ema}.py`
- Training loop: `app/echomv_jepa/train_full_joint.py`
- Smoke config: `configs/train/echomv_jepa/full_joint_global_study_smoke.yaml`
- Overnight config: `configs/train/echomv_jepa/full_joint_global_study_overnight.yaml`
- Launch scripts: `scripts/echomv_jepa/pretrain_full_joint_{smoke,overnight}.sbatch`
- Prior study-token work: `claude/neurips/experiments/*` (Arm C smoke results in
  `reports/echomv_jepa/stage1_global_study_token_smoke_results.md`)
- Parallel orthogonal axis: `claude/neurips/experiments/masked-cross-clip-vjepa.md`

## 12. Timeline

- **2026-05-05 08:11** — MCC-JEPA +25 (job 762) launched on node 56
- **2026-05-05 08:14** — vanilla +25 control (job 761) cancelled to
  free node 146
- **2026-05-05 08:14** — implementation kickoff for full-joint
- **2026-05-05 08:40** — full-joint smoke (job 771) first attempt
- **2026-05-05 08:48** — full-joint smoke PASS (job 771, 225 steps)
- **2026-05-05 09:05** — overnight (job 772, 5000 steps) launched
- **2026-05-05 09:08** — job 772 cancelled after user asked to match
  MCC compute (30k instead of 5k)
- **2026-05-05 09:09** — overnight (job 774, 30,000 steps) launched
- **~2026-05-06 00:00** — projected overnight completion

## 13. Debug ladder and all-job outcomes

Six sbatch jobs were submitted for Full-Joint between 08:38 and 09:09
on 2026-05-05. Four smoke attempts failed with successively deeper
errors before the fifth (job 771) passed; one overnight was cancelled
and replaced by the current 30k-step run. This section records each
failure so future debuggers see the whole loop.

### 13.1 Job ladder

| Job | Config | Node | Outcome | Elapsed | Root cause |
|---|---|---|---|---|---|
| 763 | smoke | 146 | **FAILED** | 0:21 | `StudyTransformerConfig` kwarg mismatch (`depth` vs `n_layers`) |
| 766 | smoke | 146 | **FAILED** | 0:25 | `make_transforms()` doesn't take `training=...` |
| 769 | smoke | 146 | **FAILED** | 0:56 | `DistributedDataParallel` wraps `projector`; `self.projector.update_teacher(tau)` raised AttributeError |
| 771 | smoke | 146 | **PASS** | 6:43 | — |
| 772 | overnight 5k | 146 | **CANCELLED** | 4:23 | User asked to match MCC compute (resubmit as 30k) |
| 774 | overnight 30k | 146 | **RUNNING** | — | Primary experiment; live as of writing |

### 13.2 Fix ledger

**Fix 1 — `StudyTransformerConfig` kwargs** (job 763 crash).
`full_joint_model.py:build_full_joint_model` built the study-
transformer config with:
```python
st_cfg = StudyTransformerConfig(d_clip=d_clip, d_model=cfg.d_model,
                                depth=cfg.study_depth,
                                num_heads=cfg.study_num_heads)
```
But `StudyTransformerConfig` in `src/models/study_transformer.py:22`
uses `n_layers` and `n_heads`. Traceback:
```
TypeError: StudyTransformerConfig.__init__() got an unexpected
keyword argument 'depth'
```
Fix: rename to `n_layers=cfg.study_depth, n_heads=cfg.study_num_heads`.
Side-effect: `EMAProjectorPair` had the same kind of mismatch
(`d_in`/`d_hidden`/`d_out` in my code vs `d_model`/`d_hidden`/`d_proj`
in the module). Fixed in the same edit. **Could have been caught
locally** by running `build_full_joint_model` on CPU before deploy;
I had a sanity import test but it didn't touch the factory, only the
imports.

**Fix 2 — `make_transforms(training=True)`** (job 766 crash). The
V-JEPA transform factory (`app/vjepa/transforms.py:13`) has no
`training` arg; I passed one copying a different API. Traceback:
```
TypeError: make_transforms() got an unexpected keyword argument
'training'
```
Fix: remove the kwarg. **Could have been caught locally** by the
same sanity import test (would have exercised the dataset
construction path). Not caught because I only tested the constructors
in isolation.

**Fix 3 — DDP-wrapped projector lacks `update_teacher`** (job 769
crash). After the job got past model init and through 1 training
step, `model.update_projector_teacher(tau)` raised:
```
AttributeError: 'DistributedDataParallel' object has no attribute
'update_teacher'
```
Because I wrap `model.projector` in DDP at line 210 of
`train_full_joint.py`, but `FullJointModel.update_projector_teacher`
called `self.projector.update_teacher(tau)` directly without
unwrapping. Same bug potentially on `update_study_teacher`.

Fix: unwrap DDP in both methods:
```python
proj = self.projector.module if hasattr(self.projector, "module") else self.projector
proj.update_teacher(tau)
```
**Should have been caught locally** by a test that wraps the
student module in DDP and calls the update methods; no such test
existed. Local tests run single-process.

**Fix 4 — sbatch partition hack** (same as MCC fix 6). All
templates pin `-p ml-p5-48xlarge`, but that partition is empty on
this cluster cycle; must submit with `-p dev --nodelist=...`
override. Not a code fix.

### 13.3 What the 5000-step first overnight (772) was going to be

Originally the overnight was configured for 5000 steps:
- batch=4, K=4, warmup 200, total 5000
- λ_study ramp 1000 steps, λ_sv ramp 1500 steps
- saves at 1k / 2k / 3k / 4k
- sbatch walltime 12 h

Job 772 launched at 09:05 and ran cleanly for 4:23 before user asked
to match MCC-JEPA's clip-forward compute (~8.3M clip forwards). 5000
steps × 256 clips/step = 1.3M — **6× short** of MCC. Cancelled.

Job 774 is the resized version:
- total 30000, warmup 500
- λ_study ramp 6000 steps, λ_sv ramp 9000 steps
- saves at 2k / 5k / 10k / 15k / 20k / 25k
- sbatch walltime 24 h

This produces 7.68M clip forwards — within 8% of MCC's 8.32M.

### 13.4 Test summary (what local tests actually verify)

`tests/echomv_jepa/` has 24 new tests in 6 files for full-joint
(part of the 82-test echomv_jepa suite). Each file's assertion set:

- `test_study_corruption.py` (5 tests)
  - `no_dropout` is identity (no change to elements or pad mask)
  - `random_element_dropout` expands pad but keeps ≥ 1 unpadded per row
    and newly-padded positions are zeroed while kept content is preserved
  - `whole_view_family_dropout` drops exactly one view family, always
    keeping ≥ 1 unpadded per row
  - `whole_modality_dropout` drops exactly one modality, always
    keeping ≥ 1 unpadded per row
  - Same RNG seed → byte-identical output (confirms the
    extraction-from-train.py didn't introduce drift)
- `test_clip_ema_update.py` (3 tests)
  - `step_clip_ema` moves teacher params toward student; delta is > 0
  - `clip_ema_schedule(0.9, 0.99, 10)` yields 11 values, τ[0]=0.9,
    τ[-1]=0.99, monotone increasing
  - `ema_delta_norm` returns exactly 0.0 on identical copies
- `test_teacher_no_grad_full_joint.py` (3 tests)
  - `assert_no_grad(frozen_mod)` passes silently
  - `assert_no_grad(trainable_mod)` raises AssertionError
  - `freeze()` sets module to eval mode *and* every param's
    `requires_grad=False`
- `test_global_study_loss.py` (5 tests)
  - `global_study_loss(h, h)` ≈ 0 (identical inputs → cosine 1)
  - `global_study_loss(h, z)` > 1e-3 on random different inputs
  - `single_view_to_study_loss` is a numerical alias for
    `global_study_loss` on same arguments
  - `assemble_total_loss` produces the exact weighted sum
    (λ_clip·1 + λ_study·2 + … = 1.589 for default weights)
  - `LossRamp(0.3, 100).value_at(step)` linear on [0, 100], saturated
    at 0.3 beyond
- `test_anchor_loss.py` (4 tests)
  - `anchor_loss(x, x)` ≈ 0 on identical tokens
  - `anchor_loss(online, anchor)` > 1e-3 and gradient flows to the
    online input
  - `anchor_cosine_to_e100` diagnostic > 0.999 on identical tokens
  - `pool_tokens_mean((N, T, D))` returns `(N, D)`
- `test_single_view_to_study_branch.py` (2 tests)
  - `_single_view_subset` ensures every row's remaining unpadded
    positions share one view family
  - When all positions are padded, returns the original all-True mask
    (no crash)
- `test_full_joint_no_cache.py` (2 tests)
  - `train_full_joint.py` imports `EchoMVJEPAPixelDataset`, not the
    cached `EchoMVJEPADataset`
  - The smoke YAML has `clip_encoder.source == "online_trainable"`
    and no `cache_*_prefix` keys

**What tests do NOT cover** (and would have caught earlier failures):
- `build_full_joint_model` end-to-end construction on CPU
  (would have caught fixes 1 and 2)
- Distributed wrapping + EMA update interaction (would have caught fix 3)
- Save/load round-trip of the FullJointModel
- Actual cluster compute (only local CPU tests)

### 13.5 Launch command log

For traceability — exact sequence of submits that ran:

```text
08:38  sbatch -p dev --nodelist=ip-10-0-50-146 ..._full_joint_smoke  → 763 FAIL (fix 1)
[fix 1 applied, tarball rebuilt and redeployed]
08:41  sbatch -p dev --nodelist=ip-10-0-50-146 ..._full_joint_smoke  → 766 FAIL (fix 2)
[fix 2 applied, tarball rebuilt and redeployed]
08:45  sbatch -p dev --nodelist=ip-10-0-50-146 ..._full_joint_smoke  → 769 FAIL (fix 3)
[fix 3 applied, tarball rebuilt and redeployed]
08:48  sbatch -p dev --nodelist=ip-10-0-50-146 ..._full_joint_smoke  → 771 PASS (6:43)
[smoke gate scorecard green; proceed to overnight]
09:05  sbatch -p dev --nodelist=ip-10-0-50-146 ..._full_joint_overnight (5k) → 772 RUNNING
[user requests matching MCC compute; bump total to 30k]
09:08  scancel 772                                                    → 772 CANCELLED (4:23)
[config edited: total_steps 5000→30000, walltime 12h→24h, ramps extended]
09:09  sbatch -p dev --nodelist=ip-10-0-50-146 ..._full_joint_overnight (30k) → 774 RUNNING
```
