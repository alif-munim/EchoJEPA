# EchoMV-JEPA Stage-1 — Root-Cause of Zero Contextualization

**Date:** 2026-05-05
**Source:** `experiments/echomv_jepa/context_sensitivity_probe.py`, smoke jobs 737 and 741.

## TL;DR

The Stage-1 teacher **can** contextualize on synthetic inputs (trained: cos 0.73) but **cannot** contextualize on real V-JEPA c_clip vectors (trained: cos 0.96). The cause is a data property, not an architecture or training bug: **within-study c_clip vectors are already nearly identical (pairwise cosine 0.87–0.91)**. When self-attention averages near-identical inputs, the result ≈ the input itself, so `forward_contextualized` ≈ `forward_isolated`.

This is not fixable by LR, seed, step count, or `λ_nce`. It requires a different input representation, a different objective, or both.

## Evidence

### 1. Within-study c_clip similarity (raw data property)

Pulled 6 c_clips from two real studies (`91265267`, `92336938`) from the S3 cache and measured pairwise cosine among the 6 clip-level V-JEPA embeddings:

| Study | Mean off-diag cosine | Min | Max |
|---|---|---|---|
| 91265267 | 0.871 | 0.804 | 0.954 |
| 92336938 | 0.908 | 0.863 | 0.975 |

Clips within a study (different views, phases, modalities of the **same patient's heart**) are **extremely similar** in V-JEPA latent space. This is a consequence of patient-level dominant features (anatomy, position, probe, scanner, body habitus) overwhelming view/modality/phase differences in the pre-trained encoder.

### 2. Init-time architecture test (synthetic random inputs)

With fresh-init weights, `d_model=512, 4L, 8H`, synthetic Gaussian `c_clip` inputs (`B=8, M=6`):

```
cos(z_full, z_iso):  mean=0.9739  min=0.9579  max=0.9860
rel_l2:              mean=0.2270  max=0.2903
per-layer ||attn_out|| / ||input_stream||:
  layer 0: 0.1442
  layer 1: 0.1528
  layer 2: 0.1425
  layer 3: 0.1299
VERDICT: BORDERLINE
```

At init, attention adds ~14% of the input stream's norm. The 97% cosine at init is partially the attention-is-small-at-init effect — standard init has attention weights N(0, 0.02²) while the residual input is a direct clip_in-projection of unit-normal inputs plus meta embeddings. This borderline is **expected at init**; what matters is whether training fixes it.

### 3. Trained teacher test (job 741's final.pt, synthetic random inputs)

After 5 000 steps of training, on synthetic Gaussian inputs:

```
cos(z_full, z_iso):  mean=0.7330  min=0.3114  max=0.9910
per-layer ||attn_out|| / ||input_stream||:
  layer 0: 0.9167     ← attention grew 6.4× at layer 0
  layer 1: 0.1864
  layer 2: 0.1028
  layer 3: 0.1005
VERDICT: HEALTHY (significant contextualization)
```

Training **did** amplify attention at layer 0 (0.14 → 0.92). On synthetic inputs with independent Gaussian elements, the trained teacher genuinely contextualizes.

### 4. Trained teacher test (job 741's final.pt, REAL c_clip inputs)

```
INIT on REAL c_clips:     cos(z_full, z_iso)  mean=0.9839
TRAINED on REAL c_clips:  cos(z_full, z_iso)  mean=0.9607
```

Training barely moved the needle on real data: 0.984 → 0.961. Because real within-study c_clips have cos 0.87–0.91 to begin with, attention's output (which is essentially a learned weighted combination of already-similar inputs) differs from identity by a tiny fraction. After the projector MLP + LayerNorm, this gets compressed further, producing the `z_cosine_vs_isolated = 0.998` observed in training.

### 5. Why training still reduces `z_cosine_vs_v1` to 0.42

During training, `z_cosine_vs_v1` drops from 1.0 → 0.42 because `z_v1 = proj_teacher(st.clip_in(tgt_elements))` and `z_echomv = proj_teacher(F_bar_psi(full_study)[target_idx])`. These projectors share weights (EchoMV and the v1 probe use the *same* teacher projector MLP, just applied to different inputs). `st.clip_in` is a pure linear projection (d_clip=1024 → d_model=512), while `F_bar_psi` applies 4 transformer blocks + LayerNorm. Even without useful cross-element contextualization, the transformer stack shifts the distribution enough that the post-projector cosine to the v1 path drops — but this "difference from v1" is not the same as "informative cross-element context."

The diagnostics distinguish these two cases correctly:
- `z_cosine_vs_v1 → 0.42` (PASS): the teacher path does something different from pure linear.
- `z_cosine_vs_isolated → 0.998` (FAIL): but that "something different" is still per-element, not contextualized.
- `z_cosine_vs_peer_drop → 0.9999` (FAIL): dropping a peer has zero effect on target output.

## Why the §20.2.a gates failed

The pre-downstream gates require **both** `z_cosine_vs_v1 ≤ 0.95` AND `z_cosine_vs_isolated ≤ 0.90`. Stage-1 passed the first (it's not collapsed to v1), failed the second (it's collapsed to isolated-per-element). The architectural claim "contextualize after full-study encoding" is vacuously true but useless: the contextualization happens, it's just redundant because there is no meaningful context.

## What won't fix it

- **Longer training.** Already tested ~5k steps; trajectory is plateaued. 27 k would take `z_iso` from 0.998 to maybe 0.99 at best — still way above 0.90.
- **More LR or aggressive schedulers.** The teacher is fine; it produces useful outputs on heterogeneous inputs. The problem is the inputs are homogeneous.
- **Tiny NCE (Stage-1b).** NCE would pressure different *studies* to differ, which they already do (different studies have different c_clip directions); it would not force cross-element context within a study to help.
- **Per-modality projector (Stage-1m).** Only splits the projection space by modality id; does not change within-study c_clip similarity.
- **No-EMA ablation.** Teacher is not the issue; real c_clip homogeneity is.
- **Architectural tweaks like attention dropout / layer scale init.** These would affect init but not steady-state; the trained attention already took over at layer 0.

## What could fix it (decreasing order of feasibility)

### Option A — Use per-clip tokens, not per-clip pooled c_clip

Currently each element is a **single** 1024-d vector (mean-pooled across clips in that element, or single clip). The study has 6–8 elements, each collapsed to 1 vector. Self-attention over 6 near-identical vectors is nearly a no-op.

Fix: keep `(T, 1024)` token-level V-JEPA outputs per clip, flatten across clips, and attend over the full token set (`K × T ≈ 8 × 256 = 2048` tokens per study). Token-level echo representations have structure that gets averaged out in mean-pooling. This is essentially what `clip_encoder.keep_tokens: true` is a config hook for in the architecture plan — but it was reserved for Stage-3/4, not tested at Stage-1.

Cost: ~50× more compute per step for the teacher, and `experiments/echomv_jepa/cache_tokens.py` does not exist yet (per `docs/echomv_jepa_architecture_plan.md` §12.3 it's a "deferred follow-up"). This is a real pipeline build.

### Option B — Change the target so the student cannot collapse to per-element

Force the student to predict the *study-level* output `h_study` (the `[STUDY]` token readout) instead of per-element target outputs. Then isolated-per-element targets are meaningless by construction.

Cost: low. Loss change: predict `h_study_student` against `h_study_teacher` via cosine regression, target-slot mask loss drops out, NCE becomes study-level. But this breaks the held-out-element masked prediction story the design doc is built around — need to re-examine whether study-level prediction is the right scientific claim or whether we need per-element prediction to validate multi-view reasoning.

### Option C — Use a stronger clip-level augmentation and accept near-identity as the background

Treat `z_cosine_vs_isolated = 0.99` as the baseline and measure *deviations* instead of absolute values. The 0.2% of signal in `1 - cos ≈ 0.002` could still carry useful information for downstream tasks. Empirical test: train with this as-is, probe downstream at step 27 000, compare LVEF R² against EchoSet v1 control. If +0.015 R² shows up, the contextualization diagnostic is overly pessimistic for Stage-1's scientific claim.

Cost: low, but goes against the spec's pre-downstream halt rule. The spec says not to waste compute on downstream probes when pre-downstream gates fail; Option C says the gates were miscalibrated and we should run the downstream probe anyway. This requires agreement from the user on the scientific framing.

### Option D — Introduce synthetic within-study diversity via element masking/dropout

Randomly drop 25% of elements' `clip_in + meta_add` contribution at the student and teacher input. Forces the model to reconstruct missing elements using peers, and the teacher cannot be isolated-per-element because every element sees dropout noise. This is like a dropout variant of BERT-style masked modeling on the element stream.

Cost: small. A ~20-line change in `training_step_echomv`. Worth probing in a new Stage-1d config before committing to pipeline changes (Option A).

### Recommendation ranked

1. **Diagnostic-only next step (no compute):** ask the user whether Option A (token-level cache rebuild) or Option D (element dropout, cheap) is in scope before the NeurIPS deadline. Option D is a ~1-day experiment; Option A is a ~1-week pipeline build.
2. **If user wants progress today:** implement Option D, re-run the smoke. 30 min of code + one 5-min smoke.
3. **Reserve Option C as a "skip the gate and report downstream numbers" fallback** only if the user explicitly wants to know whether the tiny contextualization delta carries downstream signal despite the cosine-near-1 diagnostic.

## What this means for the NeurIPS plan

The Stage-1 claim as written — "full-study EMA target encoder with target selected after encoding is the JEPA-faithful minimum addition over EchoSet v1" — is architecturally true but **empirically null on MIMIC K=8 cached c_clips.** Adding a transformer with self-attention over 6 near-identical vectors does not produce meaningfully different targets than running each through the same transformer alone. This is an honest negative result for the Stage-1 MVP as specified.

The paper needs either:
- A demonstration that Stage-1 produces better downstream results **despite** `z_cosine_vs_isolated ≈ 1.0` (which would undermine the §15.1a probe as a valid gate — would require discussion); or
- A different input representation (Option A) that gives attention something to do; or
- A different objective (Option B) that doesn't collapse to per-element predictions.

## Holds

Same as the smoke-results report — no Stage-1b / 1m / ablation / breadth / downstream launches.

## Files added this session

- `experiments/echomv_jepa/context_sensitivity_probe.py` — standalone probe, CLI + callable.
- `experiments/echomv_jepa/__init__.py`.
- `reports/echomv_jepa/stage1_context_debug.md` (this file).

## What is NOT broken (from the probes)

- Training loop, DDP, EMA updates, sbatch pipeline — all healthy.
- `StudyTransformerEMA.forward_contextualized` is mathematically correct (layer-ratios show attention doing work; trained weights contextualize synthetic inputs).
- `ModalityProjectorPair`, losses, manifest schema, cache — all validated.

The single broken piece is: per-study cached V-JEPA c_clips are too similar for cross-element attention to carry signal.
