# Phase-Relational Pretraining: Design Reference

Detailed reference for the phase-relational JEPA objective
(EchoJEPA-Rel) used in the current NeurIPS experiment set. This doc
explains the design as implemented on disk, contrasts it with vanilla
V-JEPA-2 pretraining, and traces why three earlier phase-aware
variants came up null.

Companion to:
- `phase-relational-hardneg.md` — running-record of 593/608/613
  pretrains and probes
- `experiments/phase-jepa.md` — design notes for the earlier
  Predictor-φ / Mask-φ variants (superseded)
- `experiments/finalbudget-phase-probes.md` — design notes for the
  earlier positive-only cross-view regression variant (superseded)

Source files this doc pins against:
- `app/vjepa_multiview/train.py:555` — `_relational_infonce_with_hard_neg`
- `app/vjepa_multiview/train.py:650` — `forward_phase_relational`
- `app/vjepa_multiview/phase_relational_head.py` — head module
- `configs/train/vitl16/pretrain-multiview-phase-relational-hardneg-25of100-paper.yaml`

## 1. The method in one picture

Per training step, the sampler produces three time-synchronised clips
from the same MIMIC study:

```
  clip_a           clip_b_pos            clip_b_neg
  (anchor)         (positive target)     (hard negative target)

  study S          study S               study S
  view X           view Y (≠X often)     view X or family(X)
  phase φ_a        phase φ_b ≈ φ_a + Δφ  phase φ_b + δ,  |δ| ≥ 0.25
```

Three forwards then occur:

```
  STUDENT(clip_a + mask) ── predictor ─► ẑ        (intra JEPA signal)
  TEACHER(clip_a)  ────────────────────► h_a
  TEACHER(clip_b_pos) ─────── pool ────► y_pos    (detached)
  TEACHER(clip_b_neg) ─────── pool ────► y_neg    (detached)
```

The **relational head** maps the anchor's pooled student context onto
a query vector in a 256-dim InfoNCE space, conditioned on `(view_a,
view_b_pos, Δφ_b)`:

```
  q = RelHead.query(pool(z_ctx), view_a, view_b_pos, Δφ_b)
  y_pos_p = RelHead.target(y_pos.detach())
  y_hard_p = RelHead.target(y_neg.detach())
```

The **loss** has two additive terms with a warmup-scaled coefficient:

```
  L_total = L_intra(ẑ, h_a)
          + λ_rel(t) · L_rel(q, y_pos_p, y_hard_p, batch_negs)

  λ_rel(t) = 0.05 · min(1, epoch_progress / 5)
```

`L_intra` is the standard V-JEPA-2 SmoothL1 latent-prediction loss
between the predictor's output at masked positions and the teacher's
corresponding features on the same clip. `L_rel` is a candidate-set
InfoNCE over `[positive, hard_neg, batch_negs]`.

## 2. Differences from vanilla V-JEPA-2 pretraining

Side-by-side of every component that differs. Everything not listed
here (encoder architecture, optimizer, LR schedule, masking, dtype,
EMA) is identical.

| Axis | Vanilla V-JEPA-2 | Phase-Relational (this method) |
|---|---|---|
| **Dataset** | `VideoDataset`, 1-clip-per-row CSV | `VideoGroupDataset` over `phase_annotations.parquet` (~500K clips with per-frame cardiac-cycle phase, HR, quality tier, view label, RR consistency) |
| **Sampler** | Uniform random window over any MIMIC clip | `PhaseMatchedStudySampler` emitting 3-clip triples (anchor + positive + hard neg) with strict filters |
| **Clips loaded / step** | 1 | 3 |
| **Quality filter** | None (any clip accepted) | `quality_tiers: [high, medium]`, `rr_filter_mode: strict`, `require_rr_consistent: true` |
| **View-pair policy** | N/A | 35% same-view / 45% same-family / 20% cross-family for the (anchor, positive) pair |
| **Δφ conditioning** | N/A | Controlled buckets `[0, 0.125, 0.25, 0.5]` with probs `[0.40, 0.30, 0.20, 0.10]`; phase tolerance 0.15 |
| **Hard-negative eligibility** | N/A | Mandatory: same-study, same-view (or same-family if unavailable), phase offset ≥ 0.25 cycles from the positive. `rel_allow_missing_hard_negative: false`; anchors without a valid hard neg are resampled (≤ 16 attempts). |
| **Teacher forward** | 1 clip under `torch.no_grad` | 3 clips concatenated into a single `no_grad` call on clip_a + clip_b_pos + clip_b_neg, LayerNorm'd, split. Saves kernel launches and keeps all teacher activations under one no-grad context. |
| **Student forward** | 1 masked context clip | 1 masked context clip (clip_a only — b_pos, b_neg never enter the student) |
| **Predictor phase conditioning** | Disabled | Disabled in this variant (`predictor_phase_token.enabled: false`); Δφ conditioning lives only in the relational head, not in the base predictor |
| **Relational head** | Absent | 4-layer fuse: source MLP + view-id embeddings (separate for anchor/positive, 14 canonical views + UNKNOWN) + Fourier-encoded Δφ (4 frequencies) → relation MLP → 256-dim query. Target projector: shared MLP applied to detached teacher-pooled latents. |
| **Loss components** | L_intra only | L_intra + λ_rel(t) · L_rel |
| **L_rel structure** | N/A | Candidate-set InfoNCE over [positive (col 0), hard neg (col 1), batch negs (cols 2..B+1)]. Labels always 0. Self-diagonal in the batch block masked to −∞; same-study off-diagonals also masked when `rel_mask_same_study_batch_negatives: true`. τ = 0.10. |
| **λ_rel schedule** | N/A | Linear warmup from 0 → 0.05 over 5 epochs, then held at 0.05 (cap 1/20 relative to L_intra) |
| **Metadata-shortcut barrier** | N/A | Head signature accepts only `(c_a_pool, view_a_ids, view_b_pos_ids, delta_phase_pos)` — **no** HR, absolute phase, quality tier, RR status, phase error, view confidence, view_b_neg_id, Δφ_neg, or any study/patient/DICOM ID. Enforced by a smoke test on `_build_predictor_inputs` arity (4 tensors, hard-capped). |
| **Teacher / target stop-gradient contract** | Teacher EMA + no-grad + no params in optimizer | Same, **plus** `.detach()` on y_pos and y_neg before they enter the target projector. Head's target projector itself is trainable; teacher parameters are not. |
| **Per-step diagnostics logged** | `loss, intraview, crossview` (7 CSV cols) | 24 CSV cols including `rel_loss, rel_top1_with_hard, rel_pos_sim_mean, rel_hard_neg_sim_mean, rel_batch_neg_sim_mean, rel_pos_minus_hard_gap, rel_pos_minus_batch_gap, effective_lambda_rel, q_var, y_var, q_prenorm_mean, y_prenorm_mean, logits_std, target_enc_grad_l2, target_proj_grad_l2, rel_target_projector_trainable, same_study_masked_count` |
| **Checkpoint metadata** | Standard (encoder, predictor, target_encoder, opt, scaler, epoch) | Standard + `relational_head.state_dict()` + `rel_config: {lambda_rel, rel_warmup_epochs, rel_temperature, target_projector_trainable, rel_mask_same_study_batch_negatives, rel_negative_mode}` + top-level `multiview_objective: "phase_relational"` |

### Why the design choices above

**Concat teacher forward + no-grad.** Three separate teacher forwards
would multiply kernel-launch overhead; concatenating them keeps one
no_grad scope and halves teacher latency relative to the naive
implementation. LayerNorm is applied after the concat, so all three
targets share the same normalization statistics.

**Relational head is pooled, not token-level.** Token-level contrastive
loss would require spatial correspondence between views — which
doesn't exist for different echocardiographic acoustic windows.
Mean-pooling across token positions aggregates clip-level information
that is comparable across views.

**View embeddings are separate for anchor and positive.** Not shared,
because the role of "I am the anchor view" differs from "I am the
requested target view." A successful query has to condition on both
independently; sharing the embedding would introduce a symmetric
inductive bias we don't want.

**Target projector is trainable.** Alternatives considered:
(a) frozen random projection, (b) MLP shared with source_proj,
(c) trainable (chosen). Option (c) gives the model flexibility to
reshape the target space during training. Guardrail: the projector
sees **only detached** teacher features, so no gradient flows back
into the encoder through this path.

**Self-diagonal and same-study masking on batch negatives.** Without
masking, the same positive would appear twice in the candidate set
(once at column 0, once at a same-row batch position), pulling the
loss toward 0 trivially. Same-study masking prevents near-duplicates
(different clips from the same heart) from being counted as negatives.

**λ_rel cap at 0.05.** The primary objective is still L_intra — the
base JEPA signal is what the encoder was already succeeding on at
e100. L_rel nudges the representation along the phase-relational axis
without overriding the intraview representation quality. Empirically
at λ_rel > 0.1, q_var grows too fast and intra loss drifts up.

**Warmup over 5 epochs.** The relational head is randomly initialized;
its query doesn't produce a useful InfoNCE signal at step 0. The
warmup lets the head calibrate before its gradient dominates — the
first 5 epochs are predominantly L_intra with L_rel smoothly
increasing.

### What is the same as vanilla

- Encoder / predictor architecture: ViT-L/16, tubelet 2, patch 16,
  predictor with 12 blocks, 384-dim embedding, 12 heads, RoPE,
  mask tokens, SDPA, activation checkpointing.
- Optimizer / LR / EMA: AdamW, weight decay 0.04, cosine LR over the
  100-epoch scheduler horizon with start LR 3.33e-5, peak LR 1.75e-4,
  5-epoch warmup, bfloat16. EMA `[0.99925, 0.99925]`.
- Masking: same `num_blocks=[8, 2]` dual-mask setup. Student sees
  masked clip_a, predictor fills masked-position features.
- Init: both start from the MIMIC standard V-JEPA e100 checkpoint
  (hash-verified via `verify_init_checkpoint.py --strict`).

So the only meaningful differences from vanilla are: the dataset
wrapper, the 3-clip sampler and its filters, the addition of
`L_rel` to the loss, and the machinery to support those two changes.

## 3. How it differs from earlier phase objectives

Four phase-aware variants have been tried across three experiment sets
leading up to this method. The first three were nulls on converged
LVEF; the mechanism framing in §8 of the paper explains why.

### 3.1 Predictor-φ (superseded)

**Variant:** Within-clip. Condition the base V-JEPA predictor on the
context-to-target phase displacement Δφ by concatenating a Fourier
embedding of Δφ onto the mask tokens. Base JEPA masking and loss
unchanged.

**Config:** `configs/train/vitl16/pretrain-mimic-224px-16f-in21k-full-phase-jepa.yaml`,
`phase_conditioned: true`.

**What it learned:** Early-training gain on LVEF (val R² 0.443 →
0.510 at e25), but converged to parity with vanilla JEPA by e100.

**Why it failed.** Predictor-φ provides the predictor with Δφ as
**conditioning information** — it tells the predictor *when in the
cycle* to predict. But the encoder itself is never given any signal
that its frame representations should *differ* by phase. If the
encoder produces phase-invariant features and the predictor somehow
inverts them using Δφ, the loss still decreases. At convergence, the
predictor's Δφ channel is effectively ignored because the intraview
SmoothL1 doesn't reward using it.

**Mechanism gap:** *conditioning, not discrimination*. The encoder is
not pressured to produce phase-distinct representations.

### 3.2 Mask-φ (superseded)

**Variant:** Within-clip. Sample the JEPA target block at specific
cardiac-cycle phases (e.g. always force the target to cover a
late-systolic window) while holding the context mask coverage fixed.

**Config:** `configs/train/vitl16/pretrain-mimic-224px-16f-in21k-phase-mask.yaml`.

**What it learned:** Statistically indistinguishable from a
localised-block control (uniformly placed short targets at the same
scale) at both e25 and e50. Both short-block variants were
significantly worse than full-duration JEPA-IN21K masking with
non-overlapping 95% CIs.

**Why it failed.** Same mechanism gap as Predictor-φ plus a new one:
constraining target blocks to short-window phase buckets **reduced the
effective target coverage**, which hurt the base JEPA signal. The
phase-aware placement didn't add anything, and the short-window
constraint cost ~0.12 val R² at e25 regardless of whether placement
was phase-aware or uniform.

**Mechanism gap:** the *target placement* encodes phase, but the
encoder is still asked to predict that placement's content from a
standard context — there is still no discrimination signal.

### 3.3 Positive-only cross-view regression (job 542, superseded)

**Variant:** Cross-clip, same study. Draw phase-matched clip pairs
from within a study and add a **SmoothL1** crossview loss: predictor's
output on clip_a should match teacher's encoding of clip_b, weighted
at 0.25 · L_crossview on top of the intraview loss.

**Config:** `configs/train/vitl16/pretrain-multiview-phase-matched-25of100.yaml`
(renamed and archived; descendants are in `finalbudget-phase-probes.md`).

**What it learned:** Null on downstream LVEF probe (val MAE 5.013
phase-matched vs 5.097 plain JEPA continuation — ~0.08 within
HP-seed noise). Null on explicit phase-decodability probe (Δ ≈
1.2° vs matched single-view arm, also within HP noise).

**Why it failed.** This was actually the first across-clip variant,
and it does shape the encoder to produce features comparable across
views. But under tight phase+view matching, `teacher(clip_b)` is
**near-redundant with** `teacher(clip_a)` — the same physiological
state viewed from a different but related window. The crossview
SmoothL1 collapses toward the intraview loss because the two targets
are too similar.

Diagnostic observation that made this clear: 542's total loss stays
≈ 0.67 across all 25 epochs because the crossview term
`0.25 · L_crossview ≈ 0.17` sits on top of the descending intraview
term without itself descending. The sampler is doing what it claims
(we verified view-pair mixture and phase-bin coverage), but the
crossview objective *under these pair conditions* collapses toward
being a noisier intraview.

**Mechanism gap:** *positive-only supervision*. There is no force
pulling different-phase representations apart — the only training
signal comes from pairs that should be *similar*, and the teacher at
matched phase produces near-identical targets.

### 3.4 Phase-relational InfoNCE with mandatory hard negative (593, current method)

**What's new vs variant 3.3.** The positive-only regression is
replaced with a **discriminative** contrastive objective:

1. The positive is still drawn to be similar to the anchor
   (cross-view, matched phase).
2. An additional **mandatory hard negative** is drawn from the
   same study and same view/family but at a wrong phase
   (|Δφ_neg − Δφ_pos| ≥ 0.25 cycles).
3. The candidate set `[positive, hard_neg, batch_negs]` enters
   cross-entropy with labels fixed at column 0.

Because the positive and the hard negative are **matched on study and
view** but **differ by phase**, the only axis of variation that
distinguishes them is cardiac-cycle phase. The encoder cannot win the
InfoNCE by learning view- or study-invariant features — it has to
produce phase-discriminative representations to separate column 0
from column 1.

The view-pair policy (35% same-view / 45% same-family / 20%
cross-family) keeps the positive from being a trivial same-view
same-phase copy; a successful query has to carry cross-view phase
correspondence. The hard negative restricts the axis of variation so
that discrimination cannot be solved by anything other than phase.

**Mechanism payoff:** the first phase-aware variant with a
*discrimination signal* instead of conditioning or regression-to-
similar-target. Empirically on LVEF: val R² 0.477 → 0.742 over 17
probe epochs (595); test R² 0.6986 vs matched-compute SV e125 test R²
0.645 — Δ = +0.054.

### 3.5 Compact comparison table

| # | Variant | Scope | Training signal | Shared failure mode |
|---|---|---|---|---|
| 1 | Predictor-φ | within-clip | Δφ conditioning on predictor | Encoder never penalised for phase-indistinct representations |
| 2 | Mask-φ | within-clip | Phase-bucketed target placement | Same as (1); additionally reduces effective target coverage |
| 3 | Positive-only cross-view | across-clip, same study | SmoothL1 of predictor(clip_a) against teacher(clip_b) at matched phase | At tight phase+view matching, teacher(b) ≈ teacher(a); objective collapses to noisier intraview |
| **4** | **Phase-relational InfoNCE (current)** | **across-clip, same study** | **Candidate-set InfoNCE with mandatory same-study wrong-phase hard negative** | **No longer shares the failure mode — discrimination is forced on an axis where pos/neg differ only by phase** |

## 4. Pre-registered controls for the current method

The method alone doesn't prove anything — the downstream gain on LVEF
could come from any of the 9 things that differ from vanilla
(sampler, eligibility filter, quality/RR filter, view-pair policy,
Δφ buckets, 3-clip forward, head architecture, λ_rel, the InfoNCE
loss itself). The experiment set has two controls to isolate which:

### 4.1 Paired intraview-only control (job 608)

Same YAML as method **except** `multiview_objective: intraview_only`
and `use_crossview_loss: false`. All 9 data-path axes identical
to method — same 3-clip sampler, same eligibility, same quality
filter, same view-pair policy, same Δφ buckets. Only the loss
differs: control computes L_intra on clip_a alone; b_pos and b_neg
clips are loaded but discarded at the loss.

This isolates **whether the InfoNCE loss itself is load-bearing**,
holding the data path fixed. `Δ = method − control` is the
paper's pre-registered comparison.

### 4.2 No-hardneg ablation (job 613)

Same YAML as method **except** `rel_negative_mode: no_hardneg`,
which sets column 1 of the InfoNCE logits to −∞ before softmax.
Everything else is byte-identical: same sampler, same teacher
forward on all 3 clips, same relational head forward, same
`q·y_hard` cosine diagnostic.

This isolates **whether the mandatory hard negative is load-bearing**
(vs. "the InfoNCE term with only batch negatives would have
sufficed"). `Δ = method − ablation` is the secondary question.

### 4.3 What each control rules out

| Outcome | Mechanism reading |
|---|---|
| Method > control > ablation > vanilla | Clean story: every component is load-bearing in the expected direction |
| Method > control, method ≈ ablation | The hard negative is redundant; batch negatives alone suffice under the 3-clip sampler's eligibility filter |
| Method ≈ control, method > ablation | The InfoNCE term adds nothing beyond the sampler/eligibility changes (would invalidate the paper's method claim) |
| Method ≈ control ≈ ablation | The downstream gain is sampler-driven; method claim fails |

## 5. What is NOT claimed by the method

The method's training objective has two components that shape
single-view features:

1. **Phase awareness (within-clip):** from the same-view wrong-phase
   hard negative. The encoder produces frame-/clip-level features
   that differ by cardiac-cycle phase.
2. **Cross-view alignment (across-clip):** from the same-study
   different-view positive. The encoder produces features aligned
   across acoustic windows at matched phase.

A **single-view downstream probe** (one clip at inference) can read
out (1) directly — it receives phase-localised features and can
decode phase-dependent targets (LVEF, TAPSE). It **cannot** directly
exercise (2): there is no second view at inference to integrate with.
What single-view features inherit from (2) is a latent
trace — the encoder was pushed toward view-invariant physiology —
but a regression probe can exploit that trace only when its target
correlates with the view-invariant axis.

So the empirically supported claim from the current single-view
evaluations (LVEF, TAPSE) is:

> Phase-relational pretraining improves phase-sensitive
> dynamic-function representations.

The claim NOT supported by these evaluations is:

> Phase-relational pretraining improves multi-view integration.

The latter claim would require multi-view downstream probes (2+
clips at inference). See `phase-relational-hardneg.md` §8.6 for
detailed treatment and `experiments/phase-relational-hardneg.md`
changelog for the framing revision dated 2026-05-02.

## 6. Quick file / artifact pointers

| Artifact | Path / location |
|---|---|
| Method YAML | `configs/train/vitl16/pretrain-multiview-phase-relational-hardneg-25of100-paper.yaml` |
| Control YAML | `configs/train/vitl16/pretrain-multiview-intraview-only-25of100-paper.yaml` |
| Ablation YAML | `configs/train/vitl16/pretrain-multiview-phase-relational-no-hardneg-25of100-paper.yaml` |
| Training entrypoint | `app/vjepa_multiview/train.py` |
| Head | `app/vjepa_multiview/phase_relational_head.py` |
| InfoNCE function | `_relational_infonce_with_hard_neg` at `train.py:555` |
| Unit tests | `tests/phase/test_relational_infonce.py` (8 tests, all passing) |
| Launch debug log | `claude/neurips/phase-relational-launch-debug.md` |
| Running experiment record | `claude/neurips/experiments/phase-relational-hardneg.md` |
| Prior variants (superseded) | `experiments/phase-jepa.md` (Predictor-φ, Mask-φ); `experiments/finalbudget-phase-probes.md` (positive-only cross-view) |
| Paper §8 | `user-default-efs/echojepa-neurips/sections/08_phase.tex` |
