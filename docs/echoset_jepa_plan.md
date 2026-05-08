# EchoSet-JEPA — Implementation Plan (v3, implementation-ready)

Status: engineering spec; scaffolding in place; no cluster compute yet.
Supersedes `claude/neurips/echoset-jepa-plan.md` (v2). Keep that file as history.

---

## 1. Executive summary

**EchoSet-JEPA** = a **two-stage, study-level** extension of V-JEPA for echocardiography.

- **Stage 1** — existing V-JEPA clip encoder. **Frozen** in the MVP. Produces per-clip latents `c_clip ∈ R^{d_clip}` that are **cached on S3** before Stage-2 training.
- **Stage 2** — a permutation-invariant **study transformer** over a variable-sized bag of **study elements** (groups of clips sharing `(view_family, modality, phase_bucket)`). Trained with **masked study-element latent prediction** using an EMA-target objective.

**Fairness unit is `K_clips = 8`.** Every baseline, every ablation, every control consumes the *exact same K-clip manifest per study* with the *same seed*. EchoSet groups those K clips into `M_elements` post-sampling. Reports always state both `K_clips` and resulting `M_elements`.

**Seven sampler-matched baselines** (A, B1, B2, C, D, E, F, G) isolate specific failure modes before any downstream win can be claimed.

**Success requires beating Control A (prediction averaging) and B1 (supervised late-fusion); strong success requires beating B2 (capacity-matched supervised study transformer) in low-label / held-out-site / view-dropout regimes.**

Prior empirical findings shaping the plan (see `claude/neurips/controlled-objectives.md`):

| # | Finding | Design implication |
|---|---|---|
| 1 | Vanilla V-JEPA is already strong on single-clip echo | Do not expect K=1 wins; claim is at K≥4. |
| 2 | EMA/co-evolving targets are load-bearing | Stage-2 uses EMA target projector + EMA study transformer. |
| 3 | Predictor-side phase conditioning was null | Phase is a **meta token** in Stage 2, not a predictor-side conditioning input. |
| 4 | Positive-only cross-view SmoothL1 was inert | Loss is `cosine + small InfoNCE`, never SmoothL1 alone. |
| 5 | Phase-rel hard negatives help LVEF but hurt RVSP | Contrastive is over **study-element latents**, not pooled encoder latents. |
| 6 | MV2SV learned retrieval but lost downstream to paired intraview controls | Retrieval is a **diagnostic**, not a success endpoint. |
| 7 | Factorized slots did not yield reliable disentanglement | No slot factorization in Stage 2; a single `z ∈ R^{d_proj}` per element. |
| 8 | Pooled latents do not solve local-motion (TAPSE) | TAPSE is deferred to a separate token-level branch; not part of v1. |
| 9 | Sampler-matched controls are mandatory | All controls share one `study_clip_sample.parquet`. |

## 2. Scientific claim

EchoSet-JEPA learns **relational structure across views/modalities/phases within a study** that:

(a) **cannot** be recovered by averaging per-clip predictions (Control A),
(b) **cannot** be matched by a lightweight supervised late-fusion pool (B1),
(c) **cannot** be matched by a capacity-equivalent supervised study transformer trained with downstream labels only (B2),
(d) is **not** explained by metadata priors (Control D) or by exploiting marginal view/modality statistics (Control C, shuffled-study),

and that (c) and (d) translate into downstream gains under **low-label**, **held-out-site**, and **view-dropout** regimes.

Explicit non-claims: K=1 wins, TAPSE wins without a token-level branch, retrieval wins alone.

## 3. Method

### 3.1 Study elements — what we group on

**Grouping key** (revision from v2):

```
element_key = (view_family, modality, phase_bucket)
```

**`quality_bucket` is NOT a default grouping dimension** (revision from v2). Quality is used as:

- an optional **pooling weight** when aggregating clips into an element (`element_agg: mean | quality_weighted`),
- a **context metadata token** added to context element representations (may be dropped),
- a **diagnostic stratifier** for downstream analysis,

but two A4C B-mode systolic clips of different quality are **the same element**.

Rationale: including `quality_bucket` in the key over-fragments elements, forces too many single-clip elements, and risks leaking a target-side quality shortcut.

| Field | Vocabulary |
|---|---|
| `view_family` | apical, parasternal_long, parasternal_short, subcostal, suprasternal, doppler_spectral, m_mode, tdi, unknown |
| `modality` | b_mode, color_doppler, pw_doppler, cw_doppler, m_mode, tdi, contrast |
| `phase_bucket` | systolic, diastolic, full_cycle, not_applicable |

**Aggregation** for element `j` with clips `{i : key_i == j}`:

```
if element_agg == "mean":
    e_j_raw = mean( c_clip_i )
elif element_agg == "quality_weighted":
    w_i     = softmax( quality_score_i / tau_quality )    # tau_quality = 0.5
    e_j_raw = sum_i  w_i * c_clip_i
e_j = LayerNorm( e_j_raw )
```

When an element has a single clip, it passes through LayerNorm directly.

### 3.2 Stage 2 objective

Given a study as a bag of `M` elements `{e_1, ..., e_M}`, partition into context `C` and target `T` indices with a **mask strategy** (§3.5). Predict each `e_t ∈ T` from:

- context elements `{e_j : j ∈ C}` (with their meta tokens),
- target element's metadata (`view_family`, `modality`, optionally `phase_bucket`) — **NOT target quality by default**,
- a `[STUDY]` token,
- target-slot positional identity (learned `[MASK]_t`).

### 3.3 Target definitions

Two target modes; plan supports both.

**Mode A — `element_target`** (MVP default):

```
z_t = stopgrad( LN( Projector_EMA( e_t_teacher_side ) ) )
h_t =           LN( Projector_student( mask_slot_output_t ) )
```

`e_t_teacher_side` is `LayerNorm(mean({c_clip_i : key_i == t}))`, computed once per step; the projector is EMA-updated. Simple, stable, no second transformer forward pass.

**Mode B — `full_study_teacher_target`** (ablation, off by default):

A second transformer forward pass with **no masking** (teacher sees every element). For masked indices `t`, `z_t = stopgrad(LN(Projector_EMA(teacher_output_t)))`, where `teacher_output_t` is the teacher transformer's output at position `t`. Closer to I-JEPA. Costs one extra forward.

Config switch: `target_mode: element_target | full_study_teacher_target`.

### 3.4 Target metadata at the mask slot

**Required at `[MASK]_t`**:

```
+ target_view_family_emb
+ target_modality_emb
+ target_phase_bucket_emb           # config: include_target_phase (default: true)
```

**Forbidden at `[MASK]_t` by default**:

```
target_quality_emb  (gated behind target_quality_token_ablation; default: false)
```

Rationale: quality is computed on the target image itself. Including it leaks an "easy target" signal and can trivialize the masked-prediction problem. Target quality is available only as an ablation to test what fraction of the objective depends on it.

### 3.5 Mask strategies

Sampled per step from a mixture:

| Strategy | Default weight | Description |
|---|---|---|
| `random_element` | 0.50 | sample `n_target` elements uniformly |
| `whole_view_family` | 0.25 | mask every element whose `view_family == v` for one random `v` |
| `whole_modality` | 0.15 | mask every element whose `modality == m` for one random `m` |
| `apical_holdout` | 0.033 | mask all apical elements (if ≥1) |
| `doppler_holdout` | 0.033 | mask all doppler elements (if ≥1) |
| `bmode_holdout` | 0.034 | mask all B-mode elements (if ≥1) |

The last three are only drawn when the corresponding elements exist in the study. Under-represented strategies fall back to `random_element`.

Invariants:

```
min_context_elements ≥ 1
min_target_elements  ≥ 1
max_target_elements  ≤ M - 1
mask_ratio_cap       = 0.6   # clamp if a stratified strategy would exceed
```

### 3.6 K-matching and M_elements

`K_clips = 8` is the unit of cross-method fairness. The sampler pre-selects K clips per study into `study_clip_sample.parquet`; all methods read that manifest.

```
For EchoSet:
  clips_K = sample_K(study_id, K=8, seed=S)
  elements_M = group_by(clips_K, key=(view_family, modality, phase_bucket))
  M_elements = len(elements_M)        # variable, ∈ [1, 8], typically 3–6
  report: K_clips, M_elements
```

A secondary `K_elements` ablation (fixed `M` by truncating/duplicating per key) is defined but **only runs after K_clips is stable**.

### 3.7 K sweep (secondary)

```
K_clips ∈ {1, 2, 4, 8, 16}   # primary
full_N                        # appendix only
```

## 4. Architecture

### 4.1 Inputs

```
c_clip_i               ∈ R^{d_clip=1024 for ViT-L}     (cached, frozen)
e_j (raw)              = agg({c_clip_i : key_i == j})  ∈ R^{d_clip}
e_j_ln                 = LN(e_j)                        ∈ R^{d_clip}

view_emb_j             ∈ R^{d_meta=64}
modality_emb_j         ∈ R^{d_meta}
phase_emb_j            ∈ R^{d_meta}                     (or <na>)
quality_emb_j (ctx only) ∈ R^{d_meta}                   (context-side only; never at mask slot by default)

[STUDY]                ∈ R^{d_model=512}
[MASK]                 ∈ R^{d_model}                    (shared; identity via meta tokens)
```

**Fusion** (context):

```
x_j = W_in  @ e_j_ln  +  view_emb_j + modality_emb_j + phase_emb_j + quality_emb_j
```

**Fusion** (target mask slot):

```
x_t = [MASK]  +  view_emb_t + modality_emb_t + (phase_emb_t if include_target_phase else 0)
# no quality_emb_t by default
```

where `W_in : R^{d_clip} → R^{d_model}`. Meta embeddings are either additively projected from `d_meta → d_model` or (simpler) directly allocated at `d_model` and added. MVP uses `d_meta == d_model` and pure addition, which halves the code path.

### 4.2 Meta-token dropout (context side only)

```
p_dropout_view     = 0.15
p_dropout_modality = 0.10
p_dropout_phase    = 0.30
p_dropout_quality  = 0.30
```

Dropped → `<unk>` id in the corresponding lookup. Target-side meta tokens are **never dropped** (the model must know what it is predicting).

### 4.3 Study transformer (MVP — base)

```
d_model:     512
n_layers:    4      # start smaller; can bump to 6 if v1 shows signal
n_heads:     8
ffn_mult:    4
dropout:     attn 0.0, ffn 0.1
norm:        pre-LN
position:    none (permutation-invariant)
permute:     shuffle element order every training step
max_M:       64
attn:        bidirectional self-attention with padding mask
```

Input sequence:

```
[STUDY]  x_1 ... x_{|C|}  [MASK]_{t_1} ... [MASK]_{t_{|T|}}
```

`[MASK]_{t}` slots carry target meta tokens (§3.4).

**Large variant** (`n_layers: 6, d_model: 512`) is gated behind a config flag and only enabled after base shows positive signal.

### 4.4 Projectors

Two independent 2-layer MLPs, `d_model → d_hidden=1024 → d_proj=256`, GELU, LayerNorm on output.

```
Projector_student (online)
Projector_teacher (EMA copy, no grad)
```

LayerNorm is applied outside the projectors to both `h_t` and `z_t` before cosine/InfoNCE.

### 4.5 EMA

```
tau_start = 0.996
tau_end   = 0.9999
schedule  = cosine over planned steps

updated_weights = stage-2 transformer + Projector_teacher
```

V-JEPA clip encoder is **not** EMA-updated (frozen).

## 5. Losses

### 5.1 Primary — masked cosine regression

For each masked index `t`:

```
z_t = stopgrad( LN( Projector_teacher( target_latent(t) ) ) )
h_t =            LN( Projector_student( mask_slot_output_t ) )

L_regress = mean_t (1 - cosine(h_t, z_t))
```

`target_latent(t)` depends on `target_mode` (§3.3).

### 5.2 Secondary — view/modality-matched InfoNCE

Rather than batch-uniform negatives, pick negatives from the same `(view_family, modality)` bucket as the target where possible (revision from v2).

```
priority_1 = same (view_family, modality, phase_bucket)        # if |pool| >= k_min
priority_2 = same (view_family, modality)                       # fallback 1
priority_3 = same modality                                       # fallback 2
priority_4 = all-batch                                           # final fallback
exclude     = same-study off-target elements (always)
```

`k_min = 4`. Logging (per step):

```
valid_neg_count_same_view       (mean, min across the batch)
valid_neg_count_same_modality
fallback_fraction               (0..1; fraction of targets falling back past priority_2)
top1_same_view                  (retrieval top-1 using same-view pool)
pos_minus_neg_gap_by_view       (dict keyed by target view_family)
```

Loss:

```
L_nce = CE( logits = h_t @ z_pool.T / tau ,  label = diagonal positives )
tau   = 0.1
```

Same-study off-targets are masked out of the logit matrix (−inf).

### 5.3 Total

```
L_total = L_regress + lambda_nce * L_nce

lambda_nce sweep: {0.01, 0.03, 0.05}
default:          0.03
```

### 5.4 Collapse safety rail (monitoring only)

```
var_t   = std over batch of h_t
cov_off = mean abs off-diag covariance of h_t
```

Logged every step. **No automatic lambda_nce adjustment** (that was a v2 idea; overrides muddy the ablation and make runs non-reproducible). If `var_t < 0.3` for 500+ steps, the run is **halted** and a diagnostic dump is saved. Operator decides the fix.

## 6. Training schedule

```
warmup     : 2,000 steps   LR 0 → peak
main       : 50,000 steps  peak LR
cooldown   : 5,000 steps   LR → 0

optimizer  : AdamW
peak_lr    : 5e-4 (d_model=512)
weight_decay: 0.05
betas      : (0.9, 0.95)
batch      : 32 studies/GPU × 8 GPUs = 256 studies/step
```

With cached `c_clip` (no V-JEPA forwards at train time), 50k steps on 8×H100 is ~6–9 hours. Short turnaround enables many ablations.

## 7. Baselines and controls

All consume the same `study_clip_sample_K{K}_seed{S}.parquet`.

### Control A — V-JEPA + per-clip prediction averaging

For each clip in the K selected clips, run a trained single-clip attentive probe (d=1). Study prediction = arithmetic mean of clip predictions.

**Ruled out**: no cross-clip reasoning learned in EchoSet.

### Control B1 — V-JEPA + lightweight supervised late-fusion attention

2-layer attention pool, `d_model=256, n_heads=4`, ~0.3M params. Inputs: the K `c_clip` embeddings + view embeddings. Trained end-to-end with downstream labels only.

**Ruled out**: a small supervised pool captures the cross-clip signal trivially.

### Control B2 — V-JEPA + capacity-matched supervised study transformer

**Identical architecture to EchoSet-JEPA**, randomly initialized, trained end-to-end with downstream labels only. Same `d_model`, `n_layers`, `n_heads`, same element grouping, same meta tokens, same K=8.

**Ruled out**: masked-prediction pretraining adds nothing beyond capacity.

### Control C — shuffled-study pretraining

Same EchoSet-JEPA architecture and objective, but **context elements come from a different random study** (matched on M and view-family mix). Target stays in the true study.

**Ruled out**: the model exploits marginal view/modality statistics rather than true study-level structure.

### Control D — metadata-only

Transformer inputs are **only meta tokens** (`view_family`, `modality`, `phase_bucket`, `quality`); zero `c_clip` content. Downstream probe trained on `[STUDY]` token.

**Ruled out**: metadata priors explain downstream performance.

### Control E — identity / no-cross-clip encoder

Stage-2 replaced with per-element LayerNorm + mean pool over elements. No attention. Downstream probe on the pooled vector.

**Ruled out**: a study encoder adds nothing over a pooled K-clip baseline at matched probe capacity.

### Control F — target-view/modality-only baseline for masked prediction

During **pretrain evaluation only** (not downstream). Predict `z_t` from **meta tokens of target only** (no context at all). Measures how much of the masked-prediction loss is recoverable from `(view, modality, phase)` alone.

If `L_regress_F ≈ L_regress_EchoSet` → the objective is trivially solved from metadata; something is wrong.

### Control G — nearest-context-element baseline

During pretrain evaluation only. For each masked target, `ẑ_t = LN(Projector_teacher(nearest context element by c_clip cosine))`. Measures whether targets are just duplicates of something in context.

If `L_regress_G ≪ L_regress_EchoSet` → targets are too easy (near-dups not fully filtered); tighten `dedup.cosine_threshold`.

### Summary — what each control rules out

| Control | Rules out |
|---|---|
| A | no cross-clip learning |
| B1 | trivial late-fusion sufficiency |
| B2 | capacity is all you need |
| C | marginal-statistics shortcut |
| D | metadata shortcut |
| E | pooled-representation sufficiency |
| F | target-metadata trivially solves pretrain loss |
| G | targets are near-duplicates of context |

## 8. Evaluation protocol

### 8.1 Downstream — main MVP tasks

All study-level, all B-mode-input where the target is extractable from B-mode:

- **Study-level LVEF** (regression)
- **RV function** (regression) — if labels available
- **LVH / chamber size** — LVIDd, LVIDs, IVSd, LVPWd (regression, multiple tasks)
- **HCM** (classification) — if labels available
- **HF / mortality** (classification) — if labels available
- **Pediatric external generalization** — LVEF on EchoNet-Pediatric (safety check)

### 8.2 Proxy B-mode cross-modal

- **RVSP from B-mode** (no TR Doppler in inputs)
- **MR severity from B-mode**
- **AS severity from B-mode**

### 8.3 Full multimodal (Phase 6, not MVP)

Only once color/CW/TDI elements are first-class:

- RVSP with TR Doppler
- AS with CW + LVOT
- MR with color Doppler
- Diastology with mitral inflow + TDI

### 8.4 Low-label sweep

All main tasks at 1%, 5%, 10%, 25%, 100% labels. Three seeds per fraction.

### 8.5 Robustness

- **Held-out-site** (Phase 5).
- **View-dropout** — at test time, drop one view family; measure Δperf vs full-K.
- **K sweep** — K ∈ {1, 2, 4, 8, 16}.

### 8.6 Probe architecture (all methods share)

```
probe: LN → Linear(d_model → 256) → GELU → Linear(256 → n_targets)
```

Applied to:
- EchoSet-JEPA: `[STUDY]` token output.
- Controls B1/B2: their respective pooled output.
- Control D: `[STUDY]` token output.
- Control E: pooled vector.
- Controls A: per-clip scores then mean (no shared probe; already uses a d=1 probe).

## 9. Diagnostics

| Name | Purpose |
|---|---|
| target-difficulty audit (§10) | detect too-easy / too-hard targets **before** pretraining |
| Control F | target-metadata-only baseline on pretrain loss |
| Control G | nearest-context-element baseline on pretrain loss |
| context ablation | drop X% of context → L_regress curve |
| shuffled-study control | Control C gap |
| attention entropy | collapse detection |
| drop-one-view attribution | per-view importance downstream |
| performance vs K | curve per task |
| performance vs unique-view-count | controls for "many-clips ≠ many-views" |
| performance vs quality tertile | stratified downstream |
| study-size bias | per-N downstream |
| retrieval top-1 / top-5 | aux diagnostic only |
| var/cov monitoring | collapse sentinel |
| metadata shortcut (Control D gap) | confirms non-trivial gain |

## 10. Target-difficulty audit (pre-pretrain)

`experiments/echoset_jepa/target_difficulty_audit.py` runs **before** any Stage-2 training. Samples ~5,000 studies; for each valid `(context, target)` mask instance, computes:

```
cos(target_element, nearest_context_element)
cos(target_element, same-view other-study mean)
cos(target_element, metadata-only prediction)      # metadata-only regressor trained on held-out audit split
cos(target_element, same-study different-view mean)
```

Output: distributions by target `view_family`, `modality`, `phase_bucket`. Flagging rules:

- `cos(target, nearest_context) > 0.9` for >10% of cases → dedup threshold too loose; tighten.
- `cos(target, metadata_pred) > 0.8` for >20% of cases → target is trivially predictable from meta; need harder masking or remove target quality token (already default).
- Very low `cos(target, same-view other-study mean)` and high within-study cos → targets are well-posed.
- Bimodal distributions within a `(view_family, modality)` bucket → phase_bucket may be under-captured or poorly labeled.

The audit's output is the **gate to kick off P2 (pretraining)**. If the audit flags a trivial-target regime, we fix the pipeline before burning compute.

## 11. Data pipeline

### 11.1 Manifest schema (`study_clip_manifest.parquet`)

```
patient_id              : stable hashed id
study_id                : stable hashed id
clip_id                 : stable hashed id
s3_uri                  : path
dicom_series_uid        : for dedup
view_label              : str
view_conf               : float [0,1]
modality                : str
phase_label             : optional {systolic, diastolic, full_cycle, na}
phase_conf              : optional float
quality_score           : float [0,1]
quality_bucket          : {high, med, low} (per-cohort tertile; context-side only)
frame_rate_hz           : float
clip_duration_s         : float
n_frames                : int
pixel_spacing_cm_per_px : optional; gated behind 'leak_safe_calib' flag
acquisition_ts          : ISO8601 — AUDIT ONLY (never input)
site_id                 : held-out-site split key
vendor                  : diagnostic only
n_duplicates            : computed
is_duplicate_of         : optional
cached_cclip_s3         : s3://.../cclip/{study_id}/{clip_id}.npy
```

### 11.2 Splits

- **Patient-level splits** — default. No patient in both train and val/test.
- **Site-level splits** — held-out site(s) for Phase 5.
- **Study-balanced sampling** — within train, weighted sampling bias `inv_log(N_clips)`.

### 11.3 Dedup

Before element grouping:

```
clips A, B are near-dup iff
  same study_id
  AND same view_label with view_conf > 0.8
  AND same modality
  AND |n_frames_A - n_frames_B| < 3
  AND |duration_A - duration_B| < 0.2s
  AND cosine(c_clip_A, c_clip_B) > 0.98
keep: higher quality_score
```

`cosine_threshold` is tunable from audit output.

### 11.4 Variable N handling

```
N_clips == 1       → excluded from pretrain
N_clips ∈ [2,3]    → included; min_context_elements=1
N_clips ≥ 4        → primary pretrain cohort
```

### 11.5 View-label confidence

```
view_conf ≥ 0.7          → use label
view_conf ∈ [0.4, 0.7)   → 50% use, 50% <unknown>
view_conf < 0.4          → <unknown>
```

Predicted-view vs oracle-view is a Phase-5 ablation.

### 11.6 Element grouping

```
for each (view_family, modality, phase_bucket) group in study:
    if len(group) > 0:
        emit element j with
            e_j = LN( agg({c_clip_i for i in group}, element_agg) )
            view_emb, modality_emb, phase_emb
            quality_emb (context-only; = bucket of mean(quality_score_i))
```

Cap `M ≤ max_M = 64`; when exceeded, drop lowest-priority element by (quality mean × clip count).

## 12. Leakage controls

- Patient-level and site-level splits.
- Overlay / measurement text masking (applied at clip ingest before caching).
- No report-derived metadata as input.
- No measurement-derived target leakage (e.g., LVEF regression target must not be in metadata tokens).
- **No target-side quality token** by default (target_quality_token_ablation defaults to false).
- No target clip latent leakage into mask slot — target slot has meta only.
- DICOM/vendor/site metadata ablations (Phase 5).
- Predicted-view-label vs oracle-view-label ablation (Phase 5).

## 13. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Cached `c_clip` drifts from online V-JEPA | freeze V-JEPA checksum; re-cache if checksum changes |
| Metadata shortcut dominates | Control D + F; auto-fail if D ≈ EchoSet |
| Masked-prediction collapse | var/cov monitor, halt-and-dump protocol |
| Sampler-driven gains | shared seeded K-manifest across all controls |
| Targets too easy (near-dups) | target-difficulty audit, Control G |
| View-label leakage | meta dropout, `<unknown>` tokens, Control D |
| Pediatric regression | pediatric in headline; ≥ 2% regression = auto-fail |
| Site / vendor leakage | held-out-site split, vendor ablation |
| Attention entropy collapse | per-layer entropy monitor |
| Stage-2 too small → underfits | start `n_layers=4`; bump to 6 if main loss plateaus above audit-derived floor |
| Stage-2 too large → overfits | cosine + InfoNCE already regularize; dropout 0.1 |

## 14. MVP implementation checklist

PR-sized units in build order:

1. `experiments/echoset_jepa/build_manifest.py` — manifest (§11.1) from UHN + MIMIC sources.
2. `experiments/echoset_jepa/dedup.py` — near-dup clustering (§11.3).
3. `experiments/echoset_jepa/element_grouping.py` — `(view, modality, phase)` grouping; emits `study_element_manifest.parquet`.
4. `experiments/echoset_jepa/cache_cclip.py` — frozen V-JEPA forward on all clips → S3.
5. `experiments/echoset_jepa/sample_K.py` — view-stratified fixed-K sampler; emits `study_clip_sample_K{K}_seed{S}.parquet`.
6. `experiments/echoset_jepa/target_difficulty_audit.py` — §10. **Gate to P2.**
7. `src/datasets/echoset_jepa_dataset.py` — dataset yielding (context elements, target indices, target meta).
8. `src/models/meta_embeddings.py` — view/modality/phase/quality lookups + dropout policy.
9. `src/models/study_transformer.py` — §4.3 architecture.
10. `src/models/study_projectors.py` — student + EMA teacher 2-layer MLPs.
11. `app/echoset_jepa/train.py` — training loop, EMA update, losses, diagnostics, CSV schema.
12. `app/echoset_jepa/__init__.py` — `from .train import main`.
13. `tests/echoset_jepa/test_element_grouping.py`, `test_k_sampler.py`, `test_mask_strategies.py`, `test_nce_negatives.py`, `test_target_metadata_no_quality_leak.py`.
14. `configs/train/echoset_jepa/echoset_jepa_v1_K8.yaml`.
15. `configs/train/echoset_jepa/echoset_jepa_v1_K8_full_teacher_target.yaml` — Mode B ablation.
16. `configs/train/echoset_jepa/echoset_jepa_v1_K8_lambda_nce_sweep.yaml` — sweep config.
17. Controls B1, B2, C, D, E — in `evals/echoset_jepa_probe/` (post-MVP, Phase 3).

### 14.1 Minimal PR sequence

| PR | Files | Lands |
|---|---|---|
| PR-0 | `docs/echoset_jepa_plan.md`, `app/scaffold.py` one-line dispatch | This plan doc, stub dispatch |
| PR-1 | `experiments/echoset_jepa/build_manifest.py` + `dedup.py` | Manifest ready |
| PR-2 | `experiments/echoset_jepa/cache_cclip.py` | Cached features on S3 |
| PR-3 | `experiments/echoset_jepa/element_grouping.py` + `sample_K.py` | Element + K manifests |
| PR-4 | `experiments/echoset_jepa/target_difficulty_audit.py` | **P2 gate** |
| PR-5 | `src/models/meta_embeddings.py`, `study_transformer.py`, `study_projectors.py` | Stage-2 model |
| PR-6 | `src/datasets/echoset_jepa_dataset.py` | Dataloader |
| PR-7 | `app/echoset_jepa/train.py` + tests | Training loop |
| PR-8 | `configs/train/echoset_jepa/*.yaml` + sbatch | Launchable |

PRs 0–3 can overlap. PR-4 audit must run and pass **before** PR-7 burns cluster compute.

## 15. Config matrix

| Config | target_mode | lambda_nce | mask strategy mix | use | purpose |
|---|---|---|---|---|---|
| `echoset_jepa_v1_K8.yaml` | element_target | 0.03 | default mix | main MVP | primary claim |
| `echoset_jepa_v1_K8_full_teacher_target.yaml` | full_study_teacher_target | 0.03 | default mix | ablation | is Mode B worth the 2× forward? |
| `echoset_jepa_v1_K8_lambda_nce_sweep.yaml` | element_target | {0.01, 0.03, 0.05} | default mix | sweep | lambda_nce calibration |
| `echoset_jepa_v1_K8_mask_random_only.yaml` | element_target | 0.03 | random_element only | ablation | does stratified masking help? |
| `echoset_jepa_v1_K8_shuffled_study.yaml` | element_target | 0.03 | default mix + shuffle | Control C | shuffled-study pretraining |

## 16. First 72-hour experiment schedule

Assumes manifest + cached `c_clip` are already built (or in progress). All items run on a single 8×H100 node.

| Hour | Task | Output |
|---|---|---|
| 00–06 | PR-4 target-difficulty audit on 5k studies | distributions by (view, modality, phase); pass/fail gate |
| 06–08 | If gate passes → compile sbatch, stage tarball | launchable v1 |
| 08–14 | **Smoke** pretrain (5k steps, lambda_nce=0.03, element_target) | loss curves, var/cov, Control F/G logged |
| 14–16 | Smoke gate check: L_regress decreasing, var_t > 0.3, Control F gap > 20% | go / no-go |
| 16–30 | **Full** pretrain v1 (50k steps) | v1 checkpoint |
| 30–34 | Extract `[STUDY]` token on val studies; Control E pooled baseline | |
| 34–44 | Downstream LVEF probe (EchoSet + Control A + B1) on same K=8 manifest | first comparison |
| 44–54 | Downstream RVSP from B-mode + LVIDd probes | multi-task snapshot |
| 54–60 | Control D (metadata-only) pretrain — smaller, 20k steps | |
| 60–72 | Low-label sweep 10% on LVEF + LVIDd for A, B1, B2, EchoSet | early low-label signal |

This schedule deliberately does **not** include B2 full pretrain in the first 72h — B2 is an end-to-end supervised run per task, queued after EchoSet v1 shows signal at 100% labels.

## 17. Success / failure gates

### Success — must hold jointly for the primary claim

1. EchoSet-JEPA beats **Control A** by ≥ 1 absolute metric unit (or ≥ 2% relative) on ≥ 3 main study-level tasks at K=8, patient-split test.
2. EchoSet-JEPA beats **Control B1** on ≥ 2 main tasks at K=8.
3. EchoSet-JEPA beats **Control B2** on the low-label regime (1%, 5%, 10%) on ≥ 2 tasks.
4. EchoSet-JEPA is **not worse** than Control A on pediatric external generalization.
5. Control C (shuffled-study) is strictly worse than EchoSet — gap ≥ 50% of the A-vs-EchoSet gap.
6. Control D (metadata-only) is strictly worse — gap ≥ 80% of the A-vs-EchoSet gap.
7. View-dropout robustness: EchoSet's Δperf when one view family is dropped is ≤ 80% of A's Δperf on ≥ 2 tasks.

### Failure — any of these is a rewrite / abandonment trigger

- Only beats A, not B1.
- Only beats A and B1, not B2, in low-label.
- Gains explained by Control D.
- Gains vanish under patient split or site split.
- Gains vanish when K matches (same K clips at train and test).
- Only wins on LVEF, ties/loses on multi-view tasks.
- Pediatric or external-site regression ≥ 2% relative.
- Attention entropy collapses (H_attn < 0.1 × log(M)) — a disguised Control A.
- Retrieval improves but downstream does not.

## 18. Timeline

| Phase | Calendar | Deliverable |
|---|---|---|
| P0 audit | week 1 | manifest, clip counts, view/modality/phase distributions, duplicate rates |
| P1 cache | week 1–2 | V-JEPA `c_clip` cached on S3 |
| P2 target-difficulty | week 2 | audit pass; v1 config finalized |
| P3 pretrain v1 | week 2–3 | 50k-step checkpoint |
| P4 controls | week 3–4 | A, B1, B2, C, D, E downstream runs on shared K manifest |
| P5 downstream + K sweep | week 4–5 | main tasks + 1/5/10/25/100% + K∈{1,2,4,8,16} |
| P6 robustness | week 5–6 | held-out-site, view-dropout, vendor ablation, predicted-view vs oracle |
| P7 modality extension | week 6–8 *(optional)* | CW/color/TDI as first-class elements |
| P8 local-motion branch | week 7–9 *(optional)* | token-level TAPSE; not EchoSet v1 |

Gates:

- **P2 → P3**: audit must pass.
- **P3 → P5**: Stage-2 smoke must pass (L_regress decreasing, var_t stable, Control F gap > 20%).
- **P5 → P6**: Success criteria 1–3 must hold on patient split.
- **P6 → P7**: Success criteria 1–7 must hold.

---

## Appendix A — what this plan explicitly does not ship

- Any per-clip token-level objective (TAPSE branch) — deferred.
- Any paper edits / Overleaf pushes.
- Any modification to V-JEPA 2.1 / EchoJEPA pretraining code.
- A production multi-view inference runtime — this is a research pipeline.
- Doppler-native element loss — P7.
- Online joint training of V-JEPA + Stage-2 — frozen is scientifically cleaner for v1.

## Appendix B — relation to v2 plan (`claude/neurips/echoset-jepa-plan.md`)

Changes from v2:

1. Grouping key drops `quality_bucket` (§3.1).
2. Target slot no longer receives target quality by default (§3.4); gated behind `target_quality_token_ablation`.
3. `K_clips` named as the primary fairness unit; `M_elements` is derived and reported (§3.6).
4. Target-difficulty audit added as P2 gate (§10).
5. NCE negatives prioritized by `(view_family, modality, phase_bucket)` with explicit fallback ladder and logging (§5.2).
6. Mask strategies expanded to a mixture: random, whole-view, whole-modality, apical/doppler/bmode holdouts (§3.5).
7. Stage-2 starts smaller (`n_layers=4`, §4.3).
8. Two target modes `element_target | full_study_teacher_target` (§3.3).
9. Loss keeps `cosine + small NCE`; `lambda_nce ∈ {0.01, 0.03, 0.05}`, default 0.03 (§5.3).
10. Two new diagnostic controls F (target-meta-only pretrain baseline) and G (nearest-context-element pretrain baseline).
11. No automatic lambda_nce adjustment on collapse; halt-and-dump instead (§5.4).
12. 72-hour first-week schedule concretized (§16).
