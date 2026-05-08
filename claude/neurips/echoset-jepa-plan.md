# EchoSet-JEPA — Implementation Plan (v2, reviewer-proof)

Status: engineering spec, pre-implementation.
Applies after Fix-1 sampler / MV2SV v5 smoke work; independent code path (`app/echoset_jepa/`).

---

## 1. Executive summary

EchoSet-JEPA is a **two-stage, study-level** extension of V-JEPA for echocardiography.

- **Stage 1**: the existing V-JEPA clip encoder. Frozen in the MVP. Produces per-clip latents `c_clip`.
- **Stage 2**: a permutation-invariant **study transformer** over a variable-sized bag of **study elements** (grouped clips). Trained with **masked study-element latent prediction**.

The method is evaluated against **sampler-matched baselines** (Controls A, B1, B2, C, D, E) at fixed **K=8** clips per study. It only wins if it beats both prediction-averaging (A) and a **capacity-matched supervised late-fusion transformer** (B2).

Prior empirical findings that shape this plan:

1. Vanilla V-JEPA is already strong on single-clip echo — do not expect large single-clip wins.
2. EMA teachers are load-bearing — use EMA for the stage-2 target encoder.
3. Predictor-side phase conditioning has been null — treat phase as a **token** in stage 2, not as a head conditioning input.
4. Positive-only cross-view regression is inert — couple regression with a small contrastive term.
5. Pooled phase-rel InfoNCE helps LVEF but hurts RVSP and external generalization — do not lift that loss wholesale; EchoSet-JEPA's contrastive is over **study-element latents**, not pooled encoder latents.
6. Factorized-slot objectives improved retrieval but failed downstream — **retrieval is not a success endpoint**.
7. TAPSE is local-amplitude; pooled latents cannot solve it — deferred to Phase 7 (token-level branch), not EchoSet-JEPA v1.
8. **Sampler effects dominate apparent gains** — every baseline gets the exact same K clips with the same seed.

## 2. Scientific claim

EchoSet-JEPA learns **relational structure across views/modalities/phases within a study** that (a) cannot be recovered by averaging per-clip predictions, (b) cannot be matched by a capacity-equivalent late-fusion attention pool trained only with downstream labels, and (c) translates into downstream gains under **low-label**, **held-out-site**, and **view-dropout** regimes.

The claim is explicitly NOT:

- that EchoSet-JEPA beats single-clip V-JEPA at K=1,
- that it improves local-amplitude tasks (e.g. TAPSE) without a token-level branch,
- that retrieval improvements alone validate the method.

## 3. Method

### 3.1 Study elements (masking unit)

Masking at **individual-clip** granularity is rejected: many studies have 3+ redundant A4C clips, so masking one teaches the model to denoise duplicate clips rather than reason across views/modalities.

Define a **study element** as a group key:

```
element_key = (view_family, modality, phase_bucket, quality_bucket)
```

| Field | Vocabulary |
|---|---|
| `view_family` | apical, parasternal_long, parasternal_short, subcostal, suprasternal, doppler_spectral, m_mode, tdi, unknown |
| `modality` | b_mode, color_doppler, pw_doppler, cw_doppler, m_mode, tdi, contrast |
| `phase_bucket` | systolic, diastolic, full_cycle, not_applicable |
| `quality_bucket` | high, medium, low (tertiles of classifier quality score) |

Examples: `A4C.b_mode.systolic.high`, `PLAX.b_mode.full_cycle.high`, `A5C.color_doppler.na.med`, `aortic.cw_doppler.na.high`, `tr.cw_doppler.na.high`, `mitral.m_mode.na.high`, `septal.tdi.na.high`.

**Aggregation of clips into an element** (for `N_clips_in_element ∈ {1..m}`):

```
e_j = LayerNorm( mean_pool( c_clip_i  for  i ∈ element j ) )
```

A weighted mean by quality score is a config knob (`element_agg: mean | quality_weighted`). Quality-weighted is default only when `element_agg_quality=true` and quality labels are trusted; otherwise plain mean.

When fewer than `min_clips_per_element` valid clips are present, the element is still emitted (single clip passes through LayerNorm) but its quality token reflects the underrepresentation.

### 3.2 Stage 2 objective

Given a study as a bag of `M` elements `{e_1,...,e_M}`, mask a subset `T` (indices). Predict each `e_t ∈ T` from:

- context elements `{e_j : j ∉ T}`,
- target element's **metadata only** (view, modality, phase, quality tokens),
- a `[STUDY]` token,
- mask slot positional identity (learned, not acquisition-order).

Target is **EMA teacher-side**:

```
z_t = stopgrad( LayerNorm( Projector_teacher( e_t_teacher ) ) )
h_t =           LayerNorm( Projector_student( predicted_target_t ) )
```

where `e_t_teacher` is computed with an EMA copy of (a) the projector and (b) the stage-2 encoder. The V-JEPA clip encoder is shared (frozen in MVP).

### 3.3 Training regime

MVP: stage-2 only. V-JEPA clip encoder is **frozen** and its outputs are **cached** on S3.

### 3.4 Inference and K-matching

Fixed-K view-stratified sampling:

```
K = 8    # primary
K_sweep ∈ {1, 2, 4, 8, 16}
full_N   → appendix only
```

All baselines receive the **exact same K clips** per study, with the **same random seed**, and the **same element grouping**. Prediction averaging (Control A), late-fusion (B1/B2), EchoSet-JEPA (C), shuffled-study (C'), metadata-only (D), and identity encoder (E) all share one `study_clip_sample.parquet` manifest per (split, seed, K). This is the single biggest reviewer-proofing decision.

## 4. Architecture

### 4.1 Inputs (per study)

```
c_clip_i        ∈ R^{d_clip}       (from frozen V-JEPA, cached)
e_j  (element)  ∈ R^{d_clip}       (LayerNorm(mean(c_clip over element)))
view_emb_j      ∈ R^{d_meta}       (lookup, with <unknown> id)
modality_emb_j  ∈ R^{d_meta}
phase_emb_j     ∈ R^{d_meta}       (or <na>)
quality_emb_j   ∈ R^{d_meta}       (tertile token)
calib_emb_j     ∈ R^{d_meta}       (optional: frame_rate bucket, pixel_spacing bucket; gated)
[STUDY]         ∈ R^{d_model}      (learned)
[MASK]_t        ∈ R^{d_model}      (learned, one shared vector; slot identity is via meta tokens)
```

No acquisition-order positional embedding. Ordering in training is **permuted every step**.

Meta tokens are **additively fused** into each element's input:

```
x_j = Linear_in(e_j) + view_emb_j + modality_emb_j + phase_emb_j + quality_emb_j + calib_emb_j
```

**Metadata dropout** (independent per token, per sample):

```
p_dropout_view = 0.15
p_dropout_phase = 0.30
p_dropout_modality = 0.10
p_dropout_quality = 0.30
p_dropout_calib = 0.50
```

Dropped tokens → `<unk>` lookup id, not zeroed.

### 4.2 Study transformer

```
d_model:          512 (base), 768 (large for Phase 6)
n_heads:          8
n_layers:         6 (base), 8 (large)
ffn_mult:         4
attn_type:        bidirectional self-attention with padding mask
position:         none (permutation-invariant)
norm:             pre-LN
dropout:          attn 0.0, ffn 0.1
```

Input sequence: `[STUDY] x_1 x_2 ... x_M_ctx [MASK]_{t_1} ... [MASK]_{t_T}`.
Padding to `max_M = 64` with attention mask.

Each `[MASK]_{t}` position is injected with the **target element's meta tokens** (view, modality, phase, quality) — so the transformer knows *what* to predict, not *where* in any sequence.

Output heads:
- `Projector_student(·) : R^{d_model} → R^{d_proj}` on mask-slot outputs.
- `Projector_teacher(·)` (EMA copy) applied to element inputs for targets.

`d_proj = 256`. LayerNorm on both sides.

### 4.3 EMA schedule

```
tau_start = 0.996
tau_end   = 0.9999
schedule  = cosine over planned steps
updated   = stage-2 transformer weights + Projector_teacher
```

V-JEPA clip encoder is **not** EMA-updated in MVP (it's frozen).

## 5. Data pipeline

### 5.1 Manifest schema

Required per-clip fields (`study_clip_manifest.parquet`):

```
patient_id                  : stable hashed id
study_id                    : stable hashed id
clip_id                     : stable hashed id
s3_uri                      : path
dicom_series_uid            : for dedup
view_label                  : str, from echo view classifier
view_conf                   : float in [0,1]
modality                    : str, from dicom metadata + classifier
phase_label                 : optional {systolic, diastolic, full_cycle, na}
phase_conf                  : optional float
quality_score               : float in [0,1] from quality classifier
quality_bucket              : {high, med, low} (per-cohort tertiles)
frame_rate_hz               : float
clip_duration_s             : float
n_frames                    : int
pixel_spacing_cm_per_px     : optional; gated behind 'leak_safe_calib' flag
acquisition_ts              : ISO8601; AUDIT ONLY, not used as input
site_id                     : for held-out-site splits
vendor                      : str, for diagnostics
n_duplicates                : computed; near-dup detection
is_duplicate_of             : optional clip_id
cached_cclip_s3             : s3://.../cclip/{study_id}/{clip_id}.npy
```

### 5.2 Splits

- **Patient-level splits** — default. No patient appears in both train and val/test.
- **Site-level splits** — a held-out site is carved out for Phase 5 (external generalization). Sites chosen to reflect vendor diversity.
- **Study-balanced sampling** — within train, weighted sampling such that studies with N∈{1, 2-3, ≥4} clips each appear with documented rates (`study_size_weighting: none | inv_log | inv_sqrt`, default `inv_log`).

### 5.3 Dedup

Near-duplicate clip filter before element grouping:

```
two clips are near-dup iff
  same study_id
  AND same view_label with view_conf > 0.8
  AND same modality
  AND |frame_count_diff| < 3
  AND |duration_diff| < 0.2s
  AND c_clip cosine > 0.98     # using cached V-JEPA embedding
```

Within a near-dup cluster, keep the highest-quality clip; mark others `is_duplicate_of`.

### 5.4 Handling variable N

```
N_clips == 1       → single element; stage-2 is a no-op supervisor (exclude from MVP pretrain)
N_clips ∈ [2,3]    → include; relax min_context_elements to 1
N_clips ≥ 4        → primary pretrain cohort
```

### 5.5 View-label confidence

```
view_conf ≥ 0.7   → view_label used
view_conf ∈ [0.4, 0.7)  → view_label used with 50% probability; else <unknown>
view_conf < 0.4   → <unknown>
```

Ablation (Phase 5): oracle view vs predicted view vs `<unknown>`-only.

### 5.6 Element grouping

For each study:

```
for each (view_family, modality, phase_bucket, quality_bucket) group:
    if len(group) == 0: skip
    else:                emit element j = LayerNorm(mean({c_clip_i})), with meta tokens
```

Elements per study: mean ~8, median 6, p95 ~18, cap at `max_M = 64`.

## 6. Losses

### 6.1 Primary — masked study-element prediction

```
for each masked index t:
  z_t = stopgrad(LayerNorm(Projector_teacher(e_t_teacher_side)))
  h_t =           LayerNorm(Projector_student(mask_slot_output_t))
  L_regress += 1 - cosine(h_t, z_t)

L_regress = L_regress / |T|
```

SmoothL1 is **explicitly rejected** — prior experiments showed SmoothL1 on unnormalized latents allowed collapse.

### 6.2 Small contrastive term

```
L_nce = InfoNCE(h_t, z_t, batch_neg=all_other_z in batch)
      = -log exp(sim(h_t, z_t)/tau) / sum_k exp(sim(h_t, z_k)/tau)
tau = 0.1
```

Negatives: other masked-element targets in the **same mini-batch** across other studies. Same-study off-targets are **excluded** (attention mask on logits).

### 6.3 Total

```
L_total = L_regress + lambda_nce * L_nce

lambda_nce sweep: {0.01, 0.03, 0.05}   # default 0.03
```

### 6.4 Secondary diagnostics (not primary objective)

```
var_monitor  = mean std over batch of h_t         — VICReg-style, threshold 1.0
cov_monitor  = off-diagonal cov of h_t             — threshold 0.1
```

If `var_monitor < 0.3` for > 500 steps, auto-raise `lambda_nce` to 0.1 and log. This is a **safety rail**, not primary loss.

## 7. Training schedule

| Phase | Duration | What |
|---|---|---|
| warmup | 2k steps | LR 0 → peak, `lambda_nce=0.01` |
| main | 50k steps | peak LR, full `lambda_nce` |
| cooldown | 5k steps | LR → 0 |

```
optimizer:    AdamW
peak_lr:      5e-4 (d_model=512)
weight_decay: 0.05
betas:        (0.9, 0.95)
batch:        32 studies/GPU × 8 GPUs = 256 studies/step
clip_sample:  K=8 per study (primary), view-stratified
mask_ratio:   0.4 of M elements, min 1, max M-1
mask_strategy: random (with stratified variant — mask a whole modality or view_family — as ablation)
```

## 8. Baselines and controls

All baselines consume the **same K=8 clips per study** with the same seeds.

### Control A — V-JEPA + per-clip prediction averaging

Each clip scored independently by a single-clip attentive probe (d=1). Study prediction = arithmetic mean.

**Isolates**: "did we need cross-clip attention at all?"

### Control B1 — V-JEPA + lightweight supervised late-fusion attention pooling

A 2-layer self-attention pool over the K clips' `c_clip`, trained with downstream labels only.

```
d_model = 256, 2 layers, 4 heads, ~0.3M params
```

**Isolates**: "can a small supervised late-fusion matches EchoSet-JEPA?"

### Control B2 — V-JEPA + capacity-matched supervised study transformer

Identical architecture to EchoSet-JEPA (same `d_model`, `n_layers`, `n_heads`, same element grouping, same meta tokens) but **randomly initialized** and trained end-to-end with downstream labels only. **No masked pretraining.**

**Isolates**: "is the masked-prediction pretraining signal doing anything beyond capacity?"

### Control C — EchoSet-JEPA with **shuffled-study** context/target pairs

Same architecture, same masking, same losses. Context elements are drawn from **other random studies** (matched on N and view mix). Targets from the true study.

**Isolates**: "is the model learning study-level structure or just marginal view/modality statistics?"

If C performs as well as EchoSet-JEPA proper → the method has not learned study-level relational structure.

### Control D — metadata-only target prediction

Study transformer receives **only meta tokens** (view/modality/phase/quality) for all positions, zero element content. Downstream probes trained on its `[STUDY]` token.

**Isolates**: "what fraction of downstream performance is driven by view-distribution priors alone?"

### Control E — no-cross-clip-attention / identity study encoder

Study transformer replaced with per-clip LayerNorm + mean pool. Downstream probe trained on pooled output.

**Isolates**: "baseline floor for a K-averaged study representation with matched probe capacity."

### Summary table — what each control rules out

| Control | If EchoSet-JEPA fails to beat it, we conclude |
|---|---|
| A | no cross-clip reasoning learned |
| B1 | cross-clip signal is trivially captured by a small supervised pool |
| B2 | masked pretraining is redundant; capacity explains gains |
| C | model exploits marginal view/modality statistics, not study structure |
| D | performance is driven by metadata priors |
| E | study encoder adds nothing over pooled baseline |

## 9. Evaluation protocol

### 9.1 Downstream tasks

**Main study-level (MVP):**

| Task | Type | Notes |
|---|---|---|
| study-level LVEF | regression | from B-mode only; pooled across all cardiac B-mode views |
| RV function (RVSP) from B-mode | regression | cross-modal; B-mode-only inputs |
| LVH (LVIDd, IVSd) | regression | multi-view advantage expected |
| HCM (if labels available) | classification | expected to benefit from PLAX + A4C fusion |
| HF / mortality (if labels available) | classification | longitudinal label, study-level |
| pediatric external generalization | regression | EchoNet-Pediatric test split |

**Proxy B-mode cross-modal:**
- RVSP from B-mode (no TR Doppler in inputs)
- MR severity from B-mode (no color Doppler)
- AS severity from B-mode (no CW Doppler)

**Full multimodal (Phase 6 only):**
- RVSP with TR spectral Doppler element
- AS with CW Doppler + LVOT element
- MR with color Doppler element
- diastology with mitral inflow + TDI elements

### 9.2 Low-label sweep

All main tasks: 1%, 5%, 10%, 25%, 100% of training labels. Seeds: 3 per fraction.

### 9.3 Robustness

- Held-out-site evaluation (Phase 5).
- View-dropout: at test time, drop each view family independently and re-score. Report degradation curve.
- K-sweep: K ∈ {1, 2, 4, 8, 16}. Report performance-vs-K curves for all methods.

### 9.4 Probe architecture

A single light head on the study transformer's `[STUDY]` token output:

```
probe: LayerNorm → Linear(d_model → d_hidden=256) → GELU → Linear(d_hidden → n_targets)
```

Same probe architecture used across all controls (applied to their respective study-level output). This is critical for fairness.

## 10. Diagnostics

| Name | What it checks |
|---|---|
| context ablation | drop % of context elements → curve of L_regress |
| target-view-only baseline | per-task performance using only target clip's view |
| shuffled-study control (C) | performance gap between real-study and shuffled-study pretraining |
| attention entropy | per-layer mean entropy; collapse if → 0 or → log(M) |
| drop-one-view attribution | Δperf when each view_family removed at probe time |
| performance vs K | curves per task |
| performance vs unique-view-count | controls for "many-clips ≠ many-views" |
| performance vs quality | tertile-stratified |
| study-size bias | is the method better on N≥10 studies and worse on N≤3? |
| retrieval (aux) | rank-1 / rank-5 of masked element from same study vs other studies |
| var/cov monitoring | per batch, per layer |
| metadata shortcut | compare Control D to full method — is the gap > noise? |

**Retrieval is a diagnostic, not a success criterion.** The MV2SV era proved retrieval gains do not imply downstream gains.

## 11. Edge cases

- **Study with N=1 clip**: excluded from stage-2 pretrain. At downstream eval, the study encoder still runs (single element + `[STUDY]` token); probe sees a degenerate input, treated as the floor.
- **Study with 20+ redundant A4C clips, no other views**: elements collapse to 1-3 by grouping; study transformer essentially averages. Expected to under-perform but not crash.
- **Missing view labels for all clips**: all meta tokens go to `<unknown>`; study transformer must use element content only. Expected to degrade but still function.
- **Doppler-only study**: Phase 6 territory; in MVP, excluded.
- **Mixed-vendor study**: vendor not input; diagnostic only.
- **Degenerate mask selection (masks all elements)**: clamped to `max_mask = M - 1`.
- **Duplicate-heavy study post-dedup**: study has M_effective < `min_context_elements`; drop from batch with a retry.

## 12. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Cached `c_clip` drifts from online V-JEPA | freeze V-JEPA checksum; re-cache if checksum changes |
| Metadata shortcut dominates | Control D quantifies; if D ≈ full method, fail the run |
| Masked-prediction collapse | var/cov monitoring; lambda_nce auto-raise; stop-grad on teacher |
| Sampler-driven gains | every control uses the **same seeded K clips** manifest |
| K-sweep advantage only at large K | report curves; claim calibrated to where it actually wins |
| View-label leakage (e.g. view label correlates with ejection fraction cohort) | metadata dropout + `<unknown>` + Control D |
| Pediatric regression | include pediatric in headline table; **hurting pediatric is an auto-fail** |
| Attention entropy collapse to a single clip | monitored; if it happens, the model is performing Control A in disguise |
| Site / vendor leakage | held-out-site split; vendor ablation |

## 13. MVP implementation checklist

Ordered; each item is a landable PR-sized unit.

1. `experiments/echoset_jepa/build_manifest.py` — build `study_clip_manifest.parquet` from existing sources (UHN + MIMIC). Fields per §5.1.
2. `experiments/echoset_jepa/dedup.py` — near-dup clustering; emits `n_duplicates`, `is_duplicate_of`.
3. `experiments/echoset_jepa/element_grouping.py` — (view, mod, phase, qual) grouping; emits `study_element_manifest.parquet`.
4. `experiments/echoset_jepa/cache_cclip.py` — batch V-JEPA forward on all clips; write to S3 (`cached_cclip_s3` column).
5. `experiments/echoset_jepa/sample_K.py` — view-stratified fixed-K sampler, seeded; emits `study_clip_sample_K{K}_seed{S}.parquet`. Used by all controls.
6. `src/datasets/echoset_jepa_dataset.py` — dataset yielding batched study elements with meta tokens and mask indices.
7. `src/models/study_transformer.py` — §4.2 architecture.
8. `src/models/projectors.py` — 2-layer MLP projectors (student and teacher EMA copy).
9. `app/echoset_jepa/train.py` — training loop, EMA update, masked-prediction losses, diagnostics.
10. `app/echoset_jepa/__init__.py` — scaffold entry.
11. `evals/echoset_jepa_probe/` — downstream study-level probe (d=1 equivalent on `[STUDY]` token); shared across all controls.
12. `evals/echoset_jepa_probe/control_A.py` — per-clip attentive probe + mean aggregation over same-K manifest.
13. `evals/echoset_jepa_probe/control_B1.py` — lightweight supervised late-fusion.
14. `evals/echoset_jepa_probe/control_B2.py` — capacity-matched supervised study transformer.
15. `evals/echoset_jepa_probe/control_D.py` — metadata-only study transformer.
16. `evals/echoset_jepa_probe/control_E.py` — identity study encoder.
17. `scripts/echoset_jepa/shuffled_study_pretrain.py` — Control C pretrain variant.
18. `tests/echoset_jepa/` — unit tests for grouping, sampling, masking, EMA update, loss values on synthetic data.
19. `configs/train/echoset_jepa/echoset_jepa_v1_K8.yaml` — primary pretrain config.
20. `configs/eval/echoset_jepa/{task}_K8.yaml` — one per downstream task.

## 14. File / module-level code changes

**New** (all under fresh paths — no modifications to existing V-JEPA training code):

```
app/echoset_jepa/
  __init__.py
  train.py
  collate.py
src/datasets/
  echoset_jepa_dataset.py              [new]
  echoset_jepa_sampler.py              [new; view-stratified K sampler]
src/models/
  study_transformer.py                 [new]
  study_projectors.py                  [new]
  meta_embeddings.py                   [new]
experiments/echoset_jepa/
  build_manifest.py
  dedup.py
  element_grouping.py
  cache_cclip.py
  sample_K.py
evals/echoset_jepa_probe/
  __init__.py
  probe.py
  control_A.py
  control_B1.py
  control_B2.py
  control_C_pretrain.py
  control_D.py
  control_E.py
tests/echoset_jepa/
  test_element_grouping.py
  test_masking.py
  test_ema.py
  test_sampler_k_matched.py
  test_loss_synthetic.py
configs/train/echoset_jepa/
  echoset_jepa_v1_K8.yaml
  echoset_jepa_v1_shuffled_study.yaml   # Control C
configs/eval/echoset_jepa/
  lvef_K8.yaml
  rvsp_bmode_K8.yaml
  lvidd_K8.yaml
  hcm_K8.yaml
  ped_lvef_K8.yaml
  k_sweep/lvef_K{1,2,4,16}.yaml
```

**Modified** — none in V-JEPA core. The only repo-wide touch is a new `app_name` dispatch path in `app/scaffold.py`:

```python
# app/scaffold.py  — add one case
if app_name == "echoset_jepa":
    from app.echoset_jepa.train import main as run
    return run(...)
```

## 15. Config templates

### 15.1 Primary pretrain — `configs/train/echoset_jepa/echoset_jepa_v1_K8.yaml`

```yaml
app: echoset_jepa
experiment:
  study_transformer:
    d_model: 512
    n_layers: 6
    n_heads: 8
    ffn_mult: 4
    dropout: 0.1
    max_M: 64
  elements:
    view_families: [apical, parasternal_long, parasternal_short, subcostal, suprasternal, doppler_spectral, m_mode, tdi, unknown]
    modalities: [b_mode, color_doppler, pw_doppler, cw_doppler, m_mode, tdi, contrast]
    phase_buckets: [systolic, diastolic, full_cycle, na]
    quality_buckets: [high, med, low]
    element_agg: mean        # or quality_weighted
    min_clips_per_element: 1
  meta_dropout:
    view: 0.15
    modality: 0.10
    phase: 0.30
    quality: 0.30
    calib: 0.50
  masking:
    mask_ratio: 0.4
    min_mask: 1
    max_mask: M_minus_1
    strategy: random        # or stratified_modality | stratified_view_family
  sampler:
    K: 8
    strategy: view_stratified
    seed: 0
  loss:
    objective: cosine_plus_nce
    lambda_nce: 0.03
    tau: 0.1
  ema:
    tau_start: 0.996
    tau_end: 0.9999
    schedule: cosine
  clip_encoder:
    source: cached
    frozen: true
    cache_s3_prefix: s3://.../echoset_jepa/cclip/
  optim:
    lr: 5.0e-4
    wd: 0.05
    betas: [0.9, 0.95]
    warmup_steps: 2000
    main_steps: 50000
    cooldown_steps: 5000
    batch_studies_per_gpu: 32
```

### 15.2 Control C (shuffled study)

Same as above with:
```yaml
  masking:
    shuffled_context: true
    shuffle_match_on: [n_elements, view_family_mix_bucketed]
```

### 15.3 Downstream probe — `configs/eval/echoset_jepa/lvef_K8.yaml`

```yaml
eval_name: echoset_jepa_probe
experiment:
  pretrain_checkpoint: <path to stage-2 ckpt>
  clip_cache_s3: s3://.../cclip/
  sampler:
    K: 8
    strategy: view_stratified
    seed: 0                # match across all controls
  probe:
    head: layernorm_mlp
    d_hidden: 256
    task: regression
    target: study_lvef
  optim:
    lr: 5.0e-5
    wd: 0.05
    warmup: 500
    total_steps: 8000
```

## 16. Success / failure gates

### Success — **must hold jointly** for the primary claim

1. EchoSet-JEPA beats A by ≥ 1 absolute metric unit (or ≥ 2% relative) on ≥ 3 of the main study-level tasks at K=8, on patient-split test.
2. EchoSet-JEPA beats B2 (capacity-matched supervised transformer) on the low-label regime (1%, 5%, 10%) on ≥ 2 tasks.
3. EchoSet-JEPA beats B1 at held-out-site eval on ≥ 2 tasks.
4. EchoSet-JEPA is **not worse** than A on pediatric external generalization.
5. Control C (shuffled-study) is strictly worse than real-study EchoSet-JEPA on primary tasks — gap ≥ 50% of the A-vs-EchoSet-JEPA gap.
6. Control D (metadata-only) is strictly worse — gap ≥ 80% of the A-vs-EchoSet-JEPA gap.
7. View-dropout robustness: EchoSet-JEPA's degradation when one view family is dropped is ≤ 80% of A's degradation on ≥ 2 tasks.

### Failure — any of these triggers a rewrite or abandonment

- Only beats A, not B1 or B2.
- Only wins on LVEF; ties or loses on multi-view tasks (RVSP, MR, AS, LVIDd).
- Gains vanish under patient split or site split.
- Gains are entirely explained by Control D (metadata priors).
- Retrieval improves but downstream does not.
- Hurts pediatric or external-site generalization (≥ 2% relative regression).
- Attention entropy collapses (`H_attn < 0.1 * log(M)`) — model is performing Control A in disguise.

## 17. Timeline

| Phase | Calendar | Deliverable |
|---|---|---|
| P0 audit | week 1 | manifest built, clip counts, view distributions, duplicate rates, quality tertiles reported |
| P1 cache | week 1-2 | V-JEPA c_clip cached for all in-scope studies; I/O throughput verified |
| P2 pretrain v1 | week 2-4 | K=8 + grouped elements, 50k steps, `lambda_nce=0.03`, stop-grad teacher EMA |
| P3 controls | week 3-5 (parallel with P2 late) | A, B1, B2, C, D, E run on same K=8 manifest |
| P4 downstream + K sweep | week 4-6 | all main tasks + 1/5/10/25/100% label sweep + K∈{1,2,4,8,16} |
| P5 robustness | week 5-7 | held-out-site, view-dropout, vendor ablation, predicted-vs-oracle view |
| P6 modality extension | week 7-9 *(optional)* | CW/color Doppler, TDI, M-mode as first-class elements; Doppler-native tasks |
| P7 local-motion branch | week 8-10 *(optional, orthogonal)* | token-level TAPSE; NOT part of EchoSet-JEPA v1 |

Gate from P4 → P5: success criteria 1-3 must hold on patient split before committing cluster time to P5.
Gate from P5 → P6: success criteria 1-7 must all hold.

---

## Appendix — what this plan explicitly does not ship

- Any per-clip token-level objective (TAPSE branch) — deferred to P7 as a separate method, not EchoSet-JEPA v1.
- Any paper edits or Overleaf pushes.
- Any modification to V-JEPA 2.1 / EchoJEPA pretraining code paths.
- A production multi-view inference runtime — this is a research pipeline.
- A Doppler-native element loss beyond treating Doppler as another element class — full Doppler modeling is P6.
- Online joint training of V-JEPA + study transformer — deferred; frozen is scientifically cleaner for v1 and eliminates one confound in the controls.
