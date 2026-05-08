# EchoMV-JEPA Implementation Sketch

**Status:** design document, PR-0. Engineer-facing reference for PR-1 onward.
**Companion docs:** `docs/echomv_jepa_architecture_plan.md` (the plan), `docs/echomv_jepa_vs_prior_variants.md` (differentiation).

This document specifies each new module's contract: inputs, outputs, tensor shapes, key YAML config fields, failure modes, and unit tests. Nothing here is implemented in PR-0; the purpose is to make PR-1..PR-9 reviewer-defensible.

---

## 0. Tensor-shape glossary

| Symbol | Meaning |
|--------|---------|
| `B` | batch size (studies per GPU) |
| `K` | selected clips per study (MVP: K=8) |
| `M` | total study elements after grouping by `(view_family, modality, phase_bucket)` (`1 ≤ M ≤ max_M=64`) |
| `M_ctx` | number of context elements after masking (`M_ctx ≥ 1`) |
| `M_tgt` | number of target (masked) elements (`1 ≤ M_tgt ≤ ceil(0.6 M)`) — note `M_ctx + M_tgt = M` |
| `T` | tokens per clip (when `keep_tokens: true`); ViT-L/16 video: T = 8·14·14 = 1568 tubelets |
| `T_e` | tokens per element (per-clip T times #clips in element, optionally subsampled) |
| `D_clip` | clip encoder embedding dimension (ViT-L/16 = 1024) |
| `d_model` | study transformer width (MVP: 512) |
| `D_proj` | projection head output dim (MVP: 256) |
| `k_anchor` | number of visible target tokens per target slot in Stage-3 anchor mode C |

Pooled path shapes (MVP):

```
c_clip            : (B, K, D_clip)
e_elem            : (B, M, D_clip)               # mean (or quality-weighted) pool of clips per element
ctx_elements      : (B, M_ctx, D_clip)
tgt_elements      : (B, M_tgt, D_clip)
ctx_meta_add      : (B, M_ctx, d_model)
tgt_meta_add      : (B, M_tgt, d_model)
ctx_pad_mask      : (B, M_ctx) bool              # True = padded
tgt_pad_mask      : (B, M_tgt) bool
full_pad_mask     : (B, M_ctx + M_tgt) bool      # for teacher full-study pass
target_idx_in_full: (B, M_tgt)                   # gather indices into full stream

h_study (student) : (B, d_model)
h_mask  (student) : (B, M_tgt, d_model)
h_t     (student) : (B, M_tgt, D_proj)

z_per_element     : (B, M, d_model)              # teacher output over full unmasked study
z_at_targets      : (B, M_tgt, d_model)
z_t               : (B, M_tgt, D_proj)           # stopgrad
```

Token path shapes (Stages 3+, optional):

```
clip_tokens       : (B, K, T, D_clip)
element_tokens    : (B, M, T_e, D_clip)
```

---

## 1. Anti-leak rules (enforced by tests)

A single, authoritative list. Any test file that references "leak" must assert these.

1. **No target visual latent in context stream.** `ctx_elements` never contains `tgt_elements`. Verified by comparing tensor contents + index sets.
2. **Target quality is off by default.** `include_target_quality=False` in `encode_target_slot` (inherits EchoSet v1).
3. **Target measurement-site is off by default.** `tgt_meta_site` never emitted unless explicitly configured.
4. **Acquisition order is off by default.** No positional embedding across elements; dataloader shuffles order every step.
5. **Vendor / site are off by default.**
6. **Report-derived metadata is off by default.**
7. **Measurement-derived labels are off by default.** Labels are never in the input.
8. **Overlay / measurement text** is masked before visual encoding where possible (existing repo practice; retained).
9. **Target anchor invariants I1–I5** (§6 below) enforced when `target_slot_mode ∈ {B, C}`.
10. **Teacher input is unchanged by anchor flag.** `F_bar_psi` receives the raw full study, never the anchor-reduced stream (I3).
11. **CALA is online-only.** `F_bar_psi` never has CALA adapters; any configuration that enables CALA on the teacher is rejected at config-load time.
12. **Same-study off-targets excluded from NCE negatives.** Inherits EchoSet v1's `_prioritized_neg_pool`.

---

## 2. Modules under `src/models/echomv_jepa/`

### 2.1 `clip_backbone.py` — `EchoMVClipBackbone`

**Purpose:** thin wrapper around the existing V-JEPA ViT clip encoder (from `src/models/vision_transformer.py`) with flags for `freeze`, `keep_tokens`, and adapter insertion.

**Inputs:**
- `x: (B*K, C=3, T_frames, H, W)` — concatenated clips across studies.
- `pool: bool = True` — if True, returns pooled; else returns token-level.

**Outputs:**
- If `pool=True`: `c_clip : (B*K, D_clip)`.
- If `pool=False`: `c_clip_tokens : (B*K, T, D_clip)`.

**Key config fields (YAML under `experiment.clip_encoder`):**

```yaml
clip_encoder:
  name: vitl16              # hub key
  checkpoint: /path/to/vjepa2_vitl16.pt
  freeze: true              # Stage-1 MVP
  keep_tokens: false        # true at Stage-3+
  adapter:
    type: none              # none | lora | top_blocks
    lora_rank: 8            # when type=lora
    top_blocks_n: 2         # when type=top_blocks
```

**Failure modes:**
- Checkpoint dim mismatch (unit test: shape of `c_clip`).
- Accidental gradient on frozen params (unit test: `.grad is None` after backward).

**Unit tests:**
- `test_frozen_clip_no_grad`
- `test_keep_tokens_shape`
- `test_adapter_param_count` (Stage-2+)

### 2.2 `study_encoder.py` — re-export / thin extension

**Purpose:** re-export `src/models/study_transformer.StudyTransformer` as `EchoMVStudyEncoder`. Optionally insert CALA zero-init residuals (Stage-4) via a mode flag.

**Inputs / outputs / shapes:** identical to `StudyTransformer.forward` (see `src/models/study_transformer.py:73–108`).

**Key config fields:**

```yaml
study_encoder:
  d_clip: 1024
  d_model: 512
  n_layers: 4
  n_heads: 8
  ffn_mult: 4
  dropout_ffn: 0.1
  dropout_attn: 0.0
  max_M: 64
  cala:
    enabled: false          # Stage-4 only
    gamma_init: 0.0
    gamma_warmup_steps: 5000
    n_heads: 8
    bias_dim: 32
    anatomy_vocab: [LV, RV, MV, AV, TV, IVC, unknown]
```

**Failure modes:**
- CALA enabled on the teacher at config load → reject.
- `d_clip` mismatch with clip backbone.

**Unit tests:**
- `test_permutation_invariance` — shuffle element order, output is equivariant modulo the shuffle permutation; `h_study` (at `[STUDY]` position) is invariant.
- `test_pad_mask_respected` — padded positions do not affect unpadded output.

### 2.3 `study_target_encoder.py` — `StudyTransformerEMA`

**This is the defining new module of EchoMV-JEPA.**

**Purpose:** EMA copy of `StudyTransformer` that operates on the full unmasked study and exposes per-element contextualized hidden states at arbitrary indices.

**Class sketch:**

```python
class StudyTransformerEMA(nn.Module):
    def __init__(self, student: StudyTransformer) -> None:
        super().__init__()
        self.teacher = copy.deepcopy(student)
        for p in self.teacher.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update_teacher(self, student: StudyTransformer, tau: float) -> None:
        # foreach for speed, identical pattern to app/vjepa/train.py:800
        params_s = list(student.parameters())
        params_t = list(self.teacher.parameters())
        torch._foreach_mul_(params_t, tau)
        torch._foreach_add_(params_t, params_s, alpha=1.0 - tau)

    @torch.no_grad()
    def forward_full_study(
        self,
        full_elements: torch.Tensor,     # (B, M, D_clip)
        full_meta_add: torch.Tensor,     # (B, M, d_model)
        full_pad_mask: torch.Tensor,     # (B, M) bool
    ) -> torch.Tensor:
        # Run teacher over full unmasked stream: no mask slots.
        # Returns (B, M, d_model) — the per-element contextualized hidden state.
        # Equivalent to StudyTransformer.forward with empty tgt_*, but returns
        # the context-position outputs (not h_study).
        ...

    @torch.no_grad()
    def select_at_targets(
        self,
        z_per_element: torch.Tensor,     # (B, M, d_model)
        target_idx: torch.Tensor,        # (B, M_tgt) long
    ) -> torch.Tensor:
        # gather
        return torch.gather(
            z_per_element, dim=1,
            index=target_idx.unsqueeze(-1).expand(-1, -1, z_per_element.shape[-1]),
        )                                 # (B, M_tgt, d_model)
```

**Inputs to `forward_full_study`:**
- `full_elements : (B, M, D_clip)` — concat of ctx_elements + tgt_elements in a known order.
- `full_meta_add : (B, M, d_model)` — concatenated meta embeddings.
- `full_pad_mask : (B, M) bool`.

**Output:**
- `z_per_element : (B, M, d_model)` — contextualized, pre-projection.

**Downstream:** caller projects `z_at_targets` through `EMAProjectorPair.teacher_forward` and applies `stopgrad`.

**Key config fields:**

```yaml
target_encoder:
  ema:
    tau_start: 0.996
    tau_end: 0.9999
    schedule: cosine        # cosine_schedule from src/models/study_projectors
  full_study_pass: true     # must be true for EchoMV Stage-1+
```

**Failure modes:**
- Teacher-student weight drift (unit test: after many updates, teacher differs from student; with `tau=1.0`, teacher does not update).
- Accidental gradient through teacher (unit test: `.requires_grad=False` for all teacher params, `torch.no_grad()` wrappers).
- Mis-gather of target indices (unit test: gathered output matches per-element output at known indices).
- CALA accidentally enabled (unit test: `hasattr(teacher, 'cala')` is False OR CALA disabled in teacher config).

**Unit tests (PR-3):**
- `test_ema_update` — after `update_teacher(tau=0.5)`, teacher param = 0.5·teacher_old + 0.5·student.
- `test_teacher_no_grad` — all teacher params `requires_grad=False`.
- `test_full_study_target_encoder_shapes` — for arbitrary B/M, `z_per_element.shape == (B, M, d_model)`.
- `test_gather_at_targets` — with known target indices, gathered output equals the corresponding per-element row.
- `test_teacher_input_unchanged_by_anchor_flag` — hashes of `full_elements`, `full_meta_add`, `full_pad_mask` are equal between anchor-off and anchor-on configurations.

### 2.4 `predictor.py` — `EchoMVPredictor`

**Purpose:** map student's `h_mask` (context-aware target-slot output) to the projection space where the target lives.

**MVP implementation:** re-uses `EMAProjectorPair.student_forward` (2-layer MLP: `d_model → d_hidden → D_proj`). No new module needed for MVP.

**Stage-2+ optional:** a small transformer predictor (1–2 blocks) that cross-attends between target slots and a summary of context, paralleling `VisionTransformerPredictor` at the element level. Deferred until Stage-2.

**Inputs:**
- `h_mask : (B, M_tgt, d_model)` — student context encoder output at target slot positions.
- `tgt_meta_add : (B, M_tgt, d_model)` — optional explicit conditioning (MVP: already fused upstream).

**Outputs:**
- `h_t : (B, M_tgt, D_proj)`.

**Key config fields:**

```yaml
predictor:
  mode: projector_only      # MVP
  d_hidden: 1024
  d_proj: 256
```

**Unit tests:**
- `test_predictor_shapes`

### 2.5 `correspondence_attention.py` — `CALA` (Stage-4)

**Purpose:** zero-init correspondence-aware latent attention, inserted into `F_psi` only (never teacher).

**Class sketch:**

```python
class CALA(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        bias_dim: int = 32,
        anatomy_vocab_size: int = 7,
        view_family_vocab_size: int = 9,   # matches src/models/meta_embeddings
        meta_dropout_prob: float = 0.15,
        gamma_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.cross_attn = CrossAttentionBlock(...)   # reuse src/models/attentive_pooler
        self.bias_table = nn.Parameter(
            torch.zeros(view_family_vocab_size, view_family_vocab_size,
                        anatomy_vocab_size, anatomy_vocab_size, bias_dim))
        self.bias_proj = nn.Linear(bias_dim, n_heads)
        self.gamma = nn.Parameter(torch.full((n_heads,), gamma_init))
        self.meta_dropout_prob = meta_dropout_prob

    def forward(self, x, view_ids_q, view_ids_k, anatomy_ids_q, anatomy_ids_k, training):
        # Build per-head attention bias from bias_table, with dropout on anatomy keys.
        # Apply cross-attn with bias; scale output by gamma (per-head).
        # Return x + gamma * cross_attn(x, ...)  — residual, zero-init by gamma.
        ...
```

**Inputs:**
- `x : (B, M, d_model)`.
- `view_ids_q, view_ids_k : (B, M)` long.
- `anatomy_ids_q, anatomy_ids_k : (B, M)` long.

**Outputs:**
- `x_out : (B, M, d_model)`.

**Key config fields (subset of `study_encoder.cala`):** see §2.2.

**Failure modes:**
- γ not zero at init (unit test: γ = 0 → output equals input).
- Bias table used as identity routing (mitigated by low bias_dim=32, plus anatomy-key dropout).
- Attention entropy collapse within 500 steps → halt.
- Enabled on teacher → config-load rejection.

**Unit tests (PR-8):**
- `test_cala_zero_init` — with `gamma=0`, `F_psi` output hash equals Stage-1 encoder's output hash on the same batch.
- `test_cala_bias_dropout_applied` — anatomy key dropout fires at configured probability.
- `test_cala_not_on_teacher` — config with `target_encoder.cala.enabled=true` raises at load.

### 2.6 `target_anchor.py` — `TargetAnchor` (Stage-3)

**Purpose:** implement the three target-slot modes A/B/C from architecture plan §7.8.

**Modes:**
- `A = metadata_only` (MVP default): target slot = `mask_token + tgt_meta_add`.
- `B = low_res_anchor`: add a low-res summary produced by a **distinct head** (`nn.Linear` on a pooled downsample of the target element's tokens) to the target slot.
- `C = partial_visible`: expose a subset of `k_anchor` tokens of the target element's token stream to the predictor *only*.

**Inputs:**
- `mask_token_at_slots : (B, M_tgt, d_model)` — the student's un-projected mask slot.
- `tgt_element_tokens : (B, M_tgt, T_e, D_clip)` (mode B and C only).
- `mode : str` ∈ {A, B, C}.
- `k_anchor : int` (mode C only).

**Outputs:**
- `anchored_slot : (B, M_tgt, d_model)` — passed into `F_psi` at mask positions.

**Invariants (required, tested):**

| ID | Invariant |
|----|-----------|
| I1 | `ctx ∩ tgt == ∅` and `ctx ∪ tgt == all_indices` per study. Anchor tokens are a strict subset of target-slot tokens, never in context. |
| I2 | Per target slot, at most `k_anchor` tokens from the full-resolution target element are visible to the online predictor; budget logged. |
| I3 | Teacher input is unchanged by anchor flag. Hash of `full_elements`, `full_meta_add`, `full_pad_mask` is identical between anchor-on and anchor-off for the same batch. |
| I4 | Low-res anchor produced by a head with parameters **distinct** from the target projector. `id(low_res_head) != id(target_projector)`. |
| I5 | `k_anchor=0` reproduces Stage-1 loss numerically (tolerance 1e-5). |

**Failure modes:**
- Anchor tokens appear in context (violates I1) — test fails.
- Anchor budget exceeded — test fails and logs.
- Teacher sees anchor — test fails.
- Low-res head shares weights with projector — test fails.

**Unit tests (PR-7):**
- `test_ctx_tgt_disjoint_under_all_strategies` — for each masking strategy, assert I1.
- `test_anchor_budget_respected` — fuzz with random (K, M, k_anchor); assert I2.
- `test_zero_anchor_equals_stage1_loss` — assert I5 within 1e-5.
- `test_anchor_tokens_not_in_context_stream` — introspect assembled context tensor's index set.
- `test_teacher_input_unchanged_by_anchor_flag` — assert I3 via hash equality.
- `test_lowres_anchor_head_distinct_parameters` — assert I4.

### 2.7 `modality_adapters.py` (Stage-6)

**Purpose:** per-modality stems / adapters for color Doppler, spectral Doppler (CW/PW), M-mode, TDI. Not in MVP; placeholder defined for Stage-6.

**Inputs / outputs:** match `EchoMVClipBackbone` I/O; route by modality id.

**Key config fields:**

```yaml
modality_adapters:
  color_doppler:   {type: stem, channels_in: 3, pretrained: null}
  pw_doppler:      {type: stem, channels_in: 1, pretrained: null}
  cw_doppler:      {type: stem, channels_in: 1, pretrained: null}
  m_mode:          {type: stem, channels_in: 1, pretrained: null}
  tdi:             {type: stem, channels_in: 3, pretrained: null}
```

**Unit tests (Stage-6):** deferred.

### 2.8 `losses.py`

**Functions:**
- `cosine_regression(h_t, z_t, valid_mask) -> Tensor` — Stage-1 core loss.
- `matched_nce(h_t, z_t, view_ids, modality_ids, phase_ids, study_ids, tau=0.1) -> Tensor` — reuses EchoSet v1's `_prioritized_neg_pool` (import from `app/echoset_jepa/train.py` or copy to shared utility).
- `anchor_to_base_clip(c_clip_adapter, c_clip_base, valid_subset) -> Tensor` — Stage-2 guardrail.
- `sigreg_diagnostic(h_t) -> Tensor` — VICReg-style variance + covariance (Stage-5 optional monitor).

**Unit tests:**
- `test_cosine_regression_known_values`
- `test_matched_nce_no_same_study_off_target`
- `test_matched_nce_fallback_ladder_logged`
- `test_anchor_to_base_clip_identity` — with `c_clip_adapter == c_clip_base`, loss is 0.

### 2.9 `ema.py`

**Purpose:** generic EMA helper shared by clip and study encoders.

**Functions:**
- `ema_update(params_teacher, params_student, tau)` — uses `torch._foreach_mul_/add_` pattern from `app/vjepa/train.py:800–809`.
- `cosine_schedule(step, total, tau_start, tau_end)` — re-exports from `src/models/study_projectors` so there is exactly one implementation.

**Unit tests:**
- `test_ema_update_correctness`
- `test_cosine_schedule_endpoints`

---

## 3. `src/datasets/echomv_jepa_dataset.py`

### 3.1 Purpose

Extends `src/datasets/echoset_jepa_dataset.EchoSetJEPADataset` with:
1. **Full-study target pass support:** emits `full_elements`, `full_meta_add`, `full_pad_mask`, `target_idx_in_full` (the gather indices) in addition to the existing ctx/tgt split.
2. **Token retention flag:** optional per-clip tokens when `clip_encoder.keep_tokens: true`.
3. **Target-anchor mode dispatch:** returns the necessary tensors for modes A/B/C (Stage-3).
4. **Nothing else changes.** Masking strategies, meta dropout, K-sampling, element grouping are identical to EchoSet v1.

### 3.2 Contract (batch dict yielded by the collate)

```python
batch = {
    # Existing EchoSet v1 keys:
    "ctx_elements":    (B, M_ctx, D_clip) float,
    "tgt_elements":    (B, M_tgt, D_clip) float,
    "ctx_meta_view":   (B, M_ctx) long,
    "ctx_meta_modality": (B, M_ctx) long,
    "ctx_meta_phase":    (B, M_ctx) long,
    "ctx_meta_quality":  (B, M_ctx) long,
    "tgt_meta_view":     (B, M_tgt) long,
    "tgt_meta_modality": (B, M_tgt) long,
    "tgt_meta_phase":    (B, M_tgt) long,
    "ctx_pad_mask":    (B, M_ctx) bool,
    "tgt_pad_mask":    (B, M_tgt) bool,
    "study_id_int":    (B,) long,

    # NEW for EchoMV:
    "full_elements":        (B, M, D_clip) float,        # ctx ∪ tgt in a fixed order
    "full_meta_view":       (B, M) long,
    "full_meta_modality":   (B, M) long,
    "full_meta_phase":      (B, M) long,
    "full_pad_mask":        (B, M) bool,
    "target_idx_in_full":   (B, M_tgt) long,             # gather indices
    "context_idx_in_full":  (B, M_ctx) long,             # complement, for assertions

    # OPTIONAL (Stage-3+ if keep_tokens=true):
    "ctx_element_tokens":   (B, M_ctx, T_e, D_clip) float,
    "tgt_element_tokens":   (B, M_tgt, T_e, D_clip) float,   # only used by target anchor B/C
}
```

### 3.3 Key config fields

```yaml
dataset:
  name: echomv_jepa
  k_sample_manifest: /.../study_clip_sample_K8_seed0.parquet
  cache_prefix: /mnt/.../cached_cclip/
  element_agg: mean              # mean | quality_weighted
  keep_tokens: false             # true at Stage-3+
  target_slot_mode: A            # A | B | C  (Stage-3 toggles B/C)
  k_anchor: 0                    # C mode only
  strategy_weights:
    random_element:              0.40
    whole_view_family:           0.25
    whole_modality:              0.20
    apical_holdout:              0.05
    parasternal_holdout:         0.05
    color_holdout:               0.025
    spectral_holdout:            0.025
  meta_dropout:
    view: 0.15
    modality: 0.10
    phase: 0.30
    quality: 0.30
  target_meta:
    include_phase: true
    include_quality: false       # anti-leak
    include_site: false          # anti-leak
```

### 3.4 Failure modes

- `full_elements` ordering inconsistent with `target_idx_in_full` → unit test.
- `target_idx_in_full ∩ context_idx_in_full ≠ ∅` → unit test (I1).
- Target anchor tokens leak into context stream → unit test.
- Element_agg="quality_weighted" with no quality field populated → graceful fallback + warning.

### 3.5 Unit tests (PR-2)

- `test_masking_no_target_leak` — assert `ctx_elements` does not contain any row from `tgt_elements`.
- `test_full_study_consistency` — `full_elements[context_idx_in_full] == ctx_elements` and `full_elements[target_idx_in_full] == tgt_elements` (modulo ordering from gather).
- `test_k_sampler_fairness` — all baselines loaded from the same manifest receive identical K clip IDs per study.
- `test_permutation_invariance` — shuffle seed changes element order but not the set; downstream loss is equivariant.
- `test_meta_dropout_applied_context_only` — target meta is never dropped.

---

## 4. `experiments/echomv_jepa/`

### 4.1 `build_multiview_manifest.py`

**Purpose:** extend `experiments/echoset_jepa/build_manifest.py` with calibration fields required for Stage-6. Fields populated in Stage-1 but not consumed until Stage-6.

**New manifest columns:**

| Column | Type | Source | Required from | Notes |
|--------|------|--------|---------------|-------|
| `velocity_scale_cm_per_s` | float | DICOM tag | Stage-6 | nullable for B-mode |
| `nyquist_limit_cm_per_s` | float | DICOM tag | Stage-6 | Doppler only |
| `sweep_speed_mm_per_s` | float | DICOM tag | Stage-6 | spectral only |
| `pixel_spacing_cm_per_px` | float | DICOM tag | Stage-6 | required |
| `frame_rate_hz` | float | DICOM tag | Stage-6 | required |
| `doppler_mode` | str | classifier / DICOM | Stage-6 | one of {none, color, pw, cw} |
| `measurement_site` | str | classifier / annotation | Stage-6 | per §3 vocabulary |

**Failure modes:** missing fields → populate as NaN (not consumed in Stage-1), log coverage percentage.

### 4.2 `cache_tokens.py` (Stage-3+)

**Purpose:** produce per-clip token cache (not just pooled) when `clip_encoder.keep_tokens=true`.

**Output:** `cached_tokens_s3: s3://bucket/tokens/<study>/<clip>.npy` of shape `(T, D_clip)`.

**Failure modes:** disk/S3 space; per-clip size ~6 MB (`T=1568 × D_clip=1024 × float32`).

### 4.3 `target_difficulty_audit.py`

**Purpose:** extends `experiments/echoset_jepa/target_difficulty_audit.py` to stratify target prediction residual by (full-study context richness, view/modality relation, phase distance, quality tertile).

**Output CSV columns:** `study_id, target_element_key, residual, M_ctx, view_distance, modality_distance, phase_distance, quality_tertile`.

### 4.4 `echometr3r_consistency.py`

**Purpose:** implement the §15 Echo-MEt3R suite from the architecture plan.

**Produces:** a single CSV per checkpoint with columns `metric_name, value, strata, ci_low, ci_high`. Metrics: same-study view consistency, cross-modality consistency, phase consistency, shuffled-study consistency, view-dropout consistency, wrong-phase inconsistency, consistency independent of quality tertile.

---

## 5. `app/echomv_jepa/`

### 5.1 `train.py`

**Purpose:** Stage-1 entry point. Extends `app/echoset_jepa/train.py` with:
- instantiation of `StudyTransformerEMA` (teacher study transformer);
- a second forward pass on the full unmasked study;
- gather at target slot indices;
- the falsification-probe log (`cosine(z_EchoMV, z_v1)`);
- EMA updates for both the projector **and** the teacher study transformer each step.

**Pseudocode (reproduces §11 of the architecture plan):** see `docs/echomv_jepa_architecture_plan.md` §11.

**Logging added over v1:**
- `z_cosine_vs_v1_mean` (falsification probe).
- `teacher_st_ema_tau` per step.
- `num_cala_warmup_steps_remaining` (Stage-4 only).
- Per-stratum target prediction residual (hourly flush).

### 5.2 `eval_consistency.py`

**Purpose:** runs Echo-MEt3R diagnostics (§4.4) on a checkpoint; CLI-callable for CI smoke tests.

---

## 6. `configs/train/echomv_jepa/`

All configs inherit v1's structure (see `configs/train/echoset_jepa/echoset_jepa_v1_K8.yaml`).

### 6.1 `stage1_frozen_clip_full_study_ema.yaml` (MVP)

Key knobs relative to v1:

```yaml
experiment:
  target_encoder:
    full_study_pass: true
    ema:
      tau_start: 0.996
      tau_end:   0.9999
      schedule: cosine
  loss:
    lambda_regress: 1.0
    lambda_nce:     0.0              # Stage-1 default
    tau_nce:        0.1
  dataset:
    target_slot_mode: A              # metadata-only
    k_anchor: 0
    keep_tokens: false
  clip_encoder:
    freeze: true
  study_encoder:
    cala:
      enabled: false
  logging:
    falsification_probe:
      enabled: true
      halt_cosine_threshold: 0.98
      halt_consecutive_steps: 5000
```

### 6.2 `stage1b_frozen_clip_matched_nce.yaml`

Same as 6.1 except `loss.lambda_nce: 0.01` (also a variant at 0.03).

### 6.3 `stage2_adapter_joint.yaml`

Key deltas:

```yaml
experiment:
  clip_encoder:
    freeze: false
    adapter:
      type: lora
      lora_rank: 8
  loss:
    lambda_regress: 1.0
    lambda_nce:    0.03
    lambda_anchor_clip: 0.1          # L_anchor_to_base_clip
    lambda_clip_vjepa: 1.0           # preserves clip-level V-JEPA
  target_encoder:
    ema:
      clip_encoder: true             # Stage-2+ EMA on clip encoder too
```

### 6.4 `stage3_target_anchor_partial_tokens.yaml`

```yaml
experiment:
  dataset:
    target_slot_mode: C
    k_anchor: 16
    keep_tokens: true
```

### 6.5 `stage4_cala_zero_init.yaml`

```yaml
experiment:
  study_encoder:
    cala:
      enabled: true
      gamma_init: 0.0
      gamma_warmup_steps: 5000
      bias_dim: 32
  target_encoder:
    cala:
      enabled: false       # teacher NEVER has CALA
```

### 6.6 `stage5_full_joint.yaml`, `ablation_*.yaml`

Stubbed with documented knobs per the ablation matrix (A9–A16 in architecture plan §17).

### 6.7 `baseline_vjepa_predavg.yaml`, `baseline_supervised_study_transformer.yaml`

Reuse existing probe configs (`configs/eval/vitl/*`) with K-matched same-manifest sampler override.

---

## 7. `tests/echomv_jepa/`

All tests use `pytest`; reuse fixtures from `tests/echoset_jepa/conftest.py` where possible.

| Test file | Invariants asserted | Dependencies |
|-----------|---------------------|--------------|
| `test_masking_no_target_leak.py` | no target visual latent in context; target quality/site off by default | PR-2 |
| `test_full_study_target_encoder_shapes.py` | `z_per_element.shape == (B, M, d_model)`; gather at target indices correct | PR-3 |
| `test_ema_update.py` | `teacher = tau*teacher + (1-tau)*student`; `tau=1.0` no-op; `requires_grad=False` on teacher | PR-3 |
| `test_permutation_invariance.py` | element shuffle → `h_study` invariant (`[STUDY]` token); `h_mask` equivariant | PR-3 |
| `test_target_anchor_masking.py` | I1–I5 (6 sub-tests, see §2.6) | PR-7 |
| `test_cala_zero_init.py` | γ=0 reproduces Stage-1 encoder hash on same input; anatomy-key dropout fires; teacher CALA config rejected | PR-8 |
| `test_matched_negatives.py` | same-study off-targets excluded; fallback ladder logged; λ_nce=0 → NCE is exactly 0 | PR-3 |
| `test_k_sampler_fairness.py` | all baselines receive identical K clip IDs per study from the same manifest with the same seed | PR-2 |

---

## 8. PR sequence with files changed

Reproduced from architecture plan for engineer reference.

```
PR-0 [this PR, docs only]
     docs/echomv_jepa_architecture_plan.md
     docs/echomv_jepa_vs_prior_variants.md
     docs/echomv_jepa_implementation_sketch.md

PR-1 [data / taxonomy / manifests]
     experiments/echomv_jepa/__init__.py
     experiments/echomv_jepa/build_multiview_manifest.py
     experiments/echomv_jepa/target_difficulty_audit.py  [scaffold]
     tests/echomv_jepa/__init__.py

PR-2 [dataset / masking / target slots]
     src/datasets/echomv_jepa_dataset.py
     tests/echomv_jepa/test_masking_no_target_leak.py
     tests/echomv_jepa/test_k_sampler_fairness.py
     configs/train/echomv_jepa/stage1_frozen_clip_full_study_ema.yaml  [dataset section only]

PR-3 [full-study EMA target encoder]
     src/models/echomv_jepa/__init__.py
     src/models/echomv_jepa/study_target_encoder.py
     src/models/echomv_jepa/losses.py
     src/models/echomv_jepa/ema.py
     tests/echomv_jepa/test_full_study_target_encoder_shapes.py
     tests/echomv_jepa/test_ema_update.py
     tests/echomv_jepa/test_permutation_invariance.py
     tests/echomv_jepa/test_matched_negatives.py

PR-4 [training loop + losses + falsification probe]
     app/echomv_jepa/__init__.py
     app/echomv_jepa/train.py
     configs/train/echomv_jepa/stage1_frozen_clip_full_study_ema.yaml  [complete]
     src/models/echomv_jepa/clip_backbone.py
     src/models/echomv_jepa/predictor.py

PR-5 [controls + diagnostics]
     configs/train/echomv_jepa/ablation_shuffled_study.yaml
     configs/train/echomv_jepa/ablation_metadata_only.yaml
     configs/train/echomv_jepa/ablation_target_meta_only.yaml
     configs/train/echomv_jepa/ablation_nearest_context.yaml
     configs/train/echomv_jepa/ablation_no_ema.yaml
     configs/train/echomv_jepa/ablation_element_target.yaml       # A12
     experiments/echomv_jepa/echometr3r_consistency.py
     app/echomv_jepa/eval_consistency.py

PR-6 [downstream K-matched eval]
     configs/train/echomv_jepa/baseline_vjepa_predavg.yaml           # B0
     configs/train/echomv_jepa/baseline_supervised_study_transformer.yaml  # B2
     evals/video_classification_frozen_multi/ [config additions]

PR-7 [target anchoring; gated on PR-5 passing §20.2]
     src/models/echomv_jepa/target_anchor.py
     tests/echomv_jepa/test_target_anchor_masking.py                  # 6 sub-tests
     configs/train/echomv_jepa/stage3_target_anchor_partial_tokens.yaml
     experiments/echomv_jepa/cache_tokens.py

PR-8 [CALA]
     src/models/echomv_jepa/correspondence_attention.py
     tests/echomv_jepa/test_cala_zero_init.py
     configs/train/echomv_jepa/stage4_cala_zero_init.yaml

PR-9 [adapter-joint clip training; gated on §20.3]
     src/models/echomv_jepa/clip_backbone.py  [adapter]
     src/models/echomv_jepa/losses.py         [L_anchor_to_base_clip]
     configs/train/echomv_jepa/stage2_adapter_joint.yaml
```
