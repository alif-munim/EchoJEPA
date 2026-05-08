# EchoMV-JEPA: Hierarchical Multi-View Study-Level JEPA for Echocardiography

**Status:** design document, PR-0
**Audience:** NeurIPS reviewers, project leads, implementers
**Cross-references:** `docs/echoset_jepa_plan.md` (frozen EchoSet-JEPA v1 plan; reframed here as Stage-0 control), `claude/neurips/README.md` (resubmission plan), `claude/neurips/controlled-objectives.md` (prior-variant results)

---

## 1. Executive summary

EchoMV-JEPA is a hierarchical, multi-view, study-level Joint Embedding Predictive Architecture for echocardiography. The prediction unit is the **clinical echo study**, not the isolated clip. Context is a masked/incomplete study; target is held-out study elements whose embeddings are selected **after** a full-unmasked study pass through an EMA target study encoder. This is the JEPA-faithful design — a study-level analogue of I-JEPA/V-JEPA target encoding — and it is the defining architectural addition over frozen EchoSet-JEPA v1, whose EMA acts only on a projector MLP and whose "full_study_teacher_target" branch is in fact an unfinished stub that falls back to pre-context element projection (`st.clip_in(tgt_elements)` at `app/echoset_jepa/train.py:200–202`).

The method is built to avoid the specific failure modes observed in prior work on this repo: single-view hallucination (MV2SV), pooled phase-axis specialization (V4 phase-relational), positional-blind token matching (TokenRel), and slot-disentanglement claims that collapsed in practice (factorized heads). It does not claim to solve TAPSE/local mechanics without an explicit token-transport branch, and it does not claim RVSP/AS/MR/hemodynamic recovery without Doppler/color/calibration. The MVP (Stage 1) freezes the clip encoder; clip-encoder updates are gated by Stage-2+.

**Three-sentence claim.** (i) The echo study is a variable, partially observed set of acquisitions under view/modality/phase/quality/calibration variation; its latent physiological state is what downstream tasks need. (ii) A trainable online study context encoder + EMA full-study target encoder with target embeddings selected after full-study encoding is the minimal JEPA-faithful design that respects this structure. (iii) This design is distinct from vanilla V-JEPA (clip-level), frozen EchoSet-JEPA v1 (pre-context linear target), MV2SV (single-view hallucination), V4 phase-relational (pooled phase axis), and TokenRel/MotionDelta (clip-pair token matching) in precise, architecturally-falsifiable ways enumerated in §4–§6.

---

## 2. Scientific claim

**C1.** The clinical echo study — not the clip — is the correct prediction unit for echocardiography foundation modeling.

**C2.** Target embeddings selected after an EMA full-unmasked-study encoder carry study-level physiological context that cannot be recovered from pre-context element projection (the EchoSet v1 element-target path). The falsification probe is a per-step log of `cosine(z_t^{EchoMV}, z_t^{v1})`; if this exceeds 0.98 for 5 k consecutive steps, C2 is falsified and the run is halted (§18).

**C3.** Co-evolving EMA is load-bearing at the study-encoder level, not just at the clip level (prior experiments established this at the clip level). Stage-0/1 ablation (A6, A12) adjudicates this.

**C4.** The method produces gains on *multiple* downstream axes, not just LVEF; gains that appear on LVEF but regress RVSP, TAPSE, or pediatric/external transfer do not count as success (the V4 failure mode, §16).

Claims that EchoMV-JEPA **does not** make, and explicit guardrails:
- Not a universal improvement over V-JEPA on all echo tasks.
- Not a solver of TAPSE or local wall mechanics without an explicit token/transport branch (Stage-5+, A13).
- Not a valid claimant of RVSP, AS, MR, or diastology from B-mode alone; those require Stage-6 multimodal extension with calibration tokens.
- Not a slot-disentanglement architecture; prior factorized attempts collapsed, and no disentanglement claim is part of this plan.
- Not evaluated by retrieval accuracy; retrieval is a diagnostic (§15), not a success criterion.

---

## 3. First-principles formulation

An echo study is a variable, partially observed set of acquisitions:

```
  x_i = obs(s | v_i, m_i, phi_i, q_i, c_i, a_i)
```

where

| Symbol | Meaning |
|--------|---------|
| `s`    | latent physiological state (the study-level quantity downstream tasks need) |
| `v_i`  | view family of acquisition `i` (A4C, A2C, A3C, A5C, PLAX, PSAX, RV_focused, subcostal, suprasternal, unknown) |
| `m_i`  | modality (`b_mode`, `color_doppler`, `pw_doppler`, `cw_doppler`, `m_mode`, `tdi`, `contrast`, `unknown`) |
| `phi_i`| phase bucket / cycle state (`systolic`, `diastolic`, `full_cycle`, `not_applicable`, `unknown`) |
| `q_i`  | quality (high/med/low/unknown) — **reliability**, *not* an element-key axis |
| `c_i`  | calibration (velocity scale, Nyquist limit, sweep speed, pixel spacing, frame rate, Doppler mode, measurement site) — required for Stage-6 hemodynamic claims |
| `a_i`  | operator/acquisition factors (vendor, site) — kept out of target-slot metadata by default |

**Element key** (the grouping key for study elements) is:

```
  element_key(i) = (view_family_i, modality_i, phase_bucket_i)
```

Quality is **deliberately excluded** from the element key. Quality is:
- a *context* reliability token (§7.4),
- an aggregation weight (optional) for clips → elements,
- a dedup tie-breaker (`experiments/echoset_jepa/dedup.py`),
- a stratification diagnostic (§15).

Quality is **never**:
- part of the default element key,
- supplied to target mask slots (default `include_target_quality=False`),
- part of a headline scientific claim.

Doppler, M-mode, and TDI are **modalities, not view families**. This keeps the key space factorizable and matches the existing `src/models/meta_embeddings.py` vocabularies (which already implement `VIEW_FAMILY_VOCAB` disjoint from `MODALITY_VOCAB`).

A measurement-site field (`none, TR, MV_inflow, LVOT, AV, MV, TV, septal_annulus, lateral_annulus, IVC, unknown`) is tracked in the manifest for Stage-6 but is **excluded from target-slot metadata** to prevent label leakage into the masked-prediction task.

---

## 4. Relation to vanilla V-JEPA

Vanilla V-JEPA (`app/vjepa/train.py`, `src/models/vision_transformer.py`, `src/models/predictor.py`) is the clip-level sub-component used as `f_theta`. EchoMV-JEPA is strictly a super-system: it embeds V-JEPA as the clip backbone and adds a study-level JEPA on top.

| Axis | Vanilla V-JEPA | EchoMV-JEPA |
|------|----------------|-------------|
| Prediction unit | single clip | clinical study (set of elements) |
| Context input | masked spatiotemporal tubelets of one clip | masked/incomplete study (elements, metadata, optional anchors) |
| Target input | full unmasked clip → target encoder | full unmasked **study** → EMA target **study** encoder |
| Target selection | target-encoder tokens at mask positions | EMA study-encoder outputs at target element slots, selected **after** full-study encoding |
| EMA scope | target clip encoder only | target clip encoder (Stage-2+) **and** target study encoder (Stage-1) |
| Study structure | none | variable K elements, permutation-invariant, metadata-aware |
| View/modality | none | first-class, in element key and as conditioning |
| Phase | none (Stage-0 phi-JEPA is an exception, not a study-level construct) | first-class, phase-bucket part of element key |
| Held-out view/modality prediction | not possible | core training signal (masking strategies §8) |
| Inference | single clip | K clips per study, set-aware probing |

V-JEPA's strength (confirmed in prior controlled experiments on this repo: V-JEPA beats BYOL/MAE/SALT at matched compute on LVEF and on robustness) is preserved. The clip-level V-JEPA loss (`L_clip_vjepa`) is retained whenever the clip encoder is trainable (Stage-2 adapter-joint, Stage-5 full joint), via `L_anchor_to_base_clip`, to prevent study-level pressure from destroying the clip representation that is already strong.

---

## 5. Relation to frozen EchoSet-JEPA v1

EchoSet-JEPA v1 (`app/echoset_jepa/train.py`, `src/models/study_transformer.py`, `src/models/study_projectors.py`) is the closest prior in this repo. Reading the code reveals three narrow but load-bearing gaps that EchoMV-JEPA closes.

**What v1 actually does (verified at `app/echoset_jepa/train.py:176–204`):**

```
if target_mode == "element_target":
    z_target_src = st.clip_in(batch["tgt_elements"])     # linear on raw element latent
    z_t = proj.teacher_forward(z_target_src).detach()    # EMA on projector ONLY
elif target_mode == "full_study_teacher_target":
    ...
    # "For clarity, use element_target path for MVP; Mode B is an
    #  ablation pathway to be finalized when its ablation YAML runs."
    z_t = proj.teacher_forward(st.clip_in(batch["tgt_elements"])).detach()
```

So v1's `full_study_teacher_target` is a stub: even in that branch, `z_t` is built by running `st.clip_in` (the student's pre-context linear) on raw target element vectors, then projecting with the EMA teacher MLP. The EMA is on the projector (`EMAProjectorPair` at `src/models/study_projectors.py:29`), not on the study transformer. There is no EMA copy of `StudyTransformer` anywhere.

**What EchoMV-JEPA adds:**

| Axis | EchoSet v1 | EchoMV-JEPA |
|------|------------|-------------|
| EMA scope | projector MLP (2-layer, d_model→d_hidden→d_proj) | projector MLP **and** full `StudyTransformer` (new `StudyTransformerEMA` wrapper) |
| Target source | pre-context linear `st.clip_in(tgt_elements)` → projector teacher | EMA study transformer applied to full unmasked study → select contextualized outputs at target slots → projector teacher |
| Target depends on other elements? | no (pure per-element linear) | yes (contextualized by full-study self-attention) |
| Stage-0 / control role | was the method | is the Stage-0 control (A16 is the apples-to-apples re-run) |
| Load-bearing claim | element target + projector EMA suffices | full-study contextualized target is the load-bearing study-level signal |

Accordingly, EchoMV-JEPA's PR sequence inherits the K-sampler, manifest builder, element grouping, meta embeddings, masking strategies, and dataset collate from `experiments/echoset_jepa/` and `src/datasets/echoset_jepa_*.py` unchanged, and adds only what closes the three gaps. See §19 for the file/module plan.

---

## 6. Relation to MV2SV, V4 phase-relational, TokenRel, MotionDelta, factorized slots

All of these live in `app/vjepa_multiview/` and in `configs/train/vitl16/*{mv2sv,phase,tokenrel,privview}*.yaml`. Prior controlled experiments produced specific negative results; EchoMV-JEPA is designed to avoid the same modes.

| Variant | Core mechanism | Observed result | EchoMV-JEPA design response |
|---------|----------------|-----------------|-----------------------------|
| **MV2SV** (`mv2sv-pilot-5to25-resume.yaml`) | Single-view student at inference; privileged target-view supervision; factorized slots (`factorized_head.py`) | Cross-view retrieval improved; downstream LVEF lost to paired intraview controls; slot collapse | EchoMV uses *observed* elements at inference — no single-view hallucination claim; target is a held-out observed element, not a hallucinated view; no slot-disentanglement claim |
| **V4 phase-relational** (`phase_relational_head.py`, `pretrain-multiview-phase-relational-hardneg-25of100-paper.yaml`) | Pooled (B, 1024)→InfoNCE with same-study wrong-phase hard negative + Fourier Δφ conditioning | LVEF R²=0.652 (best on repo); RVSP Pearson=0.458; TAPSE not addressed; pediatric/external generalization regressed | EchoMV distributes pressure across physiology (full study), view, modality, **and** phase (element key), avoiding pooled phase-axis specialization; matched NCE default `λ=0.0` in Stage 1 to prevent V4-style over-reliance on a pooled contrastive axis |
| **TokenRel** (`token_relational_head.py`, `pretrain-multiview-tokenrel-25of100.yaml`) | Per-token set InfoNCE across views, no spatial positional awareness; token subsample K=64 | Required pooled safety loss (λ=0.005) to preserve LVEF; cross-view token matching positional-blind; no RVSP/TAPSE reported | EchoMV defers token-level / transport objectives to Stage-4 CALA (anatomy/physiology-keyed cross-element attention); does not make token-level correspondence the MVP objective |
| **MotionDelta** (`token_relational_head.py::MotionDeltaHead`) | Same-view-only per-token Δφ→latent-delta SmoothL1 + InfoNCE | Never enabled in production; restricted to same-view (discards cross-view motion); did not solve TAPSE | EchoMV does not claim TAPSE in Stage-1–4; local metric motion is Stage-5+ with an explicit token-transport branch |
| **Factorized slots** (`factorized_head.py`, `view_predictor.py`, `phase_query_head.py`) | z_shared / z_phase / z_view decomposition with three independent MLPs | Slot collapse; z_phase pooled-from-whole-frame, discards motion; no downstream eval | EchoMV uses full-study target conditioning + metadata embeddings, **no** slot decomposition and **no** post-hoc disentanglement claim |
| **Privileged multiview** (`privview`) | Adds extra `target_clip` and optional fused pool (p_fused=0.0 in practice) | Sparse fused loss never activated; privileged target assumption breaks at inference | EchoMV's target is a *masked subset of the observed study*, not a privileged extra acquisition; inference uses the same observed study |

**Reuse from `app/vjepa_multiview/` for EchoMV-JEPA:** phase-matched sampler (`classifier/phase/sampler/phase_matched_sampler.py`) and `src/datasets/video_group_dataset.py` anchor-centering logic (opt-in, Stage-6); Fourier phase encoding pattern as a utility (not as a head). No head from `app/vjepa_multiview/` is reused.

---

## 7. Architecture

Eight components. Block diagram:

```
  study (K clips, metadata)
     │
     ▼
  ┌──────────────────────┐        ┌─────────────────────────────┐
  │ f_theta  clip encoder│        │ f_bar_theta clip enc  (EMA) │
  │  V-JEPA init         │        │  EMA of f_theta OR fixed    │
  │  frozen / adapter    │        │  copy (Stage-1 frozen)      │
  │  / fully joint       │        └─────────────────────────────┘
  └──────────────────────┘                       │
             │                                   │
    (per-clip tokens / pooled)       (per-clip tokens / pooled)
             │                                   │
  ┌──────────────────────┐        ┌─────────────────────────────┐
  │ Element builder      │        │ Element builder (teacher)   │
  │  group K clips -->   │        │  group K clips -->          │
  │  M elements by       │        │  M elements by              │
  │  (view, mod, phase)  │        │  (view, mod, phase)         │
  └──────────────────────┘        └─────────────────────────────┘
             │                                   │
   (context mask applied)          (no mask — full unmasked study)
             │                                   │
             ▼                                   ▼
  ┌──────────────────────┐        ┌─────────────────────────────┐
  │ F_psi online study   │        │ F_bar_psi EMA target        │
  │ context encoder      │        │ study encoder               │
  │  sees M_ctx + M_tgt  │        │  sees M (all elements)      │
  │  mask slots w/ meta  │        │  NO mask slots              │
  │  [STUDY] token       │        │  returns contextualized     │
  │  optional CALA       │        │  per-element hidden states  │
  └──────────────────────┘        └─────────────────────────────┘
             │                                   │
             ▼                                   ▼
  ┌──────────────────────┐        ┌─────────────────────────────┐
  │ Predictor g_phi at   │        │ Select outputs at target    │
  │ target mask slots    │        │ element indices; apply      │
  │  -> h_t, projected   │        │ EMA projector teacher_MLP   │
  │ through student MLP  │        │  -> z_t (stopgrad)          │
  └──────────────────────┘        └─────────────────────────────┘
             │                                   │
             └──────────────┬────────────────────┘
                            ▼
                   L_study_jepa = mean_t [1 - cos(LN(h_t), sg(LN(z_t)))]
                   + lambda_nce * L_nce   (default 0.0 in Stage 1)
                   + optional lambda_anchor_clip * L_clip_anchor (Stage-2+)
```

### 7.1 Clip encoder `f_theta`

- Initialized from an existing V-JEPA checkpoint (ViT-L/16 by default).
- Three training modes:
  - **A. frozen** (Stage 0–1 default) — matches `clip_encoder.freeze = True` in existing configs; reuses the cached `c_clip` infrastructure (`experiments/echoset_jepa/cache_cclip.py`).
  - **B. adapter-joint** (Stage 2) — LoRA/top-block training; keeps base weights frozen.
  - **C. fully joint** (Stage 5) — all clip parameters updated; anchored by `L_clip_vjepa + L_anchor_to_base_clip`.
- Produces **pooled** element-contribution vectors `c_clip ∈ R^{D_clip=1024}` by default. Optionally preserves **token-level** outputs `c_clip_tok ∈ R^{T × D_clip}` for CALA (Stage-4) and target anchoring (Stage-3); gated by `clip_encoder.keep_tokens: true`.
- B-mode only in Stages 1–5; modality-specific adapters/stems (color Doppler, CW/PW Doppler, M-mode, TDI) added in Stage-6.

### 7.2 EMA target clip encoder `f_bar_theta`

- Stage 1 (frozen `f_theta`): `f_bar_theta` is a fixed deep copy.
- Stage 2+ (trainable `f_theta`): `f_bar_theta` is an EMA copy updated via `torch._foreach_mul_/add_` with a `momentum_scheduler` identical to `app/vjepa/train.py:382–405`. This preserves V-JEPA's co-evolving target behavior at the clip level.

### 7.3 Study element builder

- Groups the K clips of a study into M elements keyed by `(view_family, modality, phase_bucket)`. Reuses `experiments/echoset_jepa/element_grouping.py` unchanged.
- Default aggregation: mean over per-clip pooled latents → element latent `e ∈ R^{D_clip}`.
- Optional quality-weighted aggregation: `weights = softmax(log q_i)`, gated by `elements.element_agg: quality_weighted`.
- Optional token-level path: keeps `T_e` tokens per element for CALA/anchor.

### 7.4 Online study context encoder `F_psi`

- Trainable permutation-invariant transformer. Reuses `src/models/study_transformer.StudyTransformer` (d_model=512, 4 layers, 8 heads, FFN mult=4) for MVP; documents a `d_model=768, 6L` option for Stage-2+ if capacity-matched controls (B2) require it.
- Sees context elements + target mask slots + `[STUDY]` token, with meta-token additives for `view_family`, `modality`, `phase_bucket`, `measurement_site` (context side), and context-side `quality_bucket`. See `src/models/meta_embeddings.py`.
- No acquisition-order positional embedding. Element order is permuted every step by the dataloader (already done in `EchoSetJEPADataset`).
- Accepts variable M elements with padding masks (`ctx_pad_mask`, `tgt_pad_mask`).
- Optional CALA inserted as zero-init residual (§7.7) only on the online path.

### 7.5 EMA full-study target encoder `F_bar_psi`

- **This is the core new module.** An EMA copy of `F_psi`'s `StudyTransformer`, wrapped in a new `StudyTransformerEMA` pair (paralleling `EMAProjectorPair` at `src/models/study_projectors.py:29`).
- Sees the **full unmasked study** (context elements ∪ target elements, with the target elements' true meta but no mask slots). Tensor shape contract: ingest `(B, M, D_clip)` + `(B, M, d_model)` meta, emit `(B, M, d_model)` contextualized per-element hidden states.
- Target selection is **after** encoding: pick hidden states at target element indices → project through EMA `teacher_MLP` → stopgrad → `z_t ∈ R^{B × M_tgt × D_proj}`.
- Does not include CALA adapters (teacher stays vanilla so correspondence bias cannot leak into the target).
- EMA schedule: `cosine_schedule(step, total, tau_start=0.996, tau_end=0.9999)` (reuses `src/models/study_projectors.cosine_schedule`).

### 7.6 Predictor `g_phi`

- Consumes online context outputs at target mask slot positions: `h_mask ∈ R^{B × M_tgt × d_model}`.
- Maps to target space via the **student** half of `EMAProjectorPair` for MVP (re-uses v1's projector to minimize new code). Stage-2+ may replace with a dedicated transformer predictor; for MVP, a lightweight Linear+GELU+Linear is sufficient.
- Optionally conditions on target metadata explicitly (already baked in through `tgt_meta_add` at the study transformer input).
- Output: `h_t ∈ R^{B × M_tgt × D_proj}`.

### 7.7 Correspondence-Aware Latent Attention (CALA), optional (Stage 4)

- Inspired by MVDiffusion. **Zero-initialized** via a per-head `gamma` scale (`x + gamma * CALA(x)`, `gamma_init=0.0`, `gamma_warmup_steps=5000`). This is the LayerScale/ReZero pattern, the only reliable "does not disrupt base model" construction.
- Inserted into `F_psi` only — **not** into `F_bar_psi`. This is critical: if the teacher also had CALA, the correspondence bias could leak into `z_t` and the model could match its own bias.
- Queries: study-encoder tokens. Keys/values: tokens from other elements in the same study, filtered by a learned correspondence prior keyed on `(view_family_q, view_family_k, anatomy_group_q, anatomy_group_k)`. Anatomy groups are a small learned table (`LV`, `RV`, `MV`, `AV`, `TV`, `IVC`, `unknown`) — **not** pixel correspondences, because no calibrated camera geometry exists for echo.
- Correspondence bias uses low-dimensional learned bias table (`bias_dim=32`), not a gated copy, so it cannot perform identity routing.
- Correspondence keys dropped with probability matching `meta_dropout` to prevent metadata shortcut.
- Halt criterion: attention entropy collapses within 500 steps → abort.
- Built on `src/models/attentive_pooler.CrossAttentionBlock`.

### 7.8 Target anchoring (Stage 3), optional

Three target-slot modes, compared head-to-head (see §8, §17 A7/A14):

- **A. metadata-only** (default, MVP): target slot = `mask_token + tgt_meta_add`. No visual latent.
- **B. low-resolution target anchor**: compress target element tokens → a small (k_anchor ≪ T_e) low-res summary, produced by a **distinct head** from the target projector (invariant I4 in §16). Added to the target slot.
- **C. partial target visible tokens**: a strict subset (k_anchor tokens per target element) of the target's own tokens is exposed to the predictor via the target slot. Highest-risk; gated by the five invariants I1–I5 and six unit tests in §16.

Rationale: pure cross-view hallucination from metadata alone is under-specified, which may make the prediction task trivially noisy. Anchoring may help. All three modes must respect the anti-leak rules (§16).

### 7.9 Metadata routing and anti-leak rules (summary)

Context side (permissive): view_family, modality, phase_bucket, measurement_site, quality_bucket — with `meta_dropout` per field.
Target side (restrictive): view_family, modality, phase_bucket by default. **Excluded by default:** quality (`include_target_quality=False`), measurement_site, vendor, site, acquisition order, report-derived metadata.

Anti-leak rules (enforced by unit tests, §19):
1. Target mask slot must not receive target visual latent (except under the explicit target-anchor modes B/C, with their own invariants).
2. Target quality off by default.
3. Acquisition order off by default.
4. Vendor/site off by default.
5. Report-derived metadata off by default.
6. Measurement-derived labels off by default.
7. Overlay/measurement text pixels masked before visual encoding where possible (inherits existing repo practice).

---

## 8. Target construction and masking

### 8.1 Masking strategies

Same taxonomy and implementation as `src/datasets/echoset_jepa_dataset.pick_mask_indices`, extended with modality/phase holdouts.

| Strategy | Description | Weight (default) |
|----------|-------------|------------------|
| `random_element` | uniform random target-element subset (size drawn from `[1, ceil(0.6 M)]`) | 0.35–0.40 |
| `whole_view_family` | hold out all elements of one view family (e.g., all apical) | 0.25 |
| `whole_modality` | hold out all elements of one modality (e.g., all color Doppler) | 0.20 |
| `apical_or_parasternal_holdout` | hold out either all apical or all parasternal | 0.10 |
| `color_or_spectral_holdout` | hold out color Doppler or spectral Doppler (gated by coverage) | 0.05–0.10 |
| `phase_holdout` (optional) | hold out all elements of one phase bucket | 0.05 (optional) |
| `same_phase_adjacent_beat_holdout` (optional, Stage-3+) | hold out same phase in adjacent beat (if annotations permit) | 0.00 (opt-in) |
| `ED_ES_holdout` (optional, Stage-3+) | hold out ED or ES if phase labels exist | 0.00 (opt-in) |

Fallback: if no valid stratified strategy exists for a study (insufficient elements), fall back to `random_element`. Invariants `min_context ≥ 1`, `min_target ≥ 1`, `max_target ≤ 0.6 M` from EchoSet v1 are preserved.

### 8.2 Target slot modes

Per §7.8: A (metadata-only, MVP default), B (low-res anchor), C (partial visible tokens). The MVP runs only A; B and C are Stage-3 ablations.

### 8.3 Sampling

- **Primary fairness unit: K_clips.** All baselines and the method receive the exact same K clips with the same seed (same K-sample manifest, `experiments/echoset_jepa/sample_K.py`).
- **Primary K=8** with view-stratified budget: ~4–5 B-mode slots for view diversity, 1–2 color Doppler if available, 1–2 spectral/M-mode/TDI if available; remaining filled by view/modality diversity and quality.
- **K sweep appendix**: `K ∈ {1, 2, 4, 8, 16}`; full-N only in an appendix.
- **Same-seed-across-controls** is enforced; `experiments/echomv_jepa/sample_K.py` inherits from EchoSet v1 unchanged.

---

## 9. Losses

### 9.1 Core study-JEPA regression

```
  L_study_jepa = mean_t [ 1 - cosine( LayerNorm(h_t), stopgrad(LayerNorm(z_t)) ) ]
```

- `h_t`: predictor output at target slot t, `[B_valid_targets, D_proj]`.
- `z_t`: EMA full-study target encoder output at target element index, projected, stopgrad, `[B_valid_targets, D_proj]`.
- `LayerNorm` applied to both before cosine (prevents scale collapse).
- Mean over valid (non-padded) target positions.

### 9.2 Matched InfoNCE (optional, `lambda_nce = 0.0` in Stage 1)

Reuses EchoSet v1's `_prioritized_neg_pool` (context-aware 4-tier): same `(view, modality, phase)` negatives from other studies preferred; fallback ladder to same `(view, modality)` → same `modality` → batch. Same-study off-targets always excluded.

Differences from V4 phase-relational:
- Per-target stratified (not pooled).
- Same-study off-targets always excluded.
- Fallback ladder capped: if >30 % of rows fall to `pri_3`/`pri_4`, NCE contribution is down-weighted and logged.
- **Default `λ_nce = 0.0` in Stage 1.** Only enabled in Stage-1b after cosine path is stable (`var_t > 0.3` for 10 k steps). Sweep `{0.0, 0.01, 0.03}`; do not go to 0.05.

### 9.3 Stage-dependent `L_total`

**Stage 1 (frozen clip encoder):**

```
  L_total = L_study_jepa + lambda_nce * L_nce    (default lambda_nce = 0.0)
```

**Stage 2 (adapter-joint clip encoder):**

```
  L_total = L_clip_vjepa
          + lambda_study * L_study_jepa
          + lambda_nce   * L_nce
          + lambda_anchor_clip * L_anchor_to_base_clip
```

where `L_anchor_to_base_clip = (1 - cos(c_clip_adapter, stopgrad(c_clip_base)))` applied to a held-out clip subset, preventing adapter drift from destroying the strong V-JEPA representation. Prior experiments: specialized losses that shifted the clip encoder hurt generality; the anchor is a guardrail.

**Stage 5 (full joint):**

```
  L_total = L_clip_vjepa + lambda_study * L_study_jepa + lambda_nce * L_nce
          + lambda_sigreg * L_sigreg_diagnostic    (optional, monitoring only)
```

`L_sigreg` (VICReg-style variance/covariance) is a **diagnostic / auxiliary**, not a replacement for EMA. Prior experiments confirmed EMA is load-bearing; SIGReg is only for Stage-5 collapse monitoring.

---

## 10. Training stages

| Stage | Name | Clip encoder | Target path | CALA | Anchor | Primary loss | Compute | Go/no-go |
|-------|------|--------------|-------------|------|--------|--------------|---------|----------|
| 0 | Frozen EchoSet control | frozen | element-target via projector EMA (v1 as-is) | — | — | v1 loss | 1× | reproduce v1 R² within ±0.01 (§20) |
| 1 | **True study JEPA (MVP)** | frozen | full-study EMA target encoder, select after encoding | — | metadata-only (A) | `L_study_jepa`, `λ_nce=0` | 1.2–1.5× Stage-0 | contextualization diagnostics pass + downstream gates (§20) |
| 1b | Stage-1 + tiny matched NCE | frozen | same | — | A | `+ λ_nce ∈ {0.005, 0.01}` (tiny by design) | +0 | NCE contribution additive; no regression on primary |
| **1m** | **Stage-1 + per-modality projector** | frozen | same; modality-routed student+teacher projector pair | — | A | `L_study_jepa`, `λ_nce=0` | +0 | beats Stage-1 on color-Doppler / spectral targets without regressing B-mode |
| 2 | Adapter-joint | LoRA / top blocks | same | — | A | `L_clip_vjepa + L_study_jepa + L_anchor` | 3–5× Stage-0 | gated on Stage-1 pass; EchoNet-Dyn probe not >2 % absolute regression |
| 3 | Target anchoring | frozen / adapter | same | — | B (low-res) or C (partial tokens) | same | +0 | gated on Stage-1 pass; I1–I5 invariants green |
| 4 | CALA | frozen / adapter | same (teacher vanilla, no CALA) | zero-init, warmup | A | same | +10 % | gated on Stage-1 pass; γ=0 reproduces Stage-1 |
| 5 | Full joint | fully joint | same, with clip EMA co-evolving | optional | optional | `L_clip_vjepa + L_study_jepa + L_anchor + L_sigreg` | 5–10× Stage-0 | only if Stages 1–2 show clear signal; high-risk |
| 6 | Multimodal | fully joint + modality adapters | same + calibration tokens | optional | optional | same + modality heads | TBD | only then eligible to claim RVSP/AS/MR/diastology |

### 10.1 Initial breadth set: Stage 1, 1b, 1m in parallel

These three are the first experimental wave and share the frozen clip encoder. They are run with the same manifest/K/seed so differences attribute cleanly to the head/loss knob. Stages 2, 3, 4 are gated follow-ups and are not part of the breadth set.

**Stage 1m rationale.** In EchoSet v1 and in the Stage-1 MVP, the projector teacher is modality-agnostic: a single 2-layer MLP projects every element (B-mode, color, spectral, M-mode) into the same target space. This places B-mode and Doppler in the same metric, and the cosine regression pressure on B-mode-dense studies (the majority) can dominate projection directions that are informative for Doppler. Stage-1m introduces a **per-modality student+teacher projector pair** indexed by target-slot modality id. Concretely: `proj_by_modality[m] = EMAProjectorPair(...)`, and for each target row the modality id selects the pair. This is a small add (one extra embedding-like table of projectors, no new loss, no new masking); if it helps, it helps by reducing modality-metric interference. If it does not help — or if it hurts — the per-modality metric is the wrong hypothesis and we fall back to Stage-1. Stage-1m inherits all of Stage-1's gates and adds one modality-stratified gate (§20.2).

Stages 1 → 2 and 2 → 3 are gated by §20 numeric thresholds.

---

## 11. Implementation sketch

Forward graph pseudocode for a single train step (Stage 1):

```python
# Inputs from EchoMVJepaDataset (extends EchoSetJEPADataset):
#   batch = {
#     "ctx_elements":   (B, M_ctx, D_clip=1024)
#     "tgt_elements":   (B, M_tgt, D_clip)      # real element latents (for teacher only)
#     "ctx_meta_*":     (B, M_ctx)              # view/modality/phase/measurement_site/quality
#     "tgt_meta_*":     (B, M_tgt)              # view/modality/phase only by default
#     "ctx_pad_mask":   (B, M_ctx) bool
#     "tgt_pad_mask":   (B, M_tgt) bool
#     "full_pad_mask":  (B, M)     bool         # for teacher full-study pass
#     "study_id_int":   (B,)
#   }

# --- ONLINE STUDENT PATH (masked study) ---
ctx_meta_add = meta.encode_context(batch["ctx_meta_view"], batch["ctx_meta_modality"],
                                   batch["ctx_meta_phase"], batch["ctx_meta_quality"])
tgt_meta_add = meta.encode_target_slot(batch["tgt_meta_view"], batch["tgt_meta_modality"],
                                       phase_ids=batch["tgt_meta_phase"],
                                       include_phase=True, include_quality=False)
h_study, h_mask = st(ctx_elements=batch["ctx_elements"],
                     ctx_meta_add=ctx_meta_add,
                     ctx_pad_mask=batch["ctx_pad_mask"],
                     tgt_meta_add=tgt_meta_add,
                     tgt_pad_mask=batch["tgt_pad_mask"])     # (B, d), (B, M_tgt, d)

h_t = proj.student_forward(h_mask)                            # (B, M_tgt, D_proj)

# --- EMA TEACHER PATH (full unmasked study) ---
with torch.no_grad():
    full_elements, full_meta_add, full_pad = _build_full_study(batch, meta)  # (B, M, D_clip), ...
    z_per_element = teacher_st(full_elements, full_meta_add, full_pad)       # (B, M, d_model)
    z_at_targets  = _gather_target_slots(z_per_element, batch["target_idx_in_full"])  # (B, M_tgt, d_model)
    z_t = proj.teacher_forward(z_at_targets).detach()                        # (B, M_tgt, D_proj)

# --- LOSS ---
valid = ~batch["tgt_pad_mask"]
h_flat = F.layer_norm(h_t[valid], h_t.shape[-1:])
z_flat = F.layer_norm(z_t[valid], z_t.shape[-1:])
loss_reg = (1.0 - (h_flat * z_flat).sum(-1)).mean()       # cosine regression, no extra LN

if lambda_nce > 0:
    loss_nce = _nce_loss(h_flat, z_flat, _prioritized_neg_pool(...), tau=tau_nce)
else:
    loss_nce = torch.zeros((), device=h_flat.device)

loss = loss_reg + lambda_nce * loss_nce

# --- FALSIFICATION PROBE (Stage-1 MVP only, §18) ---
with torch.no_grad():
    z_v1 = proj.teacher_forward(st.clip_in(batch["tgt_elements"])).detach()  # v1's z
    falsify = F.cosine_similarity(
        F.layer_norm(z_t[valid], z_t.shape[-1:]),
        F.layer_norm(z_v1[valid], z_v1.shape[-1:]), dim=-1).mean().item()
    # log falsify; halt if > 0.98 for 5000 consecutive steps.

# --- EMA UPDATES ---
loss.backward(); optimizer.step(); optimizer.zero_grad()
tau = cosine_schedule(step, total, tau_start, tau_end)
proj.update_teacher(tau)
teacher_st.update_teacher(tau)                            # NEW: StudyTransformerEMA.update_teacher
```

Checkpoint save/load contract extends v1 with two additional keys: `teacher_st_state_dict`, `teacher_st_ema_step`.

Compute cost per step:
- Stage 1 teacher full-study pass ≈ 0.3–0.5× student pass (no mask slots, M ≤ M_ctx + M_tgt). Expected total ≈ 1.2–1.5× v1.
- Stage 2 adds clip-encoder adapter forward/backward. Expected 3–5× v1.
- Stage 5 adds full clip-encoder forward/backward twice (student + teacher). Expected 5–10× v1.

---

## 12. Data pipeline requirements

- **Reuse without modification** (from `experiments/echoset_jepa/`): `build_manifest.py`, `sample_K.py`, `element_grouping.py`, `taxonomy.py`, `phase_bucket.py`, `quality_proxy.py`, `dedup.py`, `splits.py`, `cache_cclip.py`, `view_modality_coverage_audit.py`.
- **New** (Stage-1, PR-1): `experiments/echomv_jepa/build_multiview_manifest.py`. Extends `build_manifest.py` with calibration fields required for Stage 6: `velocity_scale`, `nyquist_limit_cm_per_s`, `sweep_speed_mm_per_s`, `pixel_spacing_cm_per_px`, `frame_rate_hz`, `doppler_mode`, `measurement_site`. Fields are **populated in Stage-1 but not consumed** by the training loop until Stage 6.
- **New** (Stage-3+): `experiments/echomv_jepa/cache_tokens.py`. Produces per-clip token cache (not just pooled `c_clip`) for CALA and target anchoring. Optional; gated by `clip_encoder.keep_tokens: true`.
- **New** (PR-5): `experiments/echomv_jepa/target_difficulty_audit.py`. Extends `experiments/echoset_jepa/target_difficulty_audit.py` to score target difficulty under full-study context.
- **New** (PR-5): `experiments/echomv_jepa/echometr3r_consistency.py`. Implements §15 suite.

---

## 13. Controls and baselines

All controls use the **same K, same seed, same manifest, same optimizer, same probe type, same splits**.

| # | Baseline | What it tests | Config |
|---|----------|---------------|--------|
| B0 | Vanilla V-JEPA + prediction averaging | study-level gain from EchoMV vs clip-avg | `configs/train/echomv_jepa/baseline_vjepa_predavg.yaml` |
| B1 | V-JEPA + supervised late-fusion attention probe | supervised attention pool at probe time suffices? | `configs/eval/.../attn_latefusion.yaml` |
| B2 | V-JEPA + capacity-matched supervised study transformer | supervised study transformer (no pretraining) matches EchoMV? | `configs/train/echomv_jepa/baseline_supervised_study_transformer.yaml` |
| C  | **Frozen EchoSet-JEPA v1 (sampler-matched)** | isolates EMA-full-study-encoder contribution | reuse `configs/train/echoset_jepa/echoset_jepa_v1_K8.yaml` **with same manifest** |
| D  | Shuffled-study EchoMV-JEPA | does the model learn study physiology or marginal view/modality stats? | `configs/train/echomv_jepa/ablation_shuffled_study.yaml` |
| E  | Metadata-only | no visual content; tests metadata shortcut | `configs/train/echomv_jepa/ablation_metadata_only.yaml` |
| F  | Target-meta-only prediction | tests whether target is under-specified | `configs/train/echomv_jepa/ablation_target_meta_only.yaml` |
| G  | Nearest-context-element baseline | tests duplicate-element / easy-masking shortcut | `configs/train/echomv_jepa/ablation_nearest_context.yaml` |
| H  | No-EMA / frozen target study encoder | tests whether study-level EMA co-evolution matters | `configs/train/echomv_jepa/ablation_no_ema.yaml` (**LANDED** — pure YAML, `tau_start=tau_end=1.0`) |
| I  | No-CALA / CALA comparison (Stage-4) | plain set transformer vs correspondence-aware | `configs/train/echomv_jepa/stage4_cala_zero_init.yaml` vs Stage-1 |
| J  | Metadata dropout ablations | `no_meta` / `v_m_only` / `v_m_p` / `full` with target-quality off by default | `configs/train/echomv_jepa/ablation_metadata_*.yaml` |

### 13.1 PR-5 work plan — what each ablation needs

Only H (no-EMA) is pure YAML. The rest require small code hooks in `training_step_echomv` or the dataset. Enumerated here so PR-5 is scoped precisely:

- **H no-EMA (LANDED):** YAML-only. `tau_start=tau_end=1.0`.
- **A12 element-target:** new `target_mode: "element_target"` knob in `training_step_echomv`. When set, skip the teacher full-study pass and compute `z_t = proj.teacher_forward(st.clip_in(tgt_elements))` — identical to v1. Keeps the teacher EMA updating (so this isolates "selection after encoding" vs "EMA at study level"). ~20 lines in `training_step_echomv` + YAML.
- **D shuffled-study:** dataset-level. Shuffle which elements go into which study across the batch: within a minibatch, reshuffle `(ctx_elements, tgt_elements, meta_*)` across the B dimension while preserving the (view, modality, phase) marginal distribution. New `src/datasets/echomv_jepa_dataset.py::shuffled_study_collate` option. ~40 lines.
- **E metadata-only:** zero out the visual content at the dataset boundary (`ctx_elements`, `tgt_elements` → zeros). Keep meta intact. Pure YAML if we add a `dataset.zero_visual: true` switch in the collate. ~10 lines in the collate.
- **F target-meta-only:** new `target_mode: "target_meta_only"` in `training_step_echomv` — `z_t = proj.teacher_forward(tgt_meta_add)` only. Tests whether the target prediction is under-specified (if yes, model can solve the task from metadata alone). ~5 lines.
- **G nearest-context:** baseline is not a training variant — it's an eval computing `z_t ← z_at_nearest_context_element(by cosine)` and reporting cosine(h_t, that) vs Stage-1. Lives in `experiments/echomv_jepa/` as a script, not a training config. ~60 lines.
- **I CALA:** needs PR-8 code.
- **J metadata dropout sweep:** pure YAML — vary `meta_dropout.view/modality/phase/quality`.

Estimated PR-5 surface: ~150 lines of new code + 6 YAML files + `experiments/echomv_jepa/nearest_context_baseline.py`.

---

## 14. Evaluation protocol

### 14.1 Main tasks (all Stages)

- Study-level LVEF
- LV dimensions / chamber size / LVH / HCM (where available)
- RV function (where available; not RVSP claim without Stage-6)
- Pediatric / external LVEF
- Held-out-site tasks
- HF / mortality (where available)

### 14.2 Proxy B-mode-only tasks (Stages 1–5; reported but guarded)

- RVSP from B-mode — reported as *proxy*, with the explicit do-not-ship cliff (§20).
- MR from B-mode — proxy only.
- AS from B-mode — proxy only.

### 14.3 Full multimodal tasks (Stage-6 only)

- RVSP with TR spectral Doppler (CW).
- AS with CW Doppler + LVOT.
- MR with color Doppler.
- Diastology with mitral inflow + TDI.

### 14.4 Low-label grid

`{1 %, 5 %, 10 %, 25 %, 100 %}` of labels per task. K-matched across all baselines.

### 14.5 Robustness

- K sweep: `{1, 2, 4, 8, 16}`.
- View dropout: drop 1, 2, 3 view families at eval.
- Modality dropout: drop color / drop spectral.
- Held-out site.
- Pediatric transfer.
- Vendor / site ablation.
- Predicted-view-label vs oracle-view-label (tests sensitivity to view classifier errors).

---

## 15. Diagnostics and Echo-MEt3R consistency suite

Inspired by MEt3R. **Feature-space** consistency, not pixel reconstruction.

### 15.1 Echo-MEt3R physiology consistency

1. Same-study A4C vs A2C vs PLAX latent consistency for LV geometry.
2. RV consistency across A4C vs RV-focused vs subcostal.
3. B-mode vs color vs spectral consistency for MR/AS/TR.
4. Phase consistency across adjacent beats.
5. Wrong-phase inconsistency (should be LOW).
6. Shuffled-study inconsistency (should be LOW).
7. View-dropout consistency degradation.
8. Target prediction residual stratified by view/modality relation.
9. Consistency independent of image quality tertile (stratified audit).

### 15.1a Teacher-contextualization diagnostics (per-step, required in Stage-1+)

Three cosine diagnostics logged every step. All are cheap; the last is computed every `diag_peer_drop_every_n_steps` (default 50) for speed.

1. **Falsification probe — `z_cosine_vs_v1`:** cosine between the Stage-1 full-study teacher target `z_t` and a second target `z_t^{v1}` computed the v1 way (`proj_teacher(st.clip_in(tgt_elements))`). **Halt rule:** if the per-step mean exceeds 0.98 for 5 000 consecutive steps, the EMA full-study teacher has collapsed to v1's pre-context projection and we halt. **Pre-downstream gate:** mean over the final 2 000 steps must be **≤ 0.95** for Stage-1 to pass its pre-downstream phase.
2. **Positive contextualization probe — `z_cosine_vs_isolated`:** cosine between `z_t` (full-study teacher, all M elements) and `z_t^{iso}` (same teacher, each target element encoded alone with its meta — no cross-element self-attention). Measures how much the teacher actually uses same-study context. **Pre-downstream gate:** mean over the final 2 000 steps must be **≤ 0.90**. If it is > 0.95, the teacher is effectively isolated-per-element; the "contextualize after full-study encoding" claim is falsified before downstream eval.
3. **Peer-drop sensitivity — `z_cosine_vs_peer_drop`:** for each target row, drop a single random context peer element and re-run the teacher; compute cosine between full-study `z_t` and `z_t^{peer_drop}`. A healthy contextualizing teacher should have this < 0.98 (target is sensitive to which peers are present); a teacher that ignores context should have this ≈ 1.0. **Diagnostic only**, not a gate; stratified by dropped peer's (view, modality).

Also monitored but not gated: `z_cosine_vs_v1` stratified by target modality (Doppler vs B-mode), by view family, and by M (number of study elements).

### 15.2 Standard JEPA health metrics

- Attention entropy (per layer, per head).
- Drop-one-view attribution.
- Performance vs number of unique views in the study.
- Performance vs modality availability.
- Target difficulty audit (`experiments/echomv_jepa/target_difficulty_audit.py`).
- Metadata shortcut tests (Stage-1 ablation J).
- Representation variance `var_t` (floor 0.3; inherits EchoSet v1's collapse monitor).
- Representation off-diagonal covariance `cov_off` (new floor 0.1).
- Collapse monitor (halt if either triggers 500 steps).
- Nearest-context cosine similarity (should not dominate the masked-prediction task).
- Same-study vs other-study target cosine separation.
- **Falsification probe `cosine(z_EchoMV, z_v1)`** (§18).

---

## 16. Risks and failure modes

### 16.1 Representation collapse

- Existing var_t floor (0.3) and new cov_off floor (0.1), each triggering halt after 500 steps.
- Independent diagnostic: nearest-context cosine should decrease over training, not increase.

### 16.2 Target leakage (five vectors)

1. **Target latent leak** into context stream — forbidden by construction; test `test_full_study_target_encoder_shapes.py` verifies target elements are never in `ctx_elements` tensor.
2. **Target quality leak** — `include_target_quality=False` by default; test `test_target_metadata_no_quality_leak.py` (inherits from EchoSet v1).
3. **Target measurement-site leak** — measurement_site excluded from target-slot meta by default; new test.
4. **Target overlay / measurement text leak** — overlays masked before visual encoding (existing repo practice).
5. **Target anchor leak (Stage-3)** — five invariants I1–I5, six unit tests (§7.8; `tests/echomv_jepa/test_target_anchor_masking.py`).

Invariants for Stage-3 anchoring:
- **I1** Partition: `ctx ∩ tgt == ∅` and `ctx ∪ tgt == all_indices`. Anchor tokens are a strict subset of target-slot tokens, never in context.
- **I2** Budget: per target slot, at most `k_anchor` visible tokens to online predictor; budget logged per step.
- **I3** Teacher invariance: `F_bar_psi` input is unchanged by the anchor flag (hash-equal before/after).
- **I4** Head distinctness: low-res anchor produced by a head with distinct parameters from `z_t` projector (`id(low_res_head) != id(target_projector)`).
- **I5** Zero-anchor equivalence: `k_anchor=0` reproduces Stage-1 loss numerically (tolerance 1e-5).

### 16.3 Matched NCE dominated by view/modality identity

- Per-target stratified pool (same `(v, m, phi)`) → hard negatives.
- Default `λ_nce = 0.0` in Stage 1.
- Fallback ladder > 30 % → NCE down-weighted and logged.

### 16.4 EMA co-evolution failure at study level

- Stage 1 MVP falsification probe `cosine(z_EchoMV, z_v1)` catches this: if the EMA study transformer's output at target slots stays close to pre-context projection (v1's `z`), the method collapses to v1.

### 16.5 CALA destabilization

- `gamma=0` init + 5 k-step warmup.
- Attention entropy halt.
- Teacher does not have CALA, so correspondence bias cannot leak into `z_t`.

### 16.6 Adapter-joint clip-encoder regression (Stage-2)

- `L_anchor_to_base_clip` constrains drift.
- EchoNet-Dynamic linear probe regression >2 % absolute → halt and roll back.

### 16.7 Overclaim on TAPSE / RVSP / AS / MR

- B-mode-only RVSP Pearson r > 0.55 on external val → mandatory audit before any external claim (prior work plateaus at ~0.46).
- No TAPSE claim in Stages 1–4.
- No multimodal Doppler / hemodynamic claims until Stage-6 with calibration tokens.

---

## 17. Ablation matrix

| # | Variant | Hypothesis | Expected benefit | Risk | Compute | Controls | Go/no-go gate |
|---|---------|-----------|------------------|------|---------|----------|--------------|
| A1 | Stage-1 MVP (full-study EMA target, λ_nce=0) | Contextualized target > pre-context projection | LVEF R² +0.015, RVSP r +0.02 vs Stage-0 | target ≈ v1 (falsification probe) | 1.2–1.5× | C, H | §20 Stage 1→2 gate |
| A2 | Stage-1 + λ_nce=0.01 | Small NCE reinforces hard-negative discrimination | marginal LVEF/RVSP | dominates cosine, V4 failure | +0 | C | no regression on primary |
| A3 | Stage-1 + λ_nce=0.03 | As A2 with more pressure | marginal | as A2 | +0 | C | no regression |
| A4 | Stage-3 target anchor B (low-res) | Anchor reduces target under-specification | smoother training; +LVEF | anchor leak (head distinctness) | +0 | I1–I5 | invariants pass; not worse than A1 |
| A5 | Stage-3 target anchor C (partial visible tokens) | As A4, stronger signal | as A4 stronger | leak risk higher | +0 | I1–I5 | invariants pass strictly |
| A6 | Stage-4 CALA (zero-init) | Physiology/anatomy-keyed cross-element attention helps multi-view tasks | RV/pediatric/held-out site | attention collapse | +10 % | I (no-CALA) | γ=0 reproduces A1 |
| A7 | Stage-2 adapter-joint | Clip-encoder adapters help | RVSP/LVEF +, pediatric + | adapter drift on EchoNet-Dynamic | 3–5× | anchor loss | EchoNet-Dyn probe not >2 % absolute regression |
| A8 | No-EMA / frozen target study encoder | EMA is load-bearing at study level | A1 > A8 | A1 ≈ A8 → EMA not needed | 1× | A1 | study-level EMA matters |
| A9 | Shuffled-study | Physiology vs marginal view/modality | A1 ≫ A9 | A9 ≈ A1 → model learns only marginals | 1× | A1 | A9 strictly worse |
| A10 | Metadata-only (no visual) | Metadata-shortcut floor | A9 < A10 ≪ A1 | A10 ≈ A1 → visual content unused | 0.1× | A1 | A10 ≪ A1 |
| A11 | Full joint (Stage-5) | Study-level pressure can improve clip rep | LVEF + RVSP + TAPSE | clip-rep destruction | 5–10× | A7 | only if A1/A7 show signal |
| **A12** | **EMA full-study target encoder BUT with element-target loss (no selection after encoding)** | **Isolates the "selection after encoding" contribution** | A1 > A12 | A12 ≈ A1 → core claim fails | 1.2× | A1 | A1 strictly beats A12 |
| **A13** | **CALA with γ unfrozen from step 0 vs γ warmup** | Zero-init warmup matters | γ-warmup > γ-step-0 | γ-step-0 destabilizes | +10 % | A6 | warmup strictly better |
| **A14** | **Anchor with k_anchor>0 BUT teacher also sees anchor** | Teacher invariance (I3) matters | leakage makes loss trivially low | *expected failure* | +0 | A4/A5 | **expected to fail** → confirms I3 |
| **A15** | **Matched NCE with same-view negatives only, no fallback ladder** | Stratification is the load-bearing piece | no-ladder ≥ ladder | ladder helps | +0 | A3 | stratification is what matters, not the ladder |
| **A16** | **Sampler-matched EchoSet v1 re-run (same K, manifest, optimizer, seed)** | Apples-to-apples v1 baseline | A1 > A16 | A1 ≈ A16 → v1 sampler mismatch explained gains | 1× | — | A1 beats A16 |

---

## 18. Minimal first experiment

**Stage 1 MVP — true study JEPA with frozen clip encoder.**

- Clip encoder: frozen V-JEPA (cached `c_clip`); same manifest as `echoset_jepa_v1_K8.yaml`.
- `F_psi`: identical architecture to `src/models/study_transformer.StudyTransformer` (d_model=512, 4 L, 8 H).
- `F_bar_psi`: **new** `StudyTransformerEMA` wrapper (`tau_start=0.996, tau_end=0.9999`, cosine schedule).
- Target: select `F_bar_psi` hidden states at target element indices after full-study pass → EMA projector teacher → stopgrad.
- K = 8 view-stratified (same as EchoSet v1).
- Element key: `(view_family, modality, phase_bucket)`.
- Masking mixture: default (§8.1).
- Meta dropout: identical to v1 (context: view 0.15 / modality 0.10 / phase 0.30 / quality 0.30; target: view/modality/phase only).
- Target slot mode: metadata-only (A).
- No CALA. No partial target tokens. No adapter-joint.
- Loss: `L_study_jepa` only; **`λ_nce = 0`**.
- **Falsification probe**: per step log `cosine(z_t^{EchoMV}, z_t^{v1})`. Halt if > 0.98 for 5 k consecutive steps.
- Budget: 5 k warmup + 20 k main + 2 k cooldown, same K, same manifest, same GPUs as EchoSet v1 short run.
- Output: logs + checkpoint + downstream linear probe on EchoNet-Dynamic LVEF (diagnostic, not headline).

### 18.1 Success criteria

- Stage-0 (EchoSet v1) re-run reproduces its reported LVEF R² within ±0.01.
- EchoMV-JEPA (Stage-1 MVP) beats Stage-0 on LVEF R² by **≥ +0.015** AND RVSP Pearson r by **≥ +0.02** AND does not regress TAPSE MAE by more than 0.2 mm, on a held-out pre-registered val split at sampler-matched + K-matched compute.
- Shuffled-study (A9) and metadata-only (A10) are strictly worse.
- No pediatric / external regression.
- No representation collapse (`var_t > 0.3`, `cov_off < 0.1` for last 5 k steps).
- Falsification probe `cosine(z_EchoMV, z_v1)` is **not** > 0.98 for 5 k consecutive steps.

### 18.2 Failure criteria (kill switches)

- Only beats B0 (prediction averaging), not B1/B2.
- Gains explained by metadata-only (A10 ≈ A1).
- Gains vanish at same K (A16 ≈ A1).
- Gains vanish under patient/site split.
- Retrieval improves but downstream does not.
- Only LVEF improves and multi-view / RV tasks do not.
- Target prediction solved by nearest-context (G) or target-metadata-only (F).
- Pediatric or external regression.
- Falsification probe: `cosine(z_EchoMV, z_v1) > 0.98` for 5 k consecutive steps.

---

## 19. File / module-level implementation plan

New files (forward references; **not yet written**):

```
docs/
  echomv_jepa_architecture_plan.md          [this file, PR-0]
  echomv_jepa_vs_prior_variants.md          [PR-0]
  echomv_jepa_implementation_sketch.md      [PR-0]

src/models/echomv_jepa/
  __init__.py                # LANDED
  study_target_encoder.py    # LANDED — StudyTransformerEMA wrapper (EMA copy + forward_contextualized + forward_isolated)
  modality_projector.py      # LANDED — ModalityProjectorPair, indexed by modality id (Stage-1m)
  losses.py                  # LANDED — L_study_jepa, prioritized_neg_pool, matched NCE (identical math to v1)
  ema.py                     # LANDED — generic EMA update helper (_foreach_mul_/add_)
  # Deferred follow-ups (PR-7 / PR-8 / PR-9):
  target_anchor.py           # target slot A/B/C modes (Stage-3)
  correspondence_attention.py # CALA, zero-init via per-head gamma (Stage-4)
  clip_backbone.py           # thin wrapper around V-JEPA ViT-L/16 with keep_tokens flag (Stage-2+)
  predictor.py               # lightweight predictor (Linear+GELU+Linear) over h_mask (Stage-2+)
  modality_adapters.py       # modality-specific clip stems for color/CW/PW/M-mode/TDI (Stage-6 — distinct from modality_projector.py)

# Note: instead of creating src/models/echomv_jepa/study_encoder.py, we added
# a pure additive method StudyTransformer.forward_contextualized(elements,
# meta_add, pad_mask) -> (B, M, d_model) to src/models/study_transformer.py.
# Existing forward() behavior is unchanged; all 109 v1 tests still pass.

src/datasets/
  echomv_jepa_dataset.py     # extends EchoSetJEPADataset with full_study target pass + token retention flag + anchor modes

experiments/echomv_jepa/
  __init__.py
  build_multiview_manifest.py  # extends experiments/echoset_jepa/build_manifest.py with calibration fields
  cache_tokens.py              # optional per-clip token cache (Stage-3+)
  target_difficulty_audit.py
  echometr3r_consistency.py

app/echomv_jepa/
  __init__.py
  train.py                  # Stage-1 entry point; extends app/echoset_jepa/train.py with teacher_st forward + EMA update
  eval_consistency.py       # runs §15 diagnostics

configs/train/echomv_jepa/
  stage1_frozen_clip_full_study_ema.yaml            # LANDED — MVP, λ_nce=0.0, num_modalities=1
  stage1b_frozen_clip_tiny_nce.yaml                 # LANDED — λ_nce=0.005 (tiny by design; sweep 0.01 separately)
  stage1m_frozen_clip_modality_projector.yaml       # LANDED — Stage-1m, num_modalities=len(MODALITY_VOCAB)=8
  # Deferred follow-ups:
  stage2_adapter_joint.yaml                         # PR-9
  stage3_target_anchor_partial_tokens.yaml          # PR-7
  stage4_cala_zero_init.yaml                        # PR-8
  stage5_full_joint.yaml                            # gated
  ablation_no_ema.yaml                              # PR-5
  ablation_element_target.yaml                      # A12 — PR-5
  ablation_metadata_only.yaml                       # A10 — PR-5
  ablation_shuffled_study.yaml                      # A9  — PR-5
  ablation_nearest_context.yaml                     # G   — PR-5
  ablation_target_meta_only.yaml                    # F   — PR-5
  baseline_vjepa_predavg.yaml                       # B0  — PR-6
  baseline_supervised_study_transformer.yaml        # B2  — PR-6

tests/echomv_jepa/
  __init__.py                              # LANDED
  conftest.py                              # LANDED — synth_cache + meta32 fixtures
  test_full_study_consistency.py           # LANDED — full = [ctx ∥ tgt], gather correctness, ctx∩tgt=∅
  test_full_study_target_encoder_shapes.py # LANDED — teacher shapes, no_grad, gather, isolated==single-element
  test_ema_update.py                       # LANDED — tau=0.5 math, tau=1.0 no-op, tau=0.0 copies student
  test_modality_projector.py               # LANDED — routing, OOB fallback, empty slice, update advances all pairs
  test_contextualization_diagnostics.py    # LANDED — z_cosine_{vs_v1,vs_isolated,vs_peer_drop} invariants
  # Inherited transitively from v1 (masking logic is unchanged):
  #   test_masking_no_target_leak.py    → covered by tests/echoset_jepa/test_mask_strategies.py
  #   test_k_sampler_fairness.py        → covered by tests/echoset_jepa/test_k_sampler.py
  #   test_matched_negatives.py         → covered by tests/echoset_jepa/test_nce_negatives.py
  # Gap to write before running real training on Stage-1:
  #   test_permutation_invariance.py    → shuffle element order, h_study is invariant; teacher
  #                                       contextualized output is equivariant modulo the shuffle.
  # Deferred follow-ups:
  test_target_anchor_masking.py           # I1–I5 (6 tests) — PR-7
  test_cala_zero_init.py                  # γ=0 → hash-equal with Stage-1 — PR-8
```

Detailed per-module specification → see `docs/echomv_jepa_implementation_sketch.md`.

---

## 20. Go/no-go gates

Quantitative, falsifiable, pre-registered.

### 20.1 Stage 0 → 1 promotion

- EchoSet v1 re-run (A16) reproduces its reported LVEF R² within **±0.01** on the same val split.
- `var_t > 0.3` for the last 5 k steps.
- Falsification probe infrastructure is wired and logs per step.

### 20.2 Stage 1 gates (two-phase)

Stage-1 is gated in **two phases**: pre-downstream (cheap, during/right after pretraining) and downstream. Both must pass to promote to Stage 2.

**20.2.a Pre-downstream gates (Stage-1, 1b, 1m):**

- `var_t > 0.3` and `cov_off < 0.1` for last 2 000 steps.
- **Teacher-contextualization gates (§15.1a):**
  - Mean `z_cosine_vs_v1` over last 2 000 steps ≤ **0.95** (target is informatively different from v1's pre-context projection).
  - Mean `z_cosine_vs_isolated` over last 2 000 steps ≤ **0.90** (teacher actually uses context).
  - No 5 000-step window with `z_cosine_vs_v1 > 0.98` (collapse to v1 halt).
- Stage-1m additionally: no per-modality projector pair has `var_t < 0.3` or is NaN (any modality pair that collapses halts the run).

**20.2.b Downstream gates (Stage-1, 1b, 1m):**

- EchoMV-JEPA Stage-1 beats Stage-0 on:
  - LVEF R² by **≥ +0.015** AND
  - RVSP Pearson r by **≥ +0.02** AND
  - TAPSE MAE not regressed by more than **0.2 mm**,
- at sampler-matched + K-matched compute, on a held-out pre-registered val split.
- **Single-axis failure = no promotion.**
- A9 (shuffled), A10 (metadata-only), A12 (EMA but element-target loss) are strictly worse (by ≥ 50 % of Stage-0→Stage-1 gap).
- Stage-1m additionally: modality-stratified — on color-Doppler and spectral-Doppler targets, `loss_regress` final mean is **≤ Stage-1's** by ≥ 2 %; no B-mode regression (final `loss_regress` within 1 % of Stage-1's).

Promotion to Stage 2 / 3 / 4 requires passing both 20.2.a and 20.2.b.

### 20.3 Stage 2 → 3 promotion

- Adapter-joint Stage-2 matches or beats Stage-1 on LVEF, and does not regress RVSP.
- EchoNet-Dynamic linear probe on the adapter-joint clip encoder does not regress by **more than 2 % absolute** vs the frozen base. If it does, halt and roll back to Stage 1.

### 20.4 Stage 3 → 4 promotion

- All Stage-3 anchor invariants I1–I5 pass.
- A14 (leakage control) shows the expected trivial-low loss, confirming I3.
- Stage-3 does not regress Stage-1 primary metrics.

### 20.5 Stage 4 → 5 promotion

- γ=0 CALA exactly reproduces Stage-1 logits on a fixed batch (hash-equal unit test).
- Attention entropy does not collapse within the first 5 k steps.
- A6 shows CALA helps multi-view tasks (RV, pediatric, held-out site) without regressing LVEF.

### 20.6 Stage 6 eligibility

- Stages 1–5 demonstrably help on B-mode tasks.
- Calibration metadata is populated for ≥ 80 % of studies in the manifest.
- Modality adapters trained; no hemodynamic (RVSP/AS/MR/diastology) claims before Stage 6 passes audit.

### 20.7 Collapse halts

- `var_t < 0.3` for 500 steps → halt.
- `cov_off > 0.1` for 500 steps → halt.
- Attention entropy collapse in CALA layer → halt.

### 20.8 Do-not-ship cliffs

- B-mode-only model reporting RVSP Pearson r > 0.55 on external val → mandatory audit before any external claim.
- Pediatric or external LVEF regression vs Stage-0 → block release.
- Retrieval-only improvement with no downstream gain → not a valid contribution.

---

## PR sequence (PR-0 → PR-9)

| PR | Name | Files | Depends on | Gate |
|----|------|-------|-----------|------|
| **PR-0** | **Docs only** (this PR) | `docs/echomv_jepa_*.md` | — | reviewer approval |
| **PR-1** | Data / taxonomy / manifests | `experiments/echomv_jepa/build_multiview_manifest.py`, `experiments/echomv_jepa/target_difficulty_audit.py` (scaffold) | PR-0 | manifests validate |
| **PR-2** | **Dataset / masking / target slots** (before target encoder!) | `src/datasets/echomv_jepa_dataset.py`, unit tests `test_masking_no_target_leak.py`, `test_k_sampler_fairness.py` | PR-1 | dataset contract frozen |
| **PR-3 (landed)** | **Full-study EMA target encoder + per-modality projector** | `src/models/echomv_jepa/{study_target_encoder,losses,ema,modality_projector,__init__}.py`; pure additive `StudyTransformer.forward_contextualized`; tests `test_full_study_target_encoder_shapes.py`, `test_ema_update.py`, `test_modality_projector.py` | PR-2 | **green — 109/109 v1 tests + 24/24 new tests** |
| **PR-4 (landed)** | Training loop + losses + contextualization diagnostics + three Stage-1 configs | `app/echomv_jepa/{train,__init__}.py`; `configs/train/echomv_jepa/stage1_frozen_clip_full_study_ema.yaml`, `stage1b_frozen_clip_tiny_nce.yaml` (λ=0.005), `stage1m_frozen_clip_modality_projector.yaml` (num_modalities=8); falsification-probe halt + two positive contextualization probes; `test_contextualization_diagnostics.py`, `test_full_study_consistency.py` | PR-3 | **green — config load + module import validated; no training launched** |
| **PR-5** | Controls + diagnostics | `configs/train/echomv_jepa/ablation_{shuffled_study,metadata_only,target_meta_only,nearest_context,no_ema,element_target}.yaml`; `experiments/echomv_jepa/echometr3r_consistency.py`; frozen-probe LVEF dry-run on EchoNet-Dynamic as diagnostic | PR-4 | all controls smoke-run |
| **PR-6** | Downstream K-matched eval | extends `evals/video_classification_frozen_multi/` configs for Stage-1 checkpoints; adds `configs/train/echomv_jepa/baseline_vjepa_predavg.yaml`, `baseline_supervised_study_transformer.yaml` | PR-5 | §20.2 gate evaluated |
| **PR-7** | Target anchoring (gated on PR-5 passing) | `src/models/echomv_jepa/target_anchor.py`, `tests/echomv_jepa/test_target_anchor_masking.py` (6 invariant tests), `configs/train/echomv_jepa/stage3_target_anchor_partial_tokens.yaml` | PR-6 passing §20.2 | I1–I5 invariants green |
| **PR-8** | CALA | `src/models/echomv_jepa/correspondence_attention.py`, `tests/echomv_jepa/test_cala_zero_init.py`, `configs/train/echomv_jepa/stage4_cala_zero_init.yaml` | PR-7 | γ=0 reproduces Stage-1 |
| **PR-9** | Adapter-joint clip training | `src/models/echomv_jepa/clip_backbone.py` (adapter), `L_anchor_to_base_clip` in losses, `configs/train/echomv_jepa/stage2_adapter_joint.yaml` | PR-8, §20.3 gate | EchoNet-Dynamic regression < 2 % absolute |

PR-0 is this document set. No code or config changes in PR-0.
