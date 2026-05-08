# EchoMV-JEPA vs Prior Variants

**Status:** design document, PR-0
**Audience:** NeurIPS reviewers; a reader who has read `docs/echomv_jepa_architecture_plan.md` §§4–6 in summary form and wants side-by-side detail.

This document differentiates EchoMV-JEPA from:
1. Vanilla V-JEPA
2. Frozen EchoSet-JEPA v1 (present in this repo)
3. MV2SV (privileged-multiview single-view student)
4. V4 phase-relational (pooled InfoNCE with hard negatives)
5. TokenRel / MotionDelta (clip-pair token/motion variants)
6. Factorized slots (z_shared / z_phase / z_view)

It does so with side-by-side tables and prose paragraphs that explain *why* each differentiation matters, grounded in the specific negative results observed on this repo.

---

## 1. EchoMV-JEPA vs Vanilla V-JEPA

Vanilla V-JEPA operates at the clip level. EchoMV-JEPA embeds V-JEPA as the clip backbone `f_theta` and adds a study-level JEPA on top. EchoMV-JEPA is strictly a super-system.

| Axis | Vanilla V-JEPA | EchoMV-JEPA |
|------|----------------|-------------|
| Prediction unit | single clip | clinical echo study (variable M elements) |
| Context input | masked spatiotemporal tubelets of one clip | masked/incomplete study: context elements + target mask slots with metadata + `[STUDY]` token |
| Target input (pre-encoder) | full unmasked clip | full unmasked **study** (all K clips → all M elements, no mask) |
| Target encoder | target clip encoder (EMA of online encoder) | EMA target clip encoder **plus** EMA full-study target encoder `F_bar_psi` |
| Target selection | target encoder output at mask token positions | `F_bar_psi` hidden state at target element index, **after** full-study encoding, then EMA projector |
| EMA scope | target clip encoder | target clip encoder (Stage-2+) **and** target study encoder (Stage-1+) |
| Masking unit | spatiotemporal token | study element (whole-view / whole-modality / phase holdout variants) |
| Metadata | none | view_family × modality × phase_bucket (element key); view/modality/phase on target side; measurement_site, quality_bucket on context side |
| Multi-view / multi-modality | not supported | first-class |
| Held-out view / modality prediction | impossible | core training signal |
| Inference unit | single clip | K clips per study, set-aware probe |
| Scientific validity of the single-clip baseline | strong | preserved as `L_clip_vjepa` anchor whenever clip encoder is trainable (Stage-2+) |

### Why it matters

Prior controlled experiments on this repo established that V-JEPA beats BYOL/MAE/SALT at matched compute on LVEF and on robustness; specialized losses that destabilized the clip encoder hurt generality. EchoMV-JEPA is designed to **preserve the clip-level V-JEPA representation** via `L_clip_vjepa + L_anchor_to_base_clip` in Stage-2+, and to add the study-level signal without destroying what works. Stage-1 freezes the clip encoder entirely, which makes Stage-1 the "study-level JEPA on top of frozen V-JEPA" configuration and isolates the study-level contribution as a clean add-on.

---

## 2. EchoMV-JEPA vs Frozen EchoSet-JEPA v1

EchoSet-JEPA v1 (`app/echoset_jepa/`, `src/models/study_transformer.py`, `src/models/study_projectors.py`) is the closest prior in this repo. It is also, by the design stated here, the **Stage-0 control for EchoMV-JEPA**, not a competing method.

### 2.1 Verified v1 behavior (from source)

At `app/echoset_jepa/train.py:176–204`, the `target_mode="full_study_teacher_target"` branch is explicitly a stub:

```python
# For clarity, use element_target path for MVP; Mode B is an
# ablation pathway to be finalized when its ablation YAML runs.
z_t = proj.teacher_forward(st.clip_in(batch["tgt_elements"])).detach()
```

So both branches in v1 compute `z_t = teacher_projector(st.clip_in(tgt_elements))`. `st.clip_in` is a single `nn.Linear(D_clip, d_model)` (defined at `src/models/study_transformer.py:65`). The EMA is on the projector MLP (`EMAProjectorPair` at `src/models/study_projectors.py:29`), not on the study transformer. There is no EMA copy of `StudyTransformer` anywhere in the repo.

### 2.2 Side-by-side

| Axis | EchoSet-JEPA v1 | EchoMV-JEPA |
|------|-----------------|-------------|
| Clip encoder | frozen V-JEPA (cached `c_clip`) | frozen / adapter / fully joint (Stage 1 / 2 / 5) |
| Clip-level EMA | n/a (clip encoder frozen) | Stage-1 n/a (frozen); Stage-2+ EMA clip encoder |
| Study transformer | trainable student only | trainable student (`F_psi`) **and** EMA teacher (`F_bar_psi`) |
| Target source | `z_t = proj_teacher(st.clip_in(tgt_elements))` — pre-context linear | `z_t = proj_teacher(F_bar_psi(full_study)[target_indices])` — contextualized after full-study encoding |
| **Target from contextualized teacher?** | **no** | **yes** |
| Target depends on other elements in same study? | no (per-element linear) | yes (self-attention over all elements) |
| EMA scope | projector MLP only | projector MLP **and** full study transformer |
| Element key | (view_family, modality, phase_bucket) | same |
| Masking strategies | 6 strategies (random, whole_view_family, whole_modality, apical_holdout, doppler_holdout, bmode_holdout) | same + optional phase / adjacent-beat / ED-ES holdouts (Stage-3+) |
| Meta embeddings | view/modality/phase/quality vocabularies implemented | reused unchanged |
| CALA | not present | Stage-4 optional, zero-init, online-only |
| Target anchoring | not present | Stage-3 optional (A/B/C modes) |
| Calibration fields | not in manifest | required in manifest (Stage-6 prerequisite) |
| Stage-0 / control role | was the method | **is the Stage-0 control** |

### 2.3 Why it matters

EchoSet v1's claim was that a permutation-invariant study transformer with element-target loss and a projector-EMA teacher is sufficient to extract study-level structure. If that claim were true, a contextualized study-level target encoder would offer no gain. The Stage-1 MVP's **falsification probe** `cosine(z_t^{EchoMV}, z_t^{v1})` logged per step directly adjudicates this: if the EMA study transformer's output at target slots stays cosine-close (> 0.98) to v1's pre-context projection for 5 k consecutive steps, the EchoMV-JEPA claim is falsified and the run halts. This is a design-level commitment to be falsifiable, not a promise.

The A12 ablation isolates the contribution further: "EMA full-study study-transformer teacher **with** element-target loss (no selection after encoding)". If A12 ≈ Stage-1, the `select after encoding` piece is inert and only the EMA-at-study-transformer is load-bearing. If Stage-1 > A12, the selection-after-encoding is what drives the gain — the precise architectural claim.

---

## 3. EchoMV-JEPA vs MV2SV

MV2SV (`configs/train/vitl16/mv2sv-*.yaml`, `app/vjepa_multiview/`) trained a single-view student at inference with a privileged different-view target during training, plus factorized slots.

### 3.1 Side-by-side

| Axis | MV2SV | EchoMV-JEPA |
|------|-------|-------------|
| Inference input | single view (e.g., A4C) | multiple observed views of the same study |
| Training target | privileged target clip from a *different* view | held-out observed element of the *same* study |
| Cross-view signal | hallucinate target-view latent from source-view | predict held-out observed element from other observed elements |
| Factorization | factorized slots (z_shared / z_phase / z_view) with three independent heads | no slot decomposition; metadata conditioning via additive embeddings |
| Retrieval claim | cross-view retrieval was improved | retrieval is a diagnostic, not a success criterion |
| Downstream result (from `claude/neurips/experiments/mv2sv-privileged-multiview.md`) | slot collapse; downstream LVEF lost to paired intraview controls | TBD; to be evaluated on primary tasks with K-matched controls |
| p_fused (fused-study aux loss) | `p_fused=0.0` in production configs (never trained) | not a construct; study-level objective is always on |

### 3.2 Why it matters

MV2SV's failure was that (a) **single-view hallucination at inference is the wrong framing** for echo: a clinical study always has multiple acquisitions available, and a model that needs to hallucinate from one view is under-using the data at inference. (b) **Slot disentanglement does not emerge from loss weighting alone** — prior factorized heads collapsed; orthogonality constraints and per-slot distinct losses were not sufficient. (c) The **privileged target clip assumption breaks at inference**: downstream tasks do not have a separate privileged target view.

EchoMV-JEPA's design response: *observed* study elements at inference; no factorized slots and no disentanglement claim; target is a masked subset of the observed study (whose observations are available at inference too, but held out of the context window during training). The cross-view signal comes from the held-out element prediction task, not from hallucination.

---

## 4. EchoMV-JEPA vs V4 Phase-Relational

V4 (`app/vjepa_multiview/phase_relational_head.py`, `configs/train/vitl16/*phase-relational*.yaml`) used pooled (B, 1024) → InfoNCE with same-study wrong-phase hard negatives and Fourier Δφ conditioning.

### 4.1 Side-by-side

| Axis | V4 phase-relational | EchoMV-JEPA |
|------|---------------------|-------------|
| Contrastive pool | pooled, batch + hard-negative, single-axis pressure along phase | per-target stratified (same view × modality × phase) NCE, optional, `λ_nce=0.0` by default in Stage 1 |
| Phase representation | pooled Fourier Δφ conditioning on a single query vector | phase as part of element key; no pooled phase-axis specialization |
| Metric hit | LVEF R² 0.652 (best on repo at ViT-L) | TBD; gated by §20.2 |
| Metric miss | RVSP r=0.458 (stalled early-epoch); pediatric/external regression | designed to not regress RVSP or external/pediatric (required by §20.2 gate) |
| TAPSE | not addressed | not claimed in Stages 1–4 |
| NCE role | primary objective, pooled | auxiliary, optional, per-target stratified |
| Hard negatives | same-study wrong-phase | excluded (same-study off-target excluded) |

### 4.2 Why it matters

V4 improved LVEF because adult LVEF is itself largely a phase-dependent measurement (ED vs ES) and pooled phase contrast compressed the representation along the phase axis. The same compression **hurt** tasks that depend on information orthogonal to phase (RVSP, pediatric, external generalization) and on information localized to anatomy (RV function).

EchoMV-JEPA design response: *distribute pressure across physiology, view, modality, and phase*, not compress onto a single axis. Phase is one axis among three in the element key; view family and modality are the other two. The study-level target encoder is physiology-level (contextualized across all elements in the study). Matched NCE is per-target stratified (same view × modality × phase negatives from *other* studies), which makes the NCE task one of within-stratum discrimination, not cross-stratum axis-finding. `λ_nce=0.0` by default in Stage 1 eliminates the V4 failure mode entirely and only reinstates the NCE auxiliary once the cosine regression path is proven.

A V4-style failure in EchoMV-JEPA would show up as LVEF gain accompanied by RVSP regression; this is explicitly a kill criterion in §18.2 and §20.2.

---

## 5. EchoMV-JEPA vs TokenRel / MotionDelta

TokenRel (`app/vjepa_multiview/token_relational_head.py`, `configs/train/vitl16/*tokenrel*.yaml`) was a per-token set InfoNCE across views; MotionDelta was a same-view-only per-token Δφ → latent-delta SmoothL1 + InfoNCE.

### 5.1 Side-by-side

| Axis | TokenRel | MotionDelta | EchoMV-JEPA |
|------|----------|-------------|-------------|
| Scope | clip pair | clip pair (same view) | study-level |
| Matching unit | token set with logsumexp (no spatial alignment) | token-level Δ | element-level (Stages 1–3), token-level optional (Stages 3–4) |
| Anatomical correspondence | absent — positional-blind | absent | optional CALA (Stage-4) with *anatomy-keyed* latent correspondence, not pixel |
| Pooled safety loss needed? | yes (`λ_pool_rel=0.005`) to preserve LVEF | — | no; cosine regression is the primary signal, no pooled safety |
| Token subsample | K=64 | K=64 | full tokens retained (Stages 3+) if `keep_tokens: true` |
| Reported downstream | no RVSP / TAPSE eval | never enabled in production | full protocol (§14) |
| Claim on TAPSE | did not solve | did not solve | **explicitly not claimed** in Stages 1–4 |

### 5.2 Why it matters

Token-level set InfoNCE without spatial or anatomical alignment can reward arbitrary cross-view token matches; when the positional prior is missing and the anatomy prior is missing, the model has no shape reason to prefer anatomically-correct matches. That is the TokenRel failure. MotionDelta was restricted to same-view pairs and thus discarded the cross-view motion signal.

EchoMV-JEPA's design response: **study-level objective first**; token-level objectives are reserved for Stage-4 CALA where correspondences are keyed on `(view_family, anatomy_group)` latent priors, not pixels. TAPSE and local wall mechanics are explicitly **not** in the Stage-1 to Stage-4 claims. A token-transport branch is a possible Stage-5+ addition, gated by showing Stage-1/2/3 gains first. No TAPSE claim is made without an explicit token/local branch.

---

## 6. EchoMV-JEPA vs Factorized Slots

Factorized heads (`app/vjepa_multiview/factorized_head.py`, `view_predictor.py`, `phase_query_head.py`) decomposed pooled encoder output into `z_shared` (phase-invariant), `z_phase` (cycle state), `z_view` (view-local residual).

### 6.1 Side-by-side

| Axis | Factorized slots | EchoMV-JEPA |
|------|------------------|-------------|
| Representation shape | three 256-dim slots with three independent MLPs | single contextualized study representation + per-element hidden states |
| Disentanglement mechanism | loss assignment per slot + different init seeds | none — metadata is a conditioning signal, not an orthogonal slot |
| Disentanglement claim | yes, explicit | **no — explicitly not claimed** |
| Observed behavior | slot collapse / leakage despite init-seed guard; `z_phase` was pooled-from-whole-frame | TBD; no disentanglement contract to violate |
| Downstream eval | never deployed post-pretraining | full protocol (§14) |

### 6.2 Why it matters

Factorized slots are an attractive idea that has repeatedly failed in practice in this repo: without strict orthogonality constraints, the three heads drift into the same representation under gradient pressure from loss weighting alone. Init-seed guards help briefly, then slots collapse. The resulting representation has neither the interpretability it promised nor a clear downstream benefit.

EchoMV-JEPA's design response: *do not attempt disentanglement*. Metadata (`view_family`, `modality`, `phase_bucket`, `measurement_site`) is injected as additive embeddings at the study-encoder input, not as a decomposition of the output. The study-level representation is entangled by construction, and that is accepted. Evaluation is on downstream task performance, not on slot interpretability.

---

## 7. Summary matrix: one row per method, one column per property

Legend: ✓ = has it; ✗ = does not have it; — = not applicable; `Stg-N` = only at stage N.

| Property | V-JEPA | EchoSet v1 | MV2SV | V4 phase-rel | TokenRel | Factorized | **EchoMV-JEPA** |
|----------|--------|------------|-------|--------------|----------|------------|-----------------|
| Clip-level target encoder with EMA | ✓ | ✗ (frozen clip) | ✓ | ✓ | ✓ | ✓ | ✓ (Stg-2+) |
| Study-level trainable encoder | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✓ |
| Study-level EMA teacher encoder | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **✓** |
| Target from contextualized full-study teacher | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **✓** |
| Held-out element prediction (view/modality) | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✓ |
| Variable-M study input | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✓ |
| Metadata conditioning (view/modality/phase) | ✗ | ✓ | partial | partial | partial | partial | ✓ |
| Permutation-invariant over elements | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✓ |
| Phase-axis as single pooled contrastive axis | ✗ | ✗ | ✗ | ✓ (V4 failure mode) | ✗ | ✓ (`z_phase` slot) | ✗ (avoided by design) |
| Single-view-hallucination inference | ✗ | ✗ | ✓ (MV2SV failure mode) | ✗ | ✗ | ✗ | ✗ (avoided by design) |
| Slot disentanglement claim | — | — | ✓ (collapsed) | — | — | ✓ (collapsed) | ✗ (not claimed) |
| Retrieval-as-success | — | — | ✓ (but downstream lost) | — | — | — | ✗ (retrieval is diagnostic only) |
| Pixel reconstruction | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Sampler-matched + K-matched controls | — | ✓ (plan) | — | — | — | — | ✓ (required) |
| Falsification probe vs nearest prior | — | — | — | — | — | — | **✓ (cosine vs v1)** |
| TAPSE claim | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **✗ (explicitly not claimed)** |
| RVSP/AS/MR B-mode-only claim | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **✗ (proxy only until Stage 6)** |

---

## 8. What EchoMV-JEPA does not attempt

Stated explicitly so reviewers can score the claims precisely:

- It does not claim a universal improvement over V-JEPA on all echo tasks.
- It does not claim to solve TAPSE or local wall mechanics in Stages 1–4.
- It does not claim RVSP, AS, MR, or diastology from B-mode alone; those require Stage-6 multimodal extension with calibration tokens.
- It does not claim slot disentanglement; no post-hoc slot-interpretability plot.
- It does not use pixel reconstruction at any stage. MV2MAE is cited as conceptual inspiration for target anchoring (latent, not pixel); MVDiffusion for CALA (latent, not pixel).
- It does not use retrieval accuracy as a success criterion. Retrieval is a diagnostic.
- It does not train on a single-view stream at inference (avoids MV2SV's framing).
- It does not rely on a single pooled phase-contrastive axis (avoids V4's failure mode).
- It does not rely on factorized slot disentanglement (avoids prior failure).
- It does not introduce new vocabularies where `src/models/meta_embeddings.py` already defines them.
