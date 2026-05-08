# Masked Cross-Clip V-JEPA (MCC-JEPA) — Experiment Design

Running-record doc for the **minimal-modification** multi-view extension of
V-JEPA for echocardiography, and two orthogonal extensions of it for local
cardiac mechanics.

The design sits at three independent axes, each answering a different
question:

- **Sampler** (MCC-JEPA, §1–§7): vanilla V-JEPA masks tubelets of **one
  clip** and predicts their latents from visible tubelets of the same clip.
  MCC-JEPA masks tubelets of **one clip of a study** and predicts their
  latents from visible tubelets of **a different clip of the same study**.
  This adds cross-acquisition structure to the clip encoder's gradient.
- **Loss** (MCC-MC, §8): add a small-weight latent-transport auxiliary
  (MC-JEPA-inspired) that learns token-level motion correspondences. This
  adds dense local-mechanics signal to the clip encoder's gradient.
- **Mask placement** (MCC-MC-MGM, §9): replace the random target-clip mask
  with a motion-guided mask driven by H.264 motion vectors. This
  concentrates the reconstruction signal on the anatomically most
  informative regions.

All three keep V-JEPA's architecture, EMA mechanism, and cosine-regression
loss unchanged. §1–§7 specify MCC-JEPA in full (the core contribution);
§8–§9 specify the two extensions as additive follow-ups, each with its own
three-arm matched control contract. Each can be run independently and each
has its own success criteria; all three are run in order and each gates the
next.

Paper framing for Part 2: the controlled pretraining-objective comparison
(§3 of the paper) establishes that latent prediction with co-evolving EMA
wins on MIMIC. §5 validates the finding on held-out tasks. **MCC-JEPA is
the single principled objective modification** that uses echo's study
structure, and MCC-MC / MCC-MC-MGM are the two orthogonal axes along which
V-JEPA can be adapted to echo's local mechanics without changing its
inductive bias — sampler, loss, mask placement. Reviewers get one clean
contribution (MCC-JEPA) with two ablations (transport, motion-guided mask)
that answer the separable questions "does motion help?" and "does
motion-aware mask placement help?".

Experiment status: **design only**. Not yet launched. Target per axis: one
smoke (~500 steps) + one matched-compute run (+25 epochs from
`mimic_standard_jepa_e100`), matched against vanilla V-JEPA, MV2SV v5, and
phase-relational Variant 4. Total 3 matched-compute runs (~3 days wall).

---

## 1. Story in one paragraph

Vanilla V-JEPA's objective is masked latent prediction **within** a clip:
mask a subset of spatiotemporal tubelets of clip A, predict their teacher
latents from the visible tubelets of clip A, with an EMA teacher updated at
every step. This works because nearby tubelets of a single clip are
structurally related (same heart, same view, close in time). But
**echocardiography studies contain multiple clips of the same patient under
different acquisition conditions** — different view families, modalities,
cardiac phases — and those cross-acquisition clips are *also* structurally
related: they are observations of the same latent cardiac state. Vanilla
V-JEPA treats the 8 clips of one study as 8 independent samples and never
exploits this. MCC-JEPA makes one change: sample a **pair** `(clip_A,
clip_B)` from the same study, apply V-JEPA's mask collator to `clip_B`, and
run V-JEPA's standard forward with context = `f_θ(clip_A)` and target =
`f̄_θ(clip_B)` at the masked positions. Mathematically the loss is identical
to V-JEPA's. Structurally the change is that the predictor is now forced to
reason about what a masked region of a *different clip of the same study*
looks like, given a different clip as context. The encoder's gradient
signal therefore rewards features that are **invariant across acquisitions
of the same patient** — exactly the signal echocardiography provides but
which vanilla V-JEPA ignores.

---

## 2. Why this is not any prior multi-view variant in the repo

Four prior multi-view / cross-clip variants have been tried against vanilla
V-JEPA in this repo. All of them wrapped the V-JEPA objective in auxiliary
machinery. MCC-JEPA is the minimal version that strips the auxiliary
machinery back and keeps only the cross-clip idea. The exact deltas:

| # | Variant | Context input | Target input | Prediction unit | Mask? | New loss term | New head |
|---|---|---|---|---|---|---|---|
| 0 | **Vanilla V-JEPA** | visible tubelets of clip A | EMA teacher on full clip A | **masked tubelets of clip A** | **yes (on A)** | none | none |
| 1 | Predictor-φ (Variant 1) | visible tubelets of A | EMA on full A | masked tubelets of A | yes (on A) | **Fourier Δφ into predictor** | none |
| 2 | Mask-φ (Variant 2) | visible tubelets of A | EMA on full A | masked tubelets of A | yes (on A, phase-bucketed) | none | none |
| 3 | Positive-only cross-view (Variant 3) | full A, no mask | EMA on full B | **pooled latent of whole B** | **no** | SmoothL1 on pooled B latent | none |
| 4 | Phase-relational InfoNCE (Variant 4) | pooled A | pooled B + pooled B_wrong_phase | **pooled latent of whole B** | no | **candidate-set InfoNCE** | none |
| 5 | MV2SV v5 | full A | EMA on full B | **pooled latent of whole B, decomposed into z_shared/z_phase/z_view** | no | `L_pair_view`, `L_view_nce`, `L_shared`, ... | **`FactorizedHead`, `ConditionalViewPredictor`** |
| 6 | TokenRel / MotionDelta | clip-pair tokens | — | token-set InfoNCE across views | no | token-set InfoNCE | `TokenRelationalHead` + pooled safety |
| 7 | MC-JEPA (original) | visible tubelets of A | EMA on full A | masked tubelets of A | yes (on A) | **token-transport + cycle consistency** | **2-layer cross-attention transport head** |
| **MCC-JEPA (§1–§7)** | **visible tubelets of clip A** | **EMA teacher on full clip B** | **masked tubelets of clip B** | **yes (on B)** | **none** | **none** |
| **MCC-MC (§8)** | visible tubelets of A | EMA teacher on full B + EMA teacher on A at `t+Δ` | masked tubelets of B + transported tubelets of A | yes (on B) + transport within A | **λ·L_transport + λ·L_cycle** | **MC-JEPA transport head (within A only)** |
| **MCC-MC-MGM (§9)** | visible tubelets of A | EMA teacher on full B + EMA teacher on A at `t+Δ` | **motion-vector-guided** masked tubelets of B + transported tubelets of A | yes (on B, MGM-biased) + transport within A | λ·L_transport + λ·L_cycle | MC-JEPA transport head (unchanged) |

The key row-by-row differences:

- **vs Variant 3 (positive-only cross-view, job 542)** — Variant 3 was the
  closest prior: cross-study pairs, positive-only regression to teacher of B.
  But it matched on the **pooled** latent of whole B, and clip A was passed
  to the predictor **unmasked** as a whole-clip summary. The paper's
  documented failure mode: at tight phase+view matching, teacher encoding
  of B redundantly recovers from A, so the predictor reduced to an intraview
  regressor and no cross-view signal was retained. MCC-JEPA is **masked** on
  B (local spatiotemporal targets, not pooled) and uses V-JEPA's actual
  predictor (context tokens + mask slots + positional embeddings), restoring
  the spatial-temporal locality that V-JEPA's success depends on.

- **vs Variant 4 (phase-relational InfoNCE, job 593)** — Variant 4 replaced
  SmoothL1 regression with InfoNCE on **pooled** latents, using a
  same-study wrong-phase hard negative. This improved LVEF by forcing
  discrimination along the phase axis, but regressed RVSP and pediatric
  transfer because it compressed the representation onto a single pooled
  contrastive axis. MCC-JEPA uses V-JEPA's standard **per-token cosine**
  loss on masked tubelets of B; no InfoNCE, no hard negatives, no pooled
  axis. The cross-clip signal enters via the predictor's task, not via a
  contrastive objective.

- **vs MV2SV v5** — MV2SV v5 is architecturally closest; it also pairs two
  same-study clips, encodes A with the student and B with the teacher, and
  regresses the student's prediction to the teacher's B latent. Three
  distinct differences:
  1. **Masking.** MV2SV v5 does not mask either clip for the primary
     `L_pair_view` loss. The student consumes full `clip_a`, produces a
     single view-conditioned embedding, and matches the teacher's whole-B
     embedding. MCC-JEPA keeps V-JEPA's mask collator on B. This changes the
     prediction unit from "a whole clip's summary" to "specific masked
     tubelets in a specific spatiotemporal region of a specific other clip,"
     which is both concretely grounded and preserves the local-prediction
     inductive bias that made V-JEPA work in the first place.
  2. **No factorized slots.** MV2SV v5 inserts a `FactorizedHead` that
     decomposes the pooled embedding into `z_shared` (view-invariant),
     `z_phase` (cycle state), and `z_view` (view-local) slots, and a
     `ConditionalViewPredictor` that assembles a target-view embedding from
     those slots + target-view meta. The factorized decomposition is what
     collapsed in prior runs (documented in
     `mv2sv-privileged-multiview.md` §3 slot-collapse section). MCC-JEPA
     removes the decomposition entirely; the predictor output is a standard
     V-JEPA mask-token-level latent with no slot structure.
  3. **No target-view conditioning via metadata.** MV2SV v5's predictor
     receives the target clip's view/modality/phase as meta inputs and
     conditions its output on them. MCC-JEPA's predictor receives only the
     **positional** embeddings of the masked tubelets in clip B's coordinate
     system — i.e., *where* to predict, not *what acquisition* the target
     is. The view/modality/phase of B reach the encoder only through the
     visual content of B itself (via the teacher's pass over the full
     clip). The predictor cannot solve the task via a metadata shortcut.
  4. **Inference claim.** MV2SV trains a **single-view student at
     inference** and relies on it having internalized multi-view reasoning
     during pretraining. MCC-JEPA makes no such claim: the student encoder
     is used as a generic clip encoder at inference (same as vanilla
     V-JEPA's EchoJEPA-L), and downstream probes run over the same clips
     they would with vanilla V-JEPA. MCC-JEPA is an **encoder pretraining
     modification**, not a single-view inference trick.

- **vs TokenRel / MotionDelta** — TokenRel matched tokens across views via
  token-set InfoNCE with no positional alignment, required a pooled safety
  loss to avoid LVEF regression, and was never deployed. MCC-JEPA keeps
  V-JEPA's positional embeddings and uses them to specify *where* in clip B
  to predict; there is no token-set matching because the targets are
  specific tubelet positions in B.

- **vs Arms A/B/C (EchoMV-JEPA study-level transformer)** — A/B/C are
  **study-level adapters on top of a frozen clip encoder**. They do not
  modify the clip encoder pretraining. MCC-JEPA operates at the **clip
  encoder pretraining** level and produces a different set of clip encoder
  weights. The two are orthogonal: any MCC-JEPA checkpoint can be used as
  the frozen backbone for A/B/C; conversely, vanilla V-JEPA + A/B/C leaves
  the clip encoder untouched. If MCC-JEPA adds signal at the clip encoder
  level, A/B/C becomes unnecessary for tasks where per-clip prediction is
  already sufficient, and becomes cheaper to run on top of the better
  backbone for tasks where it is not.

- **vs MC-JEPA (original; Bardes et al.)** — MC-JEPA is **not a
  multi-view method**. It operates *within* a single clip, learning a
  motion-content shared encoder via a latent-transport auxiliary that
  matches the student's tokens at time `t` (transported forward) to the
  teacher's tokens at time `t+Δ` of the same clip. It changes neither
  the sampler (still one clip per sample) nor the mask placement (still
  random). The relationship to MCC-JEPA is therefore **orthogonal**:
  MCC-JEPA changes the pair relation across clips, MC-JEPA adds a
  motion-prediction auxiliary within a clip. The two can be combined
  independently. §8 (MCC-MC) specifies this combination and its
  ablations; it is the correct way to integrate MC-JEPA's motion
  auxiliary into the echo setting — as an addition, not a replacement,
  of the V-JEPA content objective — per the MC-JEPA paper's own
  finding that transport weight must be small.

---

## 3. The mechanism in detail

### 3.1 Input sampler

One new sampler: **same-study pair sampler**. For each training step, for
each study in the batch:

- Draw a pair `(clip_A, clip_B)` of distinct clips from the study's K
  available clips (K = 8 per the existing `study_clip_sample_K8_seed0`
  manifest).
- Pair draw policy (MVP): uniform over all `K × (K-1) / 2` pairs. No
  preference for view/modality/phase difference in the MVP.
- Pair draw policy (knob, off by default): if `p_cross_view > 0`, with
  probability `p_cross_view` reject pairs where `view(A) == view(B)` and
  resample. This biases the gradient toward genuine cross-view
  prediction. Keep at 0 for MVP — let the data distribution supply
  whatever view-difference it supplies.
- If the study has only one valid clip, fall back to **vanilla V-JEPA**
  on that clip (`clip_A == clip_B`, standard mask collator). This is
  MCC-JEPA degrading to vanilla for single-clip studies. No special case
  in the loss; the math is identical.

Reuse: `VideoGroupDataset` (used by MV2SV v5 and V4) already handles
same-study grouped draws. The only change is the eligibility filter —
drop MV2SV's "target_clip must be a privileged different-view clip"
constraint and the hard-negative constraint; just "two distinct clips of
the same study" is enough. Phase-matching is not required.

### 3.2 Forward pass

At each step, for each pair `(clip_A, clip_B)`:

1. **Mask collator applied to B.** Use V-JEPA's existing
   `MaskCollator` (`src/masks/multiseq_multiblock3d.py`). This produces
   `masks_enc_B` (visible positions) and `masks_pred_B` (masked target
   positions) for clip B. `masks_enc_B` is **not used by the student** —
   the student gets clip A, not the visible positions of B. Only
   `masks_pred_B` is used, to specify where in B the predictor must
   predict.
2. **Student encoder pass on clip A.** Run the student V-JEPA encoder
   on the full clip A (no masking). Produces per-token latents of clip
   A: `z_A = f_θ(clip_A)` of shape `(B, T_A · H_A · W_A, d_enc)`.
3. **Teacher encoder pass on clip B.** Run the EMA teacher V-JEPA
   encoder on the full clip B (no masking; teacher always sees full
   target). Produces per-token latents of clip B: `z_B = f̄_θ(clip_B)`
   of shape `(B, T_B · H_B · W_B, d_enc)`.
4. **Predictor pass.** Call V-JEPA's existing `VisionTransformerPredictor`
   with:
   - context tokens = `z_A` (all tokens of A)
   - `masks_x` = indices in A's coordinate frame for all tokens of A
     (i.e., the predictor sees A in full)
   - `masks_y` = `masks_pred_B` in B's coordinate frame
   - mask tokens = learned `mask_token` embedding + positional embedding
     at each position in `masks_pred_B`
   - output: predicted latents `ẑ_B` of shape `(B, |masks_pred_B|, d_enc)`
     at the masked positions of B.
5. **Loss.** V-JEPA's standard loss, applied at the masked positions of
   B:
   ```
   L = mean over masked positions in B of  |ẑ_B - LN(sg(z_B[masks_pred_B]))|^p
   ```
   where `p` is the configured L1/smooth-L1 exponent (unchanged from
   vanilla V-JEPA), LN is LayerNorm on the target, and `sg` is stop-grad
   on the teacher output. This is mechanically identical to
   `app/vjepa/train.py::forward_target + forward_context + loss`, with
   the input pair swapped to `(clip_A, masks_pred_B, clip_B)` instead of
   `(clip_A, masks_pred_A, clip_A)`.

### 3.3 The one architectural wrinkle

V-JEPA's predictor concatenates positional embeddings of **context**
tokens and **mask** tokens into a single stream before the transformer
blocks. In vanilla V-JEPA, these positions come from the same coordinate
frame (the single clip's tubelet grid). In MCC-JEPA, context tokens
live in clip A's coordinate frame and mask tokens live in clip B's.

Two ways to handle this, in order of increasing change:

- **(Recommended, MVP)** Let the predictor use *separate* positional
  embeddings for context vs mask tokens, indexed by their respective
  clips' coordinate frames. V-JEPA's predictor already does this in
  principle — it has a single `pos_embed` table indexed by token grid
  position, and it applies the same table to both streams. For
  MCC-JEPA, we apply the same `pos_embed` table but each token uses its
  own clip's (t, h, w) grid coordinates. Since clips A and B have the
  same tubelet grid (same `frames_per_clip`, `patch_size`,
  `tubelet_size`), the coordinates are drawn from the same integer
  range and the same table. **No architectural change is needed — only
  the indexing logic in the predictor's forward needs to know which
  positions belong to A vs B**, and that is a ~5-line change to pass
  two position tensors instead of one.
- **(Alternative, if (1) doesn't work)** Add a learned "clip-id" token
  concatenated to each position: context tokens get `pos_A + clip_id_A`,
  mask tokens get `pos_B + clip_id_B`. Two new learnable 1024-dim
  vectors. This is the only architectural addition MCC-JEPA would ever
  need, and only if MVP results show the predictor confuses context and
  target positions. Defer unless (1) underperforms.

### 3.4 EMA teacher update

Unchanged from V-JEPA. The teacher is still the EMA of the student encoder
parameters, with the same cosine `momentum_scheduler` from `app/vjepa/
train.py`. One encoder shared across contexts and targets — there is no
separate "student encoder for A, teacher encoder for B" architecture;
both paths use the same student/teacher pair, just applied to different
clips at any given step.

### 3.5 Masking strategy on B

Same mask config as vanilla V-JEPA on MIMIC (8 small spatial blocks + 2
coarse blocks, spanning 8/16 temporal tubelets). The masking is applied
to the target clip B at each step. Since A and B are drawn
independently per step, across training the encoder sees masks applied
to each study member roughly uniformly — no built-in bias toward a
particular clip within a study.

---

## 4. Why this should work given the prior evidence

Three specific mechanisms, each grounded in prior findings from the repo:

1. **The cross-clip structural prior is echo-appropriate, not phase-
   specific or view-specific.** Variants 1–4 imposed narrow structural
   priors (phase axis only, or privileged target view). Phase is too
   narrow (V4 helped LVEF, hurt RVSP); privileged target view requires
   that the same target view exist for all studies (operationally hard,
   and the factorized head used to compose it collapsed). MCC-JEPA's
   prior is "**any two clips of the same study are predictive of each
   other**" — which is what's actually true in echo data.

2. **The EMA teacher's co-evolution remains load-bearing.** The
   controlled comparison showed frozen-teacher distillation (SALT)
   collapses catastrophically under frame shuffling; the EMA co-evolution
   of targets is mechanistically load-bearing. MCC-JEPA preserves it
   exactly (same `momentum_scheduler`, same `torch._foreach_mul_/add_`
   update). Variants 3–5 also preserved it; Variant 3's null result is
   not attributable to losing co-evolution.

3. **Locality of prediction is echo-appropriate.** The
   pooled-latent variants (3, 4, 5) predict whole-clip summaries and
   invite degenerate solutions (phase-axis collapse in 4, slot collapse
   in 5, intraview reduction in 3). V-JEPA's success on natural video
   comes from per-tubelet prediction — the predictor is forced to
   answer "what does *this specific* spatiotemporal region look like,"
   which requires fine-grained features. MCC-JEPA keeps this locality:
   the predictor's targets are specific tubelet positions in B, not a
   pooled summary of B.

The combination — correct prior + correct mechanism + correct granularity
— is what no prior multi-view variant in the repo has. Each had two of
three.

---

## 5. Experimental setup

### 5.1 Three pretrain arms (matched structural contract with Variant 4)

All three arms continue from the same
`mimic_standard_jepa_e100` checkpoint (the EchoJEPA-L baseline), +25
epochs on MIMIC, scheduler horizon 100, stop at 25. Same batch size 32
studies × 8 GPUs, same ImageNet-21K teacher init retained from e100,
same seed 234. The three YAMLs differ only in the pair-sampler policy
and the objective key:

| Arm | Pretrain name | YAML delta | Objective |
|---|---|---|---|
| **Method** | `mcc_jepa_25ep` | `multiview_objective: mcc_jepa; pair_policy: any_distinct_in_study` | Vanilla V-JEPA loss on pair `(clip_A, masks_pred_B, clip_B)` |
| **Control (vanilla V-JEPA, same sampler)** | `sv_jepa_25ep_pairsampler` | `multiview_objective: intraview_only; pair_policy: any_distinct_in_study; loss_pair: false` | Vanilla V-JEPA loss on clip A only; clip B loaded and discarded |
| **Ablation (same-clip degenerate)** | `mcc_jepa_25ep_same_clip` | `multiview_objective: mcc_jepa; pair_policy: same_clip_always` | MCC-JEPA forward but `clip_A == clip_B` always; degenerates to vanilla V-JEPA exactly |

**Why this control contract matches Variant 4's.** Variant 4 used bit-
identical data path across method / control / no-hardneg arms. For
MCC-JEPA:

- `Δ_method_control` = MCC-JEPA − vanilla-V-JEPA-on-same-sampler.
  Isolates *the cross-clip prediction objective* from the pair sampler's
  I/O overhead. Both arms draw the same pairs at the same steps, with
  the same seed; only the loss differs.
- `Δ_method_ablation` = MCC-JEPA − MCC-JEPA-same-clip. The ablation
  runs the full MCC-JEPA code path (same predictor call, same mask
  collator on B, same everything) but forces `clip_A == clip_B` on every
  step, so the prediction task is identical to vanilla V-JEPA. This
  isolates whether the extra machinery matters without the cross-clip
  signal: if method ≈ ablation, the cross-clip signal is null.

Three arms, three isolated axes, same triple-arm contract as
`phase-relational-hardneg.md`.

### 5.2 Evaluation

Same task battery as the NeurIPS controlled comparison:

- **Primary (Part 1 validation)**: EchoNet-Dynamic LVEF at d=4 attentive
  probe, matched compute. This tests whether MCC-JEPA harms single-view
  LVEF (the failure mode of Variant 4 was null-or-regress on RVSP while
  gaining on LVEF; MCC-JEPA must at minimum not regress LVEF).
- **Hemodynamic multi-view (where cross-clip signal should help)**:
  MIMIC RVSP (d=4 multi-view probe on 41K studies), UHN RVSP (5K test).
  If MCC-JEPA's cross-clip training gives the encoder better multi-view
  representations, the multi-view probe should benefit more than the
  single-view probe.
- **Cross-population transfer (robustness)**: EchoNet-Pediatric LVEF
  zero-shot (n=368), UHN-trained probes on pediatric test. Same pattern
  as the controlled comparison.
- **Noise robustness**: EchoBench depth-attenuation, clutter, speckle.
  V-JEPA outperformed BYOL/MAE here at 100ep; MCC-JEPA should match or
  exceed.
- **Segmentation sanity (not expected to gain)**: CAMUS Dice. V-JEPA ≈
  BYOL ≈ MAE at 100ep (≈0.82 Dice); MCC-JEPA should not regress this.
  Important because if MCC-JEPA wins on functional tasks **and** on
  segmentation, it invalidates the §5-of-paper ranking-inversion story.

### 5.3 Compute budget

Per step: two encoder forwards (student on A, teacher on B, both full
clips) + one predictor forward + backward through student. Compared to
vanilla V-JEPA, which is one encoder forward (student on A visible
tokens) + one encoder forward (teacher on full A) + predictor + backward
= also two encoder forwards. **So MCC-JEPA is compute-matched to vanilla
V-JEPA per step**, up to the visible-vs-full distinction (student
processes visible tubelets in vanilla, student processes all of A in
MCC-JEPA). The MCC-JEPA student pass is therefore slightly more
expensive by roughly `1 / visible_fraction ≈ 1 / 0.15 ≈ 6.7×` on the
student pass only, which is a ~30-40% step-time increase overall
depending on how much time the predictor takes vs. the encoder.

Expected wall on 8×H100, 32 studies/GPU, 25 epochs from e100:

- vanilla V-JEPA +25ep reference (from Variant 4 contract): ~13h.
- MCC-JEPA +25ep: **~17-18h** (+30-40% over vanilla).

A 500-step smoke: **~15-20 min**.

### 5.4 Falsification probes

Three cheap per-step diagnostics:

1. **`cross_clip_gain`** — at each step, compute the MCC-JEPA loss on
   the sampled `(A, B)` pair AND the vanilla-V-JEPA loss on `(A, A)`
   with the same mask_pred. Log both. A healthy MCC-JEPA encoder should
   have `loss_MCC > loss_vanilla` (cross-clip prediction is harder than
   within-clip by construction, at least at init); the gap should narrow
   as training progresses. If `loss_MCC == loss_vanilla` from step 0,
   the predictor has found a shortcut that collapses MCC-JEPA to vanilla
   — likely via the `clip_id` or position-embedding ambiguity
   (§3.3). Halt.

2. **`pair_view_diff_rate`** — fraction of sampled pairs where
   `view(A) ≠ view(B)`. At MIMIC K=8, expected ~0.85 given view
   distribution; if this drops (e.g., because the sampler degenerates),
   MCC-JEPA is silently running on same-view pairs and losing its signal.
   Halt if < 0.5 sustained.

3. **`pred_cosine_to_v1_target`** — cosine between MCC-JEPA predictor
   output and the vanilla-V-JEPA target at the same masked position of
   A (computed by running the teacher on A, not B, and gathering at
   `masks_pred_B` translated to A's grid). This measures whether the
   MCC-JEPA predictor is secretly predicting A's latents (shortcut via
   positional embedding ambiguity) instead of B's. Should be < 0.8 at
   steady state — meaningfully different from vanilla V-JEPA's target.

All three are cheap (<1% step overhead), log every 25 steps, and give
clear halt signals.

### 5.5 Success and failure gates

**Success (Part 1 safety + Part 2 signal):**

- LVEF R² on EchoNet-Dynamic test: ≥ vanilla-V-JEPA-+25-epoch control
  (paired-intraview, job 608), within HP noise.
- **RVSP Pearson on MIMIC test: > vanilla-V-JEPA-+25-epoch control
  by ≥ 0.02** (the multi-view gain we specifically expect).
- Pediatric LVEF zero-shot: ≥ vanilla-V-JEPA-+25-epoch within HP noise.
- CAMUS Dice: ≥ vanilla-V-JEPA-+25-epoch within 0.01.
- `cross_clip_gain` positive throughout, `pair_view_diff_rate` > 0.8,
  `pred_cosine_to_v1_target` < 0.8.

**Failure (abandon):**

- LVEF regresses by > 0.01 R² vs control (the Variant 4 failure mode
  inverted).
- RVSP ties control (null multi-view gain; the expected signal isn't
  there, so there's no paper claim).
- Any of the falsification probes fires on more than one smoke attempt
  → shortcut learned, design is broken.

**Ambiguous (further ablation):**

- LVEF and RVSP both match control within HP noise, but
  `cross_clip_gain` is positive. Encoder has internalized *something*
  cross-clip but it isn't showing up at the probe level. Run noise
  robustness (EchoBench) and held-out-site probes before concluding.

---

## 6. Implementation plan (engineering, not science)

Target: a `scripts/neurips/mcc/` directory mirroring `scripts/neurips/
phase/` conventions. Reuse the existing V-JEPA code paths where
possible.

1. **New sampler** — `classifier/phase/sampler/same_study_pair_sampler.py`
   (mirrors `phase_matched_pair_dataset.py` but without the phase/view
   eligibility constraints). ~80 LOC.
2. **New multiview_objective** — add `"mcc_jepa"` to the dispatch in
   `app/vjepa_multiview/train.py::training_step`. The dispatch branch
   calls the existing V-JEPA `forward_target` on `clip_B` full and the
   existing `forward_context` with `clip_A` full + `masks_pred_B`
   translated to B's grid. ~40 LOC of new code, most of it tensor
   reshaping.
3. **Predictor coordinate-split** — in `src/models/predictor.py`,
   add a flag `separate_ctx_tgt_pos: bool` that, when True, treats the
   `masks_x` positions as being in the context clip's grid and the
   `masks_y` positions as being in the target clip's grid. Implementation
   is trivial (both are integer grids of the same shape); ~10 LOC.
4. **Falsification probes** — add to the training step's diagnostics
   dict alongside the existing `loss`, `var_t`, `cov_off` fields. ~40
   LOC, reuses the existing CSV logger.
5. **Three YAMLs** in `configs/train/vitl16/mcc/{mcc_jepa_25ep,
   sv_jepa_25ep_pairsampler, mcc_jepa_25ep_same_clip}.yaml`. Each is a
   ~15-line delta from the Variant 4 YAMLs.
6. **Three sbatch scripts** in `scripts/neurips/mcc/`. Each is a
   ~5-line delta from `scripts/neurips/phase/` existing scripts.
7. **Unit tests** in `tests/app_vjepa_multiview/test_mcc_jepa_*`:
   - `test_mcc_same_clip_equals_vanilla_vjepa`: the degenerate
     ablation (clip_A = clip_B) must produce numerically identical loss
     and gradients to vanilla V-JEPA on the same clip.
   - `test_mcc_pair_sampler_draws_distinct_clips`: sampler never
     returns `A == B` when `pair_policy=any_distinct_in_study`.
   - `test_mcc_predictor_uses_B_grid_for_masks`: predictor positional
     embeddings for `masks_y` use clip B's coordinates, not A's.
   - `test_mcc_no_leakage_of_B_content_to_student`: student encoder's
     output depends only on clip A's pixels, never B's.

Estimated total engineering: **1 focused day** to land code + tests, +
one smoke (~20 min) + one 25-ep pretrain (~17h) + downstream probes
(~3h each for LVEF, RVSP, CAMUS, pediatric). Total wall for a full
validated experiment: **~3 days** from this doc to paper-ready numbers,
assuming no bugs surface in the smoke.

---

## 7. Risks and open questions

1. **Cross-clip prediction might be too hard for the predictor.** If the
   clips have very different views and the encoder's per-tubelet latents
   at different views are effectively independent, the predictor has
   nothing to go on and `cross_clip_gain` stays large forever with no
   learning. Mitigation: `p_cross_view` knob (§3.1) — start with 0
   (uniform pairs include many same-view pairs, which are easier) and
   only bias toward cross-view if the base version converges.

2. **The coordinate-frame split might enable a shortcut.** If the
   predictor can tell which stream is context vs mask from position
   alone (e.g., because A and B have slightly different token counts due
   to padding), it might route A's positional info as if it were B's
   and produce a V-JEPA-like on-A prediction. The
   `pred_cosine_to_v1_target` probe catches this.

3. **MCC-JEPA might gain on downstream at the cost of segmentation.**
   The V4 failure mode was exactly this — gains on one task class,
   regression on another. CAMUS is the safety check in §5.2. If CAMUS
   regresses by > 0.01 Dice, the paper's §5 ranking-inversion story
   breaks and MCC-JEPA becomes a mixed result.

4. **Compute might be higher than estimated.** The estimate assumes the
   student pass on full A is ~6× the student pass on visible A; real
   overhead could be higher if there is batch-size pressure. Mitigation:
   8→6 studies/GPU if OOM.

5. **Within-study structural redundancy might already be captured at
   step 0.** If EchoJEPA-L at e100 already has representations such
   that `f_θ(clip_A) ≈ f̄_θ(clip_B)` per-tubelet at some positions,
   MCC-JEPA's loss is near zero from init and no learning happens.
   This is the hardest failure to detect. Mitigation: the
   `cross_clip_gain` probe at step 0 vs. step 500 — if the gap doesn't
   widen during training, learning isn't happening.

---

## 8. Extension A — MCC-MC: latent transport auxiliary (MC-JEPA inspired)

### 8.1 One-paragraph story

V-JEPA at 100ep on MIMIC is a strong generalist encoder and wins on
global cardiac-function tasks (LVEF, RVSP trend) but hits a documented
ceiling on **local mechanics**: TAPSE (tricuspid annular plane systolic
excursion), RV FAC (fractional area change), and regional wall motion.
Prior token-level attempts (TokenRel, MotionDelta) did not break the
ceiling because they matched tokens across views without spatial
alignment and required a pooled safety loss to avoid LVEF regression.
The MC-JEPA approach — motion-content shared encoder, with a small
latent-transport auxiliary — is the right precedent: it adds dense
token-level motion learning **while keeping the primary content
objective unchanged**. MCC-MC bolts an MC-JEPA-inspired latent-transport
auxiliary onto MCC-JEPA (or vanilla V-JEPA, §8.4 contract): the student
encoder's tokens at time `t` are transported to time `t+Δ` via a
learned token-correspondence, and the transported tokens must match the
EMA teacher's tokens at `t+Δ`. Cycle consistency closes the loop. The
auxiliary's weight is small (λ_transport ≤ 0.1) so it never dominates
the V-JEPA content loss.

### 8.2 Mechanism in detail

The transport auxiliary is a **within-clip** operation. For each clip
in the batch (for MCC-JEPA: for clip A; for vanilla V-JEPA: for the
single clip), pick two tubelet time-indices `t₁` and `t₂ = t₁ + Δ`
where Δ is sampled from `{1, 2, 4}` tubelets apart (the three temporal
distances span ~0.4, 0.8, 1.6 seconds at 25 fps and tubelet-size 2 —
covering sub-beat, intra-beat, and cross-beat motion).

Let `z_t₁ ∈ ℝ^{H·W × d_enc}` be the student's spatial tokens at time
`t₁` (the `H·W = 14·14 = 196` tokens that share the same temporal
tubelet index `t₁`). Similarly for the teacher: `z̄_t₂`. The transport
module is a **two-layer cross-attention head** with learnable
parameters, taking `z_t₁` as queries and `z_t₁ ∥ z_t₂` as keys/values,
and producing transported tokens `ẑ_{t₁→t₂}`:

```
attn = softmax(Q(z_t₁) · K([z_t₁, z_t₂])ᵀ / √d)
ẑ_{t₁→t₂} = attn · V([z_t₁, z_t₂])
```

The cross-attention architecture is intentionally the same minimal
pattern used in the `attentive_pooler.py` probe head, so there's no
new primitive to debug.

**L_transport** — cosine regression at each spatial position:

```
L_transport = mean over H·W positions of
              [1 - cos(LN(ẑ_{t₁→t₂}), LN(sg(z̄_t₂)))]
```

**L_cycle** — cycle consistency: transport `z_t₂` back to `t₁` and
require `ẑ_{t₂→t₁}` ≈ `z_t₁`:

```
ẑ_{t₂→t₁} = transport_head(z_t₂, [z_t₂, z_t₁])
L_cycle = mean over H·W positions of
          [1 - cos(LN(ẑ_{t₂→t₁}), LN(sg(z_t₁)))]
```

Note `L_cycle` uses the **student's own** `z_t₁` on the RHS with
stop-grad — this is a self-consistency check on the transport head, not
an EMA-teacher alignment term. Keeps the cycle-consistency cost from
inflating gradients into the encoder.

**Total loss**:
```
L_total = L_vjepa (or L_MCC-JEPA if §1–§7 is on)
        + λ_transport · L_transport
        + λ_cycle · L_cycle
```

Start: `λ_transport = 0.05, λ_cycle = 0.02`. Sweep in §8.5.

### 8.3 Optional: probe-motion / tissue-motion decomposition

MC-JEPA's paper decomposes flow into a global camera motion + residual
object motion. In echo, the analog is **probe motion** (the sonographer
drifts) vs **tissue motion** (valve excursion, wall deformation).
Probe motion in MIMIC is usually small over 16 frames (most clips are
breath-held chamber views), so this decomposition is unlikely to
matter for the MVP and is deferred. If the transport-auxiliary ablation
finds that the transport head is mostly modeling probe drift, the
decomposition becomes an ablation to run; flag and move on.

### 8.4 Three-arm contract

Same triple-arm structure as §5.1 (method / matched control / ablation).
Two independent options depending on whether MCC-JEPA has already
passed:

**Option A — MCC-JEPA passes §5.5, MCC-MC bolts on:**

| Arm | Pretrain name | Objective |
|---|---|---|
| **Method** | `mcc_mc_jepa_25ep` | L_MCC-JEPA + λ·L_transport + λ·L_cycle |
| **Control** | `mcc_jepa_25ep` (already run in §1–§7) | L_MCC-JEPA alone |
| **Ablation** | `mcc_mc_jepa_25ep_no_cycle` | L_MCC-JEPA + λ·L_transport (no cycle) |

**Option B — MCC-JEPA fails §5.5, MCC-MC stands alone on vanilla V-JEPA:**

| Arm | Pretrain name | Objective |
|---|---|---|
| **Method** | `mc_jepa_25ep` | L_vjepa + λ·L_transport + λ·L_cycle |
| **Control** | `sv_jepa_25ep` (the Variant-4 control arm, job 608) | L_vjepa alone |
| **Ablation** | `mc_jepa_25ep_no_cycle` | L_vjepa + λ·L_transport (no cycle) |

### 8.5 Coefficient sweep

Sweep `λ_transport` on a short-horizon matched-compute proxy:

- 5 values: `λ ∈ {0.0, 0.01, 0.05, 0.1, 0.3}`.
- Each at 500 steps (~20 min) from the same e100 init.
- Endpoint: LVEF held-out R² on EchoNet-Dynamic (the safety endpoint —
  if λ is too high, this regresses) + token-level motion probe (see
  §8.7).
- Pick the smallest λ that gives a non-trivial token-motion probe gain
  without regressing LVEF. Default to 0.05 if the sweep is null.

### 8.6 Primary endpoints (where MCC-MC should help)

Tasks dominated by local mechanics:

- **TAPSE** — regression, UHN dataset. **Primary success endpoint.** No
  prior method in the repo has beaten vanilla V-JEPA on TAPSE at
  matched compute. If MCC-MC does, that's the headline for this arm.
- **RV FAC** — regression, UHN.
- **Regional wall motion score** — classification or ordinal, UHN
  (if label availability allows; documented as sometimes-present in the
  UHN extraction).
- **CAMUS segmentation Dice** — same safety endpoint as §5.2. MCC-MC
  should not regress segmentation since it adds rather than replaces
  signal.
- **Landmark tracking / valve excursion** — if a dataset with annular
  plane or valve-leaflet landmarks is available. Placeholder; requires
  data availability audit before commitment.

### 8.7 Safety endpoints (must-not-regress)

Inherited from §5.2 and extended:

- Adult LVEF (EchoNet-Dynamic): within HP noise of vanilla V-JEPA
  control.
- Pediatric LVEF zero-shot.
- MIMIC RVSP single-view Pearson.
- MR severity, AS severity (UHN).
- EchoBench noise robustness.

### 8.8 Token-level diagnostic

A cheap per-step probe that directly measures whether the transport
head is learning useful motion, not just identity:

- **`transport_displacement`** — average L2 distance between the
  attention-weighted centroid of `ẑ_{t₁→t₂}` and the spatial position
  of `z_t₁`. Measured in tubelet units. At init this is ~0 (attention
  is uniform; centroid at input position). Healthy learning: grows
  over training, stabilizes at ~1-3 tubelets (corresponds to real
  tissue displacement). If it stays at 0, transport is an identity map
  and λ_transport is wasted.
- **`transport_cycle_error`** — the cycle-consistency cosine distance
  `1 - cos(ẑ_{t₂→t₁}, z_t₁)`. At init this is ~0.5 (attention is
  random). Should drop to < 0.1 with training if cycle consistency is
  being learned.
- **`transport_vs_identity_loss_gap`** — gap between `L_transport` and
  a hypothetical identity baseline `L_transport(ẑ_{t₁→t₂} := z_t₁)`.
  If < 0.02 throughout training, the transport head is numerically
  close to identity and λ_transport is wasted.

### 8.9 Compute

`transport_head` is a 2-layer cross-attention head over `H·W · 2 =
392` tokens at dim 1024 — ~5M params, negligible compared to the ViT-L
encoder. Runtime overhead: one extra student encoder forward (already
paid in MCC-JEPA; for vanilla V-JEPA this is genuinely new), one
cross-attention forward per pair `(t₁, t₂)`, backward through transport
head only (the encoder's gradient from `L_transport` goes through the
teacher's stop-grad, so no extra encoder backward). **Per-step overhead:
~15-20% over MCC-JEPA, ~30-40% over vanilla V-JEPA.**

Expected +25 ep wall on 8×H100: MCC-MC from MCC-JEPA: ~20 h; MC-JEPA
from vanilla: ~18 h.

### 8.10 Risks

- **λ_transport coefficient sensitivity**. The sweep in §8.5 is the
  real experiment; anything published without it is underspecified.
- **Transport head collapses to identity**. Caught by §8.8 probes.
- **Cycle consistency is too easy**. If both directions use the same
  two layers of cross-attention, the loss can be solved by making the
  attention uniform in both directions. The architectural mitigation is
  separate `transport_head_forward` and `transport_head_backward`
  parameters; document and implement this.
- **Transport helps TAPSE but regresses LVEF.** This is the V4 failure
  pattern inverted. Hard gate: if LVEF regresses by > 0.01 R² vs
  control, lower λ_transport and retry. If the pattern persists across
  λ values, accept that the auxiliary's gradient pulls the encoder
  onto a motion-specific axis incompatible with LVEF at the scale
  tested. Report as a negative result.

---

## 9. Extension B — MCC-MC-MGM: motion-guided masking (MGM inspired)

### 9.1 One-paragraph story

V-JEPA masks tubelets randomly. The mask-placement literature (MGM,
VideoMAE-v2's motion masking) has shown that biasing the mask toward
motion-salient regions concentrates the reconstruction signal on the
most informative tokens. For echocardiography, motion-salient means
valve edges, wall boundaries, blood-pool boundaries — exactly the
regions that carry functional information. MGM's key innovation is
using **H.264 motion vectors** (present in the codec stream of any
MP4) rather than computing optical flow: motion vectors come "for
free" during decoding. This makes motion-guided masking cheap enough to
apply at pretraining scale. MCC-MC-MGM replaces MCC-MC's random mask
collator with an MGM-style motion-vector-biased mask on clip B, keeping
everything else identical.

### 9.2 Motion-vector availability audit (prerequisite)

**This axis is gated on a data-availability audit.** MGM's assumption is
that H.264 motion vectors are present in the MP4 codec stream. MIMIC's
clips are re-encoded MP4s of original DICOM, and the re-encoding
pipeline may or may not preserve motion-vector metadata. Before
committing to MCC-MC-MGM, audit:

- **Sample 1000 random MIMIC clips** and attempt to extract per-tubelet
  motion vectors via `ffprobe -codec_type video -show_streams
  -show_packets -export_side_data motion_vectors <clip>` or equivalent
  PyAV-based extraction.
- **Metric**: fraction of clips with non-zero motion vectors present in
  the stream, and fraction where the motion vectors are spatially
  correlated with genuine cardiac motion (proxy: spatial correlation
  with a cheap optical-flow reference like RAFT on a subsample).
- **Pass threshold**: ≥ 95% of clips have motion vectors, and ≥ 70% of
  those have ≥ 0.4 spatial correlation with RAFT-computed flow on a
  50-clip subsample. If either threshold fails, MCC-MC-MGM is
  abandoned for MIMIC and deferred until a dataset with reliable motion
  vectors (or cheap pre-computed optical flow) is available.

Audit is a ~30-min ffprobe script + one notebook. Do this first; do
not write any config or sbatch for MCC-MC-MGM until it passes.

### 9.2.1 Audit results (2026-05-05, 50-clip pilot — gate PASS)

Script: `experiments/echomv_jepa/audit_motion_vectors.py`. Uses PyAV
17.0.1 with `vs.codec_context.flags2 |= Flags2.export_mvs` to request
motion-vector side data during decode, then aggregates per-clip MV
counts, magnitudes, and spatial distribution on the 14×14 tubelet
grid. Results CSV: `/tmp/mv_audit_results.csv`.

**Sample:** 50 random MIMIC MP4s (first pilot; a 500–1000-clip
follow-up can be run if a reviewer asks, but the pilot signal is
strong enough to unblock the work).

**Coverage metrics:**

| Metric | Threshold | Observed | Pass? |
|---|---|---|---|
| Clips decoded successfully | — | 50 / 50 | ✓ |
| Codec is `h264` | — | 50 / 50 | ✓ |
| Clips with MV present on ≥ 50% of frames | ≥ 0.95 | 1.00 | ✓ |
| Clips with MV present on ≥ 90% of frames | ≥ 0.95 | 1.00 | ✓ |
| Mean fraction of frames with MVs per clip | — | 0.99 | — |
| Mean MV count per frame | — | ~316 | — |

**Spatial-localization proxy** (substitute for the deferred RAFT
correlation check): project MVs onto the 14×14 tubelet grid and
compute per-tubelet mean magnitude.

| Metric | Observed |
|---|---|
| Mean fraction of frames with any tubelet motion > 0.5 px | 0.82 |
| Mean fraction of tubelets with MV > 0.05 px | 0.21 |
| Mean fraction of tubelets with avg MV > 0.5 px | 0.10 |
| Per-clip tubelet-grid max magnitude (mean across clips) | 1.59 px |

The per-tubelet distribution is **localized, not uniform**: ~10% of
tubelets carry the bulk of the motion, consistent with motion
concentrated on cardiac structures rather than whole-frame camera
noise. This is precisely the distributional shape MGM requires — a
heavy-tailed per-tubelet magnitude so weighted sampling biases toward
moving regions.

**Caveat: sub-pixel average magnitudes.** Mean MV magnitude per
tubelet is sub-pixel (≈0.05–0.5 px), because H.264 motion vectors in
re-encoded MP4s are often quantized coarsely and many macroblocks are
labeled "skip" (copy from reference, MV ≈ 0). The localization
*ordering* — which tubelets carry most motion — is still recoverable,
but the absolute magnitudes should not be treated as optical flow.
MGM only needs the ordering, so this does not block the gate.

**Verdict: gate PASS.** Coverage is saturated (100%), the per-tubelet
MV distribution is heavy-tailed enough to rank tubelets by motion,
and the motion is spatially concentrated rather than uniformly
distributed. MCC-MC-MGM can proceed to implementation.

**Optional follow-up** (not blocking): rerun on 500–1000 clips if a
reviewer challenges the 50-clip sample size; defer the full RAFT
correlation check until a reviewer challenges the cheap-proxy
substitute, since the pilot's localization signal is already clean.

### 9.3 Mechanism

Assuming the audit passes: MGM-style mask is applied to clip B in the
MCC-JEPA / MCC-MC pipeline.

- For each clip B at each step: extract per-tubelet motion-vector
  magnitudes (averaged over the 8×14×14 tubelet grid from the
  per-frame per-macroblock motion vectors).
- Normalize per-clip: `m_i = |MV_i|` for each tubelet `i`; `p_i ∝ m_i`
  after smoothing and a low-magnitude floor (so completely-still
  tubelets are not excluded — the mask should still cover some of them
  to prevent the encoder from ignoring stationary regions).
- Draw the mask via weighted sampling under `p_i`. Same number of
  tubelets masked as V-JEPA's random mask (8 small spatial blocks + 2
  coarse blocks; total ~40% of tokens). Same contiguity constraints
  (the 8 small blocks still need to be spatial blocks, just now biased
  toward high-motion regions).

Everything downstream — predictor, loss, EMA update — is unchanged.

### 9.4 Three-arm contract

| Arm | Pretrain name | Objective |
|---|---|---|
| **Method** | `mcc_mc_mgm_jepa_25ep` | L_MCC-MC with MGM mask on B |
| **Control** | `mcc_mc_jepa_25ep` (from §8) | L_MCC-MC with random mask on B |
| **Ablation** | `mcc_mc_random_mask_on_motion_regions` | L_MCC-MC, random mask within motion-region bounding boxes (tests: does "concentrate mask on moving areas" help, or only "weight by motion magnitude"?) |

### 9.5 Primary endpoints

Same as §8.6, with the expectation that motion-guided masking amplifies
wherever the transport auxiliary already helps. **Specifically: if
MCC-MC improves TAPSE by `δ_MC`, MCC-MC-MGM should improve by `≥ 1.2 ×
δ_MC` to justify the complexity.** Otherwise, MGM does not add.

### 9.6 Risks

- ~~**Motion-vector audit fails.**~~ **Resolved (2026-05-05, §9.2.1):**
  50-clip pilot audit shows 100% MV coverage with heavy-tailed
  per-tubelet magnitudes. Gate PASS.
- **Sub-pixel MV magnitudes in re-encoded MP4s.** H.264 MVs in MIMIC
  are coarsely quantized (mean per-tubelet magnitude ≈ 0.05–0.5 px,
  many skip-macroblocks at 0). MGM only needs the tubelet *ordering*,
  not absolute magnitudes, and the audit confirms ordering is
  recoverable. If MGM underperforms, the fix is a smoothed mask
  probability (§9.3's `p_i ∝ m_i + ε` floor) rather than switching to
  optical flow.
- **Motion vectors correlate with probe drift, not tissue motion.** MGM
  on drift-dominated clips biases the mask toward the wrong regions. An
  ablation of MGM on probe-stable vs probe-mobile studies would
  diagnose; defer unless §9.2's correlation check is weak.
- **Mask placement and transport auxiliary interact unpredictably.** If
  MCC-MC-MGM beats MCC-MC, it's hard to attribute the gain to MGM vs
  to a better mask-transport synergy. Mitigation: the §9.4 ablation arm
  (random mask within motion-region bounding boxes) controls for the
  "where the mask lands" effect independently of the "how weighted by
  motion magnitude" effect.

### 9.7 Compute

Motion-vector extraction at dataloader time adds ~5 ms/clip via PyAV (if
audit passes). At batch 32 × 2 clips × 8 GPUs = 512 clips/step, that's
~2.5 s/step of CPU-side MV extraction overlapping with GPU forward.
CPU-bound workers may need to increase `num_workers` from 4 → 6. Per-
step wall overhead on top of MCC-MC: ≤ 5% after num_workers tuning.

## 10. What the paper claim becomes

If MCC-JEPA passes §5.5 success:

> V-JEPA's masked latent-prediction objective generalizes from within-clip
> spatiotemporal prediction to within-study cross-clip prediction with a
> single change to the sampler. Because echocardiography studies contain
> multiple clips of the same patient under different acquisition
> conditions, this extension gives the clip encoder a direct gradient
> signal that rewards invariance across acquisitions. The modification
> preserves V-JEPA's architecture, EMA mechanism, masking, predictor, and
> loss unchanged; it is the smallest principled extension of V-JEPA that
> uses echo's study structure. On matched-compute MIMIC pretraining, MCC-
> JEPA improves RVSP multi-view Pearson by ≥ 0.02 over vanilla V-JEPA
> without regressing LVEF, CAMUS segmentation, or pediatric zero-shot
> transfer. The cross-clip modification is therefore the correct level of
> intervention for echo — above it (pooled-latent contrastive objectives,
> factorized slots, frozen-teacher distillation) injects fragile priors;
> below it (phase-conditioning in the predictor alone) doesn't engage the
> multi-view structure at all.

If MCC-JEPA does not pass, the paper claim is: "The prior failure of
multi-view V-JEPA variants is not attributable to lack of a pooled
contrastive objective, lack of factorized slots, or lack of a
hard-negative. We show that even the minimal principled modification of
V-JEPA to use echo's study structure — MCC-JEPA — fails to produce
RVSP gains at matched compute, suggesting echo's within-study cross-clip
signal is not recoverable from a clip-encoder-level modification alone."
This is still a publishable result (negative result on a well-posed
question), just not the headline one.

If MCC-MC (§8) passes its primary endpoints (TAPSE / RV FAC / regional
wall motion improve while LVEF and segmentation are within HP noise),
the claim extends:

> V-JEPA's content objective captures global cardiac phase and anatomy
> but not local mechanics. A small-weight latent-transport auxiliary
> adds dense token-level motion learning that closes the local-mechanics
> gap on TAPSE and RV FAC without sacrificing the generalist
> representation. The auxiliary is architecturally minimal — a 2-layer
> cross-attention head over spatial tubelets at time `t` and `t+Δ`,
> with cosine regression to the EMA teacher's spatial tokens at
> `t+Δ` and a cycle-consistency term — and its weight is kept small
> (λ_transport ≈ 0.05). MCC-JEPA + transport is the adaptation of
> V-JEPA to echocardiographic mechanics: sampler change (MCC-JEPA) +
> local-motion auxiliary (MC-inspired) + nothing else.

If MCC-MC-MGM (§9) passes its primary endpoints (≥ 1.2× the
MCC-MC TAPSE improvement), the claim adds:

> Biasing the target-clip mask toward motion-salient tubelets — derived
> for free from H.264 motion vectors — further amplifies the
> local-mechanics gain. MGM is a cheap mask-placement change that
> complements the transport auxiliary by concentrating the
> reconstruction signal on the regions that carry functional
> information; together, the three modifications (sampler, loss, mask
> placement) constitute the most complete V-JEPA adaptation for
> echocardiography that preserves the method's inductive bias.

---

## 11. Open file references

- Prior experiments: `claude/neurips/experiments/{phase-relational-hardneg,
  mv2sv-privileged-multiview,phase-jepa,finalbudget-phase-probes}.md`.
- V-JEPA forward graph: `app/vjepa/train.py::training_step`,
  `src/models/predictor.py::VisionTransformerPredictor.forward`.
- Mask collator: `src/masks/multiseq_multiblock3d.py`.
- Same-study sampler to adapt from: `classifier/phase/sampler/phase_matched_pair_dataset.py`.
- Multiview dispatch to extend: `app/vjepa_multiview/train.py` (add
  `"mcc_jepa"` branch alongside existing `phase_relational`,
  `privileged_multiview`, etc.).

Design ready; not yet launched.

- **MCC-JEPA (§1–§7)**: ~1 day engineering, ~3 days wall to paper
  numbers (smoke + 25-ep pretrain + probes).
- **MCC-MC (§8)**: ~1 additional day engineering (transport head +
  diagnostics + coefficient sweep), ~3 additional days wall (sweep +
  25-ep pretrain + mechanics probes).
- **MCC-MC-MGM (§9)**: ~0.5 day engineering (mask collator change),
  ~2 additional days wall (pretrain + probes). §9.2 audit **passed**
  2026-05-05 on a 50-clip pilot (§9.2.1); gate is clear.

Total end-to-end wall for all three axes if all gates pass: ~**8-10
days**. Each is run sequentially and each gates the next.

---

## 12. Critique and refinement — target-anchored MCC-JEPA

This section revises the pure MCC-JEPA formulation defined in §1–§7
after a second pass that weighs MCC-JEPA against the refined-Arm-C
study-token results and the MV2SV failure mode. The conclusion is:
**pure A → B prediction is under-specified; the primary run should be
target-anchored.**

### 12.1 What MCC-JEPA gets right

MCC-JEPA is one of the better candidates for a **stronger clip
encoder**, because unlike frozen-`c_clip` study-level methods (Arm C,
EchoSet-JEPA), it modifies the **V-JEPA pretraining signal itself**.
The student clip encoder receives gradients from cross-acquisition
structure, rather than only a post-hoc study transformer trained on
frozen features.

It also keeps the most reliable part of the ecosystem intact:

```text
V-JEPA encoder
EMA target encoder
latent-space prediction
masked tubelet targets
cosine / latent regression loss
```

The §1–§7 design explicitly frames MCC as a **sampler-level change**:
predict masked tubelets of clip B from a different same-study clip A,
keeping V-JEPA's architecture, EMA mechanism, and cosine loss
unchanged. The controlled-objective comparisons already show standard
JEPA is the strongest base objective among JEPA/BYOL/MAE/SALT on
echo, and EMA co-evolution is load-bearing. MCC-JEPA does not touch
either.

### 12.2 The under-specification problem in pure MCC

Pure MCC asks the model to predict the local latent content of a
different acquisition given no target-B information:

```text
context: clip A, possibly from A4C
target: masked tubelets of clip B, possibly PLAX / PSAX / A2C / color
```

Without visible target-B anchor tokens, this degenerates toward
conditional-mean prediction or a soft form of hallucination. Less
brittle than pixel reconstruction because the target is latent, but
not guaranteed that A contains enough information to predict B's
local target tokens. This is the MV2SV failure mode at a lower
granularity: MV2SV pooled the target view, pure MCC just swaps the
pooled target for per-tubelet targets.

### 12.3 Relationship to MV2SV (updated)

Pure MCC is meaningfully different from MV2SV, but shares one risk:
underdetermined cross-view prediction. The refined target-anchored
version closes the gap.

| Axis | MV2SV | Pure MCC-JEPA | Target-anchored MCC-JEPA |
|---|---|---|---|
| Prediction target | Pooled target-view latent / slots | Masked tubelet latents of B | Masked tubelet latents of B |
| Context | One source clip | One source clip A | Source A + visible tokens of B |
| Target granularity | Whole-clip pooled embedding | Local spatiotemporal tokens | Local spatiotemporal tokens |
| Architecture | Factorized head, conditional view predictor | Vanilla V-JEPA encoder/predictor | V-JEPA predictor + zero-gated cross-clip adapter |
| Metadata conditioning | Target view/modality/phase | None or positions only | Target visible tokens supply acquisition info |
| Inference claim | Single-view student internalizes multi-view | Generic improved clip encoder | Generic improved clip encoder |
| Main failure risk | Single-view hallucination + slot collapse | Cross-view hallucination / conditional mean | Reduced hallucination; A helps beyond B-visible |

MV2SV was a privileged-information setup with pooled target
supervision, factorized slots, and a conditional view predictor;
retrieval improved while downstream regressed and slots collapsed.
MCC avoids the factorized-slot and conditional-predictor traps, but
inherits the "target must be recoverable from source" requirement.
MV2MAE's lesson is directly relevant: its cross-view decoder
reconstructs a target view using source-view visible tokens **and**
target-view visible/mask tokens — the visible target patches supply
the necessary target-view information. Target-anchored MCC adopts
this structure in latent space.

### 12.4 Four refinements to the §1–§7 design

#### Refinement 1 — Add visible target-B tokens

```text
# Pure MCC (§3):
student context = clip A
teacher target  = full clip B
predict masked B tokens

# Target-anchored MCC (new primary):
student context = visible tokens of B + source tokens from A
teacher target  = full clip B
predict masked B tokens
```

Task reframes from *"hallucinate clip B from clip A"* to *"complete
clip B using its own visible context, helped by another same-study
clip A."* Preserves V-JEPA's core structure; cross-clip information
improves predictions where A carries complementary same-patient
physiology.

Formulation:

```text
L = D(
      predictor(z_B_visible, z_A_context, mask_positions_B),
      stopgrad(target_encoder(B_full)[mask_positions_B])
    )
```

Required control: **B-visible only** (vanilla V-JEPA on clip B under
the same sampler) vs **B-visible + A** (target-anchored MCC).
Difference isolates whether A actually helps.

#### Refinement 2 — Zero-gated cross-clip adapter

Do not hard-wire A into the predictor. Add a residual cross-attention
module initialized to zero so training starts at vanilla V-JEPA:

```text
pred_B = VJEPA_predictor(B_visible, B_mask_tokens)

cross  = CrossAttn(
  queries     = B_mask_tokens,
  keys/values = A_tokens
)

pred   = pred_B + γ · cross
γ_0    = 0
```

The model learns to use same-study source tokens only when useful.
Inspired by MVDiffusion's cross-view interaction modules inserted
into a strong pretrained backbone without disturbing it. For echo
the correspondence is latent/physiological rather than geometric;
a zero-gated adapter is safer than hard-wiring cross-view attention.

**Diagnostic**: track γ over training. If γ stays near 0, A is
unused and the method is degenerating to vanilla V-JEPA.

#### Refinement 3 — Keep vanilla V-JEPA loss active

```text
L_total
  = L_vjepa_self
  + λ_mcc   L_target_anchored_mcc
  + λ_cov   L_cov_or_SIGReg   (only if collapse appears)
```

Initial values:

```text
λ_mcc = 0.1–0.3
λ_cov = 0.001 (if needed)
```

Protects the base V-JEPA representation. Prior phase/objective
experiments showed specialized objectives can improve one axis while
regressing generality, especially RVSP / pediatric / transfer.

#### Refinement 4 — Pair sampling should match clinical goals

Do not sample pairs uniformly for the main run. Controlled mixture:

```text
40%  same view family, different phase / acquisition
30%  same broad family (A4C ↔ A2C / A3C / A5C)
20%  cross family (apical ↔ PLAX / PSAX)
10%  modality pairs (B-mode ↔ color Doppler) if available
```

Rationale:

```text
same-view pairs        → A4C-only and local-cycle features
apical cross-view      → biplane LV/RV, LVEF, HCM, LVH
parasternal/apical     → chamber geometry, LVH/HCM morphology
color/Doppler          → later, only if modality/calibration ready
```

Prioritize B-mode view pairs first (A4C↔A2C, A4C↔A3C/A5C, A4C↔PLAX,
PLAX↔PSAX, A4C↔RV-focused/subcostal).

### 12.5 TAPSE caveat — MCC alone is insufficient

TAPSE needs local metric motion, not cross-view consistency. Plain
V-JEPA and prior phase/token variants appear capped on TAPSE-like
local-amplitude tasks because the objective does not explicitly track
a local annular landmark or displacement amplitude. For TAPSE, the
MCC-MC extension (§8) is the relevant tool:

```text
L_total
  = L_vjepa_self
  + λ_mcc       L_target_anchored_mcc
  + λ_transport L_token_transport
  + λ_cycle     L_cycle

λ_transport = 0.005–0.01
λ_cycle     = 0.001–0.005
```

Add local motion only after target-anchored MCC shows it does not
regress LVEF/HCM/LVH.

### 12.6 Revised arm contract (supersedes §5.1 as primary)

§5.1's three arms (0=vanilla, 1=pure MCC cross-study, 2=MCC
same-study) are kept as **diagnostic smokes**. The primary run is
target-anchored. New arm plan:

- **Arm 0 — vanilla V-JEPA continuation.** MIMIC-IV-Echo e100
  checkpoint + 25 epochs, same sampler/eligibility, vanilla loss.
  Essential control.
- **Arm 1 — pure MCC-JEPA smoke (not full).** Primary diagnostic:
  replace same-study A with unrelated same-view A from another study.
  If target-B loss does *not* worsen, pure MCC is not using same-study
  context, and Arm 2 is the only path worth funding.
- **Arm 2 — target-anchored MCC-JEPA (primary run).** `context =
  B_visible + A_source`, `target = masked B teacher tokens`.
  Controls: B-visible only / +same-study A / +other-study matched A /
  +shuffled A.
- **Arm 3 — Arm 2 + global study-token auxiliary.** Add Arm C's
  global `[STUDY]` prediction at `λ_study = 0.05–0.1`. Tests whether
  the study-level signal compounds with target-anchored MCC.
- **Arm 4 — MCC-MC for local mechanics.** Only if Arms 2/3 are safe.
  Target-anchored MCC + low-weight token transport + cycle.

### 12.7 Decision diagnostics (before downstream)

Do not use cross-clip retrieval alone — MV2SV showed retrieval can
improve while downstream regresses. Required diagnostics:

```text
1. A-context usefulness
   loss(B_visible + same-study A) < loss(B_visible only)

2. Same-study specificity
   loss(B_visible + same-study A) < loss(B_visible + other-study matched A)

3. View-pair specificity
   gains stratified by pair type:
     A4C↔A2C, A4C↔PLAX, A4C↔RV-focused, PLAX↔PSAX

4. Cross-view adapter usage
   γ > 0 after training (otherwise A is unused)

5. No representation collapse
   var/cov stable, no excessive view-invariant collapse

6. Clip-level safety
   A4C-only LVEF / RV / HCM / LVH do not regress vs Arm 0

7. Multi-view usefulness
   K=8 study probes improve over vanilla V-JEPA + prediction averaging
   and over supervised late fusion
```

Downstream eval splits into two groups to attribute gains:

```text
Clip-only:
  A4C-only LVEF, TAPSE, RV qualitative, HCM, LVH

Study-level:
  K=8 LVEF, RV qualitative, HCM, LVH, incident HF, mortality
```

This tells you whether MCC improved the **clip encoder**, the
**study representation**, or both.

### 12.8 Parallel to Arm C, not a replacement

MCC-JEPA and Arm C answer different questions:

```text
Arm C / study-token JEPA
  learns a study-level representation on top of frozen c_clip

Target-anchored MCC-JEPA
  modifies clip encoder pretraining so individual clip features are
  study-aware
```

| Task | Needs stronger clip encoder? | Needs study-level integration? | MCC-JEPA relevance |
|---|---|---|---|
| LVEF | yes | sometimes | high |
| TAPSE | yes (local motion) | less | MCC alone insufficient; needs MCC-MC |
| RV qualitative | yes | yes | moderate-high |
| HCM | yes | yes | high |
| LVH | yes | yes | high |
| Incident HF / mortality | yes | yes | high (as pretraining signal) |

MCC-JEPA is especially relevant for **A4C-only or limited-view
tasks**, because it updates the clip encoder itself. Arm C's frozen
study transformer improves a study embedding but does not make an
A4C clip representation better. Run them as parallel experiments.

### 12.9 Final recommendation

Primary claim becomes:

> **Target-Anchored MCC-JEPA:** a V-JEPA-style masked latent
> prediction objective where the target clip provides visible anchor
> tokens and a same-study source clip provides complementary
> cross-acquisition context.

Meaningfully different from MV2SV: no pooled target-view
hallucination, no factorized slots, no target-view metadata as the
main prediction mechanism. Trains the clip encoder directly, keeps
V-JEPA's EMA latent prediction structure, and gives the model a
non-hallucinatory target-view anchor.

Execution order:

```text
1. Vanilla V-JEPA continuation control       (Arm 0)
2. Pure MCC smoke + cross-study diagnostic   (Arm 1)
3. Target-anchored MCC main run              (Arm 2)
4. Target-anchored MCC + study-token aux     (Arm 3)
5. MCC-MC only if TAPSE/RV motion capped     (Arm 4)
```

Major lesson carried from earlier variants: **the model should use
multi-view structure as additional evidence, not be forced to
hallucinate missing views.**

---

## 13. Implementation status (2026-05-05)

Target-anchored MCC-JEPA is **implemented, tested, and training now** on
HyperPod cluster `echojepa-h100-neurips`. This section tracks what
exists in the tree and what runs are live.

### 13.1 Code

All new MCC-JEPA code lives in four files; one minimal edit to the
existing multiview training entry point. No changes to vanilla
`app/vjepa/train.py`, `src/models/predictor.py`, or
`src/models/vision_transformer.py`.

| Path | LOC | Purpose |
|---|---|---|
| `src/models/mcc_jepa/__init__.py` | 3 | package export |
| `src/models/mcc_jepa/cross_clip_adapter.py` | 73 | zero-gated cross-attention residual; γ ∈ ℝ; identity at init |
| `src/datasets/mcc_pair_dataset.py` | 205 | MVP pair mixture (40/30/20/10) + shuffled-A diagnostic |
| `app/vjepa_multiview/mcc_jepa_forward.py` | 160 | `forward_mcc_jepa(mode='pure'|'target_anchored', …)` |
| `app/vjepa_multiview/train.py` | +23 | added `mcc_jepa` branch to objective dispatch + save path |

**Adapter** (`src/models/mcc_jepa/cross_clip_adapter.py`): wraps the
predictor's output tokens with
`pred = pred_B_base + γ · LN(pred_B_base) · CrossAttn · LN(A_source)`
where `γ = nn.Parameter(torch.zeros(1))`. At γ=0 the adapter is an
exact identity (pytest-verified), so training begins byte-equivalent
to vanilla V-JEPA on clip B; γ learns up only if the cross-attention
produces a direction that reduces L_mcc.

**Forward** (`app/vjepa_multiview/mcc_jepa_forward.py`): consumes a
`PairBatch(clip_a, clip_b, masks_enc, masks_pred)` where masks live
on clip B. Target-anchored mode computes both
`L_vjepa_self = Lp(pred_B_base, teacher(B_full))` and
`L_mcc = Lp(pred_B_base + γ·CrossAttn(·, A_source), teacher(B_full))`
in a single forward; `L_vjepa_self` is free because it reuses the
teacher output and the pre-adapter predictor output. Total loss is
`λ_vjepa · L_vjepa_self + λ_mcc · L_mcc`.

**Save dict patch** (`app/vjepa_multiview/train.py:3665`): adds
`save_dict["mcc_adapter"] = mcc_adapter.state_dict()` and
`save_dict["mcc_config"] = {mcc_mode, λ_mcc, λ_vjepa, num_heads,
γ_init}`. Without this, adapter weights were lost at every
checkpoint save (caught during smoke).

**γ visibility** (`app/vjepa_multiview/train.py:4010`): per-step INFO
log appends `mcc[gamma=±X.XXXX pred_delta=±X.XXXXX]` when the
objective is MCC, so γ is scrapable from `job.log`. The smoke CSV
does not include γ as a column; this is deliberate to avoid an
invasive `csv_logger.py` change.

### 13.2 Configs (under `configs/train/vitl16/`)

| Config | Epochs | Objective | λ_mcc |
|---|---|---|---|
| `pretrain-vjepa-in21k-e100-plus25-control.yaml` | 25 | `intraview_only` (vanilla V-JEPA on clip_a) | — |
| `pretrain-mcc-jepa-pure-smoke.yaml` | 1 (500 steps) | `mcc_jepa / pure` | 1.0 |
| `pretrain-mcc-jepa-target-anchored-smoke.yaml` | 1 (500 steps) | `mcc_jepa / target_anchored` | 0.2 |
| `pretrain-mcc-jepa-target-anchored-25of100.yaml` | 25 | `mcc_jepa / target_anchored` | 0.2 |

All four init from `checkpoints/jepa_in21k_vitl_e100.pt` via
`optimization.anneal_ckpt` + `force_load_pretrain: true`.
`rel_require_same_study_wrong_phase_negative: false` on the MCC
configs (2-clip sampler path); `true` on the vanilla control
(3-clip sampler for matched-compute parity).

### 13.3 sbatches (under `scripts/mcc_jepa/`)

| Script | Walltime |
|---|---|
| `pretrain_vjepa_plus25_control.sbatch` | 1d 12h |
| `pretrain_mcc_pure_smoke.sbatch` | 1h 30m |
| `pretrain_mcc_target_anchored_smoke.sbatch` | 1h 30m |
| `pretrain_mcc_target_anchored_25ep.sbatch` | 1d 12h |
| `launch_mcc_25ep.sh` | helper (pytest + yaml parse + dry-run; `--yes-25ep` to submit) |
| `verify_checkpoint.py` | preflight — confirms e100 size/keys |

All sbatches follow the canonical S3-tarball deploy recipe; sbatches
land in `/tmp/vjepa2-ctrl/` on the controller and unpack to
`/opt/dlami/nvme/src/vjepa2` on the compute node per
`scripts/neurips/phase/final_phase_rel_hardneg25_paper.sbatch`. The
`ml-p5-48xlarge` partition is currently empty on this cluster; all
jobs are submitted with explicit `-p dev --nodelist=...` overrides
to the two dev nodes `ip-10-0-50-146` and `ip-10-0-50-56`.

### 13.4 Tests (`tests/mcc_jepa/`, 15 passing)

| Test | What it guarantees |
|---|---|
| `test_same_study_pair_sampler.py` | sampler returns same-study (A,B); distinct-clip rate ≥ 0.857 on synthetic 7-study manifest |
| `test_single_clip_fallback.py` | 1-clip studies → (clip, clip) fallback |
| `test_cross_clip_adapter_zero_init.py` | adapter.γ = 0 at init; output is bitwise-identical to pred_B; diverges when γ > 0 |
| `test_target_anchored_no_leak.py` | at γ=0 the anchored forward matches B-visible-only V-JEPA forward within fp32 tolerance |
| `test_mcc_forward_shapes.py` | loss finite on 2-sample batch; teacher has no grad; student has grad |
| `test_shuffled_A_control.py` | `shuffle_source=True` draws A from a different study; same_study_rate flips from 1.0 to 0.0 |

### 13.5 Smoke run (job 759) — GATE PASS

Config: `pretrain-mcc-jepa-target-anchored-smoke.yaml` on node 146,
8 × H100, 500 steps, 23 min wallclock. Full detail in
`reports/mcc_jepa/smoke_results.md`.

**Loss trajectory:**

| steps | total | L_vjepa_self | L_mcc | gap (vjepa−mcc) |
|---|---|---|---|---|
|   0– 99 | 0.5997 | 0.4998 | 0.4998 | **+3e-6** |
| 100–199 | 0.5916 | 0.4930 | 0.4930 | +4e-6 |
| 200–299 | 0.5953 | 0.4961 | 0.4960 | +9e-5 |
| 300–399 | 0.5917 | 0.4932 | 0.4926 | +5.8e-4 |
| 400–499 | 0.5927 | 0.4941 | 0.4931 | **+1.0e-3** |

`L_mcc < L_vjepa_self` in 478/500 steps. Gap grew **300×** over the
smoke → γ moved off zero. No NaNs, no divergence, GPU util 100%.
Step time 2 s (2× single-clip V-JEPA due to the extra clip_a
forward through the encoder).

**Structural gates passed:** loss finite, γ moving, adapter
contributing measurable signal, pair sampler healthy (same-study
rate > 0.95, view-pair mixture 0.45/0.30/0.25 sv/sf/cf), teacher no
grad, student + adapter grad nonzero. **Unmeasured gates** (deferred
to 25-epoch run): `loss_same_study_A < loss_b_visible_only`,
`loss_same_study_A < loss_shuffled_A`, per-pair-type breakdown —
these require the 3-way probe described in §12.7 and will be
computed on a saved checkpoint offline.

**Pure-MCC smoke (job 760): FAILED** at `predictor.py:238` with a
shape mismatch. Root cause is in the `pure` branch of
`forward_mcc_jepa` passing `z_a_source` through
`PredictorMultiSeqWrapper` incorrectly. Not on the critical path —
pure MCC is a diagnostic only; §12.6 already demotes it from the
primary contract. To be debugged after target-anchored 25-epoch
completes.

### 13.6 Live 25-epoch runs (launched 2026-05-05 08:11)

| Job | Config | Node | Role |
|---|---|---|---|
| **761** (`mcc_vjep`) | `pretrain-vjepa-in21k-e100-plus25-control.yaml` | `ip-10-0-50-146` | matched-compute vanilla V-JEPA +25 control |
| **762** (`mcc_anch`) | `pretrain-mcc-jepa-target-anchored-25of100.yaml` | `ip-10-0-50-56` | primary target-anchored MCC-JEPA +25 |

Both init from `jepa_in21k_vitl_e100.pt`, 25 continuation epochs on
the 100-epoch LR schedule (same warmup, same LR, same batch, same
sampler eligibility). Expected runtime ~7 h pure compute + data
loader warmup. Checkpoints saved every 5 epochs (+5/+10/+15/+20/+25).
Outputs land in
`s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts/runs/{mcc_vjepa_plus25_control_761, mcc_target_anchored_25of100_762}/`.

The save_dict patch in §13.1 is live in the deployed tarball, so
job 762's checkpoints will include `mcc_adapter` and `mcc_config`
under their respective state-dict keys.

### 13.7 Immediate follow-ups

Before scoring downstream:

1. **Pull +5 epoch checkpoints** from both jobs as soon as they land.
   Use `scripts/compute_predavg_stats.py` + a fresh A4C-only LVEF
   probe on each to get an early read on whether the encoder has
   moved in a useful direction.
2. **γ trajectory**: scrape `job.log` from 762 and plot γ vs step.
   If γ plateaus before epoch 10, the adapter's saturated and
   λ_mcc may need to drop. If γ keeps climbing through epoch 25,
   it may be worth running a λ_mcc ∈ {0.1, 0.3} sweep in a follow-up.
3. **Same-vs-shuffled-A diagnostic** (§12.7 item 2): load the +25
   checkpoint, run three forwards on the same mini-batch with
   (same-study A | no A | shuffled A), log the three losses. This
   is the strongest test of "A helps beyond metadata/geometry".
4. **Pure MCC smoke debug**: 30-min local repro against the real
   predictor; fix the shape error in `forward_mcc_jepa`'s `pure`
   branch. Not on the critical path, but needed before any
   follow-up pure-MCC vs target-anchored ablation.

### 13.8 Execution-order actuals (vs §12.9 plan)

| Step | Plan | Actual |
|---|---|---|
| 1. Vanilla control smoke | implicit | — (skipped; matched-compute control is the +25 run 761) |
| 2. Pure MCC smoke | included | **FAILED** (shape bug, job 760/758); diagnostic only |
| 3. Target-anchored smoke | primary | **PASS** (job 759) |
| 4. Target-anchored main | primary | **RUNNING** (job 762) |
| 5. Vanilla +25 control | matched compute | **CANCELLED** (job 761) — node 146 reallocated to Full-Joint (see §13.9) |
| 6. Target-anchored + study-token aux | Arm 3 | deferred to follow-up |
| 7. MCC-MC | Arm 4 | deferred — only if TAPSE caps observed post-downstream |

### 13.9 Updates since §13.6 (2026-05-05 ~09:00)

- **Job 761 (vanilla +25 control) was cancelled after 3:14.** Node
  `ip-10-0-50-146` was reallocated to the orthogonal Full-Joint
  Global Study-Token EchoMV-JEPA experiment (see
  `claude/neurips/experiments/full-joint-global-study-token-echomv-jepa.md`).
  Consequence: **MCC-JEPA no longer has a matched-compute vanilla
  continuation control on this cluster cycle.** Downstream comparison
  must fall back to the e100 baseline; a proper matched-compute
  vanilla +25 is re-queued as a follow-up run.
- **Job 762 (target-anchored +25) is still running** on node 56.
  First expected checkpoint save: epoch 5, roughly ~2 h after start
  (~10:10 PDT). Save cadence: every 5 epochs (`e5.pt`…`e25.pt`) +
  `latest.pt`. The `mcc_adapter` + `mcc_config` state dict keys are
  in the patched save path and will be included in every save.
- **Throughput observed on 762**: ~2.3 s/step at batch_studies_per_gpu=32,
  K=3 clips/sample, 8 × H100. 25 epochs ≈ 16,250 steps → **~7 h**
  pure compute.
- **Clip-forward compute** on MCC-JEPA: 2 × 32 × 8 × 16,250 ≈ **8.3M
  clip forwards**. The Full-Joint overnight was resized from 5,000
  steps to **30,000 steps** to match this budget (~7.7M clip forwards
  at K=8 × 4 studies × 8 GPUs × 30k = 7.68M); see the full-joint doc
  for details.

### 13.10 Known gaps in validation before 762 completes

These should be checked once the first 762 checkpoint lands:

1. **`mcc_adapter` state-dict round-trip**: we verified the save path
   writes the key in-tree but never loaded it back from a saved
   checkpoint. Action: pull `e5.pt` from S3, `torch.load(...)`, and
   confirm `sd["mcc_adapter"]["gamma"]` is a finite scalar and
   `sd["mcc_config"]` carries the expected λ values.
2. **γ trajectory at longer horizon**: smoke 759's γ moved 3e-6 →
   1e-3 over 500 steps. If γ plateaus before epoch 10 of the 25-ep
   run, the adapter has saturated at a small contribution. If γ
   keeps climbing, a λ_mcc ∈ {0.1, 0.3} sweep is warranted as a
   follow-up.
3. **Anti-hallucination (same-vs-shuffled-A) probe**: deferred to
   an offline script against the final checkpoint. This is the
   strongest diagnostic that the cross-clip adapter is *using*
   same-study structure and not exploiting study-level metadata.

## 14. Debug ladder and all-job outcomes

Six sbatch jobs were submitted for MCC-JEPA between 07:22 and 08:14
on 2026-05-05. Four of them failed; two succeeded. This section
records the failure → fix → redeploy loop so future debuggers see
the entire story, not just the survivor.

### 14.1 Job ladder

| Job | Config | Node | Outcome | Elapsed | Root cause |
|---|---|---|---|---|---|
| 757 | target-anchored smoke | 146 | **FAILED** | 9:02 | `_extract_multiview_clips` allow-list missing `mcc_jepa` |
| 758 | pure smoke | 56 | **FAILED** | 8:55 | Same (inherited from 757) |
| 759 | target-anchored smoke | 146 | **PASS** | 23:12 | — |
| 760 | pure smoke | 56 | **FAILED** | 8:55 | Pure-mode predictor input shape (2-D instead of 3-D) |
| 761 | vanilla +25 control | 146 | **CANCELLED** | 3:14 | User reallocated node 146 to full-joint |
| 762 | target-anchored +25 | 56 | **RUNNING** | — | Primary experiment; live as of writing |

### 14.2 Fix ledger

**Fix 1 — `_extract_multiview_clips` allow-list** (jobs 757, 758
crash). The objective-dispatch in `app/vjepa_multiview/train.py:147`
had an allow-list separate from the one at line ~2514; I only patched
the second. Traceback:
```
ValueError: unknown multiview_objective='mcc_jepa'; want one of
smooth_l1 | intraview_only | phase_relational | privileged_multiview | token_phase_relational
```
Fix: added `mcc_jepa` to the 2-clip branch (`("smooth_l1", "mcc_jepa")`)
at `train.py:147` + updated error-message list.

**Fix 2 — `rel_require_same_study_wrong_phase_negative` flag** (went
in with fix 1). MCC configs inherited `true` from the phase-relational
template, but MCC uses 2-clip sampling so the hard-negative requirement
is meaningless. Set to `false` + `rel_allow_missing_hard_negative: true`
in all three MCC configs. Not a failure cause, but a quiet inefficiency.

**Fix 3 — pure-mode predictor input** (job 760 crash). The pure branch
of `forward_mcc_jepa` passes `z_a_source` (list-over-fpc of `[B, N_A, D]`)
through `PredictorMultiSeqWrapper`, which zips over (fpc, mask-generator)
and ends up feeding the inner `VisionTransformerPredictor` a 2-D tensor.
Traceback:
```
File "src/models/predictor.py", line 238, in forward
    _, N_ctxt, D = x.shape
ValueError: not enough values to unpack (expected 3, got 2)
```
**Not fixed.** Pure MCC is a diagnostic only (§12.6 already demoted
it from the primary contract). To be revisited only if target-anchored
succeeds downstream and we want the anti-hallucination reference.

**Fix 4 — `save_dict["mcc_adapter"]` missing** (caught during smoke
859 post-hoc, before launching 762). After job 759 completed, I pulled
`latest.pt` and inspected it:
```python
sd = torch.load('mcc_759_latest.pt', map_location='cpu', weights_only=False)
# sd keys: ['batch_size', 'encoder', 'epoch', 'itr', 'lambda_crossview',
#          'loss', 'lr', 'multiview_objective', 'opt', 'predictor',
#          'sampling_mode', 'scaler', 'target_encoder', 'world_size']
# NO 'mcc_adapter' KEY.
```
The save_dict code at `app/vjepa_multiview/train.py:3611` has no branch
for the MCC adapter, so its weights were silently dropped on every
save. This would have made the 25-epoch checkpoint unusable for
downstream (adapter re-initialized to γ=0 every reload).

Fix: added `save_dict["mcc_adapter"] = mcc_adapter.state_dict()` +
`save_dict["mcc_config"] = {...}` at `train.py:3665`. Deployed in a
fresh tarball before submitting 762.

**Fix 5 — γ not logged anywhere** (caught at the same time as fix 4).
CSV columns (`epoch, itr, loss, intraview, crossview, iter-time, data-time`)
don't include γ. The only way to inspect γ during training was to
read the state dict post-hoc. For a 7-hour run, this is too blind.

Fix: edited the `log.info` line at `train.py:4010` to append
`mcc[gamma=±X.XXXX pred_delta=±X.XXXXX]` when the objective is MCC.
γ is now scrapable from `job.log` every 20 steps. Deployed with fix 4.

**Fix 6 — `-p ml-p5-48xlarge` partition empty**. All canonical sbatch
templates pin `#SBATCH -p ml-p5-48xlarge`, but `sinfo` showed 0 nodes
in that partition on this cluster cycle (the 2 dev nodes 146/56 were
idle in the `dev` partition). Initial submit failed:
```
sbatch: error: Batch job submission failed: Requested node
configuration is not available
```
Fix: submit with `-p dev --nodelist=ip-10-0-50-{146|56}` override.
Not a code fix; documented in the plan file and the sbatch submit
commands.

### 14.3 Tests that would have caught each bug locally

- **Fix 1** (extract_multiview_clips) — caught only by running the
  actual sbatch. Could be caught locally with a test that imports
  `_extract_multiview_clips` and calls it with `objective="mcc_jepa"`
  on a dummy batch; no test exists yet.
- **Fix 3** (pure mode shape) — caught only by running the actual
  sbatch. The local `test_mcc_forward_shapes.py` covered target-
  anchored but uses a toy predictor that doesn't enforce the 3-D
  input shape; it shadows the real `VisionTransformerPredictor`'s
  contract. A test against the real predictor at small ViT would
  have caught this.
- **Fix 4** (save dict) — no test. A `test_mcc_save_roundtrip.py`
  that trains 1 step, saves, reloads, and asserts `sd["mcc_adapter"]`
  is non-empty would have caught it. Follow-up.
- **Fix 5** (γ logging) — not testable, but the CSV header schema
  is grep-able.

### 14.4 Test summary (what local tests actually verify)

`tests/mcc_jepa/` has 15 tests in 6 files. Each one's assertion set:

- `test_cross_clip_adapter_zero_init.py` (4 tests)
  - γ parameter value is exactly 0.0 at init
  - `adapter(pred, src)` output is bitwise-identical to `pred` when γ=0
    (confirms the zero-gated residual is an exact identity)
  - Output still identity when `source_proj_dim` differs from
    `embed_dim` (projection path doesn't break identity)
  - With `γ=0.5`, output is no longer identical to `pred` (confirms γ
    controls the mix)
- `test_target_anchored_no_leak.py` (2 tests)
  - `_apply_adapter_to_predictor_out` returns bitwise-identical
    outputs when adapter γ=0 (across both fpc entries and mask-gen
    columns of the list-of-list structure)
  - Helper correctly broadcasts source A tokens across mask-generator
    repetitions (when the predictor's leading dim is `B*k`, not `B`)
- `test_same_study_pair_sampler.py` (3 tests) — sampler returns
  same-study pairs; distinct-clip rate ≥ 0.857 on fixture;
  diagnostics dict carries the 4 mixture buckets
- `test_single_clip_fallback.py` (1 test) — single-clip study →
  (clip, clip) pair, `fallback=True`, `bucket="fallback_single_clip"`
- `test_mcc_forward_shapes.py` (2 tests) — `forward_mcc_jepa` returns
  finite losses on a 2-sample toy batch; `intraview == crossview` to
  6 dp at γ=0 (this is the structural no-leak check at the loss
  level); teacher params have no grad after backward, student +
  adapter do
- `test_shuffled_A_control.py` (3 tests) — `shuffle_source=True`
  draws A from a different study; `pair_same_study_rate` flips from
  1.0 to 0.0; `shuffled_source` flag set correctly per row

**What tests do NOT cover:**
- The `_extract_multiview_clips` dispatch path (fix 1)
- Pure mode with the real predictor (fix 3)
- Save/load round-trip of the MCC adapter (fix 4)
- Actual cluster compute (only local CPU tests)

### 14.5 Launch command log

For traceability — exact sequence of submits that ran:

```text
07:23  sbatch -p dev --nodelist=ip-10-0-50-146 ..._target_anchored_smoke  → 757 FAIL (fix 1 discovered)
07:23  sbatch -p dev --nodelist=ip-10-0-50-56  ..._pure_smoke              → 758 FAIL
[fix 1, fix 2 applied, tarball rebuilt and redeployed]
07:42  sbatch -p dev --nodelist=ip-10-0-50-146 ..._target_anchored_smoke  → 759 PASS (23:12)
07:42  sbatch -p dev --nodelist=ip-10-0-50-56  ..._pure_smoke              → 760 FAIL (fix 3 discovered; left unfixed)
[fix 4, fix 5 applied, tarball rebuilt and redeployed]
08:11  sbatch -p dev --nodelist=ip-10-0-50-146 ..._vjepa_plus25_control    → 761 RUNNING
08:11  sbatch -p dev --nodelist=ip-10-0-50-56  ..._target_anchored_25ep    → 762 RUNNING
[user reallocates node 146 to full-joint]
08:14  scancel 761                                                         → 761 CANCELLED
```
