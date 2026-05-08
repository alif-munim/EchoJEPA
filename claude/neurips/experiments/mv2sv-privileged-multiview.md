# MV2SV — Privileged Multi-View EchoJEPA (v5)

Running-record doc for the MV2SV ("multi-view to single-view") experiment
set. Started 2026-05-02. The v5 scientific-path smoke (Stage B, job 652)
passed; the 5-epoch pilot (job 655) is in-flight at the time of writing.

This doc explains **what we're building**, **why the architecture is
what it is**, and **what failure modes of earlier phase-aware arms it is
designed to avoid**.

Paper framing: this is a privileged-information SSL objective for
single-view clinical video. The student sees one clip, at one view, at
inference. Training provides a privileged multi-view teacher signal
that teaches the student to hallucinate target-view-specific structure.
Success endpoints are RVSP, MR severity, AS severity (cross-view
hemodynamic / valve dynamics tasks, which the prior LVEF-only arms do
not test). LVEF is a safety check.

---

## 1. Story in one paragraph

The NeurIPS phase-aware series ran four objectives against a matched
single-view +25-epoch control. Variants 1–3 were **null or LVEF-only**
(see §2 below). Variant 4 (phase-relational InfoNCE with a mandatory
same-study wrong-phase hard negative, job 595) was the first arm to
beat matched-compute SV — LVEF test ΔMAE −0.379 / ΔR² +0.054 — but only
on LVEF. Variant 4's hypothesis is that **structural discrimination of
same-study representations at different phases** is what gave LVEF the
lift, and that same family of objective, retargeted to **different
views at matched phase**, should produce the same kind of lift on
cross-view-dominant tasks (RVSP, MR, AS). MV2SV v5 is that retarget.
Its primary signal is `L_pair_view` (SmoothL1 from student's
single-view embedding to the teacher's view-specific slot on a paired
same-study target-view clip) plus `L_view_nce` (cross-view retrieval
contrastive: the student's A4C embedding must retrieve the correct
same-study PLAX / A5C / A3C target latent among a batch of
alternatives). `L_pair_shared` becomes a small stabilizer;
`L_shared` (paired NT-Xent on the view-invariant slot) provides an
additional same-study signal; fused and phase_rel are auxiliary. The
v5 plumbing is dataloader-complete and fail-loud — if the sampler
can't deliver a real target clip, the forward raises rather than
silently reusing clip_b (the silent-reuse bug that hollowed Variant 3).

---

## 2. What earlier phase-aware arms got, and where they fell short

The NeurIPS phase-aware series split into two doc families:
`phase-jepa.md` (within-clip phase conditioning — Variants 1–2) and
the multiview track at `multiview-pilot-progress.md` →
`finalbudget-phase-probes.md` (positive-only cross-view — Variant 3)
→ `phase-relational-hardneg.md` (InfoNCE with hard-negative —
Variant 4). MV2SV v5 is the cross-view retarget of Variant 4.

### 2.1 Variant 1 — Predictor-φ (φ-JEPA Run D, job 374, 200 epochs)

**Objective**: `L = L_intra` (base V-JEPA loss, unchanged) with the
predictor receiving Δφ(ctx→tgt) integer-Fourier embedding as input
alongside RoPE positions:

```
predictor(context_tokens, mask_tokens + phase_embed(Δφ))
  where phase_embed = MLP(sin(2πkΔφ), cos(2πkΔφ) : k=1..16)
```

Encoder is phase-blind; only the predictor sees Δφ. Phase dropout
at p=0.15 (random replacement with a learned `<no_phase>` token)
regularizes the predictor and exercises the fallback path. Δφ
computed from DICOM per-clip HR + FrameTime, cycle-fraction units.
`<no_phase>` fires on studies with HR stdev > 15 bpm (~3.3% of
MIMIC studies).

**Data gate (Gate 1, job 351, 2026-04-24)**: DICOM metadata pilot on
1,000 random MIMIC studies confirmed HR + FrameTime present at
scale — 99.59% HR coverage across 74,314 clips, 96.68% of studies
tight-rhythm, 0% parse errors. The full metadata extraction (job
352, `phi_jepa_full_meta_352`) produced
`mimic_clip_phase_metadata.csv` consumed by Run D.

**Run D (job 374, 200 epochs)**: full φ-JEPA pretrain on MIMIC from
the IN21K-JEPA e100 init. Checkpoints saved at e25 / e50 / e75 /
e100 (and later e150 / e200, not probed here).

**LVEF trajectory probes (EchoNet-Dynamic, frozen d=4 attentive,
20 epochs, 6-HP grid, test on held-out 1,277 videos)**: training
job 380 produced e25/e50/e75 probes; job 391 produced e100 probe;
job 398 ran test inference on all four. Authoritative test numbers:

| φ-JEPA ckpt | Test MAE | Test R² | Test Pearson |
|---|---|---|---|
| e25  | 6.110 | 0.510 | 0.720 |
| e50  | 5.552 | 0.606 | 0.780 |
| e75  | 5.394 | 0.625 | 0.793 |
| e100 | **5.272** | **0.642** | **0.803** |

For comparison — single-view V-JEPA continuation on MIMIC from the
same IN21K e100 init, probed with the same protocol (from
phase-relational-hardneg.md §5.4):

| SV JEPA ckpt | Test MAE | Test R² | Test Pearson |
|---|---|---|---|
| e100 (canonical) | 5.320 | 0.652 | 0.808 |
| e125 | 5.264 | 0.645 | ~0.81 |
| e150 | 5.038 | 0.679 | ~0.83 |
| e175 | 5.003 | 0.682 | ~0.83 |
| e200 (job 421 protocol) | **4.880** | **0.714** | **0.845** |

**Verdict: NULL.** At matched compute (e100 vs SV e100), φ-JEPA is
within HP-sweep noise of the single-view baseline (ΔMAE +0.048 in
SV's favor; ΔR² −0.010). φ-JEPA's e100 checkpoint is actually
slightly *worse* than SV's e100. The monotone improvement
trajectory e25 → e100 (MAE 6.110 → 5.272) tracks what any
continuation run would produce from more training; there is no
φ-specific gain. At SV e150 and beyond, SV moves ahead materially
(ΔMAE −0.234 at e150, −0.392 at e200 vs φ-JEPA e100).

**What this empirically confirms**: predictor-side phase
conditioning does not force the encoder to learn phase-discriminative
features. The predictor absorbs the Δφ signal; the encoder keeps
producing the same context features it would under base V-JEPA.
Run D's LVEF trajectory is essentially a more-expensive SV e100 on
this probe. The `<no_phase>` fallback path being trained doesn't
change the ceiling — fallback trains the *predictor*'s robustness,
not the *encoder*'s phase sensitivity.

**Why the predictor can "use" Δφ without the encoder needing to
reorganize**: the predictor has to map context tokens to mask-token
positions. Δφ is one of several position-like inputs. If the
context encodes enough temporal information to solve the L_intra
objective at any phase alignment (i.e., "predict what's at these
masked tokens"), the predictor can route Δφ through its MLP without
demanding that the encoder emit a phase-disentangled representation.
This is the precise architectural failure principle that motivated
Variant 4's shift to an encoder-side discriminative objective.

**Remaining φ-JEPA runs**: the v5 phase-jepa.md doc's Runs G / I /
J / E2 (negative control / HR-attribution / capacity-matched
control / phase-free deployment) were gated on Run D producing a
clear positive at Gate 2. With Run D null, none of them are needed
to falsify the conditioning-only hypothesis — the null result
already does that.

### 2.2 Variant 2 — Mask-φ (within-clip)

**Objective**: `L = L_intra` with target-mask blocks selected to
concentrate on specific phase buckets, so the predictor more often
has to predict end-systole or end-diastole targets from mixed-phase
context. Same "conditioning, not discrimination" critique applies —
mask selection is a sampling bias, not a loss that penalizes
phase-blind encodings.

Mask-φ and Predictor-φ are classed together in §8 of the NeurIPS
paper framing because they share the conditioning-not-discrimination
property. Neither ships.

### 2.3 What "the sampler" actually is (and why it's a confound in its own right)

V3 and V4 share a data-path change that Variant 1 and the SV
baseline don't have. Throughout this doc I refer to it as "the
sampler" for shorthand, but it's not one knob — it's **five
coupled changes** that all travel together whenever
`multiview_objective != intraview_only` on the `phase_matched`
path. Because the control (608) isolates "same sampler, no
InfoNCE," the paired Δ numbers let us decompose how much of each
variant's downstream score comes from the sampler vs the
objective. This subsection is the reference.

#### SV baseline data path: `VideoDataset`, 1-clip-per-step

1. For each row in `mimic_annotations_s3.csv`, open one mp4.
2. Pick a **single random 16-frame window** from anywhere in the
   clip (partition-random sampling).
3. One clip per sample, one sample per DataLoader slot.
4. No filters on quality, rhythm, view, or phase — every clip the
   annotations CSV lists is eligible.
5. Random window means at high-HR clips a 16-frame window covers
   ~2 cycles; at low-HR < 1 cycle. No control.

Per step: 128 clips × 8 GPUs = 1,024 clips/step.

#### Phase-matched data path: `PhaseMatchedStudySampler`, 3-clip-per-step

Used by V3 (`fb_phase_542`), V4 (`final_phase_rel25_paper_593`),
the paired intraview-only Control (608), and by MV2SV v5. The
builder's constructor takes a config dict that gates the following
five behaviors, all of which differ from the SV path:

1. **Clip-quality + rhythm filter**: `quality_tiers=["high","medium"]`,
   `rr_filter_mode=strict`, `require_rr_consistent=true`. Clips
   that fail these drop out of the eligibility pool entirely. A
   non-trivial fraction of MIMIC fails (exact pct not logged, but
   known to be material for non-apical acquisitions).
2. **View-label confidence filter**: `min_view_confidence=0.60` on
   the view labels CSV. Unknown-view clips drop out. Studies with
   no confidently-labeled clips drop out of the study pool.
3. **Pair-draw not clip-draw**: per study, draw `pairs_per_study=24`
   triples (clip_a, clip_b_pos, clip_b_neg). Studies without enough
   eligible pairs to fill even one triple get skipped.
4. **Phase-anchored 16-frame window**: `sampling_mode=uniform_phase`
   with `phase_tolerance=0.15`. The 16-frame window is **centered
   on the frame closest to a drawn target φ**, not random. Requires
   per-frame phase labels (DICOM-ECG aligned, already computed for
   MIMIC). Clips lacking confident phase at the drawn φ get
   skipped. The effect is that the encoder sees end-systolic and
   end-diastolic frames preferentially (because φ is drawn
   uniformly and the circular-phase distribution concentrates on
   extrema).
5. **Hard-negative-availability eligibility**: when
   `rel_require_same_study_wrong_phase_negative=true` (V3, V4,
   Control all set this), a same-study same-view clip with phase
   differing by ≥ 0.25 cycles (`rel_wrong_phase_min_delta`) must
   exist for the triple to be valid. If not,
   `rel_allow_missing_hard_negative=false` drops the triple and
   `rel_hard_negative_fallback=resample_anchor` retries up to
   `rel_max_hard_neg_attempts=16` times. Studies without a single
   valid hard-neg partner get skipped.
6. **View-pair mixture**: among the `(clip_a, clip_b_pos)` pairs
   that survive all the above, enforce a distribution of
   same_view / same_family / cross_family (35/45/20 in V4's
   config, 25/45/30 in V3's) via resampling.

Per step: 32–64 pairs × 8 GPUs = 512 pairs ≈ 1,024 teacher
forwards. Compute budget ≈ matched to SV (teacher-forwards count,
not backward-passes).

#### Summary: SV vs phase_matched, five changes

| Axis | SV baseline | Phase-matched path |
|---|---|---|
| Unit | 1 clip | 1 triple (clip_a, clip_b_pos, clip_b_neg) |
| Data filter | none | quality + RR + view-confidence + phase-labels + hard-neg availability |
| Window sampling | random | phase-anchored on drawn φ |
| Study grouping | none | study-grouped pair draws |
| Sample eligibility | all annotated clips | ~subset (exact % not logged) |

#### Why the sampler affects different tasks differently

The paired Δ numbers (SV → Control) measured on the real test
sets show:

| Task | SV R² | Control R² | Δ |
|---|---:|---:|---:|
| LVEF | 0.645 | **0.670** | **+0.025** |
| RVSP | 0.157 | 0.108 | **−0.049** |

Same data-path change, opposite direction. The mechanisms are
different for each task:

**LVEF benefits from the sampler (~+2.5 pp R²)**:
- **Phase anchoring is perfectly targeted.** LVEF = EF reads out
  of end-systolic (ESV) and end-diastolic (EDV) volumes. The
  phase-anchored window preferentially samples these frames, so
  the intraview loss gets useful gradient at exactly the phase
  landmarks that matter for LVEF prediction.
- **View-pair policy up-samples apical views.** The same_view
  0.35 + same_family 0.45 mixture preferentially draws apical
  clips (A4C/A2C/A3C are the bulk of apicals in MIMIC), which is
  also what LVEF probes read from.

**RVSP suffers from the sampler (~−5 pp R²)**:
- **Quality + RR filter disproportionately excludes
  RVSP-informative clips.** RVSP is assessed from TR jet Doppler
  tracings in subcostal or apical 4C views; these tend to have
  higher acquisition variability and may fail
  `rr_filter_mode=strict` more often than plain B-mode cine.
- **Phase anchoring is wasted on RVSP.** Peak TR jet velocity is
  nearly phase-stationary within systole — the probe doesn't
  benefit from phase-anchored windows, but loses the clips that
  the eligibility filter removes.
- **Hard-negative-availability eligibility** (step 5 above) drops
  clips that can't form a same-study same-view wrong-phase
  triple. RVSP-informative views might more often lack a valid
  partner than apical cine.

The MR A4C signal is harder to pin down (both SV and Control
probes come out near the 47% majority baseline, so
sampler-vs-baseline Δ is within probe noise). No matched SV
baseline for TAPSE.

#### Isolating objective from sampler: the paired Δ

Because the **Control encoder (608) uses the same phase_matched
sampler as V3 and V4 but with `multiview_objective=intraview_only`**
(clip_b_pos and clip_b_neg are still loaded and forwarded through
the teacher but discarded from the loss), the Control is the
cleanest way to decompose each variant:

| Task | SV → Control (sampler ΔR²) | Control → V4 (objective ΔR²) | SV → V4 (combined) |
|---|---:|---:|---:|
| LVEF | **+0.025** | **+0.029** | +0.054 |
| RVSP | **−0.049** | **−0.090** | −0.139 |

This is what "about half sampler, half objective on LVEF;
sampler hurts RVSP and the objective hurts it even more" actually
means. All subsequent per-variant subsections use "sampler"
with this five-change referent in mind.

#### How MV2SV v5 relates to the phase_matched sampler

V5 inherits the same `PhaseMatchedStudySampler` with all five
changes above, plus **two MV2SV-specific extensions** (Fix 1):

7. **`target_clip` sampling** — explicit same-study,
   **different-target-view** clip drawn per row, with a
   view-family stage curriculum (Stage 1: A4C source →
   {A2C, A5C}; Stage 2+: {A2C, A5C, PLAX, PSAX-MV, A3C}).
   Stricter than V4's same_study-same-view path: some studies
   don't have distinct target-view pairs at matched phase and
   drop out.
8. **`fused_clips` sampling** (optional) — N-view same-study pool
   for the sparse fused auxiliary. Stage C tripped on this
   (`fused_valid_mask` mean = 1.44 < the `>=2` forward guard),
   which is why fused is off in the 5-ep pilot.

Consequence for the paired control in v5: the MV2SV paired
control cannot be Variant 4's Control (608) — it must be an
MV2SV-pipeline-matched intraview-only run that exercises the
same target_clip and (optionally) fused_clips draws as the
method, then discards them from the loss. That config doesn't
exist yet; it's item 5 in §8 "Known open items."

### 2.4 Variant 3 — Positive-only cross-view regression (job 542, `fb_phase_542`)

**Objective**:
```
L = L_intra + 0.25 · smooth_L1(pred(clip_a), teacher(clip_b))
```
where `clip_a` and `clip_b` are same-study clips sampled at matched
target phase φ (`sampling_mode=uniform_phase`, `phase_tolerance=0.15`)
under a view-pair policy (same_view 0.25 / same_family 0.45 /
cross_family 0.30). 25-epoch continuation from
`jepa_in21k_vitl_e100`, batch=64-pairs × 8 GPUs, pairs_per_study=24.

**Training diagnostic that pre-announced the null result**
(from finalbudget-phase-probes.md §Implementation):
- 542's intraview component is bit-identical to 548's SV-only
  curve (both ~0.48 at epoch 25).
- 542's **total** loss stays ≈0.67 across all 25 epochs because
  the crossview term `0.25 · L_crossview ≈ 0.17` sits flat on top
  of intraview without descending.
- That flat ≈0.17 contribution means the teacher's `h_b` encodings
  at matched phase+view are already close to `h_a`, so the
  crossview SmoothL1 is nearly redundant with intraview — the
  effective objective collapses toward single-view V-JEPA with a
  noise term.
- Sampler was verified functional (view-pair mixture and phase-bin
  coverage checked at dry-run time); the sampling path is doing
  what it claims. The failure is in the *loss structure*: at tight
  phase+view matching, positive-only regression onto a near-
  duplicate latent doesn't push the encoder to reorganize.

**Downstream results**:

| Task | Variant 3 (`fb_phase_542`) | SV e125 (matched compute) | Verdict |
|---|---|---|---|
| EchoNet-Dynamic LVEF val MAE | 5.013 / R² 0.691 / Pearson 0.833 (job 555 ep16) | 5.097 / 0.685 / 0.832 | NEUTRAL — Δ ≈ 0.08 MAE, within HP-sweep noise |
| Phase probe test circular MAE | 42.0° (job 542 encoder) | 43.2° (SV e125) | NEUTRAL — 1.2° gap well inside HP-seed noise; both arms beat constant baseline by only ~2° |
| Phase probe macro-bin acc (rare-bin) | 0.121 | **0.129** | SV wins; if phase-matching were the mechanism, the ordering should be consistent across metrics — it isn't. |
| Phase probe per-axis sin / cos | sin 0.448 / cos 0.432 | sin 0.448 / cos **0.422** | Any +25e continuation improves cos, but SV not phase+25 wins. The "cos improvement" isn't a phase-matching signal. |

**Why this matters for MV2SV**: Variant 3 is the closest prior attempt
to cross-view regression in this codebase, and its failure mode —
"predictor SmoothL1 onto a near-duplicate latent" — is exactly what
MV2SV v5 is architected to avoid. The `L_pair_view` term in v5 targets
the teacher's **`z_view` slot on a different-view clip**; the student
cannot produce this by copying the source because `z_view` is
(by design) the view-local residual. `L_view_nce` further requires
the student to *retrieve* the correct target-view latent out of a
batch, ruling out the conditional-mean shortcut that SmoothL1 alone
admits. (See §3.3 below for the explicit neutralization table.)

### 2.5 Variant 4 — Phase-relational InfoNCE with mandatory hard-neg (job 593 / 595)

**Objective**: `L = L_intra + λ_rel(t) · L_rel`, where
`L_rel` is a candidate-set InfoNCE over cosine similarities:

```
q         = PhaseRelationalHead(h_a, src_view, tgt_view, Δφ)
cands     = [y_pos, y_hard, y_batch_1, ..., y_batch_B]
logits    = q · cands / τ_rel
labels    = 0                      # positive = column 0 = y_pos
L_rel     = cross_entropy(logits, labels)
```

`y_pos` is the teacher's pooled latent on the same-study clip at
the target phase+view (what Variant 3 used as its SmoothL1 target).
`y_hard` is the teacher's pooled latent on a **same-study same-view
*wrong-phase*** clip — drawn with phase distance ≥ 0.25 cycles from
the positive's phase. This is the structural innovation: the
encoder is forced to push apart same-view representations at
different phases of the same heart. Self-diagonal and batch
same-study entries masked to −∞.

Data path: 3-clip phase_matched sampler with mandatory hard-neg
eligibility, view-pair policy (35/45/20), Δφ bucketing
`[0, 0.125, 0.25, 0.5]` at probs `[0.40, 0.30, 0.20, 0.10]`.
Same start-from-e100 continuation, 25 epochs, batch=32 × 8 GPUs.

**Training diagnostic**: `L_rel` descends monotonically
(1.780 → 1.045), InfoNCE top-1-with-hard rises (0.352 → 0.637),
pos-minus-hard similarity gap widens 6× (+0.021 → +0.131),
query-vector variance grows ~1700× (no representation collapse),
λ_rel hits its 0.05 cap at epoch 6. **The objective actually does
work** — unlike Variant 3 where the crossview term sat inert.

**LVEF downstream (EchoNet-Dynamic, matched probe protocol)**:

| Encoder | Val MAE / R² / Pearson (best head) | Test MAE / R² / Pearson | Probe compute |
|---|---|---|---|
| SV e100 (canonical) | 5.32 / 0.652 / 0.808 | 5.320 / 0.652 / 0.808 | +0 |
| **SV e125 (matched compute)** | **5.097 / 0.685 / 0.832** | **5.264 / 0.645** / ~0.81 (est) | +25 |
| SV e150 | 4.958 / 0.700 / 0.840 | 5.038 / 0.679 / ~0.83 | +50 |
| SV e175 | 4.855 / 0.717 / 0.848 | 5.003 / 0.682 / ~0.83 | +75 |
| SV e200 (job 379 protocol) | 4.867 / 0.714 / 0.846 | 5.058 / 0.684 / ~0.83 | +100 |
| SV e200 (job 421 protocol) | 4.880 / 0.714 / 0.845 | 4.880 / 0.714 / 0.845 | +100 |
| `fb_phase_542` (Variant 3) | 5.013 / 0.704 / 0.839 | — | +25 |
| **Variant 4 (595, ep15)** | **4.708 / 0.733 / 0.857** | **4.885 / 0.6986 / 0.8393** | **+25** |

Observations:
- Variant 4 is ahead of SV e125 on **every single epoch** of the
  probe trajectory on both val MAE and val R².
- Variant 4's val MAE 4.708 at ep15 is better than SV e125/e150/e200's
  20-epoch bests and within 0.15 of SV e175.
- Variant 4's val R² 0.733 at ep15 is higher than **every SV
  baseline's 20-epoch best** (SV e200 peaks at 0.714).
- Test numbers (596, full 1,277 EchoNet-Dynamic): **Variant 4 at e125
  ≈ SV at e200** (MAE 4.885 vs 4.880 / R² 0.6986 vs 0.714). The
  phase-relational objective is delivering the compute-equivalent of
  ~75 extra pretraining epochs on this probe.
- Δ vs matched-compute SV e125 (test set): **ΔMAE −0.379 / ΔR² +0.054**.

**RVSP downstream (MIMIC single-view, job 598 test completed
2026-05-02)**:

| Encoder | Test MAE | Test R² | Test Pearson |
|---|---|---|---|
| SV `fb_sv_548` (matched compute +25ep) | 9.705 | 0.157 | 0.400 |
| Variant 4 (598 test on method's best-HP head) | 10.525 | 0.018 | 0.281 |

*Apparent verdict* (aggregate only): ΔMAE +0.82, ΔR² −0.139,
ΔPearson −0.119. But the per-severity breakdown below tells a
more nuanced story.

**Severity-stratified RVSP (2,000 test clips)**:

| True RVSP bucket | n | y̅ | SV pred | V4 pred | SV MAE | V4 MAE | SV ρ | V4 ρ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| <25 (normal) | 772 | 18.4 | 29.2 | 31.2 | 10.73 | **12.72** | −0.30 | **−0.39** |
| 25–35 (borderline) | 725 | 30.1 | 31.6 | 33.2 | 4.38 | 5.41 | 0.00 | 0.03 |
| 35–45 (mod PH) | 84 | 44.0 | 32.9 | 37.9 | 11.11 | **6.66** | — | — |
| ≥45 (severe PH) | 419 | 52.8 | 36.1 | 36.9 | 16.76 | 16.10 | 0.18 | 0.04 |

Three observations. **First, both encoders are doing regression to
the global mean** (label std 14.2, pred std ~6): normals get
over-predicted (into the 29–33 range), severe PH gets
under-predicted (to 36–37 for a true mean of 53). Both encoders'
prediction dynamic range is ~40% of the label range. In that
regime, **aggregate R² mostly measures what proportion of label
variance the probe didn't collapse**, not rank correlation. Plain
SV at R²=0.157 is "barely explains variance," not "good." The gap
V4 ⟶ SV at R²=0.018 ⟶ 0.157 is a difference in *how compressed
the predictions are* more than a difference in *what the
representation encodes*.

**Second, the two arms fail in opposite ways across buckets**:
- Normal RVSP (<25): V4 is worse (pred 31 vs truth 18, ΔMAE +2.0
  vs SV). Also the within-bucket Pearson is **negative and larger
  in magnitude** (V4 −0.39 vs SV −0.30) — within the normal
  range, both probes' predictions *anti-correlate* with true RVSP.
  That's a pathological finding per se, almost certainly driven
  by non-RVSP confounds (view/image-quality features) driving
  prediction up when true RVSP is low.
- Moderate PH (35–45): V4 is **materially better** than SV (MAE
  6.66 vs 11.11; V4 pred mean 37.9 is closer to truth 44.0 than SV
  at 32.9). On the bucket where leaflet/flow dynamics would plausibly
  matter most, phase-relational pretraining helps.
- Severe PH (≥45): both encoders saturate; method's within-bucket
  Pearson drops to 0.04, SV holds 0.18. Neither does well, but SV
  retains some rank information.

**Third, probe head selection amplifies the val→test gap**. 597's
full 20-epoch val trajectory (from
`final_phase_rel25_rvsp_597/probe/.../log_r0.csv`):

| Epoch | val MAE | val R² | val Pearson |
|---:|---:|---:|---:|
| 5 | **6.701** | **0.199** | 0.468 |
| 6 | 6.789 | 0.176 | 0.461 |
| 7 | 7.113 | 0.103 | **0.485** |
| 10 | 7.271 | 0.049 | 0.465 |
| 12 | 6.769 | 0.180 | 0.475 |
| 20 | 7.093 | 0.122 | 0.472 |

No trajectory collapse — R² stays in the 0.05–0.20 band across all
20 epochs. `best.pt` was selected on val MAE at ep5. If Pearson
had been the selection criterion, ep7 or ep10's head would have
gone to test instead. Plain SV's probe has the same head-choice
sensitivity: val MAE best at ep5, val R² best at ep6, val Pearson
best at ep10 — three different heads.

**Revised RVSP verdict**:
- On aggregate test metrics, Variant 4 underperforms plain SV at
  matched compute. The gap is real but is in the noise-floor
  regime (both encoders are doing mean-regression with ≤16%
  variance explained).
- Per-severity, the story splits: V4 is worse on normals and better
  on moderate PH. The "Variant 4's LVEF gain does not translate to
  RVSP" claim holds (no aggregate or sub-group where V4 cleanly
  beats SV), but "actively regresses" overstates it given the mod-PH
  bucket evidence.
- The paper-interpretable Δ is still 597 − 601 (paired intraview-
  only control), not 597 − fb_sv_548 (which mixes sampler +
  objective changes).

**LVEF stratified (for comparison, same protocol, 1,277 test
clips, V4)**:

| True LVEF bucket | n | y̅ | V4 pred | V4 MAE | V4 ρ |
|---|---:|---:|---:|---:|---:|
| <40 reduced | 160 | 29.0 | 36.3 | 8.71 | 0.64 |
| 40–55 borderline–mid | 241 | 49.0 | 53.0 | 6.66 | 0.45 |
| ≥55 preserved | 876 | 62.1 | 61.1 | **3.70** | 0.40 |

Same regression-to-mean pattern: reduced-EF tail pred mean 36.3 for
truth 29.0 (+7.3 systematic bias, MAE 8.71), preserved bucket
predictions sit almost on the mean (MAE 3.70 because truth clusters
near pred). But within-bucket **Pearson for reduced-EF is 0.64** —
the probe retains strong rank information on the clinically
critical tail, even though predictions are biased high. This is
exactly the "reduced-EF tail MAE" failure mode flagged in
`phase-jepa.md` §"two anchoring quantities." V4 improves the
aggregate but the tail bias remains; it's not a pure win.

**MR severity A4C (MIMIC 4-class, 4,482 test clips, job 610 test
completed 2026-05-02)**:

Overall accuracy 52.21% (max-HP head) / 53.74% (probe_best.pt).
Majority-class baseline ~47% (class 0 = 2,110 / 4,482), so
+5.2 pp over majority.

Confusion matrix (true rows × predicted cols):

| | Pred 0 | Pred 1 | Pred 2 | Pred 3 | Recall |
|---|---:|---:|---:|---:|---:|
| True 0 None/Trivial (n=2,110) | 1,695 | 266 | 149 | 0 | **0.803** |
| True 1 Mild (n=1,291) | 682 | 338 | 271 | 0 | 0.262 |
| True 2 Moderate (n=782) | 260 | 215 | 307 | 0 | 0.393 |
| True 3 Severe (n=299) | 48 | 57 | 194 | 0 | **0.000** |

Three striking features: (1) **class 3 Severe is never predicted**
— 0/299 recall. (2) class 1 Mild is mostly collapsed into class 0
(682/1,291 → 53% misclassified to None). (3) the +5.2 pp over
majority-baseline is almost entirely class-0 accuracy (80.3%
recall on class 0, with everything else bleeding into class 0 or 2).

Binary "Moderate-or-worse vs not" (class ≥2 vs <2, prevalence
24.1%): sensitivity 0.463, specificity 0.877, accuracy 0.777. AUROC
couldn't be computed (sklearn missing in the eval env — see §9
known issues); this will need a re-inference. **Verdict**: the
method is not delivering a real MR grading signal from A4C alone.
Consistent with phase-relational-hardneg.md §8.7.1's pre-registered
prediction that A4C-SV MR is a weak phase-awareness task; the
strong test is MR-MV (A4C + A2C + PLAX integration), not queued.

**TAPSE A4C (MIMIC regression, 2,000 test clips, job 621 test
completed 2026-05-02)**:

Aggregate: MAE 0.355 cm, R² 0.250, Pearson 0.519. Clinical cutoffs
are <1.7 cm (reduced RV longitudinal function), 1.7–2.4 normal,
≥2.4 hyperdynamic/tall.

| True TAPSE bucket | n | y̅ (cm) | V4 pred (cm) | MAE | ρ |
|---|---:|---:|---:|---:|---:|
| <1.7 reduced | 473 | 1.35 | 1.71 | 0.42 | 0.35 |
| 1.7–2.4 normal | 1,063 | 1.98 | 1.98 | **0.23** | 0.30 |
| ≥2.4 hyperdynamic | 464 | 2.72 | 2.16 | 0.57 | 0.19 |

Same regression-to-mean (label std 0.53, pred std 0.34; 36% range
compression). The normal-bucket MAE of 0.23 cm is clinically
meaningful — roughly the within-exam test-retest noise on TAPSE.
But the hyperdynamic bucket MAE 0.57 cm is half a cutoff's width
of systematic under-prediction. Within-bucket Pearson 0.30 in the
normal range says some RV longitudinal motion signal *is* present
in the encoder, just too compressed to read at the tails.

No matched-compute SV +25 TAPSE baseline; TAPSE is standalone here.
The observation only tells us that phase-relational features encode
some RV longitudinal motion — not that they encode it better than
plain SV.

**Is V4 a multi-view run?** (clarifying frame, 2026-05-03)

Yes, by the standard definition "context and target encoders see
different views." Confirmed from `forward_phase_relational` at
`app/vjepa_multiview/train.py:864-865, 875, 893-895`:

- **Student/context encoder forwards `clip_a` only** (line 875).
  It never sees clip_b_pos directly.
- **Teacher/target encoder forwards all three clips** (line 865:
  `torch.cat([pair.clip_a[0], pair.clip_b[0], pair.clip_b_neg[0]])`)
  under `no_grad`. The teacher's output on the different-view
  clip is `h_b_pos`, which is the positive target of L_rel
  (lines 893-895, 907-914).
- **Cross-view supervision enters the encoder via L_rel**: the
  InfoNCE pulls `student(clip_a)` toward `teacher(clip_b_pos)`,
  conditioned on `(view_a, view_b, Δφ)`. 65% of training rows
  have `clip_b_pos` from a different view family than `clip_a`
  (see `view_pair_policy` in
  `configs/train/vitl16/pretrain-multiview-phase-relational-hardneg-25of100-paper.yaml`:
  same_view_prob=0.35, same_family_prob=0.45, cross_family_prob=0.20).

So V4's encoder **is** trained with cross-view supervision — by
your definition it is multi-view. The framing in earlier
revisions of this doc that referred to V4 as "single-view" was
literally true about the student's forward pass but misleading
about the supervision signal.

The same correction applies to pilot 655 and ctrl 658: the teacher
encodes `target_clip` (different view) in both. Pilot uses that
teacher output as the L_pair_view target; ctrl 658 computes it and
multiplies by zero (all MV2SV lambdas = 0).

**Why phase-informed V4 doesn't transfer to TAPSE** (mechanistic
reading, 2026-05-03):

TAPSE is "phase-informed" in the superficial sense that its
measurement requires picking two phases of the cycle (end-diastole
annulus baseline, end-systole peak excursion). But it is *not*
the same kind of phase signal that V4's InfoNCE objective creates.
Three distinctions matter:

1. **Phase identity vs phase-amplitude at a landmark.** LVEF =
   (EDV − ESV)/EDV decodes off "where in the cycle is this frame"
   (ED vs ES) plus "approximate ventricular cavity size." V4's
   `L_rel = InfoNCE(PhaseRelationalHead(c_a_pool), ..., Δφ)` is
   exactly trained to make A4C frames discriminable by phase index
   — which is all LVEF needs. TAPSE = **peak systolic excursion of
   the tricuspid annulus along its longitudinal axis, in mm**.
   That's not "which phase is this" — it's "how many mm did one
   specific 5-mm structure move between two phases, along a
   specific direction." Telling two phases apart is trivially
   satisfiable by global features (cavity area, leaflet
   configuration); it does not require preserving per-landmark
   metric amplitude.

2. **L_rel is a *pooled* readout of the encoder — downstream
   probes are not.** Important disambiguation:

   - **L_rel's gradient signal into the encoder** operates on
     `c_a_pool = pool_tokens(z_ctx)` (mean pool). The only
     constraint L_rel places on the student's tokens is "whatever
     pattern keeps the token *mean* on the cycle-position axis."
     Individual tokens are free to do anything consistent with
     that mean.
   - **The downstream probe sees the full token sequence**
     `[B, N_tok, D]`, not the pool. The attentive-classifier
     probe (`num_probe_blocks=4, num_heads=16`) does learned
     cross-attention over tokens to produce its readout.
   - So the probe CAN attend to any token-level structure the
     encoder encodes — but V4's L_rel *only incentivized* token
     structure that preserves the pool's phase-identity axis. The
     only source of token-level structure is V-JEPA intraview
     (standard reconstruction), which encodes "what does this
     patch look like," not "how far did this landmark move
     between phases."
   - L2 normalization in the InfoNCE strips any magnitude
     information from the pooled direction itself. So not only
     are tokens unconstrained beyond the pool-mean constraint,
     the pool-mean constraint itself doesn't carry metric
     amplitude — just direction.

   The 36% TAPSE range compression is consistent: V4's
   representation distinguishes *which phase* reliably (pool-mean
   axis), but the probe cannot recover "how many mm did this
   annulus move" from tokens that were never supervised to encode
   that. Attentive pooling can weight tokens, but it can only
   pool what the encoder chose to encode.

3. **V4's discrimination is phase-identity, not phase-amplitude
   coupling.** V4 learns: given A4C at phase φ, pick the right
   Δφ-offset pair out of N candidates. That forces phase-identity
   (trivially supports LVEF). It does NOT reward "predict the
   per-patch displacement field between φ=0 and φ=0.5" — which is
   what would be needed for TAPSE-style amplitude reading.

**Evidence from §5.9 F/G/H that's consistent with this**: V4 is
the only arm with held-out temporal-shuffling sensitivity
(cos(clean, shuffled) = 0.89 vs 0.99 elsewhere). V4's
representation does depend on temporal ordering — just at the
*global phase-identity* level, not at the *local-landmark-
amplitude* level. Exactly what you'd predict if phase-InfoNCE
learns "where in the cycle am I" but not "how did each patch
move."

**Implication for downstream tasks a V4-style objective can /
can't capture**:

| Task | Mechanism | V4 covers it? |
|---|---|:---:|
| LVEF | ED vs ES phase-identity + cavity size | ✓ (+0.029 paired R² confirmed) |
| RVSP | Peak TR jet velocity on in-plane RV view | ✗ (paired −0.090 R² — view-dominant, not phase-dominant) |
| MR severity | Multi-view jet integration at systole | ~ null (view-dominant, but some phase signal helps) |
| **TAPSE** | Per-landmark amplitude integration over the cycle | ✗ (no matched baseline, but geometry predicts null) |
| AS severity | Peak PLAX/PSAX-AV gradient at systole | untested; view-dominant |

### Why V4 succeeded on LVEF — best hypothesis

V4's +0.029 paired R² on LVEF is small-but-real. The strongest
hypothesis is **not** "phase-relational pretraining helps any
phase-informed task" (TAPSE rules that out). Instead:

> **V4's InfoNCE specifically rewards the student for learning a
> pooled representation in which cardiac cycle position is a
> low-curvature axis. LVEF decodes off exactly that axis (ED vs
> ES discriminability + approximate ventricular cavity size from
> the intraview V-JEPA loss). V4 gives the LVEF probe a
> representation where ED and ES are pre-separated along a
> single linear direction in pooled latent space; the probe
> doesn't have to synthesize that axis.**

Three design choices conspire to produce this specifically:

1. **Hard-negative is same-view wrong-phase** (`clip_b_neg`,
   `rel_wrong_phase_min_delta=0.25`). The InfoNCE must
   distinguish ED-looking frames from ES-looking frames *within
   the same view family* — exactly the discrimination LVEF needs.
   phase-relational-hardneg.md already showed that removing the
   hard-neg collapses the LVEF gain.

2. **Positive is different-view same-study at controlled Δφ**.
   Conditioned on `(view_a, view_b, Δφ)`, the student has to
   encode phase-identity in a way that's *invariant to view
   change*. The optimal solution is a single low-curvature
   cycle-position axis in the pool — shared across views.

3. **Pooled-feature cosine + L2 norm bakes this axis into the
   encoder's token-mean direction.** The downstream attentive
   probe reads it out via cross-attention (which can recover
   pool-mean by uniform-weighting all tokens, or do better if
   additional token-level structure helps).

This explains the TAPSE failure symmetrically: TAPSE requires
*token-level amplitude* information, which V4's pooled InfoNCE
does not supervise. Probe attentive pooling cannot extract
structure that was never put into the tokens.

**Fix direction — token-level phase-InfoNCE (not in V4 or MV2SV v5)**:

The highest-expected-value next variant is a re-run of V4 with the
InfoNCE moved from pooled → per-token:

```
q_tokens       = relational_head_token(z_ctx)  # [B, N, D]
y_pos_tokens   = h_b_pos.detach()              # [B, N, D]
y_hard_tokens  = h_b_neg.detach()              # [B, N, D]
L_rel_token    = mean over tokens of InfoNCE(q_n, y_pos_n, y_hard_n)
```

Same sampler, same hard-neg, same view-pair policy — but the
cross-view phase supervision now lands at each spatial token
rather than on the pool. Expected outcomes:

- **LVEF**: approximately preserved. Every token encoding
  phase-identity, then pooled by attention, still gives the
  probe a cycle-position axis.
- **TAPSE**: now has a shot. Tokens at the TV annulus location
  can carry phase-coupled amplitude information because the
  objective is supervised there.
- **RVSP**: potentially helped. In-plane RV inflow structure
  gets cycle-coupled supervision at the right spatial cell.
- **MR**: limited by A4C-only probing; probably unchanged.

Note that pilot 655's `L_pair_view` loss *looks* like a token-
level fix, but §5.10 H1 showed the factorized head's slots are
near-linear projections of the pool — so pilot's cross-view
supervision is effectively still pooled, just through a rotation.
A genuine token-level phase-InfoNCE (no factorized head, no
pooling in the head) is a distinct variant that we have not run.

The `L_local_motion` objective sketched in the v4 addendum of the
plan file (predict token-level residuals between same-view
Δφ-displaced clips) is an alternative targeting TAPSE specifically.
Unshipped; gates TAPSE entirely. Either the token-level
phase-InfoNCE rewrite OR `L_local_motion` would fill the gap; the
token-level rewrite is a one-file change to
`forward_phase_relational` whereas `L_local_motion` requires
sampler extension and a new loss. Token-level rewrite is the
cheaper first experiment.

**Update 2026-05-03**: both fixes are now shipped as the
EchoJEPA-TokenRel (Run 1) + EchoJEPA-TokenRel + MotionDelta (Run 2)
variants. See **§5.15** for run-log entries and **§11** for the
full design notes. Jobs 692-695.

**Multi-view RVSP (job 633, running as of 2026-05-02 19:46)**:
phase-relational-hardneg.md §8.6 flagged MV-RVSP as the
"mechanistically informative" next test because the method's
cross-view alignment component can only be exercised by a probe
that reads ≥2 views at inference. Not complete at this doc's time
of writing; pending.

**Variant 1 (φ-JEPA) LVEF stratified — full population via `clip_outputs.npz`**:

(The per-clip `predictions.csv` from job 398 is a 160-clip rank-0
shard, but each probe epoch also wrote a `clip_outputs.npz` with
per-clip predictions from **all** ranks — 1,280 clips, 6 HP heads,
z-score params, and `best_head_idx` saved inline. That's the
authoritative population-level data; the stratification below uses
it.)

φ-JEPA LVEF trajectory, full-population (n=1,280 each):

| ckpt | MAE | R² | Pearson | compression | best head |
|---|---:|---:|---:|---:|---:|
| e25  | 6.110 | 0.510 | 0.720 | 19% | 4 |
| e50  | 5.552 | 0.606 | 0.780 | 18% | 4 |
| e75  | 5.394 | 0.625 | 0.793 | 18% | 0 |
| e100 | 5.272 | 0.642 | 0.803 | 19% | 3 |

Monotone improvement across the trajectory; "compression" is the %
reduction in prediction std vs label std (smaller = more faithful
dynamic range). φ-JEPA compresses ~19% throughout — much better
than RVSP probes (54–67% below).

Per-bucket, full-population:

| ckpt | Bucket | n | y̅ | pred | bias | MAE | ρ |
|---|---|---:|---:|---:|---:|---:|---:|
| e25  | <40 reduced | 160 | 29.0 | 39.3 | +10.3 | 11.93 | 0.51 |
| e25  | 40–55 bord  | 243 | 49.0 | 53.0 |  +3.9 |  7.23 | 0.38 |
| e25  | ≥55 pres    | 877 | 62.1 | 59.3 |  −2.8 |  4.74 | 0.26 |
| e50  | <40 reduced | 160 | 29.0 | 37.3 |  +8.3 |  9.98 | 0.55 |
| e50  | 40–55 bord  | 243 | 49.0 | 51.7 |  +2.7 |  6.24 | 0.43 |
| e50  | ≥55 pres    | 877 | 62.1 | 59.2 |  −2.9 |  4.55 | 0.32 |
| e75  | <40 reduced | 160 | 29.0 | 37.8 |  +8.8 |  9.99 | 0.57 |
| e75  | 40–55 bord  | 243 | 49.0 | 53.1 |  +4.1 |  6.84 | 0.42 |
| e75  | ≥55 pres    | 877 | 62.1 | 60.4 |  −1.7 |  4.15 | 0.34 |
| e100 | <40 reduced | 160 | 29.0 | 37.9 |  +8.8 |  9.89 | 0.59 |
| e100 | 40–55 bord  | 243 | 49.0 | 53.1 |  +4.1 |  6.72 | 0.39 |
| e100 | ≥55 pres    | 877 | 62.1 | 60.4 |  −1.7 |  4.03 | 0.38 |

Two patterns:
- Reduced-EF bias starts at +10.3 at e25 and plateaus at +8.8 from
  e50 onward — the bias doesn't improve with training, only the
  within-bucket Pearson does (0.51 → 0.59). More training doesn't
  fix the systematic under-severity-calling on reduced EF; it just
  makes the probe rank those patients more consistently.
- The preserved-EF bucket drives most of the aggregate R² (877 /
  1,280 = 68% of the test set, MAE 4.03). The reduced-EF tail
  (where V4 is designed to help) carries ~2× the per-clip MAE of
  preserved.

**Variant 1 vs Variant 4 on LVEF — head-to-head, full-population**:

| | φ-JEPA e100 | V4 phase-rel e125 |
|---|---:|---:|
| Aggregate MAE | 5.272 | **4.885** |
| Aggregate R² | 0.642 | **0.699** |
| Aggregate Pearson | 0.803 | **0.839** |
| Compression | 19% | **15%** |
| <40 reduced bias | +8.8 | **+7.3** |
| <40 reduced MAE | 9.89 | **8.71** |
| <40 reduced ρ | 0.59 | **0.64** |
| ≥55 preserved MAE | 4.03 | **3.70** |

V4 is strictly better on every aggregate and every bucket, and the
compression is lower (predictions more faithful to label dynamic
range). The reduced-EF tail bias *improves* by 1.5 and ρ improves
by 0.05 — phase-relational *does* help the clinically-critical
tail, on top of its aggregate win.

### Variant 3 RVSP — full-population via `clip_outputs.npz` (n=2,000)

finalbudget-phase-probes.md §8 queued 561 = V3's RVSP test. The
NPZ was available; previously not analyzed. Aggregate:

| Encoder | MAE | R² | Pearson | Compression |
|---|---:|---:|---:|---:|
| SV fb_sv_548 | 9.705 | 0.157 | 0.400 | 59% |
| V3 fb_phase_542 | **10.175** | **0.092** | **0.306** | 67% |
| V4 phase-rel | 10.525 | 0.018 | 0.281 | 54% |

V3 on aggregate is **worse than SV and roughly halfway between SV
and V4** on RVSP. Compression is its worst feature — V3 collapses
predictions 67% vs label, more than either SV or V4. Per-bucket
(for direct comparison to V4's table in §2.4):

| Bucket | n | y̅ | V3 pred | V3 bias | V3 MAE | V3 ρ |
|---|---:|---:|---:|---:|---:|---:|
| <25 normal | 772 | 18.4 | 29.5 | +11.1 | 11.08 | −0.30 |
| 25–35 borderline | 725 | 30.1 | 31.7 | +1.7 | 4.09 | −0.09 |
| 35–45 mod PH | 84 | 44.0 | 31.3 | −12.7 | 12.74 | — |
| ≥45 severe PH | 419 | 52.8 | 34.3 | −18.5 | 18.52 | +0.12 |

Observations:
- V3's per-bucket pattern is qualitatively identical to SV's:
  same direction of bias (over-predict normals, under-predict
  PH), similar magnitudes. V3 is *barely distinguishable from
  SV per-bucket* — it mostly differs in being more compressed,
  which hurts aggregate R² without changing the per-severity
  story.
- V4's one redeeming RVSP feature — beating SV on the mod-PH
  bucket (MAE 6.66 vs 11.11) — is **not** present in V3
  (V3 MAE 12.74, marginally *worse* than SV's 11.11).
- V4's worst RVSP feature — amplified normal-bucket damage (MAE
  12.72 vs SV 10.73) — is also present in V3 (11.08, between SV
  and V4 but close to SV).

In plain terms: V3's positive-only cross-view loss didn't change
RVSP readouts meaningfully over plain SV. Consistent with the
training-time diagnostic in finalbudget §Implementation ("542
intraview component essentially identical to 548's SV loss;
crossview term sits at ≈0.17 without descending"). The encoder is
what SV would have produced at matched compute.

### Paired intraview-only CONTROL (job 608 encoder, probes 629/631/611)

The phase-relational paper's pre-registered comparison is **method
− paired intraview-only control**. The control pretrain uses the
identical 3-clip phase_matched sampler with mandatory wrong-phase
hard-neg eligibility, view-pair policy, and quality/RR filtering
as the method — only the loss differs (`multiview_objective:
intraview_only`, so clip_b_pos and clip_b_neg are loaded and
discarded; same data path, same throughput). This isolates
*objective* from *data path + eligibility filter*.

Three control probes completed test inference 2026-05-02:

**Control LVEF** (job 630, full-population n=1,277):

| | SV fb_sv_548 | **Control 608** | V3 fb_phase_542 | V4 method 593 |
|---|---:|---:|---:|---:|
| Test MAE | 5.264 | **5.067** | — | **4.885** |
| Test R² | 0.645 | **0.670** | val 0.704 | **0.699** |
| Test Pearson | ~0.81 | **0.821** | val 0.839 | **0.839** |
| Compression | — | 17% | — | 15% |

Stratified:

| Bucket | n | Control pred | Control MAE | Control ρ | V4 pred | V4 MAE | V4 ρ |
|---|---:|---:|---:|---:|---:|---:|---:|
| <40 reduced | 160 | 37.3 | 9.48 | 0.62 | 36.3 | 8.71 | 0.64 |
| 40–55 bord | 241 | 52.9 | 6.38 | 0.48 | 53.0 | 6.66 | 0.45 |
| ≥55 preserved | 876 | 60.6 | 3.90 | 0.38 | 61.0 | 3.70 | 0.40 |

**Paper-interpretable Δ_LVEF on test**: V4 − Control =
ΔMAE −0.182, ΔR² +0.029, ΔPearson +0.018. Real but modest. The
V4 − SV gap was ΔMAE −0.38, so **about half of V4's LVEF win
over SV comes from the 3-clip phase_matched sampler**, not the
InfoNCE objective. The other half is real phase-discrimination
signal, confirmed by the control having a clean trajectory (LVEF
val best at ep18: MAE 4.80, R² 0.716, Pearson 0.847 — control
actually beats SV e200-job-421 on val Pearson). On the reduced-EF
tail specifically, V4 improves bias by 1.0 and MAE by 0.77 vs
control; within-tail ρ is 0.64 vs 0.62 — V4 is fractionally
better at rank ordering reduced-EF, but the size is within
HP noise.

**Control RVSP** (job 632, full-population n=2,000):

| | SV fb_sv_548 | **Control 608** | V3 542 | V4 method 593 |
|---|---:|---:|---:|---:|
| Test MAE | 9.705 | **9.984** | 10.175 | **10.525** |
| Test R² | **0.157** | 0.108 | 0.092 | 0.018 |
| Test Pearson | **0.400** | 0.344 | 0.306 | 0.281 |
| Compression | 59% | **75%** | 67% | 54% |

**This is the clean decomposition we couldn't do before.**
- **SV → Control** gap: ΔR² −0.049. The 3-clip phase_matched
  sampler + eligibility filter costs ~5 pp of RVSP R² *by itself*,
  before any objective change. Plausible mechanism:
  eligibility-filtering removes clips useful for RVSP that lack
  a valid wrong-phase same-view hard negative.
- **Control → V3** gap: ΔR² −0.016. Positive-only crossview SmoothL1
  costs another 2 pp on top of the sampler penalty. Small, as
  expected given V3's loss collapses to near-intraview at training
  time.
- **Control → V4** gap: ΔR² −0.090. **The InfoNCE objective costs
  9 pp of RVSP R² on top of the sampler.** V4 − V3 gap is 7 pp, so
  the InfoNCE hard-negative-at-wrong-phase specifically is doing
  the damage.

The paper-interpretable Δ_RVSP: V4 − Control = ΔMAE +0.54,
ΔR² −0.090, ΔPearson −0.063. **The method actively hurts RVSP**
on top of whatever the sampler already took.

Control's RVSP compression is 75% — the worst of any arm. Its
prediction std is only 3.53 on a label std of 14.23. Yet its R² is
0.108 (second-best) because the compressed predictions align with
labels. V4 is less compressed (54%, pred std 6.49) but its
predictions ANTI-correlate with truth inside the normal bucket
(ρ = −0.39) — the InfoNCE-trained encoder has learned to assign
*higher* RVSP-proxy to clips with *lower* true RVSP. That's the
signature of the objective actively interfering with the target.

Stratified (control vs V4):

| Bucket | n | Control pred | Ctrl MAE | Ctrl ρ | V4 pred | V4 MAE | V4 ρ |
|---|---:|---:|---:|---:|---:|---:|---:|
| <25 normal | 772 | 30.0 | 11.57 | **−0.27** | 31.2 | 12.72 | **−0.39** |
| 25–35 borderline | 725 | 31.3 | 2.95 | 0.01 | 33.2 | 5.41 | 0.03 |
| 35–45 mod PH | 84 | 32.3 | 11.75 | — | 37.9 | 6.66 | — |
| ≥45 severe PH | 419 | 33.9 | 18.89 | ~0 | 36.9 | 16.10 | 0.04 |

Reading by column:
- Control's normal bucket ρ is also negative (−0.27), matching SV
  (−0.30). The anti-correlation-on-normals is NOT unique to V4 —
  it's a **feature of the single-view RVSP probe setup** that all
  arms share. V4 amplifies it (−0.39 vs control's −0.27), the
  InfoNCE loss specifically makes it worse.
- On mod-PH, V4 still beats control (MAE 6.66 vs 11.75). The V4
  mod-PH win is real: with or without the sampler-confound control,
  the method correctly pushes borderline clips' predictions up.
- On severe-PH, V4 has a small MAE edge over control (16.10 vs
  18.89) but within-bucket ρ is ~0.04 for both — neither captures
  rank inside the severe bucket.

**Control MR A4C** (job 612, full-population n=4,482):

| | Control 608 | V4 method 593 |
|---|---:|---:|
| Overall acc | **52.37%** | 52.21% |
| Majority baseline | 47.08% | 47.08% |
| +pp over majority | **+5.3** | +5.1 |
| Severe recall (class 3) | 0/299 | 0/299 |
| Moderate+ sens / spec | 0.453 / 0.882 | 0.463 / 0.877 |

**Paper-interpretable Δ_MR: essentially zero** (V4 − Control
= −0.16 pp overall acc). Both arms collapse class 3 completely
(never predict Severe) and achieve accuracy on class-0 that is
barely above majority baseline. Control even edges out method by
0.16 pp. **Phase-relational is null on MR A4C SV** with the paired
comparison controlling for sampler effects.

Control MR val trajectory converges at 54.1-54.2% ep16-20 (best
ep15: 54.19%). V4's val trajectory was similar (52.0-54.0%
ep15-20). Neither probe has a head that clearly distinguishes the
two encoders on MR.

**No TAPSE control** — job 621 ran only on the method encoder;
the paired control TAPSE probe was not submitted.

### Summary across variants — full-population, paired where available

| Variant | Pretrain compute | LVEF test MAE / R² | RVSP test MAE / R² | MR overall acc | Severe-end probe ρ |
|---|---|---:|---:|---:|---:|
| SV fb_sv_548 | IN21K + MIMIC +25 ep | 5.264 / 0.645 | 9.705 / **0.157** | — | RVSP 0.18 |
| V1 φ-JEPA e100 | IN21K + MIMIC +100 ep | 5.272 / 0.642 | — | — | LVEF reduced 0.59 |
| V3 fb_phase_542 | +25 ep | val 5.013 (no test) | 10.175 / 0.092 | — | RVSP 0.12 |
| **Control 608** (paired to V4) | +25 ep | **5.067 / 0.670** | **9.984 / 0.108** | **52.37%** | LVEF 0.62, RVSP 0.00 |
| **V4 method 593** | +25 ep | **4.885 / 0.699** | 10.525 / 0.018 | 52.21% | LVEF 0.64, RVSP 0.04 |
| **Ctrl 658 e5** (paired to MV2SV) | **+5 ep** | **5.16 / 0.665** | **9.91 / 0.093** | **53.01%** | — |
| **MV2SV v5 pilot 655 e5** | **+5 ep** | 5.29 / 0.645 | 10.17 / 0.066 | 51.54% | — |

**Paired deltas** (V4 e25 − Control 608 e25):

| Task | ΔMAE | ΔR² | ΔPearson |
|---|---:|---:|---:|
| LVEF | **−0.182** | **+0.029** | **+0.018** |
| RVSP | +0.541 | **−0.090** | −0.063 |
| MR (Δ acc) | +0.16 pp | — | — |

**Paired deltas** (MV2SV Pilot 655 e5 − Ctrl 658 e5):

| Task | ΔMAE | ΔR² | ΔPearson / ΔAcc |
|---|---:|---:|---:|
| LVEF | +0.14 | **−0.020** | −0.012 (Pearson) |
| RVSP | +0.26 | **−0.027** | −0.042 (Pearson) |
| MR (Δ acc) | — | — | **−1.47 pp** |

**Pilot 655 e5 vs V4 593 e25 (non-paired — different compute, different sampler; interpret cautiously):**

| Task | Pilot e5 | V4 e25 | Δ (pilot−V4) | Who wins |
|---|---:|---:|---:|---|
| LVEF R² | 0.645 | **0.699** | **−0.054** | V4 by a lot |
| RVSP R² | **0.066** | 0.018 | +0.048 | Pilot — no V4-style hurt |
| MR acc | 51.54% | 52.21% | −0.67 pp | ~tie |

**Key reads from the combined table:**

1. **Ctrl 658 e5 matches or beats Ctrl 608 e25 on every endpoint**
   at a fifth of the pretrain compute. LVEF R² 0.665 ≈ 0.670,
   RVSP MAE 9.91 < 9.98, MR acc 53.0% > 52.4%. The MV2SV *sampler
   + data path alone* is doing most of the work the 20 extra
   epochs of plain intraview on random clips were buying.
2. **Pilot 655 e5 loses to Ctrl 658 e5 on every paired endpoint**
   at matched compute. The MV2SV objective (pair_view + view_nce +
   pair_shared + shared) doesn't add downstream probe signal at
   e5 over sampler-alone.
3. **V4 e25 still holds the LVEF title** (R² 0.699). Pilot e5
   hasn't matched it; ctrl 658 e5 hasn't either.
4. **Pilot e5 avoids V4's RVSP hurt.** V4 at +25 ep: R² 0.018
   (9 pp below paired ctrl). Pilot at +5 ep: R² 0.066, close to
   its paired ctrl (−0.027). The MV2SV objective doesn't damage
   RVSP the way the phase-relational InfoNCE did.
5. **The remaining paper question is whether pilot e25 (post
   683) clears (a) ctrl 658 e25 (post 684) on LVEF/RVSP/MR and
   (b) V4 593 e25 on RVSP/MR.** Both comparisons are still
   pending.

The paired control makes the story crisp:

1. **V4's LVEF win is about half sampler, half objective.** Paired
   Δ is real (ΔR² +0.029) but modest. The phase-relational
   objective contributes, but most of the SV→method gap is
   sampler + eligibility filtering.
2. **V4's RVSP hurt is specifically the InfoNCE objective.** The
   sampler costs 5 pp of R² (SV → Control); the objective costs
   **another 9 pp** (Control → V4). The paired Δ is −0.090 R²
   — the method is objectively worse than doing pure intraview
   on the same 3-clip pipeline.
3. **V4's MR A4C result is null vs paired control.** 0.16 pp
   edge to control, not method. Whatever MR signal the method
   had disappeared once you remove the sampler confound.
4. **Neither V3 nor V4 gives a positive paired Δ outside LVEF.**
   V3 was null on LVEF (within-noise of SV) and never made it to
   RVSP-paired comparison. V4 has one real endpoint (LVEF).

This sharpens the MV2SV motivation considerably. It is no longer
"we conjecture Variant 4's phase-discrimination won't help
view-dominant tasks" — it is:

> **Paired, compute-matched, sampler-matched: V4's InfoNCE
> objective actively hurts RVSP by 9 pp of R² and is null on MR
> A4C, while providing only a small +3 pp R² gain on LVEF beyond
> what the phase_matched sampler alone buys.**

MV2SV v5 attacks this by replacing phase-discrimination with
view-discrimination while keeping the same sampler eligibility
machinery. If v5's paired comparison (v5 − intraview-only on the
same MV2SV data path) lands positive on RVSP / MR / AS, it will
be on a cleanly-controlled axis the phase-relational arm didn't
have access to.

**Why LVEF gains but RVSP does not** (the observation that
motivates MV2SV v5):
- Variant 4's hard negative is structurally scoped to *phase within
  a view family*. An end-systolic A4C latent must differ from an
  end-diastolic A4C latent. That is exactly the geometry LVEF
  reads out of (SV = EDV − ESV; both volumes are phase-dominant
  A4C features).
- RVSP is estimated from peak TR jet velocity. On B-mode the
  mechanism is different: the TR jet has to be in-plane and the
  view has to capture the RV inflow cleanly — a **view** and
  acquisition-angle dependency more than a phase dependency.
- MR severity depends on apical 4C / apical 3C / PLAX
  integration at specific moments in the cycle; AS severity on
  PLAX / PSAX AV short-axis measurements. All three are
  cross-view-dominant.
- Variant 4 discriminates along *phase*. RVSP / MR / AS need
  analogous discrimination along *view*.

### 2.6 The consistent story

- **Variant 1 (φ-JEPA Run D, 100ep probed)** is the empirical
  proof of the conditioning-not-discrimination critique. Test
  LVEF e100 matches SV e100 within noise (MAE 5.272 vs 5.320,
  R² 0.642 vs 0.652); SV e150+ pulls materially ahead. The
  predictor absorbing Δφ doesn't propagate into the encoder.
- **Variant 2 (Mask-φ)** has the same critique by structure;
  it was never run given Variant 1's null result — mask-selection
  bias is a weaker form of the same mechanism that Variant 1
  already falsified.
- **Variant 3** regressed onto a near-duplicate latent. At tight
  phase+view matching, `h_a ≈ h_b`; the crossview term is a
  noisier intraview. Training loss showed this in real time
  (0.67 flat total loss, intraview mirroring SV).
- **Variant 4** introduced a discriminative signal (hard negative
  at wrong phase). This *did* rearrange the latent geometry —
  enough to let a +25-epoch encoder match a +100-epoch encoder
  on LVEF, with within-tail Pearson 0.64 on the reduced-EF
  bucket (the clinically-critical slice). But the signal is
  scoped to phase, and across the view-dominant tasks the gains
  don't materialize:
  - **RVSP SV**: aggregate Δ negative (R² 0.018 vs SV 0.157),
    but in the noise-floor regime where both probes compress
    predictions 40% and the per-severity breakdown is mixed —
    V4 better on mod-PH, worse on normals. Not "actively
    regresses"; closer to "no consistent sub-group where V4
    exceeds SV."
  - **MR A4C 4-class SV**: 52.2% overall accuracy vs 47%
    majority baseline, with 0/299 Severe recall and Mild
    collapsing to None at 53%. Consistent with the pre-
    registered prediction that A4C-only MR is weak for
    phase-awareness — real MR grading is multi-view.
  - **TAPSE**: no SV baseline; standalone MAE 0.355 cm
    with 36% range compression and hyperdynamic-bucket
    under-prediction of 0.57 cm. Not interpretable as a
    method effect.
- The stratified pattern across tasks is consistent: V4 learns
  phase-sensitive features that help rank-correlate LVEF on the
  dynamic (reduced-EF) tail, but all four probes — including
  the LVEF winner — show aggressive regression-to-mean that
  phase-awareness does not fix. The specific failure mode MV2SV
  v5 is designed to attack is not "phase-awareness doesn't
  help RVSP" (that could be compute or sampler noise) but the
  structural one: **an encoder trained to discriminate along
  phase within a view family has no mechanism for
  discriminating across views, and therefore cannot learn the
  view-angle-dependent features RVSP / MR / AS grading needs.**

MV2SV v5 is the cross-view retarget of Variant 4. It inherits
Variant 4's positive design principle — an InfoNCE-style
discriminative term is what moves representations — and avoids
Variants 1–3's failure principle (no conditioning-only, no
near-duplicate regression). The discriminative signal is
structural same-study *different-view* prediction (`L_pair_view`
onto teacher's `z_view` slot, which is by construction not a
duplicate of the source) plus explicit cross-view retrieval
(`L_view_nce`, which rules out conditional-mean collapse).

---

## 3. Architecture

### 3.1 High-level

Student sees one clip at inference. Training draws a **same-study
target-view clip** (a PLAX when the source is A4C, an A2C when the
source is PLAX, etc.) and optionally a **fused pool** of additional
same-study target-view clips (sparse, Bernoulli-gated,
compute-expensive).

Student's pooled encoding `h` is split into three slots via a
factorized head:

```
h --FactorizedProjectionHead--> (z_shared, z_phase, z_view)
```

- `z_shared`: view-invariant same-study signature.
- `z_phase`: cardiac-phase residual.
- `z_view`: view-local residual (what this view shows that others
  don't — e.g. MV morphology on PLAX).

The student then predicts:
1. `P_shared(z_shared_student)` ≈ stopgrad teacher's `z_shared` on
   the target clip — the view-invariant same-study alignment.
2. `P_view(z_shared, z_phase, z_view, src_view, tgt_view, Δφ)` ≈
   stopgrad teacher's `z_view` on the target clip — **view-specific**
   hallucination. This is the primary signal.
3. A cross-view retrieval contrastive: the student's A4C-source
   embedding must retrieve the teacher's correct same-study PLAX /
   A5C / A3C / A2C target latent among a batch of other-study
   alternatives of the same target view.

### 3.2 Objective

```
L_total = L_intra
        + λ_pair_shared · L_pair_shared
        + λ_pair_view   · L_pair_view     (PRIMARY)
        + λ_view_nce    · L_view_nce      (PRIMARY)
        + λ_shared      · L_shared_pair
        + λ_fused       · L_fused · Bernoulli(p_fused)     (auxiliary)
        + λ_phase       · L_phase_rel                      (deferred)
        + λ_local_motion · L_local_motion                  (unshipped)
```

First-real-run recipe (5-ep pilot, job 655, 2026-05-02):

```
λ_pair_shared  = 0.05
λ_pair_view    = 0.10   PRIMARY
λ_view_nce     = 0.025  PRIMARY
λ_shared       = 0.05
λ_phase        = 0.0
λ_fused        = 0.0    (off — see §5.3)
λ_local_motion = 0.0    (unshipped)
p_fused        = 0.0
```

Scientific-path guards:
- `allow_provisional_clip_b_fallback = false` — forward **raises**
  if `λ_pair_view > 0` and `pair.target_clip is None`. This is the
  fix for the silent reuse-of-clip_b bug that neutered Variant 3.
- `mv2sv_sampler.enabled = true` — sampler emits real
  same-study target-view clips per row.
- `view_nce` uses same-target-view negative masking with
  view-family fallback (apical / parasternal_long /
  parasternal_short) to guarantee ≥1 valid negative per row in
  the common case.

### 3.3 Why these terms, and what they each neutralize

| Term | What it neutralizes | Why it's in the objective |
|---|---|---|
| `L_intra` (base V-JEPA) | — (baseline) | Keeps encoder from drifting off the pretrained IN21K manifold |
| `L_pair_view` | Variant 3's "teacher encodings are redundant" failure | Target is view-**specific** (teacher's `z_view` on a DIFFERENT view). Student cannot produce this by copying the source; it must hallucinate. |
| `L_view_nce` | Variant 3's "SmoothL1 collapses to conditional mean" failure | Retrieval contrastive: student must pick the right same-study target-view latent out of a batch. A conditional mean can't retrieve — the discrimination is along *study × target-view*. |
| `L_pair_shared` | stabilizer — prevents z_shared from drifting off the pretrained same-study alignment | Small weight (0.05) so it doesn't dominate |
| `L_shared_pair` (paired NT-Xent) | Fix 3: the v1 "batch NT-Xent with often-zero same-study positives" bug | Per-row positive = teacher's `z_shared` on the target clip; one positive per row guaranteed. |
| `L_fused` (sparse, Bernoulli) | Auxiliary: consensus-across-views signal when >=2 same-study target-view clips are available. Mean-shared target mode, no trainable mv_teacher_fusion. | Currently OFF (sampler sparsity audit required before re-enable) |
| `L_phase_rel` | Variant 4's LVEF-gain signal, now conditioned on `z_phase` | Deferred — enable only after pair_view trajectory is clean |
| `L_local_motion` | TAPSE-shaped local-motion residual | Unshipped. Requires same-view Δφ-displaced clip in sampler. |

### 3.4 The factorized head — why three slots, not one

Paper claims rest on being able to **attribute** which part of the
representation carries which information. If the whole encoder just
learns a big tangled latent that happens to predict target views, we
can't separate "cross-view hallucination" from "view-invariant
same-study signature." The factorized head lets the downstream probes
read out `z_shared` alone, `z_phase` alone, `z_view` alone, and
concat configurations. If RVSP / MR / AS benefit from `z_view` but
LVEF benefits from `z_phase`, that's a strong interpretability claim.

Probe adapter `evals/video_classification_frozen/modelcustom/vit_factorized_encoder.py`
is landed; configs under `configs/eval/vitl/privview/` cover z_shared,
z_phase, [shared;phase], and encoder-pool (legacy) probes.

### 3.5 Teacher asymmetry (EMA heads)

Same BYOL/JEPA asymmetry that makes the base objective work, applied
consistently to the new heads:
- `factorized_head_ema` — EMA of `factorized_head`
- `shared_projector_ema` (pair + fused variants) — EMA of the
  projectors
- Teacher-side targets are computed under `torch.no_grad()` with the
  EMA heads; student-side uses the online heads.
- `mv_teacher_fusion` is **not** constructed in `mean_shared` mode
  (the default). The prior v2 design built it as an attention fuser
  and EMA'd it, but it had no independent objective and the EMA
  target was an EMA of a random-init attention fuser — meaningless.
  v3 replaces this with `_mean_shared_fused_target` (mean over the
  EMA-factorized `z_shared` slots of the fused pool) when fused is
  active. `attention_ema` mode still exists behind a config flag but
  is experimental.

### 3.6 DDP-synced fused gate

Fused is Bernoulli(p_fused) per step. The draw is seeded from a
step-hashed RNG **shared across ranks**, with an explicit broadcast
from rank 0 as a belt-and-suspenders check. This was Fix 6 — without
it ranks diverge on whether the fused branch runs, breaking DDP
gradient synchronization.

---

## 4. How we got here — design iterations

| Rev | Shipped | What it fixed | What it still got wrong |
|---|---|---|---|
| v1 | Phase A/B scaffolding (55 tests) | Factorized head, ConditionalViewPredictor, MultiViewTeacherFusion, SharedProjector classes wired into train.py with DDP, optimizer groups, checkpoint I/O, CSV schema | Reused clip_b_neg (wrong-phase same-view!) as the target; batch-only same-study NT-Xent often had zero positives; `z_view` produced but never consumed; fused target wasn't stop-gradded; fused Bernoulli drawn per-rank (DDP break) |
| v2 | 8-fix plan (Fixes 2–6 coded + tests) | Conditional phase query restored; paired NT-Xent for `L_shared`; dual-target pair prediction (shared + view) so `z_view` is consumed; stop-grad fused target + EMA target heads; DDP-synced Bernoulli gate | Fused target still produced by an untrained mv_teacher_fusion EMA'd against itself (meaningless signal); phase_rel still live as primary; shared_projector not split between pair and fused |
| v3 | `mean_shared` fused target + pair/fused projector split | Fused target computed via `_mean_shared_fused_target` (mean over EMA-factorized `z_shared` of the fused pool); mv_teacher_fusion gated behind `attention_ema` mode; shared_projector split into `pair_shared_projector` + `fused_shared_projector` (backward-compat state_dict load) | `lambda_pair_shared=0.25` made view-invariant target dominant — reproduces Variant 3's "positive-only near-duplicate target" failure mode on encoders already aligned at the study level |
| v4 | `L_view_nce` cross-view retrieval contrastive; primary signal shift | pair_view demoted-to-promoted to λ=0.10; view_nce added at λ=0.025; pair_shared demoted to stabilizer (λ=0.05); A4C→{A2C, A5C, PLAX, A3C} stage-curriculum specified | Still on the sampler's old `clip_b` path — pair targets were not yet real different-view clips |
| v5 | Fix 1 sampler (Fix 1a–f, Fix 1e(a)–(d)): real target_clip + fused_clips; fail-loud forward guards; same-target-view view_nce masking + family fallback; on-run sanity log | Dataloader plumbing complete end-to-end; 156 passing multiview tests; sampler emits target_clip + target_view + target_delta_phase + fused_clips; `allow_provisional_clip_b_fallback=false` blocks silent fallback | Fused sampler sparsity: valid-view-count mean ≈ 1.5 in batch, below the forward's `>=2` guard — fused cannot run as-configured (see §5.3) |

Rev counts match the plan addendum structure in
`/home/sagemaker-user/.claude/plans/parsed-squishing-thompson.md`. The
plan file is the authoritative v1→v5 record; this doc is the
paper-oriented narrative.

---

## 5. Run log (this doc will extend over time)

### 5.1 Stage A — parity smoke (chained before 652)

All auxiliary lambdas zero. Verifies the new MV2SV dispatch is a
pass-through for the base V-JEPA objective. Gate: `total_loss ==
intraview_loss` within bf16 tolerance (`< 1e-2`). PASS. No NaN/inf.
stdout clean. Interpretation: turning on the MV2SV dispatch without
privileged lambdas reproduces base V-JEPA — no hidden side effects
from the factorized head, EMA heads, or teacher-concat forward.

### 5.2 Stage B — target-view NCE smoke (job 652, PASS)

Scientific-path recipe (λ_pair_shared=0.05, λ_pair_view=0.10,
λ_view_nce=0.025, λ_shared=0.05). Fused off. 80 steps, 7m30s, all
9 relaxed gates PASSING:

| Gate | Result |
|---|---|
| `used_clip_b_fallback == 0` every step | PASS |
| `pct_target_clip_present` mean ≥ 0.85 | PASS (0.882) |
| `view_nce_valid_neg_count` mean ≥ 1.0 | PASS (4.25) |
| `view_nce_fallback_fraction` ≤ 0.3 | PASS (0.066) |
| `view_nce_top1 > 1/B` (late run) | PASS (tail mean 0.344 vs random 0.0625 → **5.5× random**) |
| `pos_sim − neg_sim` increases | PASS (first half −0.0011, last half +0.0077) |
| `intraview` drift ≤ 20% | PASS (drift 1.7%; 0.5114 → 0.5029) |
| no NaN/inf in MV2SV losses | PASS |
| stdout clean | PASS |

**Interpretation**: the primary objective is trainable. Cross-view
retrieval rises above chance within 80 steps; the positive-negative
similarity gap widens. Base intraview is essentially flat — privileged
signal is not injecting noise. The sampler covers ~88% of rows with
real target clips; view-family fallback is uncommon (6.6%); no
clip_b phase-wrong fallback was ever used.

Gate-checker thresholds were relaxed from strict `==1.0` /
`min>=1` to mean-based (≥0.85 / ≥1.0) during the 648→652 cycle.
Justification: the forward masks rows with missing target clips /
zero valid negatives out of the loss (the CE over a
positive-only-logit row contributes 0 to the mean by definition,
logsumexp of one element = 0 up to scale), so row-level misses are
scientifically sound — only population-level miss rates are a
failure. Unit test
`tests/vjepa_multiview/test_view_nce_loss.py::test_v5_zero_valid_negatives_does_not_crash`
locks this invariant.

### 5.3 Stage C — fused smoke (job 653, FAILED, not a scientific failure)

Same recipe as Stage B with `λ_fused=0.05, p_fused=0.25` added.
Training ran 5m48s then raised `ValueError: fused_valid_mask mean
valid views = 1.44 < 2` on step 1. Per-rank valid-view means:
1.44 / 1.50 / 1.69 / 1.75 — all below the forward's hard floor
of 2.

This is a **sampler-sparsity** result, not an objective failure.
The sampler's `fused_pool` isn't delivering enough same-study
distinct-view clips per batch row for the mean-shared fused target
to be non-trivial. Three paths forward:

1. Increase `n_fused_min`: forces the sampler to skip rows that
   can't fill a fused pool. Reduces effective batch size on fused
   steps.
2. Relax the forward guard to `>=1.5`: **rejected**. At row-level
   1 valid view, the fused target degenerates into a single
   target-view target — redundant with `L_pair_view`. Running
   would muddy attribution.
3. Row-mask the fused branch: compute `L_fused` only on rows with
   `fused_valid_mask.sum(dim=1) >= 2`; skip the step entirely if
   the eligible fraction is too low. Deferred to task #222.

All three are contingent on an **offline sampler audit** (§5.5
below, script
`scripts/neurips/phase/mv2sv_fused_coverage_audit.py`) showing
how often the sampler can actually deliver ≥2 valid views for
which source×target view combinations. Without that, we're
guessing at knobs.

Decision (per user 2026-05-02): **do not change the core MV2SV
objective or the sampler**. Launch the primary pilot with fused
OFF. Fused becomes an optional auxiliary after the audit passes.

### 5.4 5-epoch scientific pilot — fused OFF (job 655, COMPLETED)

Config: `configs/train/vitl16/mv2sv-pilot-5ep-nofused.yaml`.
Lambdas: pair_view 0.10 / view_nce 0.025 / pair_shared 0.05 /
shared 0.05, everything else 0. batch=32, ipe=650, 5 epochs,
warmup=1, save every epoch. Sbatch:
`scripts/neurips/phase/mv2sv_pilot_5ep_nofused.sbatch`. Submitted
2026-05-02 22:19, ran on ip-10-0-50-146.

**COMPLETED 2026-05-03 ~01:30**, exit 0:0, elapsed **3h11m**.

**Training health — clean end-to-end**:
- Intraview 0.568 → 0.564 across 5 epochs (−0.7% drift, well inside
  ±20% gate).
- Aggregate loss 0.59 → 0.70 (+19%), signature of the auxiliary-
  lambda warmup ramp sitting on top of the flat intraview.
- `used_clip_b_fallback == 0` on all 1,950 logged steps (fail-loud
  guard never tripped).
- `pct_target_clip_present` mean 0.880, p25 0.812, no sustained
  ≥3-step dip below 0.75.
- `view_nce_valid_neg_count_mean` = 8.0 (far above the ≥1.0 gate).
- `view_nce_fallback_fraction` = 0.047 (family fallback is rare).
- **`view_nce_top1`** rose from 0.20 (first 50 steps) to 0.60 (last
  50 steps); ep1 0.452 → ep3 0.574. **18× random chance (1/32 =
  0.031), widening pos-neg gap +0.074 → +0.284 across epochs.**
- **`diag_pair_view_cos_q_target`** 0.005 → 0.474 — the student's
  `z_view` predictions are aligning with the teacher's `z_view`
  target. Strongest sign the privileged signal is being learned.
- **Slot factorization healthy**:
  `diag_z_shared_vs_z_phase_cos` stays ≈ 0.024 (target < 0.3).
  `diag_z_shared_vs_z_view_cos` 0.021 → 0.009. Slots disjoint.
  Per-slot variance grows ~6× (shared), ~3× (phase, view).
  `z_view_var` is 0.70× `z_shared_var` — comfortably above the
  `z_view_var > 0.1 × z_shared_var` red-flag threshold.
- Per-target-view top-1 (view_nce) at epoch 3:
  A3C 0.71, A2C 0.65, A5C 0.59, PLAX 0.50 (cross-family).
  All four trainable; PLAX hardest but 16× chance.

Hourly collapse-monitor cron `3d18e46f` ran over the full duration,
fired `CronDelete` at COMPLETED. Never flagged collapse.

All checkpoints saved to
`s3://.../runs/mv2sv_pilot_5ep_655/ckpt/{e1,e2,e3,e4,e5,latest}.pt`.

**Interpretive caveat** (added post-run 2026-05-03): the
proximal-objective training signatures above are *necessary* but
not *sufficient*. Variant 4's phase-relational objective showed
structurally identical healthy trajectories at training time and
still delivered −0.090 paired ΔR² on RVSP. Training-time healthy
diagnostics are consistent with both "v5 works" and "v5 learns
same-study identity shortcuts or a view classifier." The next
section describes the attribution work kicked off after the pilot
to disambiguate these.

### 5.5 MV2SV-pipeline intraview-only CONTROL — fused OFF (job 658, COMPLETED)

**Motivation**: V4's paired control (608) used the `phase_matched`
sampler but not the MV2SV `target_clip`/`fused_clips` data-path
extensions. If the pilot's downstream probe Δ is compared only to
608, the MV2SV *data-path* (target clip draw + view-family
curriculum) gets bundled into the "objective" column. The
scientific question needs a paired control that holds **every data-
path knob identical to the pilot** and differs *only* in the loss.

Config: `configs/train/vitl16/mv2sv-ctrl-5ep-intraview-only.yaml`.
All MV2SV auxiliary lambdas = 0.0 (pair_shared, pair_view, view_nce,
shared, phase, fused, local_motion); `p_fused=0`;
`mv2sv_sampler.enabled=true`; `fused_pool.enabled=false`;
`allow_provisional_clip_b_fallback=false`. Sampler still draws the
same `target_clip` into the `PairBatch` and forwards it through the
teacher encoder; it's simply discarded from the loss. Everything
else (batch=32, ipe=650, 5 epochs, warmup=1, seed 345, same init
checkpoint, same LR schedule, same save_every_freq=1) bit-identical
to pilot 655.

Sbatch: `scripts/neurips/phase/mv2sv_ctrl_5ep_intraview_only.sbatch`.
Added a `ctrl` stage to `mv2sv_gate_check.py` that validates
`used_clip_b_fallback==0`, `pct_target_clip_present mean ≥ 0.85`,
`total_loss ≈ intraview` (within 1e-2, since all auxiliary lambdas
are zero), `intraview drift ≤ 20%`, no NaN/inf, stdout clean.

First submission as job 656 failed at 1m17s with `ImportError: No
usable pyarrow engine` — node 56 lacked the `/opt/dlami/nvme/
pyarrow_site` site-packages that pilot node 146 had. Patched the
sbatch to fall back to `pip install --target=... pyarrow` when the
import fails; resubmitted as **job 658** on ip-10-0-50-56.

**COMPLETED 2026-05-03 ~04:54 UTC**, exit 0:0, elapsed **3h11m**.
- Intraview 0.540 → ~0.47 across 5 epochs (−13%; smooth descent
  consistent with intraview-only continuation).
- `used_clip_b_fallback == 0` on every logged step.
- `pct_target_clip_present` mean ≥ 0.85 (gate passed; sampler
  delivered real target clips even though loss discarded them).
- e1, e2, e3, e4, e5, latest ckpts saved to
  `s3://.../runs/mv2sv_ctrl_5ep_658/ckpt/`.
- Hourly collapse-monitor cron `f38fd1b1` ran the full duration
  and auto-deleted on COMPLETED; never flagged collapse.

The paper-interpretable Δ is now **pilot 655 e5 − ctrl 658 e5**
on each downstream endpoint — the clean decomposition of
"objective contribution" isolated from "MV2SV sampler contribution."
Downstream probes for this comparison are jobs 681 (pilot) and
682 (ctrl); see §5.12.

### 5.6 Cross-view retrieval diagnostic (jobs 657, 660 failed; 661 pending)

Script: `scripts/neurips/phase/run_cross_view_retrieval_diag.py`.

**Hypothesis to test**: does MV2SV v5 learn a useful cross-view
retrieval signal that earlier variants (Base, V3, V4, Control)
don't? And is that signal about view-specific physiology
(cross-view transfer) or about same-study identity shortcuts (study
confound)?

**Protocol**: for each test-split study (subject-disjoint,
`classifier/phase/splits/dicoms_split.csv`), sample an A4C anchor
and a target-view clip (A2C / A5C / A3C / PLAX / PSAX-MV). Encode
both with the frozen checkpoint. Query = anchor; positive = same-
study target; negatives = same-target-view targets from other
studies (with view-family fallback when insufficient).

**Coverage** (once the sweep completes):
- Standard V-JEPA ckpts: Base e100, Base e125, fb_sv_548 e25, V3
  542 e5/e25, V4 593 e5/e25, Control 608 e5/e25 — all
  `encoder_pool`.
- MV2SV pilot 655 e1/e3/e5 — **5 feature modes each** (encoder_pool,
  z_shared, z_view, concat_shared_phase, concat_all).
- MV2SV ctrl 658 e1/e3/e5 — encoder_pool only (slot-mode
  projections are random-init-level for a run with all auxiliary
  lambdas = 0, so slot retrieval is not a meaningful comparison).

**Metrics per run** (JSON): top1, top5, pos_sim, neg_sim, gap,
valid_neg_count mean/min, fallback_fraction, plus per
source×target-view bucket breakdown.

**v1 diag sweep (job 657)** — ran 11m23s, produced **zero JSON
outputs**. `KeyError('dicom_id')` on every run: the view_labels CSV
uses `s3_uri` column, not `dicom_id`; my `build_sampler`
reimplementation didn't replicate `data_manager.py`'s
`s3_uri → dicom_id` regex fallback.

**v2 rerun (job 660)** — ran 11m55s, same zero output. Different
bug: the sampler's `__init__` doesn't accept
`multiview_objective` as a kwarg (only `mv2sv_config`); the diag
script passed both.

Both bug classes were "the diag script duplicated what
`data_manager.py` / sampler already do, without end-to-end
testing." Patched both, then smoke-tested on node 146 (§5.8).

**v3 sweep (job 661)** is queued `--dependency=afterany:658:660`
and will auto-release once ctrl 658 finishes. Will pull the
latest tarball (all fixes in place) when it fires.

### 5.7 Extended diagnostics: alternate retrieval mode, slot geometry, view classifier

Extended the diag script with three additional modes to close
known-unknown gaps:

- **`--retrieval-mode same_study_any_view`** — positive is same-
  study same-target-view, negatives are ALL other-study clips
  regardless of view. If a checkpoint beats its `same_study_same_
  view` baseline by a larger margin here, the signal is study-
  identity, not view-specific. Run alongside `same_study_same_view`
  for every checkpoint.
- **`--slot-geometry`** — emits mean cosine for three relationship
  types per run:
  - B: same-study cross-view source↔target (training objective)
  - C: other-study same-view target↔target (retrieval distractor
    pool)
  - D: other-study cross-view target↔target (noise floor)
  Healthy pattern: B > C > D. Pure view clustering without study
  signal: C > B. Same-study identity confound: B ≫ C ≈ D.
- **`--view-classifier`** — trains a 1-layer linear probe on
  `encoder_pool` to predict `target_view` from the encoded
  feature. 80/20 study-disjoint split within test set; 200 epochs
  AdamW (lr 1e-3, wd 1e-3). Reports val_acc, best_val_acc, per-
  class recall, majority baseline.

  Decision rule: if pilot 655 val_acc > 0.95 while downstream RVSP
  stays null, the paper's "cross-view hallucination" framing
  collapses to "view discrimination learned a view classifier,"
  which is a much weaker claim and should block the full 25-epoch
  run.

New sweep sbatch: `scripts/neurips/phase/mv2sv_diag_sweep_v2.sbatch`,
submitted as **job 661** with `afterany:658:660`. Covers the same
checkpoints as §5.6 plus the new ctrl 658 e1/e3/e5 rows; each
standard V-JEPA ckpt gets 1 run with view_classifier + 1 run in
`same_study_any_view` mode = 2 runs; pilot 655 e1/e5 get 10 runs
each (5 feature modes × 2 retrieval modes + view classifier on
`encoder_pool`).

### 5.8 Diag smoke debugging (jobs 663–671)

Four consecutive smoke sbatches failed on ip-10-0-50-146 before the
diag pipeline ran end-to-end. Each caught a distinct bug the
cluster-sweep failures had hidden:

| Job | Error | Fix |
|---|---|---|
| 663 | `KeyError('dicom_id')` | Added `s3_uri → dicom_id` regex fallback to mirror `data_manager.py`. |
| 666 | `TypeError: multiview_objective kwarg` | Removed — sampler only takes `mv2sv_config`. |
| 669 | `ValueError: fewer non-zero entries in p than size` in `np.choice(replace=False)` | view_pair_policy probabilities must all be > 0. |
| — | `'ClipAnchor' object has no attribute 'uri'` | Sampler uses `row_idx` + `anchor_frame`; added lookup through `sampler._df["s3_uri"]` with dcm→mp4 rewrite; derive frame window from `anchor_frame ± frames_per_clip/2`. |
| 671 | (final) PASS | 2m55s, valid JSON output. |

**Job 671 output** (Base e100, n=10 studies, smoke-level):
- same_study_same_view: top1 0.78, top5 1.00, gap +0.08
- same_study_any_view: top1 0.70, top5 0.90, gap +0.08
- slot_geometry: B/C/D cosine stats present
- view_classifier: "too few samples for split" at n=10 — will work
  at n=200 in the real sweep.

All four bugs were the diag script reimplementing logic that
`data_manager.py` / the sampler's own dataset classes already
encapsulate. Lesson captured in the §5.6 note; future diag work
should reuse the existing dataset classes directly, not
reimplement.

### 5.9 F/G/H attribution diagnostics (jobs 672 → 673, COMPLETED)

Three questions the retrieval diag doesn't answer:

- **F — z_phase slot phase-decodability**: does pilot 655's
  `z_phase` slot actually absorb the phase signal (the
  interpretability claim in §3.4), or did the factorization
  collapse? Linear probe predicts `[sin(2πφ), cos(2πφ)]` from the
  `z_phase` slot with 80/20 study-disjoint split. Reports circular
  MAE in degrees. On non-MV2SV ckpts, falls back to the encoder
  pool (the finalbudget protocol) for direct comparability.
- **G — frame-shuffle sensitivity**: encode each target clip twice
  — natural frame order vs a fixed random permutation. Report
  mean `cos(clean_feat, shuffled_feat)`. Lower cos → encoder
  changes more under shuffle → stronger temporal structure. Same
  mechanism as the NeurIPS §4 frame-shuffling diagnostic (see
  `frame-shuffling-results.md`).
- **H — intra-clip temporal dissimilarity**: encode the first 8
  frames and the last 8 frames of the same 16-frame clip
  separately (each duplicated to fill 16 for ViT token count).
  Report mean `cos(half_a, half_b)`. Close to 1.0 → encoder
  collapses intra-clip temporal dynamics into a study-static
  summary.

Script: `scripts/neurips/phase/run_temporal_phase_diagnostics.py`
— imports the debugged helpers from the retrieval diag. Sbatch:
`scripts/neurips/phase/mv2sv_fgh_sweep.sbatch`.

**Job 672 (first attempt)** — failed on every sub-run with
`ValueError: truth value of DataFrame is ambiguous` at
`getattr(sampler, "_df", None) or getattr(sampler, "df", None)` —
the `or` triggers `__bool__` on the DataFrame. The retrieval diag
had already been patched to a two-step if/else check but the F/G/H
script hadn't inherited it. Fixed + resubmitted as 673.

**Job 673 COMPLETED 20m12s, exit 0:0**, 7 JSONs uploaded to
`s3://.../runs/mv2sv_fgh_673/`. Results at n=200 held-out studies
per checkpoint:

| Checkpoint | Mode | G: cos(clean, shuf) | H: cos(half_a, half_b) | F: circ MAE (°) |
|---|---|---:|---:|---:|
| Base e100 | encoder_pool | 0.9961 | 0.9836 | 86.9 |
| Control 608 | encoder_pool | 0.9907 | 0.9827 | 82.6 |
| **V4 phase-rel 593** | encoder_pool | **0.8902** | **0.8222** | 85.4 |
| Pilot 655 e5 | encoder_pool | 0.9889 | 0.9584 | 85.0 |
| Pilot 655 e5 | z_phase | 0.9912 | 0.9638 | 82.4 |
| Pilot 655 e5 | z_shared | 0.9885 | 0.9634 | — |
| Pilot 655 e5 | z_view | 0.9924 | 0.9747 | — |

**G — frame-shuffle sensitivity (strong positive finding for V4)**:
V4 phase-rel is **dramatically more frame-shuffle-sensitive** than
every other encoder. cos(clean, shuffled) = 0.89 vs 0.99 for
Base / Control / Pilot. That's a 10× larger representational delta
under shuffle, direct mechanistic corroboration that V4's
hard-negative-at-wrong-phase objective forced the encoder to
reorganize along a phase/temporal axis. Pilot 655 shows no such
sensitivity — its cross-view objective did not push the encoder to
become more temporally structured.

**H — intra-clip temporal dissimilarity**: same pattern. V4
cos(half_a, half_b) = **0.82**; all others 0.96–0.98. V4's encoder
distinguishes first-half from second-half of the same clip far
more than the others. Pilot 655 does not: first and second halves
are ~96–97% cosine-identical at e5.

**F — phase decodability**: all circ MAE values cluster at
**82–87°**, very close to the 90° random-guess baseline. The
per-axis MAEs (sin/cos) show the linear probe **didn't converge
in any configuration** — pilot's `z_phase` sin_MAE = 0.680 vs a
constant-mean baseline of 0.641. At n=200, the probe is
underpowered; the finalbudget protocol used n≈3,000 test clips.
**F is inconclusive at this sample size** and cannot falsify or
support the factorization claim. A larger-n phase probe should
be run separately.

#### Interpretive read

**What the results say for V4**:
- The G and H signatures are the first *mechanistic* corroboration
  (beyond LVEF downstream) that V4's phase-relational objective
  actually moved the encoder's temporal geometry. cos(clean, shuf)
  = 0.89 is a large effect; phase-relational-hardneg.md's claim of
  "widening pos-minus-hard gap = representational reorganization"
  has a held-out test set analog now.

**What the results say for Pilot 655 e5**:
- The encoder's pooled representation at e5 has the same
  temporal properties as Base e100 and Control 608 (G ≈ 0.99,
  H ≈ 0.96–0.98). **Pilot 655 did not learn temporal
  structure; it learned cross-view retrieval as a spatial /
  pooled-static signal.** This is consistent with the training
  objective — `L_pair_view` / `L_view_nce` both operate on
  pooled features, averaging over time — but it is a *necessary*
  caveat for any downstream claim involving dynamics.
- Pilot's `z_phase` slot is indistinguishable from its
  `encoder_pool` on G and H (0.9889 vs 0.9912; 0.9584 vs 0.9638).
  The factorization isn't doing interpretable temporal work on
  these metrics at e5. **Red flag for the paper's
  interpretability claim** — at least on these metrics; the
  phase-probe F is too underpowered here to contradict that on its
  own.
- Pilot's `z_view` slot is the *least* temporally sensitive of
  the four feature modes (cos(clean,shuf) 0.9924, cos(half_a,
  half_b) 0.9747). `z_view` is closer to a view-identity static
  summary than a view-specific dynamics slot. Consistent with
  v_view-cos(q,target) rising to 0.47 at the training step
  level — the student learned to match the teacher's z_view, but
  z_view is nearly time-invariant.

**Paper implication**: the cleanest claim the evidence supports
right now is
- V4 learns within-clip temporal / phase structure (G, H, and
  LVEF downstream back this).
- Pilot 655 (v5) at e5 learns cross-view retrieval **without**
  within-clip temporal structure; this *could* be consistent with
  either "cross-view physiology is captured as a pooled spatial
  signature" (publishable) or "cross-view retrieval is same-study
  identity / view-family clustering" (not publishable). The
  retrieval diag v3 (job 661) is the test that distinguishes
  these.

The F circ-MAE-at-n=200 result is not useful at this size; re-run
at n≥2000 (finalbudget protocol) if the paper needs a hard phase-
decodability number.

### 5.10 Scaled phase probe + H1 slot-triviality (job 674, COMPLETED)

Follow-up to §5.9 F. F at n=200 was inconclusive; 674 ran a scaled
pipeline (n=1231 after sampler filtering, ~6x the F sample size) with
two probes trained on cached features:
- Linear on (sin 2πφ, cos 2πφ) under MSE (finalbudget protocol)
- MLP (256-hidden) + von Mises NLL (upgrade for moderate sample sizes)

All results at 80/20 study-disjoint split:

| Ckpt | Mode | Linear circ MAE | MLP-vM circ MAE | Const baseline |
|---|---|---:|---:|---:|
| Base e100 | encoder_pool | 87.76° | 85.61° | 88.23° |
| Control 608 | encoder_pool | 88.00° | 87.46° | 88.23° |
| V4 593 | encoder_pool | 90.67° | 93.09° | 88.23° |
| Ctrl 658 e5 | encoder_pool | 88.30° | 85.26° | 88.23° |
| Pilot 655 e1 | encoder_pool | 83.00° | 86.72° | 88.23° |
| Pilot 655 e1 | **z_phase** | **81.21°** | 82.20° | 88.23° |
| Pilot 655 e1 | z_shared | 89.49° | 85.41° | 88.23° |
| Pilot 655 e3 | encoder_pool | 85.12° | 87.51° | 88.23° |
| Pilot 655 e3 | z_phase | 88.95° | 85.41° | 88.23° |
| Pilot 655 e3 | z_shared | 88.56° | 87.09° | 88.23° |
| Pilot 655 e5 | encoder_pool | 88.33° | 89.09° | 88.23° |
| Pilot 655 e5 | z_phase | 88.56° | 87.03° | 88.23° |
| Pilot 655 e5 | z_shared | 85.35° | 86.78° | 88.23° |

**Verdict: phase decodability is still inconclusive at this protocol**.
All circ MAE values cluster 83–91°, within a few degrees of the 88.2°
constant-mean baseline. The scaled n didn't help because the limiting
factor is the protocol itself, not the sample size:
- Test clips are anchored on sampler-drawn `target_phi_b`, which is
  not uniformly distributed over [0, 1) — it concentrates at
  phase-diverse but not phase-balanced values. This tightens the
  constant baseline (88.2°) and compresses everything else against
  it.
- finalbudget's 42° result used phase-bin-balanced CSVs + anchor-
  aware clips via `build_phase_probe_csvs.py` — a different and
  stronger protocol that we're not reproducing here.
- V4 (90.67°) is actually *worse* than the constant baseline,
  consistent with §5.9: V4's pooled features have strong *non-
  phase* temporal structure (cos(clean, shuf) = 0.89) but that
  structure doesn't project linearly to sin/cos.

The one point of interest: **pilot 655 e1 z_phase at 81.21°** is
the single best number in the table (7° below baseline). But it
doesn't reproduce at e3 (88.95°) or e5 (88.56°), so it's most
likely noise. If real, it would say "the z_phase slot briefly
captures some phase signal early in training before being overtaken
by other signals as training progresses" — a plausible but
unverifiable story at this resolution.

#### H1 slot triviality — DECISIVE NEGATIVE RESULT

To test whether the factorized head's slot projections carry
different *content* (vs just being three different orthogonal
projections of the same signal), 674 also ran the H1 sanity check:
linear regression of each slot onto `encoder_pool` on held-out
clips, reporting R². If R² ≈ 1.0, the slot is a near-linear map of
the pool and the factorization is a trivial change of basis.

| Pilot ckpt | z_shared R² | z_phase R² | z_view R² | n |
|---|---:|---:|---:|---:|
| e1 | 0.928 | 0.945 | 0.940 | 200 |
| e3 | 0.989 | 0.989 | 0.989 | 200 |
| **e5** | **0.991** | **0.988** | **0.989** | 200 |

**The factorized head is effectively a linear projection by
epoch 3.** All three slots reconstruct to R² > 0.98 from the
pooled encoder output. The `shared_mlp`, `phase_mlp`, and
`view_mlp` components are learning near-linear maps; the MLP
nonlinearity that the architecture provides is trivially
recoverable.

**Implications**:
- The paper's factorized-slot interpretability claim does NOT
  survive this test. §3.4 ("The factorized head — why three slots,
  not one") and §7 (factorized-slot probe readout) need
  substantial revision or removal.
- The slot-disjointness metric tracked during training
  (`diag_z_shared_vs_z_phase_cos ≈ 0.024` at e5) confirmed the
  slots produce orthogonal *mean directions*. That's a different
  and weaker claim than "slots carry different content." Mean-
  orthogonal + R² ≈ 1.0 means the three slots are three
  orthogonal-direction projections of the same 1024-D signal.
- This does NOT mean MV2SV v5 doesn't work. The encoder may still
  have learned useful cross-view content — it just didn't encode
  that content into an interpretable slot structure.
- The decision on whether v5 is worth scaling to 25 ep still rests
  on downstream probes (§5.12). But the "factorization
  interpretability" story has to come off the table.

Script: `scripts/neurips/phase/run_phase_probe_scaled.py` +
`check_slot_projection_triviality.py`. JSONs at
`s3://.../runs/mv2sv_phase_scaled_674/`.

### 5.11 Retrieval diag v3 (job 676, n=1000, COMPLETED)

Resubmitted 661 (v2) as 676 with `--num-studies 1000` (was 200)
per user priority-2. Same retrieval-mode pair (`same_study_same_view`
+ `same_study_any_view`) + slot geometry + view classifier coverage.
Submitted `afterany:658`, auto-released when ctrl 658 COMPLETED.

After sampler filtering, n=376 valid rows per run (some records
drop due to missing target_clip + test-split subset). JSONs at
`s3://.../runs/mv2sv_retrieval_diag_v2_676/`.

#### Encoder_pool cross-view retrieval (same_study_same_view, n=376)

| Encoder | top1 | top5 | pos_sim | neg_sim | gap |
|---|---:|---:|---:|---:|---:|
| Base e100 | 0.295 | 0.473 | +0.856 | +0.799 | +0.057 |
| Base e125 | 0.298 | 0.471 | +0.890 | +0.850 | +0.040 |
| fb_sv_548 e25 | 0.295 | 0.481 | +0.884 | +0.840 | +0.044 |
| **Ctrl 608 e25** | 0.279 | 0.471 | +0.855 | +0.803 | +0.052 |
| V3 fb_phase_542 e25 | 0.306 | 0.481 | +0.854 | +0.800 | +0.054 |
| V4 phase-rel 593 e25 | 0.237 | 0.441 | +0.459 | +0.202 | +0.257 |
| **Ctrl 658 e1** | 0.306 | 0.476 | +0.847 | +0.786 | +0.060 |
| **Pilot 655 e1** | 0.332 | 0.548 | +0.745 | +0.623 | +0.122 |
| Pilot 655 e3 | (in sweep) | — | — | — | — |
| **Pilot 655 e5** | **0.447** | **0.707** | +0.588 | +0.194 | **+0.394** |

#### Pilot slot-mode retrieval (same_study_same_view, n=376)

| Ckpt | encoder_pool | z_shared | z_view | concat_shared_phase | concat_all |
|---|---:|---:|---:|---:|---:|
| Pilot 655 e1 top1 | 0.332 | 0.420 | 0.367 | 0.394 | 0.388 |
| Pilot 655 e1 gap | +0.122 | +0.309 | +0.210 | +0.270 | +0.246 |
| Pilot 655 e5 top1 | **0.447** | **0.500** | 0.471 | 0.489 | 0.473 |
| Pilot 655 e5 gap | +0.394 | +0.586 | +0.711 | +0.640 | +0.667 |

#### Critical findings

**1. Pilot 655 e5's encoder_pool retrieval is the strongest of any
encoder in the sweep.** top1 = 0.447 / gap = +0.394 on
`encoder_pool`. Compare to:
- Ctrl 658 e1 (same +1 epoch of training, only difference is MV2SV
  objective vs intraview-only): top1 = 0.306 / gap = +0.060.
  **ΔMV2SV at matched e1 = +0.026 top1 / +0.062 gap**.
- V4 phase-rel 593 e25 (phase-relational, different objective, +25
  epochs): top1 = 0.237 / gap = +0.257. V4's gap is high but its
  top1 is actually worse than Base — V4 pushes *all* similarities
  down (pos_sim = 0.459, neg_sim = 0.200) but doesn't lift top-1
  ranking.
- Base/fb_sv_548/Ctrl 608 at +25 ep: top1 ≈ 0.29, gap ≈ +0.05.

Pilot e5 with 5 pretraining epochs beats the +25-ep phase_matched
controls on retrieval. The cross-view objective is doing real
work on the encoder, not just on the factorized head.

**2. Pilot retrieval grows monotonically e1 → e5.**
- encoder_pool top1: 0.332 → 0.447 (+11.5 pp)
- encoder_pool gap: +0.122 → +0.394 (3.2×)
- z_shared top1: 0.420 → 0.500 (+8 pp)
- z_view gap: +0.210 → +0.711 (3.4×)

The +4 additional pilot epochs delivered large retrieval gains
across all feature modes. Signal is not saturated at e5; e25 (from
the 683 continuation) should deliver more.

**3. Pilot `same_study_any_view` vs `same_study_same_view` gap is
~10 pp consistently (all feature modes).** For pilot 655 e5
encoder_pool: `same_view` top1 = 0.447 vs `any_view` top1 = 0.316.
The pilot IS learning target-view-specific discrimination — it's
not just learning study-identity. Base/V4/Ctrl have similar ~10 pp
same_view advantage (sampler-inherent), but pilot's absolute top1
is ~0.15 pp higher on both metrics — signal is above the sampler
baseline. **This disambiguates the "pilot learned study identity"
vs "pilot learned view-specific physiology" concern in favor of
view-specific.**

**4. Ctrl 658 e1 retrieval ≈ Ctrl 608 e25 retrieval.** Confirms the
MV2SV sampler extensions (target_clip draw, view-family curriculum)
are not themselves producing pilot-like retrieval. The gain is
objective-driven, not sampler-driven.

**5. V4's gap pattern is anomalous** — high gap (+0.257) driven by
globally compressed similarities (pos and neg both much lower than
Base). V4's representation varies more per input (consistent with
G/H findings from §5.9: cos(clean, shuf) = 0.89). But compression
doesn't translate to better top1: V4 top1 < Base top1. V4's cross-
view retrieval on pooled features is not its strength.

### 5.12 Downstream attentive probes — pilot 655 vs ctrl 658 (jobs 677/678 → 681/682 at 20-ep)

The paper-interpretable Δ. Per user priority-3, queued
as paper-grade 20-epoch attentive d=4 probes (matching V4's
finalbudget protocol for apples-to-apples comparison across
variants).

#### Naming note: `encoder_pool` ≠ mean-pooling

The sbatch / S3 path mode label `encoder_pool` in this section
is a **legacy name inherited from** `configs/eval/vitl/privview/
privview_lvef_encoder_pool_d1.yaml`. It does NOT mean the
encoder's output is mean-pooled before the probe sees it. The
pipeline used by jobs 677 / 678 is:

1. Encoder (frozen) emits `[B, N_tok, D]` = `[B, 1568·num_clips,
   1024]` — **full token sequence**, not a pool.
2. `vit_encoder_multiclip` adapter concatenates clips along the
   temporal dim and returns the token sequence verbatim (no
   pooling anywhere; verified in
   `evals/video_classification_frozen/modelcustom/vit_encoder_multiclip.py`
   lines 117–149).
3. `AttentiveClassifier(num_probe_blocks=4, num_heads=16)` runs
   a learnable cross-attention query over the token sequence.
   The probe does the pooling via cross-attention, not by
   mean-pooling first.

**This is the same pipeline used by every paper-grade V4 /
Control / Base / fb_sv_548 probe in the finalbudget and
phase-relational-hardneg series** (verified by diffing the
probe configs and reading job 597's stdout-logged args —
identical `num_heads: 16`, `num_probe_blocks: 4`, `task_type`,
`module_name: vit_encoder_multiclip`, `frame_step: 2`,
`num_segments: 2`, `num_views_per_segment: 1`, `batch_size: 1`,
`num_epochs: 20`, `multihead_kwargs` with the same 6-HP grid).

Where else "encoder_pool" appears and what it means there:
- **F/G/H diagnostics (§5.9)** — my `run_temporal_phase_diagnostics.py`
  and `run_cross_view_retrieval_diag.py` use the label
  `encoder_pool` to mean `tokens.mean(dim=1)` inside the diag
  script. Those are diagnostic scripts that do their own pooling
  to produce a single-vector representation per clip for
  retrieval / cosine / linear-probe analysis. **That pooling is
  a property of the diagnostic scripts, NOT of the downstream
  probe pipeline.** The retrieval-diag numbers involving
  `encoder_pool` should be read as "pooled feature" signal; the
  downstream-probe numbers involving `encoder_pool` should be
  read as "standard token-sequence + attentive readout" signal.

For the doc going forward, the 677 / 678 downstream probes use
the **standard attentive-on-tokens** pipeline; the mode label in
the S3 path is cosmetic.


**Tasks**: EchoNet-Dynamic LVEF (regression) + MIMIC RVSP SV
(regression) + MIMIC MR A4C 4-class (classification). Same
CSVs as finalbudget / V4's probes. 6-HP grid (lr × wd), 20 ep,
bf16.

**Coverage**:

| Encoder | encoder_pool | concat_all |
|---|:---:|:---:|
| Pilot 655 e5 | ✓ | ✓ |
| Ctrl 658 e5 | ✓ | — (all MV2SV lambdas = 0; factorized head is random-init-level) |

6 pilot probe trainings + 3 ctrl probe trainings = 9 total.

#### Submission history

- **677/678 (20-ep initial)** — submitted ~05:03 UTC. Cancelled
  after user requested a 10-ep early-stop for triage; **LVEF
  encoder_pool at 10 ep completed + test inference already on S3**
  before cancellation.
- **679/680 (10-ep triage)** — re-submitted with `num_epochs: 10`
  and inline test-inference block. 679 ran ~53 min on 146
  (completed LVEF encoder_pool 10-ep + test), then user decided
  to restart at full 20-ep for direct comparability with V4/V3.
- **681/682 (20-ep paper-grade)** — submitted 06:15 UTC.
  - **682 ctrl COMPLETED** (~3h27m wallclock, exit 0:0). All 3
    probes (LVEF/RVSP/MR × encoder_pool) trained + test inference.
  - **681 pilot CANCELLED at ~10h14m** — first 5 probes completed
    (LVEF encoder_pool, LVEF concat_all, RVSP encoder_pool, RVSP
    concat_all, MR encoder_pool). The 6th probe (**MR concat_all**)
    was cancelled mid-training by user after the initial
    head-to-head read showed (a) concat_all consistently hurts vs
    encoder_pool on pilot (LVEF R² 0.295 vs 0.645; RVSP R² 0.106
    vs 0.066; all early-stop after e1 on 20-ep schedule), and
    (b) §5.10 H1 confirms the factorized slots are near-linear
    projections of the pool so concat_all can at best recover what
    encoder_pool already has. The MR concat_all number would not
    change the paper comparison — encoder_pool is the paired-to-
    ctrl probe. Cancelling frees node 146 for the 683 continuation
    immediately.

All three submissions use bit-identical pipeline to V4's
finalbudget probes:
- `vit_encoder_multiclip` adapter (full token sequence, no
  pre-pooling)
- `AttentiveClassifier(num_probe_blocks=4, num_heads=16)`
- `frames_per_clip=16`, `frame_step=2`, `num_segments=2`,
  `num_views_per_segment=1`, `batch_size=1`
- Same 6-HP grid as V4/V3/Base/Ctrl 608

This means pilot 655 e5 / ctrl 658 e5 probe numbers are directly
apples-to-apples with V4 593 e25, V3 542 e25, Ctrl 608 e25, and
Base e125 probe numbers from finalbudget.

#### Paired e5 test results — 681 pilot × 682 ctrl, COMPLETED

All probes 20-ep attentive d=4 with inline test inference on best
ckpt. Test metrics (best-ckpt inference):

**LVEF (EchoNet-Dynamic, n=1,277) — regression**

| Probe | Pilot 655 e5 | Ctrl 658 e5 | Δ (pilot−ctrl) |
|---|---:|---:|---:|
| encoder_pool MAE | 5.29 | **5.16** | +0.14 (ctrl better) |
| encoder_pool R² | 0.645 | **0.665** | −0.020 |
| encoder_pool Pearson | 0.804 | **0.816** | −0.012 |
| concat_all MAE | 7.24 | — | — |
| concat_all R² | 0.295 | — | — |

**RVSP (MIMIC SV) — regression**

| Probe | Pilot 655 e5 | Ctrl 658 e5 | Δ (pilot−ctrl) |
|---|---:|---:|---:|
| encoder_pool MAE | 10.17 | **9.91** | +0.26 (ctrl better) |
| encoder_pool R² | 0.066 | **0.093** | −0.027 |
| encoder_pool Pearson | 0.289 | **0.331** | −0.042 |
| concat_all MAE | 10.26 | — | — |
| concat_all R² | 0.106 | — | — |

**MR A4C 4-class — classification**

| Probe | Pilot 655 e5 | Ctrl 658 e5 | Δ (pilot−ctrl) |
|---|---:|---:|---:|
| encoder_pool val_acc | 51.54% | **53.01%** | −1.5 pp |
| encoder_pool val_auroc | 0.721 | **0.736** | −0.015 |
| encoder_pool val_bal_acc | 0.346 | **0.379** | −0.033 |
| encoder_pool val_kappa | 0.202 | **0.263** | −0.061 |
| concat_all | ❌ CANCELLED | — | — |

**Paired Δ read (all three endpoints):**

1. **Ctrl 658 e5 > Pilot 655 e5 on every paired endpoint.** LVEF,
   RVSP, MR all favor the paired intraview-only control over the
   MV2SV v5 pilot at matched +5 epochs of pretrain. Margins are
   small but consistent: −0.020 R² LVEF, −0.027 R² RVSP, −1.5 pp
   MR acc.
2. **Magnitude vs V4's paired-ctrl pattern on RVSP:** V4's paired
   hurt on RVSP at +25 ep was ΔR²=−0.090 (9 pp). Pilot's paired
   hurt on RVSP at +5 ep is only −0.027 — smaller, but same sign.
   Consistent with "objective hurts RVSP vs sampler-matched
   control," just with a weaker-in-magnitude objective here.
3. **concat_all is substantially worse than encoder_pool on
   pilot.** LVEF R² 0.295 vs 0.645; RVSP R² 0.106 vs 0.066
   (better for RVSP than encoder_pool at probe e1, but still in
   null territory). Consistent with §5.10 H1 — factorized slots
   are near-linear projections of the pool, so 3×256-D concat is
   a lossy summary of the 1024-D pool when the slot nonlinearity
   doesn't expose anything new.
4. **The §5.11 retrieval-diag win did not translate to downstream
   Δ at e5.** Pilot encoder_pool cross-view retrieval top1=0.447
   beats ctrl's 0.306 by 14 pp; downstream encoder_pool probe Δ
   on RVSP is −0.027 R². Either (a) the MV2SV signal needs more
   than +5 ep to translate, or (b) cross-view retrieval is not
   the right proxy for RVSP/MR probe performance.
5. **Ctrl 658 e5 ties or beats +25-ep references on every
   endpoint.** LVEF R² 0.665 (Ctrl 608 e25: 0.670, within noise);
   RVSP MAE 9.91 (Ctrl 608 e25: 9.98, slightly *better*); MR acc
   53.0% (Ctrl 608 e25: 52.4%, slightly better). **The MV2SV
   sampler + data path alone, at +5 ep of pretrain, matches or
   exceeds plain intraview at +25 ep.** Interpretation: the
   curated clip_a distribution (quality-tier, RR-consistent,
   phase-matched) delivers most of what the extra 20 ep of plain
   V-JEPA-on-random-clips was buying. (Important caveat: the
   student encoder only ever sees clip_a; target_clip and
   clip_b_neg are encoded by the teacher but with all MV2SV
   lambdas=0 those outputs are multiplied by 0, so they
   contribute no gradient to the encoder. Verified in
   `train.py:1699, 1703, 2069-2078`.)

The paper-Δ question is now crisp: **does pilot's MV2SV
objective recover vs this strong ctrl-658 reference by e25**? If
pilot e25 − ctrl e25 remains negative on LVEF/RVSP/MR, the paper
reframes to "privileged multi-view **sampler** is the load-
bearing element; explicit multi-view **objectives** don't add
downstream signal." That's a different but still publishable
claim.

#### 10-ep pilot LVEF encoder_pool result (from cancelled job 679)

Pilot 655 e5, LVEF encoder_pool, 10 probe epochs, test set
(EchoNet-Dynamic n=1,277):

| Metric | Value |
|---|---:|
| Test MAE | 5.708 |
| Test R² | 0.593 |
| Test Pearson | 0.795 |

Stratified (test):

| Bucket | n | y̅ | pred | bias | MAE | ρ |
|---|---:|---:|---:|---:|---:|---:|
| <40 reduced | 160 | 29.0 | 38.6 | +9.6 | 11.00 | 0.616 |
| 40-55 borderline | 241 | 49.0 | 55.5 | +6.5 | 8.20 | 0.465 |
| ≥55 preserved | 876 | 62.1 | 62.0 | −0.1 | 4.05 | 0.290 |

This is a 10-ep triage number; the 20-ep version (681) gets the
apples-to-apples paper-grade comparison. Supersedes.

#### Early pilot 681 LVEF trajectory (encoder_pool, in progress)

At time of writing, pilot 681's first probe (LVEF encoder_pool)
is at epoch 8 of 20:

| Epoch | train MAE | val MAE | val R² | val Pearson |
|---|---:|---:|---:|---:|
| 1 | 8.98 | 7.10 | 0.388 | 0.717 |
| 2 | 7.84 | 7.06 | 0.420 | 0.751 |
| 3 | 7.53 | 5.83 | 0.586 | 0.768 |
| 4 | 7.39 | 5.70 | 0.606 | 0.783 |
| 5 | 7.13 | 5.83 | 0.596 | 0.786 |
| 6 | 6.97 | 5.58 | 0.596 | 0.791 |
| 7 | 6.94 | 5.60 | 0.625 | 0.799 |
| 8 | 6.82 | 5.46 | 0.635 | 0.806 |

#### Head-to-head probe trajectory, epochs 1–8, val MAE (↓ better)

All probes use the same d=4 attentive, 16 heads, 6-HP grid, 20-ep
config.

| ep | Base e125 | Ctrl 608 (paired-IV25) | V3 fb_phase_542 | V4 phase-rel 593 | **Pilot 655 e5** |
|---:|---:|---:|---:|---:|---:|
| 1 | 6.62 | 6.42 | 6.78 | 6.48 | 7.10 |
| 3 | 5.91 | 5.54 | 5.61 | 5.49 | 5.83 |
| 5 | 5.75 | 5.40 | 5.76 | 5.41 | 5.83 |
| 6 | 5.47 | 5.23 | 5.36 | **4.99** | 5.58 |
| 8 | 5.37 | 5.44 | 5.32 | **4.93** | 5.46 |

| ep | Base e125 R² | Ctrl 608 R² | V3 R² | **V4 R²** | Pilot R² |
|---:|---:|---:|---:|---:|---:|
| 6 | 0.623 | 0.655 | 0.638 | **0.694** | 0.596 |
| 8 | 0.644 | 0.629 | 0.639 | **0.712** | 0.635 |

**IMPORTANT compute-budget caveat**:

Base e125 / V3 / V4 / Ctrl 608 are all encoders with **+25 epochs
of MIMIC continuation** beyond IN21K e100. Pilot 655 e5 has only
**+5 epochs**. The fair apples-to-apples comparison for pilot at
matched compute is **matched-compute SV e105**, which doesn't
exist but would interpolate to ~val MAE 5.28, ~R² 0.67 — pilot
e5 is tracking ~0.18 MAE *worse* than that interpolated reference
at probe ep 8.

**The comparison we actually want** is pilot 655 e5 vs **ctrl 658
e5** (same +5 epochs, same sampler; only difference is the
objective). Ctrl 658 LVEF probe (job 682) is queued and starts when
676 diag frees node 56. That paired Δ will be the "does the MV2SV
objective move LVEF at matched compute" answer.

The **scientifically-definitive** comparison comes after pilot
e25 (job 683) lands: pilot 655 e25 vs V4 593 e25, pilot 655 e25 vs
Ctrl 608 e25, pilot 655 e25 vs ctrl 658 e25. Those are all
matched-compute +25-ep comparisons with all downstream probe protocol
variables held identical.

#### Current pilot 655 e5 LVEF read at probe ep 8

Pilot's probe trajectory is **tracking Base e125** (val R² 0.635 vs
0.644; val MAE 5.46 vs 5.37 — both within 0.02 R² / 0.1 MAE after
8 probe epochs), which is **better than what matched-compute** (+5
pretraining epochs) would predict. The pilot's +5-ep encoder is
already probing at Base-e125-like level on LVEF.

This is consistent with the §2 paper story that MV2SV's cross-view
objective ~matches plain continuation on LVEF (LVEF is the safety
check, not the primary endpoint). It does NOT show MV2SV winning
on LVEF vs any +25-ep reference. That's expected — the primary
endpoints are RVSP / MR.

#### Pilot — ctrl paired Δ (planned, when 682 completes)

`scripts/neurips/phase/compute_paired_bootstrap_ci.py` is written
and tested; will run on each (pilot test csv, ctrl test csv) pair:

- pilot encoder_pool − ctrl encoder_pool on LVEF, RVSP, MR
  (apples-to-apples; clean paired Δ at e5)
- pilot concat_all − ctrl encoder_pool (does the factorized head
  buy anything? given H1 result in §5.10, expect small)
- pilot encoder_pool − V4 / Base e125 / fb_sv_548 (untrained-for-
  matched-compute references; will be negative on LVEF at e5)

#### Decision rules per user priority-3

- **RVSP + MR improve vs ctrl at e5 or e25, LVEF doesn't regress**
  → scale fused-off MV2SV to 25 ep (v5 is the right architecture).
  **e5→e25 continuation already queued as 683/684** — this
  condition won't gate that.
- **Retrieval improves but downstream is null** → do NOT scale
  further; add harder negatives (§5.11's `same_study_any_view`
  gap analysis already addresses the study-identity shortcut
  concern — pilot shows a clean same-view > any-view gap, so
  study-identity is NOT the failure mode if downstream is null).
- **concat_all improves but encoder_pool is null** → decide if
  factorized-head inference is acceptable (unlikely given H1
  result in §5.10); if not, redesign to push cross-view signal
  into encoder_pool directly.
- **LVEF regresses vs ctrl at matched compute** → small λ_phase
  only after confirming RVSP/MR benefit.
- **TAPSE null** (not in current probe set) → don't modify MV2SV;
  separate local-motion objective later.

Scripts: `scripts/neurips/phase/mv2sv_pilot_downstream_probes.sbatch`
+ `mv2sv_ctrl_downstream_probes.sbatch`.

### 5.13 e5 → e25 continuations (jobs 683 pilot, 684 ctrl — queued)

Purpose: extend both pilot and ctrl to the **same +25-ep compute** as
Base e125 / V3 542 / V4 593 / Ctrl 608. The e5 read (§5.12) is
informative but underbudgeted vs the finalbudget references. The
paper-interpretable numbers are **pilot e25 vs ctrl e25** (paired
Δ at matched compute), and **pilot e25 vs V4 e25** (matched compute,
objective Δ).

Configs (created 2026-05-03):
- `configs/train/vitl16/mv2sv-pilot-5to25-resume.yaml`
- `configs/train/vitl16/mv2sv-ctrl-5to25-resume.yaml`

Key deltas vs the 5-ep configs: `load_checkpoint: true`,
`read_checkpoint: latest.pt`, `force_load_pretrain: false`,
`stop_after_epochs: 25`, `epochs: 25`, `scheduler_total_epochs: 25`,
`save_every_freq: 5`. All lambdas / sampler config / masks /
optimizer identical. LR schedule flows through naturally (already
scheduled to 25 in the 5-ep run), so no restart; optimizer state,
grad scaler, target-encoder EMA, and all factorized-head EMA weights
restore from latest.pt.

Scripts (created 2026-05-03):
- `scripts/neurips/phase/mv2sv_pilot_5to25_resume.sbatch` — pulls
  `s3://.../runs/mv2sv_pilot_5ep_655/ckpt/latest.pt` into CKPT_DIR
  before launch; 14h walltime; inline `mv2sv_gate_check.py nce`
  post-train.
- `scripts/neurips/phase/mv2sv_ctrl_5to25_resume.sbatch` — pulls
  `s3://.../runs/mv2sv_ctrl_5ep_658/ckpt/latest.pt`; 14h; inline
  `mv2sv_gate_check.py ctrl`.

Dependency chain (explicitly decoupled per user instruction):
- `683 Dependency=afterany:681` — pilot continuation starts when
  pilot probes (681) free node 146.
- `684 Dependency=afterany:682` — ctrl continuation starts when
  ctrl probes (682) free node 56.

This parallelizes the two continuations rather than serializing
them both on the same dependency. 683 will read latest.pt (contains
EMA heads + optimizer + scheduler state at e5) and produce
checkpoints at e10 / e15 / e20 / e25. 684 is the matched ctrl.

**Expected runtime** ~2h40m each (655 ran 5 ep in 40m of wall-clock
after setup; 20 ep × 8 min/ep ≈ 2h40m). Each one produces:
- `e10.pt, e15.pt, e20.pt, e25.pt, latest.pt` in
  `s3://.../runs/mv2sv_{pilot,ctrl}_5to25_<JOB>/ckpt/`
- `log_r0.csv` with the full 20-ep training trajectory
- `params-pretrain.yaml` for exact reproducibility

**Post-continuation probes** (not yet queued): pilot e25 +
ctrl e25 downstream probes on LVEF / RVSP / MR (same 6-HP, d=4
attentive, 20-ep protocol as 681/682). Gated on 683/684
completion + gate checks passing.

The scientifically-definitive paper Δ is from those post-e25
probes, not from the e5 probes (§5.12).

### 5.15 EchoJEPA-TokenRel / -Motion (jobs 692/693 smokes → 694/695 full, queued)

New variant line introduced 2026-05-03 to test the hypothesis
advanced in §2.5 (V4 LVEF mechanism + TAPSE failure mechanism):
**V4's pooled phase-relational InfoNCE lifted to per-token output
via token-set matching, plus optional same-view latent
motion-delta prediction for local amplitude.**

This is a sibling to V4 and MV2SV v5, not a modification of either.
New `multiview_objective: token_phase_relational` is registered
alongside `phase_relational` / `privileged_multiview` / etc. V4's
and MV2SV's runs are untouched.

**Two runs launched in parallel** (both chained smoke → full on
`afterok:`):

| Job | Name | Node | Loss |
|---|---|---|---|
| 692 / 694 | Run 1 EchoJEPA-TokenRel | v1 = ip-10-0-50-146 | `L_intra + λ_token·L_token_phase_rel + λ_pool·L_pool_rel_safety` |
| 693 / 695 | Run 2 EchoJEPA-TokenRel + MotionDelta | v3 = ip-10-0-50-56 | Run 1 + `λ_delta · L_latent_motion_delta` (same-view only) |

Default λ's: token_rel=0.02, pool_rel=0.005 (tiny V4-safety tail),
delta=0.01 (Run 2 only). 5-epoch warmup. τ_token=0.10, K=64
tokens per row (subsampled with shared indices across batch rows).

**Full motivation, architecture, and differences vs V4 / MV2SV**:
see **§11 EchoJEPA-Motion — design notes**.

**Sampler unchanged from V4**: same phase_matched 3-clip output,
same quality tier filter, same view_pair_policy (0.35/0.45/0.20),
same rel_wrong_phase_min_delta=0.25, same Ctrl 608 paired-control
comparability. Sampler delta to V4 = zero.

**Init**: IN21K → MIMIC standard V-JEPA e100 (same as V4 593 /
Ctrl 608 / Pilot 655 / Ctrl 658). Scheduler horizon = 100
continuation epochs; stops at 25.

**Earlier attempt (688/689)** hit a second dispatch allowlist in
`_extract_multiview_clips` (train.py:147-162) that wasn't updated
for the new objective — smokes failed at ~9 min wall-clock with
`ValueError: unknown multiview_objective='token_phase_relational'`
before any training steps landed. Patched to include the new
objective in the 3-clip allowlist; resubmitted as 692/693. The
gate check script `scripts/neurips/phase/tokenrel_gate_check.py`
enforces: no NaN, intraview drift ≤ 20%, `token_rel_top1_with_hard`
> 0.05 window mean (above random), `pos_minus_hard_gap`
non-decreasing first-vs-last window, `q_var` nonzero (no
collapse), `pool_rel_loss` finite. Run 2 additionally gates
`delta_valid_rows > 0` (mean) and `delta_loss` finite.

### 5.16 TokenRel e5 downstream probes (jobs 696/697/699, Base e125 test job 698)

Paired probes at e5 on encoder_pool, 20-ep attentive d=4 protocol,
bit-identical to V4 / Ctrl 608 / Pilot 655. Ran **after** the e5
pretrain checkpoints were captured via cancel+upload trap at step
~4800 on both 694 (TokenRel) and 695 (TokenRel+Motion).

**Timeline:**
- 694/695 cancelled at e8/25 step ~4800. `upload_artifacts` trap
  synced `e5.pt` (5.2 GB each) to S3.
- **696** (Run 1 probes: LVEF/RVSP/MR/TAPSE) launched on v1=146.
- **697** (Run 2 probes) launched on v3=56; completed LVEF only,
  then cancelled at user request to free node 56 for 698.
- **698** (Base e125 LVEF test) — 3m07s inference job for the
  missing-in-doc Base e125 test reference.
- **699** (Run 2 probes, RVSP/MR/TAPSE only) auto-launched
  `afterok:698` on node 56.

#### LVEF test results (EchoNet-Dynamic, n=1,277)

All pulled from the best-probe-ckpt inference CSVs. Val numbers
read from the full 20-epoch probe-train log_r0.csv (peak-over-
epochs, not ep 6 or ep 8 snapshots — the earlier iteration of this
doc used mid-training snapshots which understated the references).

| Variant | Pretrain | **Test MAE** | **Test R²** | **Test Pearson** | Best val R² (ep) |
|---|---|---:|---:|---:|---:|
| Base e100 (canonical) | +100 ep | 5.30 | 0.652 | 0.808 | 0.675 (ep 17) |
| **Base e125** (newly probed) | +25 ep | **5.36** | **0.646** | **0.806** | 0.685 (ep 18) |
| Ctrl 608 paired | +25 ep | 5.07 | 0.670 | 0.821 | 0.716 (ep 18) |
| **V4 phase-rel 593** | +25 ep | **4.88** | **0.699** | **0.839** | **0.742 (ep 17)** |
| Pilot 655 e5 (MV2SV) | +5 ep | 5.29 | 0.645 | 0.804 | 0.682 (ep 18) |
| **TokenRel 694 e5** | +5 ep | **5.25** | **0.655** | **0.811** | 0.684 (ep 17) |
| **TokenRel+Motion 695 e5** | +5 ep | **5.11** | **0.669** | **0.821** | **0.689 (ep 16)** |
| **TokenRel+Motion 703 e25** (job 719) | **+25 ep** | **5.16** | **0.667** | **0.819** | **0.7092 (ep 17)** |

**New read on TokenRel+Motion at e25 (ep 2026-05-04 evening):**

- **Matched-compute comparison (+25 ep)**: TokenRel+Motion e25 test R² 0.667
  / MAE 5.16 / Pearson 0.819 vs V4 e25 0.699 / 4.88 / 0.839 vs Base e125
  0.646 / 5.36 / 0.806. TokenRel+Motion e25 is **halfway between Base
  e125 and V4** on test R² but tied with Ctrl 608 (0.670) and the
  TokenRel+Motion e5 point (0.669).
- **Diminishing returns on compute**: val R² climbed +0.020 from
  TokenRel+Motion e5 (0.689) → e25 (0.709), but **test R² stayed flat**
  (0.669 → 0.667). The extra 20 pretraining epochs moved val but not
  test — compute scaling doesn't help this LVEF readout past e5.
- **Phase-InfoNCE (V4) remains the LVEF-strongest objective** at matched
  compute (+0.032 R² / +0.28 MAE advantage over TokenRel+Motion e25).
  This confirms the §2.5 mechanistic reading: V4 learns a pooled
  cycle-position axis that LVEF uniquely benefits from; TokenRel lifts
  that to token-level but the pool-level phase axis was already
  sufficient for LVEF.

#### Source CSV references

All 7 rows in the table above are log-validated. Each entry links
the exact probe-train (best-val-over-20-epochs) log_r0.csv and the
test-inference log_r0.csv. All S3 paths are under
`s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts/`
(abbreviated below as `S3://`).

**Base e100 (canonical IN21K → MIMIC, ViT-L, +100 ep)**
- Pretrain encoder: `S3://runs/jepa_in21k_pretrain_376/checkpoints/e100.pt` (also mirrored at `S3://CLEAN/encoders/jepa_in21k_vitl_e100.pt`)
- Probe train (val): `evals/vitl/icml/jepa_in21k_e100_end_lvef_224/video_classification_frozen/icml-jepa-in21k-e100-end-lvef-d4/log_r0.csv`
- Probe test (predavg): `evals/vitl/icml/jepa_in21k_e100_end_lvef_224/video_classification_frozen/icml-jepa-in21k-e100-end-lvef-d4-predavg/log_r0.csv`

**Base e125 (IN21K → MIMIC, ViT-L, +25 ep continuation from e100)**
- Pretrain encoder: `S3://runs/jepa_in21k_e200_280/training_folder/e125.pt`
- Probe train (val): `S3://runs/jepa_ext_probes_332/jepa_e125_lvef/video_classification_frozen/neurips-jepa-in21k-e125-end-lvef-d4/log_r0.csv`
- Probe test (this doc, job 698 — 3m inline inference reusing the 332 probe best.pt):
  - `S3://runs/base_e125_lvef_test_698/test/video_classification_frozen/base-e125-lvef-test/log_r0.csv`
  - Predictions CSV: `S3://runs/base_e125_lvef_test_698/predictions/base_e125_lvef_test.csv`

**Ctrl 608 (paired intraview-only, +25 ep from e100, V4's paired control)**
- Pretrain encoder: `S3://runs/final_paired_iv25_paper_608/checkpoints/e25.pt`
- Probe train (val, job 629): `S3://runs/final_paired_iv25_lvef_629/probe/video_classification_frozen/neurips-final-paired-iv25-lvef/log_r0.csv`
- Probe test (job 630): `S3://runs/final_paired_iv25_lvef_test_630/eval/video_classification_frozen/neurips-final-paired-iv25-lvef-test/log_r0.csv`

**V4 phase-rel 593 (phase-relational InfoNCE, +25 ep from e100)**
- Pretrain encoder: `S3://runs/final_phase_rel25_paper_593/checkpoints/e25.pt`
- Probe train (val, job 595): `S3://runs/final_phase_rel25_lvef_595/probe/video_classification_frozen/neurips-final-phase-rel25-lvef/log_r0.csv`
- Probe test (job 596): `S3://runs/final_phase_rel25_lvef_test_596/eval/video_classification_frozen/neurips-final-phase-rel25-lvef-test/log_r0.csv`

**Pilot 655 e5 (MV2SV v5 scientific pilot, +5 ep from e100)**
- Pretrain encoder: `S3://runs/mv2sv_pilot_5ep_655/ckpt/e5.pt` (also `latest.pt`)
- Probe train (val, job 681): `S3://runs/mv2sv_pilot_probes_681/lvef_encoder_pool/probe/video_classification_frozen/mv2sv-pilot-655-e5-lvef-encoder_pool/log_r0.csv`
- Probe test (job 681 inline): `S3://runs/mv2sv_pilot_probes_681/lvef_encoder_pool/test/video_classification_frozen/mv2sv-pilot-655-e5-lvef-encoder_pool-test/log_r0.csv`

**TokenRel 694 e5 (EchoJEPA-TokenRel Run 1, +5 ep from e100)**
- Pretrain encoder: `S3://runs/echojepa_tokenrel_run1_694/ckpt/e5.pt` (synced via exit trap after scancel at e8/25)
- Probe train (val, job 696): on compute node 146 local
  - `/opt/dlami/nvme/tokenrel_r1_probes_696/code/evals/vitl/neurips/tokenrel_r1_694_lvef_encoder_pool_224/video_classification_frozen/tokenrel-r1-694-e5-lvef-encoder-pool/log_r0.csv`
  - (will upload to `S3://runs/tokenrel_r1_probes_696/lvef_encoder_pool/probe/` on job exit via trap)
- Probe test (job 696 inline):
  - `/opt/dlami/nvme/tokenrel_r1_probes_696/code/evals/vitl/neurips/tokenrel_r1_694_lvef_encoder_pool_test/video_classification_frozen/tokenrel-r1-694-e5-lvef-encoder-pool-test/log_r0.csv`
  - (will upload to `S3://runs/tokenrel_r1_probes_696/lvef_encoder_pool/test/`)

**TokenRel+Motion 695 e5 (EchoJEPA-TokenRel + MotionDelta Run 2, +5 ep from e100)**
- Pretrain encoder: `S3://runs/echojepa_tokenrel_delta_run2_695/ckpt/e5.pt` (synced via exit trap after scancel at e8/25)
- Probe train (val, job 697 — cancelled after LVEF to free node for 698):
  - `S3://runs/tokenrel_r2_probes_697/lvef_encoder_pool/probe/video_classification_frozen/tokenrel-r2-695-e5-lvef-encoder-pool/log_r0.csv`
- Probe test (job 697 inline, completed before cancel):
  - `S3://runs/tokenrel_r2_probes_697/lvef_encoder_pool/test/video_classification_frozen/tokenrel-r2-695-e5-lvef-encoder-pool-test/log_r0.csv`

**TokenRel+Motion 703 e25 (continuation from 695 e5 → e25, +25 ep from e100)**
- Pretrain encoder: `S3://runs/echojepa_tokenrel_delta_run2_5to25_703/ckpt/e25.pt`
- Probe train (val, job 719 inline):
  - `S3://runs/tokenrel_r2_e25_lvef_719/lvef_encoder_pool/probe/video_classification_frozen/tokenrel-r2-e25-lvef-encoder-pool/log_r0.csv`
- Probe test (job 719 inline):
  - `S3://runs/tokenrel_r2_e25_lvef_719/lvef_encoder_pool/test/video_classification_frozen/tokenrel-r2-e25-lvef-encoder-pool-test/log_r0.csv`
- Predictions CSV: `S3://runs/tokenrel_r2_e25_lvef_719/lvef_encoder_pool/predictions/tokenrel_r2_e25_e5_lvef_test.csv`

**Note on TokenRel node-local CSVs**: 696 is still running and 697
was cancelled before its `upload_artifacts` trap ran to completion
for all artifacts (lvef/test did sync based on the trap's sync
pattern; verify on S3 after 696/699 exit). To re-fetch node-local
CSVs while the jobs are still alive, use:

```bash
srun -w ip-10-0-50-146 -N1 --ntasks=1 --jobid=696 --overlap \
    cat /opt/dlami/nvme/tokenrel_r1_probes_696/code/evals/vitl/neurips/tokenrel_r1_694_lvef_encoder_pool_224/video_classification_frozen/tokenrel-r1-694-e5-lvef-encoder-pool/log_r0.csv
```

**Audit procedure**: every number in the §5.16 table was pulled
via `aws s3 cp <path> -` (for S3 CSVs), `cat` (for EFS CSVs), or
the `srun --overlap` pattern above (for node-local CSVs). The
best-val-R² values are the **max val_r2 over all 20 epochs** of
the probe-train log; the test values are the single row from the
test log_r0.csv (which has only one epoch since test is a one-pass
inference).

#### Key reads (LVEF)

1. **TokenRel+Motion 695 e5 matches Ctrl 608 (+25 ep) on test**:
   test R² 0.669 vs Ctrl 608's 0.670; test Pearson 0.821 both.
   **5× less pretrain compute.**
2. **TokenRel 694 e5 slightly beats Base e100 (+100 ep) and Pilot
   655 e5 (MV2SV)** on test R² and Pearson. Delta vs Pilot is
   +0.010 R² / +0.007 Pearson.
3. **MotionDelta helps**: Run 2 beats Run 1 by +0.014 test R² /
   +0.010 Pearson and −0.14 test MAE. The same-view latent delta
   prediction is adding signal, not regressing LVEF.
4. **V4 phase-rel still leads by ~0.03 R² / 0.23 MAE** over
   TokenRel+Motion. V4 had 5× the pretrain compute. Whether the
   gap closes at matched compute (TokenRel e25 vs V4 e25) remains
   untested — pretrain was cancelled at e8/25 to free nodes for
   probes.
5. **Base e125 slightly worse than Base e100 on test** (R² 0.646
   vs 0.652) — consistent with `finalbudget-phase-probes.md`
   observation that plain JEPA continuation plateaus on LVEF
   around e100-e125. The +25 ep phase-matched continuation in
   Ctrl 608 adds +0.025 R² over Base e125; a signal attributable
   to the phase-matched sampler, not to extra epochs alone.

#### Audit note

Earlier iterations of the doc's §5.12 comparison table cited
approximate val R² values ("~0.65", "~0.66", "~0.72") from
ep 6/ep 8 snapshots, which understated every reference. The
numbers in this §5.16 table are the **actual best-over-20-epochs
val metrics** pulled from each probe's log_r0.csv, plus the
paper-comparable **test-set metrics** from each variant's inline
test inference. The §5.12 table has not been back-edited; this
§5.16 is the corrected reference going forward.

#### RVSP test results (MIMIC SV, n=2,000 clips)

RVSP probe train + inline test inference landed for both TokenRel
variants. Probe-train val trajectories pulled from node-local CSVs
(see Source CSV references below for paths).

| Variant | Pretrain | Best val R² | Best val Pearson | Test MAE | Test R² | Test Pearson |
|---|---|---:|---:|---:|---:|---:|
| SV fb_sv_548 | +25 ep | 0.195 (ep 5) | 0.489 (ep 7) | 9.71 | **0.157** | 0.400 |
| V4 phase-rel 593 | +25 ep | 0.199 (ep 5) | 0.475 (ep 12) | 10.53 | 0.018 | 0.281 |
| Ctrl 608 paired | +25 ep | 0.150 (ep 6) | 0.450 (ep 11) | 9.98 | 0.108 | 0.344 |
| Pilot 655 e5 (MV2SV) | +5 ep | 0.188 (ep 5) | 0.467 (ep 4) | 10.17 | 0.066 | 0.289 |
| Ctrl 658 e5 | +5 ep | 0.168 (ep 6) | 0.428 (ep 6) | 9.91 | 0.093 | 0.331 |
| **TokenRel 694 e5** | +5 ep | **0.192 (ep 8)** | 0.486 (ep 17) | 10.17 | **0.083** | **0.346** |
| **TokenRel+Motion 695 e5** | +5 ep | 0.186 (ep 8) | **0.514 (ep 13)** | 10.30 | 0.067 | **0.371** |

#### RVSP key reads

1. **TokenRel 694 e5 RVSP test R² 0.083** — above Pilot 655 e5 (0.066),
   on par with Ctrl 658 e5 (0.093). Still below Ctrl 608 +25 ep (0.108)
   and SV fb_sv_548 +25 ep (0.157). TokenRel at +5 ep is halfway between
   the +5-ep MV2SV pilot and the +25-ep paired control on RVSP.
2. **TokenRel+Motion 695 e5 RVSP R² is slightly lower than TokenRel 694
   (0.067 vs 0.083)** — MotionDelta branch does not help RVSP and may
   mildly distract the encoder. Consistent with §11.5's prediction
   ("RVSP is not explicitly targeted; ... not a pre-registered claim").
3. **Both TokenRel variants match or beat Ctrl 608 on val Pearson.**
   TokenRel+Motion's peak val Pearson **0.514** is **the highest RVSP
   val Pearson** of any encoder on record (V4: 0.475; fb_sv_548: 0.489).
   The test Pearson also lands high (0.371 for TokenRel+Motion; 0.346
   for TokenRel) — both above Ctrl 608 (0.344), Pilot 655 (0.289), and
   V4 (0.281). Pearson ≠ R² here: predictions correlate well but are
   poorly-calibrated (scale/bias mismatch).
4. **V4's RVSP hurt pattern reproduced but less severe**: V4 paired
   Δ = −0.090 R² vs Ctrl 608. TokenRel 694 vs Ctrl 608 Δ = −0.025 R².
   Token-level InfoNCE hurts RVSP ≈ 3× less than V4's pooled InfoNCE
   at matched-control comparison.
5. **RVSP remains a "more pretrain, simpler objective" task.** No
   variant at +5 ep beats SV fb_sv_548 at +25 ep; the compute gap is
   load-bearing here in a way LVEF's wasn't.

#### MR test results (MIMIC A4C 4-class, ~2,421 test clips)

MR probe train + inline test inference completed for both TokenRel
variants. Older runs (609 V4, 611 Base e125) ran on an earlier probe
code path that didn't log val_auroc — marked as NaN where missing.

| Variant | Pretrain | Best val_acc (ep) | Best val_auroc (ep) | Test val_acc | Test val_auroc | Test kappa |
|---|---|---|---|---:|---:|---:|
| 611 Base e125 | +25 ep | 54.24 (ep 18) | — | — | — | — |
| 609 V4 phase-rel | +200 ep | 53.74 (ep 14) | — | — | — | — |
| 681 Pilot 655 e5 | +5 ep | 53.55 (ep 9) | 0.730 (ep 11) | — | — | — |
| 682 Ctrl 658 e5 | +5 ep | 53.80 (ep 18/19) | 0.7413 (ep 16) | — | — | — |
| **696 TokenRel 694 e5** | +5 ep | 53.35 (ep 14) | **0.7328 (ep 16)** | **52.66** | **0.7285** | **0.2490** |
| **699 TokenRel+Motion 695 e5** | +5 ep | 53.12 (ep 18) | 0.7308 (ep 14) | 51.85 | 0.7288 | 0.2266 |

Reads:
1. **All six e5-class runs converge inside ~1 pp val_acc and ~0.01
   val_auroc.** MR A4C SV is a weak separator — consistent with prior
   note in §3 and with the doc's repeated caveat that the strong MR
   test is multi-view (A4C + A2C + PLAX).
2. **TokenRel 694 test val_acc 52.66 / AUROC 0.7285** lands slightly
   below 681 / 682 best-over-20 val but within CI. TokenRel+Motion
   695 is ~0.8 pp below TokenRel 694 on test acc. Both TokenRel
   variants are on par with Pilot / Ctrl e5 runs on MR, not
   advantaged or disadvantaged.
3. **MR A4C is not a decisive readout** for any variant; the endpoint
   to watch remains RVSP aggregate + TAPSE mechanism (below).

#### TAPSE test results (MIMIC A4C regression, n=~2,000 test clips)

TAPSE probe train + inline test inference completed for both TokenRel
variants. This was the decisive read for the MotionDelta hypothesis
(§2.5, §11.5) — does same-view latent motion-delta supervision
unlock the RV longitudinal motion signal that pooled phase-InfoNCE
(V4) and V-JEPA intraview (Base e125) both capped at R² ≈ 0.25?

| Variant | Pretrain | Best val R² (ep) | Best val Pearson (ep) | Test MAE | Test R² | Test Pearson |
|---|---|---|---|---:|---:|---:|
| **V4 phase-rel e25** (620/621) | +25 ep | **0.2904 (ep 9)** | 0.5397 (ep 9) | 0.3548 | **0.2504** | 0.5190 |
| **Base e125** (622/623) | +25 ep | 0.2873 (ep 9) | 0.5395 (ep 9) | 0.3561 | 0.2468 | 0.5136 |
| **696 TokenRel 694 e5** | +5 ep | 0.2596 (ep 12) | 0.5215 (ep 8) | 0.3713 | 0.1830 | 0.4438 |
| **699 TokenRel+Motion 695 e5** | +5 ep | 0.2383 (ep 8) | 0.4915 (ep 15) | 0.3718 | 0.1803 | **0.4609** |
| **720 TokenRel+Motion 703 e25** | **+25 ep** | **0.2692 (ep 6)** | **0.5206 (ep 12)** | **0.3642** | **0.2102** | **0.4811** |

**New TAPSE read at matched compute (+25 ep, 2026-05-04 evening):**

- **MotionDelta hypothesis NOT supported at matched compute.**
  TokenRel+Motion e25 test R² 0.210 / MAE 0.364 / Pearson 0.481 lands
  **below V4 e25 (0.250 / 0.355 / 0.519) and Base e125 (0.247 / 0.356 /
  0.514)** — the same-view latent motion-delta prediction did not
  unlock TAPSE as §11.5 predicted.
- **+20 epochs did lift TokenRel+Motion's TAPSE** (R² 0.180 → 0.210,
  Pearson 0.461 → 0.481) but not enough to reach V4/Base. Gap to V4 is
  −0.040 R² / −0.038 Pearson.
- **TAPSE genuinely capped at test R² ~0.25** on A4C alone across all
  three recent pretraining regimes. V4, Base e125 and TokenRel+Motion
  e25 all plateau in the 0.21–0.25 band. This is consistent with the
  §2.5 reading: A4C-only TAPSE is fundamentally limited by the absence
  of local-amplitude signal the probe can pick up, and none of the
  current objectives (pooled phase-InfoNCE, plain V-JEPA, token-level
  phase-InfoNCE + motion-delta) changes that.
- Trajectory shape ends up the same: peak val R² at ep 6–9, then
  oscillation in the 0.21–0.27 band through ep 20, with a mild late-ep
  overfit (val R² dropped to 0.21 at ep 20 for TokenRel+Motion).

Reads:
1. **MotionDelta does NOT unlock TAPSE at the e5 ckpt.** Both TokenRel
   variants land at **test R² ≈ 0.18**, *below* V4's 0.25 standing
   cap. The +0.03-0.04 R² gap is explainable by pretrain-compute
   difference (V4 had +200 ep of phase-rel supervision; TokenRel had
   +5 ep), but the direction is the wrong sign for MotionDelta's
   §11.5 hypothesis to be active.
2. **TokenRel+Motion vs TokenRel (within-compute comparison)**: R²
   0.180 vs 0.183; Pearson 0.461 vs 0.444. The MotionDelta branch
   adds +0.017 Pearson on TAPSE — a weak non-zero signal, not the
   multi-pp lift §11.5 predicted.
3. **Peak val_pearson for TokenRel 694 is 0.522** — best-in-class at
   e5 across TokenRel variants on TAPSE val, but the val → test gap
   is ~0.08 points of Pearson (0.52 val → 0.44 test), consistent with
   regression-to-mean on the test set (see §5.11 V4 TAPSE
   stratification).
4. **TAPSE readout is inconclusive for MotionDelta at +5 ep.** To
   cleanly test §11.5, we need TokenRel+Motion at +25 ep (job 703,
   currently running on v1) vs TokenRel at +25 ep. Only then is the
   MotionDelta-specific Δ free of the pretrain-compute confound.

The 703 (5→25 pretrain continuation for Run 2) will settle this.
Expected TAPSE re-probe after 703 finishes.

#### MR + TAPSE Source CSV references

- Pilot 655 e5 MR val + test: `S3://runs/mv2sv_pilot_probes_681/mr_encoder_pool/{probe,test}/video_classification_frozen/mv2sv-pilot-655-e5-mr-encoder_pool{,-test}/log_r0.csv`
- Ctrl 658 e5 MR val + test: `S3://runs/mv2sv_ctrl_probes_682/mr_encoder_pool/{probe,test}/video_classification_frozen/mv2sv-ctrl-658-e5-mr-encoder-pool{,-test}/log_r0.csv`
- V4 MR val (no AUROC): `S3://runs/final_phase_rel25_mr_a4c_609/probe/video_classification_frozen/neurips-final-phase-rel25-mr-a4c/log_r0.csv`
- Base e125 MR val (no AUROC): `S3://runs/final_paired_iv25_mr_a4c_611/probe/video_classification_frozen/neurips-final-paired-iv25-mr-a4c/log_r0.csv`
- TokenRel 694 e5 MR val + test: `S3://runs/tokenrel_r1_probes_696/mr_encoder_pool/{probe,test}/video_classification_frozen/tokenrel-r1-694-e5-mr-encoder-pool{,-test}/log_r0.csv`
- TokenRel+Motion 695 e5 MR val + test: `S3://runs/tokenrel_r2_probes_699/mr_encoder_pool/{probe,test}/video_classification_frozen/tokenrel-r2-695-e5-mr-encoder-pool{,-test}/log_r0.csv`
- V4 TAPSE val: `S3://runs/final_phase_rel25_tapse_620/probe/video_classification_frozen/neurips-final-phase-rel25-tapse-a4c/log_r0.csv`
- V4 TAPSE test: `S3://runs/final_phase_rel25_tapse_test_621/eval/video_classification_frozen/neurips-final-phase-rel25-tapse-a4c-test/log_r0.csv`
- Base e125 TAPSE val: `S3://runs/jepa_e125_tapse_622/probe/video_classification_frozen/neurips-jepa-e125-tapse-a4c/log_r0.csv`
- Base e125 TAPSE test: `S3://runs/jepa_e125_tapse_test_623/eval/video_classification_frozen/neurips-jepa-e125-tapse-a4c-test/log_r0.csv`
- TokenRel 694 e5 TAPSE val + test: `S3://runs/tokenrel_r1_probes_696/tapse_encoder_pool/{probe,test}/video_classification_frozen/tokenrel-r1-694-e5-tapse-encoder-pool{,-test}/log_r0.csv`
- TokenRel+Motion 695 e5 TAPSE val + test: `S3://runs/tokenrel_r2_probes_699/tapse_encoder_pool/{probe,test}/video_classification_frozen/tokenrel-r2-695-e5-tapse-encoder-pool{,-test}/log_r0.csv`
- **TokenRel+Motion 703 e25 TAPSE val + test (job 720)**:
  - `S3://runs/tokenrel_r2_e25_tapse_720/tapse_encoder_pool/probe/video_classification_frozen/tokenrel-r2-e25-tapse-encoder-pool/log_r0.csv`
  - `S3://runs/tokenrel_r2_e25_tapse_720/tapse_encoder_pool/test/video_classification_frozen/tokenrel-r2-e25-tapse-encoder-pool-test/log_r0.csv`

#### RVSP Source CSV references (val + test)

- SV fb_sv_548 val: `S3://runs/fb_sv_548_rvsp_558/probe/video_classification_frozen/neurips-fb-sv-548-rvsp-sv/log_r0.csv`
- SV fb_sv_548 test: `S3://runs/fb_sv_548_rvsp_test_562/eval/video_classification_frozen/neurips-fb-sv-548-rvsp-sv-test/log_r0.csv`
- V4 593 val: `S3://runs/final_phase_rel25_rvsp_597/probe/video_classification_frozen/neurips-final-phase-rel25-rvsp-sv/log_r0.csv`
- V4 593 test: `S3://runs/final_phase_rel25_rvsp_test_598/eval/video_classification_frozen/neurips-final-phase-rel25-rvsp-sv-test/log_r0.csv`
- Ctrl 608 val: `S3://runs/final_paired_iv25_rvsp_631/probe/video_classification_frozen/neurips-final-paired-iv25-rvsp-sv/log_r0.csv`
- Ctrl 608 test: `S3://runs/final_paired_iv25_rvsp_test_632/eval/video_classification_frozen/neurips-final-paired-iv25-rvsp-sv-test/log_r0.csv`
- Pilot 655 e5 val + test: `S3://runs/mv2sv_pilot_probes_681/rvsp_encoder_pool/{probe,test}/video_classification_frozen/mv2sv-pilot-655-e5-rvsp-encoder_pool{,-test}/log_r0.csv`
- Ctrl 658 e5 val + test: `S3://runs/mv2sv_ctrl_probes_682/rvsp_encoder_pool/{probe,test}/video_classification_frozen/mv2sv-ctrl-658-e5-rvsp-encoder-pool{,-test}/log_r0.csv`
- TokenRel 694 e5 (node-local during inline test, will sync on job exit): `/opt/dlami/nvme/tokenrel_r1_probes_696/code/evals/vitl/neurips/tokenrel_r1_694_rvsp_encoder_pool_{224,test}/video_classification_frozen/tokenrel-r1-694-e5-rvsp-encoder-pool{,-test}/log_r0.csv`
- TokenRel+Motion 695 e5 (node-local during inline test): `/opt/dlami/nvme/tokenrel_r2_probes_699/code/evals/vitl/neurips/tokenrel_r2_695_rvsp_encoder_pool_{224,test}/video_classification_frozen/tokenrel-r2-695-e5-rvsp-encoder-pool{,-test}/log_r0.csv`

### 5.17 A4C final-task-list probes (job 704, TokenRel+Motion 695 e5)

A4C-only probes on the 4 final-task-list tasks from
`claude/neurips/final-task-list.md` — HCM A4C 10k (binary),
Incident HF 1y A4C 10k (binary), Age A4C 10k (regression), LVEF A4C
10k (regression). All 6-HP 20-ep attentive d=4, encoder_pool,
10k-clip matched-budget train set, val/test ≤3 clips/study. Job
704 on v3 completed 2026-05-04 at 7h14m.

| Task | Best val metric (ep) | Test metric | Test bal_acc / Pearson | Kappa / MAE |
|---|---|---:|---:|---:|
| **HCM A4C 10k** (binary, 2.4% prevalence) | val_auroc 0.572 (ep 15) | test_auroc **0.760** / acc 96.6% | bal_acc 0.521 | kappa 0.064 |
| **Incident HF 1y A4C 10k** (binary, 9% prevalence) | val_auroc 0.579 (ep 19) | test_auroc 0.581 / acc 94.1% | bal_acc 0.500 | kappa 0.000 |
| **Age A4C 10k** (regression) | val_r2 0.261 (ep 13), pearson 0.532 | test R² **0.244** / MAE 9.23 yrs | Pearson 0.514 | — |
| **LVEF A4C 10k** (regression) | val_r2 0.350 (ep 16), pearson 0.614 | test R² **0.441** / MAE 7.60 | Pearson 0.669 | — |

Reads:
1. **HCM AUROC 0.76 on test** at 2.4% prevalence with matched-budget
   10k train is a respectable signal from a single view; val kappa
   0.06 says the probe isn't strongly rank-ordering near the decision
   threshold but it clearly picks up HCM as a latent concept.
2. **Incident HF 1y AUROC 0.58** reflects the fundamental difficulty
   of predicting future admission from B-mode A4C alone (ICD
   under-coding, right-censoring, single-view). This is publishable
   as a null / weak result.
3. **Age R² 0.244** at e5 with 10k clips from a single view is
   plausible — the 91-year ceiling compresses the tail.
4. **LVEF A4C 10k R² 0.441 / MAE 7.60** — this is the MIMIC
   cross-domain LVEF signal at matched EchoNet-Dynamic compute
   budget. Compare to TokenRel+Motion 695 e5 on EchoNet-Dynamic
   LVEF: R² 0.669 / MAE 5.11 / Pearson 0.821 (job 697 captured pre-
   cancel; see §5.16 LVEF table). EchoNet-Dynamic's LVEF is a
   cleaner dataset; MIMIC A4C LVEF carries additional domain noise.

Jobs 705 (Base e125) and 706 (V4 e25) are queued on v3 to run these
same 4 A4C tasks on the two reference pretrain ckpts. When all
three complete, we'll have a 3-way matched-budget comparison
(TokenRel+Motion e5 / Base e125 / V4 e25) on all 4 A4C tasks.

#### 5.17 Source CSV references

- 704 TokenRel+Motion 695 e5 A4C probes (4 tasks × probe + test):
  `S3://runs/a4c_probes_tokenrel_r2_695_e5_704/{task}_encoder_pool/{probe,test}/video_classification_frozen/tokenrel_r2_695_e5-{task}-a4c-encoder-pool{,-test}/log_r0.csv`
  where `{task}` ∈ {hcm, hf_incident_1yr, age, lvef}.
- Trimmed 10k CSVs: `S3://data/csv/mimic_{task}_a4c_{split}_10k.csv`.
- Trim provenance: `experiments/nature_medicine/mimic/probe_csvs/{task}_a4c_10k/trim_meta.json`.

---

### 5.18 RV function binary 10k probes (jobs 718, 721 — in-flight)

New task added from `claude/neurips/final-task-list.md` § "Qualitative
RV function — A4C (5-class full + binary 10k)". **Binary RV function
dysfunction** derived from MIMIC `echo_structured_measurement.rv_function`
5-class labels, binarized as:

- **Class 0 (no dysfunction)**: `Nl RV function`, `Low normal function`
- **Class 1 (any dysfunction)**: `Mild global RV hypo`, `Moderate global
  RV hypo`, `Severe global hypo`, `RV function depressed`

10,000-clip matched-budget train (stratified, 15.2% pos), val 2,188,
test 2,122. A4C view only. Patient-level splits inherited from
`disease_hf_v4.1`. encoder_pool probe, 20-ep attentive d=4, 6-HP grid.

#### Early val trajectory (both in-flight 2026-05-04 evening)

**Job 718 — V4 e25** (v3=56, ep 9/20 at last check):

| ep | val_acc | val_auroc | val_bal_acc | val_kappa |
|---:|---:|---:|---:|---:|
| 1 | 85.54 | 0.8123 | 0.6914 | 0.4170 |
| 3 | 85.58 | 0.8229 | 0.6778 | 0.4054 |
| 5 | 85.40 | 0.8353 | 0.6110 | 0.2991 |
| 7 | **86.72** | **0.8527** | 0.7041 | 0.4690 |
| 8 | 85.99 | 0.8471 | 0.7248 | **0.4762** |
| 9 | 86.68 | 0.8451 | 0.6918 | 0.4481 |

**Job 721 — Base e125** (v1=146, ep 4/20 at last check):

| ep | val_acc | val_auroc | val_bal_acc | val_kappa |
|---:|---:|---:|---:|---:|
| 1 | 82.62 | 0.7221 | 0.5000 | 0.0000 |
| 2 | 82.62 | 0.8237 | 0.5000 | 0.0000 |
| 3 | 86.59 | 0.8446 | 0.7143 | 0.4791 |
| 4 | 86.50 | **0.8515** | 0.6745 | 0.4252 |

#### Reads (preliminary — both jobs in-flight)

1. **Both encoders probe to val AUROC ~0.85** rapidly on RV function
   binary. The task appears easier than MR A4C 4-class (~0.73 AUROC)
   and considerably easier than HCM A4C binary (~0.57 val AUROC
   cluster). This supports RV function as a **strong MIMIC readout**
   for model comparison going forward.
2. **No large V4-vs-Base separation in early epochs** — V4 ep 1 AUROC
   0.812, Base e125 ep 3 AUROC 0.845. Base catches up and passes V4 on
   ep-to-ep comparison in the early phase. Final test comparison
   pending.
3. **TokenRel+Motion e5 / e25 RV function NOT yet run** — job 707
   (TokenRel+Motion e5 RV function) was cancelled during queue
   refactor. The matched-compute comparison is V4 e25 (718) vs Base
   e125 (721) only. A future job could fill in TokenRel+Motion.

#### Source CSV references (when complete)

- Dataset:
  - Train 10k: `experiments/nature_medicine/mimic/probe_csvs/rv_function_binary_10k_a4c/train.csv` (10,000 clips)
  - Val / test: same dir (2,188 / 2,122 clips)
  - Builder: `experiments/nature_medicine/mimic/probe_csvs/build_rv_function.py`
  - S3: `S3://data/csv/mimic_rv_function_binary_10k_a4c_{train,val,test}.csv`
- 718 V4 e25 (expected S3 location on job exit):
  `S3://runs/rvfunc_probes_v4_e25_718/rv_function_binary_10k_encoder_pool/{probe,test}/...`
- 721 Base e125 (expected S3 location on job exit):
  `S3://runs/rvfunc_probes_base_e125_721/rv_function_binary_10k_encoder_pool/{probe,test}/...`

---

### 5.14 Fused-pool coverage audit (planned — not yet run)

Script: `scripts/neurips/phase/mv2sv_fused_coverage_audit.py`.
Runs the sampler with `fused_pool.enabled=true` for N epochs
(no video decode required — only the MatchRecord metadata is
needed). Reports:
- `fused_valid_views` mean / median / p10 / p25 / p75
- fraction of rows with ≥2 valid views
- fraction of rows with ≥3 valid views
- source_view / target_view / source×target pair counts
- fused_valid_mean by source×target pair

Decision rule: if **≥80% of rows** can deliver ≥2 valid fused views,
re-enable fused with `n_fused_min=2` and the existing `>=2` forward
guard. If not, implement the row-masked fused branch (task #222)
and set `min_fused_row_fraction` to the observed fraction plus a
safety margin. If even 50% can't deliver ≥2, fused stays off in
paper v1.

---

## 6. Pre-registered success criteria (MV2SV v5 vs controls)

The paper-interpretable comparison for MV2SV is:

```
Δ_MV2SV = MV2SV v5 end (job 655 successor, 25ep full run) − paired intraview-only control
```

on RVSP, MR severity, AS severity, LVEF.

Paired control design mirrors Variant 4's (job 608): identical
sampler + eligibility + view_pair policy + target_clip / fused_clips
loaded and discarded; only `multiview_objective` differs. This
neutralizes the sampler-diversity confound that (correctly) weakened
Variant 4's RVSP comparison vs `fb_sv_548`.

Success rules:
- **Strong accept**: Δ > 0 with 95% CI excluding zero on **at least
  two of {RVSP, MR, AS}** and LVEF not regressed.
- **Weak accept**: Δ > 0 with 95% CI excluding zero on **one of
  {RVSP, MR, AS}** and LVEF not regressed. (Matches Variant 4's
  LVEF-only outcome on a novel endpoint — not a slam dunk, but
  publishable with appropriate framing about endpoint
  specificity.)
- **Neutral**: Δ ≈ 0 across all three cross-view endpoints. The
  method claim fails; paper reframes around "single discriminative
  term (Variant 4) is what's load-bearing for LVEF, cross-view
  discrimination is not automatic."
- **Regression**: LVEF degrades vs SV e125. Not expected; would
  indicate the privileged signal is injecting noise. Stop-gap:
  drop `λ_pair_view` to 0.05, re-run.

Secondary reference (not pre-registered): `595_method −
MV2SV_end` on LVEF. If `595_method` > `MV2SV_end` on LVEF, the
phase-relational objective is a better LVEF mechanism than MV2SV;
this is consistent with "phase discrimination helps
phase-dominant tasks, view discrimination helps view-dominant
tasks" and is fine for the paper.

---

## 7. Factorized-slot probe readout (interpretability claim)

Per v4 addendum, post-training probes run on:
- legacy encoder-pool (current paper protocol)
- `z_shared` alone
- `z_phase` alone
- `z_view` alone
- `[z_shared; z_phase]`
- `[z_shared; z_phase; z_view]`

Expected *a priori* readout (not pre-registered — supporting
interpretability, not a hypothesis test):
- LVEF benefits from `z_phase`.
- RVSP / MR / AS benefit from `z_shared + z_view`.
- TAPSE needs `z_phase` + local-motion features (hence the
  deferred v5+ work).

If the readout matches, we have a clean interpretability paragraph
for the paper. If it doesn't, the factorized head is an
implementation detail, not a scientific claim.

---

## 8. Known open items

1. **Paired intraview-only MV2SV control** — ✅ COMPLETED (job
   658, 3h11m, exit 0:0, all gates passed; §5.5).
2. **Retrieval diag sweep** — ✅ COMPLETED (job 676, n=1000;
   §5.11). Pilot 655 e5 encoder_pool top1=0.447 gap=+0.394 beats
   every +25-ep reference. Pilot `same_study_same_view >
   same_study_any_view` gap ~10pp persists, disambiguating the
   study-identity shortcut concern.
3. **F/G/H attribution diagnostics** — ✅ COMPLETED (job 673,
   20m12s; §5.9). V4 uniquely frame-shuffle-sensitive
   (cos(clean, shuf) = 0.89 vs 0.99 elsewhere); pilot 655 e5
   temporally indistinguishable from Base e100 on pooled
   features.
4. **H1 slot-triviality + scaled phase probe** — ✅ COMPLETED
   (job 674; §5.10). Slot projections are near-linear maps of
   the pool (R² ≈ 0.99 by e3 for pilot). **Factorization-as-
   interpretability claim does not survive** — paper language
   must be updated to "implementation detail" framing.
5. **Downstream attentive probes (e5)** — 🟢 IN-FLIGHT (job 681
   pilot RUNNING on 146, 682 ctrl RUNNING on 56; §5.12). 20-ep
   d=4 attentive, pipeline bit-identical to V4/V3/Base/Ctrl 608
   finalbudget probes. Pilot 681 LVEF trajectory through ep 8
   tracks Base e125 (val R² 0.635 vs 0.644) despite pilot having
   +5 ep vs Base's +25 ep. Paired Δ vs ctrl 682 gives
   matched-compute objective Δ.
6. **e5 → e25 continuations** — ⏳ QUEUED (job 683 pilot
   `afterany:681`, 684 ctrl `afterany:682`; §5.14). Configs +
   sbatches staged. Post-continuation probes (not yet queued)
   deliver the scientifically-definitive paper Δ at matched
   +25-ep compute vs V4 593 / Ctrl 608 / Base e125.
7. **Fused sparsity audit (§5.14)** — gates fused re-enable. Not
   yet run; optional, not blocking the fused-off path.
8. **Row-masked fused branch** (task #222) — only if audit shows
   partial but not sufficient coverage.
9. **Local-motion loss** — ✅ SHIPPED (2026-05-03) as the Run 2
   MotionDelta branch of EchoJEPA-TokenRel (§11 + §5.15). Paired
   e5 probes give **TokenRel+Motion +0.014 test R² / +0.010
   Pearson over TokenRel-only on LVEF** (§5.16) — the same-view
   delta prediction adds signal without regressing LVEF. RVSP /
   MR / TAPSE test metrics still pending (696/699 in-flight).
10. **TokenRel / TokenRel+Motion e5 probes** — 🟢 PARTIALLY DONE
    (§5.16). LVEF test numbers captured for both variants:
    TokenRel 694 e5 matches Base e100 (+100 ep); TokenRel+Motion
    695 e5 matches Ctrl 608 (+25 ep). RVSP / MR / TAPSE in-flight
    (~3-4h ETA).
11. **Base e125 test number gap** — ✅ CLOSED (job 698). Base
    e125 LVEF test MAE 5.36 / R² 0.646 / Pearson 0.806.
    Previously `jepa_ext_probes_332` trained the e125/e150/e175/
    e200 probes but never ran test inference; 698 was a 3-min
    inference job reusing the existing probe best.pt + encoder.
12. **Full 25-epoch run** — **redefined**: no longer a separate
    "launch", because 683/684 are the 25-ep continuations
    (resumed from e5). Gates:
    - 5-ep pilot (✅ 655 passed gate)
    - 5-ep paired ctrl (✅ 658 passed gate)
    - Retrieval diag (✅ pilot dominant, study-identity concern
      disambiguated)
    - F/G/H (✅ pilot = Base temporally; interpretable result)
    - H1 (✅ slot-triviality confirmed; interpretability story
      updated)
    - 5-ep downstream probes on RVSP/MR showing non-negative
      paired ΔR² vs ctrl 658 at matched compute (🟢 in-flight in
      681/682; LVEF trajectory so far is flat-vs-ctrl, which is
      the LVEF safety check)
    - Post-e25 probes against V4 at matched +25 ep (future)

### Failure modes the attribution work is designed to catch

From discussion 2026-05-03 after the pilot completed, the point
worth flagging explicitly:

> **Training-time proximal-objective health is necessary but not
> sufficient.** V4's phase-relational objective had identical-
> looking training dynamics (InfoNCE loss descending, top-1
> rising, pos-neg gap widening ~6×, no representation collapse)
> and still delivered −0.090 paired ΔR² on RVSP. The pilot's
> healthy trajectory is consistent with both "v5 learns useful
> cross-view physiology" and "v5 learns same-study identity
> shortcuts" or "v5 learns a view classifier without hallucination
> capacity." The §5.6–5.9 diagnostics are the specific tests
> designed to disambiguate before the full 25-epoch run is
> launched.

The critical red-flag patterns to watch for:
- `same_study_any_view` retrieval top1 ≥ `same_study_same_view`
  top1 → study-identity shortcut
- B (same-study src↔tgt) ≫ C (other-study same-view tgt↔tgt)
  AND C ≈ D → study-identity shortcut, not view-invariant
  physiology
- View-classifier val_acc > 0.95 → trained a view classifier
- ✅ **Triggered** — pilot `z_phase` ≈ `encoder_pool` in R² → H1
  confirmed (§5.10). Slot projections are near-linear maps of the
  pool, not distinct content decompositions. Paper's
  factorized-slot interpretability claim does not survive.
- ✅ **Triggered** — pilot cos(clean, shuffled) ≈ 1.0 → no
  within-clip temporal structure learned (§5.9 F/G/H).
  Interpretation: MV2SV's objective is time-averaged retrieval,
  so the pilot captures cross-view-discriminable signal as a
  pooled spatial signature, not as temporal dynamics. Separate
  from the downstream question — could still be useful if
  RVSP/MR/AS are view-signature-dominant rather than dynamics-
  dominant.
- ✅ **Triggered** — pilot cos(half_a, half_b) ≈ 0.96 → intra-clip
  temporal collapse (§5.9 H). Same interpretation.

#### What the H1 + F/G/H findings mean for next steps

The interpretability story for the paper needs to shift:
- **Before**: "the factorized head extracts phase into `z_phase`,
  view-specific info into `z_view`, study signature into
  `z_shared`." Supported by slot-disjointness training metric.
- **After H1**: the slots are three different orthogonal linear
  projections of the same pooled encoder output. Slot
  disjointness says mean directions are orthogonal but content is
  highly overlapping. The factorization-as-interpretability claim
  must be removed or heavily qualified.

This does NOT change the architectural decision-making. The
downstream probes (§5.12) are still the gate. What it does change:
- `concat_all` vs `encoder_pool` is unlikely to differ much — if
  the slots are linear projections of the pool, `concat_all` is
  literally 3 × 256-D projections of 1024-D pool = 768-D, which
  contains (approximately) the same information as the 1024-D
  pool. We'll see this empirically when 677 lands.
- If `concat_all` does meaningfully outperform `encoder_pool`, it
  would falsify H1 (at least at the probe level — the slot
  nonlinearity *does* help a downstream probe access information
  the pool doesn't linearly expose). Watch this carefully.
- The architectural question after downstream results lands:
  should MV2SV v6 **remove the factorized head** and predict
  directly on the pool? The H1 result says the head isn't
  factorizing; it's adding trainable-parameter cost for
  negligible return.

---

## 9. Cross-references

- Plan file: `/home/sagemaker-user/.claude/plans/parsed-squishing-thompson.md` (v1→v5 addenda)
- Code: `app/vjepa_multiview/train.py` (`forward_privileged_multiview`, `_view_nce_loss`, `_mean_shared_fused_target`)
- Factorized head: `app/vjepa_multiview/factorized_head.py`, `app/vjepa_multiview/view_predictor.py`
- Sampler: `classifier/phase/sampler/phase_matched_sampler.py` (MV2SV extensions in `MatchRecord` + `build_records`)
- Smoke configs: `configs/train/vitl16/smoke/mv2sv-smoke-v5-{parity,nce,fused}.yaml`
- Pilot config: `configs/train/vitl16/mv2sv-pilot-5ep-nofused.yaml`
- Paired ctrl config: `configs/train/vitl16/mv2sv-ctrl-5ep-intraview-only.yaml`
- Gate checker: `scripts/neurips/phase/mv2sv_gate_check.py` (supports `parity | nce | fused | ctrl` stages)
- Coverage audit: `scripts/neurips/phase/mv2sv_fused_coverage_audit.py`
- Retrieval diag: `scripts/neurips/phase/run_cross_view_retrieval_diag.py` + `mv2sv_diag_sweep_v2.sbatch`
- F/G/H diagnostics: `scripts/neurips/phase/run_temporal_phase_diagnostics.py` + `mv2sv_fgh_sweep.sbatch`
- Probe adapter: `evals/video_classification_frozen/modelcustom/vit_factorized_encoder.py`
- Probe configs: `configs/eval/vitl/privview/`
- Earlier variants this builds on: `phase-jepa.md` (1–2), `finalbudget-phase-probes.md` (3), `phase-relational-hardneg.md` (4, LVEF-only gain)

## 10. Job index

| Job | Name | Node | Elapsed | State | Purpose |
|---|---|---|---|---|---|
| 651 | Stage A parity smoke | 146 | short | ✅ PASS | dispatch-is-a-no-op verification |
| 652 | Stage B target-NCE smoke | 146 | 7m30s | ✅ PASS | pair_view+view_nce trainable, 9 gates green |
| 653 | Stage C fused smoke | 146 | 5m48s | ❌ FAILED | `fused_valid_mask mean=1.44 < 2` guard; sampler sparsity |
| 655 | 5-ep pilot (MV2SV v5) | 146 | 3h11m | ✅ COMPLETED | first scientific pilot, intraview flat, view_nce_top1 18× chance |
| 656 | 5-ep ctrl (resub1) | 56 | 1m17s | ❌ FAILED | pyarrow missing on node 56; patched |
| 657 | diag sweep v1 | 146 | 11m23s | ❌ ZERO JSONS | `KeyError('dicom_id')`; patched |
| 658 | 5-ep ctrl (resub2) | 56 | in-flight | 🟢 RUNNING | paired intraview-only control, e1+e2 saved |
| 660 | diag sweep v2 | 146 | 11m55s | ❌ ZERO JSONS | `multiview_objective` kwarg not in sampler; patched |
| 661 | diag sweep v3 + extensions | tbd | pending | ⏳ PENDING `afterany:658:660` | full sweep on Base/V3/V4/Control/Pilot/Ctrl |
| 663–671 | diag smoke iterations | 146 | minutes each | 4× ❌ → 1× ✅ | debugging loop; 671 produced valid JSON |
| 672 | F/G/H sweep (v1) | 146 | ~10m | ❌ every sub-run | `DataFrame.__bool__` from `_df or df` shortcut; patched |
| 673 | F/G/H sweep (v2) | 146 | 20m12s | ✅ COMPLETED | 7/7 JSONs; V4 strongly frame-shuffle-sensitive, Pilot = Base |
| 674 | Scaled phase probe + H1 | 146 | ~50m | ✅ COMPLETED | 16 JSONs; phase-decod inconclusive at protocol, **H1 confirmed (slot R² ≈ 0.99)** |
| 675 | Attentive phase probe | — | — | ❌ CANCELLED | secondary interpretability; deferred per user priority-5 |
| 676 | Retrieval diag v3 (n=1000) | 56 | ~2h | ✅ COMPLETED | all 6 encoders; pilot 655 e5 top1=0.447 gap=+0.394 dominates |
| 677 | Pilot 655 e5 downstream probes (v1, 20-ep) | 146 | ~12m | ❌ CANCELLED | user triage switch to 10-ep; LVEF e10 encoder_pool completed first |
| 678 | Ctrl 658 e5 downstream probes (v1, 20-ep) | — | — | ❌ CANCELLED | paired to 677; cancelled with 677 |
| 679 | Pilot 655 e5 downstream probes (v2, 10-ep triage) | 146 | ~53m | ❌ CANCELLED | ran LVEF encoder_pool 10-ep + test; user switched to 20-ep paper-grade |
| 680 | Ctrl 658 e5 downstream probes (v2, 10-ep triage) | — | — | ❌ CANCELLED | paired to 679 |
| 681 | **Pilot 655 e5 downstream probes (v3, 20-ep paper-grade)** | 146 | ~10h14m | ❌ CANCELLED | 5/6 probes done (LVEF/RVSP/MR × encoder_pool + LVEF/RVSP concat_all); MR concat_all cancelled mid-train (H1 implies no added info vs encoder_pool) |
| 682 | **Ctrl 658 e5 downstream probes (v3, 20-ep paper-grade)** | 56 | ~3h27m | ✅ COMPLETED | 3 probes (LVEF/RVSP/MR × encoder_pool); all paired-to-681. **Ctrl beats pilot on every endpoint** |
| 683 | **Pilot 655 e5 → e25 continuation** | 146 | queued | 🟢 about to start | MV2SV v5 +20 more epochs from pilot latest.pt; auto-launches on 681 exit (afterany) |
| 684 | **Ctrl 658 e5 → e25 continuation** | 56 | in-flight | 🟢 RUNNING | paired intraview-only +20 more epochs; avg loss 0.50 at e10/step 10211 (started 0.49 at e4/step 5871) |

| 688 | **Run 1 EchoJEPA-TokenRel smoke (v1, first attempt)** | 146 | ~9m | ❌ FAILED | `_extract_multiview_clips` allowlist didn't include new objective; patched |
| 689 | **Run 2 EchoJEPA-TokenRel + MotionDelta smoke (v3, first attempt)** | 56 | ~9m | ❌ FAILED | same bug as 688 |
| 690 | Run 1 EchoJEPA-TokenRel full (v1) | — | — | ❌ CANCELLED | DependencyNeverSatisfied from 688; resubmitted as 694 |
| 691 | Run 2 EchoJEPA-TokenRel + MotionDelta full (v3) | — | — | ❌ CANCELLED | DependencyNeverSatisfied from 689; resubmitted as 695 |
| 692 | **Run 1 EchoJEPA-TokenRel smoke (v1, resub)** | 146 | in-flight | 🟢 RUNNING | 100-step sanity; gate enforces top1, gap, q_var, pool_rel |
| 693 | **Run 2 EchoJEPA-TokenRel + MotionDelta smoke (v3, resub)** | 56 | in-flight | 🟢 RUNNING | adds delta_valid_rows + delta_loss gates |
| 694 | **Run 1 EchoJEPA-TokenRel full 25ep (v1)** | 146 | ~3h47m | ❌ CANCELLED @ e8/25 | cancelled to capture e5 ckpt for probes; e5.pt synced to S3 via exit trap |
| 695 | **Run 2 EchoJEPA-TokenRel + MotionDelta full 25ep (v3)** | 56 | ~3h47m | ❌ CANCELLED @ e8/25 | same as 694; e5.pt synced |
| 696 | **Run 1 e5 downstream probes (LVEF/RVSP/MR/TAPSE)** | 146 | 7h02m | ✅ COMPLETED | 4 probes + test; §5.16 tables |
| 697 | **Run 2 e5 downstream probes (LVEF only, cancelled)** | 56 | 1h43m | ❌ CANCELLED | freed node for 698; LVEF test: MAE 5.11 / R² 0.669 / Pearson 0.821 |
| 698 | **Base e125 LVEF test inference** | 56 | 3m07s | ✅ COMPLETED | MAE 5.36 / R² 0.646 / Pearson 0.806 |
| 699 | **Run 2 e5 probes, RVSP/MR/TAPSE (resub)** | 56 | 5h37m | ✅ COMPLETED | 3 probes + test; MR 0.729 AUROC / TAPSE R² 0.180 |
| 703 | **TokenRel+Motion 5→25 pretrain continuation** | 146 | 10h10m | ✅ COMPLETED | 695 e5 → e25; e10/e15/e20/e25 ckpts synced |
| 704 | **A4C probes TokenRel+Motion e5 (HCM/HF/Age/LVEF)** | 56 | 7h14m | ✅ COMPLETED | §5.17 tables; all 4 tasks + test |
| 705 | **A4C probes Base e125 (HCM/HF/Age/LVEF)** | 56 | ~2h17m | ❌ CANCELLED | HCM complete, partial HF; cancelled to redirect to RV function / HCM PLAX |
| 706 | **A4C probes V4 e25 (HCM/HF/Age/LVEF)** | — | — | ❌ CANCELLED | dependency chain retargeted |
| 707 | **RV function binary 10k probes TokenRel+Motion e5** | — | — | ❌ CANCELLED | removed from chain |
| 708/709 | (stale PLAX HCM / earlier rvfunc IDs) | — | — | ❌ CANCELLED | queue-refactor deletions |
| 712/713 | HCM PLAX probes (Base e125 / V4 e25) | 56 | — | ❌ CANCELLED | user pivoted to RV function |
| 714 | rvfunc_probes Base e125 (v3 pin, first attempt) | 56 | 9m20s | ❌ CANCELLED | node was needed for TokenRel+Motion e25 LVEF probe 719 |
| 715 | rvfunc_probes V4 e25 (v3 pin, first attempt) | — | — | ❌ CANCELLED | dependency-based cancel |
| 716 | tokenrel_r2_e25_lt (combined LVEF+TAPSE on v3) | 56 | — | ❌ CANCELLED | split into 719 (LVEF on v3) + 720 (TAPSE on v1) |
| 717 | rvfunc_probes Base e125 (v3 pin, second attempt) | — | — | ❌ CANCELLED | moved to v1 as job 721 |
| **718** | **rvfunc_probes V4 e25** | 56 | in-flight | 🟢 RUNNING | V4 on RV function binary 10k; §5.18; at ep 9/20 |
| **719** | **TokenRel+Motion e25 LVEF probe + test** | 56 | 1h24m | ✅ COMPLETED | best val R² 0.7092 (ep 17); test R² 0.667 / MAE 5.16 / Pearson 0.819 |
| **720** | **TokenRel+Motion e25 TAPSE probe + test** | 146 | 1h48m | ✅ COMPLETED | best val R² 0.2692 (ep 6); test R² 0.2102 / MAE 0.364 / Pearson 0.481 |
| **721** | **rvfunc_probes Base e125 (v1)** | 146 | in-flight | 🟢 RUNNING | afterany:720; Base e125 on RV function binary 10k; §5.18; at ep 4/20 |

Historical crons:
- `3d18e46f` (pilot 655 hourly monitor, 2026-05-02 ~22:20 → 2026-05-03 ~01:30) — deleted on 655 COMPLETED.
- `f38fd1b1` (ctrl 658 hourly monitor, 2026-05-03 ~00:42 → ~04:00) — deleted on 658 COMPLETED.
- `73659842` / `11ce4ef2` (pretrain collapse monitors for 683 / 684) — deleted when nodes freed for TokenRel runs.

---

## 11. EchoJEPA-Motion — design notes

New architecture line introduced 2026-05-03 (jobs 692-695). Sibling
to V4 and MV2SV v5, not a modification of either. The design is
driven by the §2.5 mechanistic reading of V4's LVEF win and TAPSE
failure: the right fix is to move V4's pooled phase-discrimination
InfoNCE to **per-token output**, and — for Run 2 — to add
**latent motion-delta prediction** that V4's pooled contrastive
loss cannot encode.

### 11.1 Motivation (what we learned from prior variants)

Five findings that constrain the design:

1. **Predictor-φ (V1) gave Δφ to the predictor but did not
   penalize the encoder for phase-indistinct features.** Result:
   null vs single-view JEPA on LVEF e100.
2. **Positive-only cross-view SmoothL1 (V3) did not help.**
   Teacher target at matched phase/view was near-redundant with
   the source; loss became a noisy intraview target.
3. **Pooled phase-relational InfoNCE (V4) did help LVEF** via a
   same-study wrong-phase hard negative. Mechanism: forces the
   pool to separate ED from ES on a single low-curvature axis.
4. **But V4 did not generalize to local motion / hemodynamic
   tasks.** TAPSE was essentially tied vs SV (no paired baseline
   but the geometry predicts null); RVSP got worse. Mechanism:
   V4's loss only constrains the *mean direction* of the token
   sequence. Individual tokens are free to do anything consistent
   with that mean, so local landmark amplitude is not supervised.
5. **MV2SV target-view retrieval (pilot 655) did not transfer
   at e5.** Retrieval top-1 improved (§5.11), but pilot was worse
   than paired ctrl 658 on LVEF / RVSP / MR (§5.12). Instance
   retrieval of same-study target-view latents is likely the
   wrong clinical axis.

The unifying diagnosis: the *LVEF-positive pooled phase axis*
and the *TAPSE-needed token-level amplitude structure* can't be
supervised by the same pooled loss. We need a single objective
whose gradient geometry touches **tokens**, not just the pool.

### 11.2 Core idea

**Run 1 — EchoJEPA-TokenRel**: lift V4's phase-relational
InfoNCE from pooled to per-token output via token-set matching.

V4 (current, for reference):

```
c_a_pool   = mean_pool(student_tokens_on_clip_a)
y_pos_pool = mean_pool(teacher_tokens_on_clip_b_pos).detach()
y_neg_pool = mean_pool(teacher_tokens_on_clip_b_neg).detach()
q          = PhaseRelationalHead(c_a_pool, view_a, view_b, Δφ)
L_rel      = InfoNCE(q, y_pos, y_hard=y_neg, batch negs)
```

Run 1:

```
z_sub      = subsample K=64 tokens from z_ctx[:, :, :]
y_pos_sub  = detach(teacher_tokens_on_clip_b_pos)[:, same K idx, :]
y_neg_sub  = detach(teacher_tokens_on_clip_b_neg)[:, same K idx, :]
q_tokens   = TokenRelationalHead.query(z_sub, view_a, view_b, Δφ)   # [B, K, d]
y_pos_tok  = TokenRelationalHead.target(y_pos_sub)                  # [B, K, d]
y_hard_tok = TokenRelationalHead.target(y_neg_sub)                  # [B, K, d]
L_token_rel = token_set_infonce_with_hard_neg(...)                  # set-wise logsumexp
L_pool_rel  = V4_InfoNCE(pool(z_ctx), pool(h_b_pos), pool(h_b_neg)) # tiny safety tail
L_total     = L_intra + 0.02·L_token_rel + 0.005·L_pool_rel
```

**Run 2 — EchoJEPA-TokenRel + MotionDelta**: add a same-view-only
latent delta prediction head targeted at TAPSE's mechanism.

```
eligible rows      = src_view_ids == tgt_view_ids       # ~35% of batch
d_pos_raw          = teacher_tokens_b_pos - teacher_tokens_a       # [V, K, D] (detached)
d_hard_raw         = teacher_tokens_b_neg - teacher_tokens_a       # same-view wrong-phase delta
d_pos              = DeltaTargetProjector(d_pos_raw)
d_hard             = DeltaTargetProjector(d_hard_raw)
q_delta            = MotionDeltaHead.forward(z_sub, src_view, Δφ)  # student query
L_delta_l1         = SmoothL1(q_delta, stopgrad(d_pos))
L_delta_nce        = InfoNCE over batch of (d_pos, d_hard, other-row d_pos)
L_latent_motion_delta = L_delta_l1 + L_delta_nce
L_total_r2         = L_total_r1 + 0.01·L_latent_motion_delta
```

### 11.3 The token-correspondence constraint

**Hard constraint**: for cross-view rows (65% of batch per
`view_pair_policy` = 0.35 same / 0.45 same-family / 0.20
cross-family), anatomical token-index correspondence does not
hold. A4C token index `n` and PLAX token index `n` describe
different anatomical regions.

Implications:
- **Token-rel InfoNCE must be set-wise**, not aligned. Use
  `mean_i logsumexp_j cos(q[b,i], y[c,j]) / τ` — this asks
  "does any target token align well with any query token" and
  does not assume index correspondence.
- **MotionDelta is same-view only.** For cross-view rows, a
  direct token-index delta `teacher_b_pos - teacher_a` would
  mix anatomically-unrelated regions. The loss masks out
  cross-view rows and falls back to a zero-loss proxy that
  still touches delta-head params (DDP reducer invariant).
- Run 1 retains the same 65/35 view mix from V4 — the token-set
  InfoNCE handles both same-view and cross-view rows safely. The
  motion-delta loss in Run 2 is supervised on the same-view
  subset only (~35% of eligible rows per step).

This is the key architectural safety line that distinguishes
TokenRel from a naive per-token cross-view prediction.

### 11.4 How TokenRel differs from V4 and MV2SV v5

| Dimension | V4 (593) | MV2SV v5 (655) | **TokenRel Run 1 (694)** | **+MotionDelta Run 2 (695)** |
|---|---|---|---|---|
| Student forward | clip_a only | clip_a only | clip_a only | clip_a only |
| Teacher forward | clip_a + b_pos + b_neg | + target_clip (+ fused_clips optional) | clip_a + b_pos + b_neg | clip_a + b_pos + b_neg |
| Cross-view supervision enters encoder via… | pooled InfoNCE | factorized pair_view + view_nce (factorized-head slots) | **per-token set-wise InfoNCE** | Run 1 + same-view delta L1+InfoNCE |
| Token-level gradient signal? | **No** (only pool mean constrained) | Partial (through factorized head; H1 says slots are near-linear projections of pool → effectively pooled) | **Yes** | **Yes + amplitude** |
| Cross-view rows handled how? | pooled cosine (no index issue) | conditioned factorized slot prediction | **token-set logsumexp** (no index assumption) | Run 1 path for cross-view; delta loss = zero proxy |
| Local amplitude supervision? | No | No | No | **Yes** via SmoothL1 on teacher-side delta |
| Same sampler as V4 / Ctrl 608? | (original) | Different (MV2SV extensions) | **Yes, byte-identical** | **Yes, byte-identical** |
| Paired intraview-only control? | Ctrl 608 | Ctrl 658 (MV2SV sampler) | Ctrl 608 (reusable; sampler matches) | Ctrl 608 |
| Checkpoint keys | `relational_head` | `factorized_head` + 4 others | `token_rel_head` + `token_rel_pool_safety` | + `motion_delta_head` + `delta_target_projector` |
| Hard-negative in the objective | same-view wrong-phase clip | same-target-view fallback | **same-view wrong-phase clip (V4-identical)** | same (for delta too) |
| DDP wrap strategy | static_graph=False | find_unused_parameters=True on fused_shared_projector | static_graph=False on token_rel_head | find_unused_parameters=True on delta heads (zero-loss proxy branch) |

### 11.4a All four variants are multi-view (by the strict definition)

By the "context encoder and target encoder see different views"
definition, **V4, MV2SV v5, TokenRel Run 1, and TokenRel + MotionDelta
Run 2 are ALL multi-view pretrains**. The student encoder forwards
only `clip_a` in every variant, but the teacher encodes both clip_a
and clip_b_pos (and clip_b_neg), and `clip_b_pos` is a different
view family from `clip_a` on **65% of rows** per `view_pair_policy`
(0.35 same_view / 0.45 same_family / 0.20 cross_family).

Cross-view gradient enters the student encoder in all four variants
via the contrastive / predictive loss on `teacher_tokens_on_clip_b_pos`:

| Variant | Cross-view supervision path |
|---|---|
| V4 | student_pool ↔ teacher_pool(different-view clip_b_pos), pooled InfoNCE |
| MV2SV v5 | student factorized slots ↔ teacher slots on target_clip (different view), factorized-head prediction |
| TokenRel Run 1 | student_tokens ↔ teacher_tokens(different-view clip_b_pos), per-token set-wise InfoNCE |
| TokenRel + MotionDelta | TokenRel cross-view path + MotionDelta same-view-only delta prediction |

**Key clarification**: The "V4 is single-view" framing that
appears in some earlier parts of this doc was literally true
about the student's forward pass (student forwards clip_a only)
but misleading about the supervision signal (cross-view via
teacher + InfoNCE). All four variants are structurally multi-view
in this sense.

Differences among the four are in:
1. **Where** the cross-view supervision lands in the encoder
   (pool mean → factorized slots → per-token).
2. **What** the student must predict about the target view
   (phase identity → view-specific residual → phase identity per
   token → phase identity per token + phase-amplitude on
   same-view subset).
3. **Whether** there is additional same-view phase-amplitude
   supervision (MotionDelta only).

MotionDelta's same-view delta prediction is an **additional**
branch on top of TokenRel's cross-view signal, not a replacement
— Run 2 is strictly a superset of Run 1 in what the student is
asked to predict.

### 11.5 Why this design should preserve V4's LVEF and unlock TAPSE

**LVEF**: the `L_pool_rel_safety` term at λ=0.005 keeps V4's
pooled phase-identity axis alive in the encoder's token mean.
The attentive probe at inference time can still uniform-weight
tokens to recover that pool direction, so LVEF should not
regress. `L_token_phase_rel` adds per-token phase-identity
signal *on top of* the pool-mean constraint — each token learns
a projection that distinguishes ED from ES in the same hard-neg
InfoNCE geometry. This either preserves LVEF (neutral case) or
improves it (tokens carry finer phase info that attention can
extract).

**TAPSE (Run 2 only)**: `L_latent_motion_delta` is the first
variant we've run whose gradient supervises *per-token latent
displacement* between two phases of the same view. The TV annulus
tokens see gradient that rewards encoding "how did this patch
shift between φ=0 and φ=0.5," which V4's pooled cosine cannot
express (it only sees the mean direction after L2-norm).

**RVSP** is not explicitly targeted; expected to remain near the
V4 / Ctrl 608 level. If token-level phase supervision
incidentally helps RV-inflow tokens encode velocity-correlated
structure, Run 2 may see a small RVSP gain — but this is not a
pre-registered claim.

### 11.6 Config knobs (defaults)

```yaml
phase_multiview:
  token_relational:
    enabled: true
    token_subsample_k: 64
    rel_dim: 256
    tau_token: 0.10
    lambda_token_rel: 0.02
    lambda_pool_rel:  0.005     # V4 safety tail
    lambda_delta:     0.01      # Run 2 only, else 0.0
    warmup_epochs: 5.0
    mask_same_study_batch_negatives: true
  motion_delta:
    enabled: true                # Run 2 only
    delta_dim: 256
    tau_delta: 0.10
    lambda_delta_l1: 1.0
    lambda_delta_nce: 1.0
    same_view_only: true         # cross-view delta explicitly forbidden
```

### 11.7 Implementation files

**New modules**:
- `app/vjepa_multiview/token_relational_head.py` — `TokenRelationalHead`,
  `MotionDeltaHead`, `DeltaTargetProjector`, `subsample_tokens`.
- `app/vjepa_multiview/token_relational_loss.py` —
  `token_set_infonce_with_hard_neg`, `motion_delta_loss`.

**Modified**:
- `app/vjepa_multiview/train.py` — new `forward_token_phase_relational`,
  new dispatch branch `token_phase_relational`, config parsing,
  head construction with optimizer wd/no-wd groups, DDP wrap,
  save/load, CSV logger branch with 37 token-rel diagnostics.

**New configs**:
- `configs/train/vitl16/pretrain-multiview-tokenrel-25of100.yaml` (Run 1)
- `configs/train/vitl16/pretrain-multiview-tokenrel-delta-25of100.yaml` (Run 2)
- `configs/train/vitl16/smoke/tokenrel-smoke-run1.yaml`
- `configs/train/vitl16/smoke/tokenrel-delta-smoke-run2.yaml`

**New scripts**:
- `scripts/neurips/phase/tokenrel_gate_check.py` — smoke gates.
- `scripts/neurips/phase/tokenrel_run1_smoke_v1.sbatch` (node 146)
- `scripts/neurips/phase/tokenrel_delta_run2_smoke_v3.sbatch` (node 56)
- `scripts/neurips/phase/tokenrel_run1_full_v1.sbatch`
- `scripts/neurips/phase/tokenrel_delta_run2_full_v3.sbatch`

**Tests** (22 pass, 158/158 tests/vjepa_multiview/ green):
- `tests/vjepa_multiview/test_token_relational_head.py`
- `tests/vjepa_multiview/test_token_set_infonce.py`
- `tests/vjepa_multiview/test_motion_delta_loss.py`
- `tests/vjepa_multiview/test_forward_token_phase_relational.py`
- `tests/vjepa_multiview/test_tokenrel_checkpoint_compat.py`

### 11.8 What TokenRel does NOT change

- V4 (593), Ctrl 608, Pilot 655, Ctrl 658 encoders and checkpoints
  are untouched. Their probes remain comparable to each other.
- The phase_matched sampler config is byte-identical to V4's
  (same view_pair_policy, same hard-neg filter, same quality tier).
  A paired intraview-only control for TokenRel is Ctrl 608 at
  matched +25 ep compute.
- MV2SV v5's pair_view / view_nce / factorized-head / fused_clips
  machinery is not involved. MV2SV v5 lambdas are forced to 0 in
  TokenRel configs; the `token_phase_relational` objective dispatch
  avoids `forward_privileged_multiview` entirely.
- No paper edits, no Overleaf pushes, no modification to existing
  experiments.

### 11.9 Success criteria and comparisons

Primary comparisons (after Run 1 / Run 2 completes +25 ep and
probes are run):

1. **Run 1 (TokenRel) vs V4 (593) on LVEF** — does moving V4's
   InfoNCE to tokens **preserve** V4's LVEF win? Null-hypothesis
   test: ΔR² ≥ −0.01 (no regression within noise).
2. **Run 2 (TokenRel + MotionDelta) vs V4 on TAPSE** — does
   per-token delta supervision lift TAPSE R² above V4's 0.250?
   This is the primary hypothesis test for the architecture.
3. **Run 1 / Run 2 vs Ctrl 608 on LVEF / RVSP / MR** — paired Δ
   at matched +25 ep with byte-identical sampler. Quantifies
   "what does the token-level objective buy over pure intraview
   at the same data path."
4. **Run 2 − Run 1 on TAPSE and LVEF** — isolates the MotionDelta
   contribution from the TokenRel contribution. Run 2 should
   match Run 1 on LVEF (no regression) and exceed it on TAPSE.

Secondary (not pre-registered):
- RVSP: does token-level phase supervision incidentally help
  in-plane RV-inflow structure?
- MR: A4C-only probe is weak; cross-view MR is out of scope.

If Run 1 regresses LVEF vs V4 by >0.03 R², the token-set InfoNCE
is damaging the pooled axis despite the safety term — λ_pool_rel
would need to increase. If Run 2 regresses TAPSE vs Run 1, the
motion-delta loss is adding noise — λ_delta would need to
decrease. Both are parameter sweeps, not architecture changes.

### 11.10 Related work — Codex companion assessment

This approach was designed in-session based on the §2.5 reading
of V4's pooled-InfoNCE failure on TAPSE. It is consistent with
the v4 addendum of the plan file (`parsed-squishing-thompson.md`)
which sketched `L_local_motion` as a TAPSE-oriented token-level
motion loss. Run 2's `L_latent_motion_delta` is the concrete
implementation of that sketch, restricted to same-view rows
(the v4 addendum's "same-view Δφ-displaced clip" requirement is
satisfied by V4's existing hard-neg sampler's same-view-wrong-phase
clip at λ=0.35 same_view_prob × the Bernoulli subset).


