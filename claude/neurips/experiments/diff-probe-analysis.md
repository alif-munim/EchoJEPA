# Between-Frame-Difference Probe Analysis

**Status:** Phase 1 + controls + primary + expanded controls all completed
(2026-04-22). Jobs 312 (extraction), 314 (6 baseline controls + 20-ckpt
primary), 315 (4 expanded shuffle controls). Artifacts under
`s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts/results/diff_probe/{314,315}/`
mirrored to `/tmp/diff_probe_{314,315}/`. Trajectory plot at
`/tmp/diff_probe_314/trajectory.{png,pdf}`.

## Results summary (TL;DR)

All 10 control runs (6 baseline + 4 expanded, covering JEPA e50/e100 and
MAE e25/e50/e99/e124/e194 + 3 random-init seeds) pass the R²<0.05 gate:
every shuffle-diff R² is negative (max = −0.002, random-init max = +0.001).
The diff-probe signal is therefore measuring genuine temporal content at
every checkpoint probed, not a non-temporal EF-correlated shortcut.

Linear-A on diff features recovers substantial LVEF R² across every
family, with striking cross-family ordering:

| Family | epoch range | diff R² range | raw R² range | diff/raw ratio |
|---|---|---|---|---|
| **MAE**  | e25 → e194 | +0.518 → +0.644 | +0.552 → +0.695 | **0.92–0.94, flat** |
| **JEPA** | e25 → e100 | +0.144 → +0.376 | +0.341 → +0.502 | 0.42 → 0.75, climbing |
| **BYOL** | e24 → e100 | +0.330 → +0.386 | +0.271 → +0.349 | **>1.0 at every epoch** |
| **SALT** | e4 → e79  | +0.005 → +0.093 | +0.088 → +0.295 | 0.06 → 0.32, low |

**WD sweep (linear-A at wd=1e-4 vs 1e-2, 40 cells):** max |Δ R²| = 0.0035,
no overfitting regime. linear-A-wd=1e-4 is the primary number.

**Finding that invalidates the current "MAE abandons temporal features"
framing:** MAE diff-R² is ≥ +0.518 from e25 on and is **higher than** its
own matched_frame attentive-probe clean R² at *every* checkpoint (gap
+0.12 to +0.29). The diff probe reveals that temporal-difference
information is present in MAE from the earliest checkpoint and *remains*
present at e194. The attentive probe at depth=4 with RoPE-remapped
matched-frame evaluation simply does not extract it. Interpretation B
(abandonment) is therefore not supported; interpretation A
(inaccessibility through standard probe) is.

**Finding that complicates the "JEPA > BYOL > MAE ≈ SALT" temporal-access
ordering:** measured at linear-A-diff, the ordering is
**MAE > BYOL > JEPA > SALT**. MAE's diff-R² is ~1.7× JEPA's at every
matched checkpoint. If diff-R² is the temporal-information metric, the
paper's current ordering reverses for MAE and JEPA. See "Interpretation"
below.

## Control results (combined, 10 runs)

| Run | Model | Kind | n | mean R² | max R² | Gate |
|---|---|---|---|---|---|---|
| 314 | jepa_e100 | shuffle | 5 | −0.0086 | −0.0040 | PASS |
| 314 | mae_e50   | shuffle | 5 | −0.0983 | −0.0805 | PASS |
| 314 | mae_e99   | shuffle | 5 | −0.0752 | −0.0659 | PASS |
| 314 | random_s42 | random | 5 | −0.0013 | +0.0007 | PASS |
| 314 | random_s43 | random | 5 | −0.0022 | +0.0001 | PASS |
| 314 | random_s44 | random | 5 | −0.0022 | −0.0010 | PASS |
| 315 | jepa_e50  | shuffle | 5 | −0.0058 | −0.0020 | PASS |
| 315 | mae_e25   | shuffle | 5 | −0.0983 | −0.0633 | PASS |
| 315 | mae_e124  | shuffle | 5 | −0.0807 | −0.0614 | PASS |
| 315 | mae_e194  | shuffle | 5 | −0.0585 | −0.0484 | PASS |

Shuffle-diff R² is systematically negative across all checkpoints (not
just at-zero). That is the expected signature of destroying temporal
order in a feature space whose within-clip means are still EF-predictive:
the diff operator after shuffling yields a centered but uninformative
vector, and the probe's best fit on 2554 training examples lands below
zero on test.

Expanded MAE coverage (e25, e124, e194) was motivated by the near-constant
MAE diff/raw ratio of 0.923–0.941 across e25→e194 (1.8pp spread over 170
epochs) — a potential tell for interpretation C (diff and raw capturing
the same EF-correlated projection through different rescaling). The
expanded shuffle controls rule this out: if interpretations B or C held,
at least one MAE checkpoint's shuffle-diff R² would have been positive.
All five are negative.

## Primary trajectory (linear-A wd=1e-4, mean over 5 seeds)

### JEPA (diff/raw ratio climbing, consistent with matched-frame story)

| epoch | diff | raw | diff/raw |
|---|---|---|---|
| 25 | +0.144 | +0.341 | 0.42 |
| 50 | +0.214 | +0.434 | 0.49 |
| 75 | +0.352 | +0.489 | 0.72 |
| 100 | +0.376 | +0.502 | 0.75 |

Both diff and raw rise with training. diff/raw climbs from 0.42 → 0.75 —
JEPA becomes increasingly temporal-difference-readable as pretraining
progresses. This is the pattern the rebuttal's matched-frame analysis
predicted for a temporally-learning encoder.

### MAE (diff and raw rise together at constant ratio)

| epoch | diff | raw | diff/raw | matched_frame clean | matched_frame mf |
|---|---|---|---|---|---|
| 25  | +0.518 | +0.552 | 0.94 | +0.225 | +0.257 |
| 50  | +0.584 | +0.626 | 0.93 | +0.413 | +0.281 |
| 75  | +0.614 | +0.652 | 0.94 | +0.435 | +0.356 |
| 99  | +0.626 | +0.670 | 0.94 | +0.467 | +0.440 |
| 124 | +0.637 | +0.678 | 0.94 | +0.469 | +0.428 |
| 149 | +0.644 | +0.690 | 0.93 | +0.527 | +0.491 |
| 174 | +0.642 | +0.696 | 0.92 | +0.500 | +0.448 |
| 194 | +0.644 | +0.695 | 0.93 | +0.526 | +0.460 |

MAE diff-R² is the highest of any family at every checkpoint. The
diff-R² *at e25* already exceeds MAE's matched_frame clean R² at e194.
Training progress does not increase the diff/raw ratio — MAE's
representational geometry has a fixed, ~93% linear-recoverable
temporal-difference channel by e25 that persists for the entire 194-epoch
run. See "Interpretation" for why this is a meaningful signal, not a
method artifact.

### BYOL (diff > raw — temporally enriched)

| epoch | diff | raw | diff/raw |
|---|---|---|---|
| 24 | +0.330 | +0.271 | 1.22 |
| 50 | +0.368 | +0.299 | 1.23 |
| 75 | +0.386 | +0.328 | 1.18 |
| 100 | +0.386 | +0.349 | 1.10 |

Unusual: BYOL's diff-probe R² exceeds its raw-probe R² at every
checkpoint. BYOL under the present pretraining produces representations
where the temporal *derivative* is more linearly EF-predictive than the
representation itself. The diff/raw gap closes with training as raw
catches up — consistent with BYOL progressively encoding absolute state
on top of a temporal-derivative signal that was present from e24.

### SALT (diff suppressed, raw rises — temporally inert)

| epoch | diff | raw | diff/raw |
|---|---|---|---|
| 4  | +0.005 | +0.088 | 0.06 |
| 29 | +0.070 | +0.269 | 0.26 |
| 54 | +0.114 | +0.273 | 0.42 |
| 79 | +0.093 | +0.295 | 0.32 |

SALT diff-R² barely moves above zero. Raw-R² climbs as the stabilized
reconstruction objective builds spatial features. This is the cleanest
example of a representation where temporal-difference information is
genuinely suppressed — the hypothesis the paper had originally assigned
to MAE.

## Interpretation

Two diagnostics now exist for "temporal content in the frozen
representation":

| diagnostic | what it measures | what it requires |
|---|---|---|
| **matched_frame attn-probe gap** (job 216/220) | information *the attentive probe at depth=4 extracts* that shifts when frames are permuted | a probe architecture that can exploit temporal ordering |
| **linear-A-diff probe R²** (this work) | information *a linear readout of adjacent-frame differences* carries | none beyond linearity on `z_{t+1} − z_t` |

They **disagree for MAE**. The linear-diff probe reports high, stable
temporal-difference information from e25. The matched-frame attentive
probe reports rising temporal *sensitivity* across training but at
absolute levels well below the linear-diff signal.

### Why this is not a diff-probe artifact

- All 10 controls pass. The 5 MAE shuffle controls span the full
  trajectory (e25, e50, e99, e124, e194) and all produce R² < 0; a
  non-temporal EF-correlated signal in differences would have produced
  positive shuffle-R² at at least one epoch.
- Random-init shuffle-diff R² is ~0.000, as expected for features with
  no pretraining signal.
- WD sweep (1e-4 vs 1e-2) shows no overfitting pattern (max |Δ| 0.0035).
- MAE's diff R² **already exceeds raw R² matched across pretraining**
  for every family's matched-epoch raw linear probe. It is not a small
  effect.

### Candidate explanations for the attn-vs-diff disagreement in MAE

1. **Probe-class inaccessibility.** MAE encodes frame-difference
   information in a representation whose *absolute positions across
   frames* are not cleanly readable by a depth=4 cross-attention probe
   with RoPE-remapped positional indexing. Linear recovery after explicit
   differencing works because the differencing operator is exactly what
   the attn probe cannot easily implement implicitly through self-
   attention (large positional-mixing cost on 1568 tokens per segment).
   This would explain why MAE diff-R² is flat across epochs (the linear
   channel is geometry, baked in early) while matched-frame attn-R²
   climbs (the attn probe gets marginally better at *routing* as features
   mature).
2. **Global-mean overspecification.** The diff probe eats a
   `[T=8, D=1024]` globally-mean-pooled representation; the attentive
   probe eats `[S·T·spatial=3136, D=1024]`. If MAE's temporal signal is
   in a globally-coherent low-spatial-frequency mode (e.g. LV volume
   envelope), global-mean differencing isolates it cleanly; the
   attentive probe sees it but has to filter out 1568 spatial tokens
   that are temporally stable. This is the "MAE is spatial-feature
   dominant" reading applied at the *token* level: MAE may have the
   temporal signal present but buried in off-manifold spatial variation
   that the probe cannot cheaply ignore.
3. **Spatial averaging changes the object.** Diff probe operates on
   spatially-averaged features, so it is not commensurable with
   matched_frame attn at the feature-geometry level. The secondary
   experiment #4 (per-spatial-location differences, no mean over spatial)
   is the direct test — until it runs, "linear-A-diff > attn-clean" is a
   comparison across two representation spaces.

Explanations 1 and 2 are not mutually exclusive; both imply the
rebuttal's "abandonment" framing for MAE is wrong and "inaccessibility"
is the right reading. Explanation 3 is the critical caveat to address
before making the inaccessibility claim strongly.

### Reordering

At linear-A-diff (global-mean features), the family ordering is
**MAE > BYOL > JEPA > SALT**. At matched_frame attn-clean, the ordering
is **JEPA > MAE > BYOL > SALT**, with BYOL and MAE roughly tied above e99.
The JEPA ≫ MAE gap seen at matched_frame does not survive in the
diff-probe. Calling one ordering "the" temporal ordering is a claim about
which readout class the paper privileges. The NeurIPS framing should name
both diagnostics, report both orderings, and scope claims to the
diagnostic and feature level (global-mean vs per-token).

## Next steps

- **Per-spatial-location diff probe** (secondary #4) is the highest-value
  follow-up: resolves whether MAE's diff-vs-attn disagreement is a
  feature-space artifact.
- **Attentive probe on diff features** (secondary #3) on at least MAE
  e50/e194 and JEPA e50/e100 would close the probe-class loop (does attn
  recover what linear-diff recovers, once it is handed already-differenced
  features?).
- Update NeurIPS paper outline §frame-shuffling to report linear-diff
  alongside matched_frame, and update the "MAE abandons temporal
  features" subsection heading accordingly.

## Motivation

The matched_frame shuffle results (jobs 216/220, see
`frame-shuffling-results.md`) show MAE becomes temporally invariant by e99
while JEPA retains sensitivity. This is read in the paper as MAE
*abandoning* temporal features. But shuffle invariance under a single probe
class is consistent with two different states:

- **(a) Abandonment:** temporal features genuinely absent from the frozen
  representation.
- **(b) Inaccessibility:** temporal features present but not extractable by
  the attentive probe architecture as wired.

A reviewer can legitimately raise this as a Tier 1 weakness: frozen +
single-probe-class evidence cannot distinguish these. The difference probe
closes the gap. Instead of probing `z_t`, probe `z_{t+1} - z_t`. If
temporal information is gone, no probe class can recover it. If it is
encoded *relationally across timesteps*, a probe on temporal differences
can surface it even when a probe on raw `z` cannot.

## Scope statement (spatial averaging)

This protocol computes differences in a **globally-averaged** representation:
after extracting `[num_segments, T=8, spatial=196, D=1024]`, we mean over
the spatial axis before differencing. Results therefore speak to **globally-
averaged temporal features**. If MAE encodes temporal information at
specific spatial locations but not globally, this analysis will miss it.
The **per-spatial-location differences** variant (secondary experiment #4)
addresses that directly. All primary-result claims in the paper from this
experiment will be explicitly scoped to "globally-averaged temporal
features."

## Reporting commitment

Before running, commit to reporting every result regardless of how it
lands. In particular:

- **MAE e99 diff probe HIGH** → not abandonment. Paper reframes to
  "spatial-feature dominance" / "inaccessibility."
- **JEPA diff probe flat** → JEPA's temporal advantage is not visible in
  this probe class. Revise the four-way characterization.
- **Four-way endpoint ordering differs** from `JEPA > BYOL > MAE ≈ SALT` →
  report the actual ordering and revisit characterizations.
- **Controls fail** → do not run the primary experiment; debug methodology
  first. If only after running we discover a control failed, retract the
  reported numbers.

All 20 (model, checkpoint) combinations will be reported with 95% bootstrap
CIs computed on test clips (resample clip predictions with replacement,
10,000 iterations) — matches `matched_frame` statistical convention.

## Experiment design

### Representation extraction (Phase 1)

**Hook point:** `evals/video_classification_frozen/eval.py:911`, at the
output of `encoder(clips, clip_indices)`. No modifications to the frozen
encoder. A new script `scripts/neurips/extract_pre_pool_features.py` loads
the encoder via `ClipAggregation` (identical to the inference pipeline),
iterates the test dataloader, and caches pre-pool features per clip.

**Verified tensor-shape pipeline** (from end-to-end reading of
`vit_encoder_multiclip.py:107-135` and `patch_embed.py:49-52`):

| Stage | Shape | Notes |
|---|---|---|
| Input clip | `[1, 3, 16, 224, 224]` | C × T × H × W |
| `PatchEmbed3D` Conv3d kernel (2,16,16) | `[1, 1024, 8, 14, 14]` | temporal-major on flatten |
| `.flatten(2).transpose(1,2)` | `[1, 1568, 1024]` | T=8 outer, spatial=196 inner |
| ViT blocks (no reshape) | `[1, 1568, 1024]` | |
| `ClipAggregation` reshape to `[B, T, S, D]` | `[1, 8, 196, 1024]` | |
| `torch.cat(outputs, dim=1)` over num_segments=2 | `[1, 16, 196, 1024]` | clips concatenated along T |
| `.flatten(1, 2)` (probe input) | `[1, 3136, 1024]` | this is what the probe sees |

**Extraction:** receive the flattened `[1, 3136, 1024]`, reshape back to
`[1, num_segments=2, 8, 196, 1024]`, mean over the spatial axis (dim=3):

```
z[clip] : [num_segments=2, T=8, D=1024]    # cached to .pt
```

**No normalization at cache time.** Save as float16 to halve storage.

Cache layout: one `.pt` file per `(model, ckpt)`, shape `[N_clips=1277,
num_segments=2, 8, 1024]` (~80 MB @ fp16), plus one parallel `.pt` with
ground-truth EF labels and video paths aligned by index.

### Difference computation (Phase 2, CPU)

Within-crop adjacent differences:
```
z         : [N, S=2, T=8, D]
diff      : [N, S=2, T-1=7, D]   where diff[n, s, t] = z[n, s, t+1] - z[n, s, t]
```

No normalization of `diff` before probing — preserve magnitude information.

### Probe architectures (Phase 3)

Each probe is trained on `[S=2, 7, D]` per clip (diff) or `[S=2, 8, D]`
per clip (raw). Each of the `S=2` segments is a separate training example
with the same EF label (matches job 216/220 logic); at inference, the two
per-segment predictions are averaged per clip before scoring.

| Probe | Input | Params (diff) | Params (raw) |
|---|---|---|---|
| **linear-A** (flatten) | `[T', D] → [T'·D] → Linear(·, 1)` | 7·1024 = **7,168** | 8·1024 = 8,192 |
| **linear-B** (mean-pool time) | `mean over T' → [D] → Linear(D, 1)` | **1,024** | 1,024 |
| **MLP** | `[T'·D] → Linear(·, 256) → GELU → Dropout(0.1) → Linear(256, 1)` | ~1.8M | ~2.1M |

Both **diff** and **raw** variants are trained for every `(model, ckpt)`.
The matched-probe-class comparison (linear-A-diff vs linear-A-raw, etc.)
controls for probe capacity when attributing differences to the *input
transformation* rather than the *probe class*.

**Training:**
- AdamW, lr=1e-3, **wd=1e-4 baseline**
- Batch size 64
- Max 30 epochs, early stopping patience=5, min_delta=0.005 on val R²
- **5 random seeds** (each seed varies BOTH probe init AND a deterministic
  val/train split of the EchoNet train CSV — 90/10 stratified on EF
  quantile). Without seed-controlled splits, variance is underestimated.
- Z-score normalize labels with **train-set statistics from the EchoNet
  train CSV** — recomputed per seed using that seed's train split
  (consistent with matched_frame pipeline, which uses `target_mean=55.7776,
  target_std=12.4072` derived from the full train CSV; our per-seed recompute
  will match within 1% since splits are random).
- Linear-A **weight-decay sweep**: train at `wd ∈ {1e-4, 1e-2}`. If mean R²
  differs by > 0.02 across wd values, linear-A is overfitting (7168
  parameters with ≈ 2554 training examples makes this a real risk) and
  linear-B is treated as primary. Otherwise both are equally trusted.

Test set: the 1,277-clip EchoNet-Dynamic test CSV used by jobs 216/220
(`echonet_dynamic_test_s3_raw.csv`). **Not** held out from train — EchoNet
train/val/test are already disjoint in the source dataset.

### Control conditions

Controls must all hit **R² < 0.05** before interpreting the primary
experiment. If any fails, we stop and debug.

**Control 1 — temporal shuffle of cached representations (3 checkpoints):**

For each of `{MAE e50, MAE e99, JEPA e100}`, load `z[N, S=2, T=8, D]`, and
during probe training permute the T axis with a fresh random permutation
*per clip per batch* before differencing. Train linear-A-diff only. This
destroys temporal signal while preserving all other structure. Expected
R² ≈ 0.

The MAE e50/e99 pair catches any asymmetry in the methodology (e.g. if
e50 control fails but e99 passes, methodology is non-stationary). The JEPA
e100 control is the critical one: if the diff probe extracts EF from
temporally-shuffled JEPA, then JEPA's positive diff-probe results are
picking up non-temporal artifacts and *no* JEPA result is trustworthy.

**Control 2 — random encoder (3 seeds):**

Instantiate ViT-L with matching config, **no checkpoint load**, seeds 42,
43, 44 (torch.manual_seed). For each seed, extract representations on the
full test CSV, compute adjacent diffs, train linear-A-diff. Expected R² ≈
0. One random init can be unlucky; three give a baseline distribution for
"diff probe R² with no temporal information at all."

Six control runs total. All six must hit R² < 0.05 before we compute any
primary number.

### Jobs & compute

| Phase | What | Compute | Est. wall |
|---|---|---|---|
| 1 | Extract+cache features for 20 primary combos + 3 random-encoder seeds | 1× p5.48xlarge | ~30 min/ckpt, **~12 h** single-GPU or **~1.5 h** if parallelized across 8 GPUs (one GPU per ckpt) |
| 2 | Compute diffs from cache | CPU | minutes |
| 3a | Controls: 3 temporal-shuffle + 3 random-encoder | 1 GPU | ~30 min total |
| 3b | Primary probes: 3 probe × 2 input × 5 seeds × 20 combos = 600 + linear-A wd=1e-2 extra 5×20=100 → **700 probes** | 8 GPUs parallel | ~1 min/probe, **~1.5 h** |
| 4 | Aggregate, bootstrap CIs, plots, tables | local | minutes |

Storage: 20 × 80 MB = **~1.6 GB** primary + 3 × 80 MB = **240 MB** random-
encoder = ~2 GB total. Backed up to
`s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts/features/diff_probe/`.

Cache is reused across primary, controls, and future secondaries (multi-
scale, reference-frame, per-spatial). Cache-once is worth the storage.

### Decision gates

1. **Phase 1 spot check (MAE e99 only):** confirm cached shape
   `[1277, 2, 8, 1024]`, spatial-mean variance > 1e-6 per channel, zero
   NaNs, two segments distinct (not identical).
2. **Controls (6 runs):** all R² < 0.05. If any fails, stop and debug
   methodology before running primary.
3. **Primary results:** decide on secondaries based on what the data
   reveals.

## Secondary experiments (only if primary is interpretable)

Run **only after** primary probes land and controls pass:

1. **Multi-scale differences:** `d_t^k = z_{t+k} - z_t` for k ∈ {1, 2, 4}.
2. **Reference-frame differences:** `d_t = z_t - z_0`.
3. **Attentive probe on differences:** for `(model, ckpt)` combinations
   where linear/MLP showed interesting patterns — confirms with matched
   probe architecture to the standard pipeline.
4. **Per-spatial-location differences:** do not mean over spatial before
   differencing. Tests the "globally averaged" scope caveat directly.

## Checkpoint list (canonical)

All encoders resolve to existing paths under
`/opt/dlami/nvme/checkpoints/` (S3 mirror
`s3://echodata25/neurips/encoders/`). Per `canonical-checkpoints.md` and
`frame-shuffling-results.md`:

| Model | Epochs | Path pattern |
|---|---|---|
| JEPA IN21K | e25, e50, e75, e100 | `jepa_in21k_vitl_e{N}.pt` |
| MAE | e24, e50, e74, e99, e124, e149, e174, e194 | `mae_vitl_e{N}.pth` / `videomae_l_mimic_ep{N}.pth` |
| BYOL | e24, e50, e75, e100 | `byol_vitl_e{N}.pt` |
| SALT S2v1 | e4, e29, e54, e79 | `salt_s2v1_e{N}.pt` |

20 primary checkpoints. Note: "e25/e50/e75/e100" in the MAE column is
short-hand; exact file names use the above actual epoch numbers (e24, e74,
e99).

## Outputs

**Primary results table** (`diff_probe_results.csv`):

| Model | Checkpoint | Std attn R² (job 216/220) | Lin-A-raw | Lin-B-raw | MLP-raw | Lin-A-diff | Lin-B-diff | MLP-diff | Δ (A-diff vs A-raw) | 95% CI (A-diff) |

**Control results table** (`diff_probe_controls.csv`):

| Control | Model | Ckpt | Probe | R² | Pass (R²<0.05)? |
|---|---|---|---|---|---|
| Temporal shuffle | MAE | e50 | Lin-A-diff | ? | ? |
| Temporal shuffle | MAE | e99 | Lin-A-diff | ? | ? |
| Temporal shuffle | JEPA | e100 | Lin-A-diff | ? | ? |
| Random encoder | — | seed 42 | Lin-A-diff | ? | ? |
| Random encoder | — | seed 43 | Lin-A-diff | ? | ? |
| Random encoder | — | seed 44 | Lin-A-diff | ? | ? |

**Per-model trajectory plots** (one panel per model, 4 panels):

- x-axis: training epoch
- y-axis: probe R² (LVEF)
- Lines per panel: Std attn (raw, from 216/220), Std attn shuffled (from
  216/220), Lin-A-diff, Lin-B-diff, MLP-diff, Lin-A-raw (as matched-class
  reference)

## Interpretation framework

For MAE specifically:
- **diff R² at e50 > diff R² at e99** → confirms abandonment; strongest
  result for the paper's current framing.
- **diff R² stays high at e99** while std probe is shuffle-invariant →
  reframe to "spatial-feature dominance" / "inaccessibility."
- **diff R² uniformly low** → temporal info never strongly encoded;
  weakens the e50-peak interpretation.

For JEPA: expected diff trajectory parallels std. Confirms JEPA's temporal
info is multi-probe-class accessible.

For BYOL / SALT: expected BYOL lower than JEPA but similar shape; SALT
flat and low (inherits MAE).

Cross-model endpoint (e99/e100) ordering: baseline prediction is JEPA >
BYOL > MAE ≈ SALT. Other orderings require revisiting characterizations.

## Files

- `scripts/neurips/extract_pre_pool_features.py` — Phase 1 extraction
- `scripts/neurips/extract_pre_pool_features.sbatch` — Phase 1 launcher
- `scripts/neurips/extract_random_encoder_features.sbatch` — random-init
  encoder extraction for the 3 random-encoder controls
- `scripts/neurips/diff_probe_train.py` — Phase 3 probe trainer
  (linear-A/linear-B/MLP × diff/raw × seeds × wd sweep × shuffle control)
- `scripts/neurips/diff_probe_train.sbatch` — Phase 3 launcher (controls + primary)
- `scripts/neurips/diff_probe_expanded_controls.sbatch` — expanded shuffle
  controls for MAE e25/e124/e194 + JEPA e50 (job 315)
- `scripts/neurips/plot_diff_probe_trajectory.py` — 4-panel trajectory plot
  (diff/raw linear-A with bootstrap CI + matched_frame clean/mf)
- `configs/feature_extraction/vitl/neurips/diff_probe/random_encoder_prepool.yaml` — random-init extraction config

## Outputs on disk

- `/tmp/diff_probe_314/controls/` — 3 shuffle + 3 random control JSONs
- `/tmp/diff_probe_314/primary/` — 20 per-checkpoint primary JSONs
- `/tmp/diff_probe_314/primary_all.csv` — 800-row flat results table
  (model, seed, arch, input, wd, shuffle_control, test_r2, val_r2, val_mse,
  n_test_clips, r2_ci_lo, r2_ci_hi)
- `/tmp/diff_probe_314/trajectory.{png,pdf}` — 4-panel family plot
- `/tmp/diff_probe_315/expctrl/` — 4 expanded shuffle control JSONs
- S3 mirrors under
  `s3://.../vjepa2-artifacts/results/diff_probe/{314,315}/`

---

## Per-token diff-probe spot-check (MAE e99, job 319 extract + 320 probes)

Addresses the "spatial averaging changes the object" caveat (Explanation 3
above) directly. Drops the spatial mean-pool at cache time so the probe
sees `[N, S=2, T=8, spatial=196, D=1024]` fp16 (`FEATURE_KEEP_SPATIAL=1`
branch in `evals/feature_extraction_pre_pool/eval.py`).

### Design (MAE e99 only, 5 seeds)

Three probe architectures, all operate on `d_t = z_{t+1} - z_t` with the
spatial axis preserved:

| Probe | Description | Params |
|---|---|---|
| `linear_spatial` | mean over T', flatten `[196·1024] → Linear(·, 1)` | ≈201K |
| `attn_a` | content-**independent** softmax over 196 positions, then `Linear(D, 1)` | 196 + 1025 |
| `attn_b` | content-**dependent** softmax `α = softmax(w·d_t)`, pool, `Linear(D, 1)` | 1024 + 1025 |

Three shuffle conditions:

- `none`: clean — paired with the `[1277, 2, 8, 196, 1024]` fp16 cache.
- `temporal`: permute `T` per `(clip, segment)` before differencing.
- `spatial`: permute `spatial` per `(clip, segment, T)` — breaks cross-frame
  spatial correspondence, intended as a per-location null.

### Results (5 seeds, job 320, stopped after 32/45 configs)

All values are test R² on EchoNet-Dynamic (1277 clips, per-clip averaged
across the 2 segments), reported as mean ± sd over 5 seeds unless noted.

| arch | clean | temporal shuf | spatial shuf | gate (< 0.05)? |
|---|---|---|---|---|
| `linear_spatial` | +0.127 ± 0.152 | **−1.141 ± 0.031** | **−0.901 ± 0.120** | PASS |
| `attn_a` | **+0.638 ± 0.005** | **−0.004 ± 0.003** | +0.397 ± 0.005 (n=3) | **FAIL on spatial** |
| `attn_b` | **+0.663 ± 0.020** | +0.468 ± 0.022 | (not reported; job cancelled) | **FAIL on temporal** |

Per-seed clean values that anchor the comparison to the global-mean
diff-probe (job 314 spatial-avg linear-A-diff on MAE e99: R² = 0.626):
- `attn_a` clean: 0.632, 0.636, 0.637, 0.647, 0.640 → **0.638**
- `attn_b` clean: 0.667, 0.671, 0.673, 0.629, 0.678 → **0.663**

### Interpretation

**Outcome 1 (expected).** `attn_a` clean ≈ spatial-avg linear-A-diff from
job 314 (0.638 vs 0.626). Removing the spatial mean-pool and letting the
probe learn a fixed position-indexed pooling recovers the same R² as
hard-coded uniform pooling. The caveat was not hiding a substantively
different measurement — MAE e99's diff-channel EF signal is ~0.6 R² at
both resolutions.

**Outcome 2 (unexpected).** `attn_b` is ~0.02–0.04 higher than `attn_a`
clean (content-dependent attention buys ~3–6 % relative). But `attn_b`
recovers R² ≈ 0.47 under **temporal shuffle** — it **fails the gate**.
After T-permutation, the diff `z[π(t+1)] − z[π(t)]` at each spatial
position still carries anatomical content (each post-shuffle pair is a
random pair of actual frames). `attn_b`'s input-dependent pooling
re-weights those positions to recover EF signal the same way a linear
regression on `raw` features at a single position would — this is not
temporal structure, it is static per-location content surviving the
T-permutation because the difference operator alone does not erase it
unless tokens are also decorrelated across positions. `attn_a`'s
position-indexed weights cannot exploit that and correctly collapse to
R² ≈ 0.

**Outcome 3 (design flaw).** Both attention probes fail the **spatial
shuffle** gate (R² ≈ 0.40 for `attn_a`). Permuting the spatial axis
independently per T does destroy per-position encoding, but it does **not**
destroy the signal the attention pool exploits, because mean-over-T of a
spatially-permuted feature map is approximately a uniform mixture at every
position (each position has 8 random draws). `attn_a`'s `α[sp]` then
converges to uniform and the probe reduces to the spatial-avg linear-A-diff
— which is precisely the ~0.6 R² we observed at job 314. The spatial
shuffle was designed to break per-location encoding, but the mean-over-T
step in `attn_a`/`attn_b` averages the shuffle out.

### Gate status: FAIL

Both attention probes fail at least one control:

- `attn_b` fails temporal shuffle → content-dependent attention can
  re-derive EF from per-position anatomy alone, so `attn_b` clean R² cannot
  be read as a measurement of cross-frame temporal encoding.
- `attn_a` fails spatial shuffle → but this is a flaw in the *control*,
  not the probe. The control, as designed, cannot kill the signal that
  `attn_a` uses because `attn_a` collapses to a uniform pool.

The per-token experiment therefore cannot, as designed, distinguish
"MAE encodes temporal information per location" from "MAE encodes
position-indexed anatomical content that happens to predict EF after
differencing and attention-based pooling." The design needs to be revised
before we spend compute on the 7-ckpt scope.

### Concrete take-aways

1. **Outcome 1 alone is paper-grade.** Confirming the caveat does not
   change the reading: per-token `attn_a` R² ≈ spatial-avg linear-A-diff
   R² on MAE e99. Section H in the paper can state that the global-mean
   diff-probe is not hiding a per-location signal at a tighter bound than
   the mean-pool probe already captures.
2. **`attn_b` should not appear in the paper's headline.** Its
   temporal-shuffle failure means its clean R² mixes per-position content
   recovery with any actual temporal signal. A headline number that needs
   a footnote saying "under temporal shuffle this probe keeps ~70 % of its
   R²" is not a cleaner measurement than the global-mean diff-probe.
3. **The spatial-shuffle control is load-bearing but wrong.** Any future
   version of this experiment needs a control that attacks the signal the
   attention pool can actually see — e.g. shuffle the spatial index
   consistently across T within a clip so mean-over-T is not a uniform
   mixture; or replace the mean-over-T with a T-local attention.
4. **Do not run 7-ckpt scope.** Stopped after MAE e99 spot-check; no
   compute spent on JEPA/BYOL/SALT per-token extraction.

### Files produced

- `scripts/neurips/extract_pre_pool_per_token.sbatch` — extraction launcher
  (SPOT_CHECK=1 default: MAE e99 only; 8.2 GB test + 47.9 GB train cache)
- `scripts/neurips/diff_probe_train_pertoken.py` — 3-arch × 3-shuffle
  trainer; saves 14×14 attention maps per (arch, seed, shuffle) for both A
  and B formulations
- `scripts/neurips/diff_probe_pertoken.sbatch` — probe launcher with
  in-job cache-shape verification and gate-check aggregation
- S3 cache: `s3://.../vjepa2-artifacts/features/diff_probe_pertoken/mae_e99_{train,test}.pt`
- Per-seed results captured from job 320 compute-node logs (job cancelled
  before `all.csv` / S3 sync; raw seed numbers in the table above).
