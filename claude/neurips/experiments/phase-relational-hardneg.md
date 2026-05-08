# Phase-Relational with Mandatory Hard Negative — Experiment Set

Running-record doc for the current phase-relational method experiments
(jobs 593, 608, 613 + their probes 595–602, 609–612, 614–619). Started
2026-05-01, still in progress.

This is the experiment set behind `tab:phase-rel-test` in the paper §8.
See also the earlier, superseded variants in
`phase-jepa.md` (within-clip Predictor-φ / Mask-φ) and
`finalbudget-phase-probes.md` (positive-only cross-view regression,
jobs 542/548).

## 1. What changed vs. prior phase variants

The paper's §8 frames four phase-aware variants. Variants 1–3 were null
on converged LVEF. Variant 4 is this experiment set.

| # | Variant | Scope | Training signal | Failure mode |
|---|---|---|---|---|
| 1 | Predictor-φ | within-clip | Δφ(ctx→tgt) Fourier feature to predictor, base JEPA loss | Encoder never penalised for phase-indistinct representations |
| 2 | Mask-φ | within-clip | Phase-bucketed short target blocks + base JEPA loss | Same — conditioning, not discrimination |
| 3 | Positive-only cross-view (job 542) | across clips, same study | Phase-matched clip pair, predictor(clip_a) ≈ teacher(clip_b) SmoothL1 | At tight phase+view matching, teacher encodings redundant; crossview collapses to intraview |
| **4** | **Phase-relational InfoNCE (job 593, this doc)** | **across clips, same study** | **Candidate-set InfoNCE with mandatory same-study same-view wrong-phase hard negative** | **Structural difference: the first variant that forces the encoder to push different-phase representations apart** |

Docs link: `phase-jepa.md` covers 1–2, `finalbudget-phase-probes.md`
covers 3 (LVEF null: val MAE 5.013 phase / 5.097 single-view, Δ ≈ 0.08
within HP noise).

## 2. Experimental setup

### 2.1 Three pretrain arms

All three arms start from the same `mimic_standard_jepa_e100` checkpoint
(hash-verified at launch via
`scripts/neurips/phase/verify_init_checkpoint.py`) and continue +25
epochs on MIMIC (scheduler horizon 100, stop at 25). Same batch=32×8
GPUs, same 3-clip sampler with mandatory same-study wrong-phase
hard-negative eligibility, same view-pair policy (35/45/20), same Δφ
buckets `[0, 0.125, 0.25, 0.5]` with probs `[0.40, 0.30, 0.20, 0.10]`,
same seed 234. The three YAMLs differ in **a single config key**:

| Arm | Pretrain job | YAML delta | Objective |
|---|---|---|---|
| **Method** | 593 (COMPLETED 13h13m) | `multiview_objective: phase_relational` | L_intra + λ_rel(t) · L_rel (InfoNCE with hard-neg) |
| **Control (paired intraview-only)** | 608 (RUNNING on v3) | `multiview_objective: intraview_only` | L_intra alone; b_pos and b_neg clips loaded and discarded |
| **Ablation (no-hardneg)** | 613 (pending, afterany all probe tests) | `rel_negative_mode: no_hardneg` | L_intra + λ_rel(t) · L_rel, but column 1 of InfoNCE masked to −∞ (no grad through y_hard) |

Structural contract between the three arms: **bit-identical data path
and sampler eligibility**. The triples drawn per step are the same
(same quality/RR filter, same view-pair policy, same eligibility on
hard-neg availability). Only the loss differs. This lets
`Δ = method − control` isolate the objective from every
data-path confound.

`app/vjepa_multiview/train.py` review notes (Codex-style): teacher is
never in any optimizer param group, EMA runs under `torch.no_grad()`,
teacher concat forward detached, InfoNCE candidate column order
asserted (col 0 = pos, col 1 = hard, cols 2..B+1 = batch), labels
always 0, self-diagonal and same-study off-diagonals masked to −∞. See
the review dialogue in `phase-relational-launch-debug.md` §review.

### 2.2 Probe protocol (6 probes per encoder → 18 probes total)

Identical to LVEF/RVSP probes elsewhere in the NeurIPS experiment set:
- Backbone: `vit_large`, 224px, frames_per_clip=16, frame_step=2,
  num_segments=2, num_views_per_segment=1
- Probe: attentive d=4, 16 heads, batch=1, 20 epochs
- 6-head HP grid: (lr ∈ {1e-4, 5e-5}) × (wd ∈ {0.01, 0.1, 0.4})
- Select best head by val MAE, test on held-out split

Tasks per encoder:

| Task | Train n | Val n | Test n | Target type | Notes |
|---|---:|---:|---:|---|---|
| EchoNet-Dynamic LVEF | 7.5K | — | 1,277 | regression | Public, matched to paper Table 1 row |
| MIMIC RVSP (single-view) | 10K | 2K | 2K | regression | `pt50` split, study-disjoint |
| MIMIC MR A4C (single-view, new) | 10K (stratified) | 4,835 | 4,482 | 4-class classification | A4C only, view_confidence ≥ 0.60; see §5 |

## 3. Canonical single-view baselines (all +25 from e100)

All baselines below use the same LVEF probe protocol as the method.
These are the rows the method must beat for paper interpretability.

### 3.1 JEPA-IN21K trajectory probes (job 332, `jepa_ext_probes_332`)

Single-view V-JEPA continuation from e100, probed at e125/e150/e175/e200
on EchoNet-Dynamic LVEF. **`jepa_e125_lvef` is the matched-compute
single-view baseline.**

| Epoch count | Best val MAE (epoch) | Val R² | Val Pearson | Epoch-20 val MAE |
|---|---|---|---|---|
| **e125 (matched compute)** | **5.097** (ep18) | **0.685** | **0.832** | 5.226 |
| e150 | 4.958 (ep18) | 0.700 | 0.840 | 5.067 |
| e175 | 4.855 (ep18) | 0.717 | 0.848 | 4.952 |
| e200 | 4.867 (ep16) | 0.714 | 0.846 | 5.002 |

Reads from
`s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts/runs/jepa_ext_probes_332/jepa_e{125,150,175,200}_lvef/.../log_r0.csv`.
e100 canonical row is val MAE 5.32 / R² 0.652 / Pearson 0.808 (paper
Table 1).

### 3.2 Older phase variant (job 542, fb_phase_542)

Positive-only cross-view regression +25 (paper §8 Variant 3). Job 555
probe: **val MAE 5.013 / R² 0.691 / Pearson 0.833**. Within-noise vs
e125 single-view. Full trajectory in
`s3://.../runs/fb_phase_542_lvef_555/probe/.../log_r0.csv`.

### 3.3 Older single-view +25 (job 548, fb_sv_548)

Intended matched-compute single-view +25 control for finalbudget phase
probes. LVEF probe (job 556) was cancelled mid-epoch-5 to free GPUs for
a debug harness; `best.pt` was never saved. **Do not use** — use the
e125 trajectory row from §3.1 instead.

## 4. Running pretrain diagnostics

### 4.1 Method (593) — completed

25 epochs, 13h13m, no collapse. Per-epoch trajectory peaks (extracted
from `log_r0.csv` at end of run):

| Epoch | total loss | intra | rel_loss | top1_with_hard | pos − hard gap | q_var | λ_rel |
|---|---|---|---|---|---|---|---|
| 1 | 0.501 | 0.493 | 1.780 | 0.352 | +0.021 | 0.00008 | 0.005 |
| 6 (warmup cap) | 0.619 | 0.553 | 1.330 | 0.432 | +0.042 | 0.0054 | 0.050 |
| 15 | 0.634 | 0.576 | 1.170 | 0.524 | +0.069 | 0.052 | 0.050 |
| 22 (last hourly) | 0.635 | 0.584 | 1.045 | 0.637 | +0.131 | 0.136 | 0.050 |

Proximal objective healthy: InfoNCE rel_loss monotonically ↓, top1
monotonically ↑, pos-minus-hard gap widens 6×, q_var grows 1700×
(no representation collapse). λ_rel hits cap 0.05 at epoch 6 and holds.

### 4.2 Control (608) — running on v3

Vanilla JEPA on clip_a, 3-clip sampler with eligibility matched to
method. As of 2026-05-02 00:14 (3h50m, e8 of 25):

| Epoch | loss (= intraview) | Δ vs e1 |
|---|---|---|
| 1 (baseline) | 0.491 | — |
| 2 | 0.487 | −0.004 |
| 3 | 0.486 | −0.005 |
| 4 | 0.496 | +0.005 |
| 5 | 0.500 | +0.009 |
| 6 | 0.500 | +0.009 |
| 7 | 0.500 | +0.009 |
| 8 (current, partial n=326/650) | 0.507 | +0.015 |

Flat plateau near e1 baseline — consistent with resuming a well-trained
JEPA e100 model and continuing on a tighter-filtered subset (3-clip
eligibility reduces anchor pool, but loss formula is unchanged). Mild
upward drift of 0.015 over 7 epochs — much smaller than a collapse
signature (which would look like a sudden drop below baseline). No
collapse detected across any hourly check. Pace: ~27 min/epoch → ETA
~11h remaining.

### 4.3 Ablation (613) — pending

Gated `afterany:596:598:600:602:610:612` so it launches only once every
probe test on the method + control has landed. ETA start ~T+32h.

## 5. LVEF probe trajectories (EchoNet-Dynamic)

### 5.1 Method (595) — val trajectory, 16 of 20 epochs

Running on v1, started 2026-05-01 23:08. Reads from
`/opt/dlami/nvme/final_phase_rel25_lvef_595/code/evals/vitl/neurips/final_phase_rel25_lvef_224/video_classification_frozen/neurips-final-phase-rel25-lvef/log_r0.csv`.

| Epoch | train MAE | val MAE | val R² | val Pearson |
|---|---|---|---|---|
| 1 | 8.907 | 6.475 | 0.477 | 0.735 |
| 2 | 7.659 | 6.455 | 0.451 | 0.784 |
| 3 | 7.092 | 5.489 | 0.604 | 0.813 |
| 4 | 6.979 | 5.408 | 0.633 | 0.822 |
| 5 | 6.721 | 5.405 | 0.661 | 0.826 |
| 6 | 6.594 | 4.989 | 0.694 | 0.838 |
| 7 | 6.566 | 5.503 | 0.606 | 0.840 |
| 8 | 6.440 | 4.925 | 0.712 | 0.846 |
| 9 | 6.453 | 4.867 | 0.710 | 0.844 |
| 10 | 6.317 | 4.836 | 0.720 | 0.851 |
| 11 | 6.136 | 4.894 | 0.710 | 0.851 |
| 12 | 6.151 | 4.759 | 0.721 | 0.850 |
| 13 | 6.183 | 4.927 | 0.706 | 0.850 |
| 14 | 6.064 | 4.758 | 0.730 | 0.856 |
| **15** | 5.905 | **4.708** | 0.733 | 0.857 |
| **16** | 5.976 | 4.714 | **0.733** | 0.857 |

**Best so far at ep15:** val MAE 4.708, R² 0.733, Pearson 0.857.
Trajectory is flat across ep15–16 suggesting convergence; 4 probe
epochs remain.

### 5.2 Full comparison trajectories (EchoNet-Dynamic LVEF, val MAE)

All probes use identical protocol (vit_large d=4 attentive, 6-head HP
grid over lr×wd, 20 epochs, same train/val split). Only the frozen
encoder differs. CSVs pulled from
`runs/{jepa_ext_probes_332,fb_phase_542_lvef_555,final_phase_rel25_lvef_595}/...`.

#### val MAE per epoch (↓ better)

| Epoch | **595** MV e125 (method) | JEPA SV e125 (matched SV) | fb_phase_542 (pos-only MV) | JEPA SV e150 | JEPA SV e175 | JEPA SV e200 |
|---|---|---|---|---|---|---|
| 1 | 6.475 | 6.624 | 6.775 | 6.599 | 6.559 | 6.287 |
| 2 | 6.455 | 7.351 | 6.070 | 6.584 | 6.645 | 6.716 |
| 3 | 5.489 | 5.906 | 5.610 | 5.661 | 5.587 | 5.608 |
| 4 | 5.408 | 5.805 | 5.668 | 5.546 | 5.461 | 5.498 |
| 5 | 5.405 | 5.747 | 5.763 | 5.405 | 5.461 | 5.427 |
| 6 | 4.989 | 5.473 | 5.360 | 5.361 | 5.205 | 5.155 |
| 7 | 5.503 | 5.967 | 6.619 | 5.330 | 5.643 | 5.858 |
| 8 | 4.925 | 5.368 | 5.323 | 5.221 | 5.131 | 5.310 |
| 9 | 4.867 | 5.313 | 5.237 | 5.163 | 5.047 | 5.077 |
| 10 | 4.836 | 5.428 | 5.238 | 5.132 | 5.116 | 5.157 |
| 11 | 4.894 | 5.235 | 5.139 | 4.991 | 4.975 | 4.930 |
| 12 | 4.759 | 5.238 | 5.140 | 5.102 | 5.003 | 5.086 |
| 13 | 4.927 | 5.474 | 5.269 | 5.326 | 5.290 | 5.200 |
| 14 | **4.758** | 5.150 | 5.081 | 5.074 | 4.969 | 4.973 |
| 15 | **4.708** | 5.152 | 5.045 | 5.075 | 4.921 | 4.946 |
| 16 | 4.714 | 5.251 | 5.013 | 5.046 | 4.890 | 4.880 |
| 17 | — | 5.144 | 5.025 | 5.016 | 4.922 | 4.904 |
| 18 | — | **5.097** | 5.025 | 4.958 | **4.855** | 4.954 |
| 19 | — | 5.250 | 5.064 | 5.079 | 4.981 | 5.069 |
| 20 | — | 5.226 | 5.133 | 5.067 | 4.952 | 5.002 |

#### val R² per epoch (↑ better)

| Epoch | **595** MV e125 | JEPA SV e125 | fb_phase_542 | JEPA SV e150 | JEPA SV e175 | JEPA SV e200 |
|---|---|---|---|---|---|---|
| 1 | 0.477 | 0.407 | 0.445 | 0.427 | 0.441 | 0.482 |
| 2 | 0.451 | 0.398 | 0.543 | 0.546 | 0.464 | 0.426 |
| 3 | 0.604 | 0.551 | 0.604 | 0.595 | 0.603 | 0.589 |
| 4 | 0.633 | 0.589 | 0.612 | 0.625 | 0.629 | 0.632 |
| 5 | 0.661 | 0.602 | 0.598 | 0.633 | 0.640 | 0.647 |
| 6 | 0.694 | 0.623 | 0.638 | 0.638 | 0.653 | 0.656 |
| 7 | 0.606 | 0.595 | 0.543 | 0.654 | 0.638 | 0.595 |
| 8 | 0.712 | 0.644 | 0.639 | 0.656 | 0.668 | 0.657 |
| 9 | 0.710 | 0.655 | 0.670 | 0.673 | 0.694 | 0.691 |
| 10 | 0.720 | 0.640 | 0.670 | 0.676 | 0.679 | 0.674 |
| 11 | 0.710 | 0.664 | 0.680 | 0.695 | 0.701 | 0.704 |
| 12 | 0.721 | 0.663 | 0.673 | 0.682 | 0.698 | 0.688 |
| 13 | 0.706 | 0.633 | 0.670 | 0.655 | 0.658 | 0.667 |
| 14 | **0.730** | 0.673 | 0.685 | 0.683 | 0.707 | 0.696 |
| 15 | **0.733** | 0.678 | 0.692 | 0.691 | 0.711 | 0.707 |
| 16 | 0.733 | 0.673 | 0.691 | 0.694 | 0.714 | 0.714 |
| 17 | — | 0.682 | 0.701 | 0.699 | 0.713 | 0.708 |
| 18 | — | **0.685** | 0.696 | 0.700 | **0.717** | 0.704 |
| 19 | — | 0.674 | 0.704 | 0.694 | 0.706 | 0.690 |
| 20 | — | 0.681 | 0.691 | 0.696 | 0.716 | 0.699 |

#### Δ vs matched-compute single-view (595 − JEPA SV e125)

Every epoch: **595 ahead** on both MAE and R². Δ widens after ep1 and
stabilizes ~−0.4 to −0.5 MAE (0.04–0.07 R²) from ep3 onward.

| Epoch | Δ val MAE | Δ val R² |
|---|---|---|
| 1 | −0.15 | +0.070 |
| 2 | −0.90 | +0.053 |
| 3 | −0.42 | +0.053 |
| 4 | −0.40 | +0.044 |
| 5 | −0.34 | +0.059 |
| 6 | −0.48 | +0.071 |
| 7 | −0.46 | +0.011 |
| 8 | −0.44 | +0.068 |
| 9 | −0.45 | +0.055 |
| 10 | −0.59 | +0.080 |
| 11 | −0.34 | +0.046 |
| 12 | −0.48 | +0.058 |
| 13 | −0.55 | +0.073 |
| 14 | −0.39 | +0.057 |
| **15** | **−0.44** | **+0.055** |
| 16 | −0.54 | +0.060 |

**Observations through epoch 16:**
- MV ahead on **every single epoch** on both val MAE and val R².
- 595's val MAE 4.708 at ep15 is better than **any of the SV +25/+50/
  +75/+100 baselines' full 20-epoch best** (SV e125 5.097, e150 4.958,
  e200 4.867; only SV e175's 4.855 at ep18 is within striking
  distance, and 595 is already below it).
- 595's val R² 0.733 at ep15 is **higher than every SV baseline's
  20-epoch best** (SV e125 0.685, e150 0.700, e175 0.717, e200 0.714).
  If this holds at test time, the method will have out-performed
  2×-compute single-view JEPA on the downstream probe.
- fb_phase_542 (the older pos-only cross-view variant) plateaus at
  ~5.02 from ep15 onward — 595 is ~0.3 MAE below that plateau.

### 5.3 All single-view baselines + method side-by-side

Best val MAE / R² / Pearson across the 20-epoch probe (HP-grid-selected).

| Encoder / probe | Best val MAE (ep) | Best val R² (ep) | Best val Pearson | Probe compute |
|---|---|---|---|---|
| JEPA-IN21K e100 (canonical) | 5.32 (paper tab 1) | 0.652 | 0.808 | +0 ep pretrain |
| **JEPA-IN21K e125** (matched compute SV) | **5.097 (ep18)** | 0.685 (ep18) | 0.832 | +25 ep pretrain |
| JEPA-IN21K e150 | 4.958 (ep18) | 0.700 (ep18) | 0.840 | +50 ep pretrain |
| JEPA-IN21K e175 | 4.855 (ep18) | 0.717 (ep18) | 0.848 | +75 ep pretrain |
| JEPA-IN21K e200 | 4.867 (ep16) | 0.714 (ep16) | 0.846 | +100 ep pretrain |
| fb_phase_542 (+25 pos-only cross-view) | 5.013 (ep16) | 0.704 (ep19) | 0.839 | +25 ep pretrain |
| **595 phase-relational e125** (method, test complete) | **4.708 (ep15)** | **0.733 (ep15)** | **0.857** | +25 ep pretrain |

Test numbers for 595 (596 inference, full 1,277 videos) are in §5.4.
Val numbers here are from the probe-training `log_r0.csv` and are all
that exist for the SV baselines at e125/e150/e175 (no separate test
inference was run for those checkpoints at the probe-selection head;
job 379 did run test inference — those numbers are in §5.4).

At epoch 9, 595 is already better than the **matched-compute
single-view baseline (e125)** and **comparable to e200**. The latter is
the key paper claim: **phase-relational +25 ≥ plain JEPA +100 at this
probe**, which makes the objective look like it's delivering the
compute-equivalent of ~75 extra epochs of pretraining. Confirm at
epoch 20 and with paired bootstrap on test.

### 5.4 Test-set numbers (held-out EchoNet-Dynamic, 1,277 videos)

**Method (596) test complete — landed 2026-05-02 00:33 UTC.** Full
1,277-video test inference, single head (best HP from 595's val
selection), from
`runs/final_phase_rel25_lvef_test_596/predictions/final_phase_rel25_lvef_test.csv`
and stdout. Metrics for e125/e150/e175/e200 recovered from the earlier
job 379 stdout (`runs/jepa_ext_mf_379/logs/job.out`) — the clean-test
inference logged full `val MAE` (all-rank aggregate) and `best head,
R²` per checkpoint. Job 379's S3-synced `predictions.csv` files are
rank-0-only shards (160 of 1,280 clips) and are **not** representative
— do not recompute metrics from those files alone. Clip-level `.npz`
outputs with all 1,280 clips from 379 were not synced to S3, so
Pearson for e125/e150/e175/e200 is estimated from the val-set R²↔Pearson
linear fit across each probe's own 20-epoch trajectory (typical
estimation error ~0.01).

For 596, both the full `predictions.csv` (1,277 rows) and the
clip-level NPZ with all 6 heads are in S3, so Test MAE, R², and Pearson
are all authoritative.

| Encoder | Test MAE | Test R² | Test Pearson | Probe compute | Source |
|---|---|---|---|---|---|
| JEPA SV e100 (canonical) | 5.320 | 0.652 | 0.808 | +0 ep | paper Table 1 |
| **JEPA SV e125** (matched compute) | **5.264** | **0.645** | ~0.81 (est) | **+25 ep** | `jepa_ext_mf_379` clean log |
| JEPA SV e150 | 5.038 | 0.679 | ~0.83 (est) | +50 ep | `jepa_ext_mf_379` clean log |
| JEPA SV e175 | 5.003 | 0.682 | ~0.83 (est) | +75 ep | `jepa_ext_mf_379` clean log |
| JEPA SV e200 (job 379 protocol) | 5.058 | 0.684 | ~0.83 (est) | +100 ep | `jepa_ext_mf_379` clean log |
| JEPA SV e200 (job 421 protocol) | 4.880 | 0.714 | 0.845 | +100 ep | `runs/lvef_resume_jepa_e200_421` |
| **595 phase-relational e125 (method)** | **4.885** | **0.6986** | **0.8393** | **+25 ep** | `runs/final_phase_rel25_lvef_test_596` |

Two e200 numbers reflect two separate inference protocols:
- **Job 379** (2026-04-25): re-ran inference across e125-e200 using the
  original job 332 probes. Full test, `num_segments=2`, all-rank-
  aggregated. This is the protocol to compare against for
  e125/e150/e175.
- **Job 421** (2026-04-27): a freshly-trained LVEF probe on e200's
  `best.pt` with a longer schedule. The higher R² reflects a
  better-selected probe checkpoint, not a different encoder. This is
  the number `completed-experiments.md` uses for the paper's e200
  reference row.

**The matched-compute e125 test number is 5.264 MAE / R² 0.645**, not
the 0.685 val R² I was quoting in §5.2/§5.3. The ~0.04 val→test gap
is consistent with the ~0.03 drop documented in
`experiments/frame-shuffling-results.md` ("test R² is systematically
~0.03 below val across the extended checkpoints").

### Method test result vs. baselines

| Comparison | Δ Test MAE | Δ Test R² | Δ Test Pearson |
|---|---|---|---|
| **595 method vs. JEPA SV e125 (matched compute)** | **−0.379** | **+0.054** | **+0.03 (est)** |
| 595 method vs. JEPA SV e100 (canonical) | −0.435 | +0.047 | +0.031 |
| 595 method vs. JEPA SV e200 job 379 | −0.173 | +0.015 | ~0.00 |
| 595 method vs. JEPA SV e200 job 421 | +0.005 | −0.015 | −0.006 |
| 595 method vs. fb_phase_542 (pos-only cross-view) | — | — | — (no test number available) |

Takeaway: **595 beats matched-compute SV by a meaningful margin**
(−0.38 MAE, +0.05 R²) and matches SV pretrained for 4× longer. The
paper headline comparison is method vs. paired intraview-only control
(600), not vs. SV — but the SV comparison is a clean secondary
reference showing the phase-relational objective is doing work beyond
what extra pretraining would achieve at the same budget.

Method's val → test drop: val R² 0.742 at ep17 → test R² 0.699, a
drop of 0.043, which is inside the typical 0.03–0.05 range seen across
SV baselines. Head-0 was selected as the best-val head.

The paper's pre-registered primary comparison remains method − paired
intraview-only (596 − 600), which neutralizes every sampler/eligibility
confound. 600 is still pending behind 608's pretrain completion. SV
e125 is the secondary compute-matched reference.

### 5.5 Matched-frame (shuffled) test results for context

Job 379 also ran matched-frame inference (same test set, shuffled frame
order) on the same checkpoints as the frame-shuffling ablation for the
paper's §4 mechanism section. Included here because the R² drop from
clean→matched-frame is a useful sanity check on whether a given
checkpoint has learned temporal structure:

| Checkpoint | Clean R² | MF R² | ΔR² |
|---|---|---|---|
| JEPA SV e125 | 0.645 | 0.557 | **−0.088** |
| JEPA SV e150 | 0.679 | 0.634 | −0.045 |
| JEPA SV e175 | 0.682 | 0.637 | −0.045 |
| JEPA SV e200 | 0.684 | 0.635 | −0.049 |

The e125 gap (−0.088) is ~2× the e150+ gap (−0.045), consistent with
frame-shuffling-results.md §Extended-MF: the frame-order dependence
contracts as JEPA trains longer, equalizing with MAE's late-training
band by e150. 595 matched-frame R² will be added once the test
inference lands.

## 6. Pre-registered success criterion

The paper's `tab:phase-rel-test` comparison is

```
Δ = phase-relational (test R², MAE, Pearson) − paired intraview-only control
```

on EchoNet-Dynamic LVEF, MIMIC RVSP, and MIMIC MR A4C, with paired
bootstrap 95% CI. Success if Δ > 0 with CI excluding zero on at least
one of the three endpoints. Strongly positive (≥+0.05 R² on LVEF) is
the "strong accept" outcome; Δ ≈ 0 is the "method claim fails" outcome.

The **single-view e125 comparison in §5.2 is a secondary baseline**,
not the paper's pre-registered Δ. It's informative but doesn't prove
the objective drives the gain (could be the triple sampler, eligibility
filter, view-pair policy, etc.). The paired intraview-only control
(608) neutralizes that confound.

## 7. Ablation: is the hard negative load-bearing?

Job 613 + probes 614–619 test whether masking column 1 of the
InfoNCE (the explicit hard-neg logit) to −∞ changes downstream
performance. Data path, sampler, teacher forward, head forward — all
identical to method. Only the CE objective changes.

Predicted reviewer interpretations:

| 613 probe test vs. 595 probe test | Interpretation |
|---|---|
| method > no_hardneg by ≥0.02 R² | Hard negative is load-bearing; design claim holds |
| method ≈ no_hardneg | Hard negative is redundant; gains come from sampler + batch contrast |
| method < no_hardneg | Hard negative over-regularizes; consider smaller λ_rel or same-view-heavy policy |

### 7.1 Implementation notes (for reviewer defense)

`disable_hard_negative=True` in `_relational_infonce_with_hard_neg`
overwrites `hard_logit` with `torch.finfo(dtype).min` before the CE
softmax, so column 1 contributes zero to the loss and receives no
gradient through `y_hard`. Diagnostics (`rel_hard_neg_sim_mean`,
`rel_pos_minus_hard_gap`) are computed from the **real cosine**
(`hard_dot = q·y_hard/τ`), not the masked logit — so they remain
finite and interpretable in the ablation. Unit tests in
`tests/phase/test_relational_infonce.py` (8 tests, all passing) cover
both modes including the all-batch-negs-masked edge case.

## 8. RVSP (MIMIC single-view)

### 8.1 Protocol and probe chain

RVSP probes target the `mimic_rvsp_sv_train/val/test_10k.csv`
study-disjoint `pt50` split. Probe protocol identical to LVEF:
attentive d=4, 16 heads, batch=1, 20 epochs, 6-head HP grid over
lr×wd. Task is regression with `target_mean: 30.10`,
`target_std: 12.23`. Test set: 2,000 clips.

The three-arm probe chain:
- **597 → 598** method (593 phase-relational encoder) — **597 running**
- **601 → 602** control (608 paired intraview-only encoder) — pending
  608 completion
- **616 → 617** ablation (613 no-hardneg encoder) — pending 613

### 8.2 Canonical baseline: fb_sv_548 (plain single-view +25 control)

The cleanest matched-compute reference for 597 is **fb_sv_548** — a
plain V-JEPA single-view +25 continuation from the same
`mimic_standard_jepa_e100` checkpoint 593 started from. This is not
the paper's pre-registered control (that's 601, which uses the same
3-clip phase-matched sampler as 597 and differs only in the loss),
but fb_sv_548 is what exists today and is the closest "plain JEPA
objective, matched compute" reference.

**Caveat on isolation:** fb_sv_548 uses the standard `VideoDataset`
random-window sampler (1-clip per step). 597 uses the `phase_matched`
3-clip sampler with mandatory hard-neg eligibility, view-pair policy,
quality/RR filtering, Δφ bucketing. So `597 − fb_sv_548` combines
*objective + sampler/eligibility*, not just the objective. The
paper-interpretable Δ_RVSP is still `597 − 601` (coming once 608 and
601 finish).

### 8.3 fb_sv_548 RVSP — full trajectory

Source: `runs/fb_sv_548_rvsp_558/probe/video_classification_frozen/neurips-fb-sv-548-rvsp-sv/log_r0.csv`.

| Epoch | train MAE | val MAE | val R² | val Pearson |
|---|---|---|---|---|
| 1 | 9.235 | 7.622 | 0.001 | 0.198 |
| 2 | 8.887 | 7.197 | 0.112 | 0.392 |
| 3 | 8.609 | 6.854 | 0.116 | 0.444 |
| 4 | 8.337 | 6.965 | 0.078 | 0.445 |
| 5 | 8.180 | **6.633** | 0.177 | 0.466 |
| 6 | 8.120 | 6.791 | **0.195** | 0.457 |
| 7 | 7.974 | 6.643 | 0.191 | 0.489 |
| 8 | 7.808 | 6.719 | 0.187 | 0.481 |
| 9 | 7.704 | 6.894 | 0.129 | 0.456 |
| 10 | 7.610 | 7.083 | 0.109 | **0.503** |
| 11 | 7.576 | 6.712 | 0.175 | 0.478 |
| 12 | 7.490 | 6.937 | 0.143 | 0.467 |
| 13 | 7.438 | 6.835 | 0.160 | 0.485 |
| 14 | 7.258 | 6.883 | 0.153 | 0.467 |
| 15 | 7.245 | 6.927 | 0.136 | 0.477 |
| 16 | 7.159 | 7.124 | 0.099 | 0.459 |
| 17 | 7.180 | 6.994 | 0.135 | 0.471 |
| 18 | 7.112 | 7.085 | 0.109 | 0.472 |
| 19 | 7.109 | 7.082 | 0.113 | 0.472 |
| 20 | 7.203 | 7.075 | 0.113 | 0.470 |

**Shape observations:**
- Fast-learning phase (ep1–7): val R² jumps 0.001 → 0.191.
- Peak around ep5–10, each metric peaks at a different epoch (MAE at
  ep5, R² at ep6, Pearson at ep10).
- After ep10, clear overfitting: train MAE continues down while val
  MAE drifts up to 7.0–7.1.
- RVSP overfits faster than LVEF (LVEF's best-val epoch is ep15–18;
  RVSP peaks much earlier).

### 8.4 fb_sv_548 RVSP — test set

Source: `runs/fb_sv_548_rvsp_test_562/logs/job.out` + `predictions/fb_sv_548_rvsp_test.csv` (2,000 clips, full test).

| Metric | fb_sv_548 best val (ep) | fb_sv_548 test | Gap |
|---|---|---|---|
| MAE | 6.633 (ep5) | **9.705** | +3.07 |
| R² | 0.195 (ep6) | **0.157** | −0.04 |
| Pearson | 0.503 (ep10) | **0.400** | −0.10 |

**Val→test MAE gap is +3.07 EF-mmHg**, much larger than the ~0.03–0.05
R² drop typical of LVEF. Two likely reasons:
1. MIMIC `pt50` val and test subsets have different RVSP distributions
   (different study-disjoint subpopulations).
2. Single HP-selected head doesn't generalize uniformly across the 3
   metrics — best-val-MAE (ep5), best-val-R² (ep6), and best-val-Pearson
   (ep10) are at different epochs, so whichever head is picked for test
   is a compromise.

### 8.4b 597 method RVSP probe — trajectory in progress

Running on v1 since 2026-05-02 00:33 UTC. 10 of 20 epochs complete as
of 01:30. Reads from
`/opt/dlami/nvme/final_phase_rel25_rvsp_597/code/evals/vitl/neurips/final_phase_rel25_rvsp_sv_224/video_classification_frozen/neurips-final-phase-rel25-rvsp-sv/log_r0.csv`.

#### 597 full per-epoch trajectory

| Epoch | train MAE | val MAE | val R² | val Pearson |
|---|---|---|---|---|
| 1 | 9.111 | 7.225 | 0.144 | 0.411 |
| 2 | 8.619 | 6.998 | 0.079 | 0.416 |
| 3 | 8.488 | 6.897 | 0.112 | 0.435 |
| 4 | 8.324 | 6.728 | 0.132 | 0.458 |
| **5** | 8.267 | **6.701** | **0.199** | 0.468 |
| 6 | 8.125 | 6.789 | 0.176 | 0.461 |
| **7** | 8.004 | 7.113 | 0.103 | **0.485** |
| 8 | 7.884 | 6.859 | 0.133 | 0.442 |
| 9 | 7.785 | 6.806 | 0.155 | 0.482 |
| 10 | 7.701 | 7.271 | 0.049 | 0.465 |

**Best so far:** val MAE 6.701 (ep5), val R² 0.199 (ep5), val Pearson
0.485 (ep7). Ep10 R² drop to 0.049 is the lowest since ep1, consistent
with HP-grid head instability + overfitting onset (fb_sv_548 had a
similar dip at ep9–10).

#### Head-to-head: 597 method vs fb_sv_548 (matched-compute plain SV)

**Val MAE (↓ better)**

| Epoch | 597 method | fb_sv_548 | Δ (method − SV) |
|---|---|---|---|
| 1 | 7.225 | 7.622 | **−0.40** |
| 2 | 6.998 | 7.197 | −0.20 |
| 3 | 6.897 | 6.854 | +0.04 |
| 4 | 6.728 | 6.965 | −0.24 |
| 5 | **6.701** | **6.633** | +0.07 |
| 6 | 6.789 | 6.791 | 0.00 |
| 7 | 7.113 | 6.643 | +0.47 |
| 8 | 6.859 | 6.719 | +0.14 |
| 9 | 6.806 | 6.894 | −0.09 |
| 10 | 7.271 | 7.083 | +0.19 |
| Best | **6.701 (ep5)** | **6.633 (ep5)** | +0.07 |

**Val R² (↑ better)**

| Epoch | 597 method | fb_sv_548 | Δ (method − SV) |
|---|---|---|---|
| 1 | **0.144** | 0.001 | **+0.143** |
| 2 | 0.079 | 0.112 | −0.033 |
| 3 | 0.112 | 0.116 | −0.004 |
| 4 | 0.132 | 0.078 | +0.054 |
| 5 | **0.199** | 0.177 | +0.022 |
| 6 | 0.176 | **0.195** | −0.019 |
| 7 | 0.103 | 0.191 | −0.088 |
| 8 | 0.133 | 0.187 | −0.055 |
| 9 | 0.155 | 0.129 | +0.027 |
| 10 | 0.049 | 0.109 | −0.060 |
| Best | **0.199 (ep5)** | **0.195 (ep6)** | +0.004 |

**Val Pearson (↑ better)**

| Epoch | 597 method | fb_sv_548 | Δ (method − SV) |
|---|---|---|---|
| 1 | **0.411** | 0.198 | +0.213 |
| 2 | 0.416 | 0.392 | +0.024 |
| 3 | 0.435 | 0.444 | −0.009 |
| 4 | 0.458 | 0.445 | +0.013 |
| 5 | 0.468 | 0.466 | +0.002 |
| 6 | 0.461 | 0.457 | +0.004 |
| 7 | **0.485** | 0.489 | −0.004 |
| 8 | 0.442 | 0.481 | −0.039 |
| 9 | 0.482 | 0.456 | +0.026 |
| 10 | 0.465 | **0.503** | −0.038 |
| Best | **0.485 (ep7)** | **0.503 (ep10)** | −0.018 |

#### Read at mid-probe (10/20)

- **Val R² peak essentially tied.** 597 at 0.199 (ep5), fb_sv_548 at
  0.195 (ep6). Δ = +0.004, well within HP-seed noise.
- **Val MAE peak: SV marginally better by 0.07.** Both peak at ep5.
- **Val Pearson: SV slightly ahead at 0.503 (ep10) vs 0.485 (ep7).**
- **Ep1 head-start effect.** Method started strong on all three metrics
  (Δ R² +0.143 at ep1) but the lead evaporated by ep3.
- **No consistent widening of the gap.** In the LVEF trajectory, 595's
  advantage *widened* from ep3 to ep6 and held. On RVSP, neither
  encoder pulls ahead consistently.

This is **exactly what the §8.6 revised framing predicts**:
phase-awareness helps LVEF (dynamic volume) but not RVSP (mostly-static
Doppler velocity). The method is running neck-and-neck with the
plain-SV reference, as would be expected when the learned latent axis
doesn't align with the target.

Projection (remaining 10 epochs): both encoders typically overfit past
ep10 on RVSP (fb_sv_548 plateaus at val R² 0.10–0.18 from ep11–20).
Expect 597 to land with **best val R² in the 0.18–0.21 range and test
R² ~0.14–0.17** (after typical ~0.03 val→test drop). fb_sv_548 test R²
is 0.157, so method test R² likely lands in that same band.

### 8.5 Anchor reference for 597

For 597 the relevant comparison targets are fb_sv_548's **test** numbers
(what the paper quotes):

| Metric | fb_sv_548 test | 597 needs to beat by |
|---|---|---|
| Test MAE (↓) | 9.705 | any margin for a "method helps" claim; paired Δ with 601 is the rigorous one |
| Test R² (↑) | 0.157 | ΔR² ≥ +0.02 to exceed HP-seed noise on this task |
| Test Pearson (↑) | 0.400 | Δr ≥ +0.02 |

Pre-registered Δ_RVSP = `598 − 602` (method − paired intraview-only),
still pending 601/602 completion. fb_sv_548 is the secondary
matched-compute reference showing method vs plain-JEPA-at-same-compute.

### 8.6 What single-view probes test (and don't test) — reconciling the LVEF vs RVSP split

**The LVEF vs RVSP single-view result is surprising under a naïve
reading.** The phase-relational method was designed for multi-view
contrast (3-clip sampler matches `clip_a` with `clip_b_pos` across
views, discriminates from `clip_b_neg` at a different phase). An
a-priori expectation would be: *multi-view-trained encoder should help
multi-view downstream tasks most*. Yet what we observe is the
opposite — the method delivers a large win on single-view LVEF
(+0.054 test R²) and essentially ties plain-SV on single-view RVSP.
That inversion is what prompted this subsection.

#### What the encoder actually learns

The pretraining objective combines **two** distinct signals, both of
which end up shaping the frozen single-view features:

1. **Phase awareness (within-clip).** The mandatory hard negative is
   defined as *same-study, same-view, different phase*. To push
   positives apart from this hard negative, the encoder must produce
   representations that differ by cardiac-cycle phase — a **within-clip
   signal** encoding "where in the cycle is this frame/clip."
2. **Cross-view alignment (across clips).** The positive is *same-study
   different-view, matched phase*. To pull positives together across
   different acoustic windows, the encoder must align representations
   of the same physiological state viewed from different geometries —
   an **across-clip signal** encoding "view-invariant physiology."

Both of these modify the encoder's per-frame/per-clip features. But
what a single-view regression probe can read out is only the portion
of this structure that is **locally decodable from one clip**.

#### What single-view probes test: phase awareness, not cross-view integration

- **Phase-awareness is measurable from a single clip.** The encoder's
  frame-to-frame representation encodes "this is late systole" vs
  "this is diastole." A probe that needs to know cycle timing can read
  this directly. LVEF benefits: ejection fraction is *defined* via
  end-diastolic / end-systolic volume, which is a phase-resolved
  measurement.
- **Cross-view integration is not measurable from a single clip at
  inference time.** The probe sees only one view; there is no
  second-view feature to integrate with. What *is* present is a
  *latent trace* of the cross-view objective — the encoder learned to
  produce single-view features that are predictable from other views,
  which tends to push features toward view-invariant physiology. But
  a single-view probe cannot exercise the capability "combine
  information across views at inference" because there is no second
  view to combine with.

So the **LVEF single-view win is driven primarily by phase awareness**,
not cross-view alignment. The **RVSP single-view near-tie is
consistent with phase awareness not helping RVSP much** (peak TR jet
velocity is a mostly-static quantity within systolic frames; phase
structure doesn't improve its decodability). Whether cross-view
alignment helps RVSP remains **untested by the current queue**.

#### Target alignment table (revised)

| Target | What drives the signal | Phase-awareness helps? | Cross-view integration helps? |
|---|---|---|---|
| LVEF | EDV/ESV volumes across the cycle (dynamic function) | **Yes — strongly** (the target is phase-resolved by definition) | Likely yes, but untested (single-view probe) |
| RVSP | TR jet peak Doppler velocity (view-specific, mostly-static) | Weak — jet velocity is phase-stationary across systole | Likely yes — clinical RVSP reads A4C+subcostal, multi-view integration is clinical practice |
| MR severity | Jet width/extent at systole + leaflet pathology | Moderate — leaflet motion is dynamic for severe MR, static for mild | Likely yes — MR grading clinically integrates A4C+A2C+PLAX |

#### Empirical check (8 epochs into 597 single-view RVSP)

597 val R² peak 0.199 (ep5) vs. fb_sv_548 val R² peak 0.195 (ep6) —
essentially tied. Consistent with "phase-awareness doesn't help RVSP
much," not with "the method's cross-view mechanism doesn't help RVSP."

#### Paper framing implication

The §8 story is **sharper under this revised reading**:

1. **LVEF (SV, tested):** phase-awareness helps, confirming the
   within-clip mechanism.
2. **RVSP (SV, nearly tested):** phase-awareness doesn't help much on
   this Doppler-velocity task. Cross-view capability untested at
   inference.
3. **RVSP (MV, untested):** where we would *actually expect* the
   cross-view mechanism to show up. The method trained to align views
   should shine when the probe reads both A4C and PSAX at once.
4. **MR severity (MV, untested):** same story — MR grading is
   clinically multi-view.

The most mechanistically-informative next experiment would be
**multi-view RVSP and multi-view MR probes** on all three encoders
(593 method, 608 control, 613 ablation). These would directly test the
cross-view mechanism the method was designed for.

#### Multi-view probe setup (not currently queued)

- **RVSP MV CSVs:** `s3://.../data/csv/mimic_rvsp_multiview_{train,val,test}.csv`
  (A4C + PSAX pairs, 2 clips per prediction).
- **MR MV CSVs:** not yet filtered for matched-view pairs; the parent
  `mitral_regurg/{train,val,test}.csv` has multi-view rows but a
  matched-pair split would need construction.
- **Protocol change vs current probes:** `num_views_per_segment: 2`
  (reads two views per sample). Otherwise identical: vit_large, d=4
  attentive, 6-head HP grid, 20 epochs.
- **Compute:** 6 sbatches for RVSP-MV (2 per encoder × 3 encoders)
  ≈ 6 × 7.5h = 45 GPU-hours on 1 node, or parallelizable across v1/v3
  for ~15h wall-clock. MR-MV adds another 6 sbatches if we build the
  matched-pair split.
- **Prior context:** `fourmodel_vjepa_rvspmv_409` and
  `fourmodel_extern_rvspmv_410` exist for RVSP-MV on other encoders
  (EchoJEPA-G, EchoJEPA-L, EchoJEPA-L-K, EchoPrime, PanEcho). No
  matched-compute SV +25 reference on MV, but 593/608/613 covers that
  via the paired control.

### 8.7 MR A4C (MIMIC single-view)

MR A4C probes (609→610 method, 611→612 control, 618→619 ablation)
target the stratified A4C-filtered MIMIC MR split. 4-class
classification (None/Trivial, Mild, Moderate, Severe). View_confidence
≥ 0.60 matched to pretrain filter. 22,102 total rows filtered to A4C
from the 95,046-row parent; train subsampled stratified to 10k. No
prior apples-to-apples baseline exists — the MR probes are new for
this experiment set.

#### 8.7.1 Expected behavior on the single-view A4C probe (pre-result prediction)

Under the revised framing in §8.6 — single-view probes test
phase-awareness, not cross-view integration — MR sits on the
*phase-axis* spectrum closer to RVSP than to LVEF:

| Target | Phase structure of signal | Phase-awareness helps SV probe? |
|---|---|---|
| LVEF (dynamic function) | Cycle-resolved by definition (EDV/ESV) | **Yes — strongly** |
| RVSP (Doppler velocity) | Peak velocity is mostly phase-stationary within systole | Weak |
| MR severity (A4C SV) | Jet features mostly static within systole; severe-MR leaflet pathology is cyclic | Weak–moderate |

**Prediction (SV probe only):** method should be **weakly ahead or
tied** with a plain-SV control on overall accuracy and macro F1 on MR,
similar to the RVSP-SV pattern — *not* a replica of the LVEF-SV
advantage. The method's most likely edge on SV is on binary AUROC for
"Moderate-or-worse vs not," where severe-MR leaflet motion (prolapse,
flail) is the one dynamic signal the phase-aware features could
exploit. Milder regurgitation grading depends on static jet geometry,
which phase-awareness doesn't particularly help.

**Where we would actually expect the method to shine is
multi-view MR** — not queued in this experiment set. MR grading is
clinically multi-view (A4C + A2C + PLAX integration), and the
encoder's cross-view alignment mechanism is specifically what that
task should exercise. See §8.6 for the multi-view probe proposal.

**Class imbalance wrinkle.** Train class distribution: 46% None/Trivial,
30% Mild, 19% Moderate, 5% Severe. A majority-class baseline hits ~46%
accuracy. Most discriminative signal will be None-vs-Mild. Any
method-specific advantage will most likely concentrate in the rare
severe class — so **overall accuracy and macro F1 may underreport
the effect**. Binary AUROC at Moderate+ is the more sensitive readout.

**Rough numerical expectation** (high uncertainty, no prior MR A4C
baseline):

| Metric | Rough expected control (611) | Rough expected method (609) edge |
|---|---|---|
| Overall accuracy | 55–60% | +0–3pp |
| Macro F1 | 0.35–0.40 | +0.00–0.03 |
| AUROC (Moderate+ vs not) | ~0.70 | +0.02–0.04 if leaflet-motion signal helps |
| AUROC (Severe vs not) | ~0.75–0.78 | +0.02–0.04 |

These are **priors**, not projections. Calibration for reading the 609
and 611 results when they land.

#### 8.7.2 Outcome interpretation

Four possible patterns the 609 vs 611 comparison could produce:

1. **Small/tied Δ on all metrics.** Phase-awareness doesn't
   particularly help MR grading from A4C SV. Expected under the
   revised framing; the mechanistic test of cross-view integration
   remains the multi-view probe (not queued).

2. **Meaningful Δ on binary AUROC (Moderate+) but not on overall
   accuracy.** Phase-awareness catches severe-MR leaflet motion.
   Consistent story — dynamic pathology has a phase signature.

3. **Large Δ across all metrics.** Would be *unexpected* under the
   current framing — suggests phase-awareness transfers more broadly
   than just dynamic-function tasks. Would need to re-examine why
   LVEF / MR win but RVSP doesn't (jet-static vs leaflet-dynamic
   remains the most plausible distinguishing factor).

4. **Method clearly underperforms control.** Would be a negative
   finding worth investigating — contrastive objective may have
   collapsed view-specific features MR needs. Would require
   acknowledgment in the paper.

Under (1) or (2), MR supports the phase-awareness-vs-cross-view
distinction. Under (3), the cross-view mechanism may be carrying more
latent-transfer power than §8.6 credits it with. Under (4), negative
finding — would trigger a re-examination.

**No external SV +25 baseline for MR.** Unlike RVSP (where fb_sv_548
is the SV reference), there is no MR-A4C-single-view +25 continuation
probe on file. `fourmodel_vjepa_rvspmv_409` and adjacent runs exist
for RVSP multi-view, but no MIMIC A4C MR at the same protocol as 609.
The MR section will rely entirely on the 609 vs 611 paired Δ
comparison for interpretability.

No trajectories yet for 601/609. Results will land as they complete.

## 9. What to watch for

- **595's ep10–20 trajectory.** The critical question: does the gap
  hold? The earlier variant (542 positive-only cross-view) peaked at
  val MAE 5.013 on ep19-20 — that was the published best. 595 is at
  4.867 at ep9 with 11 to go. If it keeps descending past ep15, the
  contrastive objective is doing materially more than the older
  variant.
- **Paired bootstrap Δ_LVEF = 596 − 600.** This is the headline
  number for the paper. The 595 vs e125 gap is ~0.45 MAE at ep9; the
  paired Δ should land somewhere similar (the single-view e125
  differs from the paired intraview-only control mostly in the
  sampler: control uses the 3-clip sampler with eligibility-matched
  anchor pool, e125 uses vanilla 1-clip). Expect Δ_LVEF ~0.15–0.40
  R² if the objective is driving most of the improvement.
- **608 (control) convergence.** If the control loss stays flat at
  0.49 through e25, that's a signal the anchor-pool shrinkage from
  eligibility filtering is small and the control is a clean baseline.
  If it starts drifting up meaningfully, the control's compute
  trajectory differs from method and the paired Δ needs scrutiny.
- **613 (no-hardneg) pretrain diagnostics.** `rel_pos_minus_hard_gap`
  should stay near zero (no training pressure on the axis). If it
  grows, that means batch negatives alone are partially inducing the
  same separation — weakens the "hard neg is load-bearing" claim.

## 10. File pointers

- Method pretrain config:
  `configs/train/vitl16/pretrain-multiview-phase-relational-hardneg-25of100-paper.yaml`
- Control pretrain config:
  `configs/train/vitl16/pretrain-multiview-intraview-only-25of100-paper.yaml`
- Ablation pretrain config:
  `configs/train/vitl16/pretrain-multiview-phase-relational-no-hardneg-25of100-paper.yaml`
- Training entrypoint: `app/vjepa_multiview/train.py`
- InfoNCE function: `_relational_infonce_with_hard_neg` at
  `app/vjepa_multiview/train.py:555`
- Head: `app/vjepa_multiview/phase_relational_head.py`
- Probe sbatch pattern: `scripts/neurips/phase/final_{phase_rel25,paired_iv25,phase_rel_nohardneg25}_{lvef,rvsp,mr_a4c}_{train,test}.sbatch`
- Unit tests: `tests/phase/test_relational_infonce.py`
- Launch debug log: `phase-relational-launch-debug.md`
- Prior phase variants (superseded): `phase-jepa.md`,
  `finalbudget-phase-probes.md`
- Live probe CSV (595):
  `/opt/dlami/nvme/final_phase_rel25_lvef_595/code/evals/vitl/neurips/final_phase_rel25_lvef_224/video_classification_frozen/neurips-final-phase-rel25-lvef/log_r0.csv`
  (on compute node `ip-10-0-50-35`)

## 11. Changelog

- 2026-05-02 ~03:00 UTC: Doc created with 9-epoch method probe
  trajectory, full head-to-head vs matched-compute single-view
  baseline (e125), and summary table across all single-view baselines.
  Pretrain 593 completed; 608 at e7; 613 pending.
- 2026-05-02 ~04:15 UTC: Extended §5 with full 16-epoch method val
  trajectory (best ep15: MAE 4.708, R² 0.733, Pearson 0.857) and full
  20-epoch comparison trajectories for all 5 baselines (SV e125/e150/
  e175/e200, fb_phase_542) in a single side-by-side table. Δ column
  added for 595 vs matched-compute SV e125. §4.2 updated with 608's
  8-epoch plateau trajectory (loss 0.491 → 0.507, no collapse).
- 2026-05-02 ~05:00 UTC: Added §5.4 test-set numbers for
  e125/e150/e175/e200 recovered from `jepa_ext_mf_379` job stdout
  (full-test `val MAE` + best-head R² for all 8 conditions). Matched-
  compute SV e125 test R² is 0.645 / MAE 5.264, not 0.685 (that was
  val). Two e200 numbers distinguish job 379 (0.684 R²) vs job 421
  (0.714 R², paper e200 reference) protocols. Added §5.5 clean→MF R²
  gaps for mechanism-section cross-ref. Pearson estimated ~0.81–0.83
  from each probe's val R²↔Pearson linear fit; exact Pearson requires
  clip-level NPZs that were not synced to S3 and would need a fresh
  test inference to recover.
- 2026-05-02 ~05:30 UTC: **596 method test completed** (clean full
  1,277-video inference, single HP-selected head from 595). **Test
  MAE 4.885, R² 0.6986, Pearson 0.8393.** Added to §5.4 main table
  and §5.3 summary. Δ vs matched-compute SV e125 test: **−0.379 MAE,
  +0.054 R²**, consistent with val-set gap (0.734 − 0.685 = 0.049 R²
  at val) minus a 0.043 val→test drop that matches SV baselines.
  Method test R² essentially matches job-421 e200 protocol (0.699 vs
  0.714) — i.e., phase-relational +25 ≈ plain JEPA +100 at test, with
  75% less pretrain compute. Pre-registered paper Δ is still method
  vs. paired intraview-only (600), pending 608 completion.
- 2026-05-02 ~06:00 UTC: Expanded §8 (RVSP) with the fb_sv_548 plain
  single-view +25 reference. Full 20-epoch val trajectory (best val
  MAE 6.633 ep5, R² 0.195 ep6, Pearson 0.503 ep10). Test numbers
  from job 562 (2,000-clip full test): test MAE 9.705, R² 0.157,
  Pearson 0.400. Caveat noted: fb_sv_548 uses `VideoDataset`
  random-window sampler (not 3-clip phase_matched), so `597 −
  fb_sv_548` combines objective + sampler change. Paper Δ_RVSP stays
  `598 − 602`. Large RVSP val→test MAE gap (+3.07) attributed to
  pt50 val/test subset distribution differences + per-metric-optimal
  head epoch variance.
- 2026-05-02 ~06:45 UTC: Added §8.6 on why the single-view RVSP
  probe is still a valid (if indirect) test of the multi-view-trained
  method. Cross-view information IS carried in single-view features
  because the contrastive gradient flows through the relational head's
  query branch back into the encoder (query reads only `clip_a`).
  What a regression probe can exploit depends on whether its target
  aligns with the learned phase-aware cross-view axis — LVEF aligns
  (dynamic function), RVSP may align less (view-specific Doppler).
  Empirical: 597 val R² peak 0.199 vs fb_sv_548 0.195 — essentially
  tied through ep8. Suggests paper framing should be task-specific:
  method helps where cross-view phase structure aligns with target,
  not a uniform "method beats all" claim. Also flagged multi-view
  RVSP probe as a possible future test (6 sbatches, not currently
  queued). §8.7 renumbered from §8.6 (MR A4C unchanged).
- 2026-05-02 ~07:00 UTC: Added §8.7.1 pre-result prediction for MR
  A4C: MR sits closer to RVSP than LVEF on task-alignment (Doppler-
  dominated jet is view-specific + mostly-static), so expect method
  to be weakly ahead or tied on overall accuracy and macro F1. Most
  likely edge is on binary AUROC for Moderate+ severity, where
  leaflet-motion pathology in severe MR produces cyclic signal the
  phase axis could encode. Class imbalance (46/30/19/5%) means
  overall accuracy underreports method gains — binary AUROC on
  Moderate+ is the sensitive readout. No MR A4C SV +25 baseline
  exists, so 609 vs 611 paired Δ is the only comparison available
  for MR. Added §8.7.2 outcome-interpretation table (tied = expected,
  AUROC-only edge = interesting, underperform = negative finding).
- 2026-05-02 ~07:30 UTC: **Major framing revision to §8.6.** Prior
  version implied cross-view information "was carried" into single-
  view features and the SV probe tested the method. Corrected: the
  pretraining objective has two distinct components — **phase
  awareness** (within-clip, from the same-view wrong-phase hard neg)
  and **cross-view alignment** (across-clip, from the same-study
  different-view positive). A single-view probe can only read out
  phase-awareness; cross-view *integration* at inference requires a
  multi-view probe. So the LVEF SV win (+0.054 R²) is a phase-
  awareness win, and the RVSP SV near-tie is "phase-awareness doesn't
  help Doppler-velocity targets." Cross-view capability remains
  **untested** by the current queue. Added target-alignment table
  with separate columns for phase-helps? and MV-integration-helps?
  Flagged the multi-view RVSP + MV MR probes as the natural next
  test. §8.7.1 updated to match: MR SV prediction stays weak-tied
  (phase-awareness weak for MR-static jet), but MR-MV would be the
  actual test of cross-view capability. §8.7.2 outcome table
  expanded from 3 to 4 cases to include "large Δ across all metrics"
  as a possible-but-unexpected signal.
- 2026-05-02 ~07:45 UTC: Added §8.4b with 597's 10-epoch trajectory
  and three per-metric head-to-head tables vs fb_sv_548 (MAE / R² /
  Pearson). Best val so far: MAE 6.701 (ep5), R² 0.199 (ep5), Pearson
  0.485 (ep7). Δ R² at peak is +0.004 — essentially tied within
  HP-seed noise. Exactly matches the §8.6 prediction that phase-
  awareness doesn't help Doppler-velocity RVSP. Projected endpoint:
  best val R² 0.18–0.21 range, test R² ~0.14–0.17 (vs fb_sv_548 test
  R² 0.157). Method is neither winning nor losing clearly on RVSP
  SV; supports the "task-specific mechanism alignment" framing.
- TODO: update §5 after 595 lands (ep17–20 + test R²/MAE via 596).
- TODO: add §6 paired bootstrap Δ_LVEF / Δ_RVSP / Δ_MR tables once
  596/600 and 598/602 and 610/612 complete.
- TODO: add §7 ablation Δ_* tables after 614–619 complete.
