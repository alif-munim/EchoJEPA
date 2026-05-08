# Vanilla V-JEPA 2 on Echo — What Works, What Doesn't

Companion doc to `controlled-objectives.md`. That doc covers the
**matched-compute comparison** between JEPA / BYOL / MAE / SALT on
EchoNet-Dynamic and robustness endpoints. This doc covers:

1. **What the analyses reveal about JEPA's mechanism** on echo
2. **The gaps plain JEPA leaves** (ranked by severity)
3. **What has and hasn't closed them** (MV2SV / V4 / TokenRel / MotionDelta / scaling)

Distilled from `controlled-objectives.md`, `mv2sv-privileged-multiview.md`,
`final-task-list.md`, `lvh.md`, and the full rebuttal / NeurIPS archive
under `claude/`.

---

## 1. Mechanism — how JEPA works on echo (observational picture)

### 1.1 Spatial → temporal layer hierarchy (moderate)

Cross-temporal attention analysis (`representation-analysis.md` §7):
fraction of attention flowing across temporal positions (random
baseline = 0.875, lower = more within-frame).

| Layer | JEPA e100 | BYOL e100 | MAE e99 | SALT S2 e79 |
|---|---|---|---|---|
| 0-1 | **0.57–0.60** | 0.77–0.86 | 0.86 | **0.44–0.49** |
| 2-10 | 0.82–0.87 | 0.81–0.86 | 0.82–0.87 | **0.39–0.56** |
| 11-23 | 0.87–0.88 | 0.87–0.88 | 0.87 | 0.83–0.88 |
| Overall | 0.839 | 0.855 | 0.861 | 0.672 |

**JEPA develops a mild spatial-bias in layers 0-1 only.** BYOL and MAE
show no such hierarchy. SALT develops a much stronger version (layers
0-10). The hierarchy **deepens with training** (SALT e29 vs e79).

**Why this matters for echo**: cardiac structures are
anatomically-localized in each frame (ventricular walls, valves), but
their clinically meaningful state evolves over the cycle. A model that
builds per-frame spatial features first and integrates them temporally
second matches the structure of cardiac endpoints (LVEF = change in
volume between systole and diastole). **MAE builds spatial features
throughout but never integrates temporally; BYOL mixes from the start;
JEPA lands in the middle.**

### 1.2 Moderate temporal-structure encoding (~17%)

Frame-shuffling severity gradient on LVEF (`severity-gradient.md`):

| Shuffle % | JEPA e100 R² | BYOL e100 R² | MAE e99 R² | SALT e79 R² |
|---:|---:|---:|---:|---:|
| 0 (clean) | **0.591** | 0.468 | 0.445 | 0.296 |
| 100 (fully shuffled) | 0.488 | 0.291 | 0.428 | −0.270 |
| **Relative drop** | **−17%** | **−38%** | **−4%** | **−191%** |

Four distinct "temporal-encoding regimes":

- **JEPA: gentle slope** (−17%) — uses temporal structure moderately
- **BYOL: steep slope** (−38%) — strong temporal dependence
- **MAE: invariant** (−4%) — temporal structure barely used
- **SALT: cliff** (−191%) — over-reliant on order, low ceiling

**Critical observation**: JEPA shuffled (0.488) **still beats BYOL
clean (0.468)**. JEPA's advantage is not purely about reading temporal
info better — its spatial features are also substantially better than
BYOL's.

### 1.3 Retracted mechanisms (don't cite)

**Speckle filtering** (`representation-analysis.md`): the ICML
rebuttal claim that JEPA filters speckle via EMA averaging does NOT
survive init-matching. Under e100 init-matched models, BYOL is the
**best** speckle filter (0.716), JEPA 0.848, MAE 0.885. Gap shrunk
from 23% to 4% and ranking changed. Mechanism not supported.

**Effective dimensionality** (`representation-analysis.md`): consistent
4-model RankMe pipeline gives 245 (JEPA) / 221 (BYOL) / 206 (MAE) / 203
(SALT). All in 200-245 range. **No 3× collapse.** The Goodfire MAE=63
number was not reproducible and should not be cited as "MAE
under-uses capacity."

**What survives**: temporal-structure encoding (from frame-shuffling)
and spatial→temporal layer hierarchy (from cross-temporal attention).
Both are mechanisms of predictive pretraining that MAE's reconstruction
objective and BYOL's contrastive objective don't induce.

### 1.4 Cross-modality falsification (CMR direction flip)

On cardiac MRI (ACDC cohort), JEPA's advantage **reverses**. Fast-EMA
JEPA peaks at e30-e100 on LVEF/Dx, then degrades; slow-EMA shows the
same pattern (peaks R²=0.138 at e30, collapses to 0.089 at e295 on
LVEF). MAE climbs monotonically and overtakes JEPA by e200-e800. See
`cmr-cross-modality.md`.

**Implication**: JEPA's "temporal consolidation" mechanism needs high
temporal turnover in the pretraining data. Echo has ~1 s cycles ×
hundreds of cycles per clip. MRI has slow phase-encoded reconstruction.
Without the fast cycle turnover, the mechanism erodes and MAE's
longer-horizon reconstruction objective catches up.

**Scope claim**: "JEPA > MAE on cardiac video" is specifically
an echo claim, not a general cardiac imaging claim.

---

## 2. Gaps plain JEPA leaves — ranked by severity

### Gap 1 — View generalization

**Endpoints affected**: RVSP, MR severity, AS severity, valve dynamics.

Plain JEPA's pretraining is view-agnostic (single-view intraview
masking). It's never explicitly asked to relate clips across views.
The learned representation doesn't know that A4C + A5C + PLAX of the
same patient depict a shared hemodynamic state.

**Data**:
- JEPA e100 RVSP test Pearson 0.458 / R² 0.175
- Adding pretrain compute: fb_sv_548 (JEPA + 25 ep) RVSP test R² 0.157 — *doesn't improve with compute alone*
- Adding cross-view objective: V4 phase-rel RVSP R² **drops to 0.018** — cross-view objectives **hurt** RVSP at matched compute
- TokenRel variants at e5: R² 0.07-0.08 — helps slightly over V4 but still below Base

**Status**: **not closed by any objective we've tested.** Pooled
phase-InfoNCE (V4) and token-level phase-InfoNCE (TokenRel) both
regress to mean on RVSP. MV2SV v5's pair_view / view_nce losses lose
to paired-intraview control at matched compute.

The only thing that helps RVSP is **more pretraining compute plus a
view-aware multi-view probe head** — and even then, multi-view pushes
+3.9 pp R² over best-single-view but doesn't break the aggregate 0.22
R² ceiling.

**Hypothesis for why it's hard**: RVSP is a Doppler-gated measurement
in reality. B-mode RVSP is inherently a regression-to-mean task because
the label correlates weakly with what's visible in B-mode. Most of the
variance is not encoded in the video.

### Gap 2 — Population shift

**Endpoints affected**: EchoNet-Pediatric LVEF, external cohorts (Stanford vs BIDMC), age/sex subgroups.

Plain JEPA transfers reasonably well zero-shot to pediatric
(UHN-trained probes → pediatric test: Pearson 0.705, MAE 6.957, R²
0.405) but has a clear gap to the specialist EchoJEPA-G (3.88 MAE) and
the Kinetics-annealed EchoJEPA-L-K.

**Matched-compute observation**: Base e125 *beats* V4 phase-rel on
pediatric LVEF (test MAE 4.80 vs 5.02, R² 0.621 vs 0.574). The V4
phase axis learned on adult MIMIC doesn't transfer to pediatric.

**What closes it**: **more diverse pretraining data**. EchoJEPA-L-K
(Kinetics annealing) is the strongest ViT-L on adult LVEF and
presumably will be on pediatric too when probed. EchoJEPA-G (UHN 18M)
is the clear leader on pediatric by a wide margin.

**What doesn't close it**: phase-aware objectives. V4/TokenRel trade
view generalization for phase fidelity, and population shift needs
generalization.

### Gap 3 — Local amplitude signals (TAPSE cap)

**Endpoints affected**: TAPSE, LVIDd/LVIDs, valve leaflet excursion, any per-landmark amplitude measurement.

All four variants (Base e125, V4 e25, TokenRel+Motion e5, TokenRel+Motion e25) cap at **test R² ≈ 0.25** on A4C TAPSE:

| Variant | Test R² | Test Pearson |
|---|---:|---:|
| Base e125 | 0.247 | 0.514 |
| V4 phase-rel | 0.250 | 0.519 |
| TokenRel+Motion e25 | 0.210 | 0.481 |

MotionDelta was designed specifically to target TAPSE — it did not
break the cap. None of the tested objectives encode per-landmark
amplitude tracking.

**Hypothesis**: TAPSE requires reading a specific point (TV annulus)
moving between specific phases (end-diastole and peak-systole). This
is a token-level localization + amplitude task, and **none of the
objectives we've tested have a per-landmark loss**. The phase axis V4
learns is pooled; TokenRel is token-level but in a different output
space; MotionDelta predicts same-view latent delta but doesn't enforce
any spatial localization.

**What might close it**: a **per-landmark SS3D / registration-style
objective** that explicitly tracks token trajectories. Not in scope
for the current architecture line. Alternatively, the probe head could
incorporate explicit landmark pooling — requires probe-architecture
work, not a pretrain objective change.

### Gap 4 — Reduced-EF tail (pathology extremes)

**Endpoints affected**: any probe with long-tailed label distributions.

Pathology-stratified LVEF MAE (EchoNet-Dynamic test):

| EF category | N | JEPA MAE | BYOL MAE | MAE MAE |
|---|---:|---:|---:|---:|
| Normal (≥55%) | 876 | 4.3 | 5.0 | 5.1 |
| Mildly reduced (40-54%) | 241 | 7.6 | 7.8 | 7.1 |
| **Reduced (<40%)** | 160 | **12.4** | 14.4 | 19.3 |

JEPA's reduced-EF MAE is 12.4 — half of MAE's 19.3 but still 3× the
normal-EF MAE. The tail is under-represented in training data.

**What this is not**: an objective problem. All three objectives
(JEPA/BYOL/MAE) show the same failure pattern.

**What might close it**: label-aware resampling during probe training,
loss re-weighting, or supervised anchor features (which breaks the
"frozen pretrain" assumption). These are **probe-level**, not
pretrain-level, fixes.

### Gap 5 — Phase-dominated tasks beyond LVEF

**Endpoints affected**: MR severity multi-view, AS severity, ventricular strain, tissue Doppler measurements.

JEPA's phase encoding on LVEF is the one success. On MR A4C SV, all
variants (Base, V4, MV2SV Pilot/Ctrl, TokenRel) land in a 0.7 pp
val_acc / 0.01 val_auroc cluster. Too weak a task to separate
objectives. Expected to be better on MR multi-view (pre-registered in
`phase-relational-hardneg.md`) but not yet probed.

**Hypothesis**: phase dominance only shows up on tasks where *global
cycle position* is the dominant signal (LVEF). MR severity depends on
*valve leaflet coaptation* during systole — needs both phase AND
localization. AS severity is Doppler-gated — phase is correlated with
ECG, not image content.

### Gap 6 — Specialist-model gap

**Endpoints affected**: all echo tasks where a specialist beats our generalist.

EchoJEPA-G (ViT-g, 384 px, UHN 18M, 100 ep) on LVEF: test MAE 3.88, R²
~0.78. Our e200 JEPA ViT-L (IN21K + MIMIC 525K): 4.88 MAE, 0.714 R².
Gap is real and largely **data + model-size driven**, not objective.

**What closes it**: scale (UHN 18M, ViT-g). PanEcho (specialist) and
EchoPrime (specialist) also beat our L-size generalist on clean LVEF.

**What EchoJEPA-L-K shows**: adding Kinetics annealing to an L-size
model brings val MAE to 4.45 — beating specialist PanEcho and getting
within 0.57 of EchoJEPA-G. **Data diversity (Kinetics) is a bigger
lever than pretrain compute (e100 → e200).**

### Gap 7 — Distillation / compute efficiency (NOT a JEPA gap, but a constraint)

We tried SALT (frozen-teacher distillation) as a compute-efficient
alternative. It strictly **underperforms** JEPA:
- SALT v1 e79 test R² 0.416 vs JEPA e100 R² 0.652 (−0.24 R²)
- JEPA-teacher SALT (S2 student from JEPA e100 teacher): R² 0.252 vs raw JEPA 0.650 — **−0.40 R² attributable to the distillation step alone**

**Conclusion**: co-evolution of the target encoder is load-bearing.
Cannot shortcut JEPA with frozen-teacher distillation. This closes off
one compute-efficiency path.

### Gap 8 — Robustness-vs-clean tradeoff

**Endpoints affected**: pure-segmentation tasks without noise.

On CAMUS clean segmentation, MAE actually beats JEPA (0.827 vs 0.815
Dice). **JEPA's advantage only emerges under perturbation** (severe
depth attenuation: JEPA 0.683 vs MAE 0.654 vs BYOL 0.368). If a
downstream task is run purely on clean video with no robustness
pressure, MAE can win.

**What this implies**: the value of JEPA's temporal-structure encoding
is *robustness*, not peak clean performance for every task. For
clean-only segmentation benchmarks (CAMUS without perturbations), MAE
is competitive.

---

## 3. What has and hasn't closed the gaps

Matched-compute (+25 ep) scorecard from `mv2sv-privileged-multiview.md`
and `controlled-objectives.md`:

| Task | Winner | V4 phase-rel vs Base e125 |
|---|---|---|
| LVEF (EchoNet-Dynamic adult) | **V4 phase-rel** | ✅ +0.053 R² / −0.48 MAE |
| LVEF (EchoNet-Pediatric) | **Base e125** | ❌ −0.047 R² / +0.22 MAE |
| RVSP | **SV fb_sv_548 (plain JEPA +25 ep)** | ❌ −0.090 R² vs Ctrl; −0.139 R² vs fb_sv_548 |
| MR A4C 4-class | tie (all within noise) | ≈ null |
| TAPSE | tie (≈0.25 cap for all) | ≈ +0.004 R² |
| RV function binary 10k | **Base e125** | ❌ −0.007 AUROC |

**V4 phase-rel: 1-for-6 at matched compute.** Only LVEF (adult,
EchoNet-Dynamic) is a clean win.

**MV2SV v5 (Pilot 655) vs Ctrl 658** at matched +5 ep: Pilot loses on
LVEF val R² (−0.034), RVSP test R² (−0.027), MR val_auroc (−0.011).
Pilot wins on cross-view retrieval diagnostic (top1 0.447 vs Ctrl
0.174) but that structural gain doesn't translate to downstream tasks.

**TokenRel+Motion e5 → e25**: diminishing returns. LVEF val R² +0.020
but test R² flat (0.669 → 0.667). TAPSE actually *worse* at e25 than e5
(R² 0.180 → 0.210 — modest gain but below V4/Base). Adding epochs
doesn't convert the architecture's compute-efficiency advantage into
an absolute improvement.

### What DID produce clean empirical wins over plain JEPA

1. **EchoJEPA-L-K (annealing on Kinetics)** — best ViT-L overall on
   LVEF (val MAE 4.45). Beats JEPA e200 extension. Data diversity >
   pretrain compute.

2. **JEPA e100 → e200 extension** — val MAE 5.32 → 4.88 (+0.06 R²).
   Simple compute scaling on the existing objective works, just
   diminishing.

3. **V4 phase-rel for LVEF specifically** — +5 pp R² on adult LVEF at
   matched compute. Narrow but real win, doesn't generalize to other
   endpoints.

### Net: plain JEPA is a hard-to-beat generalist

The surprise of this whole investigation is that **plain V-JEPA 2 at
e100-e200 on MIMIC 525K is stubbornly good across echo endpoints.**
Task-specific objectives (V4 phase-rel, MV2SV, TokenRel, MotionDelta)
can beat plain JEPA on the endpoint they target, but lose on every
other endpoint. The generalist hypothesis wins by aggregate.

The specific observation — "a phase-aware cross-view sampler that
nobody has a good name for beats every fancy loss at matched compute"
— is the recurring §2.4 finding (Ctrl beats Pilot, paired-iv beats MV2SV
losses). The sampling regime matters more than the objective at matched
compute.

---

## 4. What would actually close the remaining gaps

Based on what we've tested and what's failed, the remaining gaps
break into three categories by what would close them:

### 4.1 Data scale + diversity

**Gaps addressed**: pediatric population shift (Gap 2), specialist gap (Gap 6), marginal LVEF improvement.

**What to do**:
- EchoJEPA-G class: UHN 18M + ViT-g + Kinetics annealing. Already done for G; would need to replicate for L-K-G on MIMIC to isolate effects.
- Expand MIMIC beyond 525K clips if more are available.

**Risk**: standard scaling. No technical breakthrough needed, just more pretrain.

### 4.2 Task-specific probe architecture

**Gaps addressed**: reduced-EF tail (Gap 4), TAPSE local amplitude (Gap 3), phase-dominated multi-view tasks (Gap 5).

**What to do**:
- **Per-landmark pooling in the probe** instead of global attentive pool — unlocks TAPSE token-level tracking
- **Label-aware re-weighting during probe training** — unlocks reduced-EF tail
- **Multi-view probe heads** — we have this for RVSP but not every task; unlocks MR multi-view
- **Doppler-informed augmentation at probe time** — for velocity-gated tasks like RVSP, AS

**Risk**: moderate. Probe-level changes don't risk the pretrain.

### 4.3 Pretrain objective changes that target specific gaps

**Gaps addressed**: depends on objective choice. V4 for LVEF (already
done, +0.053 R²). TokenRel for phase-dominated tokens (ambiguous).

**What to do cautiously**:
- **Mixed-objective pretrain** with a small phase-rel weight to get LVEF boost without crippling RVSP/pediatric. Not yet tested at careful loss-balance sweeps.
- **View-contrastive with wrong-view negatives** (to force view discrimination) — not yet tested.
- **Frame-level local amplitude loss** for TAPSE — unshipped (local_motion_loss in MV2SV v5 raises NotImplementedError by design).

**Risk**: high. History says specialized objectives hurt other endpoints at matched compute.

### 4.4 Ensemble / dispatch

**Gaps addressed**: potentially all of them, at inference cost.

**What to do**:
- Use V4 for LVEF, Base e125 for RVSP/pediatric/TAPSE/RV-function. A task-aware encoder dispatch is a 1-line probe config change if we already have both checkpoints.
- **Drawback**: doesn't work for foundation-model narratives. Fine for engineering a production system; does not advance the science.

---

## 5. Recommended next steps (paper-relevant, ranked)

Based on what we know:

1. **Finish the e25 matched-compute matrix** for V4, Base e125, TokenRel+Motion e25 on the remaining unprobed endpoints (MR, RVSP, HCM, HF, Age, LVEF A4C, RV function, LVIDd). Several already in-flight or queued. This is the publishable table.

2. **Probe MV2SV Pilot 655 + Ctrl 658 on the NeurIPS task list** (TAPSE, RV function, pediatric LVEF, A4C 4-task list). Currently only LVEF/RVSP/MR covered for MV2SV. Fills the scorecard.

3. **Run EchoJEPA-L-K on the full downstream suite** (currently only LVEF probed). Our strongest ViT-L; its downstream story is publishable on its own.

4. **EchoNet-LVH as external generalization test** for V4, Base, TokenRel+Motion on LVIDd / IVSd / LVPWd / LVIDs. In-flight (jobs 724/725 in progress on LVIDd). Strong reviewer response to cross-institution validation.

5. **Pre-register the "phase-rel trades generalization for LVEF specificity" finding** as a paper subclaim. Consistent across 6 endpoints; the aggregate story is a reliable finding with 8 data points.

6. **Document the null findings carefully**: MV2SV objective effects, TokenRel diminishing returns, MotionDelta not unlocking TAPSE. Nulls are load-bearing for the "what doesn't work" sections.

7. **Do NOT run another speckle / effective-dim analysis** — both were retracted and the mechanism story stands without them.

---

## 6. Summary

| Finding | Status | Confidence |
|---|---|---|
| JEPA > BYOL / MAE / SALT on LVEF at matched compute | **Established** | High (all paired-bootstrap CIs exclude zero) |
| JEPA builds spatial→temporal hierarchy | **Established** | High (cross-temporal attention) |
| JEPA encodes temporal structure at ~17% level | **Established** | High (severity gradient) |
| JEPA advantage specific to short-horizon temporal video (echo, not MRI) | **Established** | High (CMR direction flip) |
| Speckle filtering is NOT the mechanism | **Retracted prior claim** | High (init-matched re-analysis) |
| Effective dim is NOT the mechanism | **Retracted prior claim** | High (consistent 4-model RankMe) |
| Phase-rel objective trades view generalization for phase fidelity | **Established** | High (6-endpoint matched-compute scorecard) |
| MV2SV v5 losses actively hurt at matched compute | **Established** | Medium (Pilot < Ctrl on every LVEF/RVSP/MR metric) |
| MotionDelta doesn't unlock TAPSE | **Established** | Medium (matched-compute e25 test 0.210 < 0.250 Base/V4) |
| Plain JEPA remains the strongest generalist | **Established** | High (aggregate across 8+ endpoints) |
| Data scale (Kinetics annealing, UHN 18M) is the biggest unresolved lever | **Supported** | Medium (EchoJEPA-L-K and -G wins) |
| Remaining gaps mostly need probe-level or data-scale fixes, not objective changes | **Working hypothesis** | Medium (pending MV2SV e25, more probes) |

---

## 7. Cross-references

- `claude/neurips/controlled-objectives.md` — matched-compute JEPA/BYOL/MAE/SALT comparison (full tables + CIs)
- `claude/neurips/experiments/mv2sv-privileged-multiview.md` — MV2SV / V4 / TokenRel / MotionDelta design + scorecard
- `claude/neurips/experiments/frame-shuffling.md` — 6-condition temporal ablation
- `claude/neurips/experiments/severity-gradient.md` — 13-model severity gradient
- `claude/neurips/experiments/representation-analysis.md` — cross-temporal attention, RankMe, speckle retraction
- `claude/neurips/experiments/salt-comparison.md` — SALT distillation failure mode
- `claude/neurips/experiments/cmr-cross-modality.md` — CMR direction flip
- `claude/neurips/experiments/echobench-e100.md` — 4-model noise robustness
- `claude/neurips/final-task-list.md` — planned NeurIPS task set
- `claude/neurips/lvh.md` — EchoNet-LVH external generalization dataset
- `claude/neurips/completed-experiments.md` — consolidated results inventory
- `claude/neurips/paper-outline.md` — paper-section mapping
