# Variant Comparison and Mechanism Analysis

*Last updated: 2026-05-06*

Living doc consolidating (1) head-to-head encoder-variant comparisons across active tasks with bootstrap CIs, (2) the sampler-vs-objective decomposition for V4's LVEF win, (3) the spatial-scale trade-off hypothesis and its tests, (4) session-derived clarifications that qualify or correct prior framings in `flops-matched-probe-results.md`.

For raw test numbers per task see `flops-matched-probe-results.md`. For the pretraining recipe details see `experiments/` subfolder files. For figures see `figures/lvh/`, `figures/echonet/`, `figures/peds/`, `figures/tapse/`.

---

## 1. Encoder variants at a glance

All six variants start from the identical `jepa_in21k_vitl_e100.pt` and run for ~16–19 EFLOPs of additional post-e100 compute.

| Variant | Objective family | Pair sampler | Adds extra loss head? | FLOPs post-e100 |
|---|---|---|---|---:|
| **V-JEPA†** (base_e125) | reconstruction, single-clip | none | no | 16.1 |
| **V-JEPA†-e130** (FLOPs-tight baseline) | same as V-JEPA†, 5 more epochs | none | no | 19.3 |
| **V-JEPA‡** (fb_sv_548) | same recipe as V-JEPA†, independent seed | none | no | ~18 |
| **MV-PhaseMatched** | reconstruction with 0.25·cross-view smooth-L1 | phase-matched pair (24/study, RR-consistent) | no (cross-view loss is reconstruction) | ~18 |
| **MV-PairedIntra** | intraview V-JEPA only | phase-matched pair + view-pair policy | no | ~18 |
| **MV-PhaseRel** (V4) | intraview V-JEPA + pooled phase-relational InfoNCE | same as MV-PairedIntra + hard negative | **yes** (pooled head) | ~18 |
| **TokenRel-Motion e25** | intraview V-JEPA + token-level relational + motion-delta | phase-matched pair | **yes** (token-level heads) | ~18 |
| **MCC-Anchored e25** | intraview V-JEPA + clip-B reconstruction through zero-gated cross-attention | 2 clips/pair, `any_distinct_in_study` | adapter, not a separate loss | 18.4 |
| **FullJoint-Study 30k** | clip V-JEPA + study-token + anchor + single-view→study | K=8 clips/study | study transformer + projector | 18.0 |

See `experiments/phase-relational-hardneg.md`, `experiments/masked-cross-clip-vjepa.md`, `experiments/full-joint-global-study-token-echomv-jepa.md` for the full objective specifications.

---

## 2. Head-to-head table — active tasks (post-2026-05-06 refresh)

**Tasks kept active for the NeurIPS paper**: LVEF EchoNet-Dynamic, LVEF EchoNet-Pediatric, LVH-LVEDD (MCC+TokenRel queued), LVH-IVSD (TokenRel running), LVH-LVIDS (probes cancelled at ep 6), LVH-LVPWD (held, not queued), HCM PLAX-balanced (TokenRel queued).

**Tasks deprioritized**: RVSP, standalone MR severity, TAPSE. Numbers retained in the flops doc for completeness.

| Task | N_test | V-JEPA† | V4 (MV-PhaseRel) | MCC-Anchored | TokenRel-Motion e25 | FullJoint-Study | MV-PairedIntra |
|---|---:|---:|---:|---:|---:|---:|---:|
| **LVEF EchoNet-Dynamic** ΔR² vs V-JEPA† [95% CI] | 1,277 | 0.646 *(ref)* | **+0.053 [+0.027, +0.079] ✅** | +0.023 [+0.002, +0.044] ✅ | +0.021 [−0.001, +0.044] (97%) | +0.003 [−0.018, +0.024] (tie) | +0.024 [+0.004, +0.045] ✅ |
| **LVEF EchoNet-Pediatric** ΔR² vs V-JEPA†-e125 [95% CI] | 368 | 0.614 [0.467, 0.728] *(ref)* | −0.049 [−0.132, +0.026] (tie, P=0.11) | +0.001 [−0.041, +0.043] (tie) | +0.003 [−0.071, +0.088] (tie) | −0.013 [−0.069, +0.044] (tie) | — |
| **LVH-LVEDD** ΔR² vs V-JEPA† | 340 | 0.455 *(ref)* | +0.041 *(small-N; val says V4 loses)* | *queued (861)* | *queued (855)* | — | — |
| **LVH-IVSD** ΔR² vs V-JEPA†-e130 [95% CI] | 339 | 0.467 [0.400, 0.537] *(ref)* | **−0.226 [−0.273, −0.179] ❌ V4 loses** | **−0.115 [−0.149, −0.078] ❌ (half of V4's loss)** *(865)* | **−0.094 [−0.132, −0.052] ❌ (half of V4's loss)** | — | — |
| **LVH-LVIDS Test R²** | 9,575 | *cancelled at ep 6* | *cancelled at ep 6* | — | — | — | — |
| **LVH-LVPWD Test R²** | 11,723 | **not probed** | **not probed** | — | — | — | — |
| **HCM PLAX-balanced Test AUROC** (all 5 collapsed on rule — protocol prior-mismatch) | 2,216 | 0.519 | 0.545 | 0.554 | 0.542 (850) | 0.530 | — |
| **HCM A4C Test AUROC** (unbalanced, prior-matched, true softmax from 862) | 2,165 | 0.546 [0.496, 0.592] *(e125)* | — | — | **0.760** [0.699, 0.817] (e5, 3.6 EF) | — | — |
| **MR any-MR Test AUROC** (true softmax) | 4,482 | **0.783** [0.770, 0.796] *(e130, 864)* | 0.767 [0.753, 0.781] *(862)* | **0.778** [0.764, 0.792] (848, 864) | 0.770 [0.757, 0.784] (e5, 862); 0.770 [0.756, 0.784] (e25, 864) | *queued (858)* | 0.778 [0.764, 0.791] *(862)* |
| **MR ≥moderate Test AUROC** (true softmax) | 4,482 | **0.790** [0.775, 0.804] *(e130, 864)* | 0.778 [0.763, 0.792] *(862)* | **0.790** [0.775, 0.806] (848, 864) | 0.781 [0.766, 0.796] (e5, 862); 0.782 [0.765, 0.797] (e25, 864) | *queued (858)* | 0.787 [0.772, 0.802] *(862)* |
| **MR 4-cls OVR macro AUROC** (true softmax) | 4,482 | **0.738** [0.727, 0.749] *(e130, 864)* | 0.726 [0.715, 0.737] *(862)* | **0.737** [0.727, 0.748] (848, 864) | 0.729 [0.718, 0.740] (e5, 862); 0.730 [0.719, 0.741] (e25, 864) | pending (858) | 0.733 [0.723, 0.744] *(862)* |
| **MR severe AUROC** (class=3, true softmax) | 4,482 | **0.819** [0.796, 0.842] *(e130, 864)* | 0.810 [0.785, 0.834] *(862)* | 0.818 [0.794, 0.840] (848, 864) | 0.816 [0.793, 0.839] (e5, 862); **0.820** [0.797, 0.843] (e25, 864) | — | 0.817 [0.794, 0.840] *(862)* |
| **MR 4-cls OVR ΔAUROC vs V-JEPA†-e130** [95% CI, paired, B=2000] | 4,482 | *ref* | **−0.012 [−0.018, −0.005] ❌ V4 loses** | **0.000 [−0.006, +0.005] (exact tie)** P=0.45 | **−0.009 [−0.014, −0.004] ❌** (e5); **−0.008 [−0.014, −0.001] ❌** (e25) | pending (858) | **−0.005 [−0.010, +0.001] borderline** P=0.96 |
| **MIMIC TAPSE Test R²** | 2,000 | 0.247 | 0.250 | 0.255 *(851)* | 0.210 ❌ | — | — |

✅ = 95% CI excludes zero. ❌ = 95% CI excludes zero in the unfavorable direction.
¹ MR AUROCs are true softmax values from patched-eval reruns (862: V4/MV-PairedIntra/TokenRel-e5; 864: V-JEPA†-e130/MCC-848/TokenRel-e25). Job **858** (FJ real, queued on node 146) still pending. All single-variant CIs are B=2000 unpaired; the last row gives paired deltas vs V-JEPA†-e130 (B=2000, aligned per-video).
⁴ **Correction (2026-05-06)**: these 0.753 / 0.732 numbers were computed from job 823's predictions, which **actually used the MCC encoder** because `mr_a4c_probes_fj_30k.sbatch` had an incorrect `CKPT_S3` pointing at MCC instead of FJ. Now relocated to the MCC column. Job **848** (MCC restart) completed 2h01m and is our first clean MCC MR probe (its 4-class OVR top-1 0.7373 closely matches 823's 0.7348). Job **858** (new FJ MR probe with corrected `CKPT_S3` = `echomv_jepa/full_joint_restart_v2_30k_runs/776/latest.pt`) queued on node 146 after chain 859 → 855 → 861 → 851 → 858.
"—" = not probed and not currently queued.

### Direct deltas for sampler-vs-objective decomposition on LVEF (see §3)

Only deltas needed for the decomposition; other pairwise comparisons removed to match "vs vanilla V-JEPA†" convention in the main head-to-head table.

| Pair | ΔR² [95% CI] | P(a better than b) |
|---|---:|---:|
| MV-PhaseRel − V-JEPA† | +0.053 [+0.027, +0.079] | 100% (total V4 lift) |
| MV-PairedIntra − V-JEPA† | +0.024 [+0.004, +0.045] | 98% (sampler only) |
| MV-PhaseRel − MV-PairedIntra | +0.029 [+0.008, +0.050] | 99.5% (objective only, needed for decomp) |

---

## 3. Sampler-vs-objective decomposition (V4's LVEF win)

V4's total lift over V-JEPA†-e125 on EchoNet-Dynamic LVEF is **+0.053 R²**. Paired-bootstrap decomposition on the test predictions:

| Component | ΔR² [95% CI] | p(Δ≠0) |
|---|---:|---:|
| Sampler alone (MV-PairedIntra − V-JEPA†) | +0.024 [+0.004, +0.045] | 0.020 |
| Objective alone (MV-PhaseRel − MV-PairedIntra) | +0.029 [+0.008, +0.050] | 0.005 |
| Total (MV-PhaseRel − V-JEPA†) | +0.053 [+0.027, +0.079] | <0.001 |

**Sampler ≈ 45% of the lift, objective ≈ 55%.** Both components are individually significant at α=0.05. Sampler alone delivers roughly the same ΔR² as TokenRel-Motion e25 / MCC / MV-PairedIntra (all cluster at ~+0.02 over base). The phase-relational InfoNCE head is what pushes V4 past that cluster to +0.053.

**Clarification**: an earlier framing claimed the sampler was ~70% of the lift. That was based on a misquoted SV baseline (~0.605) from an MV2SV-protocol MIMIC-A4C number. On the canonical EchoNet-Dynamic test split (V-JEPA†-e125 = 0.646), the correct decomposition is ~45/55, not 70/30.

---

## 4. Spatial-scale trade-off hypothesis

V4's pooled phase-relational InfoNCE pulls phase-matched clips together in pooled embedding space and pushes away batch negatives. The loss rewards discrimination along the **phase axis conditioned on view**. Features orthogonal to that axis — fine wall texture, tissue-boundary sharpness — are compression-friendly under the loss and get smoothed during +25 epochs of pretraining.

### Per-task predictions vs observations

| Task | Spatial scale needed | Phase-anchored? | Predicted V4 Δ | Observed V4 Δ | Confirmed? |
|---|---|---|---:|---:|:---:|
| LVEF (EchoNet-Dynamic) | coarse (EDV − ESV) | yes (ED − ES) | +large | **+0.053 R²** | ✅ |
| LVEDD (EchoNet-LVH) | coarse (chamber diameter) | yes (end-diastole) | +small | +0.041 R² test / −0.025 R² val | inconclusive — val/test disagree |
| **IVSD** (EchoNet-LVH) | **fine (wall thickness, mm)** | yes (end-diastole) | **−large** | **−0.225 R² test** | ✅ sharpest confirmation |
| LVIDS (EchoNet-LVH) | coarse (end-systolic diameter) | yes (end-systole) | +small | *cancelled at ep 6 (842/843); no result* | N/A |
| LVPWD (EchoNet-LVH) | fine (posterior wall thickness) | yes (end-diastole) | −medium | *held (844/845), not queued* | TBD |
| RVSP (MIMIC A4C/SV) | fine (TR jet velocity via Doppler) | no | −medium | **−0.139 R² vs V-JEPA‡** | ✅ |
| TAPSE | coarse (RV annular excursion) | weak | ≈0 | +0.003 R² (tie) | consistent |
| Peds LVEF | coarse (same as LVEF but smaller scale) | yes | +large or tie | −0.048 R² (underpowered, N=368) | not conclusive |

### Confirmed wins for the hypothesis

- **LVEF (+0.053)** and **IVSD (−0.225)** together span the full range of the hypothesis. Same encoder, same phase anchor, same patient cohort, opposite outcomes depending on whether the task reads coarse vs fine spatial features.
- **RVSP (−0.139)** extends the pattern to fine-Doppler tasks.

### Open questions

- **Does the sampler alone cause IVSD collapse (mechanism B), or is it the InfoNCE head (mechanism C)?** MV-PairedIntra on IVSD is the clean test. Not yet queued.
- **Does token-level phase supervision avoid the collapse?** TokenRel-Motion e25 on IVSD would settle whether pooling is the culprit vs phase supervision in general.
- ~~**Does MCC preserve fine spatial via its reconstruction target?**~~ **Resolved by job 865**: MCC-IVSD test R² = 0.353 [0.286, 0.422], ΔR² vs base = **−0.115 [−0.149, −0.078] ❌**. MCC does *not* preserve fine spatial features on IVSD — its loss is about half of V4's and comparable to TokenRel's. All three +25-ep continuation variants (V4 / TokenRel / MCC) damage IVSD relative to the starting checkpoint; the mechanism is not specifically pooled InfoNCE.

All three sbatches are uniform-cost (~40 min each), and together they would provide a definitive mechanism attribution section for the paper.

---

## 5. Variant-level conclusions (2026-05-06 snapshot)

### V4 (MV-PhaseRel)

- **Decisively wins LVEF EchoNet-Dynamic** (+0.053 R² test, CI excludes zero, val and test agree). Sampler contributes +0.024, phase-relational InfoNCE contributes +0.029.
- **Decisively loses LVH-IVSD** (−0.225 R²). Consistent with spatial-fidelity trade-off.
- **Decisively loses RVSP** (−0.139 R²). Consistent with pooling-collapse mechanism + view-integration deficit.
- **Ties or underperforms** on peds LVEF (−0.048 directional, underpowered), TAPSE (+0.003), HCM PLAX-balanced (probe collapse confound).

### V-JEPA† (base_e125) / V-JEPA†-e130 / V-JEPA‡

- Baseline. No extra loss, no pair sampler. Underperforms V4 on LVEF Dynamic but overperforms V4 on IVSD (+0.226 R²) at tight or favorable FLOPs.
- **V-JEPA†-e130 leads MR at matched FLOPs** (rerun 864, true softmax): 4-class OVR 0.738 [0.727, 0.749], any-MR 0.783, ≥mod 0.790, severe 0.819. Paired ΔOVR vs V-JEPA†-e130: V4 **−0.012 [−0.018, −0.005] ❌**; TokenRel-e5 **−0.009 [−0.014, −0.004] ❌**; TokenRel-e25 **−0.008 [−0.014, −0.001] ❌**; MV-PairedIntra −0.005 [−0.010, +0.001] P=0.96 (borderline); **MCC-Anchored 0.000 [−0.006, +0.005] exact tie P=0.45** — only variant that matches the baseline. Phase/token-rel objectives actively hurt MR; reconstruction-target supervision (MCC) preserves it.

### MV-PairedIntra

- **Most promising non-V4 variant.** Captures the sampler's +0.024 R² LVEF lift without V4's failure modes on RVSP (+0.090 R² over V4 there) or (likely) IVSD.
- **On MR, beats V4** (rerun 862 paired bootstrap): Δ 4-cls OVR +0.007 [+0.001, +0.014] P=0.987 ✅. But **borderline loses to V-JEPA†-e130** (−0.005 [−0.010, +0.001] P=0.95, rerun 864 reference). Net: better than V4, still (marginally) below the plain baseline.
- Would be the cleanest "V4 control" for the mechanism story. **Not yet probed on LVH or peds.**

### TokenRel-Motion e25

- Similar LVEF lift to MV-PairedIntra (+0.021). **Significantly worse on TAPSE** (−0.037 R², P=2%).
- **Peds LVEF (job 852)**: test R² = 0.617 [0.503, 0.711]; ΔR² vs V-JEPA†-e125 = **+0.003 [−0.071, +0.088] P=0.51 (exact tie)**. TokenRel-e25 lands at the top of the peds cluster with base, MCC, and FJ — no differentiation. V4 remains the outlier below (0.567).
- **IVSD mechanism result** (job 859, test N=339): Test R² = 0.374 [0.310, 0.440]. **TokenRel-e25 beats V4 by ΔR² = +0.132 [+0.097, +0.170] P=1.000 ✅** — token-level phase supervision is strictly better than pooled on fine spatial features. But TokenRel still **loses to base** by ΔR² = −0.094 [−0.132, −0.052] — half of V4's loss. Pearson Δ vs base is not significant (CI straddles zero). Now that MCC's IVSD result (job 865) is also in, the full triad reads V-JEPA†-e130 0.467 > TokenRel 0.374 ≈ MCC 0.353 > V4 0.243. **Revised mechanism claim**: the pooled InfoNCE head damages IVSD most severely, but it is not the only +25-ep continuation objective that damages IVSD — token-level phase supervision and reconstruction-anchored supervision both damage IVSD by comparable intermediate amounts (−0.09 and −0.12 R²). The right reading is a "spatial-scale / extended-continuation trade-off" rather than "pooling is uniquely bad."
- **HCM A4C AUROC confirmed via rerun 862 with true softmax**: TokenRel-e5 = **0.760 [0.699, 0.817]** vs V-JEPA†-e125 = **0.546 [0.496, 0.592]**. Non-overlapping CIs → +0.214 AUROC gap is real. Largest single-task classification signal any variant has shown. Caveats: best.pt selected on val AUROC (which peaks ~0.575 for TokenRel — not high). Val/test-flip interpretation: V-JEPA† overfits val-specific positives → test collapses (0.65 → 0.55); TokenRel underfits val → test reflects encoder-level HCM signal (0.57 → 0.76).
- **HCM PLAX-balanced (job 850)**: TokenRel-Motion-e25 **test AUROC = 0.542** — falls into the 0.52–0.55 cluster with all 4 other variants (V-JEPA† 0.519, V4 0.545, MCC 0.554, FJ 0.530). Protocol prior-mismatch (33% train / 2.9% test) collapses every variant. A4C→PLAX transfer question **not resolved** by this run; confounds: balanced prior, view change, e5 vs e25. Would need TokenRel-e5 on PLAX-unbalanced to disentangle.
- **MR vs V-JEPA†-e130** (rerun 864, paired): Δ 4-cls OVR = **−0.009 [−0.014, −0.004] ❌ (e5)** and **−0.008 [−0.014, −0.001] ❌ (e25)**. TokenRel (both checkpoints) loses to baseline on MR — HCM A4C is where its advantage lives. Compute doesn't help: e5 and e25 both at ~0.730 OVR, both losing.
- **Only +25 EF variant with uniform ΔMAE < 0 across all four LVEF strata** (Reduced −0.05, Mild −0.53, Normal −0.20, Hyper −0.54). MV-PairedIntra and MCC both regress on the Reduced stratum by +0.25 to +0.36 EF %; TokenRel does not. Caveats: Reduced gain is marginal (−0.05, within noise); Hyperdynamic MAE win (−0.54) coincides with Pearson drop of −0.059 → partly regression-to-mean. Per-stratum details in `reports/stratified_results/README.md` §1 "TokenRel-Motion e25 — only +25 EF variant with uniform MAE improvement."

### MCC-Anchored

- Matches MV-PairedIntra on LVEF (+0.023). No differentiation there.
- **Peds LVEF: essentially tied with V-JEPA†-e125** (ΔR² +0.001 [−0.041, +0.043] P=0.51, exact tie). **Directional win over V4** (ΔR² +0.049 [−0.015, +0.127] P=0.92) but CI straddles zero — **R² Δ is NOT significant** at 95%. Only the Pearson Δ reaches significance: +0.042 [+0.001, +0.090] P=0.977. Retracts earlier claim of "MCC wins peds by +0.041 [+0.002, +0.090] ✅ P=96%" — that CI appears to have been from a different run/bootstrap seed or used a different scaler.
- **MR: exact tie with V-JEPA†-e130** (rerun 864, paired): Δ 4-cls OVR = 0.000 [−0.006, +0.005] P=0.45. **Only variant that matches the baseline on MR** — all phase/token-rel variants lose significantly. 4-class OVR 0.737 [0.727, 0.748]; any-MR 0.778 [0.764, 0.792]; ≥mod 0.790 [0.775, 0.806]; severe 0.818 [0.794, 0.840]. Reconstruction-style pretraining preserves the MR signal that phase-relational supervision degrades.
- **IVSD (job 865, N=339)**: Test R² = 0.353 [0.286, 0.422]. **MCC loses to base by ΔR² = −0.115 [−0.149, −0.078]** — roughly half of V4's loss (−0.226), comparable in magnitude to TokenRel-e25's loss (−0.094). **Mechanism reading**: reconstruction-anchored supervision does *not* preserve fine spatial features on IVSD; it damages them, similarly to how token-level phase supervision does, and the common factor across all three +25 ep variants (V4 / TokenRel / MCC) is that every continuation objective hurts IVSD relative to the starting checkpoint. V4's pooled InfoNCE is the most damaging, but neither MCC nor TokenRel rescue IVSD back to base-level performance. The paper's "any phase-aware or reconstruction-aware +25ep continuation damages IVSD" reading now has full triad support.
- **TAPSE (job 851, N=2000)**: Test R² = 0.255 — essentially tied with V-JEPA† (0.247) and V4 (0.250); comfortably above TokenRel-e25 (0.210). No paired CI computed.
- **Not demonstrated on multi-clip inference** (its design use-case).

### FullJoint-Study

- **Does not beat V-JEPA† on any single-clip headline regression task** (LVEF +0.003, peds −0.013).
- **MR severity result was wrong** — what I reported earlier as "FJ MR AUROC 0.753" was actually a second MCC run (job 823's sbatch incorrectly loaded MCC's checkpoint). Real FJ MR pending job 858. So FJ has no confirmed single-clip advantage yet.
- Designed for K=8 study-level inference, which we don't currently use for most tasks.
- **Status**: low defensibility across current task list. Wait for real FJ MR (858) before deciding its paper role.

---

## 6. Session-derived clarifications and corrections

### a. Class weights in the probe

The attentive probe in `evals/video_classification_frozen/eval.py` **already applies inverse-frequency class weights** to `CrossEntropyLoss` when `task_type == "classification"` and `use_focal_loss=False`. HCM PLAX-balanced probes (jobs 806-809) logged `Class weights (inverse freq): [0.667, 1.333]` from a 2:1 train prior. The PLAX HCM probe collapse is **not** from missing class weights — it's from the **train/test prior mismatch** (train balanced 33% pos, test at 2.9% pos). To fix: either use unbalanced-prior train (matches test), or apply post-hoc threshold recalibration at inference time.

### b. TokenRel-Motion e5 HCM A4C result is honest, not degenerate

TokenRel A4C HCM (AUROC 0.760) was on **matched train/test priors** (both ~2.8%). Probe made 18 positive flags, 3 correct (precision 0.167, recall 0.049). Not "collapse" — a conservative-but-non-degenerate classifier. The +0.21 AUROC gap vs V-JEPA† (which collapsed entirely to all-negative) is an **encoder-level signal**, not probe-head luck. Larger than the PLAX-balanced comparison suggested because the PLAX runs suffered prior mismatch while the A4C runs did not.

### c. MR nan AUROC on V4/MV-PairedIntra was a missing-sklearn bug, not probability collapse (corrected 2026-05-06)

**Original (incorrect) narrative** — now retracted: "V4 and MV-PairedIntra MR probes produce class-3 probabilities that degenerate to near-zero across all samples, causing sklearn's multi-class OVR AUROC to return nan. MCC and FJ produce non-degenerate probability matrices → non-nan AUROCs."

**Actual cause**: Jobs 609 (V4) and 611 (MV-PairedIntra) MR probe runs ran on 2026-05-02 with a source tarball that did not include `scikit-learn` in the Python environment. `eval.py`'s AUROC block caught the `ImportError: No module named 'sklearn'` and fell through to nan for `val_auroc`, `val_bal_acc`, and `val_kappa` across all 20 epochs. The nan pattern is consistent with this: all three metrics nan from epoch 1, because the import failed at probe startup.

**Spot-checking other runs** (jobs 620, 622, 704, 705, and all newer jobs through 849) shows non-nan AUROCs + clean sklearn imports — the bug was isolated to early May 2 tarball revisions.

**Corrected ≥moderate AUROC values** (recomputed from saved test prediction CSVs using top-1 probability as positive-class score):

| Model | any-MR AUROC | ≥mod AUROC |
|---|---:|---:|
| V4 (MV-PhaseRel) | 0.689 | 0.670 |
| MV-PairedIntra | 0.719 | 0.688 |
| TokenRel-Motion-e5 | 0.725 | 0.701 |
| **FullJoint-Study 30k** | **0.753** | **0.732** |

These differ from the earlier values in `flops-matched-probe-results.md` (0.778/0.782/0.777) — those were incorrectly mapped from a different metric source. The corrected ranking is a **clear gradient, not a tight band**: FullJoint > TokenRel > MV-PairedIntra > V4.

**Implication**: the earlier "MCC/FJ uniquely preserve rare-class features where V4 / MV-PairedIntra don't" framing is retracted. On the corrected numbers, FJ and TokenRel-Motion e5 both rank-order MR classes better than V4 / MV-PairedIntra, but there's no mechanism claim about probability-level degeneration — V4's probabilities are likely perfectly non-degenerate, we just haven't computed true multi-class OVR AUROC for it (would require re-running val inference against probe checkpoints in the current clean environment).

### d. V-JEPA† was never probed on RVSP; V-JEPA‡ is the V-JEPA reference there

Only V-JEPA‡ (`fb_sv_548` / `finalbudget_singleview_25of100_548`) has an RVSP probe. Claims like "V-JEPA beats V4 on RVSP" should reference V-JEPA‡ specifically, not treat V-JEPA† and V-JEPA‡ as interchangeable. The two are seed-replicates of the same recipe, but we only have one seed on RVSP.

### e. The +0.041 LVEDD test win is not a reliable headline

- Test Δ = +0.041 R² [+0.002, +0.081]: CI lower bound +0.002, barely excludes zero.
- Val Δ on the 3.4× larger val split = −0.025 to −0.030 R² across all selection strategies (best-val, final-epoch, last-5-avg). Val says V4 loses.
- Same pattern as peds LVEF (V4 wins val, loses test — or vice versa, depending on split N).
- Cause is most likely **small-N test set** (N=340) sampling variance amplified by R²'s SST-denominator sensitivity.
- Paper recommendation: drop from headline or report both splits with N flagged.

### f. "V4 is multi-view" is architecturally wrong

The data path is multi-view (3 clips per sample across same-study view pairs). The encoder is single-view — each clip flows through the shared ViT-L independently. View identity enters only via the 64-d view embedding into the pooled relational head. V4 does **not** perform cross-view latent prediction. Contrast with MV2SV / `factorized_head` which do fuse multi-view targets.

### g. V4 FLOPs ≈ 18 EFLOPs, but calculation has ±30% convention uncertainty

Non-trivial because V4 runs 3 teacher forwards per sample (concat), 1 masked-student forward, predictor, + relational head, with different conventions weighting teacher/student forwards differently. Published 16.1 EFLOPs (V-JEPA†-e125) vs ~18 (V4) gap is within accounting uncertainty, but the V4-e25 resume-to-e50 + L-K-init V4 50-ep runs (held jobs 825-827) provide strict <1%-FLOPs matches for the final-paper comparison.

### h. "MCC should work better with K=8 inference" is unlikely

MCC's cross-attention adapter was trained on K=2 pairs. Scaling inference to K=8 doesn't buy the "cross-clip context" MCC was designed for — the adapter just sees 8 copies of a 2-clip pattern. MCC's architectural claim can be tested via **2-clip inference on LVEF EchoNet-Dynamic**; that would be one sbatch, ~40 min, settles whether MCC's adapter adds value beyond single-clip.

### i. 2-view probe inference only helps on dropped tasks

Adding a second view at the probe head helps on RVSP / MR / AS (all cross-view-dependent). **None of these are in the active task list.** On LVEF / LVEDD / IVSD / LVPWD / peds / age / mortality / TAPSE / HCM, the canonical view carries the signal and a second view is noise or redundant. For the current task list, 2-view probe inference is not a high-value follow-up.

### j. Val/test disagreements and reversals are common on small-N probes — trust test

Several tasks in this evaluation suite show val and test giving **different variant rankings** or even **different signs of Δ**. Known cases (2026-05-06):

| Task | N_val | N_test | Val says | Test says |
|---|---:|---:|---|---|
| LVH-LVEDD | 1,141 | 340 | V4 loses by −0.025 R² | V4 wins by +0.041 R² (small-N) |
| HCM A4C | ~2,070 | 2,165 | V-JEPA†-e125 0.66, TokenRel-e5 0.57 | V-JEPA†-e125 0.55, TokenRel-e5 **0.76** (non-overlapping CIs) |
| LVEF Pediatric | ~200 | 368 | V4 wins val R² | V-JEPA† / MCC / FJ / TokenRel-e25 all tie at top; V4 is outlier below |
| MR A4C | ~1,230 | 4,482 | mostly agrees with test ranking | V-JEPA†-e130 leads, all SSL variants lose |

**Three conditions that together produce these flips:**
1. **Small test set or small positive pool** — HCM has 61 positives in test; LVEDD has 340 total test clips. At those scales the 95% CI on R² is ~0.1 wide and on AUROC is ~0.06 wide. Most observed "flips" are noise-range crossings, not real reversals.
2. **Balanced-prior or stratified probe training** — when the probe is trained on a non-natural distribution (HCM PLAX-balanced 33% train vs 2.9% test), the decision boundary is fit to a train-adjacent val prior, not the test prior. Val AUROC becomes a misleading model-selection signal.
3. **Probe capacity vs encoder signal.** When a probe easily fits val, it's likely overfitting val-specific features (strong val-AUROC that doesn't generalize). When a probe barely fits val, the best.pt reflects encoder-level geometry rather than probe-head overfitting (modest val-AUROC that does generalize). The HCM A4C val/test flip is a clean example of this.

**Operational rule for this work:** when val and test disagree:
- **If test N is ≥1,000** and test CI excludes zero, trust test.
- **If test N is small** (≤500) and the flip sign is within bootstrap CI, report both and note inconclusiveness.
- **Do not report val-only findings as paper claims** — tested this was the LVEDD "+0.041 test win" (flagged in §6e) and the HCM PLAX-balanced cluster (§6 below).
- **For small positive pools** (<100), a `latest.pt` vs `best.pt` test comparison is the cheapest probe of whether the flip is selection-noise or real.

This pattern is stronger evidence that our probe protocol is near-saturated on most of the active tasks — a better probe architecture or multi-clip probe would likely shrink these gaps.

---

## 7. Currently running / queued (as of 2026-05-06 ~13:10 UTC)

### Running

| Job | Task | Node | Purpose |
|---|---|---|---|
| **864** | `classif_rerun_base_mcc` | 56 | Patched-eval rerun for V-JEPA†-e130 (849), MCC-Anchored (848), TokenRel-Motion-e25 (853) MR probes. Writes `prob_class_0..3` → true 4-class OVR AUROC + paired bootstrap CIs for the 3 MR variants not in 862. |
| **855** | `lvh_lvedd_tokenrel_r2_e25` | 146 | TokenRel-Motion-e25 on EchoNet-LVH LVEDD (N=340). Tests whether token-level phase supervision resolves V4's val/test disagreement on LVEDD. |

### Completed this session

| Job | Task | Result |
|---|---|---|
| 840/841 | LVH-IVSD (base_e130, V4) | V4 loses ΔR² = −0.226 [−0.273, −0.179]. Base R² = 0.467 [0.400, 0.537]; V4 = 0.243 [0.169, 0.316]. |
| 859 | `lvh_ivsd_tokenrel_r2_e25` (2h25m) | TokenRel-e25 IVSD: R² = 0.374 [0.310, 0.440]. **Beats V4 by +0.132 [+0.097, +0.170] P=1.000 ✅**, still loses to base by −0.094. Mechanism test partial. |
| 849 | `mr_a4c_base_e130` (2h02m) | V-JEPA†-e130 MR. 4-class OVR (top-1 approx) 0.738; any-MR 0.715; ≥mod 0.698. 864-pending refresh with true softmax. |
| 848 | `mr_a4c_mcc_e25_restart` (2h01m) | MCC-Anchored MR: 4-class OVR (top-1 approx) 0.737. First cleanly-labeled MCC MR probe. |
| 853 | `mr_a4c_tokenrel_r2_e25` (2h02m) | TokenRel-Motion-e25 MR: 4-class OVR (top-1) 0.730. Tied with TokenRel-e5 (0.729) despite 5× compute. |
| **862** | `classif_rerun_optA` (29m) | Patched-eval rerun for V4/MV-PI/TokenRel-e5 MR + V-JEPA†/TokenRel-e5 HCM-A4C. **MV-PairedIntra > V4 on all 3 MR metrics (CI excludes zero)**; TokenRel-e5 HCM A4C 0.760 [0.699, 0.817] vs V-JEPA†-e125 0.546 [0.496, 0.592] confirmed with non-overlapping CIs. |
| **850** | `hcm_plax_bal_probes_tokenrel_r2_e25` (37m) | TokenRel-Motion-e25 HCM PLAX-balanced: test AUROC 0.542 — falls into 0.52–0.55 cluster with all other variants. Protocol prior-mismatch collapse. |
| **852** | `pediatric_probes_tokenrel_r2_e25` (35m) | TokenRel-Motion-e25 peds LVEF: R² = 0.617 [0.503, 0.711]. ΔR² vs V-JEPA†-e125 = +0.003 [−0.071, +0.088] P=0.51 (tie). Sits at top of peds cluster with base/MCC/FJ. |
| 857 | `classif_rerun_optA` (1st try, 21m) | Failed — pre-patch tarball, wrong HCM CSV fallback. Superseded by 862. |

### Cancelled earlier this session

| Job | Task | Reason |
|---|---|---|
| 842/843 | `lvh_lvids_*` (probes) | Cancelled at ep 6; LVIDS task dropped as non-load-bearing. |
| 854 | `lvh_lvedd_mcc_e25` (first try) | Cancelled to prioritize IVSD mechanism test (859). Requeued as **861** after 865. |
| 856 | `mr_test_rerun_v4_pi` | Superseded by 857→860→862. |
| 860 | `classif_rerun_optA` (resub 1) | Superseded by 862. |
| 863 | `classif_rerun_base_mcc` | Superseded by 864 (adds TokenRel-Motion-e25 to rerun list). |

### Queued (automatic, dependency-chained)

**Node 146 chain** (after 855, currently running):

| Job | Task | Depends on | Purpose |
|---|---|---|---|
| **865** | `lvh_ivsd_mcc_e25` | afterany:855 | **Completes IVSD mechanism triad.** MCC-Anchored-e25 on IVSD — disambiguates whether V4's collapse is specifically pooled-InfoNCE or shared with other phase-aware objectives. |
| **861** | `lvh_lvedd_mcc_e25` | afterany:865 | MCC-Anchored-e25 on LVEDD. Third encoder on LVEDD after base/V4. Requeue of cancelled 854. |
| **851** | `tapse_mcc_e25` | afterany:861 | MCC-Anchored-e25 on MIMIC TAPSE. Fills MCC TAPSE coverage gap. |
| **858** | `mr_a4c_fj_30k` (corrected) | afterany:851 | **First real FullJoint-Study MR probe.** Corrected CKPT_S3. Uses patched eval — writes `prob_class_*` directly. |

Node 56 is empty after 864 completes (~15 min from now).

Expected completion: Node 56 ~20m (864 remaining). Node 146 ~10m (855 remaining) → ~45m (865) → ~45m (861) → ~2h (851) → ~2h (858). All queued work done ~6h from 2026-05-06 13:10 UTC.

### Post-862+864, MR test coverage with full per-class probabilities:

| Variant | Rerun job | Probe best.pt source |
|---|---|---|
| V4-e25 | 862 | 609 |
| MV-PairedIntra | 862 | 611 |
| TokenRel-Motion-e5 | 862 | 699 |
| V-JEPA†-e130 | 864 | 849 |
| MCC-Anchored-e25 | 864 | 848 |
| TokenRel-Motion-e25 | 864 | 853 |
| FullJoint-Study 30k | 858 (direct) | 858 (patched eval) |

At that point we have proper 4-class OVR AUROC + paired bootstrap CIs for all 7 MR variants.

### Held (user-held, not in automatic queue)

| Job | Task | Why held |
|---|---|---|
| 844 | `lvh_lvpwd_base_e130` | LVPWD probe — low-priority follow-up to LVIDS (which was cancelled mid-run); paper story probably doesn't need a 4th LVH regression. |
| 845 | `lvh_lvpwd_v4_e25` | Same as 844 |
| 846 | `lvedd_latest_test` | Selection-strategy sensitivity check on LVEDD (best.pt vs latest.pt test). Low compute (~5min) but low information given val-CI answer from 847 will be definitive. |
| 847 | `lvedd_val_infer` | **High-value hold** — paired-bootstrap CI on LVEDD val Δ would definitively resolve whether LVEDD +0.041 claim should be dropped. Held for compute prioritization, not value. |
| 825 | `final_phase_rel25_resume_e50` | Pretrain: V4 resume e25→e50 (global e150). ~11h. FLOPs-tight V4 vs V-JEPA†-e155 comparison. |
| 826 | `echojepa_lk_resume_e155` | Pretrain: L-K e130→e155. ~11h. Third FLOPs-tight lineage. Chained after 825. |
| 827 | `final_phase_rel_lkinit50` | Pretrain: V4 from L-K e100, 50ep. ~22h. Tests whether L-K init compounds with phase-relational objective. |

---

## 8. Runs to do (ranked by information / compute)

Ordered by information-per-compute ratio. Only includes experiments not currently running or queued.

### High priority — small compute, high information

1. **MV-PairedIntra on IVSD** (~40 min, 1 sbatch, either node) — tests mechanism B: does the phase-matched sampler alone cause V4's IVSD collapse, without the relational head? If yes → V4's IVSD collapse is sampler-side. If no → pooled InfoNCE head is the culprit. Not yet queued.

2. ~~**TokenRel-Motion e25 on IVSD**~~ — **completed as job 859**. Result: beats V4 by +0.132 R² P=1.000; still loses to base by −0.094. Mechanism claim partial.

3. ~~**MCC on IVSD**~~ — **completed as job 865**. Test R² = 0.353 [0.286, 0.422]. MCC loses to base by ΔR² = −0.115 [−0.149, −0.078], comparable to TokenRel's −0.094 loss, about half of V4's −0.226 loss. Mechanism triad result: all three +25-ep variants damage IVSD, V4 by the most.

4. ~~Re-run V4 + MV-PairedIntra val/test inference against saved probe checkpoints for MR~~ — **completed as 862** + **running as 864**. Once 864 + 858 land, all 7 MR variants will have true 4-class OVR AUROC + paired bootstrap CIs.

5. **Release `847 lvedd_val_infer`** — ~5-10 min. Gives paired-bootstrap val ΔR² CI for LVEDD. Held but not for compute reasons; can release anytime. If val CI excludes zero in V4's negative direction, the "V4 wins LVEDD" claim should be dropped from the paper.

6. **TokenRel-Motion e5 on HCM PLAX-unbalanced** (~40 min) — disentangles the A4C vs PLAX confound for the HCM story. Current data: TokenRel A4C unbalanced 0.760 vs TokenRel PLAX **balanced** 0.542. The comparison is confounded by both view (A4C→PLAX) and protocol (unbalanced→balanced). Running TokenRel-e5 on PLAX-unbalanced isolates the view factor. If ≈0.76 → view is irrelevant, the PLAX-balanced collapse is entirely protocol. If ≈0.55 → TokenRel's HCM signal is view-specific to A4C.

7. **TokenRel-Motion e25 on HCM A4C (complete job 726)** (~40 min) — isolates the e5-vs-e25 factor at matched view/protocol. Job 726 stalled at ep 6 with no final result. If e25 A4C ≈ 0.76, TokenRel's HCM signal is preserved through compute. If e25 A4C < 0.76, extended supervision absorbs HCM-discriminative directions into phase features.

### Medium priority — moderate compute

8. **Unbalanced-prior HCM PLAX re-run** for all 5 variants (~40 min × 5 = 200 min). Confirmed by 850: **balanced-prior protocol collapses every variant to 0.52–0.55 regardless of encoder** (including TokenRel which wins on A4C). A clean 5-way HCM-PLAX comparison requires matching the train prior to the test prior.

9. **L-K-init V4 + V-JEPA†-e155 + V4-e50 downstream probes** — depends on pretrain jobs 825/826/827 being released. All three are held. Once those complete (~22h each), their downstream comparisons are the FLOPs-tight 35-EFLOPs head-to-head. Load-bearing for the paper's FLOPs-matched claim.

10. **MV-PairedIntra on peds LVEF** (~40 min) — adds sampler-only variant to peds. Current peds: V-JEPA†-e125 0.614, V4 0.567, MCC 0.615, FJ 0.602, TokenRel-e25 0.617 — missing only MV-PairedIntra.

### Low priority — high compute or low marginal information

9. **LVPWD probes** (jobs 844/845, held) — fourth LVH regression task. LVH story is already clear from LVEDD+IVSD+LVIDS (partial). Low marginal information vs the IVSD mechanism isolation in high-priority list.

10. **MCC + FJ on HCM A4C (original split, unbalanced)** — expected AUROC 0.60-0.74; won't beat TokenRel's 0.760. Adds coverage but no new paper claim.

11. **2-clip inference for MCC on LVEF EchoNet-Dynamic** — tests MCC's architectural intent. Expected ΔR² +0.005-0.015 over single-clip (generic 2-clip prediction averaging). Low marginal value.

### Dropped / out of scope

- MV-PairedIntra / MCC / FJ on RVSP or standalone MR as headlines — tasks deprioritized.
- Further V4 LVH experiments — IVSD pattern is already clear.
- FullJoint-Study on single-clip tasks beyond MR — design mismatch (FJ's niche is MR and study-level aggregation).
- K=8 prediction-averaging tasks — would require new evaluation pipeline; out of scope for the NeurIPS timeline.

---

## 9. Paper claim status (what's defensible right now)

| Claim | Status | Evidence |
|---|---|---|
| V4 beats V-JEPA† on LVEF EchoNet-Dynamic at matched FLOPs (±12%) | ✅ **Defensible** | Test ΔR²=+0.053 [+0.027, +0.079], val agrees (+0.057), three selection strategies consistent |
| V4's LVEF lift decomposes into sampler (+0.024) + objective (+0.029) | ✅ **Defensible** | Both components individually significant; MV-PairedIntra as the sampler-only ablation |
| V4 beats V-JEPA† on LVH-LVEDD | ❌ **Not defensible** | Val says V4 loses; test CI touches zero; job 847 val-CI still held |
| V4 loses on LVH-IVSD — phase-relational InfoNCE degrades wall-thickness prediction | ✅ **Defensible** | ΔR²=−0.226 [−0.273, −0.179], val and test agree, large effect size |
| Token-level phase supervision (TokenRel) is strictly better than pooled (V4) on IVSD | ✅ **Defensible** | ΔR²=+0.132 [+0.097, +0.170] P=1.000 (TokenRel-e25 vs V4-e25, test N=339) |
| Token-level phase supervision fully preserves IVSD (no damage vs base) | ❌ **Not defensible** | TokenRel still loses to base by ΔR²=−0.094 [−0.132, −0.052] — less than V4 but not zero |
| V4 loses on RVSP | ✅ **Defensible** | ΔR²=−0.139 vs V-JEPA‡, 100% P(V-JEPA better) |
| V4 "multi-view" in the architectural sense | ❌ **Not defensible** | Single-view encoder; pair sampler only |
| MCC beats V4 on pediatric LVEF (R²) | ❌ **Not defensible at 95%** | Recomputed ΔR²=+0.049 [−0.015, +0.127] P=0.92 (my B=10,000 bootstrap). Earlier "+0.041 [+0.002, +0.090] ✅ P=96%" wasn't reproduced — likely different scaler/seed in source. Pearson Δ IS significant (+0.042 [+0.001, +0.090] P=0.977), but R²/MAE are not. |
| MCC matches V-JEPA†-e125 on pediatric LVEF | ✅ **Defensible** | ΔR²=+0.001 [−0.041, +0.043] P=0.51 — exact tie, no encoder-level signal either way |
| TokenRel-Motion features encode HCM signal V-JEPA lacks | ✅ **Defensible at A4C split, matched prior** | AUROC 0.760 vs 0.546, p<0.001, probe made real positive flags |
| Spatial-scale trade-off mechanism (pooled InfoNCE damages fine features) | ⚠️ **Partially supported** | LVEF+IVSD contrast strong; sampler-vs-objective isolation not yet run on IVSD |
| **FullJoint-Study leads on MR severity classification** | ❌ **Retracted** — job 823 labeled "fj_30k" used MCC checkpoint (sbatch bug). Job 858 queued with corrected FJ checkpoint; true FJ MR result TBD. Current "0.753 / 0.732" numbers are a second MCC run. |

---

## 10. References

- **Raw test numbers + bootstrap CIs**: `flops-matched-probe-results.md`
- **Phase-relational design**: `experiments/phase-relational-hardneg.md`
- **MCC design**: `experiments/masked-cross-clip-vjepa.md`
- **FullJoint-Study design**: `experiments/full-joint-global-study-token-echomv-jepa.md`
- **Stratified clinical metrics**: `reports/stratified_results/README.md`
- **Trajectory figures**: `figures/lvh/`, `figures/echonet/`, `figures/peds/`, `figures/tapse/`
- **Plotting script**: `scripts/neurips/plot_probe_trajectory_multimetric.py`
- **Bootstrap CI script**: `scripts/neurips/phase/compute_paired_bootstrap_ci.py`
- **Decomposition JSONs (local)**: `/home/sagemaker-user/tmp-boot/delta_*.json`, `/home/sagemaker-user/tmp-boot/lvef_tests/delta_*.json`, `/home/sagemaker-user/tmp-boot/ivsd_tests/delta_*.json`
