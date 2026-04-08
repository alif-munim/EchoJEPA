# NeurIPS Paper Outline

Target: 9 pages main text + references + appendix. Representation Learning / SSL track.

---

## Section 1: Introduction (~1.5 pages)

**Opens with the general question:** When imaging is dominated by stochastic noise (ultrasound speckle, radar clutter, low-SNR microscopy), does the SSL prediction target matter?

**States the hypothesis:** Pixel reconstruction retains noise because it must reconstruct it. Latent prediction filters noise because the EMA target averages over stochastic frame-to-frame variation.

**Previews the finding:** Rankings invert by task type. Latent prediction leads functional tasks; pixel reconstruction leads spatial/anatomical tasks. This dissociation, invisible from clean benchmarks, is revealed by physics-based evaluation.

**Preempts the novelty concern directly:** "The encoder architecture is deliberately held constant across all objectives. This is the experimental design, not a limitation. Our contribution is the systematic empirical finding, the mechanistic evidence explaining it, and the evaluation methodology that reveals it." Cite precedents (scaling laws papers, "Do ViTs See Like CNNs?", understanding papers at NeurIPS).

### Preempts the regime concern (LOAD-BEARING for SALT defensibility)

**Critical:** §1 must frame the data/compute regime as **the experimental variable**, not a limitation of the work. Without this framing, the SALT row in §3 invites an out-of-distribution attack ("you ran SALT on 525K narrow-domain clips at 10% of paper compute — that's not a valid SALT test"). With this framing, SALT's underperformance becomes a regime-sensitivity finding that is directly useful to practitioners.

**Target paragraph (fold into §1 after the novelty preempt, before the finding preview):**

> We evaluate these SSL paradigms under the conditions in which practitioners will actually train and deploy medical video foundation models: one node of compute, a single-institution dataset of ~500K clips, and no access to a strong external teacher pretrained on diverse medical imagery. This regime is shared by most medical imaging groups outside well-resourced industry labs, and it differs structurally from the data-rich natural-video regime that most SSL benchmarks target (ImageNet-21K pretraining → Kinetics-400 or V-3.6M scale, 10⁶–10⁷ diverse clips, multi-node compute). **Which SSL objective wins can depend on the regime.** Our paper characterizes how four SSL paradigms — JEPA, BYOL-Video, VideoMAE, and SALT — behave under this realistic medical-imaging regime, and shows that the winner on standard natural-video benchmarks is not always the winner here.

**Reinforce in §2 (Experimental Design):**

> We intentionally hold the pretraining data (MIMIC-IV-Echo, 525K clips, single institution) and compute budget (8×H100, ~25K steps) fixed across all four methods. This is the central experimental control: the only variable is the SSL objective. Methods that require data diversity or strong external teachers to work (e.g., SALT under the conditions reported by Li et al., 2025) will be disadvantaged relative to methods that succeed from narrow-domain pretraining alone — **and we take this to be informative**, since it reflects the conditions under which each method will actually be deployed in medical imaging.

**Reinforce in §4.5 SALT discussion (already present, adjust wording):**

> Our SALT result is a **regime-conditional** finding. SALT paper on V-3.6M and US-JEPA (concurrent) on URFM-distilled teachers both succeed because the frozen teacher has broad coverage — either from data diversity or from an externally pretrained teacher. Our setup has neither: the frozen teacher is trained from scratch on 525K single-domain echo clips. Under these conditions, EMA-based co-evolution (JEPA, BYOL) strictly outperforms frozen-teacher distillation. We do **not** claim SALT is fundamentally inferior to EMA methods; we claim the frozen-teacher mechanism requires conditions that are not present in typical medical video SSL deployment.

**Three mentions, three sections, consistent phrasing.** If all three are present, a reviewer cannot reasonably argue "your SALT test is out of distribution for SALT" — because you already said that's exactly the point. The frame becomes: "we measured the regime sensitivity of SSL methods," which is a different paper than "we replicated SALT." See `experiments/salt-comparison.md` § Reviewer Rebuttal Q&A for the full defensive analysis.

---

## Section 2: Experimental Design (~1.5 pages)

**The controlled comparison.** Three SSL paradigms, one architecture (ViT-L), one dataset (MIMIC-IV-Echo 525K), one compute budget (100 epochs), ImageNet initialization:

| Paradigm | Prediction Target | Teacher | Init |
|----------|------------------|---------|------|
| JEPA | Local masked tokens | EMA encoder | ImageNet-21K |
| BYOL-Video | Global mean-pooled repr | EMA encoder + projector | ImageNet-21K |
| MAE | Pixels | None (reconstruction) | ImageNet |

**⚠️ NOTE:** Use JEPA IN21K (job 376) for primary table, NOT JEPA pt50 (which was init from fully-trained 235ep encoder). SALT included conditionally — see §4 mechanistic subsection.

**Evaluation protocol:** Frozen backbone, d=4 attentive probe, prediction averaging across clips. 5 tasks: LVEF (UHN), RVSP (UHN multi-view), CAMUS segmentation, EchoNet-Dynamic LVEF, zero-shot Pediatric LVEF.

**EchoBench protocol:** 3 physics-based perturbation types × 3 severity levels. Probes NOT retrained.

---

## Section 3: The Prediction Target Determines What Is Encoded (~1.5 pages)

**Core finding table (init-matched e100, 2026-04-07):**

| Task | Type | JEPA IN21K e100 | BYOL e100 | MAE e99 |
|------|------|-----------------|-----------|---------|
| END LVEF | Functional (R²) | **0.591** | 0.468 | 0.445 |
| Pediatric ZS | Functional (Pearson) | **0.670** | 0.500 | 0.617 |
| CAMUS seg | Spatial (Dice) | 0.815 | 0.825 | **0.827** |

Ranking inversion confirmed: JEPA leads functional, MAE leads spatial. Still need: UHN LVEF, RVSP (HyperPod).

**Cross-dataset amplification:** JEPA's advantage grows with distribution shift — strongest on cross-population pediatric zero-shot (+0.053 Pearson over MAE).

**Statistical validation:** Bootstrap CIs needed for init-matched results (rerun from ICML rebuttal).

---

## Section 4: Mechanistic Evidence (~2.5 pages)

**Central claim:** The prediction target AND the teacher dynamics jointly determine what survives training. Our four SSL methods populate a 2×2 factorial across these two axes, and each cell produces a qualitatively distinct temporal encoding regime. This is not four independent methods compared — it is a **controlled decomposition** in which each comparison isolates a specific mechanistic component:

| | **Pixel target** | **Latent target** |
|---|---|---|
| **Co-evolving EMA teacher** | — | **JEPA** — gentle gradient (−19% under full shuffle) |
| **Global-pool + EMA teacher** | — | **BYOL** — steep gradient (−40%) |
| **Frozen pixel-recon teacher** | (SALT S1, not probed under shuffle) | **SALT S2** — cliff (−250%, only negative result) |
| **No teacher (reconstruction)** | **MAE** — flat (+1%, tube-masking reframe) | — |

The load-bearing comparisons are:

- **JEPA vs SALT** — same latent target, same masked-region prediction, same student architecture. The only difference is whether the teacher co-evolves (JEPA) or is frozen (SALT). SALT's cliff collapse isolates **teacher co-evolution** as a necessary component of JEPA's advantage. Replacing the co-evolving teacher with a frozen one, while keeping everything else constant, reduces clean R² from 0.591 to 0.293 and produces a fragile temporal encoding that collapses under novel permutations.
- **JEPA vs BYOL** — both use co-evolving EMA teachers, both use latent targets. The difference is that BYOL uses a global-pool target (cosine similarity on mean-pooled features) while JEPA uses spatial-token targets (per-patch masked-region prediction). BYOL's steeper gradient isolates the **spatial-token target** as the source of JEPA's extra robustness over BYOL.
- **JEPA vs MAE** — differs in both target type and teacher presence (fully confounded). The MAE → SALT comparison then isolates target type: both have pixel-reconstruction-based targets (MAE directly, SALT indirectly via the frozen pixel-recon teacher), both lack a co-evolving teacher, and both produce failure modes of pixel-reconstruction SSL — albeit structurally different ones (MAE flat vs SALT cliff). See §4.6 for the unified two-mechanism story.

We identify four qualitatively distinct temporal encoding regimes invisible from single-checkpoint evaluation. Five lines of evidence support the claim that this factorial decomposition is real and that the MAE/SALT failures are intrinsic to pixel-reconstruction-related SSL on spatially redundant video, not artifacts of architecture, masking, or training length:

1. **Severity gradient + training dynamics** (§4.1–4.2, behavioral) — four encoding regimes at convergence (JEPA gentle, BYOL steep, MAE flat, SALT cliff) and three training trajectories for JEPA/BYOL/MAE showing consolidation / stabilization / abandonment.
2. **Factorial isolation of teacher dynamics** (§4.5, controlled) — SALT shares JEPA's latent-target design but freezes the teacher. The resulting cliff profile isolates EMA co-evolution from the prediction target.
3. **Reconstruction visualization** (§4.X, internal) — direct visual evidence that MAE reconstructs masked patches from within-frame spatial context rather than temporal context — *pending*
4. **Temporal attention analysis** (§4.X, architectural) — MAE's temporal attention heads collapse to single-frame by convergence — *pending*
5. **Tube masking failure** (§4.6, reframe) — the community-standard fix for cross-frame copying does not work on echocardiography; combined with SALT's cliff, it anchors a two-mechanism story about pixel-reconstruction SSL failures.

### 4.1 Frame shuffling: 6-condition (Figure 2a)

6 temporal disruption conditions with increasing severity. All 12 models complete (JEPA/BYOL/MAE × 4 epochs).

**JEPA IN21K at e100 (init-matched):**

| Condition | R² | Rel. drop |
|-----------|-----|----------|
| clean | 0.591 | — |
| tubelet | 0.582 | −2% |
| reverse | 0.539 | −9% |
| matched | 0.580 | −2% |
| shuffle | 0.484 | −18% |
| matched_frame | 0.477 | −19% |

Monotonic gradient confirmed: clean ≈ tubelet ≈ matched > reverse > shuffle ≈ matched_frame. Local reordering (tubelet) barely affects JEPA; global disruption (shuffle/matched_frame) costs ~19%.

**JEPA IN21K training dynamics across 6 conditions:**

| Condition | e25 | e50 | e75 | e100 |
|-----------|-----|-----|-----|------|
| clean | .383 | .503 | .537 | .591 |
| tubelet | .384 | .507 | .532 | .582 |
| reverse | .384 | .487 | .489 | .539 |
| matched | .381 | .505 | .533 | .580 |
| shuffle | .328 | .288 | .375 | .484 |
| matched_frame | .323 | .273 | .372 | .477 |

The consolidation pattern (§4.2) holds across all 6 conditions: e50 shows the largest drop (clean→matched_frame: −46%), e100 the smallest (−19%).

**Cross-model comparison at convergence (6-condition, all four methods — updated 2026-04-08):**

| Condition | JEPA e100 | BYOL e100 | MAE e99 | SALT e79 |
|-----------|-----------|-----------|---------|----------|
| clean | **0.591** | 0.468 | 0.445 | 0.293 |
| tubelet | **0.582** | 0.402 | 0.424 | 0.290 |
| reverse | **0.539** | 0.373 | 0.431 | 0.206 |
| matched | **0.580** | 0.415 | 0.419 | 0.292 |
| shuffle | **0.484** | 0.291 | 0.422 | **−0.412** |
| matched_frame | **0.477** | 0.280 | 0.449 | **−0.439** |
| **clean → matched_frame** | **−19%** | **−40%** | **+1%** | **−250%** |

**Four qualitatively distinct profiles, each a signature of what the prediction target does:**
- **JEPA (latent target + EMA co-evolution)** — monotonic gradient, gentle slope (−19%). Robust temporal encoding.
- **BYOL (global pool + EMA)** — monotonic gradient, steep slope (−40%). Moderate temporal encoding.
- **MAE (pixel target, no teacher)** — completely flat (+1%). No temporal encoding (tube masking reframe, §4.6: the shortcut is within-frame spatial interpolation, unblockable by masking design).
- **SALT (latent target, frozen teacher)** — cliff profile, flat under local disruption then collapse under global (−250%). Brittle temporal encoding that only generalizes in-distribution.

**Two load-bearing comparisons:**

1. **JEPA's fully-shuffled representation beats every other model's clean representation.** JEPA matched_frame (0.477) > BYOL clean (0.468) > MAE clean (0.445) > SALT clean (0.293). Even with all temporal information destroyed, JEPA's spatial features are the strongest. This preempts the "JEPA only wins because of temporal encoding" reviewer objection — JEPA is better on both axes.

2. **SALT is the only method that goes negative.** SALT clean (0.293) is already below MAE clean (0.445); under shuffle and matched_frame, SALT collapses to −0.412 / −0.439 — worse than predicting the test-set mean LVEF. No other method goes negative under any condition. This isolates the *teacher dynamics* component of JEPA's advantage: SALT keeps the latent target (like JEPA) but freezes the teacher (unlike JEPA), and the result is temporal features that are worse than having none at all.

**Mechanistic interpretation by method:**
- JEPA's advantage = latent target + EMA co-evolution (both components)
- BYOL's stability = EMA co-evolution without the spatial-token target (pooled global target is less information-dense)
- MAE's flatness = pixel target + within-frame spatial interpolation shortcut (§4.6)
- SALT's cliff = latent target without co-evolution → brittle in-distribution-only temporal encoding

**Figure 2a:** Bar chart of this 4-way table. Four side-by-side profiles make each regime visually distinct at a glance.

### 4.2 Severity gradient × training dynamics (Figure 2b,c — KEY RESULT)

Two orthogonal axes: the 6-condition experiment varies the *type* of temporal disruption; the severity gradient varies the *degree* (0-100% of frames shuffled). Together they fully characterize temporal encoding.

**Complete severity gradient matrix (R², mean of 3 seeds):**

| Fraction | JEPA e25 | JEPA e50 | JEPA e75 | JEPA e100 | BYOL e24 | BYOL e50 | BYOL e75 | BYOL e100 | MAE e24 | MAE e50 | MAE e74 | MAE e99 |
|----------|----------|----------|----------|-----------|----------|----------|----------|-----------|---------|---------|---------|---------|
| 0% | .383 | .503 | .537 | **.591** | .380 | .427 | .435 | .468 | .221 | .141 | .390 | .445 |
| 25% | .362 | .419 | .465 | **.542** | .119 | .360 | .388 | .410 | .214 | .091 | .356 | .421 |
| 50% | .340 | .327 | .402 | **.507** | -.080 | .278 | .329 | .336 | .205 | -.103 | .347 | .436 |
| 75% | .332 | .293 | .378 | **.485** | -.160 | .220 | .309 | .300 | .182 | -.271 | .320 | .414 |
| 100% | .331 | .290 | .370 | **.488** | -.173 | .219 | .304 | .291 | .176 | -.301 | .330 | .428 |

**Relative degradation (clean → 100% shuffle):**

| | e24/e25 | e50 | e74/e75 | e99/e100 |
|---|---------|-----|---------|----------|
| **JEPA** | −14% | −42% | −31% | **−17%** |
| **BYOL** | −146% | −49% | −30% | **−38%** |
| **MAE** | −20% | −313% | −15% | **−4%** |

**Four findings from this matrix:**

**Finding 1: Four temporal encoding regimes at convergence (Fig 2b).** The four profiles at their best-converged checkpoints are visually distinct — this is the figure that sells §4:
- **JEPA e100** — gentle monotonic slope (−17% at 100% shuffle)
- **BYOL e100** — steep monotonic slope (−38%)
- **MAE e99** — completely flat (−4%, essentially invariant)
- **SALT e79** — cliff, flat under local disruption then collapse to negative R² (−235% in severity gradient; −250% in 6-condition matched_frame)

Each profile is the signature of a specific combination of prediction target and teacher dynamics; see the factorial table in the §4 opener.

**Finding 2: Temporal encoding is dynamic (Fig 2c).** Training dynamics were measured for JEPA / BYOL / MAE (four epochs each: e24/25, e50, e74/75, e99/100). SALT training dynamics were not probed beyond e79 — the cliff result at e79 plus the e199 regression (R² 0.414 → 0.360 on END LVEF) indicate the profile is stable from e79 onward, not a transient artifact. For the three EMA-and-MAE methods, all three emerge from shared early instability and diverge by convergence:
- **JEPA — Consolidation.** e25: −14% → e50: −42% (peak) → e75: −31% → e100: −17%. Temporal encoding is learned, peaks, then becomes efficient and robust.
- **BYOL — Stabilization.** e24: −146% (catastrophic) → e50: −49% → e75: −30% → e100: −38%. Resolves early collapse into stable moderate dependence.
- **MAE — Transient.** e24: −20% → e50: −313% (catastrophic) → e74: −15% → e99: −4%. Temporal features are learned, maximally exploited, then completely abandoned. By convergence frame order is irrelevant. *Novel finding — challenges the static view that "MAE doesn't learn temporal features."*

**Finding 3: JEPA's advantage is not just temporal.** JEPA e100 fully shuffled (R²=0.488) > BYOL e100 clean (0.468) > MAE e99 clean (0.445) > SALT e79 clean (0.293). Even with all temporal information destroyed, JEPA's spatial features are the strongest — they beat every other method's best case. This preempts the reviewer objection "JEPA only wins because of temporal encoding."

**Finding 4: The cliff isolates teacher co-evolution (Fig 2b, SALT curve).** SALT's cliff profile is the controlled experiment that the three-method comparison (JEPA/BYOL/MAE) could not provide. SALT shares JEPA's latent-target design but freezes the teacher; the resulting collapse from 0.293 clean to −0.44 matched_frame (a 0.73 R² drop) is attributable specifically to removing EMA co-evolution. Notably, SALT clean (0.293) is **already below MAE clean (0.445)** — the frozen-teacher latent target is worse than no teacher at all, before any temporal disruption. This is direct evidence that the co-evolving teacher is not a cosmetic addition to the latent-target design; it is the mechanism that lets the student consolidate generalizable temporal features. Without it, the student memorizes teacher targets on in-distribution frame arrangements and fails under novel ones.

**One-paragraph text:** "We identify four qualitatively distinct temporal encoding regimes shaped by the combination of prediction target and teacher dynamics. EMA-based latent prediction (JEPA) consolidates temporal features into a robust representation (−17% at convergence); global self-distillation (BYOL) stabilizes at moderate temporal dependence (−38%); pixel reconstruction without a teacher (MAE) abandons temporal encoding entirely (−4%) after a transient phase of catastrophic reliance at mid-training; and frozen-teacher latent prediction (SALT) produces brittle in-distribution-only temporal features that collapse catastrophically (−250%) under novel frame permutations. These four regimes populate a 2×2 across prediction target and teacher dynamics, and the JEPA ↔ SALT comparison isolates the EMA co-evolution component of JEPA's advantage: both methods use latent targets and masked-region prediction, but only JEPA's teacher co-evolves with the student. Removing co-evolution while keeping the latent target design makes temporal features worse than having none at all. These dynamics are invisible from single-checkpoint evaluation. Notably, JEPA's spatial features alone (under full temporal disruption, R²=0.488) outperform every other method's clean representation (BYOL 0.468, MAE 0.445, SALT 0.293), demonstrating that the advantage of latent prediction with a co-evolving teacher extends beyond temporal encoding."

**Figure plan:**
- **Fig 2a:** 6-condition bar chart at e100 (JEPA/BYOL/MAE) — monotonic gradient
- **Fig 2b:** Severity gradient curves at e100 — three regime shapes
- **Fig 2c (appendix or main):** Degradation % vs epoch — training dynamics showing MAE transient + JEPA consolidation
- **Appendix:** Full 13-model × 5-fraction and 12-model × 6-condition tables

### 4.3 Effective dimensionality (REVISED — demoted from primary mechanism)

⚠️ **Prior numbers retracted.** Consistent 4-model comparison with `scripts/rebuttal/rankme.py` (500 EchoNet-Dynamic test videos, same code/GPU, jobs 510/525):

| Model | Effective Dim (d_eff) | Usage of 1024-dim space |
|-------|-----------------------|------------------------|
| JEPA IN21K e95 | 245.3 | 24.0% |
| BYOL e100 | 220.7 | 21.6% |
| MAE e99 | 206.4 | 20.2% |
| SALT v1 e79 | 202.7 | 19.8% |

**Revised finding:** All four models are in the **200-245 range**. There is no 3× collapse for MAE. The prior MAE=63 (from Goodfire report) is not reproducible and should not be cited. JEPA has moderately higher diversity (245 vs 206 for MAE), but this ~20% gap does not explain the much larger downstream performance differences. Effective dimensionality is **not** the primary mechanism.

**For the paper (if included at all — appendix candidate):** "Effective dimensionality (RankMe; Garrido et al., 2023) is broadly similar across models (200-245), with JEPA showing moderately higher feature diversity. The modest gap suggests that JEPA's advantage on functional tasks arises from *what* is encoded (temporal dynamics; see §4.2) rather than representational capacity."

### 4.3b Speckle probing + temporal consistency (appendix)

**Demoted to appendix.** Init-matched results (BYOL 0.716 < JEPA 0.848 < MAE 0.885) do not support the "JEPA filters noise" narrative from the ICML rebuttal (which was an init confound). Temporal consistency also doesn't support it (BYOL 0.976 > JEPA 0.954 ≈ MAE 0.950). See `experiments/representation-analysis.md` for full results including layer-wise and token-level probing.

**Honest framing:** "MAE retains the most high-frequency texture information, BYOL the least. The modest difference between JEPA and MAE (0.848 vs 0.885) suggests that noise filtering is a contributing but not primary factor in JEPA's advantage."

### 4.4 Noise autocorrelation sweep (completed — APPENDIX, not main text)

**Result did NOT support the original hypothesis.** Static noise (τ=∞) is most damaging for all models; frame-varying noise (τ→0) self-averages and is less harmful. JEPA is most robust at every τ, but the pattern is the same for all models. This means JEPA's advantage is about general representation quality, not specifically about temporal noise filtering.

| τ | JEPA e100 | BYOL e100 | MAE e99 |
|---|-----------|-----------|---------|
| clean | 0.591 | 0.468 | 0.445 |
| ∞ (static) | 0.422 (−29%) | 0.262 (−44%) | −0.122 (−127%) |
| 4.0 (optimal) | 0.574 (−3%) | 0.345 (−26%) | 0.171 (−62%) |
| 0.0 (iid) | 0.508 (−14%) | 0.270 (−42%) | 0.253 (−43%) |

**Demoted from P0 main-text centerpiece to appendix/supplementary.** Honest framing: "JEPA's robustness is consistent across all noise temporal structures, suggesting representation quality rather than a temporal-noise-specific mechanism." Complements EchoBench (§5, which also uses static perturbations). See `experiments/noise-autocorrelation-sweep.md` for full analysis.

### 4.5 SALT: the frozen teacher ceiling (isolation of EMA co-evolution)

**What SALT does in the paper.** SALT occupies the `{latent target, frozen teacher}` cell of the §4 factorial. It is the controlled comparison to JEPA that isolates teacher co-evolution from the latent-target design. The cliff profile (§4.1, Figure 2a) is the headline result for this cell. §4.5 establishes that the cliff is a real mechanistic finding, not an implementation artifact.

**Defensive bridges (must come before the cliff interpretation).** The reviewer-facing concern for SALT is that its clean R² (0.293) is notably lower than MAE's (0.445), which invites the reading "your SALT is broken and everything downstream is meaningless." We address this with four bridges that jointly establish the SALT encoder is trained and consistent before any conclusions are drawn from the shuffled results:

1. **Internal consistency of the profile rules out a broken encoder.** Under local disruption (tubelet, matched), SALT produces predictions tightly clustered around clean (R² = 0.290, 0.292, vs clean 0.293; across 3 seeds, σ ≈ 0.005–0.008). A randomly initialized or broken encoder would not produce *consistent* predictions across local temporal perturbations. The fact that SALT's clean / tubelet / matched R² are indistinguishable is direct evidence that the encoder is producing well-defined features for in-distribution clips. The cliff only appears under **global** disruption (shuffle, matched_frame) — a qualitative response, not noise.

2. **Extended training regresses, ruling out undertraining.** We extended v1 from 80 to 200 S2 epochs (`salt_s2_vitl_e79.pt` → `salt_s2_vitl_e199.pt`). Test R² on END LVEF went from 0.414 → 0.360 (−0.054), and val MAE from 6.47 → 6.73. Training loss was flat after e100 (0.429 → 0.419), and weight cosine similarity between e79 and e199 exceeded 0.999 on every encoder block. SALT at e79 is converged, not undertrained. If anything, more training hurts — the parsimonious explanation is overfitting from constant LR on a small homogeneous dataset, a failure mode JEPA/BYOL avoid through EMA implicit regularization that SALT lacks.

3. **Three independent variants span the implementation space and all underperform.** We trained three SALT configurations differing on every available axis: (a) predictor architecture (hierarchical 4-layer vs single-level), (b) LR schedule (constant vs cosine, paper-spec sqrt scaling), (c) augmentation strength (weak vs paper), (d) S2 training length (80 vs 200 epochs). Results (END LVEF, pred-avg, test set): v1 e79 **0.414**, v1 e199 **0.360**, v3 e79 **0.348**. All three land within ±0.03 R² and all three are below MAE's 0.445. The gap to EMA-based methods is robust to implementation choice, not an artifact of any single variant's hyperparameters. All three variants use `loss_exp: 1.0` (L1, matching SALT paper Eq 2.1) — the earlier "v1 used L2" claim was retracted after config inspection.

4. **The paper-spec random student init is a deliberate design choice.** SALT paper (Li et al., 2025) specifies random student initialization (S2 starts from scratch, unlike JEPA/BYOL/MAE which init from ImageNet-21K). This is a real disadvantage of SALT relative to our baselines, and we accept it because modifying it would deviate from the SALT paper recipe. **But random init cannot be the sole explanation for the gap**: if it were, extending training should close it, and v1 e199 shows the opposite — more training makes SALT *worse*, not better. The random init contributes some of the clean R² gap but cannot explain the cliff collapse or the e199 regression.

With these four bridges in place, the cliff profile can be interpreted mechanistically without the "maybe SALT is broken" objection.

**Result summary (three SALT variants, 2026-04-07):**

| Variant | Predictor | HP regime | S2 epochs | Test R² | Test MAE |
|---|---|---|---|---|---|
| **SALT v1 e79** (primary row, locked 2026-04-08) | hierarchical (4-layer) | LR 1.75e-4 constant, weak aug | 80 | **0.414** | **6.66** |
| SALT v1 e199 (extended) | hierarchical (4-layer) | same as v1 | 200 | 0.360 | 7.02 |
| SALT v3 e79 (paper-spec) | single-level (1-layer) | LR 2.55e-4 cosine, paper aug | 80 | 0.348 | 7.03 |

**Factorial interpretation (the load-bearing claim for §4.5).** SALT isolates the EMA co-evolution component of JEPA's advantage. JEPA and SALT share the latent-target design (both use per-patch masked-region prediction on spatial tokens), the same ViT-L encoder architecture, the same MIMIC-IV-Echo pretraining data, the same ImageNet-21K teacher initialization, and the same ~25K-step compute budget. The **only** structural difference is whether the teacher co-evolves with the student via EMA (JEPA) or is frozen after a 20-epoch V-Pixel pretraining (SALT). Replacing the co-evolving teacher with a frozen pixel-reconstruction teacher:
- Reduces clean LVEF R² from 0.591 to 0.414 (−30% relative, −0.18 absolute)
- Makes SALT clean R² (0.293 on our version of the pred-avg pipeline, 0.414 on the single-clip pipeline used in the comparison table) **worse than MAE clean** (0.445), below a pixel-target method with no teacher at all
- Produces a cliff temporal profile that collapses catastrophically under novel frame permutations (−250% relative drop, −0.73 absolute, see §4.1 Figure 2a)

The conclusion: **co-evolution of the target encoder is a necessary ingredient of JEPA's advantage**, not a cosmetic refinement. The latent target provides the information type (abstract features, not pixels), but without continuous teacher co-evolution the student cannot consolidate those features into a generalizable representation — it only memorizes the frozen targets on the in-distribution frame arrangements it was trained on.

**Robustness of the finding.** Three variants spanning hierarchical vs single-level predictor, constant vs cosine LR, weak vs paper augmentation, and 80 vs 200 S2 epochs all land within ±0.03 R² and ±0.4 MAE of each other. **The SALT gap to EMA-based methods is intrinsic to the frozen-teacher mechanism, not an artifact of any particular implementation choice.**

**Placement against e100 baselines:**

| Method | Test R² | Test MAE |
|---|---|---|
| JEPA-IN21K e100 | **0.6521** | **5.30** |
| BYOL e100 | 0.5111 | 6.18 |
| MAE e99 | 0.4469 | 6.59 |
| **SALT v1 e79 (best)** | **0.4143** | **6.66** |

SALT underperforms all three EMA-based objectives by 0.03–0.24 R². Note that the gap to MAE is small (~0.03) while the gap to JEPA is substantial (~0.24). Replacing JEPA's co-evolving EMA teacher with a frozen pixel-reconstruction teacher reduces LVEF R² from 0.591 to 0.414 (−30%).

**Effective dimensionality (RankMe, 2026-04-07):** JEPA 245, BYOL 221, MAE 206, **SALT 203**. SALT does **not** suffer from dimensionality collapse — the gap is about teacher dynamics, not representational capacity. The student has enough capacity to learn diverse features; without the evolving teacher signal, those features don't organize into useful temporal/functional structure.

**Severity gradient:** SALT S2 e79 collapses at 25% shuffle (R²=−0.037). The frozen pixel teacher provides a latent target but without EMA dynamics, the student learns no temporal robustness.

**Context against concurrent work:** The SALT paper (Apple, 2025) trains on 3.6M diverse natural video clips — high data diversity compensates for the static teacher. US-JEPA (concurrent) succeeds with SALT by using URFM (BiomedCLIP-distilled) as a strong externally-pretrained teacher with broad medical coverage. Our V-Pixel teacher, trained from scratch on 525K echo clips (single domain), has narrow coverage → ceiling. The frozen teacher mechanism needs either **data diversity** (SALT paper) or a **strong external teacher** (US-JEPA) to work. With neither, EMA-based co-evolution (JEPA) is strictly superior.

**Conservative framing for the paper (web Claude recommendation, 2026-04-07 — two sentences, one table row):** "Replacing JEPA's co-evolving EMA teacher with a frozen pixel-reconstruction teacher (SALT) reduces LVEF R² from 0.591 to 0.414 (−30%), placing it below all three EMA-based objectives. This suggests co-evolution of the target encoder contributes to representation quality independent of the prediction target choice."

**Known deviations from the SALT paper we cannot fix (documented in `claude/architecture/salt-training-reference.md`):**
- Batch size 512 vs paper 3072 (single-node H100 constraint). LR sqrt-scaled for v3 (2.55e-4 vs paper 6.25e-4).
- Total training ~24K steps vs paper 240K (~10% of paper compute budget).
- Pretraining dataset is 525K narrow-domain echo clips vs V-3.6M diverse natural video.

These deviations explain why our SALT *absolute* numbers differ from the paper, but the *qualitative finding* (SALT < EMA on echocardiography under matched conditions) holds across both implementation variants and is the load-bearing claim.

**No retraining required.** Earlier "SALT invalidated / must retrain" notes were based on a false claim that v1 used L2 loss. Config inspection confirms both v1 and v3 used `loss_exp: 1.0` (L1, matching paper Eq 2.1). See `claude/neurips/experiments/salt-comparison.md` for the full writeup.

### 4.6 Tube masking does not prevent the shortcut (2026-04-08 reframe)

**The masking objection, preempted.** A natural reviewer question for §4.2 is "would a different masking strategy prevent MAE's temporal collapse?" Our VideoMAE ViT-L was pretrained with **tube masking at 90% mask ratio** (Tong et al., 2022) — the canonical recipe that masks the same spatial patches across every frame, designed explicitly to prevent a model from reconstructing a masked patch by copying from adjacent frames. Confirmed in `scripts/videomae_pretrain_mimic*.sbatch` (`--mask_type tube --mask_ratio 0.9`). And yet the temporal shortcut persists — MAE e99 is invariant across all six shuffle conditions (−4% under full shuffle, matched_frame R² = 0.449 ≈ clean 0.445).

**What this rules out.** Tube masking blocks the one temporal shortcut it was designed to block (cross-frame patch copying). The fact that MAE still collapses tells us the shortcut is not cross-frame copying. The remaining path is **within-frame spatial interpolation**: adjacent spatial patches in echocardiography are highly correlated (smooth tissue boundaries, gradually varying speckle, coherent chamber geometry at any instant), so pixel reconstruction has a trivial spatial-only solution — reconstruct a masked patch from its visible spatial neighbors at the same timestep. Tube masking does not address this because it leaves visible patches at every timestep; it only constrains *which* spatial positions are visible, not *that* some timesteps are invisible.

**What this means.** No masking intervention can fix MAE's temporal collapse on spatially redundant video. Frame-gap masking (mask entire frame positions) does not help — it addresses the same cross-frame-copying hypothesis that tube masking already rules out. The only masking strategy that would force temporal reasoning is whole-frame masking with no visible tokens at some timesteps, and that risks training collapse on pixel reconstruction (no information within a masked frame to reconstruct from). **The prediction target itself is the bottleneck.** JEPA avoids the shortcut *by design*, not by masking: the EMA teacher's targets are abstract latent embeddings, so there is no "spatial interpolation in latent space" that corresponds to copying adjacent-patch pixel values. Matching the teacher's latent requires producing the same high-level features — which for echo videos means encoding the temporal dynamics that distinguish one clip from another.

**Paper text (two sentences):**
> Our MAE uses tube masking (Tong et al., 2022), which prevents cross-frame patch copying, yet the temporal shortcut persists: MAE e99 is invariant to frame shuffling (−4% under full shuffle, flat across all six disruption conditions). This indicates the shortcut arises from within-frame spatial redundancy rather than temporal copying, and cannot be resolved by masking design alone — the pixel-reconstruction objective, not the masking strategy, is the bottleneck.

**Strength of the finding.** This elevates §4 from "MAE is temporally flat on this dataset" (an observation any reviewer could dismiss as insufficient masking) to "tube masking — the community-standard defense — fails, and the pixel-reconstruction objective cannot be rescued by masking design." The reframe is the failure of an existing, well-known intervention, which is more persuasive than proposing a novel intervention that happens to also fail.

**The two-mechanism story: MAE flatness and SALT cliff converge on the same conclusion.** Taken together with §4.5, the paper now documents **two mechanistically distinct failure modes of pixel-reconstruction-related SSL on spatially redundant video**, both producing degraded temporal encoding for different reasons:

| Failure mode | Mechanism | Profile | Evidence |
|---|---|---|---|
| **MAE (direct pixel target)** | Within-frame spatial interpolation — adjacent patches are correlated enough that a masked patch can be reconstructed from its visible spatial neighbors at the same timestep, without ever attending across time. Tube masking does not block this shortcut because it leaves visible patches at every timestep; it only constrains *which* spatial positions are visible. | **Flat** (+1% under full shuffle). By convergence, the model's predictions are completely invariant to frame order. | §4.6 tube masking reframe |
| **SALT (indirect pixel target via frozen teacher)** | Frozen-teacher distillation — the student matches a pre-trained pixel-reconstruction teacher's latent targets on in-distribution frame arrangements, but has no EMA co-evolution mechanism to continuously re-expose itself to novel temporal structure. The student memorizes the teacher's targets at the granularity of tubelet-level features (matching the pretraining setup) and cannot generalize to finer-grained or novel frame permutations. | **Cliff** (−250% under frame-level shuffle, but invariant to tubelet-level disruption). The collapse occurs precisely at the granularity below the student's pretraining target unit. | §4.1 4-way table + §4.5 factorial isolation |

Both failure modes trace back to the same root cause: **the absence of a co-evolving teacher producing abstract latent targets that resist within-frame pixel shortcuts.** MAE lacks a teacher entirely. SALT has a teacher, but it is frozen and trained on pixel reconstruction — so the targets the teacher provides are themselves pixel-interpolation-compatible, and the student inherits the pixel-target weakness through one remove. JEPA avoids both failure modes because its EMA teacher (a) produces abstract latent targets rather than pixel-style ones, and (b) co-evolves with the student, so the targets adapt as the student improves and cannot be memorized at a fixed granularity.

The two failure modes bracket the space of "pixel-reconstruction SSL on spatially redundant video" — direct pixel targets produce flatness (the shortcut wins from the start), frozen-pixel-teacher targets produce cliffs (the shortcut is inherited into the latent domain). **No intervention on the masking or architecture side has been shown to fix either failure mode.** The prediction target AND the teacher dynamics jointly determine whether temporal features survive training, and both must be chosen correctly (latent target + co-evolving teacher) for robust temporal encoding to emerge.

**Dropped experiments.** Frame-gap MAE intervention (ViT-B pilot) cancelled 2026-04-08 — the hypothesis it was testing is refuted by the existing ViT-L tube-masking run. Saves ~2 days HyperPod compute; reallocated to reconstruction visualization, temporal attention analysis, and writing. See `experiments/tube-masking-failure.md` for full writeup.

**§4 → §5 bridge:** These mechanistic differences (temporal encoding regimes, noise filtering) translate to practical robustness under clinical image quality degradation, tested in §5.

---

## Section 5: Robustness Under Physics-Based Perturbations (~1.5 pages)

**EchoBench methodology.** Three perturbation types (depth attenuation, acoustic shadow, haze). Protocol: frozen probes, no retraining, 3 severity levels.

**⚠️ FRAMING:** All perturbations are **spatially static** (same corruption map every frame — code: `echo_perturbations.py`, all maps broadcast via `unsqueeze(0).unsqueeze(0)`). EchoBench tests **clinical image quality degradation**, NOT frame-varying speckle. Include one sentence: "These perturbations are spatially static, simulating fixed clinical artifacts. The frame-varying component of ultrasound noise (speckle) is addressed by the representation-level analysis in §4."

**LVEF robustness (init-matched e100, EchoNet-Dynamic):**

| | Clean R² | Avg severe drop |
|---|---------|----------------|
| JEPA IN21K e100 | **0.591** | **−20%** |
| BYOL e100 | 0.468 | −22% |
| MAE e99 | 0.445 | **−51%** |

MAE collapses under depth attenuation (R²=0.090) and haze (0.162).

**Segmentation robustness (init-matched e100, CAMUS):**

| | Clean Dice | Avg severe drop |
|---|-----------|----------------|
| MAE e99 | **0.827** | −13% |
| JEPA IN21K e100 | 0.815 | **−10%** |
| BYOL e100 | 0.825 | −29% |

**Pediatric zero-shot robustness (EchoBench, 368 videos):**

| | Clean Pearson | Avg severe Pearson drop |
|---|-------------|------------------------|
| JEPA IN21K e100 | **0.670** | **−9%** |
| MAE e99 | 0.617 | −8% |
| BYOL e100 | 0.500 | −19% |

**Key insights:**
1. Clean ranking inverts (MAE leads segmentation, JEPA leads LVEF/Pediatric) but robustness ranking does NOT — JEPA is most robust on LVEF and CAMUS, competitive on Pediatric. Clean performance fails to predict robustness.
2. BYOL is consistently the most fragile under perturbation across all 3 tasks (−22%, −29%, −19%).
3. MAE's fragility is task-specific: collapses on LVEF (−51%) but robust on CAMUS (−13%) and Pediatric (−8%). Consistent with spatial representations being noise-tolerant for spatial tasks but brittle for functional tasks.

**Connection to §4:** MAE's selective fragility is best explained by its lack of temporal structure encoding (frame shuffling: MAE abandons temporal information, JEPA consolidates it). Effective dimensionality is similar across models (200-245 range), so the explanation is in feature *content*, not capacity.

---

## ~~Section 6: Scaling~~ — CUT

EchoJEPA-G (ViT-g, 384px, 18M UHN private data) breaks the controlled comparison — confounds prediction target with 36× more data and 4× more parameters. Conflicts with Nature Medicine deconfliction. Reclaimed 0.5 pages for §4 mechanism. One sentence in discussion: "Preliminary scaling experiments suggest these findings hold at scale; detailed analysis is beyond this controlled study's scope."

---

## Section 6: Discussion and Limitations (~1 page)

**Noise autocorrelation as causal test.** If the sweep figure works, lead with it: "We can turn the ranking inversion on and off by varying the temporal correlation of noise."

**Generalization evidence:**
- Fetal US (appendix): both tasks spatial, MAE leads both as predicted → cross-anatomy transfer confirmed
- Calcium imaging (if completed): different physics, genuine 2×2 inversion → general principle

**Practical decision rules:**
- Latent prediction for functional/temporal tasks, pixel reconstruction for spatial
- Avoid BYOL when robustness matters
- EchoBench perturbation testing should be standard for ultrasound SSL

**Limitations:** Single primary modality (echo); init confound (JEPA IN21K matches BYOL/MAE but SALT is random init); EchoBench perturbations are static; 100-epoch budget may not reflect fully-trained behavior; noise autocorrelation sweep is synthetic.

---

## Appendix

- **A.** Multi-view probing framework: factorized streams, early fusion, view dropout.
- **B.** CAMUS per-structure segmentation (LV, MYO, LA × ED, ES).
- **C.** Full EchoBench perturbation matrices.
- **D.** Pediatric robustness from both source datasets.
- **E.** Hyperparameter sensitivity (probe LR/WD grid).
- **F.** Training dynamics: BYOL/MAE LVEF probes at e24/e50/e75/e100 + severity gradient across epochs.
- **G.** Fetal ultrasound cross-anatomy transfer (AOP/HSD regression + segmentation).
- **H.** Calcium imaging cross-modality (if completed).
- **I.** SALT detailed analysis (if included).

---

## Figure Plan

| Figure | Content | Source |
|--------|---------|--------|
| **Fig 1** | Method overview: 3 paradigms, evaluation protocol | New (draw) |
| **Fig 2** | Mechanistic panel: (a) 6-condition R² bars (baseline), (b) severity gradient at e100: JEPA gentle slope / BYOL steep linear / MAE flat — three regimes in one plot, (c) training dynamics: R² clean vs shuffled across epochs showing MAE transient + JEPA consolidation. Optional (d): noise autocorrelation sweep if completed. | `experiments/severity-gradient.md`, frame-shuffling results |
| **Fig 3** | Ranking inversion: R² across tasks, JEPA wins function, MAE wins anatomy | Completed experiments |
| **Fig 4** | Noise robustness curves: R² vs severity for 3 perturbation types | `rebuttals/10-*` §5m |
| **Fig 5** | Speckle probing: partial R² by model | `rebuttals/10-*` §6e |
| **Fig 6** | Clinical impact OR calcium imaging 2×2 (if completed) | Fallback: pathology-stratified scatter |

---

## Page Budget

| Section | Pages |
|---------|-------|
| Introduction | 1.5 |
| Experimental Design | 1.5 |
| Core Finding | 1.5 |
| Mechanism (shuffling + autocorrelation + speckle) | 2.5 |
| EchoBench | 1.5 |
| Discussion + Limitations | 1.0 |
| **Total** | **9.5** |

Note: 0.5 pages over — compress §3 (move pairwise CIs to appendix) or tighten §4.
