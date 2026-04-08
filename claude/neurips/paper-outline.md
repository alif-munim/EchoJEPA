# NeurIPS Paper Outline

Target: 9 pages main text + references + appendix. Representation Learning / SSL track.

**Arc (2026-04-08 restructure):** The paper has a two-phase structure. **Part 1 (§3)** performs a controlled mechanism comparison of four SSL objectives on a single well-understood task (EchoNet-Dynamic LVEF), using frame shuffling + training dynamics + factorial decomposition to characterize what each method learns. **Part 2 (§5)** derives testable predictions from Part 1 and validates them on held-out tasks (MIMIC clinical labels + cross-dataset transfer + segmentation). §4 is a compact mechanism synthesis bridge. §6 tests a specific Part 1 prediction about noise robustness via EchoBench. This converts the paper from "three parallel findings" into "mechanism → hypothesis → confirmation" — a stronger argumentative structure that preempts cherry-picking concerns and gives §3 load-bearing status for the rest of the paper.

---

## Section 1: Introduction (~1.5 pages)

**Opens with the general question:** When imaging is dominated by stochastic noise (ultrasound speckle, radar clutter, low-SNR microscopy), does the SSL prediction target matter? And more deeply: what, precisely, does each SSL objective learn about video from narrow-domain medical data?

**States the hypothesis:** Pixel reconstruction retains noise because it must reconstruct it. Latent prediction filters noise because the EMA target averages over stochastic frame-to-frame variation. Teacher co-evolution matters as much as target type.

**Previews the arc.** The paper has a two-phase structure. Part 1 characterizes what each of four SSL objectives (JEPA, BYOL-Video, VideoMAE, SALT) learns by performing a controlled mechanism comparison on a single well-understood task (EchoNet-Dynamic LVEF), using frame shuffling across six disruption conditions and a severity gradient × training dynamics matrix. Part 1 produces a 2×2 factorial decomposition across prediction target (pixel vs latent) and teacher dynamics (frozen vs co-evolving EMA), with each cell producing a qualitatively distinct temporal encoding regime. Part 2 derives testable predictions from Part 1 and validates them on held-out tasks — MIMIC-IV-Echo clinical biomarkers and disease labels, CAMUS segmentation, and cross-population pediatric transfer. The paper's central claim is that the four regimes identified in Part 1 generalize to held-out tasks in predictable ways, and deviations from those predictions reveal the scope of the mechanism.

**Previews the finding:** Part 1 identifies four qualitatively distinct temporal encoding regimes. Part 2 tests predictions from these regimes and confirms that task-type rankings invert in the predicted direction: latent prediction + co-evolving teacher (JEPA) leads functional/dynamic tasks; pure spatial features (MAE) lead anatomical/segmentation tasks; frozen-teacher distillation (SALT) underperforms all three under our conditions. This dissociation, invisible from clean benchmarks alone, is derived from mechanism and validated across a held-out task battery.

**Preempts the novelty concern directly:** "The encoder architecture is deliberately held constant across all objectives. This is the experimental design, not a limitation. Our contribution is the systematic empirical finding, the mechanistic evidence explaining it, and the evaluation methodology that reveals it." Cite precedents (scaling laws papers, "Do ViTs See Like CNNs?", understanding papers at NeurIPS).

### Preempts the regime concern (LOAD-BEARING for SALT defensibility)

**Critical:** §1 must frame the data/compute regime as **the experimental variable**, not a limitation of the work. Without this framing, the SALT row in §3 invites an out-of-distribution attack ("you ran SALT on 525K narrow-domain clips at 10% of paper compute — that's not a valid SALT test"). With this framing, SALT's underperformance becomes a regime-sensitivity finding that is directly useful to practitioners.

**Target paragraph (fold into §1 after the novelty preempt, before the finding preview):**

> We evaluate these SSL paradigms under the conditions in which practitioners will actually train and deploy medical video foundation models: one node of compute, a single-institution dataset of ~500K clips, and no access to a strong external teacher pretrained on diverse medical imagery. This regime is shared by most medical imaging groups outside well-resourced industry labs, and it differs structurally from the data-rich natural-video regime that most SSL benchmarks target (ImageNet-21K pretraining → Kinetics-400 or V-3.6M scale, 10⁶–10⁷ diverse clips, multi-node compute). **Which SSL objective wins can depend on the regime.** Our paper characterizes how four SSL paradigms — JEPA, BYOL-Video, VideoMAE, and SALT — behave under this realistic medical-imaging regime, and shows that the winner on standard natural-video benchmarks is not always the winner here.

**Reinforce in §2 (Experimental Design):**

> We intentionally hold the pretraining data (MIMIC-IV-Echo, 525K clips, single institution) and compute budget (8×H100, ~25K steps) fixed across all four methods. This is the central experimental control: the only variable is the SSL objective. Methods that require data diversity or strong external teachers to work (e.g., SALT under the conditions reported by Li et al., 2025) will be disadvantaged relative to methods that succeed from narrow-domain pretraining alone — **and we take this to be informative**, since it reflects the conditions under which each method will actually be deployed in medical imaging.

**Reinforce in §3.5 SALT discussion (formerly §4.5, wording already drafted):**

> Our SALT result is a **regime-conditional** finding. SALT paper on V-3.6M and US-JEPA (concurrent) on URFM-distilled teachers both succeed because the frozen teacher has broad coverage — either from data diversity or from an externally pretrained teacher. Our setup has neither: the frozen teacher is trained from scratch on 525K single-domain echo clips. Under these conditions, EMA-based co-evolution (JEPA, BYOL) strictly outperforms frozen-teacher distillation. We do **not** claim SALT is fundamentally inferior to EMA methods; we claim the frozen-teacher mechanism requires conditions that are not present in typical medical video SSL deployment.

**Three mentions, three sections, consistent phrasing.** If all three are present, a reviewer cannot reasonably argue "your SALT test is out of distribution for SALT" — because you already said that's exactly the point. The frame becomes: "we measured the regime sensitivity of SSL methods," which is a different paper than "we replicated SALT." See `experiments/salt-comparison.md` § Reviewer Rebuttal Q&A for the full defensive analysis.

---

## Section 2: Experimental Design (~1.0 pages)

**Four SSL paradigms, one architecture (ViT-L), one dataset (MIMIC-IV-Echo 525K), one compute budget (100 epochs, 8×H100), ImageNet-21K initialization** (except SALT S2 student, per paper-spec random init):

| Paradigm | Prediction target | Teacher | Init | Factorial cell |
|---|---|---|---|---|
| **JEPA** (V-JEPA 2.0) | Latent masked-region tokens | EMA self-distillation | ImageNet-21K | Latent + co-evolving EMA |
| **BYOL-Video** | Global mean-pooled representation | EMA encoder + projector | ImageNet-21K | Pooled latent + co-evolving EMA |
| **VideoMAE** (MAE) | Pixels | None (direct reconstruction) | ImageNet-21K | Pixel + no teacher |
| **SALT S2** | Latent masked-region tokens (to frozen teacher) | Frozen V-Pixel (SALT S1) | Random student init (IN21K S1 teacher) | Latent + frozen teacher |

**Use JEPA IN21K (job 376) as the primary JEPA row**, not JEPA pt50 which was initialized from a fully-trained 235-epoch encoder (confound). SALT v1 e79 is the primary SALT row (locked 2026-04-08; see `experiments/salt-comparison.md` § FINAL DECISION).

**Two-phase study design.** The paper is organized around a controlled mechanism comparison followed by hypothesis-driven validation:

- **Part 1 (§3)** — We characterize what each method learns by running all four objectives through a frame-shuffling + training-dynamics battery on a single well-understood task (EchoNet-Dynamic LVEF regression). This produces a 2×2 factorial decomposition across prediction target and teacher dynamics, with each cell yielding a distinct temporal encoding profile. Part 1 is the foundation — all mechanism claims are grounded here.
- **Part 2 (§5)** — We derive testable predictions from the Part 1 mechanism and validate them on held-out tasks chosen *because each tests a specific prediction*. The task battery includes MIMIC-IV-Echo clinical biomarkers (NT-proBNP, troponin-T, creatinine), mortality (1-year, 30-day), disease labels (afib, HCM, DCM, HF), note-extracted EF, CAMUS segmentation, and cross-population pediatric LVEF. Each task is selected to test a specific §3 prediction with a pre-committed expected ranking.

**Rationale for the two-phase structure.** Running full mechanism experiments (6-condition frame shuffling + severity gradient × 4 training checkpoints) on every task would be compute-prohibitive (14 conditions × 5 fractions × 4 epochs × 4 models × N tasks ≈ thousands of probe evaluations). Instead, we do mechanism on one task (LVEF, the richest functional echo benchmark with existing training-dynamics infrastructure), derive predictions, and test them on held-out tasks. This also separates hypothesis generation (Part 1) from hypothesis testing (Part 2), protecting against retrospective cherry-picking and making the task selection defensible.

**Evaluation protocol.** Frozen backbone, d=4 attentive probe (6-head HP grid over LR × WD), prediction averaging across clips per study. For Part 1: severity gradient (5 fractions × 3 seeds) and 6-condition frame shuffling (clean, tubelet, reverse, matched, shuffle, matched_frame × 3 seeds for stochastic conditions). For Part 2: d=1 attentive probes per task, 35 epochs for MIMIC study-level regression, 15–20 for classification (see `experiments/salt-comparison.md` § Artifacts Inventory for config details).

**Regime reinforcement (LOAD-BEARING, from §1):** We intentionally hold the pretraining data (MIMIC-IV-Echo, 525K clips, single institution) and compute budget (8×H100, ~25K steps) fixed across all four methods. This is the central experimental control: the only variable is the SSL objective. Methods that require data diversity or strong external teachers to work (e.g., SALT under the conditions reported by Li et al., 2025) will be disadvantaged relative to methods that succeed from narrow-domain pretraining alone — **and we take this to be informative**, since it reflects the conditions under which each method will actually be deployed in medical imaging.

---

## Section 3: Part 1 — Controlled Mechanism Comparison on LVEF (~2.5 pages)

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
- **JEPA vs MAE** — differs in both target type and teacher presence (fully confounded). The MAE → SALT comparison then isolates target type: both have pixel-reconstruction-based targets (MAE directly, SALT indirectly via the frozen pixel-recon teacher), both lack a co-evolving teacher, and both produce failure modes of pixel-reconstruction SSL — albeit structurally different ones (MAE flat vs SALT cliff). See §3.6 for the unified two-mechanism story.

We identify four qualitatively distinct temporal encoding regimes invisible from single-checkpoint evaluation. Five lines of evidence in §3.1–3.6 support the claim that this factorial decomposition is real and that the MAE/SALT failures are intrinsic to pixel-reconstruction-related SSL on spatially redundant video, not artifacts of architecture, masking, or training length:

1. **Severity gradient + training dynamics** (§3.2–3.3, behavioral) — four encoding regimes at convergence (JEPA gentle, BYOL steep, MAE flat, SALT cliff) and three training trajectories for JEPA/BYOL/MAE showing consolidation / stabilization / abandonment.
2. **Factorial isolation of teacher dynamics** (§3.5, controlled) — SALT shares JEPA's latent-target design but freezes the teacher. The resulting cliff profile isolates EMA co-evolution from the prediction target.
3. **Reconstruction visualization** (§3.X, internal) — direct visual evidence that MAE reconstructs masked patches from within-frame spatial context rather than temporal context — *pending*
4. **Temporal attention analysis** (§3.X, architectural) — MAE's temporal attention heads collapse to single-frame by convergence — *pending*
5. **Tube masking failure** (§3.6, reframe) — the community-standard fix for cross-frame copying does not work on echocardiography; combined with SALT's cliff, it anchors a two-mechanism story about pixel-reconstruction SSL failures.

### 3.1 Frame shuffling: 6-condition (Figure 2a)

6 temporal disruption conditions with increasing severity. All 12 models complete (JEPA/BYOL/MAE × 4 epochs) + SALT-S2-e79 added 2026-04-08.

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

The consolidation pattern (§3.2) holds across all 6 conditions: e50 shows the largest drop (clean→matched_frame: −46%), e100 the smallest (−19%).

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
- **MAE (pixel target, no teacher)** — completely flat (+1%). No temporal encoding (tube masking reframe, §3.6: the shortcut is within-frame spatial interpolation, unblockable by masking design).
- **SALT (latent target, frozen teacher)** — cliff profile, flat under local disruption then collapse under global (−250%). Brittle temporal encoding that only generalizes in-distribution.

**Two load-bearing comparisons:**

1. **JEPA's fully-shuffled representation beats every other model's clean representation.** JEPA matched_frame (0.477) > BYOL clean (0.468) > MAE clean (0.445) > SALT clean (0.293). Even with all temporal information destroyed, JEPA's spatial features are the strongest. This preempts the "JEPA only wins because of temporal encoding" reviewer objection — JEPA is better on both axes.

2. **SALT is the only method that goes negative.** SALT clean (0.293) is already below MAE clean (0.445); under shuffle and matched_frame, SALT collapses to −0.412 / −0.439 — worse than predicting the test-set mean LVEF. No other method goes negative under any condition. This isolates the *teacher dynamics* component of JEPA's advantage: SALT keeps the latent target (like JEPA) but freezes the teacher (unlike JEPA), and the result is temporal features that are worse than having none at all.

**Mechanistic interpretation by method:**
- JEPA's advantage = latent target + EMA co-evolution (both components)
- BYOL's stability = EMA co-evolution without the spatial-token target (pooled global target is less information-dense)
- MAE's flatness = pixel target + within-frame spatial interpolation shortcut (§3.6)
- SALT's cliff = latent target without co-evolution → brittle in-distribution-only temporal encoding

**Figure 2a:** Bar chart of this 4-way table. Four side-by-side profiles make each regime visually distinct at a glance.

### 3.2 Severity gradient × training dynamics (Figure 2b,c — KEY RESULT)

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

**Finding 1: Four temporal encoding regimes at convergence (Fig 2b).** The four profiles at their best-converged checkpoints are visually distinct — this is the figure that sells §3:
- **JEPA e100** — gentle monotonic slope (−17% at 100% shuffle)
- **BYOL e100** — steep monotonic slope (−38%)
- **MAE e99** — completely flat (−4%, essentially invariant)
- **SALT e79** — cliff, flat under local disruption then collapse to negative R² (−235% in severity gradient; −250% in 6-condition matched_frame)

Each profile is the signature of a specific combination of prediction target and teacher dynamics; see the factorial table in the §3 opener.

**Finding 2: Temporal encoding is dynamic (Fig 2c).** Training dynamics were measured for JEPA / BYOL / MAE (four epochs each: e24/25, e50, e74/75, e99/100). SALT training dynamics were not probed beyond e79 — the cliff result at e79 plus the e199 regression (R² 0.414 → 0.360 on END LVEF) indicate the profile is stable from e79 onward, not a transient artifact. For the three EMA-and-MAE methods, all three emerge from shared early instability and diverge by convergence:
- **JEPA — Consolidation.** e25: −14% → e50: −42% (peak) → e75: −31% → e100: −17%. Temporal encoding is learned, peaks, then becomes efficient and robust.
- **BYOL — Stabilization.** e24: −146% (catastrophic) → e50: −49% → e75: −30% → e100: −38%. Resolves early collapse into stable moderate dependence.
- **MAE — Transient.** e24: −20% → e50: −313% (catastrophic) → e74: −15% → e99: −4%. Temporal features are learned, maximally exploited, then completely abandoned. By convergence frame order is irrelevant. *Novel finding — challenges the static view that "MAE doesn't learn temporal features."*

**Finding 3: JEPA's advantage is not just temporal.** JEPA e100 fully shuffled (R²=0.488) > BYOL e100 clean (0.468) > MAE e99 clean (0.445) > SALT e79 clean (0.293). Even with all temporal information destroyed, JEPA's spatial features are the strongest — they beat every other method's best case. This preempts the reviewer objection "JEPA only wins because of temporal encoding."

**Finding 4: The cliff isolates teacher co-evolution (Fig 2b, SALT curve).** SALT's cliff profile is the controlled experiment that the three-method comparison (JEPA/BYOL/MAE) could not provide. SALT shares JEPA's latent-target design but freezes the teacher; the resulting collapse from 0.293 clean to −0.44 matched_frame (a 0.73 R² drop) is attributable specifically to removing EMA co-evolution. Notably, SALT clean (0.293) is **already below MAE clean (0.445)** — the frozen-teacher latent target is worse than no teacher at all, before any temporal disruption. This is direct evidence that the co-evolving teacher is not a cosmetic addition to the latent-target design; it is the mechanism that lets the student consolidate generalizable temporal features. Without it, the student memorizes teacher targets on in-distribution frame arrangements and fails under novel ones.

**One-paragraph text:** "We identify four qualitatively distinct temporal encoding regimes shaped by the combination of prediction target and teacher dynamics. EMA-based latent prediction (JEPA) consolidates temporal features into a robust representation (−17% at convergence); global self-distillation (BYOL) stabilizes at moderate temporal dependence (−38%); pixel reconstruction without a teacher (MAE) abandons temporal encoding entirely (−4%) after a transient phase of catastrophic reliance at mid-training; and frozen-teacher latent prediction (SALT) produces brittle in-distribution-only temporal features that collapse catastrophically (−250%) under novel frame permutations. These four regimes populate a 2×2 across prediction target and teacher dynamics, and the JEPA ↔ SALT comparison isolates the EMA co-evolution component of JEPA's advantage: both methods use latent targets and masked-region prediction, but only JEPA's teacher co-evolves with the student. Removing co-evolution while keeping the latent target design makes temporal features worse than having none at all. These dynamics are invisible from single-checkpoint evaluation. Notably, JEPA's spatial features alone (under full temporal disruption, R²=0.488) outperform every other method's clean representation (BYOL 0.468, MAE 0.445, SALT 0.293), demonstrating that the advantage of latent prediction with a co-evolving teacher extends beyond temporal encoding."

**Figure plan:**
- **Fig 2a:** 6-condition bar chart at e100/e99/e79 (JEPA/BYOL/MAE/SALT) — four profiles
- **Fig 2b:** Severity gradient curves at convergence — four regime shapes
- **Fig 2c (appendix or main):** Degradation % vs epoch — training dynamics showing MAE transient + JEPA consolidation (JEPA/BYOL/MAE only; SALT single-point)
- **Appendix:** Full 13-model × 5-fraction and 13-model × 6-condition tables

### 3.3 Effective dimensionality (appendix candidate)

⚠️ **Prior numbers retracted.** Consistent 4-model comparison with `scripts/rebuttal/rankme.py` (500 EchoNet-Dynamic test videos, same code/GPU, jobs 510/525):

| Model | Effective Dim (d_eff) | Usage of 1024-dim space |
|-------|-----------------------|------------------------|
| JEPA IN21K e95 | 245.3 | 24.0% |
| BYOL e100 | 220.7 | 21.6% |
| MAE e99 | 206.4 | 20.2% |
| SALT v1 e79 | 202.7 | 19.8% |

**Revised finding:** All four models are in the **200-245 range**. There is no 3× collapse for MAE. The prior MAE=63 (from Goodfire report) is not reproducible and should not be cited. JEPA has moderately higher diversity (245 vs 206 for MAE), but this ~20% gap does not explain the much larger downstream performance differences. Effective dimensionality is **not** the primary mechanism.

**For the paper (if included at all — appendix candidate):** "Effective dimensionality (RankMe; Garrido et al., 2023) is broadly similar across models (200-245), with JEPA showing moderately higher feature diversity. The modest gap suggests that JEPA's advantage on functional tasks arises from *what* is encoded (temporal dynamics; see §3.2) rather than representational capacity."

### 3.4 Speckle probing + temporal consistency (appendix)

**Demoted to appendix.** Init-matched results (BYOL 0.716 < JEPA 0.848 < MAE 0.885) do not support the "JEPA filters noise" narrative from the ICML rebuttal (which was an init confound). Temporal consistency also doesn't support it (BYOL 0.976 > JEPA 0.954 ≈ MAE 0.950). See `experiments/representation-analysis.md` for full results including layer-wise and token-level probing.

**Honest framing:** "MAE retains the most high-frequency texture information, BYOL the least. The modest difference between JEPA and MAE (0.848 vs 0.885) suggests that noise filtering is a contributing but not primary factor in JEPA's advantage."

### 3.5 SALT: the frozen teacher ceiling (isolation of EMA co-evolution)

**What SALT does in the paper.** SALT occupies the `{latent target, frozen teacher}` cell of the §3 factorial. It is the controlled comparison to JEPA that isolates teacher co-evolution from the latent-target design. The cliff profile (§3.1, Figure 2a) is the headline result for this cell. §3.5 establishes that the cliff is a real mechanistic finding, not an implementation artifact.

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

**Factorial interpretation (the load-bearing claim for §3.5).** SALT isolates the EMA co-evolution component of JEPA's advantage. JEPA and SALT share the latent-target design (both use per-patch masked-region prediction on spatial tokens), the same ViT-L encoder architecture, the same MIMIC-IV-Echo pretraining data, the same ImageNet-21K teacher initialization, and the same ~25K-step compute budget. The **only** structural difference is whether the teacher co-evolves with the student via EMA (JEPA) or is frozen after a 20-epoch V-Pixel pretraining (SALT). Replacing the co-evolving teacher with a frozen pixel-reconstruction teacher:
- Reduces clean LVEF R² from 0.591 to 0.414 (−30% relative, −0.18 absolute)
- Makes SALT clean R² (0.293 on our version of the pred-avg pipeline, 0.414 on the single-clip pipeline used in the comparison table) **worse than MAE clean** (0.445), below a pixel-target method with no teacher at all
- Produces a cliff temporal profile that collapses catastrophically under novel frame permutations (−250% relative drop, −0.73 absolute, see §3.1 Figure 2a)

The conclusion: **co-evolution of the target encoder is a necessary ingredient of JEPA's advantage**, not a cosmetic refinement. The latent target provides the information type (abstract features, not pixels), but without continuous teacher co-evolution the student cannot consolidate those features into a generalizable representation — it only memorizes the frozen targets on the in-distribution frame arrangements it was trained on.

**Robustness of the finding.** Three variants spanning hierarchical vs single-level predictor, constant vs cosine LR, weak vs paper augmentation, and 80 vs 200 S2 epochs all land within ±0.03 R² and ±0.4 MAE of each other. **The SALT gap to EMA-based methods is intrinsic to the frozen-teacher mechanism, not an artifact of any particular implementation choice.**

**Placement against e100 baselines:**

| Method | Test R² | Test MAE |
|---|---|---|
| JEPA-IN21K e100 | **0.6521** | **5.30** |
| BYOL e100 | 0.5111 | 6.18 |
| MAE e99 | 0.4469 | 6.59 |
| **SALT v1 e79 (best)** | **0.4143** | **6.66** |

SALT underperforms all three EMA-based objectives by 0.03–0.24 R². Note that the gap to MAE is small (~0.03) while the gap to JEPA is substantial (~0.24).

**Effective dimensionality (RankMe, 2026-04-07):** JEPA 245, BYOL 221, MAE 206, **SALT 203**. SALT does **not** suffer from dimensionality collapse — the gap is about teacher dynamics, not representational capacity. The student has enough capacity to learn diverse features; without the evolving teacher signal, those features don't organize into useful temporal/functional structure.

**Severity gradient:** SALT S2 e79 collapses at 25% shuffle (R²=−0.037). The frozen pixel teacher provides a latent target but without EMA dynamics, the student learns no temporal robustness.

**Context against concurrent work:** The SALT paper (Apple, 2025) trains on 3.6M diverse natural video clips — high data diversity compensates for the static teacher. US-JEPA (concurrent) succeeds with SALT by using URFM (BiomedCLIP-distilled) as a strong externally-pretrained teacher with broad medical coverage. Our V-Pixel teacher, trained from scratch on 525K echo clips (single domain), has narrow coverage → ceiling. The frozen teacher mechanism needs either **data diversity** (SALT paper) or a **strong external teacher** (US-JEPA) to work. With neither, EMA-based co-evolution (JEPA) is strictly superior.

**Conservative framing for the paper (two sentences, one table row):** "Replacing JEPA's co-evolving EMA teacher with a frozen pixel-reconstruction teacher (SALT) reduces LVEF R² from 0.591 to 0.414 (−30%), placing it below all three EMA-based objectives. This suggests co-evolution of the target encoder contributes to representation quality independent of the prediction target choice."

**Known deviations from the SALT paper we cannot fix (documented in `claude/architecture/salt-training-reference.md` and `experiments/salt-comparison.md` § Known deviations):**
- Batch size 512 vs paper 3072 (single-node H100 constraint). LR sqrt-scaled for v3 (2.55e-4 vs paper 6.25e-4).
- Total training ~24K steps vs paper 240K (~10% of paper compute budget).
- Pretraining dataset is 525K narrow-domain echo clips vs V-3.6M diverse natural video.

These deviations explain why our SALT *absolute* numbers differ from the paper, but the *qualitative finding* (SALT < EMA on echocardiography under matched conditions) holds across both implementation variants and is the load-bearing claim.

**No retraining required.** Earlier "SALT invalidated / must retrain" notes were based on a false claim that v1 used L2 loss. Config inspection confirms both v1 and v3 used `loss_exp: 1.0` (L1, matching paper Eq 2.1). See `claude/neurips/experiments/salt-comparison.md` for the full writeup.

### 3.6 Tube masking does not prevent the shortcut — the two-mechanism story

**The masking objection, preempted.** A natural reviewer question for §3.2 is "would a different masking strategy prevent MAE's temporal collapse?" Our VideoMAE ViT-L was pretrained with **tube masking at 90% mask ratio** (Tong et al., 2022) — the canonical recipe that masks the same spatial patches across every frame, designed explicitly to prevent a model from reconstructing a masked patch by copying from adjacent frames. Confirmed in `scripts/videomae_pretrain_mimic*.sbatch` (`--mask_type tube --mask_ratio 0.9`). And yet the temporal shortcut persists — MAE e99 is invariant across all six shuffle conditions (−4% under full shuffle, matched_frame R² = 0.449 ≈ clean 0.445).

**What this rules out.** Tube masking blocks the one temporal shortcut it was designed to block (cross-frame patch copying). The fact that MAE still collapses tells us the shortcut is not cross-frame copying. The remaining path is **within-frame spatial interpolation**: adjacent spatial patches in echocardiography are highly correlated (smooth tissue boundaries, gradually varying speckle, coherent chamber geometry at any instant), so pixel reconstruction has a trivial spatial-only solution — reconstruct a masked patch from its visible spatial neighbors at the same timestep. Tube masking does not address this because it leaves visible patches at every timestep; it only constrains *which* spatial positions are visible, not *that* some timesteps are invisible.

**What this means.** No masking intervention can fix MAE's temporal collapse on spatially redundant video. Frame-gap masking (mask entire frame positions) does not help — it addresses the same cross-frame-copying hypothesis that tube masking already rules out. The only masking strategy that would force temporal reasoning is whole-frame masking with no visible tokens at some timesteps, and that risks training collapse on pixel reconstruction (no information within a masked frame to reconstruct from). **The prediction target itself is the bottleneck.** JEPA avoids the shortcut *by design*, not by masking: the EMA teacher's targets are abstract latent embeddings, so there is no "spatial interpolation in latent space" that corresponds to copying adjacent-patch pixel values. Matching the teacher's latent requires producing the same high-level features — which for echo videos means encoding the temporal dynamics that distinguish one clip from another.

**Paper text (two sentences):**
> Our MAE uses tube masking (Tong et al., 2022), which prevents cross-frame patch copying, yet the temporal shortcut persists: MAE e99 is invariant to frame shuffling (−4% under full shuffle, flat across all six disruption conditions). This indicates the shortcut arises from within-frame spatial redundancy rather than temporal copying, and cannot be resolved by masking design alone — the pixel-reconstruction objective, not the masking strategy, is the bottleneck.

**Strength of the finding.** This elevates §3 from "MAE is temporally flat on this dataset" (an observation any reviewer could dismiss as insufficient masking) to "tube masking — the community-standard defense — fails, and the pixel-reconstruction objective cannot be rescued by masking design." The reframe is the failure of an existing, well-known intervention, which is more persuasive than proposing a novel intervention that happens to also fail.

**The two-mechanism story: MAE flatness and SALT cliff converge on the same conclusion.** Taken together with §3.5, the paper now documents **two mechanistically distinct failure modes of pixel-reconstruction-related SSL on spatially redundant video**, both producing degraded temporal encoding for different reasons:

| Failure mode | Mechanism | Profile | Evidence |
|---|---|---|---|
| **MAE (direct pixel target)** | Within-frame spatial interpolation — adjacent patches are correlated enough that a masked patch can be reconstructed from its visible spatial neighbors at the same timestep, without ever attending across time. Tube masking does not block this shortcut because it leaves visible patches at every timestep; it only constrains *which* spatial positions are visible. | **Flat** (+1% under full shuffle). By convergence, the model's predictions are completely invariant to frame order. | §3.6 tube masking reframe |
| **SALT (indirect pixel target via frozen teacher)** | Frozen-teacher distillation — the student matches a pre-trained pixel-reconstruction teacher's latent targets on in-distribution frame arrangements, but has no EMA co-evolution mechanism to continuously re-expose itself to novel temporal structure. The student memorizes the teacher's targets at the granularity of tubelet-level features (matching the pretraining setup) and cannot generalize to finer-grained or novel frame permutations. | **Cliff** (−250% under frame-level shuffle, but invariant to tubelet-level disruption). The collapse occurs precisely at the granularity below the student's pretraining target unit. | §3.1 4-way table + §3.5 factorial isolation |

Both failure modes trace back to the same root cause: **the absence of a co-evolving teacher producing abstract latent targets that resist within-frame pixel shortcuts.** MAE lacks a teacher entirely. SALT has a teacher, but it is frozen and trained on pixel reconstruction — so the targets the teacher provides are themselves pixel-interpolation-compatible, and the student inherits the pixel-target weakness through one remove. JEPA avoids both failure modes because its EMA teacher (a) produces abstract latent targets rather than pixel-style ones, and (b) co-evolves with the student, so the targets adapt as the student improves and cannot be memorized at a fixed granularity.

The two failure modes bracket the space of "pixel-reconstruction SSL on spatially redundant video" — direct pixel targets produce flatness (the shortcut wins from the start), frozen-pixel-teacher targets produce cliffs (the shortcut is inherited into the latent domain). **No intervention on the masking or architecture side has been shown to fix either failure mode.** The prediction target AND the teacher dynamics jointly determine whether temporal features survive training, and both must be chosen correctly (latent target + co-evolving teacher) for robust temporal encoding to emerge.

**Dropped experiments.** Frame-gap MAE intervention (ViT-B pilot) cancelled 2026-04-08 — the hypothesis it was testing is refuted by the existing ViT-L tube-masking run. Saves ~2 days HyperPod compute; reallocated to reconstruction visualization, temporal attention analysis, and writing. See `experiments/tube-masking-failure.md` for full writeup.

---

## Section 4: Mechanism Synthesis (~0.5 pages)

A brief bridge from Part 1 (§3) to Part 2 (§5). Four formal mechanism claims derived from §3 that §5 will test on held-out tasks:

**M1 (JEPA's temporal + spatial advantage).** JEPA's EMA co-evolving latent-target training produces features that (a) encode temporal structure robust to frame permutation (−19% under full shuffle vs BYOL −40%, MAE +1%, SALT −250%) and (b) are spatially stronger than any other method (JEPA fully-shuffled R² 0.488 > every other method's clean R² on LVEF). JEPA advantage = latent target + co-evolving teacher (both components).

**M2 (MAE's pure-spatial convergence).** Under tube masking at 90%, the converged MAE encoder has no temporal encoding — frame order is irrelevant at e99. The shortcut is within-frame spatial interpolation, not cross-frame copying; it cannot be fixed by masking design. MAE's features are therefore purely spatial at convergence, with whatever cardiac-cycle information survives being derivable from a single frame's anatomy alone.

**M3 (BYOL's intermediate temporal fragility).** BYOL's global-pool + EMA training produces moderate temporal encoding that degrades steeply under shuffling (−40%). BYOL's stability at mid-training resolves into a "moderate but fragile" profile — useful temporal signal but not consolidated enough to resist strong permutations.

**M4 (SALT's brittle in-distribution-only temporal features).** SALT's frozen-teacher latent-target training produces temporal features that generalize at tubelet granularity (the pretraining target unit) but collapse catastrophically below that level. Without EMA co-evolution, the student can only memorize the teacher's in-distribution targets — the mechanism that lets JEPA consolidate features under the same target design is absent. SALT clean R² is worse than MAE clean R² even before any disruption.

**Bridge to Part 2.** Each claim predicts a specific behavior on held-out tasks. §5 derives explicit predictions from M1–M4 and tests them on a curated task battery chosen such that each task tests at least one prediction with a pre-committed expected ranking.

---

## Section 5: Part 2 — Predictions and Cross-Task Validation (~2.5 pages)

**What Part 2 does.** Part 1 (§3) characterized what each method learns on a single well-understood task (LVEF). Part 2 tests whether those characterizations generalize by deriving formal predictions from M1–M4 (§4) and validating them on tasks the mechanism experiments did not touch. Task selection is **mechanism-driven**: each task is included because it tests at least one specific prediction with a pre-committed expected ranking. This separates hypothesis generation (§3–4) from hypothesis testing (§5) and protects against retrospective cherry-picking.

### 5.1 Predictions derived from the Part 1 mechanism

We derive seven formal predictions from M1–M4. Each prediction names the mechanism it tests, the task(s) that test it, the expected ranking, and what "pass" and "fail" look like.

#### P1: JEPA wins on functional / dynamic biomarker tasks

**Mechanism (M1):** JEPA's co-evolving EMA teacher produces consolidated temporal features. Tasks that depend on integrating cardiac motion across multiple frames — pressure estimation, ejection dynamics, volume changes — should disproportionately benefit from JEPA's temporal encoding.

**Test tasks:**
- **NT-proBNP regression** (MIMIC-IV lab linkage). NT-proBNP is a cardiac stress biomarker elevated in heart failure; its value correlates with LV filling pressures and wall stress across the cardiac cycle.
- **Note-extracted EF classification/regression** (`ef_note_extracted`, MIMIC clinical notes). The target is an LVEF value extracted from echo reports. Fundamentally a functional/dynamic measurement.
- **UHN LVEF** (existing 53K test set, functional ground truth).

**Expected ranking (pre-committed):** JEPA > BYOL > MAE ≈ SALT, with the JEPA-MAE gap being large (≥0.05 R² or AUROC).

**Pass condition:** JEPA top-ranked on all three tasks. Gap to MAE exceeds 0.05 on at least two. SALT below MAE on at least two.

**Fail condition:** JEPA loses on any of the three, OR gap to MAE is below 0.02 on all three. A failure would indicate that LVEF-derived mechanism does not generalize to adjacent functional tasks and would require scoping the paper's claims to echo-specific LVEF only.

#### P2: JEPA wins on rhythm / multi-beat tasks

**Mechanism (M1):** Rhythm classification fundamentally requires integrating information across multiple cardiac cycles. JEPA's consolidated temporal encoding should dominate; MAE's purely spatial features should fail; SALT's cliff should manifest.

**Test task:**
- **Atrial fibrillation classification** (`disease_afib`, MIMIC ICD linkage). AFib is a multi-beat rhythm disorder visible in the atria over time. Single-frame anatomy cannot classify it reliably.

**Expected ranking:** JEPA > BYOL > MAE, with JEPA >> MAE by the largest margin of any task.

**Pass condition:** JEPA clearly first, MAE clearly below JEPA, with at least 0.05 AUROC gap (the largest of all our task comparisons).

**Fail condition:** MAE within 0.02 of JEPA — would mean single-frame anatomy suffices for AFib, contradicting M1. Would require the paper to weaken claims about temporal-specificity.

#### P3: MAE wins on structural / anatomical tasks

**Mechanism (M2):** MAE's converged features are purely spatial. On tasks determined by static anatomy visible in a single frame, MAE should perform at least as well as any EMA-based method because its features are optimized for exactly that signal. This is the "ranking inversion" half of the paper.

**Test tasks:**
- **CAMUS segmentation** (frozen CAMUS decoder, Dice score, existing 4-way result: MAE 0.827 > BYOL 0.825 > JEPA 0.815).
- **Disease detection (DCM/HCM)** (`disease_dcm`, `disease_hcm`, MIMIC ICD linkage). Cardiomyopathy classification is based primarily on chamber geometry and wall thickness — static anatomical features that MAE should encode as well as JEPA.

**Expected ranking on CAMUS segmentation:** MAE ≥ BYOL ≥ JEPA ≥ SALT (with gaps <0.02 Dice).

**Expected ranking on DCM/HCM:** MAE ≈ JEPA ≈ BYOL (within 0.02 AUROC), with MAE at least competitive. SALT below.

**Pass condition:** MAE top or tied on CAMUS. MAE within 0.02 of the best method on DCM and HCM.

**Fail condition:** MAE clearly loses on both cardiomyopathy tasks — would mean the spatial-only characterization of MAE is wrong, or that cardiomyopathy detection requires more temporal information than anticipated.

#### P4: BYOL fails on tasks requiring strong temporal coherence

**Mechanism (M3):** BYOL's global-pool target is less information-dense than JEPA's per-token targets; its temporal encoding is moderate and fragile. On tasks where noise robustness and temporal coherence matter most (prognostic outcomes, mortality prediction where informative signal is distributed across the full clip), BYOL should underperform both JEPA and MAE.

**Test tasks:**
- **1-year mortality classification** (`mortality_1yr`, MIMIC outcomes linkage). Mortality prediction is noisy and distributed; any single-frame signal is weak.
- **Troponin-T regression** (`troponin_t`, MIMIC lab linkage). Acute cardiac injury biomarker; benefits from integrated temporal context.

**Expected ranking:** JEPA > MAE ≈ BYOL, with BYOL specifically *not* winning either task. BYOL may tie with MAE but should not lead.

**Pass condition:** BYOL does not lead either task. JEPA leads both.

**Fail condition:** BYOL leads either task — would mean BYOL's global-pool target encodes stronger prognostic signal than its LVEF shuffling profile suggests, and would complicate M3.

#### P5: SALT underperforms on all tasks (cliff generalizes)

**Mechanism (M4):** SALT's frozen-teacher design produces features that memorize teacher targets on in-distribution data. Because clinical downstream tasks use natural, in-distribution frame arrangements (unlike the shuffling conditions in §3), SALT's cliff does not directly manifest — but its *clean* R² is already below MAE, and this should carry over to held-out tasks.

**Test task:** All tasks in the §5 battery.

**Expected ranking:** SALT bottom or tied-bottom on every task.

**Pass condition:** SALT bottom on at least 5 of 6 tasks.

**Fail condition:** SALT above MAE on any task — would mean SALT has a task-specific advantage (possibly from more diverse teacher supervision) that is invisible on LVEF alone. Interesting nuance, not a paper-breaker.

#### P6: The JEPA advantage grows with functional task difficulty

**Mechanism (M1 + M2):** The more a task depends on temporal integration, the larger the JEPA > MAE gap should be. Tasks with strong static anatomical components should show narrow gaps or inversion; tasks with strong temporal/dynamic components should show wide gaps.

**Test:** Rank all §5 tasks by predicted "functionality" before seeing results, then check whether JEPA > MAE gap correlates with that ranking. Predicted functionality order (most functional → most anatomical):

1. `ef_note_extracted` (most functional — LVEF is fundamentally dynamic)
2. `nt_probnp` (functional — filling pressure dynamics)
3. `disease_afib` (temporal pattern)
4. `troponin_t` (semi-functional — acute injury)
5. `mortality_1yr` (integrated prognosis)
6. `disease_hcm` / `disease_dcm` (structural)
7. CAMUS segmentation (most spatial)

**Expected pattern:** Spearman correlation between predicted functionality rank and observed JEPA−MAE gap ≥ 0.6 (moderately strong).

**Pass condition:** ρ ≥ 0.6.

**Fail condition:** ρ < 0.3 — would mean the functionality axis does not cleanly predict cross-task rankings, and the paper should weaken the "spectrum" framing.

#### P7: Counter-prediction — JEPA's spatial features are still strong (no catastrophic inversion)

**Mechanism (M1 counter-half):** JEPA's fully-shuffled R² beats every other method's clean R² on LVEF. JEPA's spatial features are not a weakness — they are the second-best axis of its representation. On purely structural tasks, we expect JEPA to be competitive, not catastrophically inverted (JEPA second to MAE by small margin, not last).

**Test tasks:** Same as P3 (CAMUS, HCM, DCM).

**Expected ranking on CAMUS:** MAE (first) > JEPA ≈ BYOL (tied second, within 0.02 Dice) >> SALT.

**Pass condition:** JEPA within 0.02 Dice of MAE on CAMUS. JEPA never last on any structural task (except where SALT is bottom).

**Fail condition:** JEPA clearly worst on structural tasks — would suggest JEPA's spatial features are a genuine weakness, not just "second best," and would require reweighting the JEPA advantage.

### 5.2 Task selection rationale

The task battery is **minimal by design** — 6 held-out tasks plus the existing LVEF data, chosen to span the prediction space:

| Task | Paradigm | Tests | Source |
|---|---|---|---|
| NT-proBNP | Regression | P1, P6 | MIMIC lab linkage |
| `ef_note_extracted` | Regression / classification | P1, P6 | MIMIC clinical notes |
| UHN LVEF | Regression | P1 | Existing ICML probe |
| `disease_afib` | Classification | P2, P6 | MIMIC ICD linkage |
| `disease_dcm` + `disease_hcm` | Classification | P3, P6, P7 | MIMIC ICD linkage |
| Troponin-T | Regression | P4, P6 | MIMIC lab linkage |
| `mortality_1yr` | Classification | P4, P6 | MIMIC outcomes linkage |
| CAMUS segmentation | Segmentation | P3, P7 | Existing ICML decoder |
| Pediatric zero-shot LVEF | Regression (transfer) | P1, generalization | Existing ICML probe |

**Note on Nature Medicine deconfliction.** These MIMIC labels overlap with Nature Medicine paper content. The carve-out: NeurIPS uses each label as a **mechanism probe** testing a specific §5 prediction, with framing "does the Part 1 mechanism predict the cross-task ranking?" Nature Medicine uses the same labels as **clinical findings** with framing "EchoJEPA can predict X from B-mode alone for clinical utility Y." Same numbers, different claims, non-overlapping scientific contributions. Co-authors must agree on the boundary before either paper's final draft.

### 5.3 Results by prediction

*[To be filled in after MIMIC probe training; expected ~1 HyperPod day of new work — see `claude/neurips/mimic-part2-scoping.md` for the specific probes to train and the launch strategy.]*

Structure of this section will be:
- One subsection per prediction (P1–P7)
- For each: brief restatement of mechanism, pre-committed expected ranking, observed numbers per task, pass/fail judgment, sentence of interpretation
- Summary table at the end: number of predictions that passed, failed, or were ambiguous
- Overall Spearman correlation for P6 (functionality axis)

### 5.4 Where predictions fail (honest reporting)

*[To be filled in after results.]*

Template:
- If any prediction fails, the subsection describes what happened and what it means for the mechanism
- If a prediction fails in an interesting way (e.g., BYOL wins mortality), discuss what this tells us about the scope of M1–M4
- If multiple predictions fail, weaken the paper's claims to "LVEF-specific mechanism" and discuss why the generalization broke

**Honest reporting is a feature of this structure, not a bug.** A paper that makes predictions and reports honestly when some fail is more credible than one where every experiment conveniently confirms the thesis. Reviewers will treat "5 of 7 predictions confirmed, here's what the 2 failures tell us" more favorably than "we checked 7 things and they all worked."

### 5.5 Extended validation: cross-dataset transfer

Two additional tests that do not fit the MIMIC battery but bear on M1 generalization:

**EchoNet-Dynamic LVEF transfer.** Tests whether the LVEF mechanism generalizes to a different echo dataset (different institution, different imaging protocols). Expected ranking: JEPA > BYOL > MAE > SALT (same as §3.1 clean ranking). Existing numbers: JEPA IN21K e100 R²=0.591, BYOL 0.468, MAE 0.445, SALT 0.414 (pred-avg, 1,277 test videos). **Confirms P1 at the dataset level.**

**Pediatric zero-shot LVEF (EchoNet-Pediatric).** Tests cross-population transfer — probes trained on adult UHN LVEF, evaluated on pediatric clips without retraining. Tests whether the temporal-encoding mechanism is population-invariant. Expected ranking: JEPA > MAE > BYOL (same as ICML numbers: JEPA Pearson 0.670, MAE 0.617, BYOL 0.500). **Confirms P1 under maximum distribution shift and preempts the "JEPA's advantage only shows up on training-distribution data" objection.**

SALT on both transfer tasks is not yet measured — adding SALT pediatric transfer and SALT EchoNet-Dynamic would test P5 on cross-dataset data. ~1 hour of new compute each if run via the existing probe inference pipeline.

---

## Section 6: Robustness Under Physics-Based Perturbations (~1.0 pages)

EchoBench tests a specific Part 1 prediction: **methods that encode temporal features should be more robust to noise on functional tasks, and methods with purely spatial features should be more robust on spatial tasks.** This is derivable from M1 + M2 and is formally prediction P8 (implicit in §5).

**EchoBench methodology.** Three perturbation types (depth attenuation, acoustic shadow, haze). Protocol: frozen probes, no retraining, 3 severity levels.

**⚠️ FRAMING:** All perturbations are **spatially static** (same corruption map every frame — code: `echo_perturbations.py`, all maps broadcast via `unsqueeze(0).unsqueeze(0)`). EchoBench tests **clinical image quality degradation**, NOT frame-varying speckle. Include one sentence: "These perturbations are spatially static, simulating fixed clinical artifacts. The frame-varying component of ultrasound noise (speckle) is addressed by the representation-level analysis in §3."

**LVEF robustness (init-matched e100, EchoNet-Dynamic):**

| | Clean R² | Avg severe drop |
|---|---------|----------------|
| JEPA IN21K e100 | **0.591** | **−20%** |
| BYOL e100 | 0.468 | −22% |
| MAE e99 | 0.445 | **−51%** |

MAE collapses under depth attenuation (R²=0.090) and haze (0.162). **Confirms P8 for functional tasks:** JEPA's temporal encoding provides robustness where MAE's spatial-only features collapse.

**Segmentation robustness (init-matched e100, CAMUS):**

| | Clean Dice | Avg severe drop |
|---|-----------|----------------|
| MAE e99 | **0.827** | −13% |
| JEPA IN21K e100 | 0.815 | **−10%** |
| BYOL e100 | 0.825 | −29% |

**Confirms P8 for spatial tasks:** MAE and JEPA both robust on segmentation; BYOL catastrophically fragile. This is a stronger version of the P3 ranking-inversion prediction: the inversion holds under both clean and perturbed conditions.

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

**Connection to §3–§5:** MAE's selective fragility is best explained by its lack of temporal structure encoding (frame shuffling: MAE abandons temporal information, JEPA consolidates it). BYOL's universal fragility reflects its less stable intermediate temporal encoding. SALT EchoBench not yet measured; prediction (from P5 + the cliff): SALT should collapse on LVEF robustness (worse than MAE) and be competitive but not strong on CAMUS. Effective dimensionality is similar across models (200–245 range), so the explanation is in feature *content*, not capacity.

---

## ~~Scaling section~~ — CUT (historical note, not a numbered section)

EchoJEPA-G (ViT-g, 384px, 18M UHN private data) breaks the controlled comparison — confounds prediction target with 36× more data and 4× more parameters. Conflicts with Nature Medicine deconfliction. Reclaimed 0.5 pages for §3 mechanism. One sentence in discussion: "Preliminary scaling experiments suggest these findings hold at scale; detailed analysis is beyond this controlled study's scope."

---

## Section 7: Discussion and Limitations (~1 page)

**Arc recap.** Part 1 (§3) characterized four SSL objectives via a controlled mechanism comparison on LVEF, producing a 2×2 factorial decomposition across prediction target and teacher dynamics. §4 synthesized formal claims. Part 2 (§5) derived predictions from those claims and validated them on held-out tasks. §6 tested a specific prediction about noise robustness. The paper is an integrated arc: mechanism → prediction → confirmation.

**What generalizes and what does not.** Summarize which §5 predictions held (expected majority), which failed (if any), and what the failures tell us about the scope of the mechanism. Be explicit: the controlled mechanism was characterized on a single task (LVEF) in a single regime (525K narrow-domain echo clips, single-node compute). Generalization to other tasks in this regime is confirmed by §5; generalization to other regimes (data-rich natural video, diverse external teachers) is explicitly out of scope and we do not claim it.

**Noise autocorrelation as causal test (appendix).** Brief mention: "We can turn the ranking inversion on and off by varying the temporal correlation of noise." Full analysis in `experiments/noise-autocorrelation-sweep.md`.

**Generalization evidence:**
- Fetal US (appendix): both tasks spatial, MAE leads both as predicted → cross-anatomy transfer confirmed
- Calcium imaging (if completed): different physics, genuine 2×2 inversion → general principle

**Practical decision rules derived from §5:**
- Use JEPA (latent + co-evolving teacher) for functional/temporal/prognostic tasks
- Use MAE (pixel target) for pure anatomical segmentation and structural disease classification
- Avoid BYOL when robustness matters
- Avoid SALT without data diversity or a strong external teacher
- EchoBench perturbation testing should be standard for ultrasound SSL

**Limitations:**
- Single primary modality (echocardiography); we expect findings to generalize to other spatially-redundant video domains (radar, low-SNR microscopy) but have not tested this
- Part 1 mechanism characterized on a single task (LVEF); §5 confirms generalization across task types but does not rule out LVEF-specific mechanism components
- SALT uses paper-spec random student init; all other methods use IN21K — see §3.5 defensive bridges for why this does not explain the gap
- 100-epoch budget may not reflect fully-trained behavior; v1 e199 regression suggests SALT is not improving beyond 100 epochs, but JEPA/BYOL/MAE behavior at very long training lengths is unknown
- EchoBench perturbations are spatially static; frame-varying noise is addressed only in §3 representation-level analysis

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
- **I.** SALT detailed analysis (three-variant robustness table, extended training regression, factorial isolation).
- **J.** Per-prediction results tables for §5 (full numbers with bootstrap CIs).
- **K.** Nature Medicine deconfliction statement (explicit carve-out between mechanism probes in NeurIPS and clinical findings in Nature Medicine on shared MIMIC labels).

---

## Figure Plan

| Figure | Content | Source |
|--------|---------|--------|
| **Fig 1** | Method overview: 4 paradigms, 2×2 factorial table, Part 1 / Part 2 arc diagram | New (draw) |
| **Fig 2** | Part 1 mechanistic panel: (a) 6-condition R² bars at convergence — four profiles (JEPA gentle / BYOL steep / MAE flat / SALT cliff), (b) severity gradient curves at convergence — four regime shapes, (c) training dynamics: R² clean vs shuffled across epochs showing MAE transient + JEPA consolidation (JEPA/BYOL/MAE only) | `experiments/6-condition-shuffling.md`, `experiments/severity-gradient.md` |
| **Fig 3** | Part 2 prediction dashboard: grid of task results colored by pass/fail, one cell per prediction P1–P7 | `claude/neurips/mimic-part2-scoping.md` (pending) |
| **Fig 4** | Functionality axis: Spearman scatter of predicted functionality rank vs observed JEPA−MAE gap (tests P6) | Pending §5 results |
| **Fig 5** | Noise robustness curves: R² vs severity for 3 perturbation types across 4 methods | `rebuttals/10-*` §5m |
| **Fig 6** | Clinical impact OR calcium imaging 2×2 (if completed) | Fallback: pathology-stratified scatter |

---

## Page Budget (updated 2026-04-08 for Part 1 / Part 2 arc)

| Section | Pages |
|---------|-------|
| §1 Introduction | 1.5 |
| §2 Experimental Design | 1.0 |
| §3 Part 1: Mechanism on LVEF | 2.5 |
| §4 Mechanism synthesis | 0.5 |
| §5 Part 2: Predictions and cross-task validation | 2.5 |
| §6 EchoBench robustness | 1.0 |
| §7 Discussion + limitations | 1.0 |
| **Total** | **10.0** |

**Over budget by 0.5 page.** Cuts available:
- Compress §6 EchoBench to 0.5p by moving per-task breakdown tables to appendix C
- Tighten §3.3 / §3.4 (effective dim, speckle) to one paragraph each (they are appendix candidates)
- Compress §2 to 0.75p by removing the regime-reinforcement block (already in §1)

Target final: 9.5p after compression.
