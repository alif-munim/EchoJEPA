# NeurIPS Paper Outline

Target: 9 pages main text + references + appendix. Representation Learning / SSL track.

---

## Section 1: Introduction (~1.5 pages)

**Opens with the general question:** When imaging is dominated by stochastic noise (ultrasound speckle, radar clutter, low-SNR microscopy), does the SSL prediction target matter?

**States the hypothesis:** Pixel reconstruction retains noise because it must reconstruct it. Latent prediction filters noise because the EMA target averages over stochastic frame-to-frame variation.

**Previews the finding:** Rankings invert by task type. Latent prediction leads functional tasks; pixel reconstruction leads spatial/anatomical tasks. This dissociation, invisible from clean benchmarks, is revealed by physics-based evaluation.

**Preempts the novelty concern directly:** "The encoder architecture is deliberately held constant across all objectives. This is the experimental design, not a limitation. Our contribution is the systematic empirical finding, the mechanistic evidence explaining it, and the evaluation methodology that reveals it." Cite precedents (scaling laws papers, "Do ViTs See Like CNNs?", understanding papers at NeurIPS).

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

**Central claim:** The prediction target doesn't just determine what is encoded — it determines what *survives training*. We identify three qualitatively distinct temporal encoding regimes, invisible from single-checkpoint evaluation.

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

**Cross-model comparison at convergence (6-condition):**

| Condition | JEPA e100 | BYOL e100 | MAE e99 |
|-----------|-----------|-----------|---------|
| clean | **0.591** | 0.468 | 0.445 |
| tubelet | **0.582** | 0.402 | 0.424 |
| reverse | **0.539** | 0.373 | 0.431 |
| matched | **0.580** | 0.415 | 0.419 |
| shuffle | **0.484** | 0.291 | 0.422 |
| matched_frame | **0.477** | 0.280 | 0.449 |

Three distinct profiles: JEPA gentle slope, BYOL steep slope, MAE flat. JEPA matched_frame (0.477) > BYOL clean (0.468).

**Figure 2a:** Bar chart of this table.

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

**Three findings from this matrix:**

**Finding 1: Three temporal encoding regimes at convergence (Fig 2b).** The severity gradient curves at e100 are visually distinct: JEPA gentle slope (−17%), BYOL steep linear (−38%), MAE flat (−4%). One figure, one glance, three regimes.

**Finding 2: Temporal encoding is dynamic (Fig 2c).** All three objectives emerge from shared early instability, diverge by convergence:
- **JEPA — Consolidation.** e25: −14% → e50: −42% (peak) → e75: −31% → e100: −17%. Temporal encoding is learned, peaks, then becomes efficient and robust.
- **BYOL — Stabilization.** e24: −146% (catastrophic) → e50: −49% → e75: −30% → e100: −38%. Resolves early collapse into stable moderate dependence.
- **MAE — Transient.** e24: −20% → e50: −313% (catastrophic) → e74: −15% → e99: −4%. Temporal features are learned, maximally exploited, then completely abandoned. By convergence frame order is irrelevant. *Novel finding — challenges the static view that "MAE doesn't learn temporal features."*

**Finding 3: JEPA's advantage is not just temporal.** JEPA e100 fully shuffled (R²=0.488) > BYOL e100 clean (0.468) > MAE e99 clean (0.445). Even with all temporal information destroyed, JEPA's spatial features are the strongest. Preempts the reviewer objection "JEPA only wins because of temporal encoding."

**One-paragraph text:** "We identify three qualitatively distinct temporal encoding regimes shaped by the prediction target. All emerge from shared early instability but diverge in resolution: EMA-based latent prediction (JEPA) consolidates temporal features into a robust representation (−17% at convergence); global self-distillation (BYOL) stabilizes at moderate temporal dependence (−38%); pixel reconstruction (MAE) abandons temporal encoding entirely (−4%) after a transient phase of catastrophic reliance at mid-training. These dynamics are invisible from single-checkpoint evaluation. Notably, JEPA's spatial features alone (under full temporal disruption) outperform BYOL's combined spatial+temporal features, demonstrating that the advantage of latent prediction extends beyond temporal encoding."

**Figure plan:**
- **Fig 2a:** 6-condition bar chart at e100 (JEPA/BYOL/MAE) — monotonic gradient
- **Fig 2b:** Severity gradient curves at e100 — three regime shapes
- **Fig 2c (appendix or main):** Degradation % vs epoch — training dynamics showing MAE transient + JEPA consolidation
- **Appendix:** Full 13-model × 5-fraction and 12-model × 6-condition tables

### 4.3 Effective dimensionality (NEW — replaces speckle probing as main mechanistic evidence)

| Model | Effective Dim (d_eff) | Usage of 1024-dim space |
|-------|-----------------------|------------------------|
| BYOL e100 | 209 | 20% |
| JEPA IN21K e100 | 197 | 19% |
| MAE e99 | **63** | **6%** |

**Key finding:** MAE representations occupy a **3× lower-dimensional subspace** than JEPA/BYOL. Pixel reconstruction produces redundant features; latent prediction encourages diversity. This directly explains MAE's functional task weakness — limited effective capacity for encoding complex temporal/functional information.

**For the paper:** "We compute the effective dimensionality of each model's representations (exponential of the spectral entropy; Garrido et al., 2023). MAE representations occupy a 3× lower-dimensional subspace (d_eff=63) than JEPA (197) or BYOL (209), indicating that pixel reconstruction produces highly redundant features while latent prediction encourages representational diversity."

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

### 4.5 SALT: the frozen teacher ceiling (confirmed)

**Result:** SALT S2 e199 probe val MAE ~6.8 — worse than SALT S2 e79 (6.47). Training loss plateaued (0.429→0.419), weight cosine similarity >0.999 between e79 and e199. The frozen teacher imposes a representation ceiling that additional student training cannot overcome.

**Severity gradient:** SALT S2 e79 collapses at 25% shuffle (R²=−0.037). The frozen pixel teacher provides a latent target but without EMA dynamics, the student learns no temporal robustness.

**Context against concurrent work:** The SALT paper (Apple, 2025) trains on 3.6M diverse natural video clips — high data diversity compensates for the static teacher. US-JEPA (concurrent) succeeds with SALT by using URFM (BiomedCLIP-distilled) as a strong externally-pretrained teacher with broad medical coverage. Our V-Pixel teacher, trained from scratch on 525K echo clips (single domain), has narrow coverage → ceiling. The frozen teacher mechanism needs either **data diversity** (SALT paper) or a **strong external teacher** (US-JEPA) to work. With neither, EMA-based co-evolution (JEPA) is strictly superior.

**One sentence for the paper:** "Concurrent work achieves competitive results with frozen teachers using either diverse pretraining data (SALT; Li et al., 2025) or strong externally-pretrained teachers (US-JEPA; Kang et al., 2025); our results show that when the teacher is trained from scratch on a narrow clinical domain, the frozen teacher mechanism imposes a representation ceiling (loss plateau at 0.42, weight cosine >0.999 between e79 and e199) that EMA-based co-evolution avoids."

**⚠️ CAVEAT (2026-04-06):** Initial SALT runs had several hyperparameter mismatches vs the SALT paper: L1 loss instead of L2, no LR cosine decay (constant 1.75e-4 vs 6.25e-4→1e-6), no WD ramp (constant 0.04 vs 0.04→0.4), ipe_scale=1.25 instead of 1.0, weaker augmentation. These are now fixed in configs. **SALT must be retrained with corrected configs before including in the paper.** The frozen teacher ceiling observation (loss plateau, weight convergence) may still hold, but the absolute performance gap is partly attributable to misconfiguration. See `ops-notes.md` for full discrepancy table.

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

**Connection to §4:** MAE's low effective dimensionality (d_eff=63 vs JEPA's 197) explains its selective fragility — redundant pixel-level features provide robustness for spatial tasks but insufficient diversity for functional tasks under noise.

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
