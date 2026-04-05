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

**Core finding table:** 3-way (or 4-way if SALT included) comparison across all 5 tasks. JEPA leads functional tasks; MAE leads spatial tasks. Rankings invert.

**Cross-dataset amplification:** JEPA's advantage grows from +2.5pp R² in-distribution (UHN) → +12.6pp cross-dataset (EchoNet-Dynamic) → +10.3pp cross-population (Pediatric zero-shot).

**Clinical impact:** Pathology-stratified analysis. JEPA advantage 8× larger on reduced EF (<40%).

**Statistical validation:** Bootstrap CIs for all pairwise comparisons. All significant.

---

## Section 4: Mechanistic Evidence (~2 pages)

**Frame shuffling (Figure 2a).** 6-condition temporal disruption gradient:
- JEPA retains most absolute signal post-shuffle (R²=0.365 > MAE clean)
- BYOL collapses to R²=0.099 under matched_frame (-79%)
- Monotonic gradient: clean ≈ tubelet ≈ reverse ≈ matched > shuffle > matched_frame

**Frame shuffling severity gradient (Figure 2b).** Partial shuffle (0/25/50/75/100%) results (2026-04-05):

| Fraction | BYOL e100 | MAE e99 | SALT S2 e79 |
|----------|-----------|---------|-------------|
| 0.00 | 0.468 | **0.445** | 0.293 |
| 0.25 | 0.410 | **0.421** | -0.037 |
| 0.50 | 0.336 | **0.436** | -0.277 |
| 0.75 | 0.300 | **0.414** | -0.382 |
| 1.00 | 0.291 | **0.428** | -0.397 |

**Key finding: MAE e99 is invariant to shuffling.** R² stays ~0.43 from clean to fully shuffled — MAE converges to purely spatial representations. MAE e50 (R² 0.14→-0.30) still had temporal reliance, which was lost by e99. This is a **training dynamics** result: MAE's temporal encoding is transient, present early but eliminated during convergence. BYOL degrades linearly (~40% drop). SALT collapses immediately.

**Speckle probing.** JEPA encodes 23% less speckle (partial R²=0.674 vs 0.875). Monotonic: JEPA < BYOL < MAE.

**Noise autocorrelation sweep (NEW — P0 for week 1).** Sweep temporal correlation of synthetic noise from τ=∞ (static, same pattern every frame) to τ=0 (iid per-frame) on echo data. Plot MAE-vs-JEPA R² as a function of noise autocorrelation time. If the ranking inverts as noise becomes more frame-varying, this is **causal proof** of the mechanism — not just correlation, but demonstrating the effect can be turned on and off with one parameter. Implement by modifying `scripts/rebuttal/echo_perturbations.py` for per-frame noise with controllable correlation.

**Two mechanistic tests of the hypothesis** (both on native data with real frame-varying speckle):
1. Speckle probing: directly measures frame-varying noise retention (confirmed)
2. Frame shuffling: directly measures temporal dependence (confirmed + severity gradient)

**Practical consequence in §5:** EchoBench shows mechanistic difference translates to practical robustness under clinical degradation (static perturbations). §4 explains *why*, §5 shows *that it matters*.

**SALT subsection (conditional):** If SALT e200 results show frozen teacher retains more speckle than EMA teacher → include as one paragraph confirming the EMA filtering mechanism. If noisy → omit.

---

## Section 5: Robustness Under Physics-Based Perturbations (~1.5 pages)

**EchoBench methodology.** Three perturbation types (depth attenuation, acoustic shadow, haze). Protocol: frozen probes, no retraining, 3 severity levels.

**⚠️ FRAMING:** All perturbations are **spatially static** (same corruption map every frame — code: `echo_perturbations.py`, all maps broadcast via `unsqueeze(0).unsqueeze(0)`). EchoBench tests **clinical image quality degradation**, NOT frame-varying speckle. Include one sentence: "These perturbations are spatially static, simulating fixed clinical artifacts. The frame-varying component of ultrasound noise (speckle) is addressed by the representation-level analysis in §4."

**LVEF robustness table.** JEPA -19% avg, MAE -37%, BYOL -40%.

**Segmentation robustness table.** Rankings invert: MAE most robust (-8%), JEPA -10%, BYOL -25%.

**Key insight:** Clean performance fails to predict robustness. All three converge on clean CAMUS (<1pp); under severe perturbation, 32pp gap emerges.

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
| **Fig 2** | Mechanistic panel: (a) 6-condition R² bars, (b) severity gradient curves, (c) noise autocorrelation sweep | frame-shuffling results + NEW autocorrelation sweep |
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
