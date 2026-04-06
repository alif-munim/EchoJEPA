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

## Section 4: Mechanistic Evidence (~2.5 pages)

**Central claim:** The prediction target doesn't just determine what is encoded — it determines what *survives training*. We identify three qualitatively distinct temporal encoding regimes, invisible from single-checkpoint evaluation.

### 4.1 Frame shuffling: 6-condition baseline (Figure 2a)

6-condition temporal disruption gradient (from ICML rebuttal, pt50 models):
- JEPA retains most absolute signal post-shuffle (R²=0.365 > MAE clean)
- BYOL collapses to R²=0.099 under matched_frame (−79%)
- Monotonic gradient: clean ≈ tubelet ≈ reverse ≈ matched > shuffle > matched_frame

### 4.2 Severity gradient × training dynamics (Figure 2b — KEY RESULT)

Partial shuffle (0/25/50/75/100%) across pretraining epochs reveals three regimes:

**Full results (R², mean of 3 seeds):**

| Fraction | JEPA e25 | JEPA e50 | JEPA e75 | JEPA e100 | BYOL e50 | BYOL e100 | MAE e50 | MAE e99 | SALT S2 e79 |
|----------|----------|----------|----------|-----------|----------|-----------|---------|---------|-------------|
| 0.00 | 0.383 | 0.503 | 0.537 | **0.591** | 0.427 | 0.468 | 0.141 | 0.445 | 0.293 |
| 0.25 | 0.362 | 0.419 | 0.465 | **0.542** | 0.360 | 0.410 | 0.091 | 0.421 | -0.037 |
| 0.50 | 0.340 | 0.327 | 0.402 | **0.507** | 0.278 | 0.336 | -0.103 | 0.436 | -0.277 |
| 0.75 | 0.332 | 0.293 | 0.378 | **0.485** | 0.220 | 0.300 | -0.271 | 0.414 | -0.382 |
| 1.00 | 0.331 | 0.290 | 0.370 | **0.488** | 0.219 | 0.291 | -0.301 | 0.428 | -0.397 |

(Still running: BYOL e24/e75, MAE e24/e74 — will complete the per-epoch matrix.)

**Three temporal encoding regimes:**

1. **JEPA — Consolidation.** Temporal reliance peaks mid-training (e50: −42%) then consolidates (e100: −17%). The EMA target incentivizes temporal encoding throughout, but the representation becomes more efficient over time. At convergence, temporal features are strong but robust to disruption.

2. **MAE — Transient.** e50 collapses under shuffling (−313%); e99 is invariant (−4%). MAE initially uses temporal consistency as a shortcut for pixel reconstruction, then discovers static spatial features suffice and *discards temporal information entirely*. The temporal encoding doesn't weaken — it vanishes.

3. **BYOL — Stable.** Consistent ~40% degradation at both e50 and e100. Global self-distillation provides a fixed incentive for temporal encoding that neither grows nor shrinks.

**Three supporting results:**

- **JEPA spatial features alone beat everything:** JEPA e100 fully shuffled (R²=0.488) > BYOL e100 clean (0.468) > MAE e99 clean (0.445). The advantage is not just temporal — latent prediction produces better features on *both* axes.
- **SALT confirms EMA is the mechanism:** SALT S2 collapses at 25% shuffle (R²=−0.037). A latent target with a frozen teacher doesn't produce temporal robustness — the *EMA dynamics* are essential.
- **MAE's transient temporal encoding is novel:** Challenges the static view that "MAE is a spatial encoder." MAE *learns then unlearns* temporal features — a training dynamics effect invisible from any single checkpoint.

**One-paragraph text for the paper:** "We identify three qualitatively distinct temporal encoding regimes determined by the prediction target. Pixel reconstruction (MAE) produces transient temporal features that are eliminated during convergence as the encoder discovers static spatial features suffice. EMA-based latent prediction (JEPA) consolidates temporal encoding into an efficient representation that is robust to temporal disruption. Global self-distillation (BYOL) maintains stable but moderate temporal dependence. These dynamics, invisible from single-checkpoint evaluation, reveal the mechanistic role of the prediction target in shaping representation structure over training."

### 4.3 Speckle probing

JEPA encodes 23% less speckle (partial R²=0.674 vs 0.875). Monotonic: JEPA < BYOL < MAE. Directly measures frame-varying noise retention in representations.

### 4.4 Noise autocorrelation sweep (planned — P0 week 1)

Sweep temporal correlation of synthetic noise from τ=∞ (static) to τ=0 (iid per-frame). If the MAE/JEPA ranking inverts as noise becomes more frame-varying → causal proof. Implement by modifying `scripts/rebuttal/echo_perturbations.py`.

### 4.5 SALT subsection (conditional on e200 results)

If SALT e200 speckle probing shows frozen teacher retains more speckle than EMA → one paragraph confirming EMA filtering mechanism. The severity gradient already shows SALT collapses — this would add the representation-level evidence.

**§4 → §5 bridge:** These mechanistic differences (temporal encoding regimes, noise filtering) translate to practical robustness under clinical image quality degradation, tested in §5.

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
