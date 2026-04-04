# NeurIPS Paper Outline

Target: 9 pages main text + references + appendix. Representation Learning / SSL track.

---

## Section 1: Introduction (~1.5 pages)

**Opens with the general question:** When imaging is dominated by stochastic noise (ultrasound speckle, radar clutter, low-SNR microscopy), does the SSL prediction target matter?

**States the hypothesis:** Pixel reconstruction retains noise because it must reconstruct it. Latent prediction filters noise because the EMA target averages over stochastic frame-to-frame variation.

**Previews the finding:** Rankings invert by task type. Latent prediction leads functional tasks; pixel reconstruction leads spatial/anatomical tasks. This dissociation, invisible from clean benchmarks, is revealed by physics-based evaluation.

**Preempts the novelty concern directly:** "The encoder architecture is deliberately held constant across all objectives. This is the experimental design, not a limitation. Our contribution is the systematic empirical finding, the mechanistic evidence explaining it, and the evaluation methodology that reveals it." Cite precedents (scaling laws papers, "Do ViTs See Like CNNs?", understanding papers at NeurIPS).

**Completed experiments used:** All (this section frames everything).
**New experiments needed:** None.

---

## Section 2: Experimental Design (~1.5 pages)

**The controlled comparison.** Four SSL paradigms, one architecture (ViT-L), one dataset (MIMIC-IV-Echo 525K), one compute budget (50 epochs):

| Paradigm | Prediction Target | Teacher | Code |
|----------|------------------|---------|------|
| JEPA | Local masked tokens | EMA encoder | `app/vjepa_2_1/` |
| BYOL-Video | Global mean-pooled repr | EMA encoder + projector | `app/byol_video/` |
| MAE | Pixels | None (reconstruction) | VideoMAE |
| SALT | Frozen teacher's latents | Frozen pixel-trained encoder | `app/salt/` |

**The 2×2 design:** SALT decouples {pixel vs latent target} from {EMA vs frozen teacher}. Table showing which cell each paradigm occupies.

**Evaluation protocol:** Frozen backbone, d=4 attentive probe (or d=1 for small datasets), prediction averaging across clips. 5 tasks: LVEF, RVSP, CAMUS, EchoNet-Dynamic, zero-shot pediatric. Multi-view framework for RVSP (brief description, full ablation in appendix).

**EchoBench protocol:** 3 physics-based perturbation types × 3 severity levels. Probes NOT retrained — tests representation robustness, not adaptation.

**Completed experiments used:** Setup descriptions from existing comparison.
**New experiments needed:** SALT pretraining + evaluation (P0).

---

## Section 3: The Prediction Target Determines What Is Encoded (~2 pages)

**Core finding table:** 4-way comparison across all 5 tasks. JEPA/SALT lead functional tasks; MAE leads spatial tasks. Rankings invert.

**Cross-dataset amplification:** JEPA's advantage grows from +2.5pp R² in-distribution (UHN) → +12.6pp cross-dataset (EchoNet-Dynamic) → +10.3pp cross-population (Pediatric zero-shot). The prediction target matters most when you generalize.

**Clinical impact:** Pathology-stratified analysis. JEPA advantage 8× larger on reduced EF (<40%). MAE predicts 48% for patients with true EF 29%.

**Statistical validation:** Bootstrap CIs for all pairwise comparisons. All significant.

**Completed experiments used:** #1-6, #12-14 from completed inventory.
**New experiments needed:** SALT results for all 5 tasks (P0).

---

## Section 4: Mechanistic Evidence (~1.5 pages)

**Frame shuffling (Figure 2).** 6-condition temporal disruption gradient. Table + figure showing:
- JEPA retains most absolute signal post-shuffle (R²=0.365 > MAE clean)
- BYOL collapses to R²=0.099 under matched_frame (-79%)
- MAE shows small relative drop because it has little temporal signal to lose
- Monotonic gradient: clean ≈ tubelet ≈ reverse ≈ matched > shuffle > matched_frame

**Speckle probing.** JEPA encodes 23% less speckle (partial R²=0.674 vs 0.875). Monotonic: JEPA < BYOL < MAE.

**Information-theoretic hypothesis** (stated as testable predictions, not a formal proof):
- Under MAE: optimal encoder retains all pixel info including noise
- Under JEPA with EMA: target is temporally smoothed, noise is averaged out
- Under SALT: frozen teacher trained on pixels → does the student inherit the filtering or learn its own?

**Three empirical tests of the hypothesis:** speckle probing (confirmed), frame shuffling (confirmed), noise robustness (confirmed).

**Completed experiments used:** #10, #11 from completed inventory.
**New experiments needed:** SALT frame shuffling + speckle probing (P0).

---

## Section 5: Robustness Under Physics-Based Perturbations (~1.5 pages)

**EchoBench methodology.** Define the three perturbation types (depth attenuation, acoustic shadow, haze) with acoustic physics motivation. Protocol: frozen probes, no retraining, 3 severity levels.

**LVEF robustness table.** 4 models × 3 perturbations × 3 severities. JEPA -19% avg, MAE -37%, BYOL -40%. JEPA under severe noise outperforms MAE's clean baseline.

**Segmentation robustness table.** Rankings invert: MAE most robust on anatomy (-8%), JEPA on function (-19%). Different objectives → different robustness profiles.

**Key insight:** Clean performance fails to predict robustness. All three converge on clean CAMUS (<1pp); under severe depth attenuation, 32pp gap emerges. Standard SSL benchmarks miss this.

**Multi-view robustness.** Multi-view at severe ≈ single-view clean. Cross-view integration halves degradation.

**Completed experiments used:** #7-9, #15 from completed inventory.
**New experiments needed:** SALT through EchoBench (P0). DINOv2 + random baselines (P2, optional).

---

## Section 6: Scaling (~0.5 pages)

Brief section showing EchoJEPA-G (1.1B params, 18M echos) as a system-level result. Not the headline — confirms that the findings from the controlled 50-epoch comparison hold at scale.

**Completed experiments used:** #17 from completed inventory.
**New experiments needed:** None.

---

## Section 7: Discussion and Limitations (~0.5 pages)

**Generalization hypothesis:** The mechanism (EMA target filtering stochastic noise) should apply to any modality where pixel-level noise dominates: fetal ultrasound, lung ultrasound, radar, sonar, low-SNR microscopy. State as prediction, not claim.

**Practical decision rule:** Use latent prediction for functional/temporal tasks, pixel reconstruction for spatial/anatomical tasks, avoid global self-distillation for robustness.

**Limitations:** Single imaging modality (echo); 50-epoch budget may not reflect fully-trained behavior; SALT epoch-matching requires careful accounting; speckle probing is linear (may miss nonlinear encoding).

---

## Appendix

- **A.** Multi-view probing framework: factorized stream embeddings, early fusion (+12.1%), view dropout (+18.3%). Full ablation table.
- **B.** CAMUS per-structure segmentation results (LV, MYO, LA × ED, ES).
- **C.** Full perturbation matrices at mild/moderate/severe.
- **D.** Pediatric robustness from both source datasets (UHN and END probes).
- **E.** Hyperparameter sensitivity (probe LR/WD grid results).
- **F.** Biplane LVEF (if P3 experiment completed).

---

## Figure Plan

| Figure | Content | Source |
|--------|---------|--------|
| **Fig 1** | Method overview: 4 paradigms shown as 2×2, evaluation protocol | New (draw) |
| **Fig 2** | Frame shuffling: 6-condition temporal disruption gradient, R² per model | `experiments/frame-shuffling.md` |
| **Fig 3** | Ranking inversion: bar chart of R² across tasks showing JEPA wins function, MAE wins anatomy | Completed experiments |
| **Fig 4** | Noise robustness curves: R² vs severity for 3 perturbation types × 4 models | `rebuttals/10-*` §5m |
| **Fig 5** | Speckle probing: bar chart of partial R² by model | `rebuttals/10-*` §6e |
| **Fig 6** | Clinical impact: scatter plot of predicted vs true EF, stratified by severity bin | `rebuttals/10-*` §6d |

---

## Page Budget

| Section | Pages |
|---------|-------|
| Introduction | 1.5 |
| Experimental Design | 1.5 |
| Core Finding | 2.0 |
| Mechanism | 1.5 |
| EchoBench | 1.5 |
| Scaling | 0.5 |
| Discussion | 0.5 |
| **Total** | **9.0** |
