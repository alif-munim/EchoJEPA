# PART I: Comprehensive Paper Audit

## Executive Summary

**Paper Strengths:**
- Novel, well-motivated hypothesis (latent prediction for noisy medical imaging)
- Large-scale pretraining (18M videos, largest echo corpus to date)
- Comprehensive evaluation (3 tasks, 4 sites, multiple axes)
- Strong results on sample efficiency and robustness
- Public model release for reproducibility

**Critical Vulnerabilities:**
- VideoMAE baseline severely underperforms literature (8.52 vs 3.78 MAE)
- Low learning rate configuration (~170x below standard)
- Appendix documentation inconsistencies
- Missing comparisons to echo-specific MAE methods (EchoCardMAE)
- Attentive probe architecture mismatch inflates view classification gaps ~2x for CNN baselines

**Key Defensive Assets:**
- Controlled comparison (EchoJEPA-L vs EchoMAE-L) eliminates all confounds except training objective
- Linear probes confirm model ranking: EchoJEPA-G (80.9%) > EchoJEPA-L (70.8%) > EchoMAE-L (59.2%) > EchoPrime (57.7%) > PanEcho (52.8%)
- Sample efficiency: EchoJEPA-G at 1% labels exceeds best baseline at 100%

---

## TIER 1: CRITICAL (Must Address Proactively)

### C1. EchoMAE-L LVEF Performance vs Literature

**The Problem:**

| Method | LVEF MAE | Dataset | Modifications |
|--------|----------|---------|---------------|
| EchoCardMAE | 3.78 | EchoNet-Dynamic | Sector masking, denoised targets, InfoNCE, 1600 epochs, fine-tuned |
| Echo-Vision-FM | ~3.9 | MIMIC-IV-Echo | 85% mask, STF-Net head, fine-tuned |
| Your EchoMAE-L | 8.52 | EchoNet-Dynamic | Vanilla VideoMAE, ~164 epochs, frozen probe |

**Anticipated Attack:** "Published VideoMAE methods achieve 3.78 MAE. Your baseline achieves 8.52 — over 2x worse. This invalidates your comparison."

**Defense Evidence:**
- EchoMAE-L achieves competitive RVSP (5.36 vs 5.01) — model isn't broken
- EchoPrime/PanEcho (properly trained) also achieve only 5.33/5.43 on LVEF — similar to EchoJEPA-L
- EchoCardMAE requires extensive modifications that explicitly work around reconstruction's limitations
- Frozen probing vs fine-tuning is a fundamental protocol difference

**Response:**
"We clarify several important distinctions:

**Protocol difference:** EchoCardMAE and Echo-Vision-FM fine-tune the encoder end-to-end; we use frozen probing to isolate representation quality. Prior work shows reconstruction methods benefit disproportionately from fine-tuning to discard noise patterns learned during pretraining.

**Baseline validity:** EchoMAE-L achieves competitive RVSP (5.36 vs 5.01 MAE), demonstrating meaningful representation learning. A broken model would fail across all tasks.

**Echo-specific modifications:** EchoCardMAE achieves strong LVEF through sector-aware masking, denoised reconstruction targets, and temporal alignment losses — modifications that explicitly compensate for reconstruction's tendency to learn speckle patterns. Our baseline tests whether vanilla reconstruction transfers to ultrasound; it does not.

**Supporting our hypothesis:** The fact that reconstruction requires extensive domain engineering (EchoCardMAE) while latent prediction succeeds without it (EchoJEPA) directly supports our claim about objective-domain alignment."

**Worst-Case Escalation:** Reviewer insists comparison is unfair.

**Fallback:** "We acknowledge this limitation and will add EchoCardMAE as an additional baseline in the camera-ready version, pending code availability. However, we emphasize that our core claim — about the superiority of latent prediction in the frozen-probe regime — remains supported by the existing evidence."

---

### C2. VideoMAE Learning Rate Configuration

**The Problem:**

| Setting | Value |
|---------|-------|
| Your LR | 8.79e-7 (base) -> 3.5e-6 (scaled) |
| Standard VideoMAE LR | 1.5e-4 (base) -> 6e-4 (scaled at batch 1024) |
| Gap | ~170x lower than standard |

**Anticipated Attack:** "The VideoMAE baseline used an extremely low learning rate. The performance gap may reflect optimization failure."

**Defense Evidence:**
- Model converged (loss 0.87 -> 0.27)
- RVSP performance is competitive (5.36 vs 5.01)
- ALL non-JEPA baselines cluster at 40-42% attentive probe accuracy (52-59% linear)

**Response:**
"We acknowledge the learning rate was conservative. However, we emphasize:

**Convergence:** EchoMAE-L converged normally (final loss: 0.27, down from 0.87), indicating successful optimization.

**Non-degenerate representations:** EchoMAE-L achieves competitive RVSP estimation (5.36 vs 5.01), demonstrating the model learned meaningful temporal features.

**Clustering pattern:** ALL non-JEPA baselines — including EchoPrime (12M videos, official training) and PanEcho (1M+ videos, official training) — show substantially lower view classification accuracy than EchoJEPA under both attentive probes (40-42%) and linear probes (52-59%). If VideoMAE's poor view classification reflected optimization failure, properly-trained baselines should perform substantially better. They do not.

The gap reflects objective-domain alignment, not optimization quality."

---

### C3. Appendix Inconsistencies

**The Problems:**

| Location | Stated | Actual | Issue |
|----------|--------|--------|-------|
| A.1 | Cooldown: 12,000 updates | 24,000 (80 epochs x 300) | Typo |
| A.2 vs A.3 | V-JEPA: 84,000 total | ~87,000 actual | Mismatch |
| S3.2 vs A.1 | 24 fps | 8 fps | Unclear which applies where |
| A.3 | LR: 8.79e-7 | After scaling: 3.5e-6 | Confusing presentation |

**Response:**
"We apologize for documentation errors in Appendix A. Corrections:
- **Cooldown updates:** A.1 should state 24,000 updates (80 epochs x 300 ipe), not 12,000.
- **Total updates:** EchoJEPA-L: ~87,000; EchoMAE-L: ~84,000. The 3.5% difference is within acceptable tolerance.
- **FPS:** Section 3.2 describes domain adaptations for EchoJEPA-G (24 fps). The compute-matched comparison (Appendix A) uses 8 fps for both models.
- **Learning rate:** We will clarify that the stated value (8.79e-7) is before batch-size scaling; the actual optimizer LR is 3.5e-6.

All corrections will appear in the camera-ready version. These are documentation errors; the experiments themselves were conducted correctly."

---

## TIER 2: HIGH PRIORITY (Likely to Be Raised)

### H1. Model Size Disparity

**The Problem:**

| Model | Params |
|-------|--------|
| EchoJEPA-G | ~1B |
| EchoJEPA-L | 307M |
| EchoMAE-L | 307M |
| EchoPrime | 35M |
| PanEcho | 28M |

**Anticipated Attack:** "The 10-30x size difference explains the performance gap."

**Defense Evidence:**

No performance gradient with size among non-JEPA models (attentive probe):

| Model | Params | View Acc |
|-------|--------|----------|
| PanEcho | 28M | 41.9% |
| EchoPrime | 35M | 42.1% |
| EchoMAE-L | 307M | 40.4% |

**Response:**
"If model size were determinative, we would expect: PanEcho (28M) < EchoPrime (35M) < EchoMAE-L (307M). Instead, all three achieve 40-42% — the 10x larger EchoMAE-L gains nothing.

The compute-matched comparison directly addresses this: EchoJEPA-L and EchoMAE-L use identical ViT-L architectures (307M params). The 45-point view classification gap is attributable only to the training objective.

We also note EchoPrime was trained on 23x more data (12M vs 525K videos). If scale compensated for objective, EchoPrime should substantially outperform EchoJEPA-L. It achieves half the accuracy."

---

### H2. Frozen-Probe Protocol Limitations

**Anticipated Attack:** "EchoPrime was designed for end-to-end use. Frozen probing disadvantages it unfairly."

**Response:**
"We acknowledge frozen probing is one evaluation paradigm. Our goal was to isolate representation quality. We note:
- **Practical value:** Frozen evaluation enables clinical researchers without large compute budgets to deploy foundation models.
- **Fair protocol:** All models underwent identical evaluation with the same probe architecture and hyperparameter grid.
- **Claim qualification:** We will revise 'state-of-the-art' to 'state-of-the-art frozen-backbone performance' where appropriate.
- **Precedent:** EchoPrime's original paper also reports frozen/linear probe results, suggesting this is a valid evaluation paradigm."

---

### H3. Missing Baseline Comparisons

**Anticipated Attack:** "You should compare to EchoFM, EchoCardMAE, DISCOVR, and other echo-specific SSL methods."

**Response:**
"We acknowledge this limitation. EchoFM, EchoCardMAE, and DISCOVR are concurrent or recent works. We note:
- **Existing diversity:** Our comparison spans supervised (PanEcho), contrastive (EchoPrime), and reconstruction (EchoMAE-L).
- **Cleanest test:** The EchoJEPA-L vs EchoMAE-L comparison controls all variables except the objective, providing the strongest evidence for our hypothesis.
- **Public release:** Our model and evaluation framework enable direct community comparison.

We will add EchoFM/EchoCardMAE comparisons in the camera-ready version if weights are publicly available."

---

### H4. Task-Specific Performance Pattern

**The Pattern:**

| Task | EchoMAE-L | Interpretation |
|------|-----------|----------------|
| RVSP (competitive) | 5.36 | Captures temporal/velocity dynamics |
| LVEF (terrible) | 8.52 | Misses precise spatial anatomy |
| View Class. (clusters) | 40.4% | Fails at semantic discrimination |

**Anticipated Attack:** "Why does EchoMAE do well on RVSP but terribly on LVEF? This inconsistency is confusing."

**Response:**
"This task-specific pattern is consistent with our hypothesis and provides important insight:
- **RVSP** requires detecting TR jet velocity from color Doppler intensity patterns — primarily temporal/intensity dynamics that pixel reconstruction captures well.
- **LVEF** requires precise LV boundary delineation and wall motion tracking — spatial anatomy that reconstruction conflates with speckle patterns.
- **View classification** requires semantic category discrimination — global structure that reconstruction does not explicitly optimize for.

The pattern demonstrates reconstruction is not useless — it captures certain features — but systematically fails on tasks requiring spatial precision or semantic understanding. Latent prediction succeeds on all three."

---

### H5. Attentive Probe Architecture Mismatch

**The Problem:** Attentive probes underperform linear probes for non-ViT models on view classification due to token starvation. See `05-probe-fairness.md` for full analysis.

**Key numbers:**

| Model | Tokens/clip | Linear (test) | Attentive (test) | Status |
|-------|-------------|---------------|------------------|--------|
| EchoJEPA-G | 1568 | 80.9% | 87.4% | Expected |
| EchoJEPA-L | 1568 | 70.8% | 85.5% | Expected |
| EchoMAE-L | 1568 | 59.2% | 40.4% | **Inverted** |
| EchoPrime | 1 (pre-pooled) | 57.7% | 42.1% | **Inverted** |
| PanEcho | 1 (pre-pooled) | 52.8% | 41.9% | **Inverted** |

Gap narrows from ~45pts (attentive) to ~23-28pts (linear) but EchoJEPA remains dominant.

---

### H6. Embedding Dimensionality Confound

**The Problem:** Linear probes on higher-dimensional embeddings have more parameters per output class.

| Model | embed_dim | Linear probe params per class |
|-------|-----------|------------------------------|
| EchoJEPA-G | 1408 | 1408 |
| EchoJEPA-L | 1024 | 1024 |
| EchoMAE-L | 1024 | 1024 |
| PanEcho | 768 | 768 |
| EchoPrime | 512 | 512 |

**Anticipated Attack:** "EchoJEPA-G's advantage partly comes from having 2.7x more probe parameters than EchoPrime."

**Response:**
"We acknowledge the dimensionality difference. However:
- The **controlled comparison** (EchoJEPA-L vs EchoMAE-L) uses identical 1024-d embeddings — the 11.6-point linear probe gap is not attributable to dimensionality.
- We will add a **PCA-512 baseline** (project all embeddings to 512-d before probing) in the camera-ready to control for this confound on system-level comparisons."

---

### H7. Pretraining Data Confound

**The Problem:**

| Model | Pretraining data |
|-------|-----------------|
| EchoJEPA-G | 18M UHN echos (domain-matched, massive) |
| EchoJEPA-L | MIMIC-IV-Echo (525K echos) |
| EchoMAE-L | MIMIC-IV-Echo (525K echos) |
| PanEcho | Own echo dataset (published) |
| EchoPrime | 12M videos (own dataset) |

**Anticipated Attack:** "EchoJEPA-G's advantage comes from 18M domain-matched echos, not the JEPA objective."

**Response:**
"The **controlled comparison** (EchoJEPA-L vs EchoMAE-L) uses identical MIMIC-IV-Echo data, isolating the objective. For the system-level comparison (EchoJEPA-G vs baselines), we acknowledge data is a confound and frame it as 'our best system vs their best system' rather than an objective comparison."

---

## TIER 3: MEDIUM PRIORITY (May Be Raised)

### M1. Narrow Hyperparameter Grid

**Response:** "Our grid follows the V-JEPA2 evaluation protocol. EchoJEPA outperforms baselines across all 6 configurations; the ranking is stable. The 45-point view classification gap is unlikely to close with hyperparameter tuning alone."

### M2. Synthetic-Only Robustness

**Response:** "We acknowledge this limitation (discussed in Section 5). Our perturbations are physics-grounded, not arbitrary. Multi-site evaluation provides some real distribution shift. Prospective validation remains future work."

### M3. Pediatric Zero-Shot Definition

**Response:** "Zero-shot means: (1) no pediatric data in pretraining corpus, (2) probes trained on adult data only, (3) frozen encoder + adult probe applied directly to EchoNet-Pediatric test split without any pediatric tuning."

---

## TIER 4: EDITORIAL FIXES (Low Priority)

| Issue | Location | Fix |
|-------|----------|-----|
| "encoder processes masked frames" | Fig 1 caption | "visible (unmasked)" |
| "40% less than reconstruction-based" | Contributions | "86% less than next-best baseline" |
| "reconstruction-based models" | Table 5 caption | "alternative foundation models" |
| 300M vs 307M params | S3.3 vs S4.1 | Standardize to 307M |
| Missing citations (?) | Appendix B.1, B.4.1 | Fix Albumentations, Smistad refs |
| Extra comma | Intro P2 | Remove after EchoPrime citation |
