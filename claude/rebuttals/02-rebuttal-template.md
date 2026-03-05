# PART II: Master Rebuttal Template

Ready-to-adapt text for reviewer responses.

---

## Opening

We thank all reviewers for their thorough and constructive feedback. We address the key concerns below and commit to incorporating all corrections in the camera-ready version.

---

## Core Defense: The Controlled Comparison

*[Use this as your anchor argument throughout]*

We emphasize that our central evidence comes from the **controlled comparison**: EchoJEPA-L vs EchoMAE-L, using identical ViT-L architectures (307M params), identical MIMIC-IV-Echo pretraining data (525K videos), and identical evaluation protocols. The only variable is the training objective.

| Model | Objective | Params | Data | View Acc (attentive) | LVEF MAE |
|-------|-----------|--------|------|---------------------|----------|
| EchoJEPA-L | Latent prediction | 307M | 525K | 85.5% | 5.97 |
| EchoMAE-L | Reconstruction | 307M | 525K | 40.4% | 8.15 |

This 45-point view classification gap with all variables controlled provides strong evidence that objective-domain alignment, not optimization or scale, determines representation quality on ultrasound.

**Supporting evidence from system-level comparisons** (linear probes, which eliminate probe architecture confounds):

| Model | Objective | Params | Training Data | View Acc (linear) |
|-------|-----------|--------|---------------|-------------------|
| EchoJEPA-G | Latent prediction | ~1B | 18M UHN | 80.9% |
| EchoJEPA-L | Latent prediction | 307M | 525K MIMIC | 70.8% |
| EchoMAE-L | Reconstruction | 307M | 525K MIMIC | 59.2% |
| EchoPrime | Contrastive | 35M | 12M videos | 57.7% |
| PanEcho | Supervised | 28M | 1M+ videos | 52.8% |

Note: System-level comparisons do not control for scale, data, or embedding dimensionality. They show EchoJEPA's advantage but should be interpreted as "our best vs their best," not as evidence for JEPA superiority in isolation.

---

## Response to Compute-Matched Comparison Concerns

### VideoMAE Learning Rate

We acknowledge the conservative learning rate. However:
- **Convergence:** EchoMAE-L converged normally (loss 0.87 -> 0.27).
- **Non-degenerate representations:** Competitive RVSP (5.36 vs 5.01) demonstrates meaningful learning.
- **Clustering pattern:** Properly-trained EchoPrime and PanEcho achieve similar view classification accuracy (52-59% linear probe), indicating the gap reflects objective limitations, not optimization failure.

### Comparison to Literature (EchoCardMAE)

EchoCardMAE achieves 3.78 MAE through:
- Sector-aware masking
- Denoised reconstruction targets
- Temporal alignment losses (InfoNCE)
- 1600 epochs (vs our 164)
- End-to-end fine-tuning (vs our frozen probing)

These modifications explicitly compensate for reconstruction's tendency to learn speckle. The fact that reconstruction requires such engineering while latent prediction succeeds without it supports our hypothesis.

### Documentation Errors

We apologize for inconsistencies in Appendix A:
- **Cooldown:** 24,000 updates (not 12,000)
- **FPS:** 24 for EchoJEPA-G, 8 for compute-matched comparison
- **Total updates:** V-JEPA ~87K, VideoMAE ~84K (within 4%)

All corrections will appear in camera-ready.

---

## Response to Baseline Fairness Concerns

### Model Size

No performance gradient exists among non-JEPA models: the 10x larger EchoMAE-L (307M) achieves similar linear probe accuracy (59.2%) to EchoPrime at 35M (57.7%). The compute-matched comparison (identical 307M architecture) isolates the objective.

### Frozen Protocol

We qualify claims to "frozen-backbone performance." Frozen evaluation isolates representation quality and has practical value for resource-constrained clinical deployment. All models underwent identical evaluation.

### Probe Architecture

We report both attentive and linear probe results. The attentive probe is architecturally optimized for ViT encoders with dense spatiotemporal tokens; CNN-based baselines output pre-pooled single-token embeddings, which causes cross-attention to degenerate. Linear probes on mean-pooled embeddings eliminate this confound and preserve the same model ranking.

### Missing Baselines

Our comparison spans supervised, contrastive, and reconstruction objectives. We will add EchoFM/EchoCardMAE if weights are available.

---

## Response to Task-Specific Pattern

EchoMAE-L's performance pattern (competitive RVSP, poor LVEF, poor views) is consistent with our hypothesis:
- **RVSP:** Requires temporal/velocity dynamics -> reconstruction captures this
- **LVEF:** Requires spatial anatomy -> reconstruction conflates with speckle
- **Views:** Requires semantic categories -> reconstruction doesn't optimize for this

This pattern would persist regardless of learning rate; it reflects what the objective incentivizes.

---

## Summary of Revisions

| Category | Action |
|----------|--------|
| Documentation | Fix all Appendix A inconsistencies |
| Claims | Qualify "SOTA" to "frozen-backbone SOTA" |
| Editorial | Fix Fig 1 caption, Table 5 caption, contributions bullet, citations |
| Baselines | Add EchoFM/EchoCardMAE comparison if feasible |
| Probe results | Report both attentive and linear probe numbers |
| Dimensionality | Add PCA-512 baseline or report embed dims in all tables |

---

## Closing

The controlled comparison — EchoJEPA-L vs EchoMAE-L with identical architecture, data, and evaluation — demonstrates a 45-point view classification advantage attributable solely to the training objective. System-level comparisons with linear probes confirm the same ranking across all models (80.9% vs 52-59%). We believe EchoJEPA makes a significant contribution to understanding how pretraining objectives should be matched to domain characteristics in medical imaging.
