# Competitive Positioning: EchoJEPA vs Related Work

How EchoJEPA differs from EchoCardMAE, Echo-Vision-FM, and EFNet. Use this for framing in the paper and rebuttals.

---

## Core Differentiators

| Aspect | EchoJEPA | Echo-Vision-FM / EchoCardMAE | EFNet |
|--------|----------|------------------------------|-------|
| Objective | Latent prediction (JEPA) | Pixel reconstruction (MAE) | Supervised |
| Evaluation | Frozen backbone | Fine-tuned encoder | End-to-end |
| Scale | 18M videos | ~10K-100K | ~10K |
| Domain modifications | None needed | Extensive (denoised targets, sector masking, alignment loss) | N/A |

---

## 1. Training Objective: Latent vs Pixel Prediction

This is the core scientific contribution.

| Method | What it predicts | Problem with ultrasound |
|--------|-----------------|------------------------|
| EchoCardMAE / Echo-Vision-FM | Masked pixels | Must faithfully reconstruct speckle noise |
| EchoJEPA | Masked embeddings | EMA target naturally suppresses stochastic noise |

EchoCardMAE achieves good results, but only by adding three modifications that explicitly work around reconstruction's limitations:
1. Denoised reconstruction targets (median blur to remove speckle)
2. Key area masking (focus on cardiac sector, ignore background)
3. Temporal alignment loss (InfoNCE to enforce cycle-invariance)

EchoJEPA needs none of these. The JEPA objective inherently downweights unpredictable artifacts.

**Argument:** "The fact that reconstruction requires extensive domain engineering while latent prediction succeeds without it supports our hypothesis about objective-domain alignment."

---

## 2. Evaluation Protocol: Frozen vs Fine-Tuned

This is a critical methodological difference that makes raw MAE comparison misleading.

| Method | Protocol | What it tests |
|--------|----------|---------------|
| EchoCardMAE | Fine-tune encoder + head | End-to-end system performance |
| Echo-Vision-FM | Fine-tune encoder + STF-Net | End-to-end system performance |
| EFNet | Train from scratch | Supervised learning capacity |
| EchoJEPA | Freeze encoder, train probe only | Representation quality |

Frozen evaluation is harder — the encoder can't adapt to the task. Yet EchoJEPA-G achieves 3.97 MAE on Stanford with a frozen backbone, competitive with fine-tuned methods (3.7-3.87).

**Argument:** "Our frozen-probe protocol isolates representation quality. Achieving comparable MAE to fine-tuned methods while keeping the encoder frozen demonstrates that EchoJEPA learns superior representations out of the box."

---

## 3. Scale: 18M vs ~10K Videos

| Model | Pretraining Data | Labeled Fine-tuning |
|-------|-----------------|---------------------|
| EchoCardMAE | 7,465 videos (EchoNet-Dynamic train) | Same |
| Echo-Vision-FM | MIMIC-IV-Echo (~100K?) | ~8,753 (EchoNet-Dynamic) |
| EFNet | None (supervised) | ~8,000 |
| EchoJEPA-G | 18.1M videos | Frozen probe on ~8,753 |
| EchoJEPA-L | 525K (MIMIC-IV-Echo) | Frozen probe on ~8,753 |

Even EchoJEPA-L (525K videos, public data) outperforms these models on view classification and shows competitive LVEF.

---

## 4. Multi-View Reasoning

Novel contribution that others don't address.

| Model | View handling |
|-------|-------------|
| EchoCardMAE | Single view (A4C) |
| Echo-Vision-FM | Single view |
| EFNet | Single view |
| EchoJEPA | Multi-view probing framework with factorized stream embeddings |

Enables:
- RVSP estimation (requires A4C + PSAX-AV integration)
- Robustness to missing views (view dropout during training)
- Standardized evaluation across models

---

## 5. Clinical Breadth

| Model | Tasks | Sites |
|-------|-------|-------|
| EchoCardMAE | LVEF, segmentation | EchoNet-Dynamic only |
| Echo-Vision-FM | LVEF, classification | EchoNet-Dynamic, CAMUS |
| EFNet | LVEF | EchoNet-Dynamic |
| EchoJEPA | LVEF, RVSP, view classification, robustness, pediatric transfer | Toronto, Chicago, Stanford, EchoNet-Pediatric |

---

## Summary: Unique Positioning

1. **OBJECTIVE:** Latent prediction instead of pixel reconstruction — no need for echo-specific hacks (denoised targets, sector masking); inherently ignores stochastic speckle

2. **PROTOCOL:** Frozen backbone evaluation — tests representation quality, not fine-tuning capacity; achieves 3.97 MAE frozen vs ~3.8 MAE fine-tuned for others

3. **SCALE:** 18M videos (35x larger than next biggest) — even public EchoJEPA-L (525K) shows strong results

4. **MULTI-VIEW:** Novel probing framework — enables RVSP and other multi-view tasks; others limited to single-view A4C

5. **BREADTH:** 3 tasks, 4 sites, robustness + pediatric transfer — most comprehensive evaluation in the literature

---

## Suggested Paper/Rebuttal Language

"Unlike prior VideoMAE-based approaches (EchoCardMAE, Echo-Vision-FM) which require domain-specific modifications — denoised reconstruction targets, sector-aware masking, temporal alignment losses — to achieve strong performance, EchoJEPA succeeds without any such engineering. This supports our hypothesis that the JEPA objective is inherently better aligned with ultrasound's signal properties.

Furthermore, we evaluate under a frozen-probe protocol that isolates representation quality, whereas prior methods fine-tune the encoder. EchoJEPA-G achieves 3.97 MAE on Stanford with frozen backbones, competitive with fine-tuned methods reporting 3.7-3.87 MAE, demonstrating that latent prediction yields superior representations without task-specific adaptation."
