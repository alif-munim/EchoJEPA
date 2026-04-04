# Competitive Landscape for NeurIPS

Concurrent and related work that the NeurIPS paper must position against.

---

## US-JEPA (UCLA, ICML 2026 submission)

**Paper:** `claude/papers/us-jepa/example_paper.tex`

**What they do:** Apply I-JEPA (image-level, not video) to static ultrasound frames. Use SALT (frozen teacher) and USrc (ultrasound region conditioning). Evaluate on 8 classification tasks across 22 anatomies.

**Key differences from our work:**

| | US-JEPA | EchoJEPA (NeurIPS) |
|---|---|---|
| Temporal | Image-level (static frames) | Video-level (16-frame clips) |
| Teacher | SALT frozen teacher (URFM) | EMA + SALT (both tested) |
| Tasks | 8 classification tasks | 5+ task types (regression, segmentation, zero-shot transfer) |
| Perturbations | Generic (Gaussian blur, contrast, speckle) | Physics-informed (depth attenuation, acoustic shadow, haze) |
| Mechanistic analysis | None | Speckle probing, frame shuffling, information probing |
| Controlled comparison | JEPA vs MAE only | 4-way (JEPA vs BYOL vs MAE vs SALT) |
| Finding | JEPA > MAE for US classification | Rankings invert by task type |

**How to cite:** Independent, concurrent validation that latent prediction outperforms pixel reconstruction for ultrasound. Strengthens the general principle. The fact that US-JEPA uses SALT (frozen teacher) while we test both EMA and SALT under controlled conditions makes our comparison strictly more informative.

**Risk:** If US-JEPA is accepted at ICML and published before NeurIPS submission, it partially scoops the "JEPA for ultrasound" angle. Our differentiation: (1) video vs image, (2) understanding vs engineering, (3) 4-way vs 2-way, (4) mechanistic evidence.

---

## SALT Paper (Apple, arXiv 2509.24317)

**Paper:** `claude/papers/vjepa-salt/arxiv.tex`

**What they do:** Two-stage pretraining — train pixel reconstruction teacher (V-Pixel), freeze it, train student to predict frozen teacher's latents. Claim compute efficiency over EMA-based V-JEPA 2.

**Key claims:**
- Frozen teacher suffices (no EMA needed)
- Small teachers train large students (ViT-L teacher → ViT-G student)
- Student loss correlates with downstream accuracy (R²=0.95)
- Outperforms V-JEPA 2 on SSv2 and K400 at matched FLOPs

**How this fits our story:** SALT is the missing cell in our 2×2 design. We implemented it (`app/salt/`) and will train epoch-matched on MIMIC. The key question: does SALT filter noise like JEPA (latent target hypothesis) or like MAE (pixel teacher hypothesis)?

**How to cite:** Concurrent work proposing frozen teachers as alternative to EMA. We test their claims in a noise-dominated medical domain with mechanistic analysis they don't provide. Our SALT experiments directly validate or challenge their compute efficiency claims in a new setting.

---

## V-JEPA 2 / V-JEPA 2.1 (Meta)

**Papers:** `claude/papers/vjepa-2.tex`, `claude/papers/vjepa-2-1.tex`

**Role in our paper:** Upstream method. The architecture is the control variable — we use V-JEPA 2's ViT-L encoder for all four paradigms. V-JEPA 2.1 adds dense hierarchical supervision (predict all tokens + multi-layer heads) which we may evaluate as a P1 experiment.

**How to cite:** Method paper. We deliberately hold their architecture constant to isolate the prediction target.

---

## EchoPrime (CLIP-style contrastive, 12.1M clips)

**Role:** System-level baseline from ICML preprint. Uses CLIP-style contrastive learning with text supervision on 12.1M clips from 58K patients.

**Key comparison:** EchoJEPA-L outperforms EchoPrime on view classification (85.5% vs 42.1%) despite 23× less data and no text supervision.

**How to cite:** System-level reference showing that objective matters more than data scale. Not part of the controlled comparison (different architecture, different data, different supervision).

---

## PanEcho (ConvNeXt-Tiny, supervised pretraining)

**Role:** System-level baseline from ICML preprint. Supervised pretraining on labeled echo data from 5× more patients.

**How to cite:** System-level reference. Not part of the controlled comparison.

---

## Other Relevant Work

**DINOv2 (Meta):** Image-level self-distillation. Could serve as an EchoBench reference baseline (P2 experiment). Shows what image-only SSL learns from video frames.

**VideoMAE v2:** Pixel reconstruction at scale. Our MAE baseline uses the same objective. The SALT paper compares against VideoMAE v2 as well.

**Perception Encoder (Meta):** Uses features from a predefined layer + external SAM teacher. Relevant as concurrent SSL work but different domain (natural video).

---

## Positioning Summary

Our paper sits at the intersection of three concurrent trends:
1. **JEPA-family methods** (V-JEPA 2, US-JEPA, SALT) — we provide the most controlled comparison
2. **Medical foundation models** (EchoPrime, PanEcho, URFM) — we show objective matters more than scale
3. **SSL understanding papers** ("Do ViTs See Like CNNs?", scaling laws) — we provide domain-specific mechanistic evidence

The unique contribution: a 2×2 experimental design that isolates prediction target from teacher mechanism, tested with physics-based evaluation revealing failure modes invisible from clean benchmarks.
