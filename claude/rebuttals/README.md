# EchoJEPA ICML Rebuttal Documentation

Organized rebuttal materials for the EchoJEPA ICML preprint. Each file is self-contained with cross-references.

## Files

| File | Purpose | Status |
|------|---------|--------|
| **`08-rebuttal-v2.md`** | **Active rebuttal plan based on actual reviews (2026-03-26)** | **PRIMARY** |
| `01-paper-audit.md` | Pre-review vulnerability inventory (TIER 1-4 issues) | Reference |
| `02-rebuttal-template.md` | Pre-review rebuttal text (largely superseded by 08) | Superseded |
| `03-worst-case-scenarios.md` | Pre-review scenarios 1-7 (largely superseded by 08) | Superseded |
| `04-competitive-positioning.md` | EchoJEPA vs EchoCardMAE / Echo-Vision-FM / EFNet | Reference |
| `05-probe-fairness.md` | Probe fairness: ICML inversion debunked, d=1 verification, Strategy E | Reference |
| `06-claim-validity.md` | Which claims are bulletproof vs confounded (largely superseded by 08) | Superseded |
| `07-camera-ready-actions.md` | Final assessment + prioritized action items | Reference |

## Key Principles

1. **Lead with the falsifiable hypothesis** — pixel reconstruction fails because US pixels are dominated by noise; objective choice matters more than scale. Own the architecture point (not novel), pivot to what IS novel (hypothesis + controlled evidence + mechanistic proof)
2. **The actual reviewer concerns were about novelty, evaluation breadth, and mechanistic evidence** — NOT the technical ML attacks we anticipated (probe fairness, VideoMAE LR, dimensionality)
3. **EchoJEPA-G gated release upon acceptance** — UHN approved, transforms reproducibility from weakness to strength
4. **BYOL-Video is the primary contrastive baseline** (replaces generic DINO/BYOL); isolates local vs global prediction targets in a three-way controlled comparison (JEPA vs BYOL vs MAE)
5. **Speckle invariance experiments** (CKA + noise-level probing) are the highest-ROI new experiments — directly address the most flippable reviewer (ncQn)
6. **Frame shuffling** (from Goodfire report, re-run independently) is the AC champion sentence — pure ML, no NatMed overlap. Also the key differentiator: video-level temporal encoding that frame-level approaches structurally cannot match.
7. **Video-level is the fundamental differentiator** — EchoBench uses echo-specific physics-based perturbations (depth attenuation, acoustic shadow, haze) + temporal corruptions (frame drop, jitter, speed variation) across classification, regression, and segmentation on EchoNet-Dynamic + EchoNet-Pediatric. Strictly more comprehensive than concurrent frame-level evaluations (US-JEPA: 3 generic corruptions, classification only). Temporal corruptions are structurally impossible for frame-level methods.
8. **Cross-modal prediction excluded** from ICML rebuttal — NatMed Pillar 2, double submission risk
9. **Do NOT cite concurrent frame-level JEPA work** — our paper has arxiv priority; only respond if raised by reviewers
10. **Goodfire report authorship constraint:** ICML author list is fixed. Only frame shuffling (independently reproducible, standard temporal ablation) can be included in the rebuttal. All other Goodfire-specific analyses (intrinsic dim, SVD, Fourier power, attribution, SAE) are excluded from ICML — reserve for NatMed where collaborators can be added as authors. See `claude/goodfire/goodfire_mar20.pdf` for the full report.

## Related Documentation

- `../preprint/` — detailed probe analysis, encoder fairness, claim validity, hindsight recommendations
- `../architecture/probe-system.md` — probe architecture details
