# EchoJEPA ICML Rebuttal Documentation

Organized rebuttal materials for the EchoJEPA ICML preprint. Each file is self-contained with cross-references.

## Files

| File | Purpose | Status |
|------|---------|--------|
| **`08-rebuttal-v2.md`** | **Active rebuttal plan based on actual reviews (2026-03-25)** | **PRIMARY** |
| `01-paper-audit.md` | Pre-review vulnerability inventory (TIER 1-4 issues) | Reference |
| `02-rebuttal-template.md` | Pre-review rebuttal text (largely superseded by 08) | Superseded |
| `03-worst-case-scenarios.md` | Pre-review scenarios 1-7 (largely superseded by 08) | Superseded |
| `04-competitive-positioning.md` | EchoJEPA vs EchoCardMAE / Echo-Vision-FM / EFNet | Reference |
| `05-probe-fairness.md` | Probe fairness: ICML inversion debunked, d=1 verification, Strategy E | Reference |
| `06-claim-validity.md` | Which claims are bulletproof vs confounded (largely superseded by 08) | Superseded |
| `07-camera-ready-actions.md` | Final assessment + prioritized action items | Reference |

## Key Principles

1. **Lead with the controlled comparison** (EchoJEPA-L vs EchoMAE-L) — it's bulletproof
2. **The actual reviewer concerns were about novelty, evaluation breadth, and mechanistic evidence** — NOT the technical ML attacks we anticipated (probe fairness, VideoMAE LR, dimensionality)
3. **EchoJEPA-G gated release upon acceptance** — UHN approved, transforms reproducibility from weakness to strength
4. **Speckle invariance experiments** (CKA + noise-level probing) are the highest-ROI new experiments — directly address the most flippable reviewer (ncQn)
5. **Frame shuffling** (from Goodfire report, re-run independently) is the AC champion sentence — pure ML, no NatMed overlap
6. **Cross-modal prediction excluded** from ICML rebuttal — NatMed Pillar 2, double submission risk
7. See `claude/goodfire/goodfire_mar20.pdf` for representation analyses; frame shuffling + temporal Fourier power are rebuttal-relevant, SAE results reserved for NatMed

## Related Documentation

- `../preprint/` — detailed probe analysis, encoder fairness, claim validity, hindsight recommendations
- `../architecture/probe-system.md` — probe architecture details
