# EchoJEPA ICML Rebuttal Documentation

Active rebuttal materials for the EchoJEPA ICML 2026 submission. Pre-review docs (01-07) archived to `claude/archive/rebuttals/`.

## Active Files

| File | Purpose |
|------|---------|
| **`08-rebuttal-v2.md`** | **Primary rebuttal plan based on actual reviews (scores 2/3/3/4)** |
| `reviewer-comments.md` | Raw reviewer comments from OpenReview |
| `review-simulation-prompt.md` | Prompt for simulating review panel in Claude web app |

## Archived (pre-review, superseded by 08)

All in `claude/archive/rebuttals/`:

| File | Why archived |
|------|-------------|
| `claude-rebuttal-master.md` | Pre-review master doc; anticipated attacks were orthogonal to actual concerns |
| `01-paper-audit.md` | Pre-review vulnerability inventory; useful reference for appendix fixes |
| `02-rebuttal-template.md` | Pre-review rebuttal text; superseded |
| `03-worst-case-scenarios.md` | Pre-review scenarios; superseded |
| `04-competitive-positioning.md` | EchoJEPA vs concurrent work; useful reference for discussion period |
| `05-probe-fairness.md` | Probe fairness / Strategy E background; useful if probe questions arise |
| `06-claim-validity.md` | Claim audit; superseded |
| `07-camera-ready-actions.md` | Editorial action items; still useful for camera-ready |

## Key Principles

1. **This is an SSL insight paper, not an application paper.** Echo is the case study, not the contribution. The contribution is understanding when/why pixel reconstruction fails in stochastic-noise-dominated domains. Generalizes to radar, sonar, low-SNR microscopy.
2. **Mechanistic evidence is the primary scientific contribution.** Frame shuffling + CKA + noise probe = first characterization of noise-filtering mechanism in latent prediction. Elevates from "ablation" to "understanding."
3. **Objective choice matters more than scale.** 525K clips with the right objective beats 12M clips with text supervision.
4. **Actual reviewer concerns:** novelty (ALL), segmentation (hfQ1, 6t2T), speckle validation (ncQn), contrastive comparison (hfQ1). NOT the technical attacks we anticipated.
5. **ncQn is the most flippable reviewer** (3→4/5, ~75-80%) — CKA + noise probe + frame shuffling directly answers their explicit ask.
6. **BYOL-Video v2** is the only lever for hfQ1 (2→3, ~30-35%). Three contingency framings prepared.
7. **EchoJEPA-G gated release** upon acceptance — zero compute, addresses reproducibility.
8. **Goodfire authorship constraint:** Only frame shuffling (standard temporal ablation) can be included. All other Goodfire analyses reserved for Nature Medicine.
9. **Cross-modal prediction excluded** — NatMed Pillar 2, double submission risk.

## Related Documentation

- `../preprint/` — detailed probe analysis, encoder fairness, claim validity, hindsight recommendations
- `../architecture/probe-system.md` — probe architecture details
