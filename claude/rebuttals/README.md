# EchoJEPA ICML Rebuttal Documentation

Active rebuttal materials for the EchoJEPA ICML 2026 submission. Pre-review docs (01-07) archived to `claude/archive/rebuttals/`.

## Active Files

| File | Purpose | Start here? |
|------|---------|-------------|
| **`08-rebuttal-v2.md`** | **Primary rebuttal plan** — reviewer concerns, narrative strategy, contingency framings, acceptance analysis | Strategy & framing |
| **`11-rebuttal-task-tracker.md`** | **Canonical task list** — P0-P3 priorities, DONE/RUNNING/NOT STARTED, day-by-day execution plan | **What to do next** |
| `10-rebuttal-experiment-results.md` | **Consolidated results** — all numbers, running jobs, key findings (§5), config↔checkpoint mapping (§6) | Latest numbers |
| `09-three-way-comparison-results.md` | 3-way comparison detail — epoch tables, BYOL architecture audit, LVEF/RVSP/CAMUS interpretation | Deep dive on 3-way |
| `12-checkpoint-reference.md` | All encoder and probe checkpoint paths | Finding checkpoints |
| `reviewer-comments.md` | Raw reviewer comments from OpenReview | — |
| `review-simulation-prompt.md` | Prompt for simulating review panel in Claude web app | — |

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
