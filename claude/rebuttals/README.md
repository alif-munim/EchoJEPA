# EchoJEPA ICML Rebuttal Documentation

Rebuttal materials for the EchoJEPA ICML 2026 submission (scores 2/3/3/4). **Rebuttal submitted Mar 31; all reviewers maintained original scores (Apr 4).** Decision now with AC. See `13-post-rebuttal-outcome.md` for full analysis.

## Active Files

| File | Purpose | Start here? |
|------|---------|-------------|
| **`13-post-rebuttal-outcome.md`** | **Post-rebuttal analysis** — reviewer responses, updated acceptance probability, lessons, path forward | **Start here** |
| `08-rebuttal-v2.md` | Rebuttal plan — reviewer concerns, narrative strategy, contingency framings, acceptance analysis | Strategy & framing |
| `11-rebuttal-task-tracker.md` | Task list — P0-P3 priorities, 35 completed experiments, execution log | Experiment status |
| `10-rebuttal-experiment-results.md` | Consolidated results — all numbers, key findings (§5), config↔checkpoint mapping (§6) | Latest numbers |
| `09-three-way-comparison-results.md` | 3-way comparison detail — epoch tables, BYOL architecture audit, LVEF/RVSP/CAMUS interpretation | Deep dive on 3-way |
| `12-checkpoint-reference.md` | All encoder and probe checkpoint paths | Finding checkpoints |
| `reviewer-comments.md` | Raw reviewer comments from OpenReview | — |
| `review-simulation-prompt.md` | Prompt for simulating review panel in Claude web app | — |

## Experiment Writeups

Experiment docs have been migrated to `claude/neurips/experiments/` for the NeurIPS resubmission. A copy of `frame-shuffling.md` remains here for reference. See `claude/neurips/experiments/` for all 7 experiment writeups.

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

## Post-Rebuttal Status (2026-04-04)

**All reviewers maintained scores: 2/3/3/4 = avg 3.0.** Revised acceptance probability: **10-20%** (down from pre-rebuttal 35-45%). Universal blocker: novelty/technical contribution. ncQn acknowledged experiments resolved their concerns but pivoted to novelty. See `13-post-rebuttal-outcome.md` for full analysis.

All rebuttal experiments (3-way comparison, EchoBench, speckle probing) feed directly into the Nature Medicine manuscript regardless of ICML outcome. The SALT implementation (added 2026-04-04) provides a fourth SSL paradigm for the Nature Medicine comparison.

## Key Principles (from rebuttal planning)

1. **This is an SSL insight paper, not an application paper.** Echo is the case study, not the contribution.
2. **Mechanistic evidence is the primary scientific contribution.** Speckle probing + noise robustness + anatomy-function dissociation.
3. **Objective choice matters more than scale.** 525K clips with the right objective beats 12M clips with text supervision.
4. **Universal reviewer concern: novelty.** All four reviewers cited this, including L8sp (4). Experiments were successful but insufficient to change the perception of the contribution type.
5. **EchoJEPA-G gated release** upon acceptance — 1.1B params, largest public echo FM.
6. **Cross-modal prediction excluded** — NatMed Pillar 2, double submission risk.

## Related Documentation

- **`../neurips/`** — NeurIPS 2025 resubmission plan. Consolidates rebuttal experiments + new SALT experiments into a reframed paper
- `../preprint/` — detailed probe analysis, encoder fairness, claim validity, hindsight recommendations
- `../architecture/probe-system.md` — probe architecture details
