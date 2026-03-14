# EchoJEPA ICML Rebuttal Documentation

Organized rebuttal materials for the EchoJEPA ICML preprint. Each file is self-contained with cross-references.

## Files

| File | Purpose |
|------|---------|
| `01-paper-audit.md` | Comprehensive vulnerability inventory (TIER 1-4 issues) |
| `02-rebuttal-template.md` | Ready-to-submit rebuttal text for reviewer responses |
| `03-worst-case-scenarios.md` | Scenarios 1-7: what if reviewers push hard on X? |
| `04-competitive-positioning.md` | EchoJEPA vs EchoCardMAE / Echo-Vision-FM / EFNet |
| `05-probe-fairness.md` | Probe fairness: ICML inversion debunked, d=1 verification results, Strategy E justification |
| `06-claim-validity.md` | Which claims are bulletproof vs confounded |
| `07-camera-ready-actions.md` | Final assessment + prioritized action items |

## Key Principles

1. **Lead with the controlled comparison** (EchoJEPA-L vs EchoMAE-L) — it's bulletproof
2. **Acknowledge confounds honestly** — claim qualification builds credibility
3. **Use d=1 attentive probes as primary evaluation** — verified to help ALL models, cite V-JEPA precedent + own verification. Report linear in Extended Data for transparency
4. **The ICML attentive inversion was an artifact** — normalization bugs, identical HP grids, depth=4 degeneration. d=1 with proper training helps every model including EchoPrime (+9.3pp) and PanEcho (+7.1pp)

## Related Documentation

- `../preprint/` — detailed probe analysis, encoder fairness, claim validity, hindsight recommendations
- `../architecture/probe-system.md` — probe architecture details
