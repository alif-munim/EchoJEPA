# ICML Preprint: Hindsight Recommendations

What we would change if re-doing the preprint from scratch, given everything learned about probe fairness, encoder output asymmetries, and confounded comparisons.

## 1. Lead with the controlled comparison

The EchoJEPA-L vs EchoMAE-L comparison is the paper's strongest result and should be the central experiment, not a supporting one. Same ViT-L backbone, same probe, same MIMIC data, same compute — the only variable is the training objective. This is what actually proves JEPA > MAE for echocardiography.

Structure the results as:
1. Controlled objective comparison (Table 1) — EchoJEPA-L vs EchoMAE-L
2. Scaling analysis (Table 2) — EchoJEPA-L vs EchoJEPA-G (does scale help within JEPA?)
3. System-level comparison (Table 3) — EchoJEPA-G vs published baselines (clearly framed as "our best vs their best", not as objective comparison)

The current structure buries the controlled comparison inside a larger table where EchoJEPA-G dominates everything, which invites the wrong takeaway.

## 2. Use linear probes as the primary evaluation

Report linear probe results in the main tables. Attentive probes can go in supplementary as "best achievable with learned pooling." This:

- Eliminates the token starvation confound entirely
- Produces defensible numbers for all models
- Makes a stronger scientific claim ("the information is linearly accessible") vs ("a 10M-parameter adapter can extract it")
- Aligns with the representation quality framing (frozen linear probes test what's in the embedding, not what a trainable adapter can learn)

If attentive probes are reported, include both linear and attentive for every model in the same table so reviewers can see the inversion pattern and assess it themselves.

## 3. Train EchoMAE-L on 18M UHN data

The current controlled comparison (EchoJEPA-L vs EchoMAE-L on MIMIC) is good but operates at the small scale. Training EchoMAE on the full 18M UHN corpus would let you make a much stronger claim: "at the same scale and data, JEPA still wins." Without this, a reviewer can argue that MAE might catch up at scale.

This is expensive but would make the objective comparison airtight at the scale that matters.

## 4. Control for embedding dimensionality

Add a PCA baseline: project all embeddings to 512-d (the smallest model's dim) before the linear probe. Report both raw-dim and PCA-512 results. If EchoJEPA still wins at 512-d, the dimensionality confound is eliminated. If the gap narrows substantially, that's important to know and report honestly.

This is a one-line experiment (PCA + retrain linear probe) and would preempt the most obvious reviewer objection for the system-level comparisons.

## 5. Add EchoJEPA-L to every table

Every table that shows EchoJEPA-G vs baselines should also show EchoJEPA-L. This gives reviewers a size-matched comparison point (300M ViT-L) alongside the 1B ViT-g result. When EchoJEPA-L at 300M still beats EchoPrime/PanEcho at 30-35M, the scale confound is reduced (10x vs 30x). When EchoJEPA-L beats EchoMAE-L at the same size, the objective comparison is clean.

## 6. Be explicit about what is and isn't controlled

Add a comparison taxonomy table early in the paper:

| Comparison | Controls for | Does NOT control for | Claim it supports |
|-----------|-------------|---------------------|-------------------|
| EchoJEPA-L vs EchoMAE-L | Architecture, probe, data, compute | Scale (both small) | JEPA objective > MAE objective |
| EchoJEPA-L vs EchoJEPA-G | Objective, probe | Architecture, scale, data | Scaling benefits within JEPA |
| EchoJEPA-G vs EchoPrime | Probe | Architecture, scale, data, objective, embed dim | System-level performance |
| EchoJEPA-G vs PanEcho | Probe | Architecture, scale, data, objective, embed dim | System-level performance |

This disarms reviewers by showing you understand the confounds rather than ignoring them.

## 7. Drop or reframe the "clustering pattern" narrative

The observation that all non-JEPA baselines cluster at ~40-42% attentive probe accuracy is partially a token starvation artifact. Linear probes spread them from 40-42% to 52-59%. The narrative that "all non-JEPA models hit the same ceiling regardless of architecture/scale/data" is weaker than presented.

Replace with: "Non-JEPA models show substantially lower linear probe accuracy (52-59%) compared to JEPA models (70-81%), with reconstruction-based models (EchoMAE-L: 59%) outperforming supervised models (PanEcho: 53%) on this metric." This is less dramatic but accurate.

## 8. Report baseline numbers more carefully

For PanEcho and EchoPrime, acknowledge the evaluation limitations:
- These models were designed for specific tasks with their own inference pipelines
- Evaluating them through a generic frozen probe (attentive or linear) may not reflect their best achievable performance
- The comparison tests "representation quality under a standardized protocol" not "which system works best in deployment"

A single sentence in the methods section would suffice and shows scientific maturity.

## 9. Include robustness analysis of the probe itself

Vary probe hyperparameters (depth, heads, learning rate, weight decay) and show that the model ranking is stable. The current paper trains multiple heads but only reports the best. Showing that EchoJEPA > baselines holds across probe configurations (not just the optimal one) strengthens the claim. Even a supplementary figure showing accuracy vs probe depth for each model would help.

## Summary: The minimal changes that matter most

If pressed for the highest-impact, lowest-effort changes:

1. **Report linear probe results in main tables** (eliminates the primary confound, ~1 day of compute)
2. **Add PCA-512 baseline** (eliminates dimensionality confound, ~1 hour)
3. **Add the comparison taxonomy table** (eliminates reviewer confusion, ~30 minutes of writing)

Everything else is incremental improvement. These three changes would make the preprint substantially more robust to review.

## Related Documents
- `claim-validity.md` — which current conclusions survive the fairness analysis
- `encoder-fairness.md` — encoder output comparison and confound analysis
- `probe-architecture-analysis.md` — attentive vs linear inversion root cause and rebuttal
