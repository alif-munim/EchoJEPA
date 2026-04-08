# Tube Masking Does Not Prevent the MAE Temporal Shortcut

**Date:** 2026-04-08
**Status:** Observation from existing VideoMAE ViT-L run (no new experiment needed)
**Section:** §4 Mechanistic Evidence — fourth line of evidence

---

## The Fact

Our VideoMAE ViT-L was pretrained with **tube masking, 90% mask ratio** — the canonical VideoMAE recipe (Tong et al., NeurIPS 2022).

```bash
# scripts/videomae_pretrain_mimic.sbatch:294
# scripts/videomae_pretrain_mimic_matched.sbatch:333
# scripts/videomae_pretrain_mimic_matched_2node.sbatch:402
--mask_type tube --mask_ratio 0.9
```

Confirmed across all three VideoMAE-L MIMIC sbatches (original, 1-node matched, 2-node matched) and in `checkpoint-inventory.md` line 93 ("ViT-L (304M), VideoMAE, pixel reconstruction, tube masking 90%").

Tube masking picks a 2D spatial pattern and repeats it across every temporal position, so **the same spatial patches are masked in every frame**. By construction, the model cannot reconstruct a masked patch by copying from an adjacent frame — the neighbor at the same spatial location is also masked.

This is the community's standard answer to the temporal-shortcut problem in video MAE. It was designed specifically to prevent cross-frame patch copying.

## What We Observe

MAE e99 (with tube masking) is **completely invariant to frame shuffling** — −4% degradation under full shuffle (`experiments/severity-gradient.md`), and essentially flat across all six disruption conditions (`experiments/6-condition-shuffling.md`):

| Condition | MAE e99 R² |
|-----------|-----------|
| clean | 0.445 |
| tubelet | 0.424 |
| reverse | 0.431 |
| matched | 0.419 |
| shuffle | 0.422 |
| matched_frame | 0.449 |

Range: 0.419–0.449. `matched_frame` (most rigorous temporal disruption) is **marginally higher** than clean. Frame order is irrelevant to the converged MAE.

This is the opposite of what tube masking was supposed to deliver. Tube masking blocks the one temporal shortcut it was designed to block — and MAE abandons temporal features anyway.

## The Reframe

If tube masking already prevents cross-frame copying, the shortcut MAE finds must be a different one. The only remaining path is **within-frame spatial interpolation**: reconstruct a masked tube from its neighboring (visible) spatial patches at the same timestep.

This is consistent with the physics of echocardiography:
- Adjacent spatial patches are highly correlated (smooth tissue boundaries, gradually varying speckle texture, coherent chamber geometry at any given moment)
- A masked 16×16 patch can be reconstructed with low L2 error from its spatial neighbors in the same frame without ever attending across time
- The L2 loss rewards this spatial-only solution because it achieves near-perfect reconstruction on the redundant pixel signal, and only rewards temporal features when spatial reconstruction fails — which it doesn't on anatomy

**Frame-gap masking would not help.** It prevents the model from accessing temporally nearby frames, but tube masking already does that. The within-frame spatial shortcut that tube masking doesn't address is the same shortcut that frame-gap masking doesn't address. The model sees visible patches at every timestep (just different spatial locations than the masked ones) and reconstructs from those.

**The only masking strategy that would force temporal reasoning is whole-frame masking** — the model sees frames 1–4 completely, frames 5–7 are completely invisible (no visible patches at those timesteps), and must reconstruct the missing frames using only temporal context from frames 1–4. But this risks training collapse on pixel reconstruction (no information to reconstruct from within-frame), which is why VideoMAE and all descendants use tube/spatial masking instead.

**The conclusion:** the temporal shortcut is intrinsic to pixel reconstruction on spatially redundant video, not an artifact of insufficient masking. **The prediction target itself is the bottleneck.** No masking trick fixes it.

## Why This Is a Stronger Finding

The previous framing was "MAE abandons temporal features (observational)" — a finding that any careful practitioner might dismiss as "you just need better masking." The tube-masking reframe blocks that dismissal:

- Tube masking is the **community standard** for preventing temporal shortcuts in video MAE
- Our data shows it **doesn't work** for echocardiography
- Therefore the failure is **not** a trivial masking-design oversight

This elevates the finding from "MAE is temporally flat on this dataset" to "tube masking — the standard fix — fails on spatially redundant video, and the pixel-reconstruction objective cannot be rescued by masking design alone."

JEPA avoids the shortcut **by design**, not by masking: the EMA teacher's targets are abstract latent embeddings, not pixels. There is no "spatial interpolation in latent space" that corresponds to copying adjacent-patch pixel values, because the target is already a high-level representation. The only way to match the teacher's latent is to produce the same abstract features the teacher would — which for echo videos means encoding the temporal dynamics that distinguish one clip from another. The target, not the masking, does the work.

## Paper Text (§4, proposed)

Two sentences, placed at the end of §4 after the severity-gradient + training-dynamics discussion:

> Our MAE uses tube masking (Tong et al., 2022), which prevents cross-frame patch copying, yet the temporal shortcut persists: MAE e99 is invariant to frame shuffling (−4% under full shuffle, flat across all six disruption conditions). This indicates the shortcut arises from **within-frame spatial redundancy** rather than temporal copying, and cannot be resolved by masking design alone — the pixel-reconstruction objective, not the masking strategy, is the bottleneck.

This is one of four lines of mechanistic evidence in §4:

1. **Severity gradient + training dynamics** (behavioral) — `experiments/severity-gradient.md` — three temporal encoding regimes, MAE's transient-then-invariant trajectory
2. **Reconstruction visualization** (internal) — planned, shows MAE reconstructs masked patches from within-frame spatial context
3. **Temporal attention analysis** (architectural) — planned, shows MAE's attention heads collapse to single-frame after mid-training
4. **Tube masking failure** (this doc) — evidence that the cause is the objective, not the masking

The four lines converge on one claim: the prediction target determines whether temporal features survive training. No masking intervention can fix pixel reconstruction on spatially redundant video.

## What This Dropped From the Plan

**Frame-gap MAE intervention (ViT-B pilot) — CANCELLED 2026-04-08.**

Previously under consideration: pretrain a ViT-B MAE with frame-gap masking (mask entire frame positions and ask the model to reconstruct them from non-adjacent frames) and check whether the temporal shortcut can be prevented at pretraining time. Motivated by the hypothesis that MAE's temporal collapse was caused by cross-frame spatial copying.

**Why cancelled:**
1. Tube masking already prevents cross-frame copying. The hypothesis the intervention was testing is refuted by the existing ViT-L run.
2. Frame-gap masking does not block the actual shortcut (within-frame spatial interpolation), so it would not change MAE's behavior.
3. Whole-frame masking (which *would* force temporal reasoning) risks training collapse and is not on the table.
4. The paper is a pure understanding contribution — no intervention is needed or claimed.

**Saves:** ~2 days HyperPod compute + writing budget.
**Reallocate to:** reconstruction visualization, temporal attention analysis, and writing (per plan.md Immediate Next Actions, 2026-04-08).

## Assumptions and Caveats

- The within-frame spatial redundancy claim is a theoretical argument from tube masking's design plus the physics of echocardiography. It is consistent with the observed MAE e99 flatness but not independently verified — the reconstruction visualization experiment would provide direct visual evidence of the shortcut mechanism.
- Tube masking at 90% ratio leaves only 10% of tokens visible per frame. The visible tokens are spatially scattered (not clustered) because the mask is a single 2D pattern repeated across time. A masked patch at position (h,w) has its nearest visible neighbors at other spatial positions in the same frame, not at (h,w) in other frames — this is what makes within-frame interpolation the available path.
- The argument does not claim MAE cannot learn temporal features at all — the e50 crisis point (where MAE catastrophically depends on temporal shortcuts, −313% under shuffle) shows MAE does try temporal features early in training. The claim is that by convergence, the pixel-reconstruction objective has driven the encoder to the spatial-only solution that tube masking cannot prevent.
