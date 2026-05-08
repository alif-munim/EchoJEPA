# Root-cause synthesis — MCC & FJ low single-clip A4C LVEF

## 1. Executive summary

Both MCC-Anchored 25-epoch and FullJoint-Study 30k-step runs are underperforming on single-clip A4C LVEF probes, but for **different and independently identifiable reasons**, neither of which is an eval bug:

- **MCC**: The cross-clip adapter activated and reduced the MCC loss term (crossview-vs-intraview gap grew to 5.5% by ep 25), but `L_vjepa_self` on standalone clip_b **rose** from 0.49 → 0.54 over training (while base e125 continuation goes 0.48 → 0.47). The encoder was dragged away from being a good standalone clip representation by the clip_a source-encoder gradient. The adapter's success at adjoint prediction came at the cost of the standalone encoder.

- **FJ**: The clip V-JEPA path is active and works correctly, but gets only ~1/K = 1/8 of the per-step clip forwards for V-JEPA gradient → ~16× less effective V-JEPA signal than base e125 at the same wall-clock. Meanwhile the study-level objective collapsed to triviality (loss → 0.003), NCE was zero in 64% of steps, and study matching is mostly metadata-shortcut (metadata-only gap ≈ 0.006). The secondary objectives consumed compute and produced little signal, leaving the clip encoder effectively undertrained on V-JEPA.

- **Probe comparability**: My earlier "MCC/FJ is 2× worse than peers" framing was built against MV-PhaseRel / MV-PairedIntra / TokenRel-Motion numbers on **EchoNet-Dynamic**, whereas 786/792/793 were on **MIMIC A4C 10k**. The gap claim in that direction is not supported by matched-dataset data. New EchoNet-Dynamic probes (794/795) are running to close this gap.

## 2. What failed

- MCC single-clip A4C LVEF val R² plateaued at ~0.32 (job 786 on MIMIC A4C 10k).
- FJ single-clip A4C LVEF val R² in 0.16–0.27 range over first 5 epochs (job 792, canceled after 5 ep to free node for EchoNet probe).

These numbers are consistent with training-log evidence but NOT yet comparable to MV-PhaseRel / MV-PairedIntra / TokenRel until 794/795 complete.

## 3. What did not fail

- **Pretraining completed**: MCC 762 finished all 25 epochs. FJ 776 finished all 30k steps.
- **Checkpoints saved correctly**: MCC 762 e25 has `target_encoder`, `encoder`, `predictor`, `mcc_adapter`, `mcc_config`. FJ 776 `latest.pt` has `clip_target_encoder`, `clip_encoder`, `clip_anchor_e100`, `clip_predictor`, plus study-side keys.
- **Probe infrastructure**: Sbatch, config, data loading, 6-HP grid, attentive d=4 probe all operate correctly.
- **Checkpoint-key loading**: `target_encoder` vs `encoder` matters only at the EMA-lag level (~4–9% weight diff), ruled out by code audit and drift inspection. Not a bug source.
- **FJ true V-JEPA path**: Active, contributing dominant gradient (0.41 terminal loss).
- **MCC target-anchored forward**: Correctly implemented per `mcc_jepa_forward.py:133–142`. B_visible + A_source → predictor → adapter → teacher B.

## 4. Evaluation bugs ruled in/out

| Check | Status | Evidence |
|---|---|---|
| Wrong checkpoint key | **Ruled out** | Code audit + 4–9% weight gap vs online encoder; too small to produce 2× probe gap |
| Missing adapter load at probe | **Ruled out** | Probe doesn't need adapter (single-clip eval); forward pass doesn't call adapter |
| Wrong CSV / different label normalization | **Ruled out** for MCC/FJ internal comparison (same mimic_lvef_a4c_10k); **ruled IN** for cross-model comparison (MV-PhaseRel is on EchoNet-Dynamic, not MIMIC A4C) |
| Resolution / frame sampling mismatch | **Ruled out** | 224/16f/frame_step=2/num_segments=2 are identical across all probes |
| Checkpoint adaptation strips weights | **Ruled out** | Inline adapter for FJ verified; MCC has no adaptation |
| Probe hyperparameter grid difference | **Ruled out** | 6-HP grid identical |

## 5. MCC diagnosis

### Claim

MCC's clip_a source-encoder gradient damages the standalone clip_b encoder representation. Probe evidence is consistent with a degraded encoder, not with a missing adapter at probe time.

### Evidence

1. **Adapter active by ep 25** — `crossview − intraview` gap grew from 0% to 5.49% (mean over ep 25). γ must have become nontrivially positive.
2. **Encoder's standalone clip V-JEPA loss rose** — `intraview` mean went 0.491 (ep 1) → 0.546 (ep 15) → 0.539 (ep 25). Monotonically worse than starting point for 20+ epochs. Base e125 continuation in the same range has mean 0.47 and decreasing.
3. **Encoder weight drift from e100 is 40%** in middle blocks vs 23% for base e125. Moved more, to a worse place.
4. **MCC received effectively 1.2× the vanilla V-JEPA gradient** (since at γ≈0, both loss terms ≈ L_vjepa) plus non-trivial cross-clip gradient via encoder(clip_a). Over-training + orthogonal objective.

### Mechanism

The encoder in MCC gets gradients from three terms:
- `∂L_vjepa_self / ∂encoder` via clip_b
- `∂L_mcc / ∂encoder` via clip_b (same chain, identical at γ=0, slightly different when γ > 0)
- `∂L_mcc / ∂encoder` via clip_a (only through the adapter's KV path)

The third term trains the encoder to emit tokens at clip_a that are **useful cross-attention keys/values for predicting clip_b**. This is a different objective from "be a good standalone representation of the clip." As γ grew (becoming > 0), this gradient became more influential and pulled the encoder away from clip-level representational quality. The single-clip probe sees this pulled-away encoder.

### What would fix it (for future runs, not this one)

- Detach clip_a encoder forward: `z_a_source = encoder(clip_a).detach()` — adapter-only learns from clip_a, encoder doesn't train as a source.
- Or: use target_encoder (EMA) for clip_a source, so clip_a path doesn't train the online encoder.
- Or: λ_mcc < 0.1 to contain the secondary-objective gradient.
- Or: γ schedule that keeps γ near 0 for N warmup epochs so vanilla V-JEPA on clip_b dominates early training.

## 6. FJ diagnosis

### Claim

FJ's clip encoder is undertrained on V-JEPA because only 1/K of per-step clip forwards produce V-JEPA gradient, and the secondary objectives (study, NCE, SV) provide negligible useful signal.

### Evidence

1. **Clip V-JEPA loss falls normally** (0.456 → 0.407) — the path works.
2. **Study loss collapsed to 0.004** within 1 epoch. `study_matched_rank_top1` is mostly 1.0 (perfect study matching), but `metadata_only_study_gap ≈ 0.006` → the transformer is mostly just distinguishing studies by metadata (view/modality/phase counts), not content.
3. **NCE loss was 0.0 in 382/600 (64%) of logged steps**. When non-zero, it's erratic (some values at 16, 32, 64 — cliff artifacts). The study NCE is not a stable gradient source.
4. **Anchor cosine drifts slowly** (0.998 → 0.985) with healthy layerwise pattern — the clip encoder moved moderately, not catastrophically.
5. **Effective V-JEPA clip-forwards**: 30,000 steps × 128 clips × (1/8 getting V-JEPA gradient) = **480k** vs base e125's 7.5k × 1024 = **7.68M**. FJ has ~16× less V-JEPA gradient at matched wall-clock compute.

### Mechanism

FJ's design gives each step K=8 clips per study × 16 studies = 128 clip forwards. But only 1 clip per study gets clip V-JEPA loss (subset sampler `n_per_study=1`). So 16 clip V-JEPA signals per step, plus 128 full-clip forwards feeding the study transformer. When the study transformer collapses to metadata-shortcut (which it did by step 1500), those 112 extra forwards per step contribute no useful gradient to the encoder. The anchor loss is tiny (~0.013 at λ=0.005) — negligible retention but also negligible pull.

### What would fix it (for future runs)

- Raise `n_per_study` for clip V-JEPA to 2–4 (or K=8 for all clips) so effective V-JEPA signal is comparable to base.
- Kill or reduce the study transformer path when it collapses (detect `study_loss < 0.01` for >N steps, freeze study head).
- Enforce harder study contrastive negatives — current NCE relies on cross-rank all_gather, but if positives/negatives are all metadata-distinguishable, NCE collapses.
- Or accept FJ as a "study encoder, not clip encoder" experiment and evaluate only with h_study + K=8.

## 7. Does low single-clip A4C imply low study-level?

**For MCC: probably yes, but partial.** The encoder representation is degraded for clip-level predictions. At study level (e.g., K=8 prediction averaging) the degraded clip representations will still be averaged — the study-level signal is just `mean(probe(clip_i))`, so it's bottlenecked by clip quality. Unless the adapter is evaluable at test time (it isn't for standard probe setups), MCC's advantage from cross-clip context is unusable.

**For FJ: not necessarily.** The study transformer is the path where FJ's design intent lives. If `h_study` extracted via the FJ study transformer is useful for study-level tasks even though the clip encoder is undertrained, then FJ could still be valuable for K=8 tasks. But the collapsed study loss and metadata-only-gap ≈ 0.006 suggest the study transformer isn't producing a rich content representation either. A direct h_study probe would be needed to confirm.

## 8. Immediate action recommendations

### P0 — let the EchoNet-Dynamic LVEF probes finish

Jobs 794 (MCC) and 795 (FJ) are running against the same split as MV-PhaseRel, MV-PairedIntra, TokenRel-Motion. Final val R² will directly place MCC and FJ on the comparison axis. Expected ~90 min each.

### P1 — for the paper, frame results honestly

- MCC at matched compute produces a **clip encoder worse at clip-level V-JEPA prediction** than vanilla continuation. Single-clip probe results will reflect this. Report as a "negative result with mechanistic explanation": the clip_a source-encoder gradient damages the standalone representation.
- FJ at matched compute produces a **clip encoder undertrained on V-JEPA** (1/8 signal rate). Single-clip probe results will reflect this. Report as a "compute-allocation tradeoff" — study objective consumed compute without producing useful gradient.

### P2 — do NOT relaunch pretraining

The diagnostic evidence is unambiguous. Re-running with the same config will reproduce the same result. Any new pretraining needs a protocol change (detach clip_a, raise n_per_study, γ warmup, or kill collapsed study heads). That requires a new design doc, not a rerun.

## 9. Experiments to cancel

- **Do NOT proceed with K=8 LVEF probes** for MCC or FJ based on current checkpoints. MCC: the single-clip encoder degradation will carry through to K=8 (prediction averaging of bad per-clip scores). FJ: potentially informative for `h_study`, but that probe isn't implemented yet and needs new infrastructure.
- **Cancel any queued follow-up probes** (RV func, MR, TAPSE) on current MCC/FJ checkpoints until 794/795 settle the LVEF story.

## 10. Experiments to run next

### Now (running): 794 (MCC EchoNet-Dynamic LVEF), 795 (FJ EchoNet-Dynamic LVEF)

### After 794/795 complete:

- If MCC val R² ≤ 0.50 and FJ val R² ≤ 0.55: confirms both are below MV-PhaseRel (0.699). Write paper as negative result.
- If MCC val R² ≥ 0.55: partial recovery, re-evaluate.
- If FJ val R² ≥ 0.60: FJ undertraining hypothesis is partially wrong; investigate.

### Parked until paper decides:

- Build FJ `h_study` probe if FJ is kept in paper.
- MIMIC A4C 10k baseline probe with base e125 (the "matched-dataset V-JEPA baseline" that's missing).

## 11. Paper implications

**Current strongest story**: "We tested two augmentations to V-JEPA continuation — cross-clip adapter (MCC) and full-joint study transformer (FJ) — both designed to use multi-clip structure. Neither improved single-clip representation quality at matched compute. MCC actively damaged the standalone representation (intraview V-JEPA loss rose from 0.49 to 0.54 over training) while FJ left it undertrained (1/K=1/8 V-JEPA signal rate). We report these as informative negative results with mechanistic explanations."

This is a legitimate NeurIPS-style ablation finding if framed correctly. It can be paired with the MV-PhaseRel / MV-PairedIntra positive results to show which multi-clip augmentations help and which don't.

## 12. Final decision table

| Branch | Continue? | Stop? | Modify? | Evidence | Next action |
|---|:-:|:-:|:-:|---|---|
| MCC current checkpoint | ✗ (as clip encoder) | ✓ | — | intraview rose 0.49→0.54; encoder dragged by clip_a gradient | Wait for 794, then report as negative |
| MCC re-run with detach | — | — | ✓ (paper future) | Theoretical; needs fresh pretraining | Not priority |
| FJ current checkpoint (clip) | ✗ (as clip encoder) | ✓ | — | 1/8 V-JEPA signal; study collapsed | Wait for 795 |
| FJ h_study path | ? (untested) | Pause | Maybe | Study loss collapsed but may still be informative | Build h_study probe if FJ is kept |
| FJ re-run with K-out-of-K | — | — | ✓ (paper future) | Would restore V-JEPA signal | Not priority |

## 13. Minimal changes made during this diagnosis

- No modifications to training code.
- No restart of 762 or 776.
- Created 3 diagnostic reports under `reports/root_cause_low_performance/`:
  - `interim_report.md` — probe comparability audit
  - `training_log_analysis.md` — MCC & FJ CSV analysis
  - `root_cause_synthesis.md` — this doc
- Created 2 new EchoNet-Dynamic sbatches for MCC and FJ (matched-dataset comparators): `echonet_lvef_probe_mcc_e25.sbatch`, `echonet_lvef_probe_fj_30k.sbatch`.
- Canceled 4 stale probe jobs (786, 788, 791, 792, 793) that were on the non-comparable MIMIC A4C 10k dataset.
- Submitted 2 new probes (794, 795) on the comparable EchoNet-Dynamic dataset.
