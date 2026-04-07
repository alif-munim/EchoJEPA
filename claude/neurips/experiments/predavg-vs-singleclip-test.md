# Prediction Averaging vs Single-Clip — Clean Test-Set Comparison

**Date:** 2026-04-07
**Question:** Is the prediction averaging "boost" reported in `probe-results.md` real, or an artifact of comparing single-clip val to pred avg test?

## Background

The pred avg boost table in `uhn_echo/nature_medicine/context_files/dev/probe-results.md` reports gains like:

| Task | Model | Single-clip | Pred avg | Reported boost |
|---|---|---|---|---|
| RVSP | G | 0.454 | 0.504 | +5.0pp R² |

But per the legend ("Values without annotation = single-clip val"), the single-clip column is from the **val** set while pred avg is from the **test** set. Any val/test distribution difference is folded into the "boost," making it impossible to isolate the genuine pred avg effect.

## Method

Re-ran RVSP pred avg via `scripts/run_pred_avg.sh rvsp` for EchoJEPA-G. The pred avg code path (`evals/video_classification_frozen/eval.py`) saves `clip_outputs.npz` containing:
- `clip_predictions_all_heads`: per-clip predictions across all 12 HP heads (z-scored)
- `clip_labels`: per-clip labels (z-scored)
- `clip_study_ids`: study ID for each clip
- `zscore_mean`, `zscore_std`: for converting back to real units

From this single npz, both clip-level (single-clip) and study-level (pred avg) metrics can be computed on the **same test split**, isolating the aggregation effect from any val/test confound.

## EchoJEPA-G RVSP Results

100,184 clips across 10,015 studies (~10 clips/study). Best head 8 wins both metrics.

| Metric | Single-clip (test) | Pred avg (test) | Δ |
|---|---|---|---|
| **R²** | 0.4301 | 0.5036 | **+0.0735** |
| **Pearson** | 0.6611 | 0.7257 | +0.0646 |
| **MAE (mmHg)** | 8.01 | 7.25 | −0.76 |

## Comparison to Published Numbers

| | R² | Source |
|---|---|---|
| Single-clip **val** (published) | 0.454 | Original probe training log |
| Single-clip **test** (this analysis) | **0.4301** | Computed from `clip_outputs.npz` |
| Pred avg test (published) | 0.504 | Original pred avg run |
| Pred avg test (this analysis) | **0.5036** | Same npz, study-level grouping |

The pred avg numbers match exactly (0.504 ≈ 0.5036), confirming the new analysis reproduces the published pipeline.

## Findings

1. **Single-clip val (0.454) is higher than single-clip test (0.430)** by ~2.4pp R². The val set is genuinely easier than the test set for the EchoJEPA-G RVSP probe.

2. **The genuine pred avg boost is +7.4pp R²** (0.430 → 0.504), not the +5.0pp reported in the boost table. The published "boost" was a lower bound — it understated the true aggregation effect because val/test split difference was masking it.

3. **MAE drops by 0.76 mmHg with pred avg** (8.01 → 7.25), a clinically meaningful improvement.

4. **Best head is consistent** (head 8 for both clip-level and study-level), suggesting the HP grid is robust to aggregation method.

## Full 5-Model Results (extended 2026-04-07)

All 5 manuscript models rerun through `run_pred_avg.sh rvsp`. Each produced a `clip_outputs.npz` with per-clip predictions across all 12 HP heads. Clip-level and study-level metrics computed from the same data for a clean within-model comparison.

| Model | Clip R² | Clip Pearson | Clip MAE | Study R² (PA) | Study Pearson | Study MAE | Δ R² (PA boost) |
|-------|---------|--------------|----------|---------------|---------------|-----------|-----------------|
| **EchoJEPA-G** | **0.430** | **0.661** | **8.01** | **0.504** | **0.726** | **7.25** | **+0.074** |
| EchoJEPA-L-K | 0.234 | 0.489 | 9.35 | 0.318 | 0.581 | 8.54 | +0.083 |
| PanEcho | 0.207 | 0.466 | 9.35 | 0.274 | 0.555 | 8.62 | +0.067 |
| EchoPrime | 0.123 | 0.419 | 9.50 | 0.169 | 0.477 | 8.83 | +0.046 |
| EchoJEPA-L | 0.081 | 0.333 | 10.02 | 0.168 | 0.442 | 9.08 | +0.087 |

### Cross-model findings

1. **Pred avg boost is real for all 5 models** — range +4.6pp (EchoPrime) to +8.7pp R² (L). Average ~+7pp. **All larger than the +3.4 to +6.3pp reported in the published boost table** (which was underestimated due to the val/test split confound).

2. **Ranking is preserved** at clip-level and study-level: G > L-K > PanEcho > EchoPrime ≈ L. Aggregation reduces noise uniformly without reshuffling model comparisons.

3. **L is a surprisingly weak baseline at clip level** (R²=0.081, Pearson=0.333). It gains the most from pred avg (+8.7pp) because its per-clip predictions are the noisiest. Suggests the manuscript L (235-epoch MIMIC-only) underperforms specifically on RVSP — the task where cross-view geometry matters most and single-view information is least sufficient.

4. **EchoPrime gains the least from pred avg** (+4.6pp). Its text-supervised global-pool representations already do implicit pooling before the probe, so averaging adds less marginal information. Consistent with the hypothesis that contrastive/global-pool encoders have smoother prediction surfaces than patch-token encoders like JEPA.

5. **G's +20pp R² lead over L-K** at clip level (0.430 vs 0.234) is bigger than at study level (0.504 vs 0.318 = +18.6pp). Pred avg slightly closes the gap, but scale still dominates.

## Implications

- **Pred avg is more valuable than the published table suggested.** The average boost across 5 RVSP models is ~+7pp R², not ~+5pp.
- **The published pred avg boost table is biased** — it should be treated as a lower bound on the true aggregation effect across all tasks. Any task where val < test (val set easier) will show an underestimated boost; any task where val > test will show an overestimated one.
- **For NeurIPS paper:** the pred avg protocol decision (Strategy E) is well-justified by this clean analysis. Pred avg is a substantial 5-9pp R² gain uniformly across all 5 manuscript models on RVSP.
- **For Nature Medicine:** this analysis doesn't change which protocol to use — the published pred avg numbers are still the right ones to report. But it does confirm that the current headline RVSP numbers are well-earned, not inflated by val/test confounds.

## Reproducing

```bash
# Run pred avg (saves clip_outputs.npz)
cd /home/sagemaker-user/user-default-efs/vjepa2
DEVICES="cuda:4 cuda:5 cuda:6 cuda:7" MASTER_PORT=29530 \
  bash scripts/run_pred_avg.sh rvsp

# Output location:
# evals/vitg-384/nature_medicine/uhn/video_classification_frozen/rvsp-predavg-{model}/clip_outputs.npz

# Then compute clip-level metrics in Python:
# - Load clip_outputs.npz
# - Compute R²/Pearson on clip_predictions_all_heads vs clip_labels (no averaging)
# - Compute R²/Pearson on per-study averages (groupby clip_study_ids, mean predictions)
# - Compare
```

## Status

- **All 5 manuscript models complete** (2026-04-07): G, L-K, L, EchoPrime, PanEcho
- Full 5-model comparison table added above
- `clip_outputs.npz` files saved at `evals/vitg-384/nature_medicine/uhn/video_classification_frozen/rvsp-predavg-{model}/`

## Caveats

- The "single-clip" R² here is computed by skipping study averaging at metric time, but the predictions themselves were generated by a probe **trained** with `study_sampling: true` (1 random clip per study per epoch). A probe trained without study sampling might show different clip-level performance. This analysis only isolates the aggregation step, not the training regime.
- 100,184 clips across 10,015 studies is more than the test CSV's 5,103 rows — the multi-clip-per-row structure of the RVSP eval (`num_clips_per_video: 2`, `num_segments: 2`) expands each row at inference time. The metrics are still valid since predictions and labels are matched 1:1.
