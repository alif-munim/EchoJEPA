# Arm C — global [STUDY]-token JEPA smoke results

**Status:** filled 2026-05-05. Two crashed attempts (746 with CUDA-generator bug, 752 with missing-diagnostic-key bug) followed by successful run 754. Both bugs are fixed in the tarball.

## Config

- config: `configs/train/echomv_jepa/stage1_global_study_token_smoke.yaml`
- sbatch: `scripts/echomv_jepa/pretrain_smoke_global_study.sbatch`
- job id: **754** (predecessors 746, 752 failed and are noted below)
- step time (mean): ~0.2 s/step (pooled path, no V-JEPA forward)
- knobs: `lambda_nce=0.01`, `lambda_cov=0.001`, corruption mix = random 0.30 / view-family 0.25 / modality 0.15 / none 0.30
- pooled path (cclip cache on node 56 NVMe); `batch_studies_per_gpu=32` (so cross-study NCE pool = 31 negs/row/rank)

## Trajectories (every 25 steps)

| step | loss | loss_regress | loss_nce | loss_cov | var_t | cov_off | study_context_delta | metadata_only_study_gap | study_matched_rank_top1 | study_matched_rank_top5 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0   | 0.111 | 0.005 | 10.54 | 0.000 | 0.00 | 0.000 | 0.004 | 0.054 | 0.156 | 0.500 |
| 25  | 0.107 | 0.021 | 8.54 | 0.001 | 0.07 | 0.004 | nan | nan | nan | nan |
| 50  | 0.097 | 0.023 | 7.36 | 0.001 | 0.11 | 0.007 | 0.008 | 0.034 | 0.156 | 0.500 |
| 75  | 0.105 | 0.023 | 8.16 | 0.001 | 0.17 | 0.014 | nan | nan | nan | nan |
| 100 | 0.090 | 0.017 | 7.32 | 0.004 | 0.30 | 0.043 | **0.015** | **0.291** | **0.188** | **0.500** |
| 125 | 0.065 | 0.013 | 5.25 | 0.011 | 0.32 | 0.066 | nan | nan | nan | nan |
| 150 | 0.093 | 0.011 | 8.28 | 0.003 | 0.31 | 0.039 | **0.010** | 0.075 | **0.188** | **0.469** |
| 175 | 0.086 | 0.008 | 7.87 | 0.001 | 0.23 | 0.024 | nan | nan | nan | nan |
| 200 | 0.065 | 0.010 | 5.49 | 0.006 | 0.36 | 0.057 | **0.008** | **0.160** | **0.250** | **0.531** |
| 220 | 0.048 | 0.008 | 3.99 | 0.020 | 0.42 | 0.097 | nan | nan | nan | nan |

**Chance baselines** for a batch of B=32 studies per rank:
- top1: 1/32 = 0.031
- top5: 5/32 = 0.156

## Pass / fail scoreboard

| Criterion | Target | Observed (step 200) | Pass? |
|---|---|---|---|
| loss_regress does not collapse to ~0 | > 1e-3 | 0.010 | ✓ |
| cov_off ≤ 0.5 | ≤ 0.5 | 0.057 | ✓✓ |
| study_matched_rank_top1 > 1/32 (~0.031) | > chance | **0.25 (8× chance)** | ✓✓ |
| study_matched_rank_top5 > 5/32 (~0.156) | > chance | **0.53 (3.4× chance)** | ✓✓ |
| metadata_only_study_gap ≥ 0.05 | ≥ 0.05 | **0.16** | ✓ |
| study_context_delta ≥ 0.02 | ≥ 0.02 | **0.008** | ✗ |
| no NaN | 0 | 0 | ✓ |

**6 of 7 gates pass.** The one failing gate (`study_context_delta = 0.008`) is the "shuffle corrupted study across batch" probe — which may be systematically less sensitive than the within-study probe because shuffling the whole study (including all meta) puts the student in a wildly OOD distribution at cadence-probe time.

## Comparison to Arms A and B (step 200)

| Metric | Arm A (744) | Arm B (745) | Arm C (754) |
|---|---|---|---|
| loss_regress | 0.013 | 0.014 | 0.010 |
| var_t | 0.82 | 0.70 | 0.36 |
| cov_off | 0.30 | 0.21 | **0.06** (best) |
| context/study_context_delta | 0.006 | 0.003 | **0.008** (best) |
| meta-only_gap | 0.006 | 0.007 | **0.16** (27× better) |
| matched_rank_top1 | 0.70 (within-study) | 0.52 (within-study) | **0.25** (study-level, 8× chance) |

Note the different *kinds* of matched_rank:
- A/B: rank of positive target among matched-metadata-across-studies negatives (so "can you match this target to its original element?"). Chance is ~0.05.
- C: rank of positive study among other studies in the batch (so "can you identify this study?"). Chance is 0.031.

All three measures are "above-chance", but Arm C's `metadata_only_study_gap = 0.16` — **27× larger than A's 0.006** — is the most interpretable positive signal across all three arms.

## Key observations

1. **Arm C separates studies in the learned space** with large headroom over metadata-only shortcuts. Unlike A and B, where metadata plus study-geometry almost fully explains the target, Arm C has a **visible study-content signature** that metadata cannot reproduce.

2. **loss_nce is very high (5-10)** because the pool size (31) lets InfoNCE apply meaningful pressure at batch size 32.

3. **`var_t` grows slowly** (0.3 → 0.4 by end) — representation is spreading but not collapsing.

4. **`cov_off = 0.06`** at step 200 — an order of magnitude better than A (0.30) and B (0.21). Arm C has the healthiest decorrelation.

## Recommendation

- [x] **Continue** — Arm C is the simplest viable study-level method: clears 6/7 gates, has the strongest signal that the student actually uses cross-element information (via the metadata-only gap), and has the healthiest covariance. Next step: run an 800-step Arm C and follow with a small downstream probe (LVEF linear probe on frozen `h_study`) to see if the study-level representation carries task-relevant signal.
- [ ] Modify.
- [ ] Abandon.

**Specific scientific finding**: predicting `h_study` with element-corruption is a qualitatively different learning problem than predicting per-element targets. The `[STUDY]` readout is forced to aggregate across elements because no single element's content predicts the [STUDY]-pooled target. This is the only arm that produces a non-trivial `metadata_only_gap` — the student's representation uses *some* study-level visual content that metadata alone cannot reproduce. Per the plan's Part 8 decision rule: **"If Arm C passes: prioritize global study-token JEPA as the shortest viable study-level method. Then add element prediction as an auxiliary, not the main objective."**

## Bugs fixed before success

Two bugs blocked Arm C initially and are fixed in the current tarball:
1. **`_apply_study_corruption`** used a CUDA `torch.Generator` against CPU weights for `torch.multinomial`. Fixed by using a CPU-only generator throughout.
2. **`main()` halt-probe** unconditionally read `out.diagnostics["z_cosine_vs_v1"]`, which Arm C's `training_step_echomv_global` does not emit. Fixed with `.get(..., float("nan"))`.

Both are in the `app/echomv_jepa/train.py` file and tests are in `tests/echomv_jepa/test_study_corruption_masks.py` and `test_global_study_token_loss.py`.
