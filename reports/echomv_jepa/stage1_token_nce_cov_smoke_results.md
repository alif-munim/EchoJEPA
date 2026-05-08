# Arm A — Stage-1 token + matched NCE + covariance smoke results

**Status:** filled 2026-05-05. Job 744 on ip-10-0-50-146, 14:04 wall, completed exit 0.

## Config

- config: `configs/train/echomv_jepa/stage1_token_nce_cov_smoke.yaml`
- sbatch: `scripts/echomv_jepa/pretrain_smoke_token_nce_cov.sbatch`
- job id: **744**
- step time (mean): ~3.3 s/step (including encoder forward)
- K=8, batch_studies_per_gpu=8, token_spatial_pool=2 (→ 392 tokens/clip, M·T ≈ 2 352/study)
- loss knobs: `lambda_nce=0.01`, `lambda_cov=0.001`, `var_floor=0.0`
- total steps: 225 (50 / 150 / 25)

## Trajectories (every 25 steps)

| step | loss | loss_regress | loss_nce | loss_cov | var_t | cov_off | z_v1 | z_iso | z_peer_drop | student_context_delta | target_meta_only_gap | matched_rank_top1 | matched_rank_top5 | pos_minus_hardneg_gap | fallback_fraction |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0   | 0.099 | 0.097 | 0.200 | 0.001 | 0.01 | 0.002 | 0.089 | 0.983 | 1.000 | — | — | — | — | — | — |
| 25  | 0.097 | 0.026 | 7.12 | 0.003 | 0.10 | 0.010 | -0.335 | 0.983 | 0.999 | 0.012 | 0.056 | 0.571 | 1.000 | 1.35 | 0.55 |
| 50  | 0.071 | 0.028 | 4.31 | 0.005 | 0.30 | 0.042 | -0.396 | 0.982 | 0.999 | 0.004 | 0.025 | 0.375 | 0.875 | 1.00 | 0.57 |
| 75  | 0.058 | 0.031 | 2.66 | 0.004 | 0.44 | 0.081 | -0.422 | 0.983 | nan | nan | nan | nan | nan | nan | 0.82 |
| 100 | 0.093 | 0.024 | 6.89 | 0.035 | 0.53 | 0.134 | -0.430 | 0.984 | 0.999 | **0.005** | **0.013** | **0.526** | 0.947 | 1.75 | 0.68 |
| 125 | 0.044 | 0.018 | 2.58 | 0.035 | 0.57 | 0.139 | -0.426 | 0.985 | nan | nan | nan | nan | nan | nan | 0.72 |
| 150 | 0.029 | 0.018 | 1.16 | 0.050 | 0.60 | 0.164 | -0.308 | 0.987 | 0.999 | **0.003** | **0.003** | **0.750** | 1.000 | 9.60 | 0.63 |
| 175 | 0.086 | 0.024 | 6.15 | 0.343 | 0.89 | 0.407 | -0.405 | 0.984 | nan | nan | nan | nan | nan | nan | 0.75 |
| 200 | 0.025 | 0.013 | 1.23 | 0.172 | 0.82 | 0.299 | -0.359 | 0.985 | 1.000 | **0.006** | **0.006** | **0.696** | 1.000 | 4.69 | 0.78 |
| 220 | 0.050 | 0.009 | 4.04 | 0.106 | 0.78 | 0.242 | -0.311 | 0.984 | nan | nan | nan | nan | nan | nan | 0.83 |

## Pass / fail scoreboard

| Criterion | Target | Observed (last cadence step 200) | Pass? |
|---|---|---|---|
| loss finite | yes | yes | ✓ |
| cov_off ≤ 0.5 | ≤ 0.5 | 0.30 | ✓ |
| matched_rank_top1 > chance | > 1/N (N≈20 valid tgts) | **0.70** | ✓✓ |
| pos_minus_hardneg_gap > 0 | > 0 | **4.69** | ✓✓ |
| student_context_delta ≥ 0.02 | ≥ 0.02 | **0.006** | ✗ |
| target_meta_only_gap ≥ 0.05 | ≥ 0.05 | **0.006** | ✗ |
| no NaN | 0 | 0 | ✓ |

4 of 6 primary gates pass; 2 of 6 fail.

## Key observations

- `loss_regress` drops to 0.01 (same collapse trajectory as Stage-1-pooled and Stage-1-token).
- `loss_nce` remains **active throughout**: 1-8 range, never collapses to 0. Matched negatives are doing work on every step.
- `z_cosine_vs_v1` sharply negative (-0.4), much further from v1's pre-context projection than any prior smoke. Matched NCE is genuinely reshaping target geometry.
- `z_cosine_vs_isolated` stays at 0.98. Teacher's target-slot output is per-element-identity despite being through 4 attention layers over 2352 tokens.
- `matched_rank_top1` climbs from 0.57 → 0.75 and `pos_minus_hardneg_gap` from 1.3 → 9.6. Arm A separates **studies in the learned space** extremely well.
- **But `student_context_delta = 0.003-0.006`** — the student does not actually use its same-study context when predicting targets. It solves the task by combining target metadata + global study-separation geometry.
- `target_meta_only_gap = 0.003-0.013` — running the student with *no* context only costs a tiny cosine; metadata is nearly sufficient.

## Step time / memory

- Per-step wall time: ~3.3 s (includes V-JEPA forward on 64 clips/step).
- GPU memory: fit comfortably on 80 GB H100 at batch_studies_per_gpu=8.

## Recommendation

- [ ] Continue — Arm A clears all gates. ← **no: only 4/6**
- [x] **Modify** — Arm A succeeds at study-level separation (strong retrieval signal) but fails at within-study context-dependence. Matched NCE is shifting geometry globally but the student does not need cross-element attention to win. Two candidate extensions:
  - **A'** — tighter matched NCE (`lambda_nce=0.03` plus same-(view,mod,phase)-only negatives, no fallback) to force finer discrimination within a study-bucket.
  - **A''** — reduce target-slot metadata richness (drop phase from target slot; use only view+modality) so metadata-only cannot solve the task.
- [ ] Abandon.

**Specific scientific finding**: matched NCE breaks the flat-target collapse at the *study-geometry* level (studies become separable) without breaking the *element-identity* collapse at the per-target level. The two are distinct failure modes; Arm A addresses the first but not the second.
