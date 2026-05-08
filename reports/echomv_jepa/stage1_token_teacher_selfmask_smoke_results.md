# Arm B — Stage-1 token + teacher target self-masking smoke results

**Status:** filled 2026-05-05. Job 745 on ip-10-0-50-146, completed, 24 CSV rows.

## Config

- config: `configs/train/echomv_jepa/stage1_token_teacher_selfmask_smoke.yaml`
- sbatch: `scripts/echomv_jepa/pretrain_smoke_token_teacher_selfmask.sbatch`
- job id: **745**
- step time (mean): ~3.3 s/step
- knobs: `p_target_token_mask=0.5`, `lambda_nce=0.01`, `lambda_cov=0.001`

## Trajectories (every 25 steps)

| step | loss | loss_regress | loss_nce | loss_cov | var_t | cov_off | z_v1 | z_iso | z_peer_drop | student_context_delta | target_meta_only_gap | matched_rank_top1 | matched_rank_top5 | pos_minus_hardneg_gap | teacher_selfmask_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0   | 0.099 | 0.097 | 0.200 | 0.001 | 0.01 | 0.002 | 0.112 | 0.984 | 1.000 | — | — | — | — | — | 0.29 |
| 25  | 0.095 | 0.022 | 7.29 | 0.003 | 0.11 | 0.011 | -0.334 | 0.981 | 0.999 | 0.011 | 0.061 | 0.476 | 1.000 | 1.36 | 0.36 |
| 50  | 0.070 | 0.030 | 3.96 | 0.006 | 0.30 | 0.040 | -0.422 | 0.979 | 0.999 | 0.007 | 0.026 | 0.333 | 0.824 | 1.10 | 0.40 |
| 75  | 0.048 | 0.024 | 2.39 | 0.005 | 0.45 | 0.080 | -0.421 | 0.983 | nan | nan | nan | nan | nan | nan | 0.39 |
| 100 | 0.085 | 0.025 | 6.07 | 0.032 | 0.50 | 0.129 | -0.475 | 0.982 | 0.999 | **0.007** | **0.008** | **0.474** | 1.000 | 1.84 | 0.39 |
| 125 | 0.085 | 0.023 | 6.19 | 0.024 | 0.54 | 0.117 | -0.474 | 0.983 | nan | nan | nan | nan | nan | nan | 0.38 |
| 150 | 0.020 | 0.018 | 0.18 | 0.041 | 0.56 | 0.144 | -0.509 | 0.984 | 0.999 | **0.010** | **0.005** | **0.875** | 1.000 | 7.91 | 0.33 |
| 175 | 0.066 | 0.019 | 4.77 | 0.144 | 0.80 | 0.276 | -0.571 | 0.980 | nan | nan | nan | nan | nan | nan | 0.25 |
| 200 | 0.059 | 0.014 | 4.43 | 0.083 | 0.70 | 0.210 | -0.481 | 0.982 | 1.000 | **0.003** | **0.007** | **0.522** | 1.000 | 2.95 | 0.48 |
| 220 | 0.056 | 0.009 | 4.69 | 0.106 | 0.79 | 0.242 | -0.458 | 0.981 | nan | nan | nan | nan | nan | nan | 0.19 |

`teacher_selfmask_rate` averages 0.34; configured target 0.5. The gap is from the interaction of Bernoulli(0.5) masking with finite targets per step (some target rows have fewer T tokens fully masked).

## Pass / fail scoreboard

| Criterion | Target | Observed (step 200) | Pass? |
|---|---|---|---|
| loss finite | yes | yes | ✓ |
| student_context_delta > Arm-A | > Arm-A's 0.006 | **0.003** | ✗ |
| target_meta_only_gap ≥ 0.05 | ≥ 0.05 | **0.007** | ✗ |
| matched_rank_top1 > Arm-A | > 0.70 | 0.52 (mid-run 0.875 at step 150, regressed) | ✗ |
| cov_off ≤ 0.5 | ≤ 0.5 | 0.21 | ✓ |
| no NaN | 0 | 0 | ✓ |
| teacher_selfmask_rate ≈ 0.5 | 0.4–0.6 | 0.34 mean, 0.19-0.48 range | marginal |
| **Anti-gate**: z_iso drops but student_context_delta ≈ 0 | z_iso flat AND delta ~0 | **z_iso=0.98, delta=0.003** | **failure mode confirmed** |

## Comparison to Arm A (side by side at step 200)

| Metric | Arm A (744) | Arm B (745) | B better? |
|---|---|---|---|
| z_cosine_vs_v1 | -0.359 | -0.481 | B further (not necessarily better) |
| z_cosine_vs_isolated | 0.985 | 0.982 | ≈ |
| student_context_delta | 0.006 | 0.003 | ✗ worse |
| target_meta_only_gap | 0.006 | 0.007 | ≈ |
| matched_rank_top1 | 0.696 | 0.522 | ✗ worse |
| pos_minus_hardneg_gap | 4.69 | 2.95 | ✗ worse |
| cov_off | 0.299 | 0.210 | ≈ |

## Key observations

Teacher-target self-masking at `p=0.5` **did not** recover contextualization. Quite the opposite:
- Same per-element-identity collapse as Arm A (`z_iso ≈ 0.98`).
- `student_context_delta` is actually *lower* than Arm A (0.003 vs 0.006), meaning the student has become *less* context-dependent with self-masking.
- `matched_rank_top1` is lower than A's 0.70 at step 200 (0.52), though a mid-run peak reached 0.875 at step 150. Training is noisier than A.
- `z_cosine_vs_v1` is more negative (-0.48 vs -0.36): self-masking moves the target space further from v1's linear projection, but into a region where the student solves the task via metadata shortcuts even more (since the target is now "teacher on incomplete data" which correlates more with metadata than with full-study context).

## Recommendation

- [ ] Continue.
- [ ] Modify.
- [x] **Abandon Arm B as a standalone fix.** The anti-gate fires precisely: the knob moves teacher output away from per-element-isolated without making the student depend on cross-element context. Worth flagging: this is the same pattern Stage-1-token exhibited at much lower `z_iso`-sensitivity.

**Specific scientific finding**: teacher-target self-masking shifts the target geometry but does not force the student to use context. The student instead re-routes through metadata and study-level geometry (the two signals Arm A already exploits). Partial masking at target positions is insufficient; a harder corruption schedule (peer-level dropout, not just target-level) would likely be needed — that's Arm C's approach.
