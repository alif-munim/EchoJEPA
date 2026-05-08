# EchoMV-JEPA Stage-1 Smoke Results (Job 741)

**Date:** 2026-05-05
**Jobs:** 737 (first attempt, srun-fix deploy), 741 (reproducibility confirmation)
**Cluster:** echojepa-h100-neurips, node ip-10-0-50-146, 8× H100 80GB
**Duration:** 4:25 wallclock per run, srun subtask 3:10 (5 000 steps @ ~2 s/50 steps).
**State:** COMPLETED, exit 0, no NaNs, no crashes.

**Decision:** **SMOKE FAIL on contextualization.** Do not launch Stage-1b / Stage-1m / breadth.

---

## Pass/fail scoreboard (§12)

| Criterion | Required | Observed | Verdict |
|---|---|---|---|
| `loss_regress` decreases | ✓ | 0.127 → 0.0002 | PASS |
| `var_t > 0.3` sustained | ✓ | 3.3 – 3.7 in last 2k steps | PASS |
| No NaNs | ✓ | zero NaNs in 101 CSV rows | PASS |
| EMA teacher updates non-zero | ✓ | teacher drift visible in `z_cosine_vs_v1` falling from ~1.0 → ~0.4 | PASS |
| `z_cosine_vs_v1` not stuck > 0.98 | ✓ | oscillates 0.20 – 0.73 in last 2k steps | PASS |
| `z_cosine_vs_isolated` drifts below ~0.95 | ✗ | **0.998 – 0.999 sustained** | **FAIL** |
| `z_cosine_vs_peer_drop` < 0.99 sometimes | ✗ | **0.9994 – 0.9999 every probe** | **FAIL** |
| `cov_off` not exploding | ✓ ideally ≤ 0.1 | **3.5 – 6.9 in last 2k steps** | **FAIL (soft)** |

Primary fail axis: **teacher does not use cross-element context.**

## Contextualization evidence

`z_cosine_vs_v1` trajectory (every 500 steps):

| Step | z_v1 | z_iso | peer_drop | var_t | cov_off |
|---|---|---|---|---|---|
| 450 | 0.733 | 0.995 | 0.9996 | 0.55 | 0.12 |
| 950 | 0.858 | 0.998 | 0.9998 | 1.12 | 0.56 |
| 1450 | 0.839 | 0.998 | 0.9999 | 1.67 | 1.39 |
| 1950 | 0.752 | 0.998 | 0.9999 | 2.06 | 2.09 |
| 2450 | 0.662 | 0.998 | 0.9999 | 2.34 | 2.53 |
| 2950 | 0.507 | 0.999 | 0.9998 | 2.53 | 3.03 |
| 3450 | 0.468 | 0.998 | 0.9998 | 2.89 | 3.92 |
| 3950 | 0.277 | 0.998 | 1.0000 | 3.24 | 5.08 |
| 4450 | 0.231 | 0.998 | 0.9998 | 3.28 | 4.96 |
| 4950 | 0.418 | 0.999 | 0.9999 | 3.50 | 5.31 |

Reading: the teacher has **drifted from v1's pre-context projection** (`z_v1 = 0.42` by end, far from the 1.0 that would indicate collapse to v1), **but** it has done so by learning a per-element map that is independent of which other elements are in the study. `z_iso ≈ 1.0` throughout means the teacher's output at a target slot is the same whether the teacher sees the full study or just that one target element alone. `peer_drop ≈ 1.0` throughout means dropping a context peer does not change the teacher output either.

## Loss and representation health

- `loss_regress`: 0.1268 (step 0) → 0.0015 (step 450) → 0.0002 (step 4950). Student trivially matches the per-element teacher via cosine regression.
- `var_t`: rises 0.12 → 3.50 — representation spreads.
- `cov_off`: rises 0.005 → 5.31 — off-diagonal covariance grows unbounded. The spread is anisotropic: the student fills a narrow subspace of high-variance directions.

## Throughput & resource

- Step rate: ~26 steps/s (50 steps every ~1.9 s).
- Total compute for 5 000 steps: ~3 min 10 s on 8× H100, 32 studies/GPU batch = 256 studies/step.
- No observed memory pressure; no OOMs.

## Diagnostic coverage

All three §15.1a probes fired as designed:

- `cos_z_echomv_vs_v1` (spec name) = `z_cosine_vs_v1` — drifted from 1.0 → 0.42. ✓ halt-wired at 0.98/5k.
- `teacher_context_sensitivity` = `1 - z_cosine_vs_isolated` — stayed at **0.002** throughout. The probe is working correctly; the teacher is the problem.
- `leave_one_context_out_delta` = `1 - z_cosine_vs_peer_drop` — stayed at **< 0.001**. Same read.

The probes caught the pathology before any downstream eval. This is the design working as intended.

## Root cause hypothesis

1. **Loss signal does not reward contextualization.** The only pressure on the student is cosine regression against the teacher. The teacher, EMA-following the student, can produce a trivial per-element map (e.g. the identity at init, slowly drifting) that the student matches without needing cross-element self-attention. Both paths self-reinforce.
2. **No anti-shortcut pressure.** `lambda_nce = 0` by design in Stage-1, but this leaves no negative signal. There is nothing that pushes the student's target-slot output to differ from `proj(clip_in(tgt_element_alone))`.
3. **EMA schedule too slow at 5 k steps.** `tau_start = 0.996` means after 5 k steps the teacher is still ~close to its initial copy; the teacher never had time to develop meaningful self-attention structure. But this is not the whole story — `z_cosine_vs_v1` drifted to 0.42, so the teacher *did* move, just not in a way that made its output depend on cross-element context.

The teacher self-attention weights may be randomly initialized, producing near-uniform attention that effectively averages inputs. With a per-element meta-add at each position and a near-uniform attention mix, the contextualized output is dominated by the mean-plus-self term, which is roughly equivalent to running the teacher isolated — explaining both `z_iso ≈ 1.0` and `peer_drop ≈ 1.0`.

## Smallest-fix recommendation (§13, spec branch D)

**Branch D** — "teacher_context_sensitivity zero → debug F_bar_psi full-study path and gather indices" — is the exact failure. Two minimal diagnostic/experimental changes before any launch:

1. **Inspect-only first.** Write a one-shot Python script that loads the initial (untrained) Stage-1 student + teacher, runs `forward_contextualized` vs `forward_isolated` on a real batch, and prints the cosines. If the untrained model already shows `z_iso ≈ 1.0`, the gather path or the self-attention residual structure is routing inputs straight to outputs (possible LayerNorm-bypass at init, or the `clip_in + meta_add` residual being too strong vs the attention update).
2. **If the init diagnostic confirms the architectural near-identity**, options in order of minimum-change:
    - (a) **Reduce the LayerNorm scale on the `clip_in + meta_add` residual** — currently the student adds `clip_in + meta_add` as the input; the attention block then adds a residual. If attention outputs are initialized near zero (standard practice), the input dominates and `forward_contextualized` ≈ `forward_isolated`. Verify attention init magnitude at `src/models/study_transformer.py:_StudyBlock`.
    - (b) **Add NCE to the loss** — Stage-1b (`λ_nce=0.005`) provides anti-collapse pressure via negatives from other studies. But this is gated on Stage-1 passing; spec says don't launch 1b until 1 passes.
    - (c) **Inspect whether `forward_isolated` and `forward_contextualized` actually yield different outputs at init**. The test `test_forward_isolated_shape_and_matches_single_element_context` in `tests/echomv_jepa/test_full_study_target_encoder_shapes.py` asserts they match for a single-element input — which is correct. But the test never checks whether a *multi-element* study gives different contextualized vs isolated outputs. If the teacher architecture produces identical outputs for both at init, the plan's core claim is untestable until the architecture is fixed.

The correct next action per spec §13:
- **Do not** launch Stage-1b, Stage-1m, ablation_no_ema, or breadth.
- **Do not** run downstream probes.
- **Debug F_bar_psi's full-study pathway.** Write a standalone script that verifies `forward_contextualized(multi-element study) != forward_isolated(same element alone)` for at least one initialized model instance. If they are equal at init, the architecture never had a chance to contextualize.

## Attachment

- Full CSV log: `s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts/echomv_jepa/smoke_runs/741/train_log.csv` (101 rows, same local copy at `/tmp/smoke_741_train_log.csv`).
- Full stdout: `/opt/dlami/nvme/logs/echomv_smoke-741.out` on `ip-10-0-50-146`.
- Run folder: `s3://.../echomv_jepa/smoke_runs/741/`.

## Holds

Per spec §12–§13, holding on:

- `scripts/echomv_jepa/launch_breadth.sh` — do not invoke.
- Stage-1b config.
- Stage-1m config.
- `ablation_no_ema` config.
- Full-duration pretrain (`scripts/echomv_jepa/pretrain.sbatch`).
- Downstream probes (`evals/video_classification_frozen_multi/`).

## Next step

Debug `F_bar_psi`: prove `forward_contextualized` and `forward_isolated` differ for a multi-element study at model init. If they do not differ, the architecture has a near-identity shortcut and the Stage-1 design does not actually test the "contextualized target" hypothesis.
