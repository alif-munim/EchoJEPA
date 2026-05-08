# EchoMV-JEPA Stage-1 Launch Readiness

**Date:** 2026-05-05
**Author:** automated validation pass
**Decision:** **GO for Stage-1 smoke only.** Do not launch Stage-1b / Stage-1m / breadth / downstream probes until Stage-1 smoke passes its gate review.

---

## S3 artifact check

| Artifact | Path | Status |
|---|---|---|
| `cache_index.parquet` | `s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts/echoset_jepa/cache_index.parquet` | ✓ 5.3 MB, 214 092 rows |
| `cclip/` dir | `s3://.../echoset_jepa/cclip/` | ✓ 214 100 `.npy` files, 904 MB total |
| Train K8 manifest | `s3://.../echoset_jepa/study_clip_sample_K8_seed0_train.parquet` | ✓ 1 027 385 B |
| Val K8 manifest | `s3://.../echoset_jepa/study_clip_sample_K8_seed0_val.parquet` | ✓ 111 764 B |
| Test K8 manifest | `s3://.../echoset_jepa/study_clip_sample_K8_seed0_test.parquet` | ✓ 99 534 B |
| Dedup clip manifest | `s3://.../echoset_jepa/study_clip_manifest_dedup.parquet` | ✓ 10.7 MB |
| Element manifest | `s3://.../echoset_jepa/study_element_manifest.parquet` | ✓ 2.1 MB |

cache count 214 100 vs cache_index 214 092 delta = 8 rows — within the few missing-clip tolerance the sampler logs at build time. All 6 089 training studies are represented in the index.

## Manifest counts (schema-validated locally)

| Split | Rows | Studies | Avg clips/study | Modalities | Top views |
|---|---|---|---|---|---|
| train | 47 955 | 6 089 | 7.88 | b_mode 36 130 / color 11 825 | apical 12k, parasternal_long 11k, parasternal_short 11k, unknown 8k, subcostal 5k, suprasternal 411 |
| val | 4 340 | 548 | 7.92 | b_mode 3 267 / color 1 073 | apical 1k, parasternal_long 1k, parasternal_short 977, unknown 734, subcostal 452, suprasternal 42 |
| test | 3 889 | 490 | 7.94 | b_mode 2 936 / color 953 | apical 989, parasternal_long 921, parasternal_short 868, unknown 667, subcostal 417, suprasternal 27 |

All six required columns present in every split: `study_id`, `clip_id`, `view_family`, `modality`, `phase_bucket`, `quality_score`.

## YAML configs

All four updated; zero REPLACE placeholders.

| File | manifest | cache_s3_prefix | cache_local_prefix | frozen | source |
|---|---|---|---|---|---|
| `stage1_frozen_clip_full_study_ema.yaml` | train K8 | S3 cclip | NVMe echoset_cache | true | cached |
| `stage1b_frozen_clip_tiny_nce.yaml` | train K8 | S3 cclip | NVMe echoset_cache | true | cached |
| `stage1m_frozen_clip_modality_projector.yaml` | train K8 | S3 cclip | NVMe echoset_cache | true | cached |
| `ablation_no_ema.yaml` | train K8 | S3 cclip | NVMe echoset_cache | true | cached |

All training configs point at the **train** manifest only. Val/test K8 manifests are reserved for downstream probe/eval (not consumed by pretraining).

## Smoke sbatch

`scripts/echomv_jepa/pretrain_smoke.sbatch`:

- Pins to `--nodelist=ip-10-0-50-146` (confirmed idle in `dev` partition).
- Syncs `s3://.../echoset_jepa/cclip/ → /opt/dlami/nvme/echoset_cache/` at line 84 (uses `aws s3 sync --only-show-errors --exclude "_index/*"`).
- Downloads train K8 manifest to `/opt/dlami/nvme/echomv_cache/k8_train.parquet`.
- Generates a self-contained YAML at `${WORKDIR}/echomv_smoke.yaml` equivalent to Stage-1 but with shorter step counts (500 warmup / 4 000 main / 500 cooldown) and `lambda_nce=0.0, num_modalities=1`.
- Launches via `srun --ntasks-per-node=8 python -m app.main_srun` with `MASTER_ADDR`/`MASTER_PORT` exports (fixed from the prior `main_distributed` attempt which needed top-level `nodes`/`tasks_per_node` keys).
- Uploads run folder to `s3://.../echomv_jepa/smoke_runs/${SLURM_JOB_ID}/`.

## Contextualization diagnostics

Already present in `app/echomv_jepa/train.py` — no code changes needed.

| Spec name | Code path | Definition |
|---|---|---|
| `cos_z_echomv_vs_v1` | `z_cosine_vs_v1` (line 187) | Cosine between `proj_teacher(F_bar_psi(full_study)[target_idx])` and `proj_teacher(st.clip_in(tgt_elements))` |
| `teacher_context_sensitivity` | `1 - z_cosine_vs_isolated` (line 196) | Cosine between teacher over full study and teacher over each target element alone (via `forward_isolated`); sensitivity = complement |
| `leave_one_context_out_delta` | `1 - z_cosine_vs_peer_drop` (line 226, every 50 steps) | Cosine between teacher over full study and teacher with one random context peer dropped |

All three logged to CSV + stdout. Halt rule: `z_cosine_vs_v1 > 0.98 for 5 000 consecutive steps → halt_falsification.pt`.

## Test status

`pytest tests/echomv_jepa/ tests/echoset_jepa/ -q` → **145 passed** in 2 s (24 echomv + 109 echoset + 12 shared fixtures).

## Validation gates (all passed)

- [x] No REPLACE placeholders in any echomv config or sbatch.
- [x] All four YAMLs point at S3 train manifest + S3 cclip prefix + NVMe local mirror.
- [x] Training configs do not point at val/test manifests.
- [x] S3 `cache_index.parquet` exists, 214 092 rows.
- [x] S3 `cclip/` has 214 100 `.npy` files.
- [x] Local K8 manifest schema validated for train/val/test.
- [x] All four YAMLs load cleanly; `app.echomv_jepa.train` imports without error.
- [x] `scripts/echomv_jepa/pretrain_smoke.sbatch` contains the S3→NVMe sync block at line 84.
- [x] Smoke sbatch launches the Stage-1 config (`num_modalities=1, lambda_nce=0.0`), not 1b/1m/ablation.
- [x] Smoke sbatch uses `app.main_srun` (fixed from prior `main_distributed` crash).
- [x] 145/145 tests pass.
- [x] `z_cosine_vs_v1`, `z_cosine_vs_isolated`, `z_cosine_vs_peer_drop` implemented, logged, and halt-wired.

## Go/no-go

**GO for Stage-1 smoke.**

Explicit holds:
- **Do not launch Stage-1b** until Stage-1 smoke passes its pre-downstream gates.
- **Do not launch Stage-1m** until Stage-1 smoke passes its pre-downstream gates.
- **Do not launch `ablation_no_ema`** until Stage-1 smoke passes.
- **Do not launch the breadth run** (`scripts/echomv_jepa/launch_breadth.sh`).
- **Do not run downstream probes.**

## Smoke pass/fail criteria (§20.2.a)

**Pass** if:
- `loss_regress` decreases.
- `var_t > 0.3` sustained (no 500-step window below).
- No NaNs.
- Teacher EMA moves (non-zero delta from its initial deep-copy state).
- `z_cosine_vs_v1` does not stay > 0.98 for the full smoke window (no falsification-halt trigger).
- `z_cosine_vs_isolated` drifts below ~0.95 and does **not** settle at ~1.0 — evidence the teacher actually uses cross-element context.
- `z_cosine_vs_peer_drop` < 0.99 at least intermittently.

**Fail** if:
- EchoMV target is indistinguishable from v1 pre-context target (`z_cosine_vs_v1 > 0.98` sustained).
- Teacher ignores context (`z_cosine_vs_isolated ≈ 1.0` throughout).
- `z_cosine_vs_peer_drop ≈ 1.0` throughout (peer elements do not change teacher output).
- Representation collapse (`var_t < 0.3` for 500 steps OR `cov_off` explodes above 1.0).
- Repeated missing-cache errors from the dataloader.

Prior session's smoke run (job 737) completed 5 000 steps with `loss_regress = 0.0002`, `var_t = 3.5`, but `z_cosine_vs_isolated = 0.998` and `z_cosine_vs_peer_drop = 0.9999` — **fail on contextualization**, not on loss or stability. Re-running the smoke with the same config to check reproducibility; if the teacher_context_sensitivity stays near zero again, we diagnose before Stage-1b.

## Exact launch command

```bash
# From HyperPod controller (via SSM):
sudo -u ubuntu sbatch /tmp/vjepa2-ctrl/scripts/echomv_jepa/pretrain_smoke.sbatch
```

Monitor:
```bash
sudo -u ubuntu squeue -u ubuntu -l
sudo -u ubuntu srun --jobid=<JOBID> --overlap bash -c "tail -f /opt/dlami/nvme/logs/echomv_smoke-<JOBID>.out"
```

Expected log path: `/opt/dlami/nvme/logs/echomv_smoke-<JOBID>.out`.

After smoke completes, results land at `s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts/echomv_jepa/smoke_runs/<JOBID>/`.

---

**Next action:** submit the smoke sbatch only. Do not launch breadth.
