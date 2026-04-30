# Multi-view JEPA pilots — progress doc

Live status for the phase-matched multiview pilot series. Updated 2026-04-29.

## One-line status

Job **516 (mv_phase_25e)** is running; jobs 517/518/519 queued behind it with SLURM dependencies. A background poll loop refreshes `/tmp/mv_poll_loop.log` every 2 h.

## Pilot goal

Measure whether a phase-matched multi-view positive pair induces cross-clip phase alignment in the encoder representation, using four +25-epoch continuations from the same single-view IN21K e100 checkpoint:

1. **phase-matched** (`uniform_phase`) — expected positive signal.
2. **same-study random** (`same_study_random`) — controls for same-study bias.
3. **wrong-phase** (`wrong_phase`) — controls for cross-clip similarity at a mismatched phase.
4. **phase-curriculum** (`phase_curriculum`) — phase-matched with a view-distance curriculum (easy→hard).

Evaluation: `Δ_within = sim_phase_within − sim_random_within` via
`classifier/phase/sampler/prepost_delta_within.py`. Pre-baseline on e100 is already captured — see "Artifacts".

## Checkpoints

| Role | Path |
|---|---|
| Starting ckpt (all pilots) | `checkpoints/jepa_in21k_vitl_e100.pt` (EFS) / `/opt/dlami/nvme/checkpoints/jepa_in21k_vitl_e100.pt` (compute NVMe) / `s3://...vjepa2-artifacts/checkpoints/jepa_in21k/jepa_in21k_vitl_e100.pt` |
| Compute-matched single-view control | **does not exist** — no e125 was trained. e200 is a longer reference only. |
| Pre-baseline Δ_within artifact | `checkpoints/prepost/baseline_e100/` (median +0.001, Wilcoxon p=0.065) |

## Active HyperPod job queue

### Pretraining (4)
| Job | Name | Config | Sampling mode | Depends on | State |
|---|---|---|---|---|---|
| 516 | `mv_phase_25e` | `configs/train/vitl16/pretrain-multiview-phase-high-25e.yaml` | `uniform_phase` | — | RUNNING |
| 517 | `mv_random_25e` | `configs/train/vitl16/pretrain-multiview-random-high-25e.yaml` | `same_study_random` | `afterany:516` | PENDING |
| 518 | `mv_wrong_25e` | `configs/train/vitl16/pretrain-multiview-wrongphase-high-25e.yaml` | `wrong_phase` | `afterany:517` | PENDING |
| 519 | `mv_curric_25e` | `configs/train/vitl16/pretrain-multiview-phase-curriculum-high-25e.yaml` | `phase_curriculum` | `afterany:518` | PENDING |

### Train probes on final ckpt (8)
d=4 attentive regression head, 6-HP multihead grid, 20 epochs. `afterok:<pretrain>` so we never probe a failed backbone.

| Job | Name | Task | Pretrain source | State |
|---|---|---|---|---|
| 520 | `mv_phase_25e_lvef` | EchoNet-Dynamic LVEF | 516 `latest.pt` | PENDING (afterok:516) |
| 521 | `mv_phase_25e_rvsp` | MIMIC single-view RVSP | 516 `latest.pt` | PENDING (afterok:516) |
| 522 | `mv_random_25e_lvef` | EchoNet-Dynamic LVEF | 517 `latest.pt` | PENDING (afterok:517) |
| 523 | `mv_random_25e_rvsp` | MIMIC single-view RVSP | 517 `latest.pt` | PENDING (afterok:517) |
| 524 | `mv_wrong_25e_lvef` | EchoNet-Dynamic LVEF | 518 `latest.pt` | PENDING (afterok:518) |
| 525 | `mv_wrong_25e_rvsp` | MIMIC single-view RVSP | 518 `latest.pt` | PENDING (afterok:518) |
| 526 | `mv_curric_25e_lvef` | EchoNet-Dynamic LVEF | 519 `latest.pt` | PENDING (afterok:519) |
| 527 | `mv_curric_25e_rvsp` | MIMIC single-view RVSP | 519 `latest.pt` | PENDING (afterok:519) |

Train probe folder: `evals/vitl/neurips/<pilot>_end_{lvef,rvsp_sv}_224/`. Tag: `neurips-<pilot>-end-<task>`. CSVs used: `echonet_dynamic_{train,val}_s3_raw.csv` (LVEF) and `mimic_rvsp_sv_{train,val}_10k.csv` (RVSP). LVEF target z-score: mean 55.7776, std 12.4072. RVSP target z-score: mean 30.0959, std 12.2321.

### Test-set inference on probe `best.pt` (8)
1-GPU inference with `val_only: true` + `probe_checkpoint: best.pt`. Depends `afterok:<probe-train>`.

| Job | Name | Task | Probe source | Test CSV |
|---|---|---|---|---|
| 528 | `mv_phase_25e_lvef_test` | EchoNet LVEF | 520 `best.pt` | `echonet_dynamic_test_s3_raw.csv` |
| 529 | `mv_phase_25e_rvsp_test` | MIMIC RVSP | 521 `best.pt` | `mimic_rvsp_sv_test_10k.csv` |
| 530 | `mv_random_25e_lvef_test` | EchoNet LVEF | 522 `best.pt` | as above |
| 531 | `mv_random_25e_rvsp_test` | MIMIC RVSP | 523 `best.pt` | as above |
| 532 | `mv_wrong_25e_lvef_test` | EchoNet LVEF | 524 `best.pt` | as above |
| 533 | `mv_wrong_25e_rvsp_test` | MIMIC RVSP | 525 `best.pt` | as above |
| 534 | `mv_curric_25e_lvef_test` | EchoNet LVEF | 526 `best.pt` | as above |
| 535 | `mv_curric_25e_rvsp_test` | MIMIC RVSP | 527 `best.pt` | as above |

Per-job test prediction CSV: `s3://...runs/<pilot>_<task>_test_<jobid>/predictions/<pilot>_end_<task>_test.csv`.

Cluster: `echojepa-h100-neurips` / compute node `ip-10-0-50-35` (ml.p5.48xlarge, 8 H100 80GB). Pretrain + probe jobs use all 8 GPUs; test inference uses 1 GPU each.

### sbatch scripts

```
scripts/neurips/phase/multiview_phase_high_25e.sbatch
scripts/neurips/phase/multiview_random_high_25e.sbatch
scripts/neurips/phase/multiview_wrongphase_high_25e.sbatch
scripts/neurips/phase/multiview_phase_curriculum_high_25e.sbatch
```

All use the NVMe-only deploy pattern (root FS on compute is 100% full with a deleted-but-held-open file; see `claude/dev/hyperpod-troubleshooting.md#19`).

## Settings shared across all 4 pilots

| Field | Value |
|---|---|
| Model | ViT-L/16, RoPE, SDPA, activation checkpointing |
| Frames / clip / fps / tubelet | 16 / 224px / 8 / 2 |
| Per-GPU batch / global | 4 / 32 |
| Sampler | PhaseMatchedStudySampler (filters: `high` tier + `strict` RR + `require_rr_consistent=true`) |
| frame_step | 1 |
| phase_tolerance | 0.15 |
| pairs_per_study | 1 |
| lambda_crossview | 0.25 |
| use_intraview_loss / use_crossview_loss | true / true |
| video_uri_mode | `mp4` (rewrites raw-staging DICOM URIs to `mimic-echo-224px` mp4 URIs) |
| debug_verify_frame_count | true, N=8 (runs once on rank 0 at epoch 0) |
| num_workers / persistent_workers | **4 / false** (sweep winner; persistent must be false because pair DataFrame swaps each epoch) |
| pin_memory | true |
| Epochs / ipe | 25 / 199 (~4,975 optimizer steps) |
| Warmup / start_lr / lr / final_lr | 5 / 3.33e-5 / 1.75e-4 / 1.75e-4 |
| EMA / weight decay | 0.99925 / 0.04 |
| save_every_freq / save_at_end | 5 / true |
| Seed | 234 |
| dtype | bfloat16 |

The curriculum pilot additionally loads view labels from `/opt/dlami/nvme/data/view_labels/mimic_view_predictions.csv` (525,328 rows, 100% coverage) with `min_view_confidence: 0.60`, and applies a 3-stage bucket schedule (easy→medium→hard) parametrized by `epoch_frac`.

## Expected throughput

At `num_workers=4` the 8-GPU sanity produced:

- iter mean 943 ms, p90 1,097 ms, data-time 0%
- 33.9 clips/s aggregate (global bs 32)
- 25 epochs × 199 iter × ~0.95 s ≈ **75 min of training per pilot**
- + 1–2 min setup (tarball unpack, parquet/ckpt stage cache hit)

With 4 pilots serialized via `afterany`, full sweep fits in about 5–6 h wallclock.

### Compute-budget reality check vs IN21K e100

The fast wallclock is correct — the pilot is doing roughly **0.5%** of the compute of the single-view IN21K e100 pretrain that produced our starting checkpoint. Numbers:

| Field | IN21K-e100 (`pretrain-jepa-mimic-224px-16f-in21k-hp.yaml`) | mv_*_25e pilot |
|---|---|---|
| Per-GPU batch | 128 | 4 |
| Global batch | 1024 | 32 |
| Steps × epochs | 300 × 100 = 30,000 opt steps | 199 × 25 = 4,975 opt steps |
| Clips processed | ~30.7 M | ~159 K (~12.5 passes over the 6.4K pair pool) |
| Ratio | — | **~0.5 % of IN21K** |

Implications:
1. The budget is big enough to produce **detectable** encoder drift (EMA-drift checks in the 2-GPU/8-GPU sanity already showed student moves 3–6e-4 mean per-param). It is NOT big enough to claim "multiview > single-view at matched compute" — we'd need a single-view e125 control (doesn't exist).
2. If the three arms diverge on post-minus-pre Δ_within, that's signal from the sampling choice, not from training volume. If they all look flat, that's not evidence against multiview — it's evidence that 0.5 % budget is too small, and we'd want a +100e follow-up.
3. The pilot is a **sampling-hypothesis** test, not a compute-matched comparison. Read diagnostics with that framing.

## Progress captured so far on job 516

- 03:14 submit, 03:15 training start, frame-guard `OK` at step 0.
- Early iters: step 1 total=0.71, step 21 total=0.65 (rank 0).
- By 03:33 (epoch ~5), rank 0 at step 996, loss avg 0.62. `e5.pt` saved at 03:32, `e10.pt` at 03:50.
- Latest (polled 04:02): step 996 visible in tail, epoch 5/25. Throughput tracking the 0.88–0.95 s/iter prediction.

## Polling loop

- Script: `/tmp/mv_poll_loop.sh`
- Launcher PID: 387924 (disowned via `nohup`; replaces earlier 381879 after probe/test queue added)
- Tracks all 20 jobs (4 pretrain + 8 probe + 8 test)
- Log: `/tmp/mv_poll_loop.log` (appended each cycle)
- Stderr: `/tmp/mv_poll_loop.stderr`
- Cadence: 7,200 s (2 h)
- Max cycles: 12 (24 h wallclock)
- Early-exit condition: `squeue -h -j 516,517,518,519` returns 0 rows

Each cycle records, per remote target:
1. `squeue -l` from the controller
2. `sacct -P -n` for the 4 jobs
3. For each job: last 3 `step   ` loss lines, last 5 `exit | saved checkpoint | frame-guard | Traceback | RuntimeError` events from `/opt/dlami/nvme/logs/<name>-<jobid>.out`

To read latest state: `tail -200 /tmp/mv_poll_loop.log`.

## Post-training diagnostics (per pilot)

After each checkpoint lands:

```bash
python classifier/phase/sampler/prepost_delta_within.py \
  --pre  checkpoints/jepa_in21k_vitl_e100.pt \
  --post <pilot_end_checkpoint>.pt \
  --out-dir checkpoints/prepost/<pilot_name>_25e
```

Expected rough comparison (hypotheses, to be confirmed):

| Pilot | Expected Δ_within shift |
|---|---|
| phase-matched | positive — training induces cross-clip phase alignment |
| same-study random | near-zero — generic same-study similarity, no phase selection pressure |
| wrong-phase | negative or near-zero — phase-mismatched positives shouldn't produce phase alignment |
| phase-curriculum | ≥ phase-matched, potentially larger — easier cross-clip targets early, harder late |

Do not overclaim until all three controls complete.

## Code pieces that landed this turn

- `classifier/phase/sampler/phase_matched_sampler.py`:
  - `SAMPLING_MODES += ("phase_curriculum",)`
  - `VIEW_FAMILIES`, `_VIEW_EASY_PAIRS`, `_VIEW_MEDIUM_PAIRS`, `view_distance_bucket()` taxonomy
  - `MatchRecord` extended with `view_distance_bucket`, `view_distance_numeric`, `view_family_a/b`, `curriculum_epoch_frac`, `curriculum_bucket_probs`
  - `__init__` accepts `view_labels`, `view_confidences`, `min_view_confidence`, `curriculum`, `total_epochs`
  - `_bucket_pairs_for_study`, `_pick_pair_rows_curriculum`, and curriculum-aware `build_records` path
  - Bug fix: `row["view"]` instead of `row.view` (latter collides with `Series.view` method and returned a bound method, making all clip views `None`)
- `src/datasets/data_manager.py`: loads `view_labels_path` CSV, passes `view_confidences / min_view_confidence / curriculum / total_epochs` through to the sampler.
- `app/vjepa_multiview/train.py`: earlier turn landed the MP4 frame-count guard (`_run_frame_count_guard`, controlled by `phase_multiview.debug_verify_frame_count`).

## Known issues / watch items

1. **Compute-node root FS full** — 47 GB unaccounted deleted-but-held-open file. NVMe-only deploy works; if NVMe ever pressures, we need a node replacement.
2. **Curriculum hard-bucket drift** — observed vs target fractions (at epoch 15): 0.18/0.28/**0.54** vs 0.20/0.40/0.40. About 13% of multi-clip studies have only UNKNOWN-view clips, so curriculum fallback drains into "hard". Acceptable for this pilot; could tighten coverage later by filtering studies with ≥1 known-view pair.
3. **NCCL teardown warning** on job exit is benign (seen on 1-GPU, 2-GPU, 8-GPU sanities and sweep).
4. **No single-view e125 exists** for compute-matched comparison. e200 is a longer-reference only; cannot be cited as apples-to-apples.
5. **Dependency policy**: 517/518/519 use `afterany` not `afterok` — a failing predecessor still lets the next job run. Intentional to not lose the 16 h cluster window on one incident, but means we need to check each pilot's exit status from `sacct` post-run.

## Operational notes

- View labels CSV staged once to NVMe; subsequent pilots cache-hit on `/opt/dlami/nvme/data/view_labels/mimic_view_predictions.csv`.
- Source tarball lives at `s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts/setup/vjepa2-src.tar.gz` (370 KB). Rebuild with the docs-recommended `find + tar -T` pattern before any code change is submitted.
- Pilot outputs: `s3://...vjepa2-artifacts/runs/mv_{phase,random,wrong,curric}_25e_<jobid>/{checkpoints,logs}/`.

## Not yet launched (explicit hold)

- +50 or +100 epoch extensions of any arm
- ED/ES-biased, high+medium, controlled-Δphase
- Raw ECG fusion, phase-relation predictor token
- `persistent_workers=true`, `frame_step>1`
