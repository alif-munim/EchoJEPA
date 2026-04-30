# Finalbudget phase-probe sweep (555-574)

**Status**: 🟡 **IN PROGRESS** — 555 LVEF complete, 563 phase_e100_sin running. Queue: 6 train + 6 test phase probe jobs; related RVSP/LVEF jobs held.

**Date started**: 2026-04-30
**Ask**: Is the phase-matched +25e multiview continuation teaching the encoder anything phase-specific that a plain +25e JEPA extension or a single-view +25e control doesn't already give you? LVEF probes can't answer this on their own — they only reward encodings that correlate with ejection fraction. A frozen phase probe that predicts sin/cos of the cardiac phase from the same encoder is the direct diagnostic.

---

## Pretraining runs feeding this sweep

| SLURM | Name | Objective | Ckpt (S3 `runs/*/checkpoints/latest.pt`) | Final total loss | State |
|-------|------|-----------|------------------------------------------|-----------------:|-------|
| 542 | `final_phase_25` | multiview phase-matched, pairs=24 | `final_phase_25_pairs24_542` | 0.6712 (e25) | ✅ complete |
| 543 | `final_random_25` | multiview same-study-random | `final_random_25_pairs24_543` | 0.3145 (e25, **collapsed**) | ⚠️ collapsed e8 |
| 544 | `final_wrong_25` (pilot) | multiview wrong-phase | (superseded) | — | superseded |
| 548 | `final_sv_25` | single-view continuation control | `final_sv_25_pairs24_548` | 0.4685 (e25) | ✅ complete |
| 549 | `final_wrong_25` (re-run) | multiview wrong-phase | `final_wrong_25_pairs24_549` (only e5, e10) | 0.6659 (e12, **cancelled**) | ⚠️ cancelled e12.19 to free GPUs |
| 550 | `final_curric_25` | curriculum phase-matched | not started (held) | — | held |

All runs start from `checkpoints/jepa_in21k/jepa_in21k_vitl_e100.pt` (IN21K-JEPA e100, job 376). +25 of 100-epoch scheduler horizon, stop_after=25. See `multiview-pilot-progress.md` for pretrain design/history.

---

## Why this is the right diagnostic now

The LVEF probe on 555 (fb_phase_542) landed at val MAE **5.013, R² 0.691, pearson 0.833** — vs JEPA-IN21K-e125 (matched compute, job 332) at **5.097, 0.685, 0.832**. A ~0.08 MAE edge at the 2nd decimal place, well within HP-sweep noise. Phase-matched continuation is *not hurting* but the signal is too weak to justify extending to +50/+100 epochs.

The LVEF MAE cannot distinguish between:
- "phase training teaches phase-sensitive features" (the hoped-for story)
- "phase training is a no-op perturbation that happens to be equivalent to JEPA continuation"
- "both arms memorized LVEF-relevant features at the same rate irrespective of multiview signal"

A frozen encoder that predicts sin/cos of the cardiac phase *directly* is the cleanest discriminator. Δ_within (the prior substrate-validation metric) touched on this but was (a) contaminated by a cache bug caught and fixed on 2026-04-29, and (b) statistically weak at n=30 anchor clips. The phase probe is n ≈ 3K test clips with a full regression readout, and is the criterion you'll actually report.

---

## Dataset

### Anchor-phase regression

Target per clip = `[sin(2π·φ), cos(2π·φ)]` where φ is the per-frame phase at a specific anchor frame. Two separate scalar probes (sin, cos) per checkpoint — total 6 train jobs across the 3 required arms.

**Why two scalar probes instead of one 2-target probe**: the eval pipeline's regression path (`evals.video_classification_frozen/eval.py`) is scalar-only end-to-end (target_mean/std, MAE computation, prediction CSV). Patching to multi-target is 50+ lines across 3 call sites, high risk of breaking the LVEF/RVSP probes. Two scalar probes are zero code change.

### CSV construction

Built by `classifier/phase/build_phase_probe_csvs.py`:

- Source: `classifier/phase/phase_annotations/phase_annotations.parquet` (525K MIMIC clips with per-frame phase + confident-mask, from ECG trace alignment)
- Subject-disjoint train/val/test from `classifier/phase/splits/dicoms_split.csv` (pre-existing splits, quality=high, rr=strict)
- Anchor: median of the confident-mask run in each clip (guarantees `confident_mask[anchor_frame] = 1`)
- Per-subject cap: 6 clips
- Phase-bin balanced: 10 bins on [0,1), aim for `n_target / 10` per bin
- DICOM→MP4 URI mapping: `s3://echodata25/mimic-raw-staging/*.dcm` → `s3://echodata25/mimic-echo-224px/*.mp4`
- CSV format: **3-column** `uri anchor_frame target` (VideoDataset auto-detects; anchor-aware sampling activates)

### Split sizes

| split | n clips | n subjects | bins (10) |
|-------|---------|------------|-----------|
| train | 10000 | 2888 | `[908, 600, 1000, 1034, 1476, 1969, 1100, 1001, 511, 401]` |
| val | 1449 | 217 | `[38, 35, 79, 155, 325, 533, 168, 70, 24, 22]` |
| test | 2887 | 434 | `[114, 74, 117, 300, 647, 982, 364, 156, 71, 62]` |

Subject-disjoint verified at build time. Distribution is diastole-heavy across all splits (reflects phase distribution in MIMIC echo clips; per-bin caps only soften not eliminate this skew).

### Target statistics (train split)

| target | mean | std |
|--------|------|-----|
| sin(2πφ) | -0.015 | 0.662 |
| cos(2πφ) | -0.285 | 0.693 |

Used as `target_mean/target_std` in the z-score-at-runtime regression pipeline.

---

## Infrastructure patches

### 1. Anchor-aware VideoDataset (`src/datasets/video_dataset.py`)

Existing VideoDataset samples a partition-random 16-frame window regardless of CSV label — for scalar targets that apply to the whole video (LVEF, RVSP) that's fine, but for anchor-specific phase targets it would mean the label corresponds to a frame the encoder never saw.

**Patch**:
- CSV loader auto-detects 3-col vs 2-col input
- 3-col: `anchors[i]` stored per sample; 2-col: `anchors = None` (old behavior)
- `get_item_video` writes `self._current_anchor` before every `loadvideo_decord` call
- `_sample_from_vr` takes a new anchor-centered branch when `_current_anchor is not None`: computes `indices = start + arange(fpc)*frame_step` with `start = anchor - (fpc//2)*frame_step`, clamped to `[0, n_frames - clip_len]`. Returns a single clip (`num_clips=1` forced in anchor mode).
- Assertion: `indices.min() >= 0 and indices.max() < len(vr)` inside the branch.

Blast radius: additive. Old 2-col CSVs take the original code path. LVEF/RVSP probes unaffected.

### 2. Debug harness (`classifier/phase/debug_anchor_probe_loader.py`)

100-sample integrity check run via `scripts/neurips/phase/phase_probe_debug_anchor_check.sbatch`:

- Loads 100 samples through the patched `VideoDataset`
- Cross-references each anchor against the parquet
- Per-sample assertions:
  1. All returned frame indices ∈ `[0, n_video_frames)`
  2. `anchor_frame` is in the 16-frame window (EXACT) or at least `[win_min, win_max]` (IN)
  3. MP4 frame count matches parquet `n_video_frames` (tolerance: ±5 or ±5%)
  4. `sin(2π·phases[anchor])` from parquet = target in CSV to within 1e-3
  5. `confident_mask[anchor] = 1`
- Aggregate: prints mean/median/min of `confident_mask` fraction across all 16 sampled frames

**Job 577 result (2026-04-30 14:35 UTC)**:

```
SUMMARY: n=100  PASS=100  FAIL=0
MP4/parquet frame-count mismatches (small, tolerated): 0
Confident-mask fraction across sampled frames (N=100): mean=1.000 median=1.000 min=1.000
ALL CHECKS PASSED
```

All 100 anchors were EXACT (present in window). All 16 sampled frames per clip had confident phase. Zero frame-count drift between MP4 mirror and parquet. Phase probes released after this check.

### 3. CSV builder (`classifier/phase/build_phase_probe_csvs.py`)

Emits 3-col probe CSVs and a full-metadata diagnostics CSV (`data/csv/mimic_phase_anchors_10k.csv` with phase, phase_bin, dicom_id, subject_id, s3_uri, anchor_idx).

### 4. Sbatches

All 12 phase-probe sbatches generated from a template in `scripts/neurips/phase/phase_probe_{arm}_{target}_{train,test}.sbatch`:
- 8-GPU DDP for train (`#SBATCH --gres=gpu:8`, 4h timeout)
- 1-GPU for test (1h)
- 2-HP grid (`lr ∈ {1e-4, 5e-5}` × `wd ∈ {0.1}`) — simpler than LVEF's 6-HP grid since phase is a more constrained target
- `num_segments: 1` (anchor mode requires single clip)
- `frames_per_clip: 16, frame_step: 2` (same as LVEF)

Debug sbatch installs `pyarrow` (not in the conda-pack env) before reading the parquet.

---

## Queue structure

Chain all 6 train jobs serially (single 8-GPU node), each test afterok its parent:

```
555 fb_phase_542_lvef [DONE]
  → 563 phase_e100_sin      → 569 phase_e100_sin_test
  → 564 phase_e100_cos      → 570 phase_e100_cos_test
  → 565 phase_phase_542_sin → 571 phase_phase_542_sin_test
  → 566 phase_phase_542_cos → 572 phase_phase_542_cos_test
  → 567 phase_sv_548_sin    → 573 phase_sv_548_sin_test
  → 568 phase_sv_548_cos    → 574 phase_sv_548_cos_test
```

**Held** (user / admin):
- 550 `final_curric_25` — pretraining continuation, awaiting phase-probe outcome before restart
- 557 `fb_phase_542_rvsp` + 558 `fb_sv_548_rvsp` + 561, 562 test — RVSP probes deferred
- 559 `fb_phase_542_lvef_test` — test probe held until we decide it's worth the 1.5h; 555 val metrics are already sufficient for the LVEF column in the main table
- 560 `fb_sv_548_lvef_test` — sv LVEF train (556) was cancelled mid-run to free GPUs for the debug check; need to requeue the train

**Resources**: single `ml-p5-48xlarge` node (8× H100). All phase probes serialize.

---

## Expected runtime

- Each train: ~3h wallclock (10K train × 2 HPs × 15 epochs on 8 H100)
- Each test: ~15 min
- Chain: 6 × 3h + 6 × 15m ≈ 19.5h serial
- ETA for all 6 arms: **2026-05-01 ~10:00 UTC** (started 14:37 UTC 04-30)

---

## Checkpoints targeted

| arm_tag | ckpt (S3 `vjepa2-artifacts/`) | source job | note |
|---------|-------------------------------|------------|------|
| `e100` | `checkpoints/jepa_in21k/jepa_in21k_vitl_e100.pt` | 376 | canonical IN21K baseline |
| `phase_542` | `runs/final_phase_25_pairs24_542/checkpoints/latest.pt` | 542 | +25e phase-matched mv |
| `sv_548` | `runs/final_sv_25_pairs24_548/checkpoints/latest.pt` | 548 | +25e single-view control |

**Not included** (exploratory, held until clean checkpoints exist):
- `random` (job 543) — collapsed at e8; latest.pt is degenerate. e5.pt pre-collapse could be probed later but is off the main comparison.
- `wrong` (job 549) — only e5.pt, e10.pt exist; cancelled at e12.19.
- `curric` (job 550) — not started.

---

## Planned comparison table

Will be populated as test inference lands. Compute from the **test-set prediction CSVs** (569–574) joined by `video_path`.

| arm | best val sin MAE | best val cos MAE | test sin MAE | test cos MAE | test circular MAE (°) | test joint R² | test phase-bin acc (10-bin) |
|-----|------------------|------------------|--------------|--------------|----------------------|---------------|-----------------------------|
| e100 | pending | pending | — | — | — | — | — |
| phase +25 (542) | pending | pending | — | — | — | — | — |
| sv +25 (548) | pending | pending | — | — | — | — | — |

**Post-processing plan**: after all 6 tests finish, join sin/cos prediction CSVs per arm by `video_path`, compute:
- `φ̂ = atan2(ŝin, ĉos) / (2π) mod 1`
- circular MAE in degrees (wrap-aware)
- joint R² on the 2D `(sin, cos)` target (vs. baseline: `(mean_sin, mean_cos)`)
- 10-bin phase accuracy using the same bin edges as train

Script: `classifier/phase/score_phase_probes.py` (to be written after first arm completes so I can verify the CSV schema matches).

---

## Decision criteria

All against **circular MAE on test** (primary) and **joint R²** (secondary):

1. **phase +25 > sv +25 > e100**: phase-matched training induces phase-sensitive representations beyond what pure continuation does. Supports extending to +50/+100.
2. **phase +25 ≈ sv +25 > e100**: +25e of any training helps phase decodability, but multiview phase-matching is not phase-specific. Kill the multiview objective, extend plain JEPA instead.
3. **phase +25 ≈ sv +25 ≈ e100**: +25e doesn't move phase decodability either way. Either the encoder already saturated the phase signal at e100, or 25 extra epochs is too short.
4. **phase +25 < sv +25**: phase-matched training *hurt* phase decodability. Would be a surprising negative result; investigate loss/data/sampler.

Curriculum (550) and wrong-phase (549) are optional follow-ups: if outcome (1), add curriculum to see if it goes further. If outcome (4), wrong-phase becomes interesting as a "negative control actually works" story.

---

## Cross-references

- LVEF 555 results and comparison: §ADDED-TO `completed-experiments.md` section 1b when complete
- Pretraining design + loss curves: `multiview-pilot-progress.md`
- Subject-disjoint splits provenance: `classifier/phase/sampler/README_multiview.md`
- Δ_within prior results (caveat: superseded by this probe): `classifier/phase/sampler/prepost_delta_within.py` docstring
