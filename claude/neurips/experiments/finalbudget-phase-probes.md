# Finalbudget phase-probe sweep (555-578)

**Status**: ✅ **COMPLETE — NEUTRAL verdict**. 6 train + 6 test phase probes, LVEF 555, scorer 578. RVSP utility check (557/558/561/562) released to run next.

**Date started**: 2026-04-30
**Date completed**: 2026-04-30
**Ask**: Is the phase-matched +25e multiview continuation teaching the encoder anything phase-specific that a plain +25e JEPA extension or a single-view +25e control doesn't already give you? LVEF probes can't answer this on their own — they only reward encodings that correlate with ejection fraction. A frozen phase probe that predicts sin/cos of the cardiac phase from the same encoder is the direct diagnostic.

---

## Headline verdict

**NEUTRAL**: phase-matched multiview sampling at +25e does not improve explicit cardiac-phase decodability over either a plain single-view continuation or the pre-continuation IN21K-e100 baseline.

Test-set circular MAE (2887 held-out clips):

| arm | circular MAE | phase-bin acc | macro bin acc | mean comp R² |
|-----|-------------:|-------------:|-------------:|--------------:|
| IN21K-e100 | 42.5° | 0.338 | 0.120 | −0.200 |
| phase +25 (job 542) | **42.0°** | 0.337 | 0.121 | **−0.194** |
| sv +25 (job 548) | 43.2° | 0.326 | **0.129** | −0.234 |
| *constant-baseline (φ̂=0.55)* | *44.7°* | — | — | — |

**Why this is NEUTRAL, not POSITIVE**:
1. All three arms beat the constant baseline by only 1.5°–2.7° — no arm is decoding phase well.
2. phase+25 vs sv+25 gap on circular MAE is 1.2° — within HP-seed noise.
3. Macro metrics (rare-bin performance) reverse the ordering: sv+25 > phase+25 > e100. If phase-matching were the mechanism, the ordering should be consistent across metrics.
4. Per-axis: **sin** decoding is flat across arms (best val MAE 0.448 each). **cos** decoding improves with any +25e continuation (sv+25: 0.422, phase+25: 0.432, e100: 0.451) — but sv+25 wins, not phase+25.

This matches the LVEF-probe finding (555: val MAE 5.013 vs e125 5.097 — within-noise), making the consistent story across two downstream tasks that **pairs=24 phase-matched multiview adds no representational benefit at +25 epochs over plain continuation**.

**Practical implication**: do not extend the multiview phase-matched objective to +50/+100 on this compute budget. The curriculum arm (550) and random/wrong arms (543/549) are held; random collapsed and wrong only reached e12, so neither is a clean control. RVSP utility check (557/558/561/562) is queued as a final sanity test.

---

## Implementation: phase-matched vs single-view pretraining

Both arms start from **identical IN21K-JEPA e100 weights** (job 376) and run for **25 continuation epochs** with the **same** encoder architecture (ViT-L/16, tubelet_size=2), predictor, EMA teacher, 8 random spatio-temporal block masks, optimizer, and LR schedule (100-ep cosine horizon, stop@25). The only deltas are in sampling and loss.

### Single-view (548 `final_sv_25`)

Training loop: `app/vjepa/train.py`. Per sample:

- Sampler: `VideoDataset` picks one random 16-frame window per clip from `mimic_annotations_s3.csv`.
- Per step: student encoder + predictor see one clip; teacher encodes the same clip.
- Loss: `L = smooth_L1(predictor_out, teacher_out)` over mask_pred.
- Per-GPU batch: 128 clips × 8 GPUs = 1024 clips/step.

### Phase-matched (542 `final_phase_25`)

Training loop: `app/vjepa_multiview/train.py`. Per sample:

- Sampler: `classifier/phase/sampler/phase_matched_sampler.py::PhaseMatchedStudySampler`
  - Filters: `quality_tiers=["high","medium"]`, `rr_filter_mode=strict`, `require_rr_consistent=true`.
  - Per study, builds cached pair index (`_pair_index_viewpair`, `_pair_index_curriculum`).
  - Per draw: target phase φ ~ U(0,1) (`sampling_mode=uniform_phase`); for clip_a and clip_b, snap to the nearest confident frame where `|per_frame_phase − φ| ≤ 0.15` (`phase_tolerance`); center a 16-frame window on each anchor with `frame_step=1`.
  - `pairs_per_study=24` — each study contributes up to 24 pairs per epoch (single-view averages ~1 clip per epoch via random sampling).
  - `view_pair_policy`: enforces (`same_view=0.25, same_family=0.45, cross_family=0.30`) distribution via resampling, independent of the phase logic.
  - `video_uri_mode=mp4` → reads from `s3://echodata25/mimic-echo-224px/*.mp4`.
- Per step (`forward_intraview_and_crossview`):
  ```
  z   = predictor( encoder(clip_a, masks_enc), masks_enc, masks_pred )  # student, ONCE on clip_a
  h_a = target_encoder(clip_a)                                           # teacher, no grad
  h_b = target_encoder(clip_b)                                           # teacher, no grad
  L_intraview = smooth_L1(z, h_a)                                        # standard JEPA
  L_crossview = smooth_L1(z, h_b)                                        # predict clip_b from clip_a context
  L_total     = L_intraview + 0.25 · L_crossview                         # lambda_crossview=0.25
  ```
- Per-GPU batch: 64 pairs × 8 GPUs = 512 pairs = 1024 teacher forwards (512 student forwards).

### What differs, and what doesn't

| axis | single-view (548) | phase-matched (542) |
|---|---|---|
| sample unit | 1 clip | 1 pair (clip_a, clip_b) |
| sampler | `VideoDataset` random window | `PhaseMatchedStudySampler` anchored on φ |
| data filter | none (all MIMIC MP4s) | quality_tier high+medium, RR-consistent, view_pair_policy |
| student forwards/step | 1 | 1 (clip_a only) |
| teacher forwards/step | 1 | 2 (clip_a, clip_b) |
| loss terms | intraview only | intraview + 0.25·crossview |
| pairs/study/epoch | ~1 random clip | **24** |
| ipe (steps/epoch) | 325 | 325 (matched) |
| encoder, predictor, masks, optimizer, LR, init, horizon | — | identical to left column |

**Key observation**: the intraview component of 542's total loss is essentially identical to 548's loss curve (both ~0.48 at e25). 542's **total** loss stays ≈0.67 across all 25 epochs because the crossview term `0.25 · L_crossview ≈ 0.17` sits on top of it without descending. That tells you the crossview signal is not driving representational change — the teacher's clip_b latents h_b are too close to h_a at matched phase+view, so the crossview loss is nearly redundant with intraview. The sampler IS doing what it claims (we verified view-pair mixture and phase-bin coverage at dry-run time), but the crossview objective under these pair conditions collapses toward being a noisier intraview.

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

## Comparison table (final)

Validation best-epoch (from each probe's log_r0.csv):

| arm | sin best ep | val sin MAE | val sin R² | cos best ep | val cos MAE | val cos R² |
|-----|:-----------:|------------:|-----------:|:-----------:|------------:|-----------:|
| e100 | 11 | 0.448 | +0.051 | 4 | 0.451 | −0.102 |
| phase +25 | 8 | 0.448 | +0.039 | 4 | 0.432 | −0.047 |
| sv +25 | 8 | 0.447 | +0.031 | 8 | 0.422 | −0.100 |

Test-set (2887 subject-disjoint clips, from predictions joined by video_path → anchors by dicom_id):

| arm | n | sin R² | cos R² | mean comp R² | circular MAE (cy) | circular MAE (°) | bin acc (10) | macro circular MAE (°) | macro bin acc | const-baseline cy |
|-----|--:|-------:|-------:|-------------:|------------------:|-----------------:|-------------:|-----------------------:|--------------:|------------------:|
| e100 | 2887 | −0.027 | −0.372 | −0.200 | 0.118 | 42.5 | 0.338 | 84.7 | 0.120 | 0.124 |
| phase_542 | 2887 | −0.045 | −0.344 | −0.194 | 0.117 | 42.0 | 0.337 | 83.9 | 0.121 | 0.124 |
| sv_548 | 2887 | −0.195 | −0.273 | −0.234 | 0.120 | 43.2 | 0.326 | 81.1 | 0.129 | 0.124 |

Per-phase-bin circular MAE and bin accuracy at `out/per_bin.csv` (local): all three arms show **0% bin accuracy** on bins 0, 1, 2, 3, 6, 7, 8, 9 (the rare tails). Only bins 4–5 (systole + early diastole, 56% of test clips) are decoded above chance. sv+25 marginally accurate at bins 6/7/8/9 (1–11% range) — this is where its macro advantage comes from.

**Scorer**: `classifier/phase/score_phase_probes.py` (runnable both via sbatch `phase_probe_score.sbatch` and as a local CLI against S3-synced predictions). Decision threshold: POSITIVE requires (i) phase+25 circular MAE < both controls by ≥0.005 cycles (~1.8°), and (ii) any arm beats the constant baseline by >10%. Neither condition met — verdict NEUTRAL.

### Scorer deployment note

Job 578 (sbatch) failed at submission due to the `--export=ALL,PHASE_TEST_JIDS=569,570,...` form: sbatch parses the comma-delimited value as multiple separate exports, so `PHASE_TEST_JIDS` only received `"569"`. Workaround used: ran the scorer locally against S3-synced prediction CSVs. If re-running the scorer on another cluster, pass the JID list as a colon- or semicolon-delimited string and adjust the IFS parser in the sbatch.

---

## Decision criteria — outcome

Criteria were defined ex-ante as:

1. **phase +25 > sv +25 > e100**: extend to +50/+100
2. **phase +25 ≈ sv +25 > e100**: kill multiview, extend plain JEPA instead
3. **phase +25 ≈ sv +25 ≈ e100**: +25e doesn't move phase decodability
4. **phase +25 < sv +25**: phase-matched training *hurt*

**Observed**: closest to (3) with a cos-axis twist.
- Raw circular MAE: phase_542 (42.0°) ≈ e100 (42.5°) ≈ sv_548 (43.2°) — all within 1.2° of each other and 1.5–2.7° of the constant baseline floor (44.7°).
- Raw bin accuracy: all ~0.33, mostly driven by bins 4–5 (systole).
- Per-axis cos: both +25e arms (phase, sv) improve over e100 by ~0.02–0.03 vMAE, but sv+25 > phase+25 on cos — the axis that showed any movement is not phase-specific.
- Macro metrics (rare bins): sv+25 > phase+25 > e100 — opposite direction from the mechanism claim.

**Action**:
- Do NOT extend to +50/+100 on this multiview phase-matched objective.
- Do NOT run curriculum (550), random (543, collapsed), wrong-phase (549, cancelled) extensions.
- Run RVSP as a final targeted utility check — 2 probe-train + 2 test jobs (557/558/561/562 released). If phase+25 RVSP ≈ sv+25 RVSP, this confirms NEUTRAL across a clinical readout; if phase+25 wins RVSP by a margin that exceeds LVEF noise, it's worth re-examining whether there's a hemodynamic-specific benefit that's not captured by EF or sin/cos phase.
- Deprioritize 559/560 LVEF test-set inference (held) — 555 val-set numbers are sufficient for the comparison with e125/e200 already on record.

---

## LVEF utility check (555 complete)

Submitted 2026-04-30 alongside the phase probes, on the EchoNet-Dynamic split (not MIMIC). 20-ep d=4 attentive probe, 6-HP grid — same protocol as `neurips/completed-experiments.md §1b`. Job 555 `fb_phase_542_lvef` completed in 1:23:46 (exit 0).

### Val-set results (20 epochs, EchoNet-Dynamic LVEF)

| Arm | best ep | val MAE | val R² | val Pearson |
|-----|:-:|-------:|-------:|-----------:|
| **555 fb_phase_542 (+25e mv)** | 16 | **5.013** | **0.691** | **0.833** |
| JEPA-IN21K-e125 (matched compute) | 18 | 5.097 | 0.685 | 0.832 |
| JEPA-IN21K-e200 (2× compute) | 16 | 4.880 | 0.714 | 0.845 |
| EchoJEPA-L-K (anneal reference) | 18 | 4.448 | 0.766 | 0.876 |

### 555 vs JEPA-IN21K-e125 (matched compute): within-noise

Both arms are +25 epochs from IN21K-e100 — 555 via phase-matched multiview, e125 via plain JEPA continuation (job 332):

| metric | e125 | phase+25 (555) | Δ |
|--------|-----:|---------------:|---:|
| val MAE | 5.097 | 5.013 | **−0.084** (-1.6%) |
| val R² | 0.685 | 0.691 | +0.006 |
| val Pearson | 0.832 | 0.833 | +0.001 |

Phase-matched is numerically better by 0.08 val MAE, but **this is at the 2nd decimal** — well within HP-sweep noise. For comparison, JEPA-e200 (4× the continuation compute) is 0.13 val MAE ahead of 555.

### Test-set inference skipped

Job 559 (fb_phase_542_lvef_test) was submitted but is now **held** — 555's val numbers are sufficient for the within-noise comparison, and test inference on EchoNet-Dynamic costs ~1.5h on 1 GPU. 560 (fb_sv_548_lvef_test) is similarly held since job 556 (sv LVEF train) was cancelled mid-run to free GPUs for the phase-probe debug harness, and its best.pt was never saved. If the RVSP NEUTRAL verdict confirms the pattern, we don't need sv LVEF either.

---

## RVSP utility check (partial — 557 done, 558 running)

Released 2026-04-30 after phase-probe verdict:
- 557 `fb_phase_542_rvsp` (8-GPU, 1:45:07 actual) — ✅ COMPLETE
- 558 `fb_sv_548_rvsp` — RUNNING (chained afterany:557)
- 561 `fb_phase_542_rvsp_test` (afterok:557 + PROBE_JOB=557, 1 GPU, ~1.5h) — pending
- 562 `fb_sv_548_rvsp_test` — afterok:558 + PROBE_JOB=558 — pending

Uses MIMIC single-view RVSP 10K subset (`mimic_rvsp_sv_{train,val,test}_10k.csv`), same HP grid as the LVEF probes. `TARGET_MEAN=30.10 mmHg, TARGET_STD=12.23 mmHg`.

### Val-set results (20 epochs)

| arm | best ep | val MAE | val R² | val Pearson | source |
|-----|:-:|-------:|-------:|-----------:|--------|
| JEPA-IN21K-e100 (reference) | 5 | **6.823** | **+0.175** | 0.482 (e13) | job 484 `rvsp_sv_484/jepa_in21k_e100_sv` |
| **phase +25 (job 542, probe 557)** | 12 | 6.995 | +0.142 (e11) | 0.451 (e17) | job 557 |
| sv +25 (job 548, probe 558) | pending | pending | pending | pending | job 558 (running) |

**e125/e200 RVSP references do not exist in S3.** The trajectory probing effort for JEPA extensions (job 332) covered LVEF only; RVSP wasn't extended past e100 in prior work. If we want e125/e200 RVSP numbers, a fresh probe run is ~1h on 8 GPU.

### 557 vs e100: phase +25 is slightly worse on RVSP

| metric | e100 best (ep) | phase+25 best (ep) | Δ (phase+25 − e100) | verdict |
|--------|---------------:|--------------------:|--------------------:|:-:|
| val MAE (mmHg) | 6.823 (e5) | 6.995 (e12) | **+0.17 worse** | below noise |
| val R² | +0.175 (e5) | +0.142 (e11) | **−0.033 worse** | below noise |
| val Pearson | 0.482 (e13) | 0.451 (e17) | **−0.031 worse** | below noise |

At every epoch from e5 onward, phase+25 val MAE sits above e100's. The gap is small (~2% of target std), within HP-seed noise floor, but the sign is consistent.

**Noise-floor threshold** pre-specified: gap > 0.3 MAE required to flag as investigable. Observed 0.17 — **below threshold**, NEUTRAL reinforced.

Once 558 (sv +25) lands, the final 3-way table will sit under this subsection. Expected direction: sv +25 will also land near 6.9–7.0 MAE; the three arms (e100, phase+25, sv+25) should cluster together on RVSP as they did on LVEF and the phase probe.

---

## Cross-references

- LVEF 555 results and comparison: §ADDED-TO `completed-experiments.md` section 1b when complete
- Pretraining design + loss curves: `multiview-pilot-progress.md`
- Subject-disjoint splits provenance: `classifier/phase/sampler/README_multiview.md`
- Δ_within prior results (caveat: superseded by this probe): `classifier/phase/sampler/prepost_delta_within.py` docstring
