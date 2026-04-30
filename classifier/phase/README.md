# Cardiac Phase Extraction from MIMIC-IV-Echo Cines

Extract the ECG waveform burned into the bottom strip of each 2D echo cine (A4C, A2C, A3C, PLAX, etc.) to recover per-frame cardiac phase — systole vs. diastole, R-peak timing, and a continuous phase signal aligned to image frames.

## Goal

For every multi-frame 2D echo DICOM in MIMIC-IV-Echo, produce:
1. A 1-D ECG trace indexed by frame number (one sample per video frame).
2. R-peak indices and an RR-derived continuous phase ∈ [0, 1) per frame.
3. A per-view phase label suitable for conditioning / grouping downstream probes (e.g. end-diastolic vs. end-systolic frame detection).

Per-frame phase lets downstream probes treat each view (A4C, A2C, A3C, PLAX) with explicit knowledge of where in the cycle the clip sits, instead of relying on the encoder to infer it implicitly.

## Data Source

Raw DICOMs: `s3://echodata25/mimic-raw-staging/files/{pNN}/{pSUBJECT}/{sSTUDY}/{STUDY}_{NNNN}.dcm`

Layout mirrors PhysioNet MIMIC-IV-Echo v1.0. Record list at:
`uhn_echo/nature_medicine/data_exploration/mimic/mimic-iv-echo/echo-record-list.csv`
(`subject_id, study_id, acquisition_datetime, dicom_filepath`, ~525k rows).

Local sample (study `s94106955`, 89 DICOMs) at `data/sample_data/mimic_p10_dcm/` for iteration without S3 round-trips. The multi-frame files (~5–8 MB, `NumberOfFrames ≈ 60–90`) are the 2D cines we target; the small single-frame files are spectral stills and are out of scope for ECG extraction.

## Where the ECG Lives in the Cine

The ECG is rendered as a colored trace (typically green / teal / yellow-green on black) along the bottom strip of every frame, outside the ultrasound sector. It is a **hardware overlay** burned into pixels — there is no separate DICOM waveform object attached to these cines (`WaveformSequence` is absent; `Modality = US`; SOP Class = Ultrasound Multi-frame Image Storage).

Observed per-frame layout on the local sample (708 × 1016, YBR_FULL_422):
- Top ~85%: echo sector (e.g. A4C fan).
- Bottom strip ~30–80 px: ECG trace with a sweep cursor advancing left→right across the clip.
- `SequenceOfUltrasoundRegions` (0018,6011) bounds the sector; everything below `RegionLocationMaxY1` is overlay real estate (ECG, vendor logo, text, depth markers).

## Directory Layout

```
phase/
  README.md                        # this file

  # Acquisition & conversion
  download_and_convert.py          # N random DICOMs → dicoms/, cines → gif/
  download_multiclip_studies.py    # prioritize multi-clip studies (for alignment tests)
  extract_lastframe.py             # last-frame PNG per DICOM; --extract-waveform strips trace on white
  extract_hr.py                    # write dicom_metadata.csv (HR + 80 DICOM fields)

  # ECG strip / signal extraction
  crop_waveform.py                 # vertical-crop ECG strip from an animated GIF cine
  crop_waveform_frame.py           # same, single still image (PNG)
  ecg_signal.py                    # 2D strip -> 1D amplitude + R-peak detect (neurokit2)
  run_ecg_signal_batch.py          # run ecg_signal across all strips -> diagnostic plots
  process_waveform.py              # CANONICAL preprocess: trace-on-white PNGs ->
                                   #   .npz bundles (full_y, observed_mask,
                                   #   trace_span_mask, x0/x1, coverage_frac)
                                   #   + 300dpi re-renders for inspection

  # Calibration
  ecg_calibration.py               # per-clip sampling-rate / sweep-speed back-solve
  run_calibration_batch.py         # calibration_results.csv for every DICOM
  scanner_defaults.json            # modal SR per scanner model
  validate_pipeline.py             # quality-classify clips vs metadata HR
  select_handlabel_set.py          # stratified sample for hand-label audit
  hand_label_compare.py            # detector vs hand-labeled R-waves, P/R/F1

  # Alignment-substrate experiments
  test_xcorr_alignment.py          # single-offset cross-correlation
  test_hilbert_phase.py            # per-sample Hilbert phase on HR-inconsistent pairs
  test_rpeak_alignment.py          # per-frame phase from detected R-peaks
  diagnose_hr_inconsistent.py      # HR-drift / scanner-mismatch arbitration

  # R-peak detection robustness
  rpeak_detectors.py               # robust_rpeaks: neurokit + Pan-Tompkins+HR
                                   #   + fallback, chosen by log-ratio distance
                                   #   to metadata HR × duration
  rpeak_failure_autopsy.py         # per-category failure breakdown on full set

  # Embedding-similarity substrate validation (JEPA-IN21K-e100)
  embedding_substrate_validation.py  # main runner: DICOM frames -> encoder ->
                                     #   Δ_within / Δ_specificity cosines
                                     #   between phase-matched vs random frames
  embedding_validate_from_cache.py   # analysis-only re-run on cached embeddings
                                     #   with regime-stratified Δ tables
  rpeak_funnel_diagnostic.py         # where clips are lost: strip -> video
                                     #   mapping funnel + ECG/video duration check

  # Dataset build for phase-aligned multi-view pretraining
  build_phase_annotations.py         # end-to-end per-DICOM worker:
                                     #   S3 cp -> decode -> strip -> 1D signal
                                     #   -> calibrate -> R-peak -> frame phase
                                     #   -> one parquet row per clip
  phase_annotations.sbatch           # SLURM array job (256 shards) driving
                                     #   build_phase_annotations on the
                                     #   full MIMIC-IV-Echo record list
  phase_annotations_aggregate.sbatch # concat shard parquets into one file

  # Data
  dicoms/                          # raw .dcm files (500 clips, 37 multi-clip studies)
  gif/                             # multi-frame cines as animated GIFs
  calibration_results.csv          # per-clip SR (scanner_default or fallback)
  dicom_metadata.csv               # per-clip HR + all DICOM metadata
  pipeline_validation.csv          # quality classes: good / hr_mismatch / irregular / no_detection
  xcorr_test_results.csv           # xcorr on image-cropped signals (legacy baseline)
  xcorr_processed.csv              # xcorr on process_waveform NPZs (current)
  rpeak_test_results.csv           # R-peak on image-cropped (legacy)
  rpeak_processed.csv              # R-peak on NPZs, neurokit2-only (intermediate)
  rpeak_robust_results.csv         # R-peak on NPZs, robust ensemble (current)
  rpeak_failures.png               # 4x3 grid of over/under-detecting failures
  rpeak_failure_categories.txt     # failure-mode counts (full 488-NPZ audit)
  rpeak_phase_bundle.npz           # per-pair phase arrays for post-hoc plotting
  rpeak_phase_bundle_processed.npz # same, against processed signals
  rpeak_robust_bundle.npz          # same, robust-ensemble run
  rpeak_funnel.csv                 # per-clip strip→video mapping diagnostic

  # Embedding validation artifacts
  frame_cache/*.npz                # cached DICOM frames (224px RGB, per clip)
  embedding_cache/*.npz            # cached JEPA-IN21K-e100 per-frame embeddings
  embedding_validation_results.*.csv  # per-anchor records, tagged by run
  embedding_validation_summary.*.txt  # regime-stratified Δ tables per run
  embedding_phase_diagnostic.png      # Δ_within vs anchor-phase bucket
  embedding_self_similarity_example.png  # sanity check: cyclic frame structure
  embedding_mapping_diagnostic.png    # sanity check: R-peak -> frame mapping
  lastframe/
    *.png                          # last frame per DICOM
    waveform/                      # vertical-cropped ECG strip
    waveform_extracted/            # trace-only on white
    waveform_processed/            # CANONICAL: {stem}.npz signal bundles +
                                   #   300dpi re-render PNGs for inspection
    ecg_diagnostics/               # amplitude + R-peak overlays
    hilbert_phase/                 # Hilbert phase plots per HR-inconsistent pair
```

## Pipeline (current)

1. **Sample & download** (`download_and_convert.py`, `download_multiclip_studies.py`): random-sample N rows from the MIMIC record list (seeded RNG) or target multi-clip studies for alignment-substrate experiments. Parallel `aws s3 cp` into `dicoms/`, convert multi-frame DICOMs to animated GIFs (YBR→RGB, PALETTE LUT, frame rate from `FrameTime` / `CineRate` / `RecommendedDisplayFrameRate`) into `gif/`. Single-frame stills are skipped by the GIF step but still produce a last-frame PNG.

2. **Last-frame extraction** (`extract_lastframe.py`): for every DICOM, save the final frame as `{stem}.png` in `lastframe/`. With `--extract-waveform`, also produces a trace-on-white PNG in `lastframe/waveform_extracted/` using the same green-family mask as the strip cropper.

3. **ECG strip localization** (`crop_waveform_frame.py`): vertical-only crop of the bottom ECG band. Full horizontal width is always preserved (ECG sweep length varies across clips).

   Detection uses per-row **colored-pixel density with a green-family color prior**. For each row `y`, count pixels that are saturated (`max_ch - min_ch > sat_thresh`) *and* green-dominant (`G > R and G > B`). Echo ECG traces are always green-family (green / teal / yellow-green / mint); color-Doppler blue (B-dominant), red annotations (R-dominant), and grayscale sector content (low saturation) all fail the test.

   Two-tier band selection:
   - **Core**: tallest contiguous run with `density >= core_density` in the bottom `1 - min_y_frac` of the frame, heights restricted to `[min_band_px, max_band_px]`.
   - **Edge expansion**: extend `y0` up / `y1` down while `density >= edge_density`, tolerating up to `edge_gap_px` below-threshold rows, capped at `edge_max_px` per side. Plus a `breathing_room=25` pixel pad to avoid clipping anti-aliased peak tips.

4. **Canonical signal preprocessing** (`process_waveform.py`): the primary artifact for downstream analysis. Reads `lastframe/waveform_extracted/*.png` (trace-on-white) and produces `lastframe/waveform_processed/{stem}.npz` with:
   - `full_y` (float32, length `W`): PCHIP-interpolated, median-filtered, sign-flipped amplitude, NaN outside the detected trace span. Preserves strip-column timing so cross-clip lags are measured in the same coordinate system.
   - `observed_mask` / `trace_span_mask` / `interpolated_mask` (bool, length `W`): gate samples by provenance — "real detection" vs "PCHIP-filled" vs "outside trace span."
   - `x0`, `x1`, `width`, `height`, `n_observed`, `coverage_frac`: geometry + a confidence score (`n_observed / (x1 − x0 + 1)`; typically ≥0.95 in practice).
   - Also writes a 300 dpi PNG render for visual inspection.
   Pipeline: luminance-threshold the white-background input → `binary_dilation` + largest-component to drop stray text/scale markers → per-column mask centroid → PCHIP fit across internal gaps → 3-px median filter → sign flip. Downstream scripts (xcorr, R-peak) read the NPZ by default and fall back to the legacy PNG → `cropped_to_signal` path only if the NPZ is absent.

5. **1-D signal** (`ecg_signal.py`): alternative signal path used by `run_ecg_signal_batch.py` and the legacy alignment tests. Converts a *cropped color strip* to a per-column amplitude (saturated AND green-dominant mask → greedy union of trace components → per-column median y → baseline-centered amplitude) and detects R-peaks with `neurokit2.ecg_peaks`. Superseded for alignment by `process_waveform.py`; kept for the diagnostic plots.

6. **Robust R-peak detection** (`rpeak_detectors.py`): `robust_rpeaks(signal, sr, hr_metadata)` cascades three detectors and picks the one whose peak-count best matches metadata HR × duration, scored by symmetric log-ratio distance.
   - NeuroKit2 (`nk.ecg_peaks`, black-box).
   - Pan-Tompkins with HR prior: refractory widened to `max(200 ms, 0.6 × expected_rr)` to suppress T-wave double-detection at slow HR; initial threshold primed from the median amplitude of local maxima spaced ~expected_rr apart (no adaptive-threshold warm-up needed on short strips).
   - Deterministic bandpass + refractory peak-picker, plus polarity-flipped re-run, used as a fallback when the best ensemble result is >~35% off expected (`log_ratio_dist > 0.3`).

   On the 30+30 test subset (same clips as the alignment tests), the robust ensemble improves strict-tolerance detection from **46% → 75%** at ±25% and **15% → 33%** at ±10%. Under the looser alignment-purpose rule (`max(±25%, ±1.5 beats)` with ≥2 peaks) it reaches **96%**. Method usage: Pan-Tompkins+HR-prior wins 57% of clips (dominant on short strips), NeuroKit2 40%, deterministic fallback 3%. Failure-mode autopsy on the full 488-NPZ set (`rpeak_failure_autopsy.py`): 54% of failures are under-detection by NeuroKit, 31% short clips (<3 expected beats — genuinely hard), 9% polarity flips, rest small.

7. **Calibration** (`ecg_calibration.py`, `run_calibration_batch.py`): determine per-clip sampling rate. `WaveformSequence.SamplingFrequency` would be authoritative but is **absent in every MIMIC clip we inspected** — the digital ECG isn't stored, only the burned-in overlay. Fall back to empirical scanner-modal sampling rate via `scanner_defaults.json` (Vivid E95/E90: 213 Hz; Vivid S70: 196 Hz; derived by back-solving `SR = 288 × displayed_HR / detected_HR_at_SR=288`). `PhysicalDeltaX` on `SequenceOfUltrasoundRegions` is **not** a valid source — when present it describes the spectral-Doppler axis, not the ECG overlay.

8. **Validation** (`validate_pipeline.py`): quality-classify each clip against metadata HR — `good` (detected HR within 10% of displayed), `hr_mismatch`, `irregular` (RR CV > 0.3), `no_detection`. Results in `pipeline_validation.csv`.

9. **Hand-label audit** (`select_handlabel_set.py` + `hand_label_compare.py`): stratified 40-clip sample with stub CSV for manual R-peak entry; `hand_label_compare.py` scores detector P/R/F1 against ground truth.

## Alignment-substrate experiments

All tests share the same 30+30 pair sample (seed 42). Three rounds: image-cropped signals, then `process_waveform.py` NPZs with neurokit-only detection, then NPZs with `robust_rpeaks`. The third is the current reference.

**Current reference numbers (`process_waveform.py` NPZs + robust_rpeaks):**

| Substrate | Within median | Cross median | Gap | HR-consistent | Pairs usable |
|---|---|---|---|---|---|
| xcorr (masked_normalized, overlap 0.8) | +0.47 | +0.17 | **+0.30** | 70% | 30 / 30 |
| R-peak phase (robust detector, offset-max `cos(2π Δφ)`) | +0.994 | +0.908 | +0.09 | 64% | **28 / 25** |

**R-peak detection progression on the 30+30 unique clips (110 clips):**

| Criterion | NK2-only (legacy) | NK2 on NPZs | Robust on NPZs |
|---|---|---|---|
| Loose rule `max(±25%, ±1.5 beats)` with ≥2 peaks | — | 72% | **96%** |
| Strict ±25% of expected (no floor) | — | 46% | **75%** |
| Strict ±10% of expected | — | 15% | **33%** |

**Legacy numbers (image-cropped signals):**

| Substrate | Within median | Cross median | Gap | HR-consistent |
|---|---|---|---|---|
| xcorr | +0.56 | +0.34 | +0.22 | 79% |
| R-peak phase | +0.997 | +0.93 | +0.07 | 80% (of 10 usable) |

**Scripts:**
- `test_xcorr_alignment.py` — masked-normalized cross-correlation with per-lag overlap denominator, padded to `max(len_a, len_b)`, `min_overlap_frac=0.8`. Reads NPZ by default; `--processed-dir ''` forces the legacy PNG path.
- `test_rpeak_alignment.py` — R-peak phase assignment + offset-max `mean cos(2π Δφ)`. Uses `robust_rpeaks` from `rpeak_detectors.py`.
- `test_hilbert_phase.py` — bandpass + Hilbert phase on HR-inconsistent within-study pairs. 1/6 pairs truly phase-locked (R > 0.6); the rest loosely track with 0.1–0.2 cycle jitter.
- `diagnose_hr_inconsistent.py` — metadata arbitration. Ruled out HR drift and scanner mismatch as causes of within-study HR-inconsistency; the residual ~21% is sub-cycle phase drift that DICOM metadata can't detect.

**Recommended substrate for multi-view JEPA:** xcorr on processed NPZs, filtered by `coverage_frac > 0.9` and `peak_correlation > 0.4`. R-peak phase (via `robust_rpeaks`) as an optional fallback for high-coverage clips where per-frame phase is wanted directly — filter by `ratio_distance < 0.1` (the strict ±10% bucket) if HR-locked alignment is required, since the added usable pairs from the robust detector come in at weaker HR-consistency (64% vs 74% under NK2-only).

## What Comes Next

9. **Frame-level phase bridge.** The ECG trace span `[x0, x1]` in the strip is **co-registered with the video**: column `x0` corresponds to video frame 0, column `x1` corresponds to frame `n_frames − 1`. Map any R-peak column `c` to its video frame via `round((c − x0) / (x1 − x0) × (n_frames − 1))`. Trace-span duration matches video duration to within ~5% (median ratio 1.00, IQR 0.98–1.05 across 268 clips). The empty ~5s-wide strip margins are not pre-video history — they're just empty PNG space and must not be treated as a time axis.

10. **Scale-out to 525k records** (done 2026-04-28 — `build_phase_annotations.py`). Produces `phase_annotations.parquet` with one row per clip: DICOM identity + `n_video_frames`, `fps_video`, `sr_ecg`, trace-span `[x0, x1]`, `coverage_frac`, detected R-peaks in video-frame coordinates, `per_frame_phase_json`, `confident_mask_json`, `regime_summary`, `full_y_b85` (float16-quantized base85-encoded raw 1D ECG so downstream can re-run R-peak detection without re-decoding DICOMs), and a `quality_tier` (`high` / `medium` / `low` / `reject`) derived from the tier thresholds below.

    Quality tiers (tune per training run; the thresholds in the script are the defaults):

    | Tier | Min in-video R-peaks | Min `coverage_frac` | Max `rpeak_ratio_dist` |
    |---|---|---|---|
    | high | ≥3 (strict phase everywhere) | ≥0.90 | ≤0.10 |
    | medium | ≥2 (permissive regime ok) | ≥0.80 | ≤0.25 |
    | low | ≥1 (HR-extrap rescue) | — | — |
    | reject | 0, or missing FrameTime/calibration/trace span | — | — |

    **Full-cohort results** (525,422 rows, 287 MB, one per MIMIC-IV-Echo record):

    | Tier | Count | % of total |
    |---|---|---|
    | reject | 239,633 | 45.6% |
    | high | 132,756 | 25.3% |
    | medium | 107,070 | 20.4% |
    | low | 45,963 | 8.7% |

    Reject reasons: 235,543 `single_or_no_frame_time` (non-cine DICOMs — spectral stills / M-mode / single-frame captures, 98.3% of rejects); 4,087 `waveform_band: no band found` (no ECG overlay detected, 1.7%); 3 `no_in_video_rpeaks`. Zero uncaught exceptions across 525k clips. The 45.6% reject rate is dominated by the record list including non-cine DICOMs, not by ECG-extraction failures — **true ECG pipeline failure rate is ~0.78%** (4,087 / 525,422).

    Multi-view yield (high+medium tiers, 239,826 clips): 7,083 studies have ≥1 good clip and **7,034 studies have ≥2 good clips** (99.3% of usable studies), totaling 239,777 good clips in multi-view-eligible studies. Median 35 good clips per study (IQR 26–42, max 118) — rich within-study density for phase-matched positive sampling.

    Artifacts: EFS `phase_annotations/phase_annotations.parquet`, S3 `s3://sagemaker-echojepa-h100-neurips-f85ad7df-bucket/phase_anno/phase_annotations.parquet`. The original SLURM `phase_annotations.sbatch` / `phase_annotations_aggregate.sbatch` exist but the production run was instead done on a 192-vCPU SageMaker c7i instance with GNU `xargs -P64` — see the development log.

11. **Within-study pair index and multi-view sampler.** Group `phase_annotations` by `study_id`, filter to studies with ≥2 clips in the target tier, compute per-pair quality scores (xcorr peak_correlation, HR drift, same-scanner flag), and build a `PhaseMatchedStudySampler` that — per epoch, per study — draws an anchor phase `φ ∈ [0, 1)` and yields (study, clip_a_frame, clip_b_frame) tuples where the frames have the closest confident phase to `φ`. The parquet already contains `per_frame_phase_json` + `confident_mask_json`, so this is a lookup in the sampler, not a compute step.

12. **Validation gate before training.** Before committing GPU-hours, verify on a held-out ~50-clip subset with ≥3 R-peaks that phase-matched frames across within-study clips land within ±1 frame of each other after round-tripping through the pipeline. Systematic offset >1 frame means there's a bias in detection or mapping to fix upstream. Post-training, re-measure Δ_within (the `embedding_substrate_validation.py` test) to confirm the phase-matched-positives training objective actually induces cross-clip phase representation.

13. **Re-detect R-peaks from `full_y_b85` to recover the ~9% clip drop** (deferred — do before the second pretraining run). The 2026-04-28 addendum showed that ~9.3% of high+medium clips (22,213 / 239,826) fail the median-RR-vs-metadata sanity check that catches every-Nth-beat detector errors. `PhaseMatchedStudySampler` now drops these via `require_rr_consistent=True`, reducing usable multi-clip studies 7,034 → 6,966. The proper fix is to re-run `robust_rpeaks` on the raw 1D ECG signal already persisted in the `full_y_b85` column, using a stricter metadata-supervised acceptance criterion (`0.80 ≤ detected_HR / metadata_HR ≤ 1.25`), and emit v2 columns (`r_peaks_video_v2_json`, `per_frame_phase_v2_json`, `confident_mask_v2_json`, `quality_tier_v2`). No DICOM re-decoding needed — all inputs are already in the parquet. CPU-bound and embarrassingly parallel: estimate 45–90 minutes wall time on a 192-vCPU host at ~50–100 clips/sec/core. Kept as v2 columns rather than overwrites so old labels remain for diagnostic comparison and downstream callers opt in via config. Recovers most of the dropped 22k clips as correctly-labeled training data, removes a ~10% supervision-quality drop across all future training runs, and reduces HR-stratified bias (fast hearts and long clips are over-represented in the dropped set). Skip before the first pretraining run — the sampler filter is enough to get a valid "does phase-matched training help" answer; do this before ablations 2+.

## Related work: Echo-SyncNet

Taheri Dezaki et al., *Echo-SyncNet: Self-supervised Cardiac View Synchronization in Echocardiography* (IEEE TMI, arXiv:2102.02287, 2021). Local copy at `refs/Echo-SyncNet/`. Code: github.com/fatemehtd/Echo-SyncNet.

Closest prior work to the multi-view phase-aligned pretraining our pipeline feeds. Worth comparing explicitly so scope is clear.

**What they do.** Train a ResNet-50 + 3D-conv frame encoder from scratch on ~1k–3k VGH patient studies with three SSL losses (weights 0.25 / 0.25 / 0.5):
1. *Temporal intra-view*: classify 3-tuples of frames from high-motion windows as sorted vs shuffled (BCE).
2. *Spatial intra-view*: N-tuplet metric loss — maximize adjacent-frame similarity, minimize beyond window radius α=5. Requires each training clip be ≈1 cardiac cycle (RepNet fallback when HR metadata unavailable).
3. *Inter-view synchronization*: soft-nearest-neighbor "temporal cycle consistency" (TCC, Dwibedi 2019) — embed cines A and B, for each `p_i ∈ A` compute its soft-NN `q̃_i` over B, then the softmax-distances from `q̃_i` back to A should peak near index `i`. Gaussian NLL enforces the round-trip.

Inference: DTW on the two embedding sequences produces frame-level alignment. No ECG used anywhere.

**Their evaluation.** AP2↔AP4 Kendall τ = 0.921 (vs 0.80 for TCC-only, +0.12 from adding the two intra-view regularizers); one-shot ED/ES transfer MAE 3–8 frames (~120–160 ms) inside reported inter-sonographer disagreement; EchoNet-Dynamic external transfer 4.0/3.1 frames ED/ES; linear probe on frozen embeddings beats a fully-supervised phase-regression baseline at low label fractions.

**How it relates to our approach.** Echo-SyncNet has to *discover* phase alignment from cycle-consistency because it has no ECG. Our pipeline extracts explicit per-frame phase labels from burned-in ECG R-peaks across 525k MIMIC clips — strictly more supervision — and will train on top of the 18M-echo EchoJEPA encoder rather than from scratch. On the "phase-aware representation for downstream clinical probes" axis we expect to dominate. On the "zero-ancillary-input sync at inference" axis Echo-SyncNet wins by construction: our training objective is disqualified anywhere the cine doesn't have an ECG overlay, even though the resulting *encoder* can be deployed without ECG once trained.

**What their ablation says we should borrow.** Table I shows inter-view TCC alone reaches κ=0.801; adding temporal-ordering + spatial n-tuplet intra-view losses moves it to 0.921. That's a large gain from regularizers we're not currently planning to include. Our multi-view JEPA loss is the analog of their inter-view loss alone; treating the two intra-view terms as a required ablation (not optional) is the cheap way to close the gap to their κ number and make a fair multi-view comparison.

**What it doesn't settle for us.** Echo-SyncNet validates against ED/ES sonographer labels, not against ECG ground truth, so its "phase alignment" is really "keyframe transfer under DTW." The `embedding_substrate_validation.py` result from this repo — frozen EchoJEPA-L-K exhibits Δ_within ≈ 0 across clips — means we have no *a priori* reason to expect our pretrained encoder to sync via DTW out of the box the way Echo-SyncNet's does after their training. The pre/post Δ_within measurement (step 12 above) is the test for whether the planned multi-view phase-matched fine-tune actually induces cross-clip phase representation on top of EchoJEPA.

## Development Log

Sessions of 2026-04-26 / 2026-04-27. Captured so future-us doesn't re-discover the dead ends.

### What we built

1. **S3 sampler + DICOM → GIF pipeline** (`download_and_convert.py`). Random sample N rows from `echo-record-list.csv`, parallel S3 download, convert multi-frame cines to animated GIFs at native frame rate. Runs `~20% yield` since most random rows are single-frame spectral stills.
2. **Last-frame extractor** (`extract_lastframe.py`). One PNG per DICOM. Reused the existing `core_lab/convert_dicoms.py` color-space handling (YBR auto-conversion by pydicom for JPEG baseline, PALETTE COLOR LUT expansion).
3. **ECG strip cropper** (`crop_waveform_frame.py` for single frames, `crop_waveform.py` for cines). Vertical-only crop, full width preserved.

### ECG detection — iterations that failed

The detector went through five versions before settling. Each failure taught us what signal actually separates the ECG from its neighbors:

**v1: row-max saturation, bottom-half search, tallest run.** Computed per-row peak saturation (`sat = max_ch - min_ch`, then `row_sat = sat.max(axis=1)`), searched below 55% of frame height, picked the tallest contiguous run above threshold.
- *Failure*: latched onto tall color-Doppler regions inside the ultrasound sector. These are saturated and tall.

**v2: narrowed search to bottom 18%, added a max-height cap.** The intuition was correct (ECG sits below the sector, is a thin strip) but selection flipped to "lowest qualifying run" which picked small footer-text bands instead of the real ECG.
- *Fix*: switched to "tallest within the bottom window" with `max_band_px=120` so Doppler gets height-rejected and the ECG wins among the remaining thin runs. Worked on ~17/20 samples.

**v3: loose edge expansion to recover peak/trough tips.** R-peak spikes and trough nadirs are thin, anti-aliased lines whose rows score below `sat_thresh`. Added a secondary `edge_thresh=15` that walks upward/downward from the core band while saturation stays above the looser threshold.
- *Failure*: `row_sat.max(axis=1)` tracks the single most-saturated pixel per row. Sparse artifacts (depth-marker numerals, yellow "15" text, vertical gradient swatch on the right margin) have saturated pixels at many rows, so expansion never stops — several crops latched onto nearly the whole 708-px frame.

**v4: colored-pixel density instead of row-max.** Counted `row_density[y] = #{x : sat(y, x) > sat_thresh}` instead of `row_sat[y]`. The ECG trace + gridlines contribute tens of colored pixels per row; sparse text contributes 2–3. Density threshold cleanly separates them, and edge expansion now uses a looser density (2 vs 10) with a gap tolerance.
- *Key insight, worth internalizing*: when your detector is getting fooled by *sparse* noise, switch from per-row max to per-row count. The robustness jump was substantial.
- *Failure*: on cines with color-Doppler sectors sitting right above the ECG strip (`94411759_0028`, `99190722_0036`), Doppler pixels cleared `sat_thresh` in bulk and the coarse band ran up into the sector.

**v5: green-family color prior.** All observed ECG traces are green / teal / yellow-green / mint (G-dominant). Color-Doppler is cyan/blue (B-dominant); red annotations are R-dominant; grayscale sector has low saturation. Redefined the density metric as `(sat > sat_thresh) & (G > R) & (G > B)`. With this prior, Doppler pixels never count toward row density, so the coarse band stays on the ECG even when color Doppler is adjacent.
- This is the current implementation.

**Tried but removed — color-distance refinement pass.** After v5 pass 1, we sampled the median RGB of green-dominant pixels inside the coarse band and reran detection using "within `color_dist_thresh` L1 of the trace color." Idea was to distinguish the exact trace hue from nearby green annotation text.
- *Failure*: the sampled median was dominated by the bright central body of the trace. Anti-aliased peak/trough tips have slightly different greens (darker, more desaturated) and fell outside `color_dist_thresh=50`, getting clipped. Pass 1 alone was already specific enough; pass 2 was strictly tighter and cut real signal. Removed.

### What actually worked (design principles learned)

- **Use density, not per-row max, when sparse artifacts are the adversary.** A handful of depth-marker pixels per row confuse a peak-detector; they don't move a density threshold.
- **Inject the smallest color prior that is always true.** "ECG is green-family" is a clean invariant across the MIMIC cohort and single-handedly solves color-Doppler contamination. A color-*distance* prior was too specific.
- **Cap edge expansion in pixels, not just by threshold.** Real peak tips are within ~20 px of the core band; infinite expansion with a gap-tolerance fallback walks into whatever colored region is above/below.
- **Two-tier thresholds (`core` strict, `edge` loose) beat single-threshold detection.** Recovers thin anti-aliased tips that a single stricter threshold clips, without dragging in noise.

### Open issues (known limitations of v5)

- A few crops still clip R-peak tops or trough bottoms by a few pixels when the anti-aliased ends never clear `edge_density=2`. Nudging `--edge-max-px` up or `--edge-density` down may help, but both trade off against walking into adjacent overlays. Case-by-case tune, or accept a small tip loss.
- `--min-y=0.82` assumes the ECG is in the bottom 18%. If a future vendor lays out the ECG higher, this would need to be relaxed.
- Scaled up from 20 to 500 MIMIC DICOMs (37 multi-clip studies, 495 cropped strips). Still GE-only; other vendors untested.

### 1-D signal + R-peak detection (post-crop)

- `ecg_signal.cropped_to_signal` converts cropped strips to per-column amplitude via green-dominant trace mask + greedy-union connected-component filter + per-column median y.
- **T-wave doubling collapse**: scipy `find_peaks` with adaptive prominence was catching T-waves and overcounting beats (120/316 clips flagged; typical `detected/displayed` ≈ 2×). Swapped to `neurokit2.ecg_peaks` (method="neurokit") — morphology + gradient-based, resistant to T-wave confusion. `good` classification jumped from 6% to 41.5%; 120/316 T-wave-doubled cases collapsed to 1.
- **Largest-valid-segment detection**: `cropped_to_signal` returns a valid mask; R-peak detection runs only on the longest contiguous valid run to avoid spurious peaks at zero-filled gaps.

### Calibration (per-clip sampling-rate recon)

Initial pass used `PhysicalDeltaX` on `SequenceOfUltrasoundRegions` → implied SR = 288 Hz, producing 92% HR-mismatch. Investigation revealed:
- `PhysicalDeltaX` is only populated on spectral-Doppler stills, where it describes the *spectrum's* time axis, not the ECG overlay's. On cines it's either absent or describes cm/px (spatial, not temporal).
- `WaveformSequence` is absent in all 200 MIMIC clips we checked — there's no digital ECG stored alongside the imaging.
- Empirical back-solve via `SR_true = 288 × displayed_HR / detected_HR_at_SR=288` converged to ~213 Hz for Vivid E95/E90 and ~196 Hz for Vivid S70 (tight clusters). These became the `scanner_defaults.json` values.
- Metadata can **only** answer "same scanner?" and "did HR change between clips?" — *not* "did sweep speed drift between clips?" The latter would require measuring from the image itself.

### Alignment substrate (closed out 2026-04-27)

The question: can we align cardiac phase across clips in the same study (for multi-view JEPA training) using only the extracted ECG? Explored four substrates; documented so we don't relitigate.

1. **Single-offset cross-correlation** (`test_xcorr_alignment.py`). Started constant-n normalization (0.21 within / 0.09 cross, 76% HR-consistent). Iterated through per-lag overlap normalization, masked invalid-region zeroing, edge-artifact clamping (`min_overlap_frac=0.8`), length padding (no effect — MIMIC clip durations cluster tightly at 4.8–5.2 s), lag-constraint pre-filter, and phase-only FFT correlation. Baseline ended at **+0.56 within, +0.34 cross, gap +0.22, 79% HR-consistent, 1.78× sharpness**. Neither lag-constraint nor phase-only meaningfully changed the numbers.

2. **Hilbert-phase per-sample** (`test_hilbert_phase.py`). On 6 HR-inconsistent within-study pairs: 1 pair cleanly locked (R=0.78, circular-std=0.11 cycles), 5 loosely tracked with 0.1–0.2 cycle jitter. The bandpass+Hilbert pipeline isn't broken but doesn't substantially out-perform single-offset xcorr on this data.

3. **R-peak phase alignment** (`test_rpeak_alignment.py`). Expected to be the cleanest substrate — per-frame phase in [0, 1) between detected R-peaks, offset chosen to maximize `mean cos(2π Δφ)`. **The bottleneck is upstream**: `cropped_to_signal`'s valid-column mask rejects 20–65% of strip columns, so the detector only sees 1.6–3.9 s of signal per clip, not the full ~5 s. At that length `neurokit2.ecg_peaks` returns 2–4 peaks where 5–10 are expected. Only 10/30 within-study and 6/30 cross-study pairs passed detection. On the pairs that survived, the phase-agreement metric saturates (both within and cross ≈ 0.99) and the discrimination gap collapses to +0.07. Not the detector's fault — it's that you can always pin one R-peak onto another when the signal is short.

4. **HR-drift / scanner-mismatch arbitration** (`diagnose_hr_inconsistent.py`). For the 7 HR-inconsistent within-study xcorr pairs: consistent and inconsistent pairs have *identical* HR-delta distributions (both medians 3 bpm; one "consistent" pair has 39 bpm drift). Zero scanner-model mismatches in either group. The remaining 21% of within-study HR-inconsistency is sub-cycle phase drift — either a too-strict T/4 threshold (T/3 saturates the null, so T/4 is the right threshold to keep) or real per-clip sweep-speed variation that isn't in the header.

**Decision (provisional, revised 2026-04-27 after NPZ rework)**: xcorr on image-cropped signals gave +0.56 within / +0.34 cross / 79% HR-consistent. Re-running on `process_waveform.py` NPZs moved the numbers to **+0.47 within / +0.17 cross / gap +0.30 / 70% HR-consistent** — within dropped modestly, but cross-study collapsed, which is what actually matters for discrimination. The cleaner signals also unblocked the R-peak path: detection success rose 48% → 72%, and R-peak full-set within median jumped from 0 to +0.99. The production substrate is now xcorr on NPZs with coverage/correlation filtering; R-peak phase is a viable per-frame fallback on high-coverage clips.

### 2026-04-27 addendum: embedding-similarity substrate validation + mapping bug

Tested whether R-peak phase alignment provides a useful supervision signal for multi-view JEPA training by measuring, on JEPA-IN21K-e100, whether frames at matched cardiac phase are more similar in embedding space than random-offset frames (`embedding_substrate_validation.py`).

**ECG→video mapping bug (caught late, fixed).** First attempt mapped ECG strip columns to video frames using a "right-edge = now" convention, assuming the full ~5s PNG width represented continuously-scrolling ECG history back to `t = -W/sr`. That was wrong: the blank left and right margins of the strip PNG are just empty container space, not pre-video history. Empirically (n=268 clips): **trace-span duration / video duration has median 1.00, IQR 0.98–1.05.** The ECG trace is drawn only across the detected trace span `[x0, x1]`, which is co-registered with the video — column `x0` is frame 0, column `x1` is the last frame. Strip width `W` is a fixed ~5s container independent of video duration.

Under the buggy mapping, only 24/104 cached clips had ≥2 R-peaks inside the video window (the rest got mapped to negative frame indices). Under the corrected linear mapping, **102/104 clips have ≥2 R-peaks in-video** and the HR-extrapolation rescue path is no longer needed. Funnel:

| Stage | Buggy mapping | Corrected mapping |
|---|---|---|
| Embedded clips | 104 | 104 |
| ≥2 R-peaks on strip | 102 | 102 |
| ≥2 R-peaks in video window | **24** | **102** |
| ≥1 confident frame (strict) | 24 | 102 |
| Multi-clip studies | 6 | 30 |
| Within-study pairs possible | 6 | 40 |

**Substrate validation results (corrected mapping, JEPA-IN21K-e100, n=320 anchors across 40 pairs, 30 studies):**

| Metric | median | IQR | frac > 0 | Wilcoxon p |
|---|---|---|---|---|
| sim_phase_within | +0.809 | [0.72, 0.86] | 100% | — |
| sim_random_within | +0.807 | [0.72, 0.86] | 100% | — |
| sim_phase_cross | +0.702 | [0.67, 0.75] | 100% | — |
| Δ_within (phase − random) | **+0.001** | [−0.007, +0.007] | 52% | 0.29 (ns) |
| Δ_specificity (within − cross) | **+0.087** | [+0.002, +0.166] | 76% | **7×10⁻³³** |

Regime stratification (strict vs permissive vs HR-extrapolated) is now largely moot because 94% of anchors are strict or permissive; HR-extrap contributes only 8/320 records. Strict-only subset (n=150, 34 pairs): Δ_within = +0.000, Δ_spec = +0.098 (84% frac>0).

Phase-bucket breakdown on the strict subset (in-video R-peaks, measured RR; the highest-precision regime) shows Δ_within flat at zero across the cycle:

| Phase bucket | n | median Δ_within | frac > 0 |
|---|---|---|---|
| [0.0, 0.2) (R-peak / early systole) | 31 | +0.000 | 52% |
| [0.2, 0.4) (peak ejection) | 42 | −0.000 | 48% |
| [0.4, 0.6) (end-systole / early diastole) | 21 | +0.001 | 57% |
| [0.6, 0.8) (mid-diastole) | 41 | +0.000 | 51% |
| [0.8, 1.0) (pre-R) | 15 | −0.004 | 47% |

Every bucket sits at chance (50% frac>0) within noise — uniform across systole and diastole.

**What this measures (and what it doesn't).** The experiment tests whether **two independent clips of the same patient**, at R-peak-matched cardiac phase, are more similar in embedding space than at random offsets. It is a **cross-clip phase transferability** test, not a test of whether the encoder represents phase at all:
- The sanity-check self-similarity matrix shows clear cyclic structure within a single clip — JEPA-IN21K's masked-latent-prediction objective on 16-frame clips necessarily encodes *something* about cyclic cardiac motion, because temporally adjacent frames have to be predictable from each other.
- But "phase representation within a clip" ≠ "phase alignment across clips." The pretraining objective has no pressure to make frame-at-phase-0.3 of acquisition A look like frame-at-phase-0.3 of acquisition B — that cross-acquisition alignment is precisely what multi-view JEPA with phase-matched positives would train.

**What the result does say:**
1. JEPA-IN21K-e100 represents **patient/scanner/view identity sharply** across clips (Δ_spec +0.087, p=10⁻³³). Same-study multi-view positives already carry a strong, workable signal on this encoder.
2. JEPA-IN21K-e100, **as pretrained**, does not exhibit cross-clip phase alignment in its embedding geometry. Matched-phase and random-offset frames from different acquisitions look indistinguishable.
3. This means **the pretrained encoder would not receive a differentiating gradient signal from phase-matched vs random-offset positives if we used it frozen**. It does **not** mean phase-matched positives wouldn't induce phase-aware cross-clip representation under a contrastive fine-tune — that is exactly what such a training objective is designed to produce, and whether it works is an empirical training question, not a frozen-encoder question.

**What it doesn't settle:**
- Whether a multi-view JEPA training run with R-peak-phase-matched positives would acquire cross-clip phase alignment. The substrate itself (R-peak detection, strip→video mapping, phase assignment) is validated and produces meaningful phase-matched frame pairs.
- Whether a phase-aware pretrain (EchoJEPA-L-K variants, or a dedicated temporal-alignment objective) would show Δ_within > 0 out of the box.

**Reasonable next steps** (not requirements):
- **Pre/post training delta**: measure Δ_within after a short multi-view fine-tune with phase-matched positives. If Δ_within grows, training with the substrate successfully teaches cross-clip phase alignment — the substrate is doing useful work through the loss.
- **Alternative encoder spot-check**: same test on an echo-specialized encoder (EchoJEPA-L) to see whether any pretrained echo encoder exhibits cross-clip phase alignment without explicit phase supervision.
- **If chasing phase signal is not the priority**: the same-study Δ_specificity signal is strong enough that multi-view JEPA can be designed around it directly, with phase alignment as an optional additional constraint rather than the primary supervision.

The finding is the same as the pre-fix version but now on a 6× larger sample with no reliance on the HR-extrapolation rescue. The mapping bug didn't flip the conclusion; it made the earlier n=48 measurement underpowered and muddled by regime confounds.

Sanity checks (`embedding_substrate_validation.py` gated sanity suite):
1. Encoder adjacency: same-clip cos=+0.99 vs cross-clip cos=+0.78 — PASS.
2. R-peak → video frame mapping: visually verified on a sample clip; R-peak frames look consistently like the same cardiac phase.
3. Per-clip self-similarity matrix: cyclic structure visible in the lag-decay.

**Takeaway (not "recommendation — don't do it"):** the substrate itself is validated — R-peak detection, strip↔video mapping, and phase assignment all work correctly, and the substrate produces meaningful phase-matched frame pairs. What this experiment can't answer from a frozen encoder is whether training a multi-view JEPA with those phase-matched positives *induces* cross-clip phase representation. If downstream decides to train with phase-matched positives, the appropriate check is a pre/post Δ_within comparison on the trained encoder, not a frozen-encoder test. Independently, the strong Δ_specificity signal means same-study positives are already a workable multi-view supervision — phase matching is an additional constraint that may or may not help on top.

### 2026-04-27 addendum: robust R-peak detection

After the NPZ preprocessing unblocked the R-peak path, focused on getting detection success up from 46% (strict ±25% on 30+30 subset) toward 90%+. Approach: **metadata-HR-supervised ensemble** rather than tuning any single detector. Built `rpeak_detectors.py` with:

1. **NeuroKit2** (`nk.ecg_peaks`, black-box).
2. **Pan-Tompkins with HR prior**: refractory = `max(200 ms, 0.6 × expected_rr)` (kills T-wave double-detection at slow HR); initial threshold primed from median amplitude of local maxima spaced ~expected_rr apart (no adaptive warm-up needed on short strips).
3. **Deterministic bandpass + refractory peak-picker** as fallback (no adaptive thresholding; relies on metadata HR for min-spacing). Plus a polarity-flipped re-run for inverted QRS.

Scoring: log-ratio distance `|log(n_detected / expected)|`, symmetric in over/under-detection. Whichever detector (or fallback) lands closest to metadata expectation wins.

Results on the 30+30 test subset (110 unique clips):

| Criterion | NK2-only | Robust |
|---|---|---|
| Loose (`max(±25%, ±1.5 beats)`, ≥2 peaks) | 72% | **96%** |
| Strict ±25% | 46% | **75%** |
| Strict ±10% | 15% | **33%** |
| Short clips (<3 expected beats), strict ±25% | 34% (13/38) | **63%** (24/38) |
| Normal clips (≥3 beats), strict ±25% | 53% (38/72) | **82%** (59/72) |

Method usage: **Pan-Tompkins+HR-prior wins 57% of clips** (dominant on short strips where NK2's adaptive threshold can't converge), NeuroKit2 40%, fallback 3%. Polarity-flipped fallback fired 0% on the test subset despite the full-488 autopsy showing 9% polarity flips — either sample variance or the HR-primed Pan-Tompkins is handling them without the flip.

**Full-set autopsy** (`rpeak_failure_autopsy.py`, all 488 NPZs under NK2-only, strict ±25%):
- `other_under` (detector misses peaks): 165 / 303 failures (54%)
- `short_clip` (<3 expected beats — genuinely hard): 95 (31%)
- `polarity_flip`: 26 (9%)
- `baseline_wander`, `saturation_clipping`, `t_wave_double`: a few each

**Caveat on the "96% vs 38%" framing.** The loose rule's 1.5-beat absolute floor does real work on short clips — an expected count of 2 passes with 1, 2, or 3 detections. That's reasonable for alignment-purpose detection (only needs ≥2 peaks) but not a general "detector accuracy" claim. The honest numbers are the strict-tolerance rows (46% → 75% at ±25%, 15% → 33% at ±10%). "Bottleneck closed" applies to alignment use; "R-peak detection solved" would require the strict ±25% at ≥90%.

### 2026-04-27 addendum: `process_waveform.py` canonicalization

Originally written as a rendering sidecar (save a 300 dpi PNG of the trace). Rewritten to produce `{stem}.npz` as the primary artifact with:
- `full_y` in strip-column coordinates (NaN outside trace span, so cross-clip lags are comparable)
- `observed_mask` / `trace_span_mask` / `interpolated_mask` for provenance-aware gating
- `x0`, `x1`, `width`, `coverage_frac` for geometry + a confidence score
`test_xcorr_alignment.py` and `test_rpeak_alignment.py` both prefer the NPZ path and fall back to `cropped_to_signal` only when the NPZ is missing. Flags: `--no-save-signal` to render only; `--no-render` to skip the PNG.

### Read-tool display caveat

While iterating, the Read tool's image preview showed stale cached renders for PNGs overwritten in place — the on-disk bytes were correct (verified via `np.array_equal(full[y0:y1], crop)`), but the rendered preview lagged. Using unique filenames per iteration, or stacking all crops into a single image with an obvious marker (red separator bars), cache-busts the preview. Trust the numbers (`crop.shape`, `band=[y0,y1]`), not the inline image when iterating.

### 2026-04-28 addendum: full-cohort phase-annotation build

Ran `build_phase_annotations.py` over all 525,422 MIMIC-IV-Echo records to produce the production `phase_annotations.parquet` that feeds the multi-view phase-matched JEPA sampler.

**Run configuration.** Not SLURM — the original plan was `phase_annotations.sbatch` (`--array=0-255%16` on HyperPod), but we opted for a SageMaker `ml.c7i.48xlarge` instance (192 vCPU, 384 GB RAM, CPU-only) with GNU `xargs -P64` over 256 shards, each processing ~2,053 records. Scratch on `/tmp` (instance-local NVMe), not EFS. Added a v2 worker change mid-run: introduced `full_y_b85` column (float16 quantized, base85-encoded raw 1D ECG signal, ~2.5 KB per clip, +1.3 GB total) so downstream re-runs of phase assignment or alternative R-peak detectors don't need to re-decode DICOMs. Required clearing shards and restarting — v2 adds the column to the dataclass; existing parquets would have lacked it.

**Throughput ceiling finding.** Steady-state per-worker rate was 0.4 rec/s, aggregate ~25 rec/s across 64 workers, 70% CPU idle. Ramped to `-P128` to test headroom: per-worker rate **halved** to 0.2 rec/s, aggregate stayed at ~25 rec/s. Reverted to `-P64`. The bottleneck is aggregate S3-GET throughput on this single host — likely VPC NAT gateway path or single-EC2 aws-cli process saturation, not bucket-side throttling (zero `SlowDown`/`ThrottlingException` observed across the entire run). Lesson: on a single-instance fan-out against S3, per-host aggregate matters more than per-worker parallelism beyond the point where S3 becomes the wait. Full run took ~17h wall-time.

**Results.**

| Tier | Count | % | Notes |
|---|---|---|---|
| reject | 239,633 | 45.6% | 98.3% `single_or_no_frame_time` (non-cine DICOMs) |
| high | 132,756 | 25.3% | ≥3 in-video R-peaks, coverage ≥0.90, ratio-dist ≤0.10 |
| medium | 107,070 | 20.4% | ≥2 R-peaks, coverage ≥0.80, ratio-dist ≤0.25 |
| low | 45,963 | 8.7% | ≥1 R-peak (HR-extrap rescue) |

True ECG-extraction failure rate is **0.78%** (4,087 clips `waveform_band: no band found`). Zero uncaught exceptions across 525k records — every reject is a known, classified reason. The high reject percentage is entirely driven by the record list including spectral stills / M-mode / single-frame captures that we don't target.

**Multi-view yield** (high+medium tiers, 239,826 clips): 7,083 studies with any good clip; **7,034 studies (99.3%) have ≥2 good clips**; 239,777 good clips sit in multi-view-eligible studies. Per-study clip density: median 35, IQR 26–42, max 118.

The step-12 validation gate ("phase-matched frames land within ±1 frame across within-study clips") can now run against this parquet; the `PhaseMatchedStudySampler` (step 11) has its upstream input.

### 2026-04-28 addendum: sampler + validation gate + before/after demo

Built the within-study multi-view sampler infrastructure on top of the parquet and ran the step-12 validation gate. Caught two detector-tier-gate bugs while building a visual demo.

**`sampler/phase_matched_sampler.py::PhaseMatchedStudySampler`** — per-epoch generator of phase-matched within-study clip pairs. Reads the parquet, filters to the target quality tier (default `high` + `medium`), groups by `study_id`, drops single-clip studies. 7,034 multi-clip studies → 7,034 records per epoch at `pairs_per_study=1` (or 28k at `pairs_per_study=4`). Construction ~7.5s, per-epoch build ~1.1s. Each `MatchRecord` carries `(study_id, clip_a, clip_b, target_phi)` with per-clip `(row_idx, dicom_id, n_frames, anchor_frame, phase_at_anchor)`. Distributed split mirrors `DistributedStudySampler` (seed + epoch → shuffle → pad → rank slice). Exposes a `torch.utils.data.Sampler`-compatible `__iter__` yielding clip_b row indices plus a `last_records` property for the paired-loader wrapper to consume both clips.

**Design gap — now closed.** `VideoGroupDataset._loadvideo_decord_multi` previously picked its window start as `i * clip_len` (deterministic strided from 0), ignoring any caller-specified anchor. Plumbed an optional `anchor_frame` kwarg through `_loadvideo_decord_multi → _get_item_row → __getitem__` and added a `set_anchors_by_index(dict)` setter on the dataset. When an anchor is provided, the K-clip span is recentered so its geometric center is at the anchor frame, clipped to `[0, V − K·clip_len]` so all K windows stay in-bounds. Matching helper `PhaseMatchedStudySampler.build_anchor_table(side="b"|"a"|"both")` emits `{row_idx: [anchor_frame]}` for the dataset. Training loop wires this before each epoch: `sampler.set_epoch(e); list(iter(sampler)); dataset.set_anchors_by_index(sampler.build_anchor_table())`. All existing callers unaffected (`anchor_frame=None` preserves the old behavior bit-for-bit). Verified for V ∈ {100, 200} × anchor ∈ {None, 50, 10, 190} and K=2 centering at V=300.

**`sampler/phase_matched_validation.py` — step-12 gate, passed.** 50 studies × 264K pair-anchor matches (8 anchors × all clip pairs per study × 50 studies): median cross-clip phase disagreement **0.0047 cycles** (≈0.4 source frames), p90 0.019 cycles (~1.5 frames), max 0.063. Well inside the 0.05-cycle gate. No systematic bias in the detection-or-mapping pipeline across within-study clips.

**`sampler/prepost_delta_within.py`** — harness for the post-training measurement. Invokes `embedding_substrate_validation.py` twice via subprocess with monkey-patched `CHECKPOINT`, diffs the Δ_within / Δ_specificity distributions, emits Wilcoxon p-values. Useful pre-only today to lock in the baseline number before the multi-view fine-tune launches.

**Before/after visualization — caught two bugs in the tier gate via visual inspection.**

Built `sampler/make_before_after_stacks.py` to produce stacked GIFs of within-study pairs (top/bottom panel separated by a 4-px gap, 30 frames at 15 fps). "Prealign" = first 30 source frames of each clip as-is. "Aligned" = 30 phases spanning [0, 1) walked by `nearest_confident_frame` inside one RR interval per clip. The aligned GIFs showed a clear speed mismatch on one pair (97877557 A4C+A4C). Root cause:

1. **`rpeak_ratio_dist ≤ 0.10` does not imply per-clip RR consistency.** On a 501-frame clip (97877557_0073), the detector marked every 4th beat — RR=[110, 111, 110, 110] against a true cycle length of 26f (HR_meta=69, fps=29.9). The clip still passed the high-tier gate because the *total* detected-count-vs-duration ratio is self-consistent: marking every 4th beat at 5 beats across ~16.5 s yields log-ratio distance 0.075, under the 0.10 threshold.
2. **"Longest RR" is the wrong interval walker.** Clips with a single missed beat end up with one RR being ~2× the others (e.g. 97691703_0021 RR=[12, 22]). Picking the longest selects the interval containing the missed beat, and the sampler then stretches two real cycles over the aligned GIF while the partner clip covers one.

Fixes landed in `make_phase_aligned_gifs.py`:

- `_pick_rr_interval(strategy=...)` gained `"nearest_meta"` (pick the RR closest to `60 × fps / HR_metadata`) and `"median"` (pick the RR closest to the median RR length). `nearest_meta` is the new default. "Longest" retained for legacy callers but no longer recommended.
- Two-layer per-clip sanity used for demo-pair selection:
  - `median(RR) / (60 × fps / HR_meta) ∈ [0.80, 1.25]` — guards against every-Nth-beat detection.
  - `max(RR) / min(RR) ≤ 1.40` — guards against missed beats inside an otherwise-good clip.

Applied to the 114 on-disk high-tier clips, the two layers drop **22 clips (19%)**. Ranking the surviving within-study pairs by naive first-30-frame phase disagreement and regenerating gave three clean before/after demos:

| Study | Views | Pre mean \|Δφ\| | Post mean \|Δφ\| | Reduction |
|---|---|---|---|---|
| 94712615 | PLAX + PLAX | 0.221 | 0.000 | >500× |
| 96166542 | PSAX-AV + A5C | 0.096 | 0.024 | 4× |
| 95624795 | A4C + Exclude | 0.061 | 0.000 | >500× |

The 96166542 residual is **not alignment failure** — it's a real HR change between acquisitions (103 → 83 bpm). Clip _0014's RR window (15f) is correctly shorter than _0043's (22f); each clip's frames are correctly labeled against its own cycle; aligning by phase reduces disagreement to one frame's worth of quantization. This is the expected behavior and actually an argument *for* phase-based alignment over any fixed-length DTW: the patient's heart rate changing mid-exam doesn't break the supervision signal.

GIFs at `examples/phase_aligned/<study>/<clip_a>_<clip_b>_stacked{,_prealign}.gif`.

**Consequence for `PhaseMatchedStudySampler` — partially fixed.** Added `require_rr_consistent=True` (default on) to the sampler constructor: drops clips where `median(RR) / (60 · fps / HR_metadata)` falls outside `[0.80, 1.25]`. Guards against every-Nth-beat detector errors that the quality-tier gate's `rpeak_ratio_dist ≤ 0.10` accepts because it's count-vs-duration-consistent. Applied to the full parquet: **22,213 of 239,826 high+medium clips (9.3%) fail the check and are dropped**, reducing usable multi-clip studies 7,034 → 6,966 (-0.97% — most studies retain ≥2 good clips after the bad one is dropped). The mirror `max/min RR ratio ≤ 1.40` check was intentionally *not* added to the sampler: legitimate arrhythmia (AFib) produces beat-to-beat variance above that threshold and we want those patients in training. Deliberate scope: the per-clip "missed beat inside a single RR" issue (the `max/min` case) is not yet addressed for training — the affected clip still enters the sampler, but `PhaseMatchedStudySampler._draw_pair` picks a global anchor and `_nearest_confident_frame` doesn't restrict to any single RR, so the bad interval contributes a minority of pairs rather than corrupting every pair from that clip. Full re-detection from `full_y_b85` is the ambitious fix — logged as roadmap step 13.

**Playback fix in the demo GIFs (visualization only — no training impact).** The aligned stacks initially played choppy because they resampled each clip's RR to a fixed 30-frame / 15-fps output, quantizing browser GIF durations below the ~20 ms floor and hiding cross-clip HR differences. Rewrote `make_before_after_stacks.py` to play each panel's native source frames at source fps (~30 fps, 33-34 ms per-frame duration). Aligned stacks now show one RR interval per panel looped 3× at each patient's real heart rate; when HR differs between the two acquisitions (e.g., 96166542 at 119 vs 81 bpm → 15f vs 22f RR), the top panel *visibly* beats faster than the bottom instead of artificially moving in lockstep. This is cosmetic: training loads tensors of N frames and doesn't have a wall-clock notion — phase φ is a cycle fraction, HR-invariant by construction, so sampler-emitted positive pairs are valid regardless of how a demo renders them.

**Demo set expanded.** From 3 pairs (one apical) to 9 pairs (7 apical, including all 6 apical view-combo pairings). Joined the parquet against `classifier/output/mimic_view_predictions.csv`, filtered to high-tier + RR-consistent + apical-mixed, ranked by pre-alignment |Δφ|, downloaded the 12 DICOMs for the top unique-study pair in each view-combo. Every new pair has pre |Δφ| ≥ 0.488 (near the theoretical max of 0.5, meaning the naive first-30-frame stacks play at opposite phases of the cardiac cycle) and post |Δφ| ≤ 0.017 — reduction factors 29×-500×+. Artifacts at `examples/phase_aligned/<study>/`:

| Study | Views | HR A → B | Pre \|Δφ\| | Post \|Δφ\| |
|---|---|---|---|---|
| 95462184 | A4C + A2C | 112 → 112 | 0.500 | 0.000 |
| 92900930 | A2C + A3C | 53 → 51 | 0.498 | 0.010 |
| 91722876 | A2C + A5C | 69 → 69 | 0.497 | 0.000 |
| 94330184 | A4C + A3C | 75 → 72 | 0.493 | 0.015 |
| 95373991 | A5C + A3C | 61 → 43 | 0.490 | 0.010 |
| 93083042 | A4C + A5C | 64 → 56 | 0.489 | 0.017 |

## Known Edge Cases

- **Color-Doppler bleed** above the ECG strip: handled by the green-dominant prior — Doppler blue/cyan fails `G > B`.
- **Red sweep-cursor arrow / red spots** adjacent to the trace: fail `G > R`, correctly excluded.
- **Green depth-marker text** near the strip: survives the green prior but contributes only a few colored pixels per row; the density threshold filters it out.
- **Tips of R-peaks and deepest troughs**: anti-aliased spikes with lower colored-pixel count per row. The two-tier edge expansion plus `breathing_room=25` padding recovers them.
- **Spectral/M-mode single-frame DICOMs**: the sampler keeps them in `dicoms/` so last-frame extraction still produces a PNG, but `download_and_convert.py` correctly skips them during GIF conversion.
- **Sparse valid-column mask**: `cropped_to_signal` flags only columns with enough trace pixels. On clips where the baseline between R-waves is thin, 20–65% of columns end up invalid and the signal the R-peak detector sees is 1.6–3.9 s instead of ~5 s. This is the main limiter of downstream analyses; fix upstream rather than in the detector.
- **Strip container vs trace span**: the ECG strip PNG is a fixed ~5s-wide image (1016 px at ~213 Hz, or ~1016 px at ~196 Hz); the actual ECG trace is only drawn across `[x0, x1]` which matches the video's duration. The empty left/right margins are not pre-video history — treating them as a time axis will map R-peaks to fictitious pre-video times. Use `[x0, x1]` from the processed NPZ and map linearly: `frame = round((col − x0) / (x1 − x0) × (n_frames − 1))`.
- **Per-clip sweep-speed drift within a study**: not detectable from DICOM metadata. Same-scanner, same-patient clips can still have slightly different `px/s` calibration between acquisitions. Shows up as a T/3-scale phase offset in within-study xcorr and has no clean fix short of per-clip HR back-solving (which re-introduces detection dependence).

## References

- `uhn_echo/nature_medicine/core_lab/convert_dicoms.py` — DICOM → MP4/PNG converter. Reused pydicom read + YBR→RGB + PALETTE LUT handling patterns.
- DICOM PS3.3 Sect. C.8.5.5 "Ultrasound Region Calibration Module" — `SequenceOfUltrasoundRegions` semantics for locating the sector boundary.
- `claude/neurips/experiments/cmr-probe-family-preliminary.md` and `claude/neurips/probing-experiments.md` — related phase-aware-training motivation for this pipeline.
