# Nature Medicine New-Tasks Build Scripts

Build scripts and documentation for the new Nature Medicine probe tasks: AV/MV Status, MV E' medial, data-efficiency subsets, and the MIMIC zero-shot test sets. The bulk of this document covers AV Status and MV Status; per-script summaries for the others appear in the second half.

## Task summary table

| Task | Type | Views | Color/B-mode | Classes / Range |
|------|------|-------|--------------|-----------------|
| **RV Basal Diam** | Regression | A4C | B-mode only | cm (1.0–7.0) |
| **VSD** | Binary | PLAX, PSAX-AV, A4C | Includes color | 0=No VSD, 1=VSD (active or repaired) |
| **MV Status** | 4-class | PLAX, A4C | Both | 0=mechanical, 1=bioprosthetic, 2=repair, 3=native |
| **AV Status** | 4-class | PLAX, A4C, A3C | Both | 0=mechanical, 1=surgical bio, 2=TAVR, 3=native |
| **MV E/A** | Regression | A4C, A2C | Color-trained | ratio (0.3–5.0) |
| **LA AP Diam** | Regression | PLAX | B-mode only | cm (1.5–7.0) |
| **Ao Root Diam** | Regression | PLAX | B-mode only | cm (2.0–6.0)¹ |
| **E' Medial** | Regression | A4C | B-mode only | cm/s (0.02–0.25) |

¹ Sinus of Valsalva diameter (true aortic root, `sinus_diam` in MIMIC). Earlier builds used `ascending_diam` at a different anatomical level; the corrected build uses `sinus_diam` with a 2.0 cm lower bound.

## Chicago (UCMC) probe artifacts

Trained probes for these tasks are mirrored on Google Drive at `gdrive:echo_foundation/nature_medicine/chicago/probes/new/`. One subdirectory per (task, model) with a single `best.pt` checkpoint inside. Use `rclone copy gdrive:echo_foundation/nature_medicine/chicago/probes/new/<dir> <local>` to pull.

### Probes covered by the build scripts in this directory

| Task | EchoJEPA-G | EchoJEPA-L-K | EchoPrime | PanEcho |
|---|---|---|---|---|
| AV Status | `av_status-echojepa-g/` | `av_status-echojepa-l-k/` | `av_status-echoprime/` | `av_status-panecho/` |
| MV Status | `mv_status-echojepa-g/` | `mv_status-echojepa-l-k/` | `mv_status-echoprime/` | `mv_status-panecho/` |
| MV E' medial | `mv_e_prime_medial-echojepa-g/` | `mv_e_prime_medial-echojepa-l-k/` | `mv_e_prime_medial-echoprime/` | `mv_e_prime_medial-panecho/` |
| MV E/A ratio | `mv_ea_ratio-echojepa-g/` | `mv_ea_ratio-echojepa-l-k/` | `mv_ea_ratio-echoprime/` | `mv_ea_ratio-panecho/` |
| LA AP Diam | `la_ap_diam-echojepa-g/` | `la_ap_diam-echojepa-l-k/` | `la_ap_diam-echoprime/` | `la_ap_diam-panecho/` |
| Ao Root Diam | `ao_root_diam-echojepa-g/` | `ao_root_diam-echojepa-l-k/` | `ao_root_diam-echoprime/` | `ao_root_diam-panecho/` |
| RV Basal Diam | `rv_basal_diam-echojepa-g/` | `rv_basal_diam-echojepa-l-k/` | `rv_basal_diam-echoprime/` | `rv_basal_diam-panecho/` |
| Disease VSD | `disease_vsd-echojepa-g/` | `disease_vsd-echojepa-l-k/` | `disease_vsd-echoprime/` | `disease_vsd-panecho/` |
| TR Vmax | `tr_vmax/echojepa-g/` | `tr_vmax/echojepa-l-k/` | `tr_vmax/echoprime/` | `tr_vmax/panecho/` |

### Related probes also under `chicago/probes/new/` (not built from this directory but commonly used alongside)

| Task | Path layout |
|---|---|
| AR / AS / MR / TR severity (color-trained) | `{ar,as,mr,tr}_severity_color/{echojepa-g,echojepa-l-k,echoprime,panecho}/best.pt` |
| Pericardial effusion | `pericardial_effusion_{echojepa-g,echojepa-l-k,echoprime,panecho}/best.pt` |
| RV function | `rv_function_{echojepa-g,echojepa-l-k,echoprime,panecho}/best.pt` |
| RVSP B-mode (multilevel) | `rv_sp_bmode_d4_multilevel/echojepa-g/best.pt` |
| RVSP B-mode (singlelayer) | `rv_sp_bmode_d4_singlelayer/echojepa-g/best.pt` |
| RVSP B-mode (multi) | `rv_sp_bmode_multi/echoprime/best.pt` |
| TAPSE B-mode (multi) | `tapse_bmode_multi/panecho/best.pt` |

---

# AV Status & MV Status — Prosthetic Valve Classification Datasets

Quick summary of how the AV Status and MV Status datasets were constructed for the prosthetic valve classification tasks.

## Task definitions

Both are 4-class study-level classification tasks asking what kind of intervention a valve has had, if any:

- **AV Status**: 0 = mechanical, 1 = surgical bioprosthetic, 2 = TAVR, 3 = native
- **MV Status**: 0 = mechanical, 1 = bioprosthetic, 2 = repair (annuloplasty / MitraClip), 3 = native

The build scripts live at `data/sample_data/new_tasks/build_av_status_dataset.py` (393 lines) and `data/sample_data/valve/build_mv_status_dataset.py` (449 lines). Both follow the same architecture.

## Pipeline overview

The end-to-end flow:

1. Pull class labels from UHN Syngo observations (2015–2019 era, ~390K studies)
2. Pull class labels from HeartLab findings (2002–2014 era, ~432K studies)
3. Map both to S3 study UIDs via `aws_syngo.csv` and `aws_heartlab_0806.csv`
4. Build a unified UID → class dictionary (Syngo wins on ties because the prosthetic observation schema is richer)
5. Filter clips to allowed views from the 18M-clip view-classifier output
6. Subsample the native class to a 50/50 healthy-vs-diseased mix capped at half the largest prosthetic class
7. Apply a patient-level 70/15/15 random split (numpy seed 42)
8. Write `train_vf.csv` / `val_vf.csv` / `test_vf.csv` + `viewfilter_meta.json`

## Syngo label sources (the 2015–2019 era)

For AV Status, the queries hit several structured observation fields:

- `AoV_Prosthetic_mfgr-ASE_obs` — manufacturer string. Mechanical values: St. Jude, Carbomedics, Bjork-Shiley, Starr-Edwards, On-X, Unkown_mech. Bioprosthetic: Hancock II, Magna Perimount, Carpentier, Freestyle, Mosaic, Toronto SPV, Edwards Magna, Trifecta, Perigon, Magna Ease, Unkown_bio. TAVR: Sapien (+ Sapien 3 / 3 Ultra), CoreValve, Evolut R, TAVI, Portico, Perceval.
- `AoV_Mechanical_type-ASE_obs` — backup mechanical signal (bileaflet, tilting_disk, SJM, etc.)
- `AoV_Bioprosthetictype-ASE_obs` — backup bio signal, with keyword check for TAVR (Sapien / CoreValve / Evolut / TAVI / TAVR) to route into the TAVR class
- `AoV_structure_uhn_obs` and `AoV_Normal_obs` — native signal (any value indicates a non-prosthetic AV)

MV Status uses the analogous fields: `MV_Prosthetic_type_sD_obs`, `Type_of_MV_Surgery_obs`, `MV_annular_repair_obs`, `MV_Structure_functionuhn_obs`, `MV_Normal_obs`. The "repair" class is the MV-specific replacement for TAVR — annuloplasty rings, MitraClip, surgical repair.

## HeartLab label sources (the 2002–2014 era)

HeartLab uses a coded finding system. The build script joins `heartlab_finding_intersects` → `heartlab_reports` → `heartlab_series` → `heartlab_studies` to associate FIN_ID codes with study UIDs:

- AV mechanical: finding group 75 (IDs 275, 276, 277, 278)
- AV surgical bioprosthetic: group 76 (IDs 279, 280, 281, 282, 283, 1426, 1427, 1522, 1523, 100270)
- AV TAVR: group 100179 (IDs 100762, 100763, 100764) + post-op TAVR finding 100766
- AV native (tricuspid/trileaflet/normal): IDs 100439, 243, 100460, 242, 310, 100445 from groups 68 and 83

MV findings follow the same pattern: mechanical (group 100), bioprosthetic (group 102), repair (annuloplasty ring 100267, MitraClip 100383), native (Group 85 IDs 316, 100467). Manufacturer-specific codes from group 100026 distinguish mechanical vs bioprosthetic when the primary finding is ambiguous.

## Native subsampling — the key design choice

The native class is the bulk of the patient population, so without subsampling it would dominate training and the model would learn "non-tricuspid morphology = prosthetic." The build scripts split native into **healthy** (normal/tricuspid structure) and **diseased** (bicuspid, rheumatic, calcified, or sclerotic native valves) using `AoV_structure_uhn_obs` and `AoV_structure_sD_obs` substring matching on `bicuspid` / `unicuspid` / `calcif` / `restricted` / `rheumatic`. Then the native class is rebuilt as a 50/50 mix of healthy and diseased, with the total capped at `largest_other_class // 2`. This keeps native in the same order of magnitude as the prosthetic classes while preserving structurally-abnormal-but-not-replaced cases — the most clinically interesting native subgroup.

MV Status uses an analogous design with a 50/50 MR-vs-no-MR mix for its native class.

## Patient split

After native subsampling, patients are shuffled with numpy seed 42 and split 70/15/15. All studies for a patient go into the same split — strict patient-level disjoint to prevent leakage.

## Final cohort sizes

**AV Status** (from the saved `viewfilter_meta.json`):

| Split | Mechanical | Surgical bio | TAVR | Native | Total |
|-------|-----------:|-------------:|-----:|-------:|------:|
| Train | 2,214 | 4,959 | 380 | 1,734 | 9,287 |
| Val   |   524 | 1,067 |  67 |   390 | 2,048 |
| Test  |   418 | 1,158 |  58 |   394 | 2,028 |

Clip counts (training): mechanical 58K / surgical bio 117K / TAVR 9.5K / native 42K, totaling ~227K training clips. View filter: PLAX + A4C + A3C, color allowed (prosthetic jets are diagnostically informative).

**MV Status:**

| Split | Mechanical | Bioprosthetic | Repair | Native | Total |
|-------|-----------:|--------------:|-------:|-------:|------:|
| Train | 2,121 | 1,072 | 2,393 | 2,331 | 7,917 |
| Val   |   414 |   254 |   392 |   513 | 1,573 |
| Test  |   412 |   220 |   547 |   488 | 1,667 |

Clip counts (training): mechanical 48K / bio 24K / repair 51K / native 45K, totaling ~167K training clips. View filter: PLAX + A4C, color allowed.

## Caveats worth knowing about

- **TAVR class is rare** — only 380 training studies (4% of AV Status training data). The model has limited supervision for distinguishing TAVR from surgical bioprosthetic, since both appear as round metal-stent profiles. AV Status macro-AUROC may mask this confusion; per-class TAVR performance is worth reporting separately.
- **Label provenance differs by era.** Syngo (2015–2019) uses structured manufacturer observations; HeartLab (2002–2014) uses numeric finding codes. The manufacturer lists and finding-ID lists were curated by inspecting top values in the source tables. New TAVR devices or manufacturer naming changes after 2019 are not covered.
- **Label priority is mechanical → bioprosthetic → TAVR → native.** A study with both a mechanical and a TAVR observation gets labeled mechanical (rare in practice).
- **Native subsampling is the largest design choice.** Without the 50/50 healthy/diseased rebalance, native would be ~60% of training. The current split reduces it to ~19% (AV) / ~29% (MV) while keeping structurally-abnormal native cases at meaningful proportion.

## Reproducibility paths

- AV build script: `data/sample_data/new_tasks/build_av_status_dataset.py`
- MV build script: `data/sample_data/valve/build_mv_status_dataset.py`
- AV probe CSVs: `experiments/nature_medicine/uhn/probe_csvs/av_status/`
- MV probe CSVs: `experiments/nature_medicine/uhn/probe_csvs/mv_status/`
- Source DB: `data_exploration/echo.db`
- UID mapping: `data/aws/aws_syngo.csv`, `data/aws/aws_heartlab_0806.csv`
- View classifier: `classifier/output/view_inference_18m/master_predictions.csv`
- Random seed: 42 (used for both native subsampling and patient split)

---

# Other Build Scripts in This Directory

The remaining scripts produce probe CSVs for individual UHN tasks and zero-shot MIMIC test sets. They share architectural conventions:

- **Path → study UID extraction**: UHN uses the regex `1.2.276.*.3.1.2.<uid>` on the S3 path; MIMIC uses `/s<study_id>/` from the `s12345678` directory naming.
- **View filtering**: all scripts intersect clips with the 18M-clip view classifier (`classifier/output/view_inference_18m/master_predictions.csv` for UHN, `classifier/output/mimic_view_predictions.csv` for MIMIC) and keep only the views appropriate to the measurement.
- **B-mode filtering**: when set, intersects with the color classifier and keeps only clips classified as B-mode (no color Doppler overlay).
- **Plausibility filtering**: every regression task applies a `VALID_RANGE` to drop outliers and unit-confused entries (e.g. mm vs cm).
- **±1 day matching (MIMIC only)**: structured measurements live on a separate timeline from echo studies; the build matches each measurement to the nearest study within ±1 day for the same `subject_id`.

## UHN regression tasks

### `build_mv_e_prime_medial_dataset.py` — MV E' medial (regression)

Predicts mitral annular tissue Doppler e' velocity (medial/septal site) from B-mode A4C video. Pulls per-study `MV E prime medial` from `syngo_measures`, averages duplicates, and applies a `[0.02, 0.25]` cm/s plausibility window. Views: A4C only. B-mode only. Patient-level 70/15/15 split (seed 42). Writes `train_vf.csv` / `val_vf.csv` / `test_vf.csv`, `viewfilter_meta.json`, and a `zscore_params.json` (training-set mean/std for normalization at probe-time).

This is the cross-modal counterpart to MV E/A — the tissue Doppler equivalent measured at the septal corner of the mitral annulus. Predicting it from B-mode tests whether the model has internalized myocardial relaxation dynamics from structural motion alone.

### `build_data_efficiency_subsets.py` — VSD and RV basal diameter at 50/25/12.5/6.25/3.125%

Builds nested, stratified subsamples of two UHN tasks for the data-efficiency curves:

- **VSD** (binary classification): stratified by class so positive/negative ratios are preserved at every fraction.
- **RV basal diameter** (regression): stratified by label quantile bins so the distribution shape is preserved.

Fractions form a log₂ halving series (50%, 25%, 12.5%, 6.25%, 3.125%). Each smaller subset is a strict subset of the next larger one — same studies, just fewer. Writes `train_vf_{50pct,25pct,12pct,6pct,3pct}.csv` to each task directory. Val and test sets are not subsampled.

## MIMIC zero-shot test sets

These scripts produce MIMIC-IV-Echo test CSVs for tasks where the probe was trained on UHN. They are not patient-split (no `train_vf.csv`); they are pure test sets used to measure cross-institution transfer. Most write to `…_all/all.csv` rather than `test.csv` to mark this. At inference, UHN-derived `zscore_params.json` is supplied, not MIMIC-derived (so the probe sees inputs on the same scale it was trained on).

### `build_mimic_biplane_lvef.py` — Simpson's biplane LVEF

Extracts the `biplane_lvef` measurement from `echo_structured_measurement` (TTE only) and matches to MIMIC echo studies via `subject_id` + ±1 day window. Filters to A2C and A4C views (the two views Simpson's biplane uses) and plausibility range `[10, 85]`%. Writes a split `(train.csv / val.csv / test.csv)` plus `all.csv`, `zscore_params.json`, and `task_meta.json`.

The script also computes overlap and label-difference statistics against the existing `lvef_structured` splits (which use the rounded visual-estimate `lvef` field). Biplane LVEF is the clinical reference standard; the visual-estimate field is rounded to 5% increments and is less precise — so this build script lets you swap the noisier label for the precise one.

### `build_mimic_dimensions.py` — LA AP / Ao Root / RV Basal diameter

Builds three B-mode-only regression test sets in one pass:

| Task | DB measurement | View | Range (cm) |
|---|---|---|---|
| LA AP diam | `la_dimen` | PLAX | [1.5, 7.0] |
| Ao Root diam | `ascending_diam` | PLAX | [1.5, 6.0] |
| RV Basal diam | `rv_diam` | A4C | [1.0, 7.0] |

All write to `…_all/all.csv` with the raw float label (the probe applies its UHN-trained z-score at runtime).

### `build_mimic_eprime_ea.py` — E' medial and MV E/A

Two related diastolic tasks. E' medial uses MIMIC's `sept_e_prime` measurement (septal = medial in echo terminology), A4C only, B-mode only, range `[0.02, 0.25]` cm/s. MV E/A uses `mv_peak_e_a`, views A4C+A2C, range `[0.3, 5.0]`. Because MV E/A was a color-trained probe at UHN, this script produces two MIMIC test sets — `all_color.csv` (B-mode + color Doppler clips, matching training conditions) and `all_bmode.csv` (B-mode only, the cross-modal stress test).

### `build_mimic_mortality.py` — 30d / 90d / 1yr mortality

Joins `echo_study_list` with `hosp_patients.dod` (date of death). For each study: label = 1 if patient died within `window_days` of the echo, label = 0 if alive at last observation (or if `dod` is null, treated as alive). Three tasks produced: `mortality_30d_v2`, `mortality_90d_v2`, `mortality_1yr_v2`. The `_v2` suffix marks these as rebuilt directly from `mimic.db` for verification; the script also cross-checks against the prebuilt CSVs in `data_exploration/mimic/csv/mortality_*.csv` and reports label agreement.

### `build_mimic_tr_vmax.py` — TR Vmax

Single-task script: extracts `tr_velocity` measurements, matches to studies within ±1 day, filters to A4C only. Color is allowed (TR Vmax probes at UHN were trained with color Doppler clips; the regurgitant jet is the diagnostic signal). Range `[0.5, 5.0]` m/s. Writes `tr_vmax_a4c/all.csv`. Note that this uses CW Doppler Vmax labels at MIMIC but the probe sees only 2D / color Doppler clips — predicting the velocity from the visual jet alone.

### `build_mimic_valve_status.py` — MV Status and AV Status (zero-shot)

The MIMIC counterpart to the UHN MV/AV Status tasks documented above. Uses `mv_leaflets` and `av_leaflets` fields from `echo_structured_measurement` and maps free-text values to the same 4-class schemas. MV: mechanical (Bileaflet, Mechanical, Ball and cage, …) / bioprosthetic (Bioprosthesis, Sapien 3 TMVR, …) / repair (Annular ring, MitraClip, PASCAL) / native (Normal, Myxomatous, Mild/Mod/Severe thick, …). AV: mechanical (Bileaflet mechanical, Single tilting disk) / surgical bioprosthetic (Bioprosthesis, AVR homograft, …) / TAVR (Sapien 3, CoreValve, Evolut, Lotus) / native (Nl, Bicuspid, Unicuspid, Quadricuspid, …). "Not well seen" values are excluded. Views match the UHN training views (MV: PLAX+A4C, AV: PLAX+A4C+A3C). Color allowed for both. Writes `…_all/all.csv` per task.

This is the most direct cross-institution test of the prosthetic valve classifier: same 4-class schema, same training-view filter, different institution and different label vocabulary (MIMIC uses freer text than UHN's controlled vocabulary).

## Reproducibility paths (other tasks)

| Task | Build script | Output dir |
|---|---|---|
| UHN MV E' medial | `build_mv_e_prime_medial_dataset.py` | `experiments/nature_medicine/uhn/probe_csvs/mv_e_prime_medial/` |
| UHN data efficiency | `build_data_efficiency_subsets.py` | `experiments/nature_medicine/uhn/probe_csvs/{vsd,rv_basal_diam}/train_vf_{50,25,12,6,3}pct.csv` |
| MIMIC biplane LVEF | `build_mimic_biplane_lvef.py` | `experiments/nature_medicine/mimic/probe_csvs/biplane_lvef_structured/` |
| MIMIC LA/Ao/RV diam | `build_mimic_dimensions.py` | `experiments/nature_medicine/mimic/probe_csvs/{la_ap_diam,ao_root_diam,rv_basal_diam}_all/` |
| MIMIC E' medial & E/A | `build_mimic_eprime_ea.py` | `experiments/nature_medicine/mimic/probe_csvs/{e_prime_medial,mv_ea_ratio}_all/` |
| MIMIC mortality | `build_mimic_mortality.py` | `experiments/nature_medicine/mimic/probe_csvs/mortality_{30d,90d,1yr}_v2/` |
| MIMIC TR Vmax | `build_mimic_tr_vmax.py` | `experiments/nature_medicine/mimic/probe_csvs/tr_vmax_a4c/` |
| MIMIC MV/AV Status | `build_mimic_valve_status.py` | `experiments/nature_medicine/mimic/probe_csvs/{mv_status,av_status}_all/` |

Common reference paths:
- MIMIC source DB: `uhn_echo/nature_medicine/data_exploration/mimic/mimic.db`
- MIMIC view manifest: `classifier/output/mimic_view_predictions.csv`
- MIMIC color manifest: `classifier/output/mimic_color_predictions.csv`
