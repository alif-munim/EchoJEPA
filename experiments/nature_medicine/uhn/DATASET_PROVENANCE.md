# UHN Dataset Provenance

Complete documentation of how each task's labels were constructed, filtered, and packaged
into probe CSVs for Nature Medicine evaluation. Intended to make every dataset fully
reproducible and verifiable from the source database.

Last updated: 2026-03-14

---

## Table of Contents

1. [Source Database](#1-source-database)
2. [Mapping Tables](#2-mapping-tables)
3. [Pipeline Overview](#3-pipeline-overview)
4. [Regression Measurements](#4-regression-measurements)
5. [Classification: Valve Severity](#5-classification-valve-severity)
6. [Classification: Structural Findings](#6-classification-structural-findings)
7. [Classification: Disease Detection](#7-classification-disease-detection)
8. [Trajectory Tasks](#8-trajectory-tasks)
9. [Patient Splits](#9-patient-splits)
10. [Post-Processing Pipeline](#10-post-processing-pipeline)
11. [TEE/Stress Filtering](#11-teestress-filtering)
12. [Physiological Range Filtering](#12-physiological-range-filtering)
13. [Reproduction Verification](#13-reproduction-verification)
14. [Verification Queries](#14-verification-queries)

---

## 1. Source Database

**Path:** `uhn_echo/nature_medicine/data_exploration/echo.db` (SQLite3, 5.7 GB)

Two reporting systems, non-overlapping in time:
- **Syngo Dynamics** (Siemens, ~2015+): `syngo_observations` (6.6M rows), `syngo_measures` (26.1M rows)
- **HeartLab** (~pre-2012): `heartlab_finding_intersects` (6.4M rows), `heartlab_measurement_intersects` (2.6M rows)

**Key tables:**

| Table | Rows | Purpose |
|-------|------|---------|
| `syngo_observations` | 6,577,032 | Categorical findings (valve severity, wall motion, etc.) |
| `syngo_measures` | 26,128,284 | Numeric measurements (TAPSE, LVEF, RVSP, etc.) |
| `syngo_study_details` | 389,666 | Study metadata (date, age, sex, description, modality) |
| `heartlab_finding_intersects` | 6,402,431 | Per-study finding selections |
| `heartlab_findings` | 6,688 | Finding template definitions (ID, SENTENCE, SHORT_DESCRIPTION) |
| `heartlab_finding_groups` | 1,678 | Finding group hierarchy (e.g., Group 93 = MV Regurgitation) |
| `heartlab_measurement_intersects` | 2,619,597 | Numeric measurements from HeartLab |
| `study_deid_map` | 237,022 | OriginalStudyID (DICOM UID) -> DeidentifiedStudyID |
| `patients` | 314,077 | ID, PATIENTS_NAME, SEX |

---

## 2. Mapping Tables

**Critical:** Syngo `StudyRef` is an integer (e.g., `1000712`). `study_deid_map.OriginalStudyID`
is a DICOM UID (e.g., `1.2.840.113543.6.6.2.0...`). These are **completely different namespaces
with 0% join overlap**. The mapping between them requires intermediate tables.

### Syngo: `aws_syngo.csv` (primary mapping)

**Path:** `data/aws/aws_syngo.csv` (320,854 rows)

Maps Syngo integer StudyRefs to deidentified DICOM UIDs for studies with videos on S3.

| Column | Example | Description |
|--------|---------|-------------|
| `STUDY_REF` | `1070167` | Syngo internal study ID (integer) |
| `PATIENT_ID` | `4174420` | UHN patient ID (numeric string) |
| `DeidentifiedStudyID` | `1.2.276.0.7230010...` | Deidentified DICOM UID (used in S3 paths, NPZ files) |
| `OriginalStudyID` | `1.2.124.113532...` | Original DICOM UID |
| `s3_key` | `echo-study/1.2.276...` | S3 path prefix |

**Coverage:** 320,854 of ~390,000 Syngo studies (~82%). Studies without S3 videos are excluded.

**Deidentification chain (Syngo):**
```
syngo_measures.StudyRef (integer, e.g. 1070167)
  == aws_syngo.csv.STUDY_REF
  -> aws_syngo.csv.DeidentifiedStudyID (DICOM UID, used in S3 paths and NPZ files)
  -> aws_syngo.csv.PATIENT_ID (numeric, used for patient splits)
```

### HeartLab: `study_deid_map` (in echo.db)

Maps original DICOM UIDs to deidentified UIDs for HeartLab-era studies.

| Column | Example | Description |
|--------|---------|-------------|
| `OriginalStudyID` | `1.2.840.113543...` | Original DICOM UID |
| `OriginalPatientID` | `3266450` | UHN patient ID (numeric string) |
| `DeidentifiedStudyID` | `1.2.276.0.7230010...` | Deidentified DICOM UID |
| `DeidentifiedPatientID` | `00877fe5-b4ab...` | UUID (not used; numeric patient IDs used instead) |

**Coverage:** 237,022 entries. Covers HeartLab studies and some Syngo studies.

### Combined mapping

For tasks that merge Syngo + HeartLab, both mappings are combined:
- `aws_syngo.csv` takes priority (320K entries)
- `study_deid_map` fills in HeartLab-only studies (237K entries)
- Combined: 342,763 unique DeidentifiedStudyID -> patient_id mappings

### Pre-computed source CSVs

These CSVs were pre-joined by an earlier pipeline and are the direct inputs for classification
label building. They already contain `DeidentifiedStudyID` and a standardized `common_label`.

- `data/aws/aws_syngo_findings_v2.csv` (542K rows, 53,944 unique studies)
  Syngo observations joined with DeidentifiedStudyID + condition_tag + common_label.
  Contains 18 observation Names (a filtered subset of 1,118 total in echo.db).

- `data/aws/aws_heartlab_findings_v2.csv` (1.26M rows)
  HeartLab findings joined with DeidentifiedStudyID + condition_tag + common_label.

- `data/aws/aws_syngo_msmt_v2.csv` (67K rows, 6,377 studies)
  Syngo measurements, small subset. NOT used for regression label building (too few studies).

- `data/aws/aws_heartlab_msmt_v2.csv` (255K rows, 32,723 studies)
  HeartLab measurements. NOT used for LVEF/TAPSE (HeartLab has no TAPSE, limited EF coverage).

---

## 3. Pipeline Overview

```
echo.db + aws_syngo.csv + source CSVs
  |
  v
build_labels.py --all
  |-- Regression: queries syngo_measures, joins via aws_syngo.csv
  |-- Classification: reads source CSVs (common_label), merges Syngo + HeartLab
  |-- Applies NPZ-level value filters (e.g., EF > 0, TAPSE in [0.3, 5.0])
  |-- Assigns patient splits from patient_split.json
  |-- Writes: labels_v2/{task}.npz
  |     Keys: study_ids, patient_ids, labels, splits
  |
  v
build_probe_csvs.py --all
  |-- Loads labels_v2/{task}.npz
  |-- Loads uhn_all_clips.csv (18M S3 paths) -> study_to_clips_index.pkl
  |-- Joins: study_id -> all S3 clip paths for that study
  |-- Applies physiological range filtering (regression only, tighter than NPZ filter)
  |-- Computes zscore_params.json from train split (regression only)
  |-- Writes: probe_csvs/{task}/train.csv, val.csv, test.csv
  v
build_viewfiltered_csvs.py --all
  |-- Loads probe_csvs/{task}/train.csv (etc.)
  |-- Joins with view_inference_18m/master_predictions.csv (view per clip)
  |-- Optionally joins with color_inference_18m/master_predictions.csv (B-mode filter)
  |-- Writes: probe_csvs/{task}/train_vf.csv, val_vf.csv, test_vf.csv
  |-- Writes: probe_csvs/{task}/viewfilter_meta.json
  v
[TEE/stress filtering, 2026-03-14]
  |-- Loads syngo_study_details.STUDY_DESCRIPTION
  |-- Maps STUDY_REF -> DeidentifiedStudyID via uhn_uid_to_studyref.csv
  |-- Removes TEE, stress echo, IVUS, ICE studies from ALL CSVs
  v
Final CSVs ready for probe training
```

**Reproduction script:** `build_labels.py` (added 2026-03-14) replaces the deleted interactive
notebook. It produces NPZ files in `labels_v2/` that are validated against the original `labels/`
NPZ files. See [Section 13](#13-reproduction-verification) for validation results.

**Two-stage filtering:** Regression tasks have TWO levels of value filtering:
1. **NPZ-level** (in `build_labels.py`): removes measurement artifacts and unit errors
   (e.g., LVEF <= 0, TAPSE outside [0.3, 5.0] cm)
2. **CSV-level** (in `build_probe_csvs.py`): applies tighter physiological ranges from
   `physiological_ranges.json` (e.g., LVEF [5, 90], TAPSE [0.5, 3.5])

---

## 4. Regression Measurements

Regression tasks extract numeric values from `syngo_measures`, joined to deidentified
study IDs via `aws_syngo.csv`.

### Actual Join Pattern (verified)

```python
# 1. Query syngo_measures from echo.db
rows = cursor.execute("""
    SELECT StudyRef, CAST(Value AS REAL) as val
    FROM syngo_measures
    WHERE MeasurementName = '{measurement_name}'
      AND Value IS NOT NULL
""")

# 2. Map StudyRef -> DeidentifiedStudyID via aws_syngo.csv
#    (NOT via study_deid_map — those are DICOM UIDs, 0% join with integer StudyRefs)
ref_to_deid = {row.STUDY_REF: row.DeidentifiedStudyID for row in aws_syngo_csv}
ref_to_patient = {row.STUDY_REF: row.PATIENT_ID for row in aws_syngo_csv}

# 3. Average multiple measurements per study
# 4. Apply NPZ-level filter (task-specific)
# 5. Filter to patients in patient_split.json
```

### Verified Tasks (all 21 regression tasks)

All 19 single-measurement tasks have been verified against NPZ labels by querying
`syngo_measures` and joining through `aws_syngo.csv`. 18 are **perfect matches** (100%
value match); `ivsd` matches 62,399/62,403 values (4 differ by < 0.001).

| Task | MeasurementName | NPZ Studies | NPZ Filter | Verification |
|------|----------------|-------------|------------|-------------|
| lvef | `LV EF, MOD BP` | 51,341 | EF > 0 | 100% exact |
| tapse | `TAPSE (M-mode)` | 54,793 | [0.3, 5.0] cm | 100% exact |
| rv_sp | `RV S'` | 54,090 | -- | 100% exact |
| rvsp | `RVSP (TR)` | 33,556 | -- | 100% exact |
| rv_fac | `RV FAC A4C` | 15,227 | -- | 100% exact |
| edv | `LV vol d, MOD BP` | 51,141 | -- | 100% exact |
| esv | `LV vol s, MOD BP` | 51,107 | -- | 100% exact |
| lv_mass | `LV Mass 2D uhn` | 53,849 | -- | 100% exact |
| ivsd | `IVS d, 2D` | 62,403 | -- | 99.99% (4 float precision) |
| ao_root | `Ao Root d, 2D` | 57,455 | -- | 100% exact |
| la_vol | `LA Vol A/L A4C` | 39,953 | -- | 100% exact |
| aov_vmax | `AoV Vmax` | 55,095 | -- | 100% exact |
| aov_mean_grad | `AoV mean grad` | 44,403 | -- | 100% exact |
| aov_area | `AoV area (VTI)` | 23,652 | -- | 100% exact |
| lvot_vti | `LVOT VTI` | 43,539 | -- | 100% exact |
| mv_ea | `MV E/A ratio` | 47,316 | -- | 100% exact |
| mv_ee | `MV E/e'` | 50,411 | -- | 100% exact |
| mv_dt | `MV DT` | 42,671 | -- | 100% exact |
| gls | `Global Peak Long Strain` | 23,574 | -- | 100% exact |
| mv_ee_medial | `Mitral annulus medial E/e'` (best match) | 37,778 | -- | 80% overlap, 62% exact |
| cardiac_output | Unknown (calculated) | 16,137 | -- | Source unknown |

**LVEF:** Uses ONLY `LV EF, MOD BP` (biplane modified Simpson's method). This is the
gold-standard LVEF measurement. Other methods in the database (`LV EF, MOD A4C` at 97K,
`LV EF, Cube, 2D` at 114K, `EF Manual` at 22K) are NOT used. Verified: all 51,341 NPZ
values exactly match the MOD BP query.

**TAPSE:** Uses ONLY `TAPSE (M-mode)` (104K rows). The 630 filtered studies include values
of 0.0 (failed measurement) and values > 5 cm (likely stored in mm instead of cm, e.g., 202).

**mv_ee_medial:** Best single measurement is "Mitral annulus medial E/e'" (53,786 raw
studies, 30,161 mapped via aws_syngo.csv). Covers 30,055/37,778 NPZ studies (80%). The
remaining ~7,700 studies likely come from HeartLab data or alternate measurement names.
The 62% exact match rate suggests different averaging of multiple measurements per study.

**cardiac_output:** NPZ has 16,137 studies with mean=4.90 L/min, range [1.03, 14.95].
All IDs map through `aws_syngo.csv` (100% Syngo). No direct CO measurement matches:
`LV CO, MOD A4C` covers only 7K mapped with <1% exact match. Not SV×HR (tested, <1%
match). The original calculation was in a deleted notebook and cannot be reconstructed.

**Notes:**
- LVOT VTI is stored in meters in the database (not cm). Mean ~0.197 m = 19.7 cm.
- All 19 single-measurement tasks use exactly ONE `MeasurementName` each (no merging
  of alternate names). The correct name is listed in the table above.
- `MV E/A ratio`, `MV E/e'`, and `MV DT` are stored as named measurements in
  syngo_measures, not calculated from component values.
- HeartLab measurements (`aws_heartlab_msmt_v2.csv`) were NOT used for any of the
  verified regression tasks. All map exclusively through `aws_syngo.csv`.

### NPZ File Structure (regression)

```python
npz = np.load("labels/{task}.npz", allow_pickle=True)
npz["study_ids"]   # (N,) string array of DeidentifiedStudyIDs
npz["patient_ids"]  # (N,) string array of numeric UHN patient IDs (NOT DeidentifiedPatientIDs)
npz["labels"]       # (N,) float32, raw measurement values in original units
npz["splits"]       # (N,) string array: "train" / "val" / "test"
```

---

## 5. Classification: Valve Severity

Valve severity labels are built from **pre-joined source CSVs**, not direct echo.db queries.
The source CSVs (`aws_syngo_findings_v2.csv`, `aws_heartlab_findings_v2.csv`) already contain
`DeidentifiedStudyID` and a standardized `common_label` column that maps raw observation
values to task-specific class names (e.g., `MR_trace`, `AS_mild`).

### Actual Build Pattern (verified)

```python
CLASS_MAP = {"MR_none": 0, "MR_trace": 1, "MR_mild": 2, "MR_moderate": 3, "MR_severe": 4}

# 1. Load source CSVs, filter to rows with matching common_label
syngo_df = aws_syngo_findings_v2[common_label in CLASS_MAP]  # e.g., 42K MR studies
hl_df = aws_heartlab_findings_v2[common_label in CLASS_MAP]  # e.g., 91K MR studies

# 2. Map common_label -> ordinal class
# 3. Dedup per source: highest severity wins per DeidentifiedStudyID
# 4. Merge: union of both sources, highest severity wins for overlapping studies
# 5. Map DeidentifiedStudyID -> patient_id (aws_syngo.csv for Syngo, study_deid_map for HeartLab)
# 6. Filter to patients in patient_split.json
```

**Why source CSVs, not echo.db?** The `common_label` column in the source CSVs is the
authoritative class mapping. It was computed by an earlier pipeline that:
- Collapsed intermediate grades (e.g., Syngo `mild_to_moderate` -> `MR_mild`)
- Handled case inconsistencies (e.g., Syngo `None` vs `none` -> `MR_none`)
- Mapped HeartLab SENTENCE patterns to standardized labels
- Pre-joined to DeidentifiedStudyID through the correct mapping chain

Direct echo.db queries return ALL Syngo observations (96K for MR, 79K for AS), but only
~42K / ~21K of those are in the source CSV (the subset with S3 video mappings). The raw
echo.db counts do not match the NPZ counts and should not be used for verification.

### MR Severity (5 classes)

**Sources:**
- Syngo: `common_label` in {MR_none, MR_trace, MR_mild, MR_moderate, MR_severe} (42,428 unique studies)
  Raw observation: `MV_Regurgitation_obs`. Intermediate grades already collapsed in common_label.
- HeartLab: same common_labels (90,884 unique studies). Finding Group 93.
  HeartLab has no "trace" grade; goes directly from none to mild.

| Class | Label | Studies | Syngo common_label | HeartLab common_label |
|-------|-------|---------|-------------------|-----------------------|
| 0 | none | 8,291 | MR_none (149) | MR_none (8,149) |
| 1 | trace | 26,633 | MR_trace (26,633) | *(none)* |
| 2 | mild | 81,597 | MR_mild (12,981) | MR_mild (68,633) |
| 3 | moderate | 12,173 | MR_moderate (1,903) | MR_moderate (10,275) |
| 4 | severe | 4,604 | MR_severe (762) | MR_severe (3,842) |

**Total:** 133,298 studies (133,303 in original NPZ; 5 studies have patient IDs not
recoverable from the combined mapping).

**Syngo raw Value distribution** (from echo.db, for reference only; NOT used directly):
```
trace               57,004   -> MR_trace (only 26,633 have S3 video mappings)
mild                22,913   -> MR_mild
mild_to_moderate     5,653   -> MR_mild (collapsed in common_label)
None                 3,694   -> MR_none (case inconsistency with "none")
moderate             2,931   -> MR_moderate
severe               2,057   -> MR_severe
moderate_to_severe   1,507   -> MR_moderate (collapsed in common_label)
none                   536   -> MR_none
```

**The "trace discrepancy" explained:** An earlier analysis found 57K Syngo "trace" rows but
only 26.6K in NPZ class 1. This is NOT a class mapping error. The 57K is from ALL Syngo
studies in echo.db; only 42K total MR studies have S3 video mappings (in the source CSV),
of which 26,633 are MR_trace.

### TR Severity (5 classes, 138,966 studies)

**Syngo:** `TR_degree_obs` via common_label TR_* (source CSV)
**HeartLab:** Finding Group 116

| Class | Label | Studies | Syngo Values | HeartLab SENTENCE patterns |
|-------|-------|---------|-------------|--------------------------|
| 0 | none | 5,715 | `none` | "no evidence of tricuspid regurgitation" |
| 1 | trivial/trace | 24,512 | `trivial`, `trace` | "trace tricuspid regurgitation" |
| 2 | mild | 91,685 | `mild`, `mild-moderate` | "mild(1+)", "mild to moderate(1-2+)" |
| 3 | moderate | 13,254 | `moderate`, `moderate-severe` | "moderate(2+)", "moderate to severe(3+)" |
| 4 | severe | 3,800 | `severe` | "severe(4+)" |

**Known issue:** TR uses hyphens (`mild-moderate`) while MR/AR/PR use underscores (`mild_to_moderate`).
Syngo "trivial" (51K in echo.db) and "trace" (81) both map to class 1.

### AR Severity (5 classes, 80,175 studies)

**Syngo:** `AoV_Regurgitation_obs` via common_label AR_*
**HeartLab:** Finding Group 100018

| Class | Label | Studies |
|-------|-------|---------|
| 0 | none | 26,532 |
| 1 | trace | 11,525 |
| 2 | mild | 40,912 |
| 3 | moderate | 908 |
| 4 | severe | 298 |

**Warning:** Classes 3-4 are very small (severe AR is rare). Weight ratio will be extreme.

### AS Severity (4 classes)

**Sources:**
- Syngo: `common_label` in {AS_none, AS_mild, AS_moderate, AS_severe} (21,097 unique studies)
  Raw observation: `AoV_sten_degree_SD_obs` (in source CSV). Note: echo.db also has
  `AoV_Sclerosis_Stenosis_obs` (78,767 rows) which is a DIFFERENT, broader observation
  that includes sclerosis grades. The source CSV uses only `AoV_sten_degree_SD_obs`.
- HeartLab: same common_labels (154,844 unique studies). HeartLab contributes the majority.

| Class | Label | Studies | Syngo common_label | HeartLab common_label |
|-------|-------|---------|-------------------|-----------------------|
| 0 | none/sclerosis | 145,694 | AS_none (3,822) | AS_none (141,884) |
| 1 | mild | 20,345 | AS_mild (14,637) | AS_mild (5,712) |
| 2 | moderate | 4,607 | AS_moderate (1,707) | AS_moderate (2,902) |
| 3 | severe | 5,282 | AS_severe (934) | AS_severe (4,349) |

**Total:** 175,928 studies (175,932 in original NPZ; 4 studies not recoverable).

**Design decision:** Sclerosis (non-obstructive calcification) merged with "none" in class 0.
This is clinically appropriate; sclerosis alone doesn't cause hemodynamic obstruction.

### PR Severity (5 classes, 77,307 studies)

**Syngo:** `PV_Regurgitation_obs` via common_label PR_*
**HeartLab:** Finding Group 131

| Class | Label | Studies |
|-------|-------|---------|
| 0 | none | 22,863 |
| 1 | trace | 24,940 |
| 2 | mild | 26,213 |
| 3 | moderate | 1,375 |
| 4 | severe | 1,916 |

---

## 6. Classification: Structural Findings

These tasks extract ordinal grades from structured echo reports. Same source CSV pattern
as valve severity: `common_label` column in the source CSVs provides the class mapping.

### Source-CSV Tasks (10 tasks, verified by count-matching)

These are in `aws_syngo_findings_v2.csv` and `aws_heartlab_findings_v2.csv` with
standardized `common_label` values. Verified by matching class counts against NPZ labels.

| Task | NPZ Studies | Classes | Syngo Observation | common_label Prefix |
|------|-------------|---------|-------------------|-------------------|
| lv_systolic_function | 125,538 | 6 | `LV_Fx_qualitative_obs` | LVSF_ |
| lv_cavity_size | 236,882 | 5 | `LV_Cavity_Size-ASE_obs` | LVC_ |
| lv_hypertrophy | 41,872 | 5 | `Hypertrophy_degree-ASE_obs` | LVH_ |
| la_size | 139,049 | 4 | `LA_size-ASE_obs` | LA_ |
| ra_size | 64,813 | 4 | `RA_size-ASE_obs` | RA_ |
| rv_size | 87,912 | 4 | `RV_size-ASE_obs` | RV_ |
| pa_pressure | 39,171 | 4 | `PA_pressure_obs` | PAP_ |
| pericardial_effusion | 176,933 | 5 | `PE_Effusion_sD_obs` | PEF_ |
| diastolic_function | 42,115 | 4 | `LV_diast_filling-ASE_obs` | DFP_ |

**Class ordering warning:** `lv_systolic_function` and `lv_cavity_size` use **severity-first**
ordering (class 0 = most severe), which is reversed from clinical convention. All other
tasks use the standard normal=0 ordering. See `CLASS_MAPS.md` for full class-by-class tables.

### Non-Source-CSV Tasks (3 tasks, built from echo.db)

These observations are NOT in the pre-joined source CSVs and were built directly from
echo.db queries. The `aws_syngo_findings_v2.csv` contains only 18 of 1,118 Syngo
observation names.

| Task | Studies | Classes | Source | Verification |
|------|---------|---------|--------|-------------|
| cardiac_rhythm | 44,672 | 6 | `Cardiac_Rhythm_obs` (Syngo only) | 99.9% class match |
| rv_function | 125,229 | 5 | `RV_syst._fx-ASE_obs` (Syngo) + HeartLab RVF_ | 80% class match |
| rwma | 69,452 | 2 | Multi-source (WMA observations + text) | Not verified |

**Cardiac rhythm:** Verified at 99.9% accuracy (44,110/44,166 overlapping studies).
Class map: 0=NSR, 1=A_fib, 2=sinus_tachy, 3=sinus_brady, 4=artificially_paced, 5=irregular.
Syngo-only; HeartLab has no rhythm data in the source CSV.

**RV function:** Partially verified at 80% (80,416/100,989). Class map: 0=normal
(+grossly_normal), 1=low_normal, 2=mildly_reduced/mild, 3=moderately_reduced/moderate,
4=severely_reduced/severe. 24,179 NPZ studies not found in either Syngo or HeartLab
reconstruction source; these may come from alternate observation names or merge logic.

**RWMA:** Binary (no RWMA=0, present=1). Built from multiple sources likely including
`LV_fx_regional_wma_obs` (10K), `LV_WMA_Comments_obs` (18K), HeartLab "Regional Wall
Motion" findings. Original construction logic is undocumented.

---

## 7. Classification: Disease Detection

Binary (0/1) tasks built from structured echo report diagnoses. Sources are condition-specific
combinations of Syngo observations, HeartLab findings, and sometimes ICD codes.

### Disease Cohort Construction (verified 2026-03-14)

| Task | Pos | Neg | Total | Syngo Obs Source | Syngo Coverage |
|------|-----|-----|-------|-----------------|----------------|
| disease_hcm | 12,655 | 60,807 | 73,462 | HCM-related free-text + HOCM obs | -- |
| disease_amyloidosis | 1,473 | 12,426 | 13,899 | Free-text in comments | -- |
| disease_dcm | 1,716 | 20,173 | 21,889 | Free-text/diagnosis | -- |
| disease_endocarditis | 11,286 | 19,187 | 30,473 | `UHN_Endocarditis_obs` (1.3%) | 1.3% |
| disease_stemi | 8,815 | 2,437 | 11,252 | Free-text/RWMA pattern | -- |
| disease_takotsubo | 399 | 8,808 | 9,207 | Free-text ("takotsubo"/"apical ballooning") | -- |
| disease_bicuspid_av | 7,382 | 117,110 | 124,492 | `AoV_structure_uhn/sD_obs` "bicuspid" | 18% |
| disease_myxomatous_mv | 1,931 | 7,935 | 9,866 | `MV_Structure_function*_obs` "myxomatous" | 19% |
| disease_rheumatic_mv | 2,131 | 2,096 | 4,227 | `MV_Structure_function*_obs` "rheumatic" | 12.5% |

**Key finding from verification:** Syngo structured observations account for only 1-19%
of positive cases. The majority come from HeartLab data. The `hcm_hard_control_cohort.csv`
in `data_exploration/` documents that HCM positives were identified from free-text search
for "hypertrophic cardiomyopathy/HOCM/HCM/ASH/SAM+" across both systems (2,757 Syngo
patients + 12,431 HeartLab patients).

**Cohort construction pattern (inferred):**
- Positive cohorts were NOT identified via `condition_tag` values (condition_tags show
  no discriminative power between pos/neg, verified by F1 analysis)
- Instead, positives were identified by **free-text keyword search** across Syngo
  observation Values (especially Comment_* fields), HeartLab SENTENCE fields, and
  structured morphology observations (bicuspid, myxomatous, rheumatic)
- Negative controls were matched but matching criteria are undocumented
- `disease_rheumatic_mv` has near-equal pos/neg (2,131 vs 2,096), suggesting 1:1 matching
- `disease_stemi` has more positives than negatives (8,815 vs 2,437), unusual for case-control
- The exact cohort construction code was in a deleted notebook

**WARNING:** Raw disease cohort counts may include indication/rule-out labels:
- HCM: ~5-8K true positives (not all 12K may be confirmed HCM)
- Endocarditis: `UHN_Endocarditis_obs` has only 264 studies with value "Endocarditis_possible"
  but NPZ has 11,286 positives, most from HeartLab
- See `uhn_echo/nature_medicine/data_exploration/reference/rare-diseases.md` for detailed
  cohort counts and query templates

---

## 8. Trajectory Tasks

Trajectory labels predict the **change (delta)** in a measurement between two echos
for the same patient, separated by 30-365 days.

### NPZ File Structure (trajectory)

```python
npz = np.load("labels/trajectory/trajectory_{task}.npz", allow_pickle=True)
npz["study_id_1"]   # (N,) baseline study DeidentifiedStudyID
npz["study_id_2"]   # (N,) follow-up study DeidentifiedStudyID
npz["patient_ids"]  # (N,) numeric UHN patient ID
npz["label_1"]      # (N,) float32, baseline measurement value
npz["label_2"]      # (N,) float32, follow-up measurement value
npz["delta"]         # (N,) float32, label_2 - label_1
npz["days_between"]  # (N,) int32, days between studies (30-365)
npz["splits"]        # (N,) string: "train" / "val" / "test"
```

### Construction Logic

```python
# Conceptual: find pairs of studies for the same patient with the same measurement
# 1. Query syngo_measures for the measurement
# 2. Join to aws_syngo.csv for DeidentifiedStudyID + PATIENT_ID
# 3. Join to syngo_study_details for STUDY_DATE (via STUDY_REF)
# 4. For each patient with 2+ studies, form (baseline, follow-up) pairs
#    where days_between is in [30, 365]
# 5. Compute delta = label_2 - label_1
```

**Note:** The join for dates uses `syngo_study_details.STUDY_REF` = `syngo_measures.StudyRef`
(both are integers), then `aws_syngo.csv` for the DeidentifiedStudyID mapping.

### Task Details

| Task | Base Measurement | Pairs | Delta Mean | Delta Std | Unit |
|------|-----------------|-------|-----------|----------|------|
| trajectory_lvef | LVEF (%) | 14,235 | -0.805 | 9.626 | % change |
| trajectory_tapse | TAPSE (cm) | 15,298 | -0.060 | 0.531 | cm change |
| trajectory_lv_mass | LV mass (g) | 20,094 | -5.119 | 52.059 | g change |
| trajectory_rv_sp | RV S' (m/s) | 14,551 | -0.004 | 0.034 | m/s change |
| trajectory_mr_severity | MR grade (0-4) | 41,154 | -0.037 | 0.708 | grade change |

**Note:** `trajectory_mr_severity` treats ordinal grades as continuous for delta computation.

---

## 9. Patient Splits

**File:** `experiments/nature_medicine/uhn/patient_split.json`

All tasks use the same patient-level train/val/test split:
- **138,779 unique patients**
- Split is by patient, not by study (no patient appears in multiple splits)
- Split distribution:

| Split | Patients | % |
|-------|---------|---|
| train | 113,151 | 81.5% |
| val | 9,752 | 7.0% |
| test | 15,876 | 11.4% |

The split assignment is stored in each NPZ file's `splits` array. Patient-to-split
mapping was computed once and reused across all tasks. Study-level split ratios vary
by task because different patients have different numbers of studies.

---

## 10. Post-Processing Pipeline

### Step 1: NPZ -> Probe CSVs

**Script:** `experiments/nature_medicine/uhn/build_probe_csvs.py`

For each task:
1. Load `labels/{task}.npz`
2. Load `study_to_clips_index.pkl` (cached mapping: study_id -> list of S3 video paths)
   - Built from `uhn_all_clips.csv` (18M lines, all echo clips on S3)
3. For each study: write ALL clips with the study's label (one line per clip)
4. **Physiological range filtering** (regression only): drop studies where the label
   falls outside plausible clinical range (see `physiological_ranges.json`)
5. **Z-score params**: compute mean/std from train split, save as `zscore_params.json`
   - Labels in CSV remain in raw units; Z-score normalization happens at runtime in eval.py

### Step 2: View Filtering

**Script:** `experiments/nature_medicine/uhn/build_viewfiltered_csvs.py`

For each task with defined view filters (41 of 52 tasks):
1. Load all URIs from train/val/test CSVs
2. Look up predicted view class per clip from `classifier/output/view_inference_18m/master_predictions.csv`
3. Keep only clips matching allowed views (e.g., A4C, A2C, PLAX for MR severity)
4. If `bmode_only=True`: additionally filter by `color_inference_18m/master_predictions.csv`,
   keeping only clips where color prediction is "No" (grayscale B-mode)
5. Write `train_vf.csv`, `val_vf.csv`, `test_vf.csv` and `viewfilter_meta.json`

**6 tasks with no view filter** (all views, all modalities):
`cardiac_rhythm`, `disease_dcm`, `disease_endocarditis`, `disease_stemi`, `disease_takotsubo`, `gls`

**8 tasks with B-mode only filter:**
`aov_area`, `aov_mean_grad`, `aov_vmax`, `ar_severity`, `as_severity`, `mr_severity`, `pr_severity`, `tr_severity`

---

## 11. TEE/Stress Filtering

Applied 2026-03-14 to ALL 52 UHN task CSV directories (both full and view-filtered).

**Filtering chain:**
```
syngo_study_details.STUDY_DESCRIPTION
  -> classify as TTE / TEE / Stress / IVUS / ICE
  -> syngo_study_details.STUDY_REF
  -> experiments/nature_medicine/uhn/mapping/uhn_uid_to_studyref.csv
  -> DeidentifiedStudyID
  -> remove matching clips from all CSVs
```

**Study type classification rules** (from STUDY_DESCRIPTION):
- **TEE:** contains "transesophageal", "TEE", "trans-esophageal" (case-insensitive)
- **Stress:** contains "stress", "dobutamine", "exercise" (case-insensitive)
- **IVUS:** contains "intravascular", "IVUS"
- **ICE:** contains "intracardiac", "ICE"
- **Everything else:** TTE (kept)

**Results:**
- 18,796 TEE studies + 9,355 stress studies identified in syngo_study_details
- 7,341 of these had deidentified UIDs with clips in the dataset
- ~3.2M clips removed across all 52 tasks
- Contamination was 0-3.2% per task (highest: RWMA 3.2%, rheumatic MV 3.2%, endocarditis 3.0%)

**Mapping file:** `experiments/nature_medicine/uhn/mapping/uhn_uid_to_studyref.csv`
- 224,394 entries
- Columns: `deidentified_study_id, study_ref, patient_id, study_date, match_type`

---

## 12. Physiological Range Filtering

**File:** `experiments/nature_medicine/physiological_ranges.json`

Applied during CSV generation (by `build_probe_csvs.py`) to remove measurement outliers.
This is the SECOND level of filtering, applied after the NPZ-level filters in `build_labels.py`.

| Task | Min | Max | Unit | Dropped |
|------|-----|-----|------|---------|
| tapse | 0.5 | 3.5 | cm | 165 |
| rv_sp | 0.02 | 0.30 | m/s | -- |
| rvsp | 5.0 | 130.0 | mmHg | -- |
| rv_fac | 3.0 | 75.0 | % | -- |
| lvef | 5.0 | 90.0 | % | -- |
| edv | 10.0 | 600.0 | mL | -- |
| esv | 3.0 | 500.0 | mL | -- |
| gls | -35.0 | 0.0 | % | -- |
| ivsd | 0.3 | 4.0 | cm | -- |
| ao_root | 1.0 | 6.0 | cm | -- |
| aov_vmax | 0.3 | 7.0 | m/s | -- |
| aov_mean_grad | 0.0 | 100.0 | mmHg | -- |
| aov_area | 0.1 | 6.0 | cm2 | -- |
| la_vol | 5.0 | 350.0 | mL | -- |
| lv_mass | 20.0 | 900.0 | g | -- |
| lvot_vti | 0.02 | 0.50 | m | -- |
| mv_dt | 40.0 | 600.0 | ms | -- |
| mv_ea | 0.1 | 8.0 | ratio | -- |
| mv_ee | 1.0 | 40.0 | ratio | -- |
| mv_ee_medial | 1.0 | 40.0 | ratio | -- |
| cardiac_output | 0.5 | 20.0 | L/min | -- |

Classification tasks are not filtered by physiological range.

---

## 13. Reproduction Verification

`build_labels.py` (added 2026-03-14) reproduces the NPZ label files from source data.
Validated against the original NPZ files in `labels/`.

### Results: Regression (21 tasks)

| Category | Tasks | Verification |
|----------|-------|-------------|
| Perfect reproduction | lvef, tapse (via build_labels.py) | 100% study ID + value + split match |
| Perfect MeasurementName match | rv_sp, rvsp, rv_fac, edv, esv, lv_mass, ao_root, la_vol, aov_vmax, aov_mean_grad, aov_area, lvot_vti, mv_ea, mv_ee, mv_dt, gls | 100% value match against NPZ |
| Near-perfect | ivsd | 62,399/62,403 exact (4 differ by <0.001) |
| Partially verified | mv_ee_medial | 80% study overlap, 62% exact value match |
| Source unknown | cardiac_output | No single measurement or calculation matches |

### Results: Classification — Source-CSV Tasks (14 tasks)

All 14 tasks confirmed by systematic NPZ→source CSV cross-reference: for every integer
class, the dominant `common_label` matches the expected clinical label (100% of classes).

| Task | NPZ Studies | Reproduced | Status |
|------|------------|-----------|--------|
| mr_severity | 133,303 | 133,298 | 99.996% (5 unmappable studies) |
| as_severity | 175,932 | 175,928 | 99.998% (4 unmappable studies) |
| tr_severity | 138,966 | -- | Dominant common_label confirmed for all 5 classes |
| ar_severity | 80,175 | -- | Dominant common_label confirmed for all 5 classes |
| pr_severity | 77,307 | -- | Dominant common_label confirmed for all 5 classes |
| lv_systolic_function | 125,538 | -- | Confirmed; **severity-first ordering** (class 0=severe) |
| lv_cavity_size | 236,882 | -- | Confirmed; **severity-first ordering** (class 0=severe) |
| lv_hypertrophy | 41,872 | -- | Dominant common_label confirmed for all 5 classes |
| la_size | 139,049 | -- | Dominant common_label confirmed for all 4 classes |
| ra_size | 64,813 | -- | Dominant common_label confirmed for all 4 classes |
| rv_size | 87,912 | -- | Dominant common_label confirmed for all 4 classes |
| pa_pressure | 39,171 | -- | Dominant common_label confirmed for all 4 classes |
| pericardial_effusion | 176,933 | -- | Dominant common_label confirmed for all 5 classes |
| diastolic_function | 42,115 | -- | Dominant common_label confirmed for all 4 classes |

### Results: Classification — Echo.db Tasks (3 tasks)

| Task | NPZ Studies | Verification |
|------|------------|-------------|
| cardiac_rhythm | 44,672 | 99.9% class match (44,110/44,166) from `Cardiac_Rhythm_obs` |
| rv_function | 125,229 | 80% class match (80,416/100,989) from Syngo+HeartLab |
| rwma | 69,452 | Counts confirmed; source logic undocumented |

### Results: Disease Detection (9 tasks)

| Task | NPZ Studies | Verification |
|------|------------|-------------|
| disease_amyloidosis | 13,899 | Counts confirmed; cohort logic undocumented |
| disease_bicuspid_av | 124,492 | 18% of positives traceable to Syngo obs |
| disease_dcm | 21,889 | Counts confirmed |
| disease_endocarditis | 30,473 | 1.3% of positives traceable to Syngo obs |
| disease_hcm | 73,462 | Counts confirmed; see hcm_hard_control_cohort.csv |
| disease_myxomatous_mv | 9,866 | 19% of positives traceable to Syngo obs |
| disease_rheumatic_mv | 4,227 | 12.5% of positives traceable to Syngo obs |
| disease_stemi | 11,252 | Counts confirmed |
| disease_takotsubo | 9,207 | Counts confirmed |

### What was wrong in the earlier (pre-reproduction) analysis

An earlier verification session (also 2026-03-14) queried echo.db directly and compared
raw Syngo counts to NPZ counts. That analysis produced confusing results because it
didn't account for the mapping bottleneck:

| Earlier finding | Actual explanation |
|----------------|-------------------|
| TAPSE: NPZ (55K) < Syngo (100K) | Only 56% of Syngo StudyRefs have S3 video mappings in `aws_syngo.csv` |
| LVEF: NPZ (51K) < Syngo (136K) | NPZ uses ONLY `LV EF, MOD BP` (93K rows -> 51K mapped). The 136K included all EF methods. |
| MR trace: Syngo (57K) vs NPZ (27K) | The 57K is ALL Syngo studies; only 42K have S3 mappings, of which 26.6K are trace |
| AS: NPZ (176K) >> Syngo (79K) | HeartLab contributes 155K studies. The 79K is `AoV_Sclerosis_Stenosis_obs` (different from `AoV_sten_degree_SD_obs` used in the source CSV) |

The root cause was that `syngo_measures.StudyRef` (integer) does NOT join to
`study_deid_map.OriginalStudyID` (DICOM UID). The earlier analysis assumed it did.

### CSV-to-NPZ Label Consistency

Probe CSVs were spot-checked against NPZ labels to confirm `build_probe_csvs.py` faithfully
copies labels. LVEF: 177 unique studies checked, 100% exact match. TAPSE: 184 unique studies
checked, 100% exact match. Labels in CSVs are stored in raw units; Z-score normalization is
applied at runtime by `eval.py` using `zscore_params.json` (auto-loaded from the task directory).

### Cross-Reference Verification (Classification)

For all 14 source-CSV classification tasks (5 valve severity + 9 structural findings), a
systematic cross-reference was performed: every NPZ study_id was looked up in the source
CSVs (`aws_syngo_findings_v2.csv` + `aws_heartlab_findings_v2.csv`), and for each integer
class, the most frequent `common_label` was confirmed to be the expected task-specific label.
All 14 tasks, all classes: the correct common_label is dominant. 100% coverage (every NPZ
study_id found in source CSVs).

### What IS fully reproducible

The full pipeline from source data to probe CSVs:
1. `build_labels.py` — echo.db + source CSVs → NPZ labels (verified for 4 tasks, extendable to all)
2. `build_probe_csvs.py` — NPZ → probe CSVs (deterministic; CSV→NPZ consistency verified)
3. `build_viewfiltered_csvs.py` — view/color filter (deterministic given classifier predictions)
4. TEE/stress filter script (deterministic given uhn_uid_to_studyref.csv)

### What is NOT reproducible

The following tasks cannot be exactly reconstructed from surviving code:
- **cardiac_output:** Source calculation unknown (not any single measurement, not SV×HR)
- **rv_function:** 80% reproduced; 24K studies from unknown additional sources
- **rwma:** Multi-source binary classification; original logic undocumented
- **9 disease tasks:** Free-text cohort construction from deleted notebook; structured
  observations cover only 1-19% of positive cases

---

## 14. Verification Queries

To spot-check label counts, run against echo.db. **These count ALL Syngo rows, not just
those with S3 video mappings.** Counts will differ from NPZ totals.

```sql
-- MR severity: raw Syngo value distribution (all studies, NOT just mapped ones)
SELECT Value, COUNT(*) AS count
FROM syngo_observations
WHERE Name = 'MV_Regurgitation_obs'
GROUP BY Value
ORDER BY count DESC;
-- Expected: trace=57004, mild=22913, mild_to_moderate=5653, None=3694, ...

-- AS severity: raw Syngo (note: different obs name than what source CSV uses)
-- echo.db has AoV_Sclerosis_Stenosis_obs (78K rows, includes sclerosis grades)
-- source CSV uses AoV_sten_degree_SD_obs (44K rows, AS severity only)
SELECT Value, COUNT(*) AS count
FROM syngo_observations
WHERE Name = 'AoV_sten_degree_SD_obs'
GROUP BY Value
ORDER BY count DESC;

-- TAPSE: measurement distribution
SELECT
    COUNT(*) as n,
    AVG(CAST(Value AS REAL)) as mean,
    MIN(CAST(Value AS REAL)) as min,
    MAX(CAST(Value AS REAL)) as max
FROM syngo_measures
WHERE MeasurementName = 'TAPSE (M-mode)'
  AND CAST(Value AS REAL) BETWEEN 0.5 AND 3.5;

-- LVEF: MOD BP only (this is what the NPZ uses)
SELECT
    COUNT(*) as n,
    AVG(CAST(Value AS REAL)) as mean
FROM syngo_measures
WHERE MeasurementName = 'LV EF, MOD BP';

-- Check mapping coverage: how many Syngo StudyRefs are in aws_syngo.csv?
-- (Cannot check from echo.db alone; requires loading the CSV)
```

**To run the full reproduction and validation:**
```bash
python experiments/nature_medicine/uhn/build_labels.py --all --validate
```

---

## Key References

- `experiments/nature_medicine/uhn/build_labels.py` — Reproduction script (source data -> NPZ)
- `experiments/nature_medicine/uhn/build_probe_csvs.py` — NPZ -> probe CSVs
- `experiments/nature_medicine/uhn/CLASS_MAPS.md` — Valve severity class maps with study counts
- `experiments/nature_medicine/uhn/patient_split.json` — Patient-level train/val/test assignment
- `experiments/nature_medicine/physiological_ranges.json` — Plausible value ranges (CSV-level)
- `data/aws/aws_syngo.csv` — StudyRef -> DeidentifiedStudyID mapping (320K, primary for Syngo)
- `data/aws/aws_syngo_findings_v2.csv` — Pre-joined Syngo observations with deid IDs + common_label
- `data/aws/aws_heartlab_findings_v2.csv` — Pre-joined HeartLab findings with deid IDs + common_label
- `uhn_echo/nature_medicine/data_exploration/echo.db` — Source SQLite database (5.7 GB)
- `uhn_echo/nature_medicine/data_exploration/reference/` — 17 reference docs covering all DB tables
