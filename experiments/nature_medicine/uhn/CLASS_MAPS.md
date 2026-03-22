# UHN Task Class Maps and Measurement Names

Comprehensive reference for all 47 UHN tasks: class mappings (classification) and
measurement sources (regression). Verified 2026-03-14 by count-matching NPZ labels
against source data.

---

## Regression Tasks (21 tasks)

### Verified MeasurementNames

All regression labels come from `syngo_measures` in echo.db, joined to deidentified study
IDs via `aws_syngo.csv` (STUDY_REF -> DeidentifiedStudyID). Values are stored as raw
measurements in original units.

| Task | MeasurementName | NPZ Studies | Mean | Unit | NPZ Filter |
|------|----------------|-------------|------|------|------------|
| lvef | `LV EF, MOD BP` | 51,341 | 54.4 | % | EF > 0 |
| tapse | `TAPSE (M-mode)` | 54,793 | 1.86 | cm | [0.3, 5.0] |
| rv_sp | `RV S'` | 54,090 | 0.12 | m/s | -- |
| rvsp | `RVSP (TR)` | 33,556 | 34.3 | mmHg | -- |
| rv_fac | `RV FAC A4C` | 15,227 | 37.3 | % | -- |
| edv | `LV vol d, MOD BP` | 51,141 | 110.6 | mL | -- |
| esv | `LV vol s, MOD BP` | 51,107 | 49.0 | mL | -- |
| lv_mass | `LV Mass 2D uhn` | 53,849 | 193.5 | g | -- |
| ivsd | `IVS d, 2D` | 62,403 | 1.04 | cm | -- |
| ao_root | `Ao Root d, 2D` | 57,455 | 3.16 | cm | -- |
| la_vol | `LA Vol A/L A4C` | 39,953 | 65.3 | mL | -- |
| aov_vmax | `AoV Vmax` | 55,095 | 1.60 | m/s | -- |
| aov_mean_grad | `AoV mean grad` | 44,403 | 11.3 | mmHg | -- |
| aov_area | `AoV area (VTI)` | 23,652 | 1.88 | cm2 | -- |
| lvot_vti | `LVOT VTI` | 43,539 | 0.197 | m | -- |
| mv_ea | `MV E/A ratio` | 47,316 | 1.23 | ratio | -- |
| mv_ee | `MV E/e'` | 50,411 | 9.74 | ratio | -- |
| mv_dt | `MV DT` | 42,671 | 211 | ms | -- |
| gls | `Global Peak Long Strain` | 23,574 | -19.4 | % | -- |

**Verification status:** 18 of 19 above are **perfectly verified** (100% value match
against NPZ). `ivsd` matches 62,399/62,403 values exactly (4 differ by < 0.001, likely
float precision).

### Partially Verified / Source Uncertain

| Task | Best Candidate Source | NPZ Studies | Overlap | Notes |
|------|---------------------|-------------|---------|-------|
| mv_ee_medial | `Mitral annulus medial E/e'` | 37,778 | 30,055/37,778 (80%) | 62% exact value match; ~7,700 studies likely from HeartLab |
| cardiac_output | Calculated or multi-source | 16,137 | -- | No single measurement matches. Not SV×HR. Source calculation unknown. |

**mv_ee_medial:** The best single measurement is "Mitral annulus medial E/e'" (53,786 raw
studies, 30,161 mapped). "MV E/E annular ratio, medial" is the second candidate (24,947
mapped). The ~62% exact match rate suggests the original pipeline may have averaged or
prioritized across multiple measurement types. All NPZ study IDs map through `aws_syngo.csv`.

**cardiac_output:** NPZ has 16,137 studies with mean=4.90 L/min, range [1.03, 14.95].
All IDs map through `aws_syngo.csv` (100% Syngo). No direct cardiac output measurement
in echo.db matches the study count or values. Likely calculated from a multi-step formula
in the deleted notebook. Direct CO measurements (LV CO, MOD A4C etc.) cover only ~7K
mapped studies with <1% value match.

---

## Classification: Valve Severity (5 tasks)

Built from pre-joined source CSVs (`aws_syngo_findings_v2.csv`, `aws_heartlab_findings_v2.csv`)
using the `common_label` column. Syngo + HeartLab union, highest severity wins per study.

### MR severity (5 classes, 133,303 studies)

| Class | Label | Studies | common_label |
|-------|-------|---------|-------------|
| 0 | none | 8,292 | MR_none |
| 1 | trace | 26,633 | MR_trace |
| 2 | mild | 81,601 | MR_mild |
| 3 | moderate | 12,173 | MR_moderate |
| 4 | severe | 4,604 | MR_severe |

Source: `MV_Regurgitation_obs` (Syngo), HeartLab Group 93. HeartLab has no trace grade.

### TR severity (5 classes, 138,966 studies)

| Class | Label | Studies | common_label |
|-------|-------|---------|-------------|
| 0 | none | 5,715 | TR_none |
| 1 | trivial/trace | 24,512 | TR_trace |
| 2 | mild | 91,685 | TR_mild |
| 3 | moderate | 13,254 | TR_moderate |
| 4 | severe | 3,800 | TR_severe |

Source: `TR_degree_obs` (Syngo, hyphens not underscores), HeartLab Group 116.

### AR severity (5 classes, 80,175 studies)

| Class | Label | Studies | common_label |
|-------|-------|---------|-------------|
| 0 | none | 26,532 | AR_none |
| 1 | trace | 11,525 | AR_trace |
| 2 | mild | 40,912 | AR_mild |
| 3 | moderate | 908 | AR_moderate |
| 4 | severe | 298 | AR_severe |

Source: `AoV_Regurgitation_obs` (Syngo), HeartLab Group 100018. Classes 3-4 very small.

### AS severity (4 classes, 175,932 studies)

| Class | Label | Studies | common_label |
|-------|-------|---------|-------------|
| 0 | none/sclerosis | 145,697 | AS_none |
| 1 | mild | 20,346 | AS_mild |
| 2 | moderate | 4,607 | AS_moderate |
| 3 | severe | 5,282 | AS_severe |

Source: `AoV_sten_degree_SD_obs` (Syngo, 21K studies), HeartLab AS findings (155K studies).
Sclerosis merged with "none" in class 0.

### PR severity (5 classes, 77,307 studies)

| Class | Label | Studies | common_label |
|-------|-------|---------|-------------|
| 0 | none | 22,863 | PR_none |
| 1 | trace | 24,940 | PR_trace |
| 2 | mild | 26,213 | PR_mild |
| 3 | moderate | 1,375 | PR_moderate |
| 4 | severe | 1,916 | PR_severe |

Source: `PV_Regurgitation_obs` (Syngo), HeartLab Group 131.

---

## Classification: Structural Findings (10 tasks)

Built from source CSVs. Verified by matching `common_label` counts against NPZ class counts.

### LV systolic function (6 classes, 125,538 studies)

**Class ordering is severity-first (reversed from clinical convention):**

| Class | Label | Studies | common_label |
|-------|-------|---------|-------------|
| 0 | severe | 5,282 | LVSF_severe |
| 1 | moderate | 4,947 | LVSF_moderate |
| 2 | mild | 5,846 | LVSF_mild |
| 3 | low normal | 9,939 | LVSF_low_normal |
| 4 | normal | 92,825 | LVSF_normal |
| 5 | hyperdynamic | 6,699 | LVSF_hyperdynamic |

Source: `LV_Fx_qualitative_obs` (Syngo), HeartLab LVSF findings. Intermediate grades
(e.g., mild-to-moderate) collapsed into the more severe class.

### LV cavity size (5 classes, 236,882 studies)

**Class ordering is severity-first (reversed from clinical convention):**

| Class | Label | Studies | common_label |
|-------|-------|---------|-------------|
| 0 | severe | 4,355 | LVC_severe |
| 1 | decreased | 11,104 | LVC_decreased |
| 2 | moderate | 6,730 | LVC_moderate |
| 3 | mild | 15,633 | LVC_mild |
| 4 | normal | 199,060 | LVC_normal |

Source: `LV_Cavity_Size-ASE_obs` (Syngo), HeartLab LVC findings. Class 4 (normal) is
84% of the dataset, creating extreme class imbalance.

### LV hypertrophy (5 classes, 41,872 studies)

| Class | Label | Studies | common_label |
|-------|-------|---------|-------------|
| 0 | normal | 160 | LVH_normal |
| 1 | mild | 29,557 | LVH_mild |
| 2 | moderate | 5,372 | LVH_moderate |
| 3 | severe | 290 | LVH_severe |
| 4 | asymmetric | 6,493 | LVH_asymmetric |

Source: `Hypertrophy_degree-ASE_obs` (Syngo), HeartLab LVH findings. "Asymmetric" includes
concentric remodeling patterns. Class 0 (normal) has only 160 studies; class 3 (severe)
has only 290. Extreme class imbalance.

### LA size (4 classes, 139,049 studies)

| Class | Label | Studies | common_label |
|-------|-------|---------|-------------|
| 0 | normal | 55,884 | LA_normal |
| 1 | mild | 34,451 | LA_mild |
| 2 | moderate | 39,238 | LA_moderate |
| 3 | severe | 9,476 | LA_severe |

Source: `LA_size-ASE_obs` (Syngo), HeartLab LA findings.

### RA size (4 classes, 64,813 studies)

| Class | Label | Studies | common_label |
|-------|-------|---------|-------------|
| 0 | normal | 28,521 | RA_normal |
| 1 | mild | 14,491 | RA_mild |
| 2 | moderate | 9,989 | RA_moderate |
| 3 | severe | 11,812 | RA_severe |

Source: `RA_size-ASE_obs` (Syngo), HeartLab RA findings.

### RV size (4 classes, 87,912 studies)

| Class | Label | Studies | common_label |
|-------|-------|---------|-------------|
| 0 | normal | 41,388 | RV_normal |
| 1 | mild | 30,414 | RV_mild |
| 2 | moderate | 11,279 | RV_moderate |
| 3 | severe | 4,831 | RV_severe |

Source: `RV_size-ASE_obs` (Syngo), HeartLab RV findings.

### PA pressure (4 classes, 39,171 studies)

| Class | Label | Studies | common_label |
|-------|-------|---------|-------------|
| 0 | normal | 5,549 | PAP_normal |
| 1 | mild | 22,888 | PAP_mild |
| 2 | moderate | 8,395 | PAP_moderate |
| 3 | severe | 2,339 | PAP_severe |

Source: `PA_pressure_obs` (Syngo), HeartLab PAP findings.

### Pericardial effusion (5 classes, 176,933 studies)

| Class | Label | Studies | common_label |
|-------|-------|---------|-------------|
| 0 | none | 153,626 | PEF_none |
| 1 | trivial | 11,902 | PEF_trivial |
| 2 | small | 8,554 | PEF_small |
| 3 | moderate | 2,453 | PEF_moderate |
| 4 | large | 398 | PEF_large |

Source: `PE_Effusion_sD_obs` (Syngo), HeartLab PEF findings.

### Diastolic function (4 classes, 42,115 studies)

| Class | Label | Studies | common_label |
|-------|-------|---------|-------------|
| 0 | normal | 1,830 | DFP_normal |
| 1 | grade I (impaired relaxation) | 36,369 | DFP_grade_I |
| 2 | grade II (pseudonormal) | 1,715 | DFP_grade_II |
| 3 | grade III/IV (restrictive) | 2,201 | DFP_grade_III_IV |

Source: `LV_diast_filling-ASE_obs` (Syngo), HeartLab DFP findings. Class 0 (normal)
has only 1,830 studies; class 1 (grade I) dominates with 86%.

---

## Classification: Non-Source-CSV Tasks (3 tasks)

These tasks are NOT in the pre-joined source CSVs and were built directly from echo.db
observations. The `aws_syngo_findings_v2.csv` contains only 18 of 1,118 Syngo observation names.

### Cardiac rhythm (6 classes, 44,672 studies)

| Class | Label | Studies | Syngo Value (`Cardiac_Rhythm_obs`) |
|-------|-------|---------|-----------------------------------|
| 0 | NSR (normal sinus rhythm) | 33,739 | `NSR` |
| 1 | atrial fibrillation | 2,664 | `A_fib` |
| 2 | sinus tachycardia | 2,453 | `sinus_tachy` |
| 3 | sinus bradycardia | 1,988 | `sinus_brady` |
| 4 | artificially paced | 1,548 | `artificially_paced` |
| 5 | irregular | 2,280 | `irregular`, `Irregular` |

**Verified:** 99.9% class match (44,110/44,166 overlapping studies) when reconstructed
from `Cardiac_Rhythm_obs` in echo.db. Syngo-only (no HeartLab). 506 NPZ studies not
in echo.db may come from minor rhythm observation variants (with_PVC, sinus_arr, etc.)
that were collapsed into major classes.

### RV function (5 classes, 125,229 studies)

| Class | Label | Studies | Syngo Value (`RV_syst._fx-ASE_obs`) | HeartLab common_label |
|-------|-------|---------|-------------------------------------|----------------------|
| 0 | normal | 59,385 | `normal`, `grossly_normal` | `RVF_normal` |
| 1 | low normal | 8,411 | `low_normal` | `RVF_low_normal` |
| 2 | mildly reduced | 36,679 | `mildly_reduced` | `RVF_mild` |
| 3 | moderately reduced | 14,674 | `moderately_reduced` | `RVF_moderate` |
| 4 | severely reduced | 6,080 | `severely_reduced` | `RVF_severe` |

**Partially verified:** 80% class match (80,416/100,989 overlapping studies) with
Option 1 mapping (normal+grossly_normal=0, low_normal=1, mild=2, moderate=3, severe=4).
Syngo raw has 94,782 studies (45K mapped), HeartLab has 58,354 unique studies. 24,179
NPZ studies not found in either reconstruction source. Syngo `hyperdynamic` (169 studies)
excluded. The merge logic (highest-severity-wins) and additional sources for the missing
studies are unknown.

### RWMA (2 classes, 69,452 studies)

| Class | Label | Studies |
|-------|-------|---------|
| 0 | no RWMA | 43,475 |
| 1 | RWMA present | 25,977 |

**Not verified from source.** Binary regional wall motion abnormality detection. Likely
built from multiple echo.db observations: `LV_fx_regional_wma_obs` (10K studies),
`LV_WMA_Comments_obs` (18K studies), HeartLab "Regional Wall Motion" findings, and
possibly text-mining of WMA comments. The original cohort construction logic is not
documented in surviving code.

---

## Classification: Disease Detection (9 tasks)

Binary (0/1) tasks. Positive cohorts built from disease-specific observations, findings,
and free-text search across Syngo reports and HeartLab data. Negative controls are
clinically-matched hard negatives (not random).

**Build script:** `build_disease_labels.py` (created 2026-03-20). Full provenance in
`UHN_DISEASE_PROVENANCE.md`. **Recommended: `labels_v7_study_level/`** (v7 = v6 + STEMI ischemic obs fix + myxomatous criteria negation + endocarditis dropped).

### Source Architecture

Three label sources queried per disease, then unioned:
1. **Syngo structured obs** (`syngo_observations`): Name/Value lookup (1-16% of positives)
2. **HeartLab findings** (SENTENCE + NOTE): SENTENCE path via `heartlab_findings.SENTENCE` (~90% of HL matches), NOTE free-text (supplementary). ~60-93% of positives.
3. **Syngo free-text**: `syngo_analytics_study` + `syngo_study_details` (StudyComments, PatientHistory). 4-41% of positives.

Negation filtering via pattern matching + proximity negation (v5). Word boundary validation for short acronyms (v5). `--exclude_indication` removes referral fields. `--require_heartlab` drops studies confirmed only by Syngo free-text.

### Disease Summary (v7.2 study-level, 8 diseases)

| Task | Pos | Neg | Neg Control | Notes |
|------|-----|-----|-------------|-------|
| disease_hcm | 10,264 | 38,072 | Concentric LVH | v7.1: -1,246 neg (HCM indication cross-check) |
| disease_dcm | 2,438 | 14,023 | HF without DCM | |
| disease_bicuspid_av | 7,390 | 177,415 | Tricuspid AV | |
| disease_amyloidosis | 747 | 10,247 | HCM | v7.2: -3 pos (family history exclusion) |
| disease_rheumatic_mv | 2,292 | 1,104 | Non-rheumatic MS | v7.2: -113 neg (finding-group exclusion, spacing fix) |
| disease_stemi | 603 | 1,125 | NSTEMI | v7: ischemic obs disabled |
| disease_myxomatous_mv | 4,989 | 121,696 | Non-myxomatous MR | v7.2: -548 pos (VALVE/ROOT DISEASE group excluded) |
| disease_takotsubo | 74 | 602 | STEMI | Most pos hedged; N=74 |
| ~~disease_endocarditis~~ | — | — | — | **DROPPED v7**: 30% precision, unfixable |

Four audit rounds: 50 pos auto (seed=789), 10+10 manual (seeds 42, 9999, 7777).
Full provenance, per-sample verdicts, and known limitations in `UHN_DISEASE_PROVENANCE.md`.

### Cohort Design Notes

- All negative cohorts are **hard negatives** (clinically similar conditions), not random controls
- Endocarditis negatives are studies where endocarditis was specifically ruled out
- **STEMI v4 was catastrophically broken** — `LIKE '%STEMI%'` matched "systemic" (89.6% of HL matches false). v5 uses word boundary regex.
- Takotsubo has only 74 positives in v5 — may be underpowered for probe training
- STEMI test set is 173 studies — small but usable
- Endocarditis 88% precision has residual RO leaks from non-adjacent negation

### Confidence Tiers (v6 automated + manual raw-text audit)

| Tier | Diseases | Audit Prec | Manual | Recommendation |
|------|----------|:----------:|:------:|----------------|
| **Use confidently** | HCM, DCM, Bicuspid AV, STEMI, Amyloidosis, Rheumatic MV | 96-100% | Clean | Include in manuscript |
| **Use with caveats** | Myxomatous MV | 100% | Neg class impure | Include; some negs have no MR |
| **Use with caveats** | Takotsubo | 98% | Pos hedged | Include; most pos are "?takotsubo" queries. N=74 |
| **Problematic** | Endocarditis | 80% | 2/3 pos false | Use with strong caveat; intervening-word negation unfixable |

---

## Data Provenance

- **Build script:** `build_labels.py` reproduces regression + source-CSV classification tasks
- **Source CSVs:** `data/aws/aws_syngo_findings_v2.csv`, `data/aws/aws_heartlab_findings_v2.csv`
- **Database:** `uhn_echo/nature_medicine/data_exploration/echo.db`
- **NPZ location:** `experiments/nature_medicine/uhn/labels/{task}.npz`
- **NPZ keys:** `study_ids`, `patient_ids`, `labels`, `splits`
- **Deduplication:** Syngo+HeartLab union by DeidentifiedStudyID, highest severity wins
- **Patient splits:** `patient_split.json` (138,779 patients: 81.5% train, 7.0% val, 11.4% test)
- **TEE/Stress filtering:** Applied to probe CSVs; 7,341 UIDs excluded, 0-3.2% per task

## Verification Summary (2026-03-14)

| Category | Tasks | Status |
|----------|-------|--------|
| Regression (single measurement) | 18/21 | Perfect reproduction (100% value match) |
| Regression (ivsd) | 1/21 | 62,399/62,403 exact (4 differ by <0.001, float precision) |
| Regression (mv_ee_medial) | 1/21 | 80% study overlap, 62% exact value match |
| Regression (cardiac_output) | 1/21 | Source calculation unknown |
| Classification (valve severity) | 5/5 | Confirmed: dominant common_label matches expected class for all classes |
| Classification (structural, source-CSV) | 9/9 | Confirmed: dominant common_label matches expected class for all classes |
| Cardiac rhythm | 1/1 | 99.9% class match from echo.db |
| RV function | 1/1 | 80% class match (merge logic uncertain) |
| RWMA | 1/1 | Not verifiable (multi-source, undocumented logic) |
| Disease detection | 9/9 | Rebuilt v6 with full provenance; 88-100% audit precision |

### Cross-reference method (classification)

For all 14 source-CSV classification tasks, every NPZ study_id was looked up in
`aws_syngo_findings_v2.csv` + `aws_heartlab_findings_v2.csv`. For each integer class,
the most frequent `common_label` among matched studies was confirmed to be the expected
task-specific label (e.g., class 2 in mr_severity → MR_mild dominant). 100% of classes
across all 14 tasks have the correct dominant common_label.

### CSV-to-NPZ label consistency

Probe CSVs (`probe_csvs/{task}/train.csv`) were spot-checked against NPZ labels for
LVEF (177 unique studies, 100% exact match) and TAPSE (184 unique studies, 100% exact
match). The `build_probe_csvs.py` script faithfully copies NPZ labels into CSVs. Z-score
normalization is applied at runtime by `eval.py` using `zscore_params.json`, not in the CSVs.
