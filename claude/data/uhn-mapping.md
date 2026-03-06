# UHN DICOM UID → Syngo StudyRef Mapping

How to link S3 echocardiogram clips (keyed by deidentified DICOM UIDs) to clinical labels in echo.db (keyed by Syngo StudyRef). This is the foundational step for all UHN experiments in the Nature Medicine paper.

## Mapping Chain

```
S3 path: s3://echodata25/results/echo-study/{study_uid}/{series_uid}/{instance_uid}.mp4
                                              │
                                              ▼
                              study_deid_map.DeidentifiedStudyID  (exact match)
                                              │
                              ┌───────────────┼───────────────┐
                              ▼               ▼               ▼
                     OriginalStudyID   OriginalPatientID   DeidentifiedPatientID
                              │               │
                     (extract date)           │
                              │               │
                              ▼               ▼
                    syngo_study_details.STUDY_DATE + PATIENT_ID  (join)
                              │
                              ▼
                    syngo_study_details.STUDY_REF  →  syngo_measures / syngo_observations
```

**Key insight:** `study_deid_map` does NOT contain StudyRef. The bridge is through `OriginalPatientID = syngo_study_details.PATIENT_ID` (same 7-digit format, verified 20/20), disambiguated by study date extracted from the original DICOM UID.

## Files

### Mapping CSVs (gitignored, never track)

All at `experiments/nature_medicine/uhn/mapping/`. Originals at `data/data/`.

| File | Rows | Columns | Purpose |
|------|------|---------|---------|
| `deid_key.csv` | 232,835 | `deidentified_study_id`, `original_study_id`, `deidentified_patient_id`, `original_patient_id` | Core mapping. Filtered to numeric patient IDs only. |
| `echo_study_deid.csv` | 237,022 | `OriginalStudyID`, `OriginalPatientID`, `DeidentifiedStudyID`, `DeidentifiedPatientID` | Complete mapping (loaded into echo.db as `study_deid_map`). 4K more rows than deid_key (includes non-numeric patient IDs). |
| `patient_to_study.csv` | 232,848 | `deidentified_patient_id`, `original_patient_id`, `deidentified_study_ids`, `original_study_ids` | Patient-to-study groupings. Study ID lists stored as Python list strings (`ast.literal_eval` to parse). Nearly 1:1 (avg 1.00006 studies/patient). |
| `ecs_master.csv` | 216,807 | `deidentified_study_id`, `original_study_id`, `deidentified_patient_id`, `original_patient_id`, `STUDY_ID`, `PATI_ID`, `STUDY_DATE`, `STUDY_DESCRIPTION`, `SERI_ID`, `MODALITY`, `SERIES_INSTANCE_UID`, `REP_ID`, `CREATION_DATE`, `FINALIZED_DATE` | HeartLab-linked master. Maps deidentified DICOM UIDs to HeartLab report IDs (REP_ID), study IDs, series, dates. Built by `data/data/ecs_mappings.ipynb`. |

### Source Notebooks (at `data/data/`)

| Notebook | Purpose | Key Output |
|----------|---------|------------|
| `identifiers.ipynb` | Connects deidentified DICOMs to HeartLab via `dataset/echo-study-deid.csv` → `dataset/STUDIES.csv` (join on `STUDY_INSTANCE_UID`). 223,450 matches. | `patient_to_study.csv` |
| `ecs_mappings.ipynb` | Full chain: deid → HeartLab studies → series → reports. Produces the master mapping with REP_ID for measurement/finding lookups. | `ecs_master.csv`, `ecs_studies.csv`, `ecs_series.csv`, `ecs_reports.csv` |
| `syngo.ipynb` | Syngo analytics exploration (measurements, observations). Uses echo.db. | Various analyses |
| `a4c_measurements.ipynb` | A4C-specific LVEF measurement extraction (basis for ICML LVEF CSVs). | LVEF label CSVs |

### Database Tables

| Table | Location | Rows | Key Columns |
|-------|----------|------|-------------|
| `study_deid_map` | echo.db | 237,022 | `DeidentifiedStudyID`, `OriginalStudyID`, `OriginalPatientID`, `DeidentifiedPatientID` |
| `syngo_study_details` | echo.db | 389,666 | `STUDY_REF` (uppercase!), `PATIENT_ID`, `STUDY_DATE`, `STUDY_TIME`, `ACCESSION_NUMBER` |
| `heartlab_studies` | echo.db | 431,710 | `ID`, `PATI_ID`, `STUDY_INSTANCE_UID`, `STUDY_DATE`, `ACCESSION_NUMBER` |

### Other Files (at `data/data/`, not copied)

| File | Rows | Why Not Copied |
|------|------|----------------|
| `dicom_df.csv` | 1,775,506 | DICOM parent/subdir/file hierarchy, but only covers 31,663 studies (10% of master index). 293 MB. Reference from `data/data/` if needed. |
| `dataset/STUDIES.csv` | 431,710 | Full HeartLab studies table. Large. Already in echo.db as `heartlab_studies`. |
| `dataset/SERIES.csv` | 1,028,007 | Full HeartLab series table. Already in echo.db as `heartlab_series`. |
| `dataset/REPORTS.csv` | 318,218 | HeartLab reports metadata. Already in echo.db as `heartlab_reports`. |
| `dataset/Patients_No_PHI.csv` | 314,077 | HeartLab patients (ID, SEX only). Already in echo.db as `patients`. |

## Coverage

| Metric | Count | % of Total |
|--------|-------|------------|
| Master index unique study UIDs | 319,818 | 100% |
| Mapped via study_deid_map | 214,149 | 66.9% |
| **Unmapped (later deid batch)** | **105,669** | **33.1%** |
| Clips in mapped studies | 11,679,005 | 64.5% |
| Clips in unmapped studies | 6,432,407 | 35.5% |

The 106K unmapped studies were deidentified ~Jan 2024+ (their deidentified UIDs have later timestamps). The deid key was generated ~Dec 2023.

### Label Coverage (for mapped patients)

| Dataset | Mapped | Total | Coverage |
|---------|--------|-------|----------|
| Unique patients in deid_map | 103,760 | — | — |
| Syngo studies for those patients | 310,818 | 389,666 | 79.7% |
| LVEF (MOD BP) studies | 46,866 | 92,989 | 50.4% |

50% LVEF coverage because many syngo studies for mapped patients don't have a corresponding DICOM study in S3. Full coverage requires the updated deid key.

## Date Extraction from OriginalStudyID

97.5% of original DICOM study UIDs embed a study date. Three patterns:

### Pattern 1: YYYYMMDD (52.5%)

Standalone or followed by time digits. Most common in GE, Philips UIDs.

```
1.2.124.113532.6.23355.16818.20090612.93443.119880
                                ^^^^^^^^
                                20090612
```

### Pattern 2: Unix Timestamp (many of the remainder)

10-digit epoch seconds embedded in the UID. Common in older GE/Philips UIDs.

```
1.2.840.113680.1.103.65868.1148668030.13782
                            ^^^^^^^^^^
                            1148668030 → 2006-05-26
```

### Pattern 3: Siemens YYMMDD (2-3%)

After `300000` prefix. Siemens UIDs (root `1.3.12.2.1107.5`).

```
1.3.12.2.1107.5.5.10.400111.30000013022508000085900000023
                             ^^^^^^^^^^^^^^
                             300000 13 02 25 → 2013-02-25
```

### Implementation

```python
import re
from datetime import datetime

def extract_date(uid):
    """Extract YYYYMMDD study date from a DICOM StudyInstanceUID."""
    # Pattern 1: YYYYMMDD with optional trailing time
    m = re.search(r'\.(\d{4}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01]))\d*[.\s]', uid + '.')
    if m:
        return m.group(1)
    # Pattern 2: Unix timestamp (2000-2030)
    m = re.search(r'\.(\d{10})\.', uid)
    if m:
        ts = int(m.group(1))
        if 946684800 <= ts <= 1893456000:
            return datetime.fromtimestamp(ts).strftime('%Y%m%d')
    # Pattern 3: Siemens 300000YYMMDD
    m = re.search(r'300000(\d{2})(\d{2})(\d{2})', uid)
    if m:
        yy, mm, dd = m.groups()
        return f'20{yy}{mm}{dd}'
    return None
```

## Same-Date Collision Resolution

1.3% of patient+date combinations have multiple syngo studies (4,895 groups, 10,143 studies). Options:

1. **STUDY_TIME matching**: syngo stores `STUDY_TIME` (e.g., `125005`). Some OriginalStudyIDs embed time after date.
2. **Accession number**: syngo `ACCESSION_NUMBER` is partially populated. HeartLab may have it too.
3. **First match**: For most collisions, same-day studies have similar measurements. Accept first match.

## HeartLab Bridge (Alternative Path)

For studies in HeartLab (223K matches), an alternative chain avoids date parsing:

```
DeidentifiedStudyID → OriginalStudyID → heartlab_studies.STUDY_INSTANCE_UID
  → heartlab_studies.ID (STUDY_ID) → heartlab_series → heartlab_reports (REP_ID)
  → measurements/findings via REP_ID
```

This is the chain used for ICML experiments (view classification, LVEF). It only reaches HeartLab labels (pre-2014), not Syngo labels (2005-2019). The `ecs_master.csv` file contains this pre-built mapping.

## Patient ID Systems

Three separate patient ID namespaces exist:

| System | Field | Format | Example |
|--------|-------|--------|---------|
| Hospital (original) | `deid_key.original_patient_id`, `syngo_study_details.PATIENT_ID` | 7-digit zero-padded | `0595291` |
| HeartLab internal | `heartlab_studies.PATI_ID`, `patients.ID` | Variable-length integer | `108221` |
| Deidentified | `deid_key.deidentified_patient_id` | UUID | `00001302-70bf-4ec4-8bde-1b3f9bf98689` |

**`original_patient_id` = `PATIENT_ID` in syngo** (confirmed). HeartLab `PATI_ID` is a separate internal ID that joins to `patients.ID` but NOT to the hospital patient ID.

## AWS Pre-Built Mapping Files (at `user-default-efs/vjepa2/data/aws/`)

These files resolve the temporal mismatch by providing the complete 2002-2019 mapping. The updated deid key is `R_21_009_011_echo_study_parts2and3_results.csv`.

| File | Rows | Purpose |
|------|------|---------|
| `aws_syngo.csv` | 320,854 | **Master mapping**: STUDY_REF ↔ DeidentifiedStudyID ↔ s3_key ↔ PATIENT_ID. All 320K studies with S3 video. |
| `R_21_009_011_echo_study_parts2and3_results.csv` | 342,763 | Updated deid key (2002-2019). STUDY_REF + OriginalStudyID + DeidentifiedStudyID. |
| `aws_syngo_msmt_v2.csv` | 67,690 | Pre-extracted Syngo measurements (18 types) with DeidentifiedStudyID, group_tag |
| `aws_syngo_findings_v2.csv` | 542,428 | Pre-extracted Syngo observations (69 conditions) with condition_tag, common_label |
| `aws_heartlab_msmt_v2.csv` | 255,176 | Pre-extracted HeartLab measurements (20 types) with DeidentifiedStudyID |
| `aws_heartlab_findings_v2.csv` | 1,263,258 | Pre-extracted HeartLab findings (69 conditions) with condition_tag, common_label |
| `es_union.csv` | 201,803 | Per-study union of all findings + measurements as dicts |
| `label2tag.csv` | 20 | Measurement group tag dictionary (N0-N19) |
| `condition_dict.csv` | 74 | Condition tag dictionary (C0-C78) |

**100% of labeled studies (272K union) have S3 paths via aws_syngo.csv.**

For full measurement coverage (e.g., 51K LVEF studies), query echo.db and join with `aws_syngo.csv` on STUDY_REF. The pre-extracted v2 files cover a curated subset (~6K Syngo measurement studies, ~54K Syngo finding studies, ~217K HeartLab finding studies).

## Action Items

1. ~~Request updated deid key from Joe~~ — **DONE.** `R_21_009_011_echo_study_parts2and3_results.csv` IS the updated key (342K rows, 2002-2019).
2. ~~Build `uhn_uid_to_studyref.csv`~~ — **DONE** (224K rows), but superseded by `aws_syngo.csv` (320K rows).
3. ~~Build patient splits~~ — **DONE.** `experiments/nature_medicine/uhn/patient_split.json` (138K patients, 70/10/20 temporal split).
4. ~~Build label NPZs~~ — **DONE.** 53 NPZs at `experiments/nature_medicine/uhn/labels/` (20 regression + 17 classification + 9 rare disease + 6 trajectory + 1 RWMA).
5. ~~Start GPU extraction~~ — **IN PROGRESS.** EchoJEPA-G (ViT-G, bf16) on 8×A100-80GB, 18.1M clips, ~25h ETA. Script: `evals/extract_uhn_embeddings.py`. Config: `configs/inference/vitg-384/extract_uhn.yaml`. Output: `experiments/nature_medicine/uhn/echojepa_g_embeddings/`.
6. **Re-extract MIMIC embeddings** for PanEcho, EchoPrime, EchoFM — normalization bugs fixed (see below).

## Normalization Bug Fixes (2026-03-06)

Three encoder adapters had incorrect normalization when used with the shared `make_transforms` pipeline (which applies ImageNet normalization):

- **PanEcho** (`panecho_encoder.py`): was applying ImageNet normalization a second time (double-normalization). Fixed: removed redundant normalization from `_preprocess`.
- **EchoPrime** (`echo_prime_encoder.py`): was multiplying ImageNet-normalized values by 255 then applying EchoPrime norm. Fixed: undo ImageNet norm → scale to [0,255] → apply EchoPrime norm.
- **EchoFM** (`echofm_encoder.py`): was passing ImageNet-normalized values to a model trained on [0,1] range. Fixed: undo ImageNet norm before encoder.

EchoJEPA (ViT-G/L) and VideoMAE (EchoMAE-L) are unaffected — both trained with ImageNet normalization.
