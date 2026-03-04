# MIMIC-IV Database Reference

Detailed context for MIMIC-IV linked to echocardiography lives in:
- `uhn_echo/nature_medicine/data_exploration/mimic/CLAUDE.md` — exhaustive schema, biomarker coverage, prediction targets, cross-validation results

## Scale

- **MIMIC-IV**: 364,627 patients, 546,028 admissions, 94,458 ICU stays
- **MIMIC-IV-Echo**: ~525,000 DICOM echocardiograms, 7,243 studies, 4,579 patients
- Data is gzip-compressed CSVs (`.csv.gz`); echo CSVs are plain CSV

## Modules

- `mimic-iv/hosp/` — demographics, admissions, diagnoses, labs, medications, procedures
- `mimic-iv/icu/` — vitals, inputs/outputs, ICU procedures (chartevents, inputevents, etc.)
- `mimic-iv-echo/` — echo study/record lists linking to DICOM files on PhysioNet
- `mimic-iv-ed/ed/` — ED visits, triage, vitals, medications (425K ED stays)
- `mimic-iv-note/note/` — discharge summaries, radiology reports (no echo notes yet)

## Key Prediction Targets

- **Mortality**: 30d 5.5%, 90d 8.9%, 1yr 17.0% (from patients.dod)
- **Readmission**: 30d 20.9%, 90d 33.8%, 1yr 49.3% (inpatient studies)
- **Rare diseases**: STEMI 237, HCM 196, Tamponade 134, DCM 97, Endocarditis 84, Amyloidosis 71, Takotsubo 43 patients
- **EF from discharge notes**: 2,743 studies (78.9% extraction rate), HFrEF 18.2%, HFmrEF 13.1%, HFpEF 68.7%
- **Biomarkers within ±24h**: Creatinine 53.6%, Troponin T 23.3%, NT-proBNP 11.8%

## Critical Data Engineering Notes

- **Troponin T**: 55% of `valuenum` is NULL — below-detection results (`<0.01`) stored in `comments` column. Parse comments or lose half the data.
- **Echo linkage**: Only 47.5% strictly inpatient; 52.5% outpatient. Use linkage tiers (strict → ±24h → ±7d → any admission).
- **Medications**: Prescriptions include in-hospital drugs (heparin, lidocaine); discharge note medication sections better reflect outpatient state. For medication changes, use medrecon as ground truth.
- **ICD coding**: Massive under-coding for structural findings (mitral regurgitation 10.3% ICD sensitivity). Use discharge notes for valve/structural conditions; ICD codes sufficient for primary cardiac diagnoses.

## Join Columns

- **subject_id**: patient (links all modules)
- **hadm_id**: admission
- **stay_id**: ICU stay
- **study_id**: echocardiogram study
