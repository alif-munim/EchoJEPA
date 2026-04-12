# Prediction Averaging: Additional Models (EchoJEPA-B, L-K, L)

**Scope: EchoJEPA-B, EchoJEPA-L-K, EchoJEPA-L** (supplementary models)

See `inference-tracker.md` for the primary 3-model comparison (EchoJEPA-G, EchoPrime, PanEcho).

Last updated: 2026-04-12

---

## EchoJEPA-B

### UHN (20 tasks, all done):

| Task | Status |
|------|--------|
| aov_mean_grad | ✅ |
| aov_vmax | ✅ |
| ar_severity | ✅ |
| as_severity | ✅ |
| disease_amyloidosis | ✅ |
| disease_bicuspid_av | ✅ |
| disease_dcm | ✅ |
| disease_hcm | ✅ |
| disease_myxomatous_mv | ✅ |
| disease_rheumatic_mv | ✅ |
| disease_stemi | ✅ |
| lvef | ✅ |
| mr_severity | ✅ |
| mv_ee_medial | ✅ |
| rv_fac | ✅ |
| rv_sp | ✅ |
| rvsp | ✅ |
| tapse | ✅ |
| tr_severity | ✅ |
| trajectory_lvef_onset | ✅ |

Note: Both `rv_sp` and `rvsp` NPZs exist (may be duplicate naming for the same task).

### MIMIC (15 tasks, all done):

| Task | Status | Type |
|------|--------|------|
| creatinine | ✅ | predavg |
| disease_afib | ✅ | predavg |
| disease_amyloidosis-mimic-xfer | ✅ | xfer |
| disease_dcm-mimic-xfer | ✅ | xfer |
| disease_hcm-mimic-xfer | ✅ | xfer |
| disease_stemi-mimic-xfer | ✅ | xfer |
| ef_note_extracted | ✅ | predavg |
| ef_note_extracted-xfer | ✅ | xfer |
| lactate | ✅ | predavg |
| mortality_1yr | ✅ | predavg |
| mortality_30d | ✅ | predavg |
| mortality_90d | ✅ | predavg |
| nt_probnp | ✅ | predavg |
| readmission_30d | ✅ | predavg |
| troponin_t | ✅ | predavg |

**EchoJEPA-B total: 35 NPZ files (20 UHN + 15 MIMIC), all complete.**

---

## EchoJEPA-L-K

### UHN (8 tasks, all done):

| Task | Status |
|------|--------|
| cardiac_output | ✅ |
| diastolic_function | ✅ |
| edv | ✅ |
| esv | ✅ |
| rvsp | ✅ |
| trajectory_lvef | ✅ |
| trajectory_lvef_onset | ✅ |
| trajectory_mr_severity_onset | ✅ |

### MIMIC (14 tasks, all done):

| Task | Status | Type |
|------|--------|------|
| creatinine | ✅ | predavg |
| discharge_destination | ✅ | predavg |
| ef_note_extracted-xfer | ✅ | xfer |
| in_hospital_mortality | ✅ | predavg |
| lactate | ✅ | predavg |
| los_remaining | ✅ | predavg |
| mitral_regurg-xfer | ✅ | xfer |
| mortality_1yr | ✅ | predavg |
| mortality_30d | ✅ | predavg |
| mortality_90d | ✅ | predavg |
| nt_probnp | ✅ | predavg |
| readmission_30d | ✅ | predavg |
| tricuspid_regurg-xfer | ✅ | xfer |
| troponin_t | ✅ | predavg |

**EchoJEPA-L-K total: 22 NPZ files (8 UHN + 14 MIMIC), all complete.**

### UHN tasks still missing for L-K:

The following UHN tasks have G/EP/Pan coverage (or echojepa-b coverage) but no L-K NPZ:

- aov_mean_grad, aov_vmax, ar_severity, as_severity
- disease_amyloidosis, disease_bicuspid_av, disease_dcm, disease_hcm, disease_myxomatous_mv, disease_rheumatic_mv, disease_stemi
- lvef, mr_severity, mv_ee_medial, rv_fac, rv_function, rv_sp, tapse, tr_severity
- trajectory_lvef_onset (wait — this is done above)

That is 19 UHN tasks without L-K NPZ (if needed for Extended Data or supplementary analysis).

---

## EchoJEPA-L

### UHN (2 tasks):

| Task | Status |
|------|--------|
| rvsp | ✅ |
| trajectory_lvef | ✅ |

### MIMIC: None.

**EchoJEPA-L total: 2 NPZ files (UHN only).**

---

## Grand Total Across All Additional Models

| Model | UHN NPZs | MIMIC NPZs | Total |
|-------|----------|------------|-------|
| EchoJEPA-B | 20 | 15 | 35 |
| EchoJEPA-L-K | 8 | 14 | 22 |
| EchoJEPA-L | 2 | 0 | 2 |
| **Total** | **30** | **29** | **59** |

Combined with the primary tracker (69 NPZ files for G/EP/Pan), the full inventory is **128 NPZ files** on disk (remaining files are trainfeat-G variants tracked in the primary doc).

---

## File Locations

Same path convention as the primary tracker:

| Asset | Path |
|-------|------|
| NPZ outputs (EFS) | `evals/vitg-384/nature_medicine/{uhn,mimic}/video_classification_frozen/{task}-predavg-{model}/clip_outputs.npz` |
| Study-level stats | `predictions/nature_medicine/study_level_statistics/{uhn,mimic}/{task}-predavg-{model}.json` |
