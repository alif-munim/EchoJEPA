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
