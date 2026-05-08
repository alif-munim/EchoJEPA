# Final NeurIPS Task List — Single-View MIMIC Probes

Probe tasks built on single-view-filtered MIMIC splits, ViT-L d=4
attentive probes, standard 20-ep protocol. CSV bundles staged at
`experiments/nature_medicine/mimic/probe_csvs/<task>/{train,val,test}.csv`.

**View filters** (all from the ConvNeXt view manifest at
`user-default-efs/vjepa2/classifier/output/mimic_view_predictions.csv`,
`view_status == OK`):
- HCM + LV morphology (IVS thickness, LVPW thickness, LVIDd, LVIDs) use
  **PLAX** (79,742 clips available) — the canonical M-mode-style
  parasternal measurements.
- All other tasks use **A4C** (75,890 clips available).

Paths match the existing `s3://echodata25/mimic-echo-224px/files/pXX/...`
layout byte-for-byte — no re-download needed.

---

## 1. HCM detection — PLAX

**Label**: binary HCM from MIMIC HCM v4.1 (4,714 studies, 177 positive).
Source splits preserved (patient-level), clips narrowed to PLAX.

**Why PLAX instead of A4C**: HCM's primary imaging features (asymmetric
septal hypertrophy, systolic anterior motion of the mitral valve,
dynamic LVOT gradient) are best visualised in PLAX. UHN's disease_hcm
probe also uses PLAX + PSAX-PM + PSAX-MV + A4C; PLAX-only matches the
single-view protocol used for the other NeurIPS tasks.

**Builder**: `experiments/nature_medicine/mimic/probe_csvs/build_disease_hcm_plax.py`
**Output**: `experiments/nature_medicine/mimic/probe_csvs/disease_hcm_plax/{train,val,test}.csv`

| Split | Clips (pre → post) | Studies (pre → post) | Positive studies | Positive clips |
|---|---|---|---:|---:|
| train | 237,071 → **35,884** | 3,226 → **3,148** (−78) | 116 → **114** | 9,443 → 1,338 |
| val | 52,593 → **7,933** | 728 → **701** (−27) | 40 → **37** | 2,961 → 410 |
| test | 56,200 → **8,582** | 760 → **746** (−14) | 23 → **22** | 1,905 → 239 |

**Notes**:
- Study-level split assignment preserved from v4.1 → fully comparable to
  the non-view-filtered HCM run; no cross-split leakage risk.
- 119 studies dropped for zero PLAX clips (6 were HCM-positive:
  2 train / 3 val / 1 test). Slightly better PLAX coverage than A4C
  (141 dropped) but similar positive-study loss.
- Test has **22 positives** — bootstrap AUROC CIs, expect wide bounds.
  Consider merging val+test if a headline result is needed.
- Median clips/study drops from ~72 to ~11 (PLAX only). Prediction
  averaging still has plenty to work with.
- Source labels are v4.1; CLAUDE.md says v4.3 is current with ICD-9
  fixes for amyloidosis/takotsubo — HCM was not affected, so v4.1 is
  still the right seed.

---

## 2. Incident HF detection (1-year) — A4C

**Label**: HF hospital admission within 365 days **after** echo study.
- Positive = ICD-10 I50.x or ICD-9 428.x, `admittime ∈ (study_datetime, study_datetime + 365d]`
- Eligible = no HF admission with `admittime ≤ study_datetime` (HF-naive at echo)
- Negative = eligible AND no HF admission within 365d (includes
  right-censored patients — standard caveat for ICD-based incident
  labels from MIMIC)
- Patients with prevalent HF dropped outright (2,750 studies).

**Builder**: `experiments/nature_medicine/mimic/probe_csvs/build_incident_hf_a4c.py`
**Output**: `experiments/nature_medicine/mimic/probe_csvs/disease_hf_incident_1yr_a4c/{train,val,test}.csv`

**Cohort**:

| | Total studies | Eligible | Positive (1y) | Prevalence |
|---|---:|---:|---:|---:|
| Total | 7,243 | 4,493 | 352 | 7.8% |
| Train | — | 3,112 | 246 | 7.9% |
| Val | — | 670 | 62 | 9.3% |
| Test | — | 618 | 37 | 6.0% |

**Clip counts**:
- train — **33,579 clips / 3,112 studies / 246 positive**
- val — **7,276 clips / 670 studies / 62 positive**
- test — **6,487 clips / 618 studies / 37 positive**

Plus `label_meta.json` with full provenance (label definition, cohort
counts, per-split stats).

**Notes**:
- Patient-level splits reused from `disease_hf_v4.1` → zero subject
  overlap between train/val/test (verified).
- ICD under-coding caveat: HF ICD sensitivity is ~73.5% vs discharge-note
  truth per CLAUDE.md. Label has ~25% false-negative rate baked in —
  biases AUROC down but keeps signal.
- Right-censoring treated as negative: a patient whose last admission in
  MIMIC is <1y after echo is labeled negative even if we simply lack
  follow-up. No survival-analysis censoring; if rigorous framing is
  needed, `days_to_hf` in the JSON could be extended to emit a censoring
  indicator.
- Test positives: 37. AUROC CIs will be ~±0.05–0.08 at bootstrap
  n=2000. Fine for a headline but **no subgroup analysis** here.
- Every clip in v4.1 splits matched a row in `echo_study_list` — no
  unmapped study_ids.

---

## 3. Age regression — A4C

**Label**: `age_at_echo` (years) from `demographics_fairness.csv` (100%
coverage for all 7,243 studies; anchor_age + (echo_year − anchor_year)).

**Builder**: `experiments/nature_medicine/mimic/probe_csvs/build_age_a4c.py`
**Output**: `experiments/nature_medicine/mimic/probe_csvs/age_a4c/{train,val,test}.csv`

**Format**: space-delimited `<s3_path> <raw_age_float>` (matches other
regression tasks: creatinine, lvef, etc.). Use `zscore_params.json`
(mean=70.5107, std=12.7812 from train clips) for normalization.

| Split | Clips | Studies | Age mean ± std | Range |
|---|---:|---:|---:|---:|
| train | 52,964 | 4,910 | 70.5 ± 12.8 | 22–91 |
| val | 11,751 | 1,078 | 70.4 ± 13.1 | 24–91 |
| test | 11,170 | 1,053 | 71.0 ± 13.4 | 29–91 |

**Notes**:
- Study counts higher than HCM/HF splits because no studies are dropped
  for prior-disease eligibility — this is the full MIMIC echo cohort
  minus studies with zero A4C clips (~2% drop).
- **Age ceiling at 91**: MIMIC deidentifies all patients >89 to age 91.
  ~6% of studies sit at this ceiling — predictions >89 are clipped by
  design. Mention this in methods if used as a headline task; otherwise
  the probe will look like it's over-predicting old patients who are
  actually ≥92.
- Patient-level split integrity inherited from `disease_hf_v4.1`
  (verified in HF build).

---

## 4. LVEF regression — A4C

**Label**: structured echo measurement "LVEF" (TTE + B-mode + value
range [10, 85] + ±1-day matching window, not note-extracted). Labels
carried over verbatim from `lvef_structured`; no re-derivation.

**Builder**: `experiments/nature_medicine/mimic/probe_csvs/build_lvef_a4c.py`
**Output**: `experiments/nature_medicine/mimic/probe_csvs/lvef_a4c/{train,val,test}.csv`

**Format**: space-delimited `<s3_path> <raw_lvef_float>`. Z-score params
re-derived on A4C-filtered train (mean=54.6569, std=13.1159) —
**use the new `zscore_params.json`, not the source one** (source was
54.74/12.93 on A2C+A4C).

| Split | Clips | Studies | LVEF mean ± std | Range |
|---|---:|---:|---:|---:|
| train | 19,138 | 3,065 | 54.7 ± 13.1 | 10–82 |
| val | 4,350 | 683 | 55.0 ± 13.0 | 10–80 |
| test | 3,961 | 653 | 55.1 ± 13.4 | 10–85 |

**Notes**:
- A2C dropped: source had 27,771 clips across A2C+A4C in train; A4C
  narrowing removes 8,633 A2C clips. Expect slightly **lower MAE** than
  the 2-view version since A4C is the dominant LVEF-imaging view.
- Few studies dropped for no-A4C (19 train / 5 val / 6 test) — the
  source was already view-filtered so almost every study already had A4C
  clips.
- Median clips/study drops to 6 (vs 11 in HCM/HF A4C splits) because the
  source already filtered to measurement-matched clips.

---

## 5. 30-day all-cause mortality — A4C

**Label**: binary death within 30 days of echo study (1 = dead within 30d,
0 = alive at 30d). Source: `data_exploration/mimic/csv/mortality_30d.csv`
(prebuilt by `build_all_labels.py`). 100% coverage of the 7,243 echo
cohort; full-cohort prevalence 5.7% (411 positive / 6,832 negative).

**Builder**: `experiments/nature_medicine/mimic/probe_csvs/build_mortality_30d_a4c.py`
**Output**: `experiments/nature_medicine/mimic/probe_csvs/mortality_30d_a4c/{train,val,test}.csv`

| Split | Clips | Studies | Pos studies | Pos clips | Prevalence |
|---|---:|---:|---:|---:|---:|
| train | **52,964** | 4,910 | **285** | 2,823 | 5.8% |
| val | **11,751** | 1,078 | **52** | 454 | 4.8% |
| test | **11,170** | 1,053 | **42** | 410 | 4.0% |

Median 11 clips/study. Zero studies unlabelled (mortality_30d covers
100% of the cohort).

### 10k subset (A4C)

**Builder**: `experiments/nature_medicine/mimic/probe_csvs/build_mortality_30d_a4c_10k.py`
**Output**: `experiments/nature_medicine/mimic/probe_csvs/mortality_30d_a4c_10k/{train,val,test}.csv`

Stratified random sample on binary class, proportional allocation,
seed=42. Scale factor **10,000 / 52,964 = 0.1888** applied to val/test.

| Split | Clips | Pos clips | Neg clips | Studies | Pos studies |
|---|---:|---:|---:|---:|---:|
| train | **10,000** | 533 | 9,467 | 4,158 | 225 |
| val | **2,219** | 86 | 2,133 | 903 | 37 |
| test | **2,109** | 77 | 2,032 | 903 | 39 |

Prevalence preserved (train 5.3%, val 3.9%, test 3.7%). Every positive
study is down-weighted but proportionally represented; no study entirely
dropped by design of stratified sampling.

**Notes**:
- Patient-level partition reused from `disease_hf_v4.1` — zero subject overlap across train/val/test.
- Prior benchmark numbers on MIMIC 30d mortality (for reference, different probe protocols):
  - CY sklearn ensemble (mean-pooled features, 230K clip-level rows): EchoJEPA-G 0.912, L-K 0.905, EchoPrime 0.893, PanEcho 0.849 — manuscript gold standard
  - Strategy E d=1 attentive (study-sampling, ~3.2k clip-level training rows): G 0.884, L-K 0.878
  - EHR baselines: XGBoost (54 features) 0.921, Elixhauser 0.888, LVEF-only 0.519
- Test positives (full): 42 studies / 410 clips. (10k: 39 studies / 77 clips.) AUROC CIs ~±0.05 at bootstrap n=2000. Adequate for headline number; not enough for subgroup analysis.
- No "prior mortality" eligibility filter needed — mortality is an event-time endpoint, not a prevalent-condition label.
- `label_meta.json` in each output dir has full provenance.

---

## 6. Qualitative RV function — A4C (5-class full + binary 10k)

**Label**: 5-class qualitative RV function from MIMIC
`echo_structured_measurement.rv_function` (TTE, non-stress), matched to
each echo study within ±1 day (closest measurement wins), mapped to the
UHN 5-class scheme:

| Class | Label | MIMIC result string |
|---|---|---|
| 0 | normal | `Nl RV function` |
| 1 | low_normal | `Low normal function` |
| 2 | mildly_reduced | `Mild global RV hypo` |
| 3 | moderately_reduced | `Moderate global RV hypo` |
| 4 | severely_reduced | `Severe global hypo`, `RV function depressed` |

Ambiguous/focal strings dropped (`RV not well seen`, `Cannot assess RV
function`, `Apical free wall hypo`, `Basal RV hypo (McConnell's sign)`,
`Hyperdynamic` — 153 studies total).

**Builder**: `experiments/nature_medicine/mimic/probe_csvs/build_rv_function.py`

**Outputs** (two dirs, both A4C-only):
- `experiments/nature_medicine/mimic/probe_csvs/rv_function_a4c/{train,val,test}.csv` — full 5-class
- `experiments/nature_medicine/mimic/probe_csvs/rv_function_binary_10k_a4c/{train,val,test}.csv` — binary ({0,1}→0, {2,3,4}→1), train downsampled to 10k clips, val/test scaled by same factor

### Full 5-class (A4C)

| Split | Clips | Studies | Class dist (studies, 0→4) |
|---|---:|---:|---|
| train | 50,461 | 4,621 | 3,688 / 239 / 342 / 197 / 155 |
| val | 11,040 | 998 | 791 / 37 / 104 / 32 / 34 |
| test | 10,706 | 994 | 808 / 47 / 65 / 46 / 28 |

Median 11 clips/study. 6,758 studies labelled in total, 289/80/59
unlabelled per split (no rv_function measurement within ±1d, or in the
excluded focal/ambiguous categories).

### Binary 10k (A4C, binarized: {0,1}→0 no-dysfunction, {2,3,4}→1 any-dysfunction)

Scale factor: **10,000 / 50,461 = 0.1982**. Val/test scaled by same
factor and stratified on binary class.

| Split | Clips | Class 0 (no dysfunction) | Class 1 (any dysfunction) | Pos rate |
|---|---:|---:|---:|---:|
| train | **10,000** | 8,482 | 1,518 | 15.2% |
| val | **2,188** | 1,807 | 381 | 17.4% |
| test | **2,122** | 1,815 | 307 | 14.5% |

Stratified random sample, seed=42.

**Notes**:
- **View filter**: A4C only (`view == A4C AND view_status == OK`), consistent with the other 4 NeurIPS tasks. (UHN rv_function probe uses A4C + Subcostal + PLAX; we restrict to A4C here to match the shared A4C-only protocol.)
- Patient-level partition reused from `disease_hf_v4.1` — zero subject overlap across train/val/test.
- "RV function depressed" (~110 studies, unspecified severity) mapped to class 4 (severely reduced) as the conservative interpretation.
- Severe class imbalance in the 5-class task (78% normal). Binary framing is the publishable version; 5-class reported as a secondary / supplementary analysis.
- `label_meta.json` in each output dir has the full raw-string histogram, class map, and build provenance.

---

## 7. LV morphology — PLAX (IVS, LVPW, LVIDd, LVIDs)

Four regression tasks for classical parasternal M-mode-style
measurements. All in cm, all measured in the PLAX view. Clinically
these are the canonical inputs to LV mass / relative-wall-thickness
formulas and LV geometry classification.

**Label source**: `echo_structured_measurement` (TTE, closest
measurement within ±1 day, closest wins), value-range filtered per
task. Patient-level splits reused from `disease_hf_v4.1`.

**View filter**: PLAX only (`view == PLAX` AND `view_status == OK`),
79,742 PLAX clips available in the manifest.

**Builder**: `experiments/nature_medicine/mimic/probe_csvs/build_lv_morphology_plax.py`

### MIMIC name → clinical name

| MIMIC `measurement` | Clinical name | Description | Value range | Unit |
|---|---|---|---|---|
| `septal_thickness` | **IVSd** (interventricular septum, diastolic) | Left Ventricle — Septal Wall Thickness | [0.4, 3.0] | cm |
| `inf_lat_thickness` | **LVPWd** (LV posterior/inferolateral wall, diastolic) | Left Ventricle — Inferolateral Wall Thickness | [0.4, 3.0] | cm |
| `lvedd` | **LVIDd** (LV internal dimension, diastolic) | Left Ventricle — End Diastolic Dimension | [2.0, 8.5] | cm |
| `lvesd` | **LVIDs** (LV internal dimension, systolic) | Left Ventricle — End Systolic Dimension | [1.0, 7.5] | cm |

BIDMC uses ASE post-2015 terminology (inferolateral wall = LVPW).
`lvedd`/`lvesd` are identical to the older LVIDd/LVIDs.

### Full PLAX splits

**Output dirs**: `<task>_plax/{train,val,test}.csv` + `zscore_params.json` + `label_meta.json`

| Task | Train | Val | Test | Mean ± std (cm) | Clips/study (median) |
|---|---:|---:|---:|---|---:|
| `septal_thickness_plax` | 54,115 | 11,648 | 11,693 | 1.10 ± 0.24 | ~12 |
| `inf_lat_thickness_plax` | 54,079 | 11,633 | 11,723 | 1.08 ± 0.22 | ~12 |
| `lvedd_plax` | 54,260 | 11,635 | 11,733 | 4.57 ± 0.80 | ~12 |
| `lvesd_plax` | 44,045 | 9,338 | 9,575 | 3.03 ± 0.81 | ~12 |

`lvesd_plax` has ~20% fewer clips because end-systolic dimension is
measured less consistently than end-diastolic (per the structured
measurements doc, `lvesd` coverage: 3,884 DICOM subjects vs 4,483 for
`lvedd`).

### 10k subsets (quantile-stratified on label, 10 bins, seed=42)

**Output dirs**: `<task>_plax_10k/{train,val,test}.csv` + same
`zscore_params.json` (inherited from parent for consistency) +
`label_meta.json`

| Task | Train | Val | Test | Scale factor |
|---|---:|---:|---:|---:|
| `septal_thickness_plax_10k` | 10,000 | 2,152 | 2,161 | 0.1848 |
| `inf_lat_thickness_plax_10k` | 10,000 | 2,151 | 2,168 | 0.1849 |
| `lvedd_plax_10k` | 10,000 | 2,144 | 2,162 | 0.1843 |
| `lvesd_plax_10k` | 10,000 | 2,120 | 2,174 | 0.2270 |

Stratification preserves label-tail coverage: bin-10 (highest-value
wall thicknesses, LVE dimensions) keeps all or proportional samples
even when small. Use the parent `zscore_params.json` so the probe sees
the same target distribution between subset and full runs.

### Z-score parameters (from PLAX-filtered train clips)

| Task | target_mean | target_std |
|---|---:|---:|
| `septal_thickness_plax` | 1.0976 | 0.2393 |
| `inf_lat_thickness_plax` | 1.0787 | 0.2150 |
| `lvedd_plax` | 4.5697 | 0.7970 |
| `lvesd_plax` | 3.0339 | 0.8121 |

### Notes

- Patient-level partition reused from `disease_hf_v4.1` → zero subject
  overlap between train / val / test.
- The existing non-view-filtered `septal_thickness/` directory
  (45,673 / 9,799 / 9,892 clips) stays on disk for reproducibility of
  the earlier run but should not be used for new experiments — use
  `septal_thickness_plax/` instead (narrower view filter + consistent
  patient splits with the rest of the task list).
- **Clinical utility**: paired with LVEF, these four measurements let
  you compute LV mass (Cube/Devereux formula), relative wall thickness
  (RWT = (IVSd + LVPWd) / LVIDd), and LV geometry class (normal /
  concentric remodeling / concentric LVH / eccentric LVH). Predicting
  them from B-mode video is a direct comparison to conventional echo
  measurements — the kind of task reviewers will find familiar.
- Test sample sizes (~9K–12K clips across ~1K studies) give tight
  bootstrap confidence intervals, ideal for headline Pearson / MAE
  numbers.
- `lvesd` has known higher measurement variability (trained sonographer
  agreement is ~0.7 for LVIDs vs ~0.9 for LVIDd) — expect lower ceiling
  than `lvedd` regardless of model choice.

---

## Common protocol

**View filter**: `view == A4C` AND `view_status == OK` from
`classifier/output/mimic_view_predictions.csv` (525,328 rows,
75,890 A4C-OK clips).

**Probe protocol** (match existing finalbudget / TokenRel probes):
- ViT-L d=4 attentive, 16 heads, `frames_per_clip: 16`, `frame_step: 2`,
  `num_segments: 2`, `num_views_per_segment: 1`, `batch_size: 1`,
  `num_epochs: 20`
- 6-HP grid: lr ∈ {1e-4, 5e-5} × weight_decay ∈ {0.01, 0.1, 0.4}
- bf16, use_pos_embed: false
- Encoder checkpoint_key: `target_encoder`
- Probe best-ckpt selected on val (classification: val_auroc;
  regression: val_r2 or −val_mae)

**Prediction averaging**: classification tasks use clip-level probs
averaged to study level before computing AUROC / acc / Moderate+
binarization. Regression tasks average per-clip continuous predictions
per study and compute MAE / R² / Pearson at the study level.

**Split integrity**: every task uses the patient-level partition from
`disease_hf_v4.1` (or `disease_hcm_v4.1` for HCM, which follows the same
patient allocation logic). Zero subject overlap across train/val/test
verified in each builder.

---

## Status

| Task | Full A4C | 10k trimmed | Probe trained | Test run | Pred averaging |
|---|:---:|:---:|:---:|:---:|:---:|
| HCM PLAX | ✅ | ✅ (447 pos) | — | — | — |
| Incident HF 1y A4C | ✅ | ✅ (957 pos) | — | — | — |
| Age A4C | ✅ | ✅ | — | — | — |
| LVEF A4C | ✅ | ✅ | — | — | — |
| Mortality 30d A4C | ✅ | ✅ (533 pos clips) | — | — | — |
| RV function A4C (5-class) | ✅ | — | — | — | — |
| RV function A4C (binary, 10k) | — | ✅ (1,518 pos) | — | — | — |
| IVS thickness PLAX (septal_thickness) | ✅ | ✅ | — | — | — |
| LVPW thickness PLAX (inf_lat_thickness) | ✅ | ✅ | — | — | — |
| LVIDd PLAX (lvedd) | ✅ | ✅ | — | — | — |
| LVIDs PLAX (lvesd) | ✅ | ✅ | — | — | — |

All 4 task datasets ready in both full and 10k-trimmed train variants.
Probe-training sbatches not yet drafted — natural candidates are all
existing finalbudget / TokenRel probe sbatches with dataset_train /
dataset_val / dataset_test paths swapped to the new A4C CSVs and the
classifier / regression head configs adjusted per task.

**10k trim outputs** (sibling `*_10k/` dirs):

Train is trimmed to ~10k clips; val/test are trimmed to **≤3 clips per
study** (all studies and all positives preserved — only per-study clip
counts reduced for inference speedup).

| Task | Train | Val | Test |
|---|---|---|---|
| disease_hcm_plax | 10,000 clips (447 pos, 2,566 studies) | **2,066 clips / 701 studies / 37 pos studies** | **2,216 clips / 746 studies / 22 pos studies** |
| disease_hf_incident_1yr_a4c | 10,000 clips (957 pos, ~2.3k studies) | **1,989 clips / 670 studies / 62 pos studies** | **1,833 clips / 618 studies / 37 pos studies** |
| age_a4c | 10,000 clips (mean 70.0, quantile-stratified) | **3,190 clips / 1,078 studies** (range 24-91) | **3,112 clips / 1,053 studies** (range 29-91) |
| lvef_a4c | 10,000 clips (mean 52.7, 8 effective bins after 55/60 tie-dedup) | **1,971 clips / 683 studies** (range 10-80) | **1,868 clips / 653 studies** (range 10-85) |

**Per-study clip cap on val/test: 3 clips/study** (prior probes show
prediction averaging saturates at ~3-5 clips). All studies kept, all
positive studies kept, tail-label coverage intact. Val inference drops
from ~9k clips → ~2-3k; test drops ~4×. Combined with the 10k train,
a full 20-ep probe + test run fits in ~2h per task (vs ~10-12h on
full A4C).

Per-task metadata files:
- `<task>_a4c_10k/trim_meta.json` — train-trim provenance
- `<task>_a4c_10k/val_trim_meta.json` — val per-study cap stats
- `<task>_a4c_10k/test_trim_meta.json` — test per-study cap stats

LVEF notes: 8 effective bins (10 requested → collapsed at 55/60 ties).
The under-populated 52-55 bin keeps all 161 available clips; the modal
55-60 bin absorbs the reallocated headroom (2,388 clips).

Trim builder: `experiments/nature_medicine/mimic/probe_csvs/trim_to_10k.py`
(one command runs all 4: `python trim_to_10k.py`; `--task <name>` for
one task; `--skip-valtest` to leave val/test at full A4C size).

**Dual availability** — both full and trimmed CSVs coexist:

| Full A4C (unchanged) | 10k trimmed (new) |
|---|---|
| `disease_hcm_plax/{train,val,test}.csv` | `disease_hcm_plax_10k/{train,val,test}.csv` |
| `disease_hf_incident_1yr_a4c/{train,val,test}.csv` | `disease_hf_incident_1yr_a4c_10k/{train,val,test}.csv` |
| `age_a4c/{train,val,test}.csv` | `age_a4c_10k/{train,val,test}.csv` |
| `lvef_a4c/{train,val,test}.csv` | `lvef_a4c_10k/{train,val,test}.csv` |

Probes can point at either. Use 10k for matched-budget comparison to
EchoNet LVEF / MIMIC RVSP / MIMIC MR; use full A4C for maximum-data
baseline. Seeded identically (seed=42); the 10k subset is deterministic
re-samples of the full parent.

---

## Train-set trimming to ~10k clips (per-task feasibility)

Reference: `echonet_dynamic_train_s3_raw.csv` (the LVEF EchoNet train set
used by all finalbudget / TokenRel probes) is **9,867 clips**. Many
existing MIMIC probes use `mimic_rvsp_sv_train_10k.csv` (~10k) and
`mimic_mr_a4c_train_10k.csv` (~10k) — so 10k is the in-session standard
for "matched-budget" MIMIC train sets.

Current A4C train counts and feasibility of trimming to ~10k:

| Task | Train clips | Train studies | Positive studies | Feasibility | Recommended trim |
|---|---:|---:|---:|---|---|
| HCM PLAX | 35,884 | 3,148 | 114 | ✅ Easy | **keep all 114 positives + downsample negatives to ~3k studies × median ~3 clips/study ≈ 9.5k clips** |
| Incident HF 1y A4C | 33,579 | 3,112 | 246 (HF+) | ✅ Easy | **keep all 246 positives + downsample 2,866 negatives to ~2.5k × median ~3 clips/study ≈ 10k clips** |
| Age A4C | 52,964 | 4,910 | — (regression) | ✅ Easy | **sample ~10k clips uniformly across studies; stratify by age quartile to preserve the 22–91 range** |
| LVEF A4C | 19,138 | 3,065 | — (regression) | ⚠️ Already close | **already only ~19k; if trimming needed, sample 10k clips while stratifying on LVEF deciles (preserves tail coverage in [10–35] and [65–85])** |

### Recommended trim strategy per task

**Classification (HCM, Incident HF)**: stratified by label + study.
Keep every positive study's clips (177 HCM / 246 HF are all load-
bearing); downsample negative-study clips to hit the ~10k target while
preserving at least ~3 clips/study for prediction averaging. This
maintains both class balance at the study level and multi-clip averaging
at test time.

**Regression (Age, LVEF)**: decile-stratified sampling on the label
distribution. Simple uniform sampling would over-represent the modal age
(~70) or LVEF (~55-65, the "normal" cluster) and hurt tail prediction.

### Caveats on trimming

1. **Statistical power**: at 10k clips and ~3 clips/study, a trimmed
   train set has ~3.3k studies per task. HCM train goes from 114 → 114
   positive studies (unchanged) but only ~2.9k negatives; class ratio
   widens from 1:27 to 1:26 — negligible effect. Incident HF ratio goes
   from 1:12.6 to ~1:11 — also fine. The only statistical concern is
   **LVEF regression tail coverage**: the [10–35] bucket has <100 clips
   in full train; sampling needs to guarantee those are all kept.

2. **Matched-budget comparisons**: if the goal is direct comparison to
   EchoNet LVEF (9,867 train clips) or MIMIC RVSP/MR 10k variants, the
   trimmed versions go in `<task>_a4c_10k/` dirs alongside the full
   `<task>_a4c/`. Keep both so we can run matched-budget *and* max-data
   variants.

3. **Prediction averaging assumption**: training clips and val/test
   clips are per-study-sampled. Trimming train has no downstream effect
   on val/test — those stay at their full A4C clip counts (reported
   above). The trim is strictly a train-side efficiency choice.

4. **Runtime impact**: a 10k train set at bs=1 × 8 GPUs × 20 epochs ≈
   25,000 steps × ~0.5 s/step ≈ ~3.5 hours per task (vs ~10-12 hours
   for the full 33k train set). A 4-task × 6-HP grid at 10k train would
   fit a single 24-hour node run.

5. **What this does NOT fix**: test-set positive counts. Incident HF
   test has 37 positives; HCM test has 21. Trimming train doesn't
   change test; those headline-number CIs still need bootstrap.

### Implementation sketch

```python
# experiments/nature_medicine/mimic/probe_csvs/trim_to_10k.py
# Reads <task>_a4c/train.csv, writes <task>_a4c_10k/train.csv
# (val/test unchanged).
#
# Strategy:
#   if task in {'disease_hcm_plax', 'disease_hf_incident_1yr_a4c'}:
#       keep all positive-study clips
#       random.sample negative studies to hit (10k - n_pos_clips) target
#       random.sample clips per kept study (cap at e.g. 4)
#   if task in {'age_a4c', 'lvef_a4c'}:
#       decile-stratify on the label
#       sample ~10k/10 = 1k clips per decile
#       ensure all tail-decile clips kept if tail has <1k available
#
# Output: train_10k.csv + trim_meta.json with per-decile / per-label counts
```

This can be a ~50-line script; all four A4C dirs get a sibling
`train_10k.csv` plus a `trim_meta.json`. Val and test are left
untouched.

---

## Cross-references

- Source builders: `experiments/nature_medicine/mimic/probe_csvs/build_<task>_a4c.py`
- View manifest: `classifier/output/mimic_view_predictions.csv` (ConvNeXt view predictions)
- Source labels:
  - HCM: `disease_hcm_v4.1/` (MIMIC ICD-based; see `uhn_echo/nature_medicine/context_files/data-auditing.md`)
  - Incident HF: built fresh from `mimic.db` `hosp_diagnoses_icd` (ICD-10 I50.x + ICD-9 428.x); patient splits from `disease_hf_v4.1/`
  - Age: `demographics_fairness.csv`
  - LVEF: `lvef_structured/` (structured measurement, not note-extracted)
- Per-task `label_meta.json` or `viewfilter_meta.json` in each output dir captures full build provenance.

---

## Additional structured-measurement candidates (MIMIC `echo_structured_measurement`, TTE only)

Inventory (2026-05-04) of `mimic.db / echo_structured_measurement` filtered to
`test_type='tte'`. Counts are non-null clips; each measurement already has
`measurement_datetime` so all can be ±1d-window-matched to a study like our
existing lvef/tapse/rv_function builders. View filtering comes from the
ConvNeXt manifest.

### Strongest regressions (continuous)

| `measurement` | non-null clips | range hint | Notes |
|---|---:|---|---|
| `lvef` | built | 10–85% | already shipped |
| `biplane_lvef` | 68,708 | % | alternate LVEF from biplane Simpson's |
| `tapse` | 39,204 | 0.5–4.0 cm | already shipped |
| `tr_mmhg` | 133,621 | mmHg | **RVSP**, already shipped |
| `tr_velocity` | 133,278 | m/s | TR peak velocity (feeds RVSP) |
| `la_vol` | 71,130 | mL | LA volume — fast regression |
| `lvedd` | 165,110 | cm | LV end-diastolic dim |
| `lvesd` | 123,920 | cm | LV end-systolic dim |
| `septal_thickness` | 164,147 | cm | **HCM-relevant**; strong on PLAX |
| `rv_diam` | 84,629 | cm | RV basal diameter |
| `av_pk_vel` | 158,978 | m/s | AV peak velocity — **AS grading** |
| `av_pk_grad` | 36,247 | mmHg | AV peak gradient |
| `av_mean_grad` | 32,993 | mmHg | AV mean gradient |
| `av_area_continuity` | 24,388 | cm² | **AVA (continuity eq.)** — AS gold-standard |
| `sept_e_prime` | 123,788 | m/s | septal tissue Doppler — **diastolic** |
| `lat_e_prime` | 125,079 | m/s | lateral tissue Doppler |
| `mv_peak_e_a` | 139,797 | ratio | E/A — diastolic function |

### Strongest ordinal classifications

| `measurement` | non-null clips | useful classes | Notes |
|---|---:|---:|---|
| `phtn_severity` | 161,941 | 4 (Normal/Mild/Mod/Sev) | **PHTN grading**; multi-view signal |
| `lv_wall_thickness` | 165,906 | 4 (Normal/Mild/Mod/Sev LVH) | **HCM-adjacent**; PLAX-dominant |
| `ra_size` | 136,129 | 4 (Normal/Mild/Mod/Sev RAE) | 4-class RA enlargement |
| `la_size` | 161,651 | 4 (Normal/Mild/Mod/Sev LAE) | 4-class LA enlargement |
| `rv_chamber_size` | 169,321 | 4 (Normal/Mild/Mod/Sev RVE) | 4-class RV enlargement |
| `lv_chamber_size` | 90,136 | 4 (Normal/Mild/Mod/Sev LVE) | 4-class LV enlargement |
| `rv_function` | built | 5 | already shipped (A4C 5-class + binary 10k) |
| `tricuspid_regurg` | 161,208 | 4+ (Trivial/Mild/Mod/Sev) | **TR grading** |
| `aortic_regurg` | 62,001 | 4 (Trace/Mild/Mod/Sev) | **AR grading** |
| `aortic_stenosis` | 21,393 | 3 (Mild/Mod/Severe) | **AS grading by AVA** |
| `mitral_stenosis` | 28,599 | mostly "No valvular" | MS rare — hard task |
| `pericardial_effusion` | 37,220 | 4+ (Trivial/Small/Mod/Large) | pericardial effusion grading |
| `mac_severity` | 52,809 | 3 (Mild/Mod/Severe) | mitral annular calcification |
| `diastolic_grade` | 21,584 | 4 (0/I/II/III) | diastolic dysfunction grade |

### Second-tier (usable with caveats)

| `measurement` | non-null | Caveat |
|---|---:|---|
| `diastolic_fcn` | 41,634 | narrative strings (E/e′<8 etc.); 6 classes |
| `rv_wall_thickness` | 6,685 | binary (RVH vs normal) — small n |
| `tamponade` | 12,742 | 3 classes; strong class imbalance |
| `septal_motion` | 16,165 | 8 abnormal patterns; small n |
| `lv_obstruction` | 144,664 | LVOT obstruction grading (HCM-adjacent) |
| `fs` | 68,708 | fractional shortening (regression) |
| `mv_sam`, `mv_prolapse` | ~180k | binary findings (imbalance) |

### Drop (too few positives or too-narrow label)

`vsd` (386), `lv_aneurysm` (870), `lv_dyssynchrony` (886), `calcified_paps`,
`chordal_thickening`.

### Recommended next-tier probe additions

In order of scientific priority:

1. **Aortic stenosis panel** (3 probes):
   - `aortic_stenosis` 3-class (21,393 clips) — severity ordinal
   - `av_area_continuity` regression (24,388 clips) — AVA gold standard
   - `av_pk_vel` regression (158,978 clips) — AV peak velocity
   Classic Doppler-gated AS readout; V-JEPA-Echo 0.908 baseline exists.
   View dependence: PSAX / A5C / PLAX Doppler.
2. **LV wall thickness 4-class** (`lv_wall_thickness`, 165,906) —
   complements HCM PLAX binary; classical cardiomyopathy readout.
3. **PHTN severity 4-class** (`phtn_severity`, 161,941) —
   ordinal multi-class PH grading, same phenomenon as RVSP regression.
4. **Chamber-size quartet** (4-class each):
   `la_size` (161k), `lv_chamber_size` (90k), `ra_size` (136k),
   `rv_chamber_size` (169k). Fast to build, all well-populated.
5. **Diastolic panel** (3 probes):
   - `diastolic_grade` 4-class (21,584) — balanced across 4 classes
   - `sept_e_prime` regression (123,788)
   - `mv_peak_e_a` regression (139,797)
6. **Valvular regurgitation panel**:
   - `tricuspid_regurg` 4-class (161k)
   - `aortic_regurg` 4-class (62k)
7. **Biplane LVEF regression** (`biplane_lvef`, 68,708) — alternate LVEF
   readout using biplane Simpson's.

### Builder notes

- Each task follows the same `build_<task>.py` pattern as
  `build_lvef_a4c.py` / `build_rv_function.py`: query
  `echo_structured_measurement` for the target `measurement`, match to
  study via `measurement_datetime ±1d`, narrow to the view set used by
  the probe (A4C vs multi-view, task-dependent).
- Patient-level split assignment inherited from `disease_hf_v4.1/` for
  every new task — zero subject overlap across train/val/test.
- Z-score params re-derived on the task-specific train split for
  regressions (see `zscore_params.json` convention in existing A4C dirs).
- `label_meta.json` per dir captures cohort counts, class histogram, and
  matching-window provenance.

### View-dependence cheat sheet

- **A4C-best**: LVEF, biplane LVEF, TAPSE, RVSP/TR, RV function, LA/RV
  chamber size, diastolic grade, sept_e_prime, lat_e_prime, mv_peak_e_a.
- **PLAX-best**: LV wall thickness, septal thickness, HCM, LA/LV chamber
  dimension (lvedd, lvesd).
- **PSAX / A5C Doppler**: AS panel (aortic_stenosis, av_area_continuity,
  av_pk_vel, av_pk_grad, av_mean_grad), aortic regurg, pulm stenosis.
- **Multi-view integration**: PHTN severity, valvular regurgitation grade
  (TR, AR, MR), MS grading. These are the strong MV2SV headline tasks.
