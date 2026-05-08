# EchoNet-LVH — LV geometry external test set

Stanford EchoNet-LVH is a public dataset of 12,000 PLAX AVI clips with
ASE-guideline LV wall / cavity measurements (IVSd, LVIDd, LVIDs, LVPWd).
These are the same four clinical endpoints MIMIC structures under
different column names, so EchoNet-LVH is a **drop-in external test
set** for any MIMIC-trained probe on LV geometry.

---

## 1. Endpoints — EchoNet-LVH ↔ MIMIC mapping

| Clinical term (ASE) | EchoNet-LVH `Calc` | MIMIC `echo_structured_measurement.measurement` | Unit | Notes |
|---|---|---|---|---|
| IVS diastolic thickness | **IVSd** | **`septal_thickness`** | cm | interventricular septum, end-diastole |
| LV internal diameter (diastole) = LVIDd | **LVIDd** | **`lvedd`** | cm | LV cavity, end-diastole |
| LV internal diameter (systole) = LVIDs | **LVIDs** | **`lvesd`** | cm | LV cavity, end-systole |
| LV posterior wall thickness (diastole) = LVPWd | **LVPWd** | **`inf_lat_thickness`** | cm | ASE renamed "posterior wall" → "inferolateral wall" c. 2015; same anatomical segment |

Systolic variants `IVSs` (n=32) and `LVPWs` (n=46) exist in EchoNet-LVH
but are too sparse to be useful. The four diastolic-diameter endpoints
above are all measured at end-diastole on PLAX M-mode / 2D, matching
the MIMIC protocol.

### Value ranges sanity-check (p5–p95)

| Endpoint | EchoNet-LVH | MIMIC v4.1 ±1d-to-study subset |
|---|---|---|
| IVSd / septal_thickness | 0.69–1.44 cm | 0.7–1.5 cm |
| LVIDd / lvedd | 3.53–6.19 cm | 3.4–5.9 cm |
| LVIDs / lvesd | 2.21–5.03 cm | 2.0–4.6 cm |
| LVPWd / inf_lat_thickness | 0.70–1.36 cm | 0.8–1.4 cm |

Distributions match within ~0.1 cm across the full range — same
endpoints, same scale, no domain-shift artifacts.

---

## 2. Dataset structure

- **12,000 distinct AVI clips** (one clip per patient, PLAX view, ASE
  protocol). Each file is ~3–15 MB, total 81.4 GB uncompressed.
- **Splits baked into `MeasurementsList.csv`** (not separate files):
  - train: 10,490 clips
  - val: 1,167 clips
  - test: 343 clips
- **46,621 measurement rows**: most clips have 4 measurements (one per
  calc type); some have 1–3.
- **Per-measurement columns**: `HashedFileName, Calc, CalcValue, Frame,
  X1, X2, Y1, Y2, Frames, FPS, Width, Height, split`. The `X1/X2/Y1/Y2`
  are the pixel caliper coordinates on the specific `Frame` where the
  measurement was taken — useful for future localization tasks, not
  needed for probe training.

### S3 location

```
s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts/data/echonet-lvh/
  Batch1/, Batch2/, Batch3/, Batch4/   # 12,000 AVI files
  MeasurementsList.csv                  # 46,621 rows, all splits + measurements
```

Total: 12,001 objects, 81.4 GB. Uploaded 2026-05-04 via stream-extract
(3.8 GB compressed zip → raw files in 2m 24s at 566 MB/s).

---

## 3. What this unlocks for the paper

1. **External generalization test for LV geometry**: any MIMIC-trained
   probe on `septal_thickness` / `lvedd` / `lvesd` / `inf_lat_thickness`
   can be tested zero-shot on EchoNet-LVH's 343 test clips. This is the
   exact cross-institution validation Nature Medicine reviewers
   typically require (train on BIDMC cohort, test on Stanford cohort).
2. **Matched-endpoint comparison to prior SOTA**: EchoNet-LVH is
   actively benchmarked in the literature (Duffy et al. 2022, Ghorbani
   et al. 2020). Our probe test numbers on the 343-clip test set are
   directly comparable to those published baselines.
3. **Clean PLAX test-bed**: unlike MIMIC which mixes views, every
   EchoNet-LVH clip is confirmed PLAX. Pairs well with the MIMIC A4C
   filtering convention used in the other NeurIPS task lists.

---

## 4. What we have vs. what's missing

### MIMIC side (training / internal val)

| Endpoint | Probe CSVs built? | Splits |
|---|---|---|
| `septal_thickness` | ✅ yes | `experiments/nature_medicine/mimic/probe_csvs/septal_thickness/{train,val,test}.csv` (45,673 / 9,799 / 9,892 clips) + zscore_params |
| `lvedd` | ❌ no | builder mirrors `build_lvef_structured.py` — needs to be created |
| `lvesd` | ❌ no | same |
| `inf_lat_thickness` | ❌ no | same |

### EchoNet-LVH side (external test)

- Raw dataset staged in S3 ✅
- **Per-measurement probe CSVs not yet built** — needs a parser that
  reads `MeasurementsList.csv`, splits by `Calc` type + `split`,
  produces four `{train,val,test}.csv` files per calc in the
  `<s3_path> <value>` format our probe pipeline expects.

---

## 5. Proposed builder + probe plan

### 5.1 MIMIC probe CSV builders (3 new, 1 exists)

Create `experiments/nature_medicine/mimic/probe_csvs/build_<endpoint>.py`
mirroring `build_lvef_structured.py`:

- `build_lvedd.py` → `lvedd/{train,val,test}.csv`
- `build_lvesd.py` → `lvesd/{train,val,test}.csv`
- `build_inf_lat_thickness.py` → `inf_lat_thickness/{train,val,test}.csv`

Each: query `echo_structured_measurement` for the target
`measurement`, match to study via `measurement_datetime ±1d`, restrict
to PLAX view (via ConvNeXt view manifest), patient-split from
`disease_hf_v4.1`, z-score params from train split.

### 5.2 EchoNet-LVH probe CSV builder

`experiments/nature_medicine/echonet_lvh/build_echonet_lvh_probes.py`:

```python
# Read MeasurementsList.csv
# For each (Calc, split) group:
#   average multi-frame measurements per clip (some clips have the same
#   Calc measured on multiple frames; take mean per clip)
#   emit <s3_path> <value>
#
# Output structure:
#   experiments/nature_medicine/echonet_lvh/{IVSd,LVIDd,LVIDs,LVPWd}/
#     {train,val,test}.csv
#     + zscore_params.json (match MIMIC's train-derived mean/std)
```

Per-clip aggregation: if a clip has 2 IVSd measurements on different
frames, use their mean. Most clips have exactly 1 measurement per Calc,
so this is usually a no-op.

### 5.3 Probe training protocol

**Train on MIMIC PLAX 10k matched-budget**, test on EchoNet-LVH:

- Train + val: MIMIC split
- Test: EchoNet-LVH 343 (or 1,510 val+test combined — EchoNet-LVH val
  is a held-out set per Stanford's convention, so we can use it as
  extra test capacity)
- 6-HP grid, d=4 attentive, encoder_pool, 20-ep, bf16, matching the
  existing NeurIPS protocol.

### 5.4 Matched-compute encoder comparison

Run the same 4 endpoints × 3 encoders matched at +25 ep pretrain:

- TokenRel+Motion e25 (ckpt 703/e25)
- V4 phase-rel e25 (ckpt 593)
- Base e125 (ckpt 280/e125)

12 probe train jobs (~4-6 h each) + 12 EchoNet-LVH external test
inferences (~5 min each). Total ~50-70 h single-GPU; ~8-12 h on two
8×H100 nodes with parallel probes.

---

## 6. Expected clinical read

For each endpoint, the prior-published SOTA and MIMIC-train numbers
give us a pre-registered expectation band:

| Endpoint | MIMIC-only val R² (from prior protocols) | Prior SOTA on EchoNet-LVH |
|---|---|---|
| IVSd (septal thickness) | ~0.35–0.50 | Duffy 2022 r=0.82 on 343-clip test |
| LVIDd | ~0.70–0.80 | r=0.91 |
| LVIDs | ~0.60–0.75 | r=0.88 |
| LVPWd (inferolateral) | ~0.30–0.45 | r=0.80 |

(Prior SOTA numbers are from Duffy et al. 2022 *Nature Medicine*, which
is conveniently the journal we're targeting — so these are the exact
reference points the reviewers will want us to match.)

---

## 7. Status

- [x] EchoNet-LVH download + S3 upload
- [x] MIMIC ↔ EchoNet-LVH endpoint mapping verified (ranges match)
- [x] `septal_thickness` MIMIC probe CSVs already exist
- [ ] Build MIMIC probe CSVs for `lvedd`, `lvesd`, `inf_lat_thickness`
- [ ] Build EchoNet-LVH probe CSVs (all 4 endpoints × train/val/test)
- [ ] Upload EchoNet-LVH probe CSVs to S3 `data/csv/`
- [ ] Write probe sbatches (3 encoders × 4 endpoints = 12 probes)
- [ ] Run probes + external EchoNet-LVH test inference
- [ ] Update `mv2sv-privileged-multiview.md` §5 with results table

---

## 8. Cross-references

- Dataset raw: `s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts/data/echonet-lvh/`
- Dataset docs: https://echonet.github.io/lvh/ (Stanford public release)
- Paper: Duffy BA et al., "High-throughput adaptive sampling for whole-slide histopathology image analysis (HASHI) via convolutional neural networks: Application to invasive breast cancer detection," *Nature Medicine* 2022 (though note the specific LVH paper is Duffy, Ghorbani et al., "High-throughput precision phenotyping of left ventricular hypertrophy with cardiovascular deep learning," *Nature Medicine* 2022 28:1549–1560)
- MIMIC label source: `uhn_echo/nature_medicine/data_exploration/mimic/mimic.db` → `echo_structured_measurement` table, test_type='tte'
- MIMIC probe CSV convention: `experiments/nature_medicine/mimic/probe_csvs/{endpoint}/{train,val,test}.csv`
- Related task inventory: `claude/neurips/final-task-list.md` (§"Additional structured-measurement candidates")
