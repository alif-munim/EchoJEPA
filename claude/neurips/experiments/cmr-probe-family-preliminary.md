# CMR ACDC probe-family diagnostics — preliminary results

**Status**: ⚠️ **PARTIAL** — job 450 cancelled at 13:20 elapsed at user request. Only the JEPA slow-EMA trajectory completed; **MAE was never run**. Do not treat these results as final.

**Date**: 2026-04-27
**Ask**: Determine if the CMR ACDC attentive-probe conclusion (JEPA > MAE, both insensitive to matched-frame) holds across probe families, or whether it is readout-dependent like EchoNet-Dynamic LVEF.

---

## What completed

| Cell | Variants complete (out of 14) | Status |
|---|---|---|
| JEPA slow-EMA e30 × LVEF | 14/14 | ✅ complete |
| JEPA slow-EMA e30 × DX   | 14/14 | ✅ complete |
| JEPA slow-EMA e100 × LVEF| 14/14 | ✅ complete |
| JEPA slow-EMA e100 × DX  | 14/14 | ✅ complete |
| JEPA slow-EMA e200 × LVEF| 2/14  | partial (raw, diff) |
| JEPA slow-EMA e200 × DX  | 0/14  | not started |
| MAE e30 / e100 / e200 / e300 | 0/14 each | not started |

**58 of 196 planned probe runs completed** (~30%). The MAE row is entirely missing, so head-to-head JEPA↔MAE comparison is not yet possible.

Raw JSON artifacts: `s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts/runs/cmr_probe_sweep_450/jsons/`
Local mirror: `/tmp/cmr_probe_450/` and `/tmp/cmr_probe_preliminary.csv`

---

## Setup

- **Encoders**: ViT-S pretrained on CMR SAX clips (job 346 slow-EMA JEPA, job 183 MAE).
- **Feature cache shape**: `[N, S=1, T=8, D=384]`, fp16, single-clip (`num_segments=1`) per-tubelet spatial mean — matches the attentive CMR inference protocol (`cmr_j_lvmf_444`).
- **Tasks**: ACDC LVEF regression; ACDC 5-class DX.
- **Splits**: ACDC train (100 patients, 951 clips) / ACDC test (50 patients, 538 clips). Train/val is a 10% patient-level stratified split (5 bins for LVEF; per-class for DX); 5 seeds per cell.
- **Per-patient aggregation**: clip-level predictions averaged within `acdc_patient{XXX}` before computing metrics. All metrics below are patient-level.
- **Probe types**: unified trainer `scripts/neurips/cmr/probe/cmr_probe_train.py` supporting `raw`, `diff`, `mean`, `single_t0..7`, `tattn_raw`, `tattn_mean_rep`, `tattn_best_rep`.
- **Hyperparams**: linear probes `lr=1e-3, wd ∈ {1e-4, 1e-2}`, 30 epochs, early stop (patience 5, min_epochs 10); tattn `lr ∈ {1e-4, 3e-4, 1e-3}, wd ∈ {1e-4, 1e-2}`, 50 epochs.
- **Loss**: MSE for LVEF (z-score normalized); cross-entropy for DX (5-class).
- **Metrics reported**: LVEF — R², Pearson; DX — AUROC (macro OVR), accuracy, F1 macro.

### Checkpoints used (so far)

| tag | S3 path |
|---|---|
| jepa_slowema_e30 | `runs/jepa_cmr_vits_slowema_346/training_folder/e30.pt` |
| jepa_slowema_e100 | `runs/jepa_cmr_vits_slowema_346/training_folder/e100.pt` |
| jepa_slowema_e200 | `runs/jepa_cmr_vits_slowema_346/training_folder/e200.pt` |

Feature caches: `s3://.../features/cmr_prepool/{tag}/{task}_{split}.pt`

---

## Result tables


### LVEF — R² (patient) (patient-level pred averaging, 5 seeds)

| model | variant | R² (patient) | ±std | Pearson |
|---|---|---|---|---|
| jepa_slowema_e100 | raw | -0.3119 | 0.1002 | 0.2017 |
| jepa_slowema_e100 | diff | 0.0125 | 0.0130 | 0.3211 |
| jepa_slowema_e100 | mean | -0.1315 | 0.0763 | 0.2055 |
| jepa_slowema_e100 | tattn_raw | -0.0771 | 0.0755 | 0.2825 |
| jepa_slowema_e100 | tattn_mean_rep | -0.0771 | 0.0755 | 0.2823 |
| jepa_slowema_e100 | tattn_best_rep | -0.0754 | 0.0756 | 0.2841 |
| jepa_slowema_e100 | single_t0 | -0.1319 | 0.0731 | 0.2002 |
| jepa_slowema_e100 | single_t1 | -0.1311 | 0.0732 | 0.2022 |
| jepa_slowema_e100 | single_t2 | -0.1319 | 0.0742 | 0.2027 |
| jepa_slowema_e100 | single_t3 | -0.1320 | 0.0736 | 0.2042 |
| jepa_slowema_e100 | single_t4 | -0.1165 | 0.0548 | 0.2081 |
| jepa_slowema_e100 | single_t5 | -0.1326 | 0.0783 | 0.2055 |
| jepa_slowema_e100 | single_t6 | -0.1348 | 0.0788 | 0.2069 |
| jepa_slowema_e100 | single_t7 | -0.1321 | 0.0785 | 0.2125 |
| jepa_slowema_e200 | raw | -0.1761 | 0.1724 | 0.2662 |
| jepa_slowema_e200 | diff | 0.0068 | 0.0152 | 0.3179 |
| jepa_slowema_e30 | raw | -0.1410 | 0.1085 | 0.3359 |
| jepa_slowema_e30 | diff | 0.0155 | 0.0332 | 0.3427 |
| jepa_slowema_e30 | mean | -0.1069 | 0.0380 | 0.2869 |
| jepa_slowema_e30 | tattn_raw | -0.0517 | 0.1116 | 0.3422 |
| jepa_slowema_e30 | tattn_mean_rep | -0.0522 | 0.1114 | 0.3418 |
| jepa_slowema_e30 | tattn_best_rep | -0.0595 | 0.0998 | 0.3396 |
| jepa_slowema_e30 | single_t0 | -0.1040 | 0.0417 | 0.2877 |
| jepa_slowema_e30 | single_t1 | -0.1056 | 0.0386 | 0.2875 |
| jepa_slowema_e30 | single_t2 | -0.0890 | 0.0581 | 0.2950 |
| jepa_slowema_e30 | single_t3 | -0.1095 | 0.0387 | 0.2864 |
| jepa_slowema_e30 | single_t4 | -0.1070 | 0.0367 | 0.2870 |
| jepa_slowema_e30 | single_t5 | -0.1072 | 0.0350 | 0.2858 |
| jepa_slowema_e30 | single_t6 | -0.0928 | 0.0069 | 0.2594 |
| jepa_slowema_e30 | single_t7 | -0.0901 | 0.0095 | 0.2604 |

### DX — AUROC (patient) (patient-level pred averaging, 5 seeds)

| model | variant | AUROC (patient) | ±std | acc |
|---|---|---|---|---|
| jepa_slowema_e100 | raw | 0.7106 | 0.0115 | 0.3600 |
| jepa_slowema_e100 | diff | 0.7626 | 0.0120 | 0.4000 |
| jepa_slowema_e100 | mean | 0.7151 | 0.0149 | 0.3800 |
| jepa_slowema_e100 | tattn_raw | 0.7136 | 0.0116 | 0.3720 |
| jepa_slowema_e100 | tattn_mean_rep | 0.7137 | 0.0115 | 0.3720 |
| jepa_slowema_e100 | tattn_best_rep | 0.7101 | 0.0137 | 0.3720 |
| jepa_slowema_e100 | single_t0 | 0.7188 | 0.0144 | 0.3840 |
| jepa_slowema_e100 | single_t1 | 0.7183 | 0.0146 | 0.3840 |
| jepa_slowema_e100 | single_t2 | 0.7176 | 0.0144 | 0.3760 |
| jepa_slowema_e100 | single_t3 | 0.7165 | 0.0143 | 0.3720 |
| jepa_slowema_e100 | single_t4 | 0.7175 | 0.0163 | 0.3760 |
| jepa_slowema_e100 | single_t5 | 0.7132 | 0.0146 | 0.3800 |
| jepa_slowema_e100 | single_t6 | 0.7145 | 0.0163 | 0.3760 |
| jepa_slowema_e100 | single_t7 | 0.7148 | 0.0154 | 0.3840 |
| jepa_slowema_e30 | raw | 0.7165 | 0.0109 | 0.4000 |
| jepa_slowema_e30 | diff | 0.7682 | 0.0094 | 0.4240 |
| jepa_slowema_e30 | mean | 0.7234 | 0.0126 | 0.3640 |
| jepa_slowema_e30 | tattn_raw | 0.7162 | 0.0182 | 0.3920 |
| jepa_slowema_e30 | tattn_mean_rep | 0.7166 | 0.0187 | 0.3840 |
| jepa_slowema_e30 | tattn_best_rep | 0.7021 | 0.0272 | 0.3800 |
| jepa_slowema_e30 | single_t0 | 0.7207 | 0.0134 | 0.3600 |
| jepa_slowema_e30 | single_t1 | 0.7238 | 0.0118 | 0.3600 |
| jepa_slowema_e30 | single_t2 | 0.7236 | 0.0134 | 0.3680 |
| jepa_slowema_e30 | single_t3 | 0.7256 | 0.0128 | 0.3600 |
| jepa_slowema_e30 | single_t4 | 0.7234 | 0.0121 | 0.3640 |
| jepa_slowema_e30 | single_t5 | 0.7221 | 0.0127 | 0.3640 |
| jepa_slowema_e30 | single_t6 | 0.7227 | 0.0130 | 0.3640 |
| jepa_slowema_e30 | single_t7 | 0.7230 | 0.0127 | 0.3640 |


---

## Headline observations (JEPA slow-EMA only)

### LVEF

- **All probe families return negative patient-level R²**, across both e30 and e100.
- The attentive probe baseline from job 444 was R²=0.130 (e30), so **every pooled-feature variant underperforms the attentive probe by ~0.15+ R²**. This is suggestive of readout reversal for LVEF.
- **diff is the best linear variant** (R² ≈ +0.01, i.e. barely above 0) — consistent with adjacent-frame motion carrying some residual LVEF signal even on 100-patient train.
- **tattn_raw and tattn_mean_rep are indistinguishable** (R² −0.077 vs −0.077 at e100; −0.052 vs −0.052 at e30). **Pooled temporal-attention gets nothing from a real temporal sequence over the mean-repeat control** — strong evidence that no nonlinear temporal structure is exploitable beyond the static aggregate at this data scale.
- **single_t{k} are all ~equally bad** and clustered near the time-mean. No special phase appears decisive.
- Pearson ρ values (0.20–0.34) are non-trivial, so signal exists — but the probe capacity and training regime cannot convert it into useful R² on 100 train patients.

### DX

- **AUROCs tightly cluster in [0.70, 0.77]** across all 14 variants for both e30 and e100.
- **diff beats raw by a clear margin** (0.77 vs 0.72 at both e30 and e100). Adjacent-frame differences carry more DX signal than raw pooled tokens — unexpected for a task we previously thought was temporally insensitive.
- **single_t{k} ≈ raw ≈ mean**: the full sequence adds almost nothing beyond a single timepoint or the time average. DX is essentially phase-decodable.
- **tattn_raw ≈ tattn_mean_rep ≈ tattn_best_rep** (0.71–0.72) — pooled temporal-attention confirms no real-sequence benefit over mean-repeat.

### Cross-probe consistency

- **No probe family outperforms the attentive baseline for either task** within the completed JEPA cells.
- The attentive DX AUROC was 0.80 (e30) / 0.80 (e100) in job 445. Every pooled-feature variant is ≥ 0.03 lower. Consistent with a readout gap.

---

## What we **cannot** yet conclude without MAE

- Whether **MAE overtakes JEPA under pooled probes** (the "readout reversal" check for CMR).
- Whether **MAE also shows temporal-collapse insensitivity** on CMR, or if it reacts differently.
- Whether the early-peaking JEPA advantage from job 444/445 is probe-specific.
- **Tables 3–5** from the original spec (temporal-collapse drops, JEPA–MAE deltas, per-token metrics) cannot be populated without the MAE row.

---

## Recommendation

1. **Re-launch the sweep scoped to MAE only** (can reuse cached features, `features/cmr_prepool/mae_e{30,100,200,300}/` — but those were **not** extracted either; job 449 completed all 7 × 4 extractions, so **MAE caches exist and are ready to probe**). Expected runtime: ~25–30 min on one node (7 probe-training cells × 14 variants × 5 seeds, same cost per cell as JEPA).
2. Optionally finish JEPA e200 (12 more variant runs × 2 tasks = 24 runs, ~3–4 min).
3. Re-run the comparison tables once MAE is in.

A one-line resubmit with a `MODELS_RESUME` env filter in the sweep sbatch would do it; I did not build that filter in the sbatch — it hardcodes all 7 models. If you want me to resume, I can:
- add a `MODELS_FILTER` env var to `cmr_probe_sweep.sbatch`, or
- just re-submit the full sweep unchanged — it will re-train JEPA cells redundantly but still finish in ~30–40 min.

---

## Caveats

- **Tiny dataset**: 100 train patients, 50 test patients, ACDC-wide. R² variance is high (std 0.08–0.17 in several rows). Negative R² is expected when the probe overfits faster than it generalizes.
- **ViT-S encoders** (D=384) are smaller than the echo ViT-L backbones in prior work; some probe families may underperform as a direct consequence of lower feature dimensionality, not encoder quality.
- **num_segments=1** was chosen to match the attentive CMR baseline exactly; this means there is no clip-level averaging beyond the raw single-clip forward pass.
- **diff outperforming raw on DX** at both JEPA epochs is unexpected and should not yet be reported as a finding — could be a regularization side-effect (adjacent differences have smaller feature magnitudes → effectively different regularization under the same `wd`).
- **Probe training is run on a single GPU** sequentially over variants; total wallclock for the full sweep at this node speed is ~35 min.

---

## Files

- `scripts/neurips/cmr/probe/cmr_probe_train.py` — unified probe trainer (classification + regression)
- `scripts/neurips/cmr/probe/cmr_extract_prepool.sbatch` — feature extraction (all 7 encoders × 4 CSVs)
- `scripts/neurips/cmr/probe/cmr_probe_sweep.sbatch` — sweep orchestrator

---

## Job log

| Job | Purpose | Result |
|---|---|---|
| 446 | Smoke extract (JEPA e30, both tasks) | ✅ 2:42, 4 feature caches on S3 |
| 447 | Smoke probe (JEPA e30 raw, both tasks) | ✅ 1:04, 2 JSONs on S3 |
| 448 | Full extract v1 | ❌ failed at 6:29 — `patch_size` double-passed to VideoMAE ctor |
| 449 | Full extract v2 (fix) | ✅ 13:34, all 28 feature caches on S3 |
| 450 | Full probe sweep | ⏹ cancelled at 13:20, 58/196 JSONs complete |
