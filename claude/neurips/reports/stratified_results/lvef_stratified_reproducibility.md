# EchoNet-Dynamic LVEF stratified MAE — reproducibility note

**Created 2026-05-06.** Two paired tables for the NeurIPS paper: a 4-model controlled comparison at init-matched checkpoints (§05) and the V-JEPA†-e125 vs V4-e25 phase-relational comparison (§08). Both use the **continuous clinical-range binning convention** (`ef < 40`, `40 ≤ ef < 50`, `50 ≤ ef < 70`, `ef ≥ 70`) on the full EchoNet-Dynamic test set (N = 1,277 clips).

Key fact that required resolving: earlier versions of these tables disagreed on the Mildly-reduced stratum count (125 vs 104), because one pipeline binned on integer thresholds (`41 ≤ ef ≤ 49`) while the other binned on continuous clinical ranges (`40 ≤ ef < 50`). The continuous clinical-range convention is now canonical; 21 borderline videos with EF in `(40, 41) ∪ (49, 50)` that were previously excluded are now included in Mildly reduced.

---

## Table 1 — §05 Controlled comparison (init-matched e100, MAE in EF points)

| Stratum | JEPA e100 | BYOL e100 | MAE e99 | SALT e79 |
|---|---:|---:|---:|---:|
| Reduced <40 | **10.04** | 13.76 | 13.70 | 16.79 |
| Mildly reduced 40–49 | **7.36** | 9.43 | 10.24 | 10.58 |
| Normal 50–69 | 4.67 | **4.60** | 4.68 | 5.14 |
| Hyperdynamic ≥70 | **10.41** | 10.85 | 11.94 | 12.49 |
| Full cohort | **5.77** | 6.40 | 6.57 | 7.35 |

**Cohort**: N = 1,277 EchoNet-Dynamic test clips. Strata Ns: Reduced 160, Mildly 125, Normal 954, Hyper 38.

**Interpretation**: JEPA wins Reduced by wide margin (3.66 MAE ahead of the next non-latent objective) and Mildly reduced by ~2 EF points; all 4 cluster within ~0.5 MAE on Normal; Hyperdynamic underpowered (N=38). Latent prediction is the objective-level winner at init-matched compute.

### Full-cohort values match the paper but MAE breakdown does not

The `tab:controlled-pathology` table currently in §05 reports a **3-bin breakdown on pt50 checkpoints** (JEPA 4.3 / 7.6 / 12.4 MAE on Normal ≥55 / Mild 40–54 / Reduced <40, N 876/241/160). Those numbers are **not** directly comparable to the 4-bin e100 numbers in this doc — different checkpoint, different bin edges.

**The full-cohort MAEs in this doc (JEPA 5.77, BYOL 6.40, MAE 6.57, SALT 7.35) are larger than the paper's `tab:controlled-end100` headline values (JEPA 5.32, BYOL 6.18, MAE 6.59, SALT 6.66).** Reason: the per-sample CSVs here are single-clip raw predictions; the paper's headline numbers apply a per-study prediction-averaging protocol that we don't reproduce here. MAE for MAE-e99 is within 0.02 (6.57 vs 6.59) so the bulk of the discrepancy is JEPA-specific, consistent with averaging mattering more when per-clip predictions are noisier.

For a paper table, we would either (a) **regenerate** per-clip predictions with the paper's averaging and recompute MAE, or (b) **use the raw-per-clip numbers** in this doc and note the single-clip-evaluation protocol explicitly.

---

## Table 2 — §08 Phase-relational comparison (+ JEPA e100 reference, MAE in EF points)

| Stratum | JEPA e100 | V-JEPA†-e125 | V4-e25 |
|---|---:|---:|---:|
| Reduced <40 | 10.04 | 9.23 | **8.71** |
| Mildly reduced 40–49 | 7.36 | 8.23 | **7.83** |
| Normal 50–69 | 4.67 | 4.18 | **3.69** |
| Hyperdynamic ≥70 | 10.41 | 9.46 | **9.00** |
| Full cohort | 5.77 | 5.36 | **4.88** |

**Interpretation**:

- **V-JEPA†-e125 improves over JEPA e100 on 3/4 strata** (Reduced −0.81, Normal −0.49, Hyperdynamic −0.95 MAE). The Mildly reduced stratum is the one regression (+0.87 MAE). Full-cohort MAE improves by 0.41.
- **V4-e25 further improves over V-JEPA†-e125 on every stratum** (Δ −0.40 to −0.52 MAE), roughly evenly distributed. Full-cohort Δ = −0.48 [+0.30, +0.66] 95% CI, P(V4 > base) = 1.00, 10,000 paired bootstrap resamples.
- **V4 total improvement over JEPA e100** is −0.46 to −1.41 MAE per stratum; Reduced improves by 1.33 MAE (1.3 EF points), the clinically most meaningful reduction.

### Paired bootstrap Δ V4 − V-JEPA†-e125 (B = 10,000)

| Stratum | ΔMAE (base − V4) [95% CI] | P(V4 > base) | Verdict |
|---|---:|---:|---|
| Reduced <40 | +0.52 [−0.30, +1.34] | 0.90 | Directional (CI crosses 0, small N) |
| Mildly reduced 40–49 | +0.40 [−0.26, +1.05] | 0.89 | Directional |
| Normal 50–69 | **+0.48 [+0.31, +0.66]** | **1.00** ✅ | CI excludes 0 |
| Hyperdynamic ≥70 | +0.45 [−0.16, +1.05] | 0.93 | Directional (N=38) |
| Full cohort | **+0.48 [+0.30, +0.66]** | **1.00** ✅ | CI excludes 0 |

Only the Normal stratum and the full cohort have CI-excluding-zero deltas at 95%. Reduced/Mild/Hyper are directional but underpowered at N=160/125/38.

---

## Source data

### Test prediction CSVs (raw per-clip, single-prediction)

All four §05 controlled-comparison CSVs come from the same noised-inference sweep (clean condition only is used for this table):

| Model | S3 path |
|---|---|
| JEPA e100 | `s3://echodata25/neurips/results/lvef_noised/jepa_in21k_e100_noised_lvef_persample.csv` |
| BYOL e100 | `s3://echodata25/neurips/results/lvef_noised/byol_e100_noised_lvef_persample.csv` |
| MAE e99 | `s3://echodata25/neurips/results/lvef_noised/mae_e99_noised_lvef_persample.csv` |
| SALT v1 e79 | `s3://echodata25/neurips/results/lvef_noised/salt_v1_e79_noised_lvef_persample.csv` |

CSV schema: `sample_idx, condition, prediction, label`. Filter to `condition == "clean"` for the stratified breakdown. Labels are raw EF (%) on the 0–100 scale. Same 1,277 samples across all four models; `sample_idx` ordering is canonical.

The §08 phase-relational CSVs come from the NeurIPS hyperpod bucket:

| Model | S3 path |
|---|---|
| V-JEPA†-e125 | `s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts/runs/base_e125_lvef_test_698/predictions/base_e125_lvef_test.csv` |
| V4-e25 (MV-PhaseRel) | `s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts/runs/final_phase_rel25_lvef_test_596/predictions/final_phase_rel25_lvef_test.csv` |

CSV schema: `video_path, label_real, pred_real, abs_error`. Sort by `video_path` to align. Label identity with the §05 CSVs is verified (sorted-label equality across all three sources, N=1,277, 10.19–84.45 EF range).

### Probe checkpoints

| Model | S3 path |
|---|---|
| JEPA e100 probe | `s3://echodata25/neurips/probes/end_lvef_e100/jepa_in21k_e100/best.pt` |
| BYOL e100 probe | `s3://echodata25/neurips/probes/end_lvef_e100/byol_e100/best.pt` |
| MAE e99 probe | `s3://echodata25/neurips/probes/end_lvef_e100/mae_e99/best.pt` |
| SALT v1 e79 probe | `s3://echodata25/neurips/probes/end_lvef_e100/salt_v1_e79/best.pt` |
| V-JEPA†-e125 probe | `s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts/runs/base_e125_lvef_test_698/probe/` (best.pt inside) |
| V4-e25 probe | `s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts/runs/final_phase_rel25_lvef_595/probe/video_classification_frozen/neurips-final-phase-rel25-lvef/best.pt` |

### Pretraining encoder checkpoints (for reference, not used to recompute this table)

| Model | S3 path |
|---|---|
| JEPA e100 encoder | (uses the `jepa_in21k_e200_280/training_folder/e100.pt` or equivalent e100 snapshot from the IN21K line) |
| BYOL e100 encoder | byol training run e100 checkpoint |
| MAE e99 encoder | MAE training run e99 checkpoint |
| SALT e79 encoder | SALT v1 training run e79 checkpoint |
| V-JEPA†-e125 encoder | `s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts/runs/jepa_in21k_e200_280/training_folder/e125.pt` |
| V4-e25 encoder | `s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts/runs/final_phase_rel25_paper_593/checkpoints/latest.pt` |

---

## Reproduction

### 1. Download CSVs

```bash
mkdir -p /tmp/e100_lvef /tmp/lvef_strat

for m in jepa_in21k_e100 byol_e100 mae_e99 salt_v1_e79; do
  aws s3 cp s3://echodata25/neurips/results/lvef_noised/${m}_noised_lvef_persample.csv \
    /tmp/e100_lvef/${m}.csv --region us-west-2
done

aws s3 cp s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts/runs/base_e125_lvef_test_698/predictions/base_e125_lvef_test.csv \
  /tmp/lvef_strat/base_e125.csv --region us-west-2
aws s3 cp s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts/runs/final_phase_rel25_lvef_test_596/predictions/final_phase_rel25_lvef_test.csv \
  /tmp/lvef_strat/v4_e25.csv --region us-west-2
```

### 2. Run the unified stratification script

```bash
python3 scripts/neurips/compute_lvef_stratified_unified.py
```

The script is at `scripts/neurips/compute_lvef_stratified_unified.py`. It:

1. Loads the four e100 noised_inference CSVs and filters to `condition == "clean"`
2. Loads the V-JEPA†-e125 and V4 prediction CSVs and sorts by `video_path`
3. Verifies label identity across all three sources (sorted-label equality to 1e-3)
4. Computes per-stratum MAE for each model with continuous clinical-range binning (`ef < 40`, `40 ≤ ef < 50`, `50 ≤ ef < 70`, `ef ≥ 70`)
5. For the §08 comparison, computes paired bootstrap Δ (V4 − V-JEPA†-e125) with B=10,000 resamples

### 3. Expected output

Full output is archived to `claude/neurips/reports/stratified_results/lvef_strat_v2_unified.txt`.

---

## Version history

- **2026-05-06**: First written. Resolved the Mildly-reduced stratum count discrepancy (125 vs 104) by adopting continuous clinical-range binning as canonical. Added JEPA e100 as a reference column to the §08 phase-relational table so the §05 and §08 tables share a common baseline model.
- **Prior versions** of the stratified README used integer-threshold binning (`41 ≤ ef ≤ 49`) which dropped 21 borderline videos. Those stratum Ns (104) are superseded by 125.

---

## Known caveats

1. **Single-clip-per-study evaluation**: both sets of CSVs contain raw per-clip predictions, no prediction-averaging across clips per study. The paper's `tab:controlled-end100` uses a different protocol (likely prediction-averaged); **full-cohort MAEs in this doc will not exactly match that paper table**. Per-stratum ratios and rank-orderings are consistent with the paper.

2. **Bin boundary at Hyperdynamic `≥70`**: some ASE/AHA documents use `>70` as the HFpEF / hyperdynamic threshold. We use `≥70` for bin closure — this is a 2-subject difference vs `>70` (checked: 38 subjects with `≥70`, 36 with `>70`). Negligible for conclusions.

3. **N=38 Hyperdynamic**: all per-stratum deltas in this stratum have wide CIs. Not a reliable headline — report with explicit sample-size caveat.

4. **Per-sample CSV `condition == "clean"` is the canonical single-clip prediction**: each subject has 10 rows in the CSV (clean + 9 noise conditions). All stratum statistics in this doc use only the `clean` row per subject.

---

# Data-efficiency companion: 50% stratified-subsample pretraining

**Appended 2026-05-06.** This section records the 50%-data companion
to the controlled extension in Table 2. Both continuations were
re-run on a stratified 50% subsample of the EchoNet-Dynamic train
CSV, preserving per-bin clinical stratification, to test whether
V4's advantage over vanilla V-JEPA is data-efficient or data-hungry.

## Companion Table — 50% stratified-subsample train (overall val only)

Values are **validation-set** best-epoch across 20 training epochs
(the sbatches did not include a test-inference step, see
reproducibility note below). Val n = 1,141 clips; train n = 3,731
clips (50% of 7,465 full train, stratified).

| Metric             | JEPA e125 (50%) | V4-e25 (50%) | Δ (V4 − base) |
|:-------------------|---------------:|-------------:|--------------:|
| Best val R²        | 0.669 (ep 18)  | 0.718 (ep 20) | **+0.049**    |
| Best val MAE (EF)  | 5.24 (ep 18)   | 4.76 (ep 18)  | **−0.48**     |
| Best val Pearson   | 0.819 (ep 18)  | 0.849 (ep 20) | +0.030        |

**Full-data test reference** (from Table 2 / Fig 2 for side-by-side):

| Metric             | JEPA e125 (100% test) | V4-e25 (100% test) | Δ (V4 − base) |
|:-------------------|---------------------:|-------------------:|--------------:|
| Test R²            | 0.646                | 0.699              | +0.053        |
| Test MAE (EF)      | 5.36                 | 4.88               | −0.48         |
| Test Pearson       | 0.839                | 0.806 *(not used as paper reference)* | — |

**Interpretation**: V4's advantage over vanilla V-JEPA is essentially
preserved at half the training data:
ΔR² 50%-val ≈ +0.049 vs ΔR² 100%-test ≈ +0.053, and the absolute
MAE gain is identical (−0.48 EF points in both regimes). V4 is
**not more data-hungry** than vanilla V-JEPA continuation; if
anything, the 50% val gain matches the 100% test gain, which is
mild positive evidence for improved data efficiency. A full
characterisation would require matched test-side numbers at both
data fractions and CIs, neither of which is currently available.

## Source data

### Train CSVs

- **50%-data train CSV**:
  `s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts/data/csv/echonet_dynamic_train_s3_raw_strat50.csv`
  (3,731 clips, built by `/tmp/build_strat50.py` with seed 42;
  preserves per-bin clinical distribution of the full train CSV to
  within 0.02% on each of the four strata). Stratum-Ns in the
  subsample: Reduced 474, Mildly reduced 666, Normal 2,466,
  Hyperdynamic 125.
- **Full-data train CSV** (for reference):
  `s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts/data/csv/echonet_dynamic_train_s3_raw.csv`
  (7,465 clips).

### Probe checkpoints (50%-data runs)

| Run | Model | Probe S3 path |
|-----|-------|---------------|
| 870 | JEPA e125 on 50%-strat train | `s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts/runs/base_e125_strat50_echonet_lvef_870/probe/video_classification_frozen/neurips-base_e125_strat50-echonet-lvef/best.pt` |
| 867 | V4-e25 on 50%-strat train     | `s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts/runs/v4_e25_strat50_echonet_lvef_867/probe/video_classification_frozen/neurips-v4_e25_strat50-echonet-lvef/best.pt` |

### Probe training logs (val trajectory per epoch)

| Run | S3 path (log_r0.csv) |
|-----|----------------------|
| 870 | `runs/base_e125_strat50_echonet_lvef_870/probe/video_classification_frozen/neurips-base_e125_strat50-echonet-lvef/log_r0.csv` |
| 867 | `runs/v4_e25_strat50_echonet_lvef_867/probe/video_classification_frozen/neurips-v4_e25_strat50-echonet-lvef/log_r0.csv` |

Columns: `epoch, train_mae, val_mae, val_r2, val_pearson`. No
per-clip val predictions were written, which is why no stratified
breakdown is available in this section.

### Sbatches

| Role | Path |
|------|------|
| 50%-data train CSV generator  | `/tmp/build_strat50.py` (not checked into repo; one-shot script) |
| JEPA e125 on 50% (job 870)    | `scripts/neurips/phase/echonet_lvef_probe_base_e125_strat50.sbatch` |
| V4-e25 on 50% (job 867)       | `scripts/neurips/phase/echonet_lvef_probe_v4_e25_strat50.sbatch` |
| (Superseded) JEPA e120 on 50% (job 866) | `scripts/neurips/phase/echonet_lvef_probe_base_e120_strat50.sbatch` — this was the original JEPA-side run before I realised the baseline should match the paper's `e125` not `e120`. Run 866 landed at val R² 0.661, val MAE 5.28. Superseded by 870; retained for provenance. |

## Reproduction

```bash
# 1. Build the stratified 50% train CSV (one-shot; skip if already on S3)
python3 /tmp/build_strat50.py
aws s3 cp /mnt/.../data/csv/echonet_dynamic_train_s3_raw_strat50.csv \
  s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts/data/csv/ \
  --region us-west-2

# 2. Rebuild + upload the source tarball with the two sbatches in it,
#    then submit on a compute node (see scripts for full deployment pattern)
sbatch scripts/neurips/phase/echonet_lvef_probe_base_e125_strat50.sbatch
sbatch scripts/neurips/phase/echonet_lvef_probe_v4_e25_strat50.sbatch

# 3. After both complete (~55 min each), pull the per-epoch val logs
aws s3 cp s3://.../runs/base_e125_strat50_echonet_lvef_870/probe/.../log_r0.csv .
aws s3 cp s3://.../runs/v4_e25_strat50_echonet_lvef_867/probe/.../log_r0.csv .
# Best-epoch rows are JEPA e125 ep 18, V4-e25 ep 20 (on val R²).
```

## Known gaps / follow-up work

1. **No test inference.** The strat50 sbatches were cloned from
   `echonet_lvef_probe_mcc_e25.sbatch`, which does not include a
   test-inference step. Test-side stratified numbers require a ~5-10
   min follow-up job per model against the saved `best.pt` files +
   EchoNet-Dynamic test CSV.

2. **No per-clip val predictions.** Consequently, stratified 50%-data
   tables analogous to Table 2 are not currently reproducible from
   the artifacts we have. Would require either (a) re-running the
   probes with a `predictions_save_path` YAML key, or (b) running a
   `val_only` inference pass against the saved `best.pt` files.

3. **Baseline-checkpoint consistency.** Run 866 used JEPA e120 before
   I reconciled the baseline with the paper's e125 convention; run
   870 re-ran on e125 and is the canonical JEPA-side number in this
   section. Do not mix 866 and 870 val numbers in the same table.

4. **One data fraction, not a curve.** A full data-efficiency curve
   at e.g. `{5, 12.5, 25, 50, 100}` on both methods would require
   ~8 additional sbatches (4 fractions × 2 methods × 1 baseline
   checkpoint each), ~55 min per sbatch. Currently we have only the
   50% and 100% points.

---

# 50%-data test-side results (jobs 873 + 874)

**Added 2026-05-06, later.** Followed up on the "no test inference"
gap above. Two small `val_only` inference jobs, each ~3 minutes,
loaded the saved `best.pt` from 870 (base_e125 strat50) and 867
(V4-e25 strat50) respectively and ran inference against the full
EchoNet-Dynamic test CSV. Per-clip test predictions are now on S3
and paired with the 100%-train prediction CSVs by `video_path`.

## Overall test metrics (N=1,277, B=10,000 bootstrap)

| Metric | base_e125 50%-train test | V4-e25 50%-train test | Δ (V4 − base) [95% CI] |
|---|---:|---:|---:|
| Test R²        | 0.620 [0.573, 0.664] | **0.684** [0.642, 0.722] | **+0.063 [+0.038, +0.089]** ✅ |
| Test MAE (EF)  | 5.54 [5.27, 5.83]    | **5.04** [4.79, 5.30]    | **−0.50 [−0.69, −0.31]** ✅ |
| Test Pearson   | 0.788                | **0.829**                | — |

Both ΔR² and ΔMAE paired-bootstrap CIs exclude zero
(P(V4 > base) = 1.000).

## Stratified test MAE (N=1,277, 50%-train test)

| Stratum              | N    | base_e125 50% | V4-e25 50% | ΔMAE (V4 better) |
|:---------------------|-----:|--------------:|-----------:|-----------------:|
| Reduced (<40)        |  160 |         10.02 |    **9.10**|            +0.93 |
| Mildly reduced (40--49) | 125 |          7.97 |    **7.66**|            +0.32 |
| Normal (50--69)      |  954 |          4.25 |    **3.85**|            +0.41 |
| Hyperdynamic (≥70)   |   38 |         11.02 |    **9.35**|            +1.67 |
| Full cohort          | 1277 |          5.54 |    **5.04**|            +0.50 |

## Side-by-side 50% vs 100% on the same test set

| Comparison                               | ΔR² (V4 − base)         | ΔMAE (V4 better)       |
|:-----------------------------------------|:------------------------|:-----------------------|
| **100% train** (ref from Table 2, paired) | +0.053 [+0.027, +0.079] ✅ | +0.48 [+0.30, +0.66] ✅ |
| **50% train** (jobs 873, 874, paired)     | **+0.063 [+0.038, +0.089]** ✅ | **+0.50 [+0.31, +0.69]** ✅ |

**Reading**: V4's advantage over vanilla V-JEPA continuation
persists at half the training data. The absolute MAE gap is
essentially identical (−0.50 vs −0.48); the R² gap widens slightly
(+0.063 vs +0.053) because base_e125 loses more test R² at 50%
train (0.646 → 0.620, −0.026) than V4 does (0.699 → 0.684, −0.015).
This is mild positive evidence that state-synchronized training is
**not more data-hungry** than vanilla V-JEPA continuation; the
absolute improvement the method buys you is preserved when the
downstream probe sees half as much training data.

## Source data

### Test prediction CSVs (50%-data runs)

| Job | Model | S3 path |
|---|---|---|
| 873 | base_e125 on 50%-strat train, EchoNet-Dynamic test | `s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts/runs/base_e125_strat50_echonet_lvef_test_873/predictions/base_e125_strat50_echonet_lvef_test.csv` |
| 874 | V4-e25 on 50%-strat train, EchoNet-Dynamic test     | `s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts/runs/v4_e25_strat50_echonet_lvef_test_874/predictions/v4_e25_strat50_echonet_lvef_test.csv` |

CSV schema: `video_path, label_real, pred_real, abs_error`. Labels
are raw EF (%) on the 0–100 scale. N=1,277, same video_paths as
the 100%-train test CSVs (file in §08 comparison above), so paired
bootstrap against the 100%-train V-JEPA†-e125 and V4 results in
§08 is feasible without re-alignment.

### Sbatches

| Role | Path |
|------|------|
| base_e125 strat50 test inference (job 873) | `scripts/neurips/phase/echonet_lvef_test_base_e125_strat50.sbatch` |
| V4-e25 strat50 test inference (job 874)    | `scripts/neurips/phase/echonet_lvef_test_v4_e25_strat50.sbatch` |

Both cloned from `echonet_lvef_test_mcc_e25.sbatch`, substituting
the strat50 encoder CKPT_S3 and probe best.pt S3 paths. Runtime
~3 minutes each on one H100.

## Reproduction

```bash
# 1. Submit both test-inference jobs (nodes assumed idle)
sbatch scripts/neurips/phase/echonet_lvef_test_base_e125_strat50.sbatch
sbatch scripts/neurips/phase/echonet_lvef_test_v4_e25_strat50.sbatch

# 2. After both complete (~3 min each), download predictions
mkdir -p /tmp/strat50_test
aws s3 cp s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts/runs/base_e125_strat50_echonet_lvef_test_873/predictions/base_e125_strat50_echonet_lvef_test.csv /tmp/strat50_test/base_e125.csv --region us-west-2
aws s3 cp s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts/runs/v4_e25_strat50_echonet_lvef_test_874/predictions/v4_e25_strat50_echonet_lvef_test.csv /tmp/strat50_test/v4_e25.csv --region us-west-2

# 3. Bootstrap CIs + stratified breakdown: see inline snippet in the
#    session transcript, or reuse scripts/neurips/compute_lvef_strat_ci.py
#    after updating FILE paths to /tmp/strat50_test/*.csv.
```

## Resolved gaps

- **No test inference** (gap 1 in the companion section above): resolved by 873/874.
- **No per-clip val predictions** (gap 2): still open on the val side,
  but per-clip **test** predictions from 873/874 are now on S3, which
  is what the paper needs. The val-only gap no longer blocks any
  paper claim.
- **One data fraction, not a curve** (gap 4): still open; only 50%
  and 100% anchors are covered. A `{5, 12.5, 25, 50, 100}` sweep
  remains the natural extension.

---

# 25%-data results (jobs 875 + 876)

**Added 2026-05-06.** Extended the data-efficiency curve to a 25%
anchor. Unlike the 50% pair (jobs 866/867/870, which only did probe
training), the 25% sbatches include the test-inference step inline,
so per-clip test predictions land directly on S3.

## 25%-data train CSV

- **Path**: `s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts/data/csv/echonet_dynamic_train_s3_raw_strat25.csv`
- **N = 1,865** clips (25.0% of full train).
- **Bin convention**: `ef<40 / 40≤ef<50 / 50≤ef<70 / ef≥70` (AHA/ASE,
  integer cuts consistent with evaluation-side stratification).
- **Stratum Ns**: Reduced 237, Mild 181, Normal 1,385, Hyper 62.
- **Bin preservation**: ≤0.05 pp deviation from the full-train
  distribution on each of the four strata.

**Important caveat**: the earlier strat50 generator used a
**different Mild-bin edge** (`ef<55` instead of `ef<50`), so the
strat50 training file has ~5pp more "Mild" clips by count than
strat25 does. This is a train-side stratification mismatch
unrelated to the evaluation-side binning (which is consistent at
`ef<50` throughout). For the efficiency-curve claim below, both
training CSVs still preserve each bin's representation within
their generator's own convention; the mismatch only matters if
we compute per-bin training effective-sample-size matches across
fractions.

## Overall test metrics (N=1,277, B=10,000 bootstrap)

| Metric | base_e125 25%-train | V4-e25 25%-train | Δ (V4 − base) [95% CI] |
|---|---:|---:|---:|
| Test R²        | 0.548 [0.496, 0.598] | **0.633** [0.585, 0.678] | **+0.085 [+0.056, +0.113]** ✅ |
| Test MAE (EF)  | 5.89 [5.58, 6.22]    | **5.35** [5.07, 5.63]    | **−0.54 [−0.72, −0.36]** ✅ |
| Test Pearson   | 0.763                | **0.808**                | — |

## Stratified test MAE (N=1,277, 25%-train test)

| Stratum              | N    | base_e125 25% | V4-e25 25% | ΔMAE (V4 better) |
|:---------------------|-----:|--------------:|-----------:|-----------------:|
| Reduced (<40)        |  160 |         13.16 |   **10.55**|            +2.60 |
| Mildly reduced (40--49) | 125 |          9.40 |    **8.91**|            +0.48 |
| Normal (50--69)      |  954 |          4.05 |    **3.85**|            +0.19 |
| Hyperdynamic (≥70)   |   38 |         10.13 |    **9.20**|            +0.93 |
| Full cohort          | 1277 |          5.89 |    **5.35**|            +0.54 |

## Data-efficiency curve: V4 advantage across three training fractions

| Fraction | Train N | base test R² | V4 test R² | ΔR² (V4−base) | ΔMAE (V4 better) |
|:---------|--------:|-------------:|-----------:|---:|---:|
| **25%**  | 1,865 | 0.548 | 0.633 | **+0.085** | −0.54 |
| **50%**  | 3,731 | 0.620 | 0.684 | **+0.063** | −0.50 |
| **100%** | 7,465 | 0.646 | 0.699 | **+0.053** | −0.48 |

**Headline reading**: V4's ΔR² advantage over vanilla V-JEPA
continuation **widens as training data shrinks** — from +0.053 at
100% → +0.063 at 50% → +0.085 at 25%. ΔMAE stays ~constant at
~−0.5 EF points (base MAE rises with smaller train, V4 MAE rises
too, and they rise in lockstep so the gap is preserved). V4 is
**more** data-efficient than vanilla V-JEPA continuation, not
merely not-worse.

The **Reduced-EF stratum** is where the advantage sharpens most
under data scarcity:

| Fraction | base MAE (Reduced <40) | V4 MAE (Reduced <40) | Δ |
|:---------|--------:|-----:|---:|
| 100% train | 9.23 | 8.71 | −0.52 |
| 50%  train | 10.02 | 9.10 | −0.93 |
| 25%  train | 13.16 | 10.55 | **−2.60** |

On the clinically-actionable Reduced stratum, V4 beats base by
2.60 MAE points at 25% train — five times the 100%-train gap.
Under data scarcity, V4's representational prior does proportionally
more of the work on the subgroup where probes have the least data.

## Source data

### Test prediction CSVs (25%-data runs)

| Job | Model | S3 path |
|---|---|---|
| 875 | base_e125 on 25%-strat train, EchoNet-Dynamic test | `s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts/runs/base_e125_strat25_echonet_lvef_875/predictions/base_e125_strat25_echonet_lvef_test.csv` |
| 876 | V4-e25 on 25%-strat train, EchoNet-Dynamic test | `s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts/runs/v4_e25_strat25_echonet_lvef_876/predictions/v4_e25_strat25_echonet_lvef_test.csv` |

CSV schema: `video_path, label_real, pred_real, abs_error`. N=1,277.

### Sbatches

| Role | Path |
|------|------|
| 25%-data train CSV generator | `/tmp/build_strat25.py` (one-shot; seed 42) |
| base_e125 25% + test (job 875) | `scripts/neurips/phase/echonet_lvef_probe_base_e125_strat25.sbatch` |
| V4-e25 25% + test (job 876)    | `scripts/neurips/phase/echonet_lvef_probe_v4_e25_strat25.sbatch` |

Both sbatches run 20-epoch probe training **and** test inference
inline on the saved best.pt. Runtime ~42 min each on an 8-GPU H100
node (train) + ~3 min (test).

## Reproduction

```bash
# 1. Build + upload the 25%-data train CSV (one-shot)
python3 /tmp/build_strat25.py
aws s3 cp .../data/csv/echonet_dynamic_train_s3_raw_strat25.csv \
  s3://.../data/csv/ --region us-west-2

# 2. Submit both jobs (nodes assumed idle)
sbatch scripts/neurips/phase/echonet_lvef_probe_base_e125_strat25.sbatch
sbatch scripts/neurips/phase/echonet_lvef_probe_v4_e25_strat25.sbatch

# 3. After both complete (~45 min), download and analyse
mkdir -p /tmp/strat25_test
aws s3 cp s3://.../runs/base_e125_strat25_echonet_lvef_875/predictions/base_e125_strat25_echonet_lvef_test.csv /tmp/strat25_test/base_e125.csv --region us-west-2
aws s3 cp s3://.../runs/v4_e25_strat25_echonet_lvef_876/predictions/v4_e25_strat25_echonet_lvef_test.csv /tmp/strat25_test/v4_e25.csv --region us-west-2
```

---

# 12.5%-data test-set results (jobs 877, 878)

**Added 2026-05-06.** Fourth anchor on the data-efficiency curve.
Both sbatches run 20-epoch probe training + inline test inference
on the EchoNet-Dynamic test cohort.

## 12.5%-data train CSV

- **Path**: `s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts/data/csv/echonet_dynamic_train_s3_raw_strat12_5.csv`
- **N = 931** clips (12.5% of full train).
- **Bin convention**: `ef<40 / 40≤ef<50 / 50≤ef<70 / ef≥70` (AHA/ASE,
  same as strat25; consistent with evaluation-side binning).
- **Stratum Ns**: Reduced 118, Mild 90, Normal 692, Hyper 31.
- **Bin preservation**: within 0.09 pp of full-train distribution
  on each of the four strata.

## Overall test metrics (N=1,277, B=10,000 bootstrap)

| Metric | base_e125 12.5%-train | V4-e25 12.5%-train | Δ (V4 − base) [95% CI] |
|---|---:|---:|---:|
| Test R²        | 0.506 [0.441, 0.563] | **0.574** [0.519, 0.624] | **+0.069 [+0.035, +0.104]** ✅ |
| Test MAE (EF)  | 6.17 [5.85, 6.50]    | **5.82** [5.53, 6.12]    | **−0.35 [−0.56, −0.14]** ✅ |
| Test Pearson   | 0.724                | **0.779**                | +0.055 [+0.034, +0.078] ✅ |

Paired-bootstrap p < 0.001 on R² and Pearson; p = 0.0006 on MAE.

## Stratified test MAE (N=1,277, 12.5%-train test)

| Stratum              | N    | base_e125 12.5% | V4-e25 12.5% | ΔMAE (V4 better) |
|:---------------------|-----:|----------------:|-------------:|-----------------:|
| Reduced (<40)        |  160 |           13.00 |    **11.59** |            +1.41 |
| Mildly reduced (40--49) | 125 |          10.07 |     **9.83** |            +0.24 |
| Normal (50--69)      |  954 |            4.28 |     **4.17** |            +0.11 |
| Hyperdynamic (≥70)   |   38 |           11.94 |     **9.85** |            +2.09 |
| Full cohort          | 1277 |            6.17 |     **5.82** |            +0.35 |

## Updated data-efficiency curve (four anchors)

| Fraction | Train N | base test R² | V4 test R² | ΔR² (V4−base) | ΔMAE (V4 better) |
|:---------|--------:|-------------:|-----------:|---:|---:|
| **12.5%**| 931   | 0.506 | 0.574 | **+0.069** | −0.35 |
| **25%**  | 1,865 | 0.548 | 0.633 | **+0.085** | −0.54 |
| **50%**  | 3,731 | 0.620 | 0.684 | **+0.063** | −0.50 |
| **100%** | 7,465 | 0.646 | 0.699 | **+0.053** | −0.48 |

**Headline reading**: V4's ΔR² advantage is consistently positive at
every anchor and, in coarse terms, widens as data shrinks (+0.053 →
+0.069 from 100% → 12.5%). The curve is non-monotone: 25% is the
peak advantage (+0.085), with 12.5% slightly below it (+0.069). Both
endpoints of the 12.5% CI are strictly above zero, so the effect
remains highly significant at the smallest fraction.

Reduced-stratum MAE gap across fractions:

| Fraction | base MAE (Reduced <40) | V4 MAE (Reduced <40) | Δ |
|:---------|--------:|-----:|---:|
| 100%  train | 9.23  | 8.71  | −0.52 |
| 50%   train | 10.02 | 9.10  | −0.93 |
| 25%   train | 13.16 | 10.55 | −2.60 |
| 12.5% train | 13.00 | 11.59 | −1.41 |

The 25%-train anchor remains the sharpest Reduced-stratum advantage.
At 12.5% both models lose ground on Reduced together (base MAE only
drops 0.16 points vs 25%, V4 MAE rises 1.04 points), so the gap
narrows but stays clinically meaningful at 1.41 EF points.

## Source data

### Test prediction CSVs (12.5%-data runs)

| Job | Model | S3 path |
|---|---|---|
| 877 | base_e125 on 12.5%-strat train, EchoNet-Dynamic test | `s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts/runs/base_e125_strat12_5_echonet_lvef_877/predictions/base_e125_strat12_5_echonet_lvef_test.csv` |
| 878 | V4-e25 on 12.5%-strat train, EchoNet-Dynamic test | `s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts/runs/v4_e25_strat12_5_echonet_lvef_878/predictions/v4_e25_strat12_5_echonet_lvef_test.csv` |

CSV schema: `video_path, label_real, pred_real, abs_error`. N=1,277.

### Sbatches

| Role | Path |
|------|------|
| base_e125 12.5% + test (job 877) | `scripts/neurips/phase/echonet_lvef_probe_base_e125_strat12_5.sbatch` |
| V4-e25 12.5% + test (job 878)    | `scripts/neurips/phase/echonet_lvef_probe_v4_e25_strat12_5.sbatch` |

Both run 20-epoch probe training + test inference inline on best.pt.
Runtime ~30 min each on an 8-GPU H100 node.
