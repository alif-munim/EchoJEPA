# EchoSet-JEPA view × modality coverage audit

- clip_manifest: `/tmp/echoset_pr_n2/study_clip_manifest_dedup.parquet`
- element_manifest: `/tmp/echoset_pr_n2/study_element_manifest.parquet`
- k_sample_manifest: `/tmp/echoset_pr_n2/study_clip_sample_K8_seed0_train.parquet`

## Gate status

- **passed**: True
- frac_studies_ge2_view_families_in_K8 = 0.989  (threshold 0.9)
- color_retention_in_K8 = 0.854  (threshold 0.8)

## Overall

- clips: 214,100
- studies: 7,127
- patients: 4,546
- clips/study: median=30, p75=37, p95=49, max=86

## Per-split

### train
- 182,919 clips / 6,089 studies / 3,891 patients
- clips/study median=30

### val
- 16,523 clips / 548 studies / 342 patients
- clips/study median=30

### test
- 14,658 clips / 490 studies / 313 patients
- clips/study median=30

## View × modality crosstab

| view_family | b_mode | color_doppler |
|---|---|---|
| apical | 49423 | 38097 |
| parasternal_long | 27876 | 11034 |
| parasternal_short | 32615 | 18027 |
| subcostal | 7181 | 5419 |
| suprasternal | 378 | 322 |
| unknown | 15640 | 8088 |

## K=8 sampler diagnostics

- studies sampled: 6,089
- clips sampled: 47,955
- view_families/study: mean=4.33, median=4.0, frac_ge2=0.989
- color slots/study: mean=1.94, median=2.0, frac_ge1=0.972
- bmode slots/study: mean=5.93

## Element manifest

- elements: 57,506
- elements/study: median=8, p95=11, max=18
- distinct (view, modality, phase_bucket) keys: 24

## Modality-presence leakage signals (for Control D / color-present-only baselines)

- frac studies with any color_doppler: 0.973
- frac studies B-mode-only: 0.027
- per-study color count: mean=11.36 ± 5.57

## Quality bucket × split

```
{
  "test": {
    "high": 4562,
    "low": 5141,
    "med": 4955
  },
  "train": {
    "high": 60999,
    "low": 60946,
    "med": 60974
  },
  "val": {
    "high": 5588,
    "low": 5461,
    "med": 5474
  }
}
```
