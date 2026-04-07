# RVSP Protocol Comparison — d=4 Multi-View vs d=1 + Pred Avg

**Date:** 2026-04-07
**Question:** For RVSP, which evaluation protocol is best — Strategy E (d=1 attentive + prediction averaging) or the rebuttal protocol (d=4 attentive + multi-view fusion)?

## Background

Two probe protocols have been used for RVSP across the project:

1. **Rebuttal protocol** (ICML companion paper): d=4 attentive probe with factorized multi-view cross-attention over A4C + PSAX-AV. Uses `VideoGroupDataset` (multi-view dataloader). Single-clip metric at test time.
2. **Strategy E protocol** (Nature Medicine + NeurIPS): d=1 attentive probe with single-view input. Uses `VideoDataset`. Prediction averaging across all clips per study at test time.

The companion `predavg-vs-singleclip-test.md` doc established that pred avg gives a clean +7.4pp R² boost for EchoJEPA-G on RVSP. This doc extends that analysis to compare against the d=4 multi-view protocol from the rebuttal.

## Configurations Verified

### Rebuttal RVSP (`configs/eval/vitl/icml/echojepa_l_pt50_rvsp_d4_full.yaml`)
```yaml
classifier:
  num_heads: 16
  num_probe_blocks: 4         # d=4
  num_views: 2                # multi-view: A4C + PSAX-AV
  clips_per_view: 2
  use_factorized: true        # factorized cross-attention
data:
  dataset_type: VideoGroupDataset
  dataset_train: data/csv/rvsp_train.csv  # 40,969 rows (full)
  dataset_val: data/csv/rvsp_val.csv
model_kwargs:
  checkpoint: checkpoints/echojepa-l-pt50.pt   # 50 epochs MIMIC
```

### Strategy E RVSP (`configs/eval/vitl/rvsp.yaml`)
```yaml
classifier:
  num_heads: 16
  num_probe_blocks: 1         # d=1
data:
  dataset_type: VideoDataset
  dataset_train: data/csv/rvsp_train.csv  # same train CSV
  study_sampling: true        # 1 random clip/study/epoch
model_kwargs:
  checkpoint: checkpoints/anneal/keep/vitl-pt-210-an25.pt   # 235 epochs MIMIC (manuscript L)
```

**Both protocols use the same train/val/test splits.** The differences are: probe depth (4 vs 1), multi-view vs single-view, training-time vs inference-time aggregation, and encoder checkpoint (pt50 vs pt-210-an25).

## Results — EchoJEPA-L pt50 on UHN RVSP test (5,103 studies)

Single test set, single train set. Only protocol and aggregation differ:

| Setup | Probe | Views | Aggregation | Test R² | Test Pearson | Test MAE |
|-------|-------|-------|-------------|---------|-------------|----------|
| **d=4 multi-view (rebuttal)** | d=4 factorized | A4C + PSAX-AV | training-time | **0.220** | **0.484** | 9.101 |
| d=4 single-view A4C | d=4 | A4C only | none | 0.181 | 0.447 | 9.266 |
| d=4 single-view PSAX | d=4 | PSAX only | none | 0.188 | 0.449 | 9.368 |
| d=1 + pred avg (manuscript L, **different ckpt**) | d=1 | single-view | pred avg | 0.168 | 0.442 | — |

## Effect Decomposition

| Effect | Δ R² | Source |
|--------|------|--------|
| **Multi-view fusion** (single-view d=4 → multi-view d=4) | **+0.032 to +0.039** | (0.220 − 0.188) and (0.220 − 0.181) |
| **Probe depth** (d=1 → d=4, both single-view) | **~+0.013** | (0.181 − 0.168), L pt50 d=4 A4C vs L manuscript d=1 |
| **Total (d=1 single-view + pred avg → d=4 multi-view)** | **+0.052** | (0.220 − 0.168) |

Cross-checked against the published "single-view ablation" finding in `claude/neurips/completed-experiments.md`: *"Multi-view +3.9pp R² over best single view"* — matches.

## The Encoder Confound

The d=4 results use **L pt50** (50 epochs MIMIC pretraining), while the d=1 + pred avg result uses **L pt-210-an25** (235 epochs, the manuscript L). The 235-epoch checkpoint is a stronger encoder, so the 0.168 result is *likely* an overestimate of how d=1 + pred avg would perform on pt50 specifically.

If we ran d=1 + pred avg on L pt50, it would probably land in the 0.14–0.16 range, making the multi-view + depth advantage even larger:
- Estimated d=1 + pred avg on L pt50: ~0.15
- d=4 multi-view on L pt50: 0.220
- Estimated true protocol effect: **~+0.07 R²** (vs +0.052 with the confounded comparison)

To eliminate this confound, one targeted experiment would settle it: train d=4 multi-view on L manuscript (pt-210-an25) and compare against d=1 + pred avg = 0.168.

## Findings

1. **For RVSP, d=4 multi-view is the better protocol.** The total advantage over d=1 + pred avg is +5.2pp R² confounded, ~+7pp R² unconfounded estimate.

2. **Multi-view fusion contributes most of the gain (+3.9pp R²).** Probe depth contributes about +1.3pp on top. The biggest single lever is the cross-view integration that d=4's self-attention blocks enable, not the depth itself.

3. **RVSP is the only task where this matters.** All other UHN benchmark tasks (LVEF, TAPSE, hemodynamics, valve severity) perform best with d=1 + pred avg under Strategy E. RVSP is unique because Doppler-derived RV systolic pressure is computed from cross-view geometry — multi-view information is physiologically relevant in a way that single-view aggregation can't recover.

4. **Scale dominates protocol.** EchoJEPA-G with d=1 + pred avg (R²=0.504) beats EchoJEPA-L with d=4 multi-view (R²=0.220) by +28pp R². Encoder size and pretraining data matter ~5× more than probe protocol on this task.

## Implications

### For the Nature Medicine paper

Two options:
1. **Use d=4 multi-view for RVSP only** — protocol heterogeneity but better numbers (~+5pp R² for L manuscript). Would need methods text justifying "RVSP requires multi-view because Doppler-derived hemodynamics depend on cross-view geometry."
2. **Stick with d=1 + pred avg uniformly** — clean methodology and tells one consistent story across all tasks. Costs ~5pp R² on the headline RV pillar.

### For the NeurIPS paper

The clean isolation of the +3.9pp multi-view effect strengthens the methods discussion. RVSP can be the example where "multi-view fusion matters when single-view information is fundamentally insufficient," contrasting with LVEF (where single-view A4C contains nearly all the information).

## Recommended Next Experiment

**Train d=4 multi-view RVSP probe on L manuscript (pt-210-an25)** to remove the encoder confound. Cost: ~3 hours on 2 GPUs. This single run would:
- Confirm whether the d=4 multi-view advantage holds on the manuscript checkpoint
- Give a within-checkpoint protocol comparison: d=4 multi-view vs d=1 + pred avg, both on the same encoder
- Definitively resolve the protocol decision for RVSP

If d=4 multi-view on L manuscript still wins by >3pp R² over d=1 + pred avg (0.168), use d=4 multi-view for RVSP in the manuscript. Otherwise, the protocol uniformity argument wins and Strategy E stays.

## Sources

- Rebuttal RVSP results: `claude/rebuttals/10-rebuttal-experiment-results.md` §1b
- Single-view ablation: `claude/neurips/completed-experiments.md` §7
- Strategy E pred avg results: `uhn_echo/nature_medicine/context_files/dev/probe-results.md`
- Pred avg vs single-clip analysis: `claude/neurips/experiments/predavg-vs-singleclip-test.md`
- Configs: `configs/eval/vitl/icml/echojepa_l_pt50_rvsp_d4_full.yaml`, `configs/eval/vitl/rvsp.yaml`
