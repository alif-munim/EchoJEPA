# Verification Experiment: Attentive Probe Fairness

Configs for the verification experiment confirming that depth=1 attentive probes are non-harmful for all model architectures, including 1-token models (EchoPrime, PanEcho).

See `uhn_echo/nature_medicine/context_files/probe_implementation_analysis.md` for the full experiment spec.

## Background

The ICML paper showed attentive probes degrading EchoPrime (-15.6pp), PanEcho (-10.9pp), and EchoMAE (-18.8pp). Analysis revealed this was an artifact of:
1. **Normalization bugs** in EchoPrime (3 bugs) and PanEcho (double ImageNet norm)
2. **Identical HP grid** for all models (no per-model tuning)
3. **Config discrepancies**: EchoJEPA had 30 epochs/11 classes/336px vs 20 epochs/13 classes/224px for others

## What's fixed

- All models: 224px, 13 classes, 50 epochs, 5-epoch warmup
- Per-model HP sweep: 5 LR × 4 WD = 20 combos via multihead_kwargs
- Fixed normalization code (post-commit 4803640)

## Configs

| Config | Model | Probe | Depth | HP combos |
|--------|-------|-------|-------|-----------|
| `echojepa_g_d1.yaml` | EchoJEPA-G (ViT-G, 1568 tokens) | attentive | 1 | 20 |
| `echomae_d1.yaml` | EchoMAE-L (ViT-L, 1568 tokens) | attentive | 1 | 20 |
| `echoprime_d1.yaml` | EchoPrime (MViT-v2-S, 1 token) | attentive | 1 | 20 |
| `panecho_d1.yaml` | PanEcho (ConvNeXt-T, 1 token) | attentive | 1 | 20 |
| `echojepa_g_d4.yaml` | EchoJEPA-G | attentive | 4 | 20 |
| `echomae_d4.yaml` | EchoMAE-L | attentive | 4 | 20 |
| `echoprime_d4.yaml` | EchoPrime | attentive | 4 | 20 |
| `panecho_d4.yaml` | PanEcho | attentive | 4 | 20 |
| `echoprime_linear.yaml` | EchoPrime | linear | - | 9 |
| `panecho_linear.yaml` | PanEcho | linear | - | 9 |

## Running

```bash
# Depth=1 only (shortcut, ~32 GPU-hours):
bash scripts/run_verification_experiment.sh

# Full (depth=1 + depth=4, ~64 GPU-hours):
bash scripts/run_verification_experiment.sh --full
```

## Decision criteria

- EchoPrime d=1 attentive >= linear (within 3pp) → **Strategy E**: uniform depth=1 attentive for all models
- EchoPrime d=1 attentive < linear (>5pp drop) → **Strategy B**: linear primary + attentive JEPA ceiling
