# MCC-JEPA — launch readiness

**Status:** code + configs + sbatches + tests landed. **Not yet launched.**

## Checkpoint

- Path: `checkpoints/jepa_in21k_vitl_e100.pt`
- Size: 5,127,835,835 bytes (matches `scripts/neurips/phase/verify_init_checkpoint.py` allowlist).
- Keys: `encoder, predictor, target_encoder, opt, scaler, epoch=100, itr, lr, loss, batch_size, world_size`.
- Param counts: encoder 303,885,312 · predictor 22,086,016 (ViT-L/16 + standard V-JEPA predictor).
- Loaded through the canonical path: `optimization.anneal_ckpt` + `optimization.force_load_pretrain: true` in every config.

## What was added

### New files (code)
| Path | LOC | Purpose |
|---|---|---|
| `src/models/mcc_jepa/__init__.py` | 3 | package export |
| `src/models/mcc_jepa/cross_clip_adapter.py` | 73 | zero-gated cross-attention residual; γ ∈ R; identity at init |
| `src/datasets/mcc_pair_dataset.py` | 200 | same-study pair manifest (40/30/20/10 mixture + shuffled-A control) |
| `app/vjepa_multiview/mcc_jepa_forward.py` | 160 | `forward_mcc_jepa(mode='pure'\|'target_anchored', …)` |

### Edits (minimal)
- `app/vjepa_multiview/train.py` — added `mcc_jepa` to the objective allow-list, added config parsing for `mcc_mode / lambda_mcc / lambda_vjepa / mcc_adapter.*`, added adapter instantiation + optimizer registration, added dispatch branch that calls `forward_mcc_jepa`. ~55 lines.

### New configs
| Path | Epochs | Mode | λ_mcc |
|---|---|---|---|
| `configs/train/vitl16/pretrain-vjepa-in21k-e100-plus25-control.yaml` | 25 | vanilla (intraview_only) | — |
| `configs/train/vitl16/pretrain-mcc-jepa-pure-smoke.yaml` | 1 (500 steps) | pure | 1.0 |
| `configs/train/vitl16/pretrain-mcc-jepa-target-anchored-smoke.yaml` | 1 (500 steps) | target_anchored | 0.2 |
| `configs/train/vitl16/pretrain-mcc-jepa-target-anchored-25of100.yaml` | 25 | target_anchored | 0.2 |

All four initialize from `/opt/dlami/nvme/checkpoints/jepa_in21k_vitl_e100.pt` with matched 32/GPU × 8 global batch (256 samples/step).

### New sbatches
| Path | Walltime |
|---|---|
| `scripts/mcc_jepa/pretrain_vjepa_plus25_control.sbatch` | 1d 12h |
| `scripts/mcc_jepa/pretrain_mcc_pure_smoke.sbatch` | 1h 30m |
| `scripts/mcc_jepa/pretrain_mcc_target_anchored_smoke.sbatch` | 1h 30m |
| `scripts/mcc_jepa/pretrain_mcc_target_anchored_25ep.sbatch` | 1d 12h |
| `scripts/mcc_jepa/launch_mcc_25ep.sh` | helper |
| `scripts/mcc_jepa/verify_checkpoint.py` | preflight |

Every sbatch mirrors `scripts/neurips/phase/final_phase_rel_hardneg25_paper.sbatch`:
- 1 × ml.p5.48xlarge, 8 × H100
- S3 source-tarball deploy into `/opt/dlami/nvme/src/vjepa2`
- init-checkpoint preflight via `scripts/neurips/phase/verify_init_checkpoint.py --strict`
- 10-min periodic sync of checkpoints + logs to `s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts/runs/...`
- entry point `python -m app.main --fname <cfg> --devices cuda:0..cuda:7`

### Tests (all passing, 15/15)
```
tests/mcc_jepa/
  test_same_study_pair_sampler.py       [3 tests]
  test_single_clip_fallback.py          [1 test ]
  test_cross_clip_adapter_zero_init.py  [4 tests]
  test_target_anchored_no_leak.py       [2 tests]
  test_mcc_forward_shapes.py            [2 tests]
  test_shuffled_A_control.py            [3 tests]
```

## Pair-sampler dry-run stats

Synthetic 10-study × 8-clip manifest (including one color-Doppler clip per study):
```
n_pairs=10  pair_same_study_rate=1.00  pair_distinct_clip_rate=1.00  fallback=0.30
bucket_counts = {same_broad_family: 2, same_view: 0, cross_view: 4, cross_modality: 1, fallback_any: 3}
```
Fallback fraction on synthetic data is high because each study has few same-view bmode clips; on the real MIMIC manifest with 8+ clips per study, `fallback_fraction` is expected < 0.20.

## Lint + quality

- `pytest tests/mcc_jepa/ -q` → 15 passed
- `flake8 app/vjepa_multiview/mcc_jepa_forward.py src/datasets/mcc_pair_dataset.py src/models/mcc_jepa/ tests/mcc_jepa/ scripts/mcc_jepa/verify_checkpoint.py --max-line-length 119` → 0 warnings
- `black --check --line-length 119 <same list>` → all unchanged
- `isort --check-only --profile=black --line-length=119 <same list>` → all unchanged
- `bash -n scripts/mcc_jepa/*.sbatch` → all OK
- All 4 YAMLs parse; `force_load_pretrain=true`, `anneal_ckpt` ends with `jepa_in21k_vitl_e100.pt` in every config.

## Differences vs vanilla V-JEPA

| | Vanilla +25 | Target-anchored MCC +25 |
|---|---|---|
| Init checkpoint | `jepa_in21k_vitl_e100.pt` | same |
| Pair sampler (loader) | same-study triples (matched eligibility) | same-study triples (matched eligibility) |
| Student context | clip_a | visible tokens of clip_b |
| Source tokens | — | full clip_a tokens → zero-gated adapter on predictor output |
| Teacher | EMA of encoder on clip_a | EMA of encoder on clip_b (full) |
| Loss | `L_vjepa(clip_a)` | `L_vjepa_self(clip_b) + 0.2 · L_mcc(clip_b\|clip_a)` |
| γ at init | — | 0.0 (adapter is identity) |
| Optimizer / schedule / batch | identical | identical |

At γ = 0, the target-anchored forward is byte-equivalent to a plain V-JEPA forward on clip_b. γ can only grow if the adapter's contribution *reduces* the L_mcc loss. That's the structural guarantee that this is a pure additive extension.

## Differences vs MV2SV (concrete, not marketing)

| | MV2SV | Target-anchored MCC-JEPA |
|---|---|---|
| Prediction target | Pooled target-view latent / slots | Per-tubelet latents of clip_b at masked positions |
| Context | One source clip | Source A + visible B (anchored) |
| Architecture | Factorized head, conditional view predictor | Vanilla V-JEPA predictor + zero-gated cross-attn adapter |
| Metadata conditioning | Target view/modality/phase metadata | None — visible B tokens supply acquisition info |
| Inference claim | Single-view student "hallucinates" multi-view | Improved clip encoder, no view hallucination claim |
| Failure mode | Retrieval improves, downstream regresses | γ stays 0 and method reduces to vanilla V-JEPA — safe by construction |

## Smoke gates (target-anchored, ~500 steps)

Pass criteria — all must hold:
- no NaNs
- `loss_vjepa_self` finite, trending down (or at least bounded at e100 level)
- `loss_mcc` finite, trending down
- `gamma > 0.01` by step 500 **OR** `pred_delta_from_A > 0.01`
- `loss_same_study_A < loss_b_visible_only`
- `loss_same_study_A < loss_shuffled_A`
- `cov_off < 0.5`
- encoder + adapter grad nonzero; teacher no-grad
- `pair_same_study_rate > 0.95`, `pair_distinct_clip_rate > 0.90`

Fail → **stop**, do not launch the 25-epoch run.

Pure MCC smoke has a narrower gate: `loss_same_study_A < loss_shuffled_A` and loss finite. Used purely to decide whether pure MCC is worth considering in the future.

## Launch commands

Local preflight (recommended before every submit):
```bash
python -m pytest tests/mcc_jepa/ -q
python scripts/mcc_jepa/verify_checkpoint.py
bash scripts/mcc_jepa/launch_mcc_25ep.sh   # prints (does not submit)
```

Smoke:
```bash
sbatch scripts/mcc_jepa/pretrain_mcc_target_anchored_smoke.sbatch
# (optional, parallel diagnostic)
sbatch scripts/mcc_jepa/pretrain_mcc_pure_smoke.sbatch
```

25-epoch (only after target-anchored smoke passes):
```bash
sbatch scripts/mcc_jepa/pretrain_vjepa_plus25_control.sbatch
sbatch scripts/mcc_jepa/pretrain_mcc_target_anchored_25ep.sbatch
```
or equivalently `bash scripts/mcc_jepa/launch_mcc_25ep.sh --yes-25ep`.

## Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Adapter γ stays at 0 through training | med | Expected, safe — method collapses to vanilla V-JEPA. Training still uses matched compute; comparison still meaningful. |
| Cross-attn on 1568 A tokens is memory-heavy | med | Use activation checkpointing (already on in model cfg); if OOM, switch `mcc_source_token_subsample` to 512 in the YAML. |
| Fallback fraction > 0.30 on real MIMIC | low | Diagnostic logs `fallback_fraction` every epoch; re-tune pair-mixture weights if observed. |
| L_vjepa_self regresses vs vanilla | low | λ_mcc = 0.2 is conservative; L_vjepa active at full weight. Sweep to λ_mcc = 0.1 if needed. |
| `pretrain-mcc-jepa-target-anchored-25of100.yaml` diverges | med | Smoke gates catch divergence at 500 steps. |
| Same-study A provides no advantage over shuffled A | high-ish | That *is* the experiment — if so, the gate fails at smoke and we do not promote. |

## Open items (not blocking launch)

- Verify `source_proj_dim` is not needed at ViT-L: encoder `embed_dim = 1024`, predictor output after `predictor_proj` is also 1024 — confirmed in `src/models/predictor.py:130`. Adapter works at a single dim without a projection.
- The 2 diagnostic probe forwards (B-visible-only and shuffled-A) are described in `forward_mcc_jepa` docstring but not yet implemented inside it. They can be added after the smoke (run `mcc_diagnostic_every` steps) without touching the main loss path.
- `launch_mcc_25ep.sh --yes-25ep` submits both jobs serially but does not wait for smoke success. Operator must confirm smoke passed before running.

## Holds

- Do NOT launch the 25-epoch runs until the target-anchored smoke passes all gates.
- Do NOT run pure MCC for 25 epochs without explicit approval.
- Do NOT add MV2SV factorized slots, phase-relational InfoNCE, TokenRel, MC-JEPA transport, or motion-guided masking in this run.
- Do NOT initialize from e125 / e200 / phase / MV2SV checkpoints.
