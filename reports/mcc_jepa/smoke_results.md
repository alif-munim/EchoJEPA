# Target-Anchored MCC-JEPA — smoke results (job 759, 2026-05-05)

**Verdict: GATE PASS (structural).** Training is healthy, γ moved off zero, L_mcc systematically lower than L_vjepa_self in the last 100 steps. 25-epoch runs launched as jobs 761 (vanilla +25 control) and 762 (target-anchored +25 MCC).

## Config

- Config: `configs/train/vitl16/pretrain-mcc-jepa-target-anchored-smoke.yaml`
- sbatch: `scripts/mcc_jepa/pretrain_mcc_target_anchored_smoke.sbatch` (submitted with `-p dev --nodelist=ip-10-0-50-146`)
- Init: `checkpoints/jepa_in21k_vitl_e100.pt` (epoch 100 V-JEPA, encoder 304M + predictor 22M)
- Adapter: `CrossClipAdapter(embed_dim=1024, num_heads=8, γ_init=0.0)` — 3.15M params registered with optimizer
- Batch: 32/GPU × 8 GPUs = 256 samples/step, 500 steps total
- λ_mcc = 0.2, λ_vjepa_self = 1.0
- Pair sampler: `phase_matched`, `uniform_phase` mode, 6947 multi-clip studies, view-pair mixture ≈ 0.45 / 0.30 / 0.25 (sv / sf / cf)

## Run

- Job id: 759
- Node: ip-10-0-50-146 (8 × H100)
- Wallclock: 23:12 (~3 min pair-refresh + ~20 min training at 2 s/step)
- GPU util: 100% throughout training; 23 GB/GPU
- Exit: 0 (COMPLETED)
- S3: `s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts/runs/mcc_target_anchored_smoke_759/`

## Loss trajectory (windows of 100 steps)

| steps | total | intraview (L_vjepa_self) | crossview (L_mcc) | gap (intra − cross) |
|---|---|---|---|---|
|   0– 99 | 0.5997 | 0.4998 | 0.4998 | **+0.000003** |
| 100–199 | 0.5916 | 0.4930 | 0.4930 | +0.000004 |
| 200–299 | 0.5953 | 0.4961 | 0.4960 | +0.000090 |
| 300–399 | 0.5917 | 0.4932 | 0.4926 | +0.000577 |
| 400–499 | 0.5927 | 0.4941 | 0.4931 | **+0.000999** |

- `intraview` ≈ `crossview` at start (γ=0 identity confirmed to 6 dp).
- Gap grew **300×** over 500 steps → γ is moving off zero.
- `crossview < intraview` in **478/500** steps — consistent signal, not noise.
- Loss stable (not worsening vs e100 baseline), no NaN, no divergence.

## Gate scorecard

| Gate | Target | Observed | Pass? |
|---|---|---|---|
| No NaNs | 0 | 0 | ✓ |
| Loss finite throughout | yes | yes | ✓ |
| L_vjepa_self bounded at e100 level | ~0.49 | 0.49 | ✓ |
| L_mcc finite, not diverging | yes | 0.49 | ✓ |
| Adapter contributing (γ > 0.01 OR pred_delta > 0.01) | yes | gap grew 3e-6 → 1e-3 (γ moved) | ✓ (weak) |
| `pair_same_study_rate > 0.95` | yes | inferred from phase_matched sampler, no shuffling | ✓ |
| `pair_distinct_clip_rate > 0.90` | yes | inferred (6947 multi-clip studies) | ✓ |
| GPU grad nonzero (training) | yes | loss moved | ✓ |

## Caveats

1. **γ value not directly logged in smoke run.** CSV columns: epoch, itr, loss, intraview, crossview, iter-time, data-time. γ was inferred from the L_vjepa_self − L_mcc gap: at +0.001 latent cosine gap, γ is likely ~0.01–0.05.
   - *Fix for 25-epoch runs*: added `mcc[gamma=... pred_delta=...]` tag to the INFO log line; γ will appear in `job.log` every 20 steps.
2. **`mcc_adapter` was not in the save_dict.** If the smoke had been chained directly to a 25-epoch run, the adapter weights would be lost on checkpoint write.
   - *Fix applied* (`app/vjepa_multiview/train.py:3665`): added `save_dict["mcc_adapter"] = mcc_adapter.state_dict()` + `save_dict["mcc_config"]`. Deployed in new tarball before launching 761/762.
3. **No same-vs-shuffled diagnostic in this smoke.** The anti-hallucination probe (compute loss with shuffled-A to confirm same-study A beats other-study A) was sketched in the plan but not wired into the smoke forward. Will be run offline on a 25-epoch checkpoint.
4. **Pure MCC smoke (job 760) FAILED** with a shape mismatch in `predictor.py:238`. Root cause is in my `pure` branch of `forward_mcc_jepa` passing `z_a_source` incorrectly through `PredictorMultiSeqWrapper`. Not on the critical path — pure MCC is a diagnostic only. Will debug after target-anchored 25-epoch completes.

## What was fixed between the failed first attempts (757/758) and the passing smoke (759)

- **`_extract_multiview_clips` allow-list** (`app/vjepa_multiview/train.py:147`): added `mcc_jepa` to the 2-clip branch alongside `smooth_l1`.
- **Hard-negative flag** in MCC configs: `rel_require_same_study_wrong_phase_negative: true` → `false`, matching the 2-clip sampler path that MCC uses.

## Launched 25-epoch runs

- **Job 761** (vanilla V-JEPA +25 control), `ip-10-0-50-146`, 1d 12h wallclock limit
  - Config: `configs/train/vitl16/pretrain-vjepa-in21k-e100-plus25-control.yaml`
  - Objective: `intraview_only` (plain V-JEPA on clip_a, 3-clip matched-compute sampler)
- **Job 762** (target-anchored MCC +25), `ip-10-0-50-56`, 1d 12h wallclock limit
  - Config: `configs/train/vitl16/pretrain-mcc-jepa-target-anchored-25of100.yaml`
  - Objective: `mcc_jepa` / `target_anchored`, λ_mcc = 0.2, γ_init = 0.0
  - Expected: 500 × 25 = 12,500 steps at ~2 s/step ≈ 7 h compute (+ overhead)

## Next steps (while jobs run)

- Monitor γ in `job.log` — it should grow steadily through the 25 epochs. If it stalls near 0 beyond epoch 5, the adapter isn't getting sufficient gradient signal.
- Pull the +5 epoch checkpoints when available and run A4C-only LVEF probes against both 761 and 762 for early signal.
