# Root-cause interim report — MCC & FJ low A4C LVEF

**Scope**: Parts 1–4 of the diagnostic plan. Interim finding: **one very large comparability bug found**; MCC implementation audit finds **the adapter's γ gate is stuck at 0 by design and no active mechanism grows it, so MCC ≈ 1.2× vanilla V-JEPA gradient on clip_b with 2-clip data distribution and 10% cross-modality noise**. FJ audit still pending (deferred to Part 8 after probe jobs 792/793 complete).

## The critical finding — comparability bug

The "MV-PhaseRel 0.70 / MCC 0.35" gap I have been reporting across tables is **not a valid comparison**. The numbers come from **different datasets**:

| Probe model | LVEF dataset |
|---|---|
| **MCC-Anchored e25** (786) | `mimic_lvef_a4c_10k` |
| **MCC-Anchored e25 online** (793) | `mimic_lvef_a4c_10k` |
| **FullJoint-Study 30k** (792) | `mimic_lvef_a4c_10k` |
| base_e125 (job 705) | ran HCM + HF-incident only; **LVEF never started** (evidence: stdout + S3 artifacts) |
| MV-PhaseRel V4 e25 (final_phase_rel25_lvef_595) | `echonet_dynamic_train_s3_raw` |
| MV-PairedIntra (final_paired_iv25_lvef_629) | `echonet_dynamic_train_s3_raw` |
| TokenRel-Motion e25 (tokenrel_r2_e25_lvef_719) | `echonet_dynamic_train_s3_raw` |
| TokenRel-Motion **e5** on MIMIC A4C (tokenrel_r2_e5 / job 704) | `mimic_lvef_a4c_10k` |
| jepa_ext_probes_332 / e125 | `echonet_dynamic_train_s3_raw` (d=4 attentive, "end-lvef" config) |

### Like-for-like trajectories (ALL on `mimic_lvef_a4c_10k`)

| Epoch | **MCC e25 / target_encoder** (786) | **FJ 30k** (792) | TokenRel e5 |
|---:|---:|---:|---:|
| 1 | 0.21 | 0.16 | −0.02 |
| 5 | 0.26 | 0.15 | 0.20 |
| 10 | 0.31 | — | 0.24 |
| 14 | 0.32 | — | 0.31 |
| 20 | — | — | **0.31** |

**Tokenrel e5 (3.6 EFLOPs, 5 epochs of continuation) reaches 0.31 at ep 20** on the same dataset MCC/FJ are being evaluated on. MCC at 0.32 and FJ tracking toward similar is **approximately par** with an early-training baseline, not "catastrophic regression." The earlier narrative that MCC is far below base_e125 or MV-PhaseRel was computed against **EchoNet-Dynamic numbers**, which are not comparable.

### What we do not know yet

- **There is no `base_e125` LVEF probe on `mimic_lvef_a4c_10k`.** Job 705's stdout ends while still in HF-incident training; the age and LVEF heads never ran. So the actual FLOPs-matched V-JEPA baseline on this dataset is **missing**, not 0.65 as previously claimed.
- **There is no `e100` LVEF probe on `mimic_lvef_a4c_10k`.** The 0.591 number cited earlier comes from ICML ECHO-Dynamic or an older split.
- **There are no MV-PhaseRel / MV-PairedIntra probes on `mimic_lvef_a4c_10k`.** All matched-FLOPs comparators I cited are actually on EchoNet-Dynamic.

**This means the "MCC is 2× worse than peers" claim is unsupported by the data currently in hand.**

## MCC implementation audit findings

Code inspected: `app/vjepa_multiview/mcc_jepa_forward.py`, `src/models/mcc_jepa/cross_clip_adapter.py`, `app/vjepa_multiview/train.py:3170–3204` (adapter build), `app/vjepa_multiview/train.py:3984–3987` (EMA update), `configs/train/vitl16/pretrain-mcc-jepa-target-anchored-25of100.yaml`.

### Verified invariants

1. **target-anchored B_visible + A_source prediction**: correctly implemented. `z_b_visible = encoder(clip_b, masks_enc)`, `z_pred_base = predictor(z_b_visible, masks_enc, masks_pred)`, `z_pred_anchored = z_pred_base + γ · CrossAttn(pred_B, A_source)`, `L_mcc = L_p(z_pred_anchored, h_B_teacher)`. ✓
2. **Vanilla V-JEPA self-loss present**: `L_vjepa_self = L_p(z_pred_base, h_B_teacher)`, weighted at λ_vjepa=1.0. ✓
3. **MCC weighted**: `total = 1.0 · L_vjepa_self + 0.2 · L_mcc`. Matches config. ✓
4. **EMA update**: `target_encoder` is EMA of online `encoder` with τ=0.99925. No frozen e100 anchor anywhere. ✓
5. **Teacher no grad**: `h_b = target_encoder(clip_b)` under `torch.no_grad()`. ✓
6. **Adapter saved**: `save_dict` includes `mcc_adapter` (verified in run log earlier this session). ✓

### Found issues

**Issue A (design ambiguity, not a bug per se)**: "Target-anchored" names the prediction anchoring (pred_B is anchored, adapter adds a residual), NOT an encoder anchor to e100. There is no frozen `f_0 = e100` in the MCC forward. The FJ design doc *did* want such an anchor; MCC does not and never claimed to.

**Issue B (expected initialization, but in practice latent failure mode)**: γ starts at 0. At γ=0:
```
z_pred_anchored = z_pred_base + 0 · CrossAttn(...) = z_pred_base
L_mcc = L_p(z_pred_base, h_B_teacher) = L_vjepa_self
total ≈ 1.2 · L_vjepa_self
```
Both loss terms become identical, differing only in how they backprop through γ. **The encoder is effectively trained under 1.2× the vanilla V-JEPA gradient** until γ moves.

**Issue C (gradient sink)**: Even when γ moves, the encoder is called **twice per step** — `encoder(clip_b, masks_enc)` AND `encoder(clip_a)` (no mask) — but only the clip_b path supervises the encoder via V-JEPA. The clip_a encoder forward exists only to feed the adapter; its gradient pathway is gated by γ. If γ≈0, clip_a encoder forward is pure compute cost with zero signal. If γ>0, the adapter's cross-attention gradient partly flows into the encoder via clip_a tokens, but **the teacher target (h_B) is still just clip_b's teacher**, so clip_a's encoder role is "produce tokens useful as keys/values for B's mask prediction" — an abstract objective that may or may not align with useful representations.

**Issue D (sampler concern, to verify)**: `view_pair_policy` config has **10% cross_modality** (B-mode ↔ color Doppler) and 20% cross_family (A4C ↔ PLAX/PSAX). Cross-modality V-JEPA targets (predict color-Doppler tokens from B-mode visible tokens) are approximately noise-only — the prediction task is not well-posed. At 10% prevalence, that's ~10% noise in the MCC gradient signal. Needs verification that the trained encoder actually received such pairs (sampler logs would show).

**Issue E (no γ schedule)**: γ is a single scalar in `nowd_params` trained at the encoder LR with no warmup or amplification. There is no explicit pressure for γ to grow; it grows only if the adapter's cross-attention residual reduces L_mcc. The **actual γ value at e25** is not in the CSV header I pulled earlier — needs extraction from the full CSV (available in S3 `runs/mcc_target_anchored_25of100_762/checkpoints/log_r0.csv`). If γ < 0.05 at e25, the "MCC" objective ≈ "1.2× V-JEPA on clip_b + wasted clip_a forward" for the whole run.

### Weight drift evidence (from earlier inspection)

Relative L2 drift from jepa_in21k_e100 (online encoder):

| Layer | MCC e25 | base e125 (ECHO-Dynamic-path run) |
|---|---:|---:|
| patch_embed | 46.9% | 27.9% |
| block 6 qkv | 38.4% | 22.5% |
| block 12 qkv | 39.1% | 22.8% |
| block 18 qkv | 39.4% | 23.0% |
| block 23 qkv | 34.8% | 19.6% |

MCC encoder drifted **~1.7× more than vanilla +25 on the same e100 init**. Consistent with 1.2× loss-magnitude scaling of L_vjepa_self over 16,250 steps plus noisier gradient from cross-modality pairs.

### Adapter γ / pred_delta_from_A (to confirm)

`mcc_jepa_forward.py:150` computes `pred_delta_from_A = 1 - cos(pred_anchored, pred_base)` as a diagnostic. At γ=0, this is exactly 0. Need to check the full MCC training CSV to see if it ever rose above ~0.02.

## FJ implementation audit (deferred)

Not yet executed in this pass. The Parts 8–10 plan calls for a full audit of:
- true MaskCollator clip V-JEPA path
- single-view→study branch fire rate
- cross-rank NCE negatives
- anchor schedule and layerwise drift
- K=8 vs K=4 actual

This will be done next, once probe job 792 (FJ 30k LVEF) completes and provides final numbers.

## Decisions to pause

1. **Do not claim MCC underperforms matched-FLOPs peers on LVEF A4C** until we have a base_e125 LVEF A4C probe on `mimic_lvef_a4c_10k`. The claim currently rests on EchoNet-Dynamic comparators, which are not comparable.
2. **Do not claim MCC is broken** until we extract γ and pred_delta_from_A from the full training CSV. Possible that MCC did learn something (γ > 0.1) but the signal is small on single-clip A4C LVEF because MCC's value is multi-clip.
3. **Do not cancel MCC/FJ experiments** based on this LVEF A4C probe alone. The right diagnostic is K=8 multi-clip study-level LVEF, which is what FJ was designed for.

## Priority actions (after probe 792/793 complete)

### P0 — fill baseline gaps on the same dataset

Train a **base_e125 A4C LVEF probe on `mimic_lvef_a4c_10k`** (same config as 786/792/793). This is the single missing number that determines whether MCC/FJ are "worse than vanilla" or "on par with vanilla." ~90 min on one 8×H100.

### P1 — extract MCC diagnostics

Pull the full `log_r0.csv` from `runs/mcc_target_anchored_25of100_762/checkpoints/log_r0.csv`. Check:
- final γ value
- `pred_delta_from_A` trajectory
- λ_mcc × L_mcc vs λ_vjepa × L_vjepa_self ratio over time
- any per-step sampler distribution logs

If γ stayed near 0, we have proven "MCC adapter never learned anything"; next step is a re-run with a γ warmup or ablation against vanilla +25.

### P2 — FJ diagnostic

Pull FJ training CSV from `runs/.../full_joint_restart_v2_30k_runs/776/log_r0.csv`. Parse the 42-column CSV for:
- `loss_clip_vjepa_true`
- `sv_valid_fraction`, `a4c_sv_count`
- `anchor_cosine_to_e100`
- `clip_grad_norm`, `study_grad_norm`, `clip_pred_grad_norm`
- `metadata_only_study_gap`

Answers whether FJ's clip-level path was active and whether the study-level signal is non-trivial vs metadata shortcut.

### P3 — native h_study probe (FJ)

After 792 completes, queue a K=8 prediction-averaging LVEF probe using FJ's clip encoder (or if time permits, an h_study probe using the FJ study transformer on K=8 clips). This is the actual test of FJ's design intent.

## Revised framing for the paper

The results from single-clip A4C LVEF are **not yet** a verdict on MCC or FJ. Both objectives were designed to exploit multi-clip or study-level information; single-clip A4C LVEF is specifically the scenario that should stress-test whether they preserved clip-level competence. A fair comparison requires:

1. All models evaluated on `mimic_lvef_a4c_10k` (matched-dataset).
2. K=8 multi-clip LVEF (prediction averaging) — the setting where MCC/FJ's investments should pay off.
3. Non-LVEF tasks (RV function, TAPSE, MR) where the encoder drift direction matters differently.

Until we have (1) for at least base_e125, we cannot publish any MCC/FJ vs vanilla comparison.
