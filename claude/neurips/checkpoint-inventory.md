# NeurIPS Checkpoint Inventory

Complete inventory of all pretraining checkpoints across objectives and initializations. Updated 2026-04-04.

**S3 bucket abbreviations:**
- `HYP`: `s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts`
- `H100`: `s3://sagemaker-echojepa-h100-march-0d224785-bucket`
- `CLEAN`: `s3://echodata25/neurips` (organized mirror for NeurIPS)

---

## 1. EchoJEPA-L (V-JEPA 2.0) — ImageNet-21K Init

**Init:** `vitl_in21k.pt` (ImageNet-21K supervised ViT-L, 2D, inflated to 3D at load time)
**Data:** MIMIC-IV-Echo 525K clips
**Architecture:** ViT-L (304M), V-JEPA 2.0, patch 16, tubelet 2, 16 frames, 224px
**Confirmed by:** HyperPod run 125 `params-pretrain.yaml`: `anneal_ckpt: vitl_in21k.pt`

**This is the init that matches BYOL and MAE.** Use for the NeurIPS controlled comparison.

| Epoch | S3 Path | Run | Date | Seed |
|-------|---------|-----|------|------|
| e0–e85 (every 5ep) | `HYP/runs/jepa_in21k_pretrain_376/checkpoints/e{0,5,...,85}.pt` | 376 (running) | Apr 5 | 234 |
| e100 (expected) | `HYP/runs/jepa_in21k_pretrain_376/checkpoints/e100.pt` | 376 | ~Apr 6 | 234 |
| e100 (old run, same init) | `HYP/runs/vjepa_mimic_pretrain_125/training_folder/e100.pt` | 125 | Jan 25 | 42 |
| e110 | `HYP/runs/vjepa_mimic_pretrain_125/training_folder/e110.pt` | 125 | Jan 25 | 42 |
| e120 | `HYP/runs/vjepa_mimic_pretrain_125/training_folder/e120.pt` | 125 | Jan 25 | 42 |
| e130 | `HYP/runs/vjepa_mimic_pretrain_125/training_folder/e130.pt` | 125 | Jan 25 | 42 |
| e122-e154 | `HYP/runs/vjepa_mimic_pretrain_137/training_folder/e{122..154}.pt` | 137 | Jan 26 | 154 |
| e146-e168 | `HYP/runs/vjepa_mimic_pretrain_148/training_folder/e{146..168}.pt` | 148 | Jan 26-27 | 24 |
| e162-e238 | `HYP/runs/vjepa_mimic_pretrain_150/training_folder/e{162..238}.pt` | 150 | Jan 27-28 | 234 |

**Gaps:** Epochs 0-99 lost (trained on HyperPod NVMe, never synced to S3). Would need to retrain for training dynamics analysis at early epochs.

**Seed changes:** The run chain used different seeds across continuations (42→83→154→24→234). Run 125 (seed 42) covers e100-e130 as a single consistent segment. Later runs resumed from `latest.pt` so the seed change only affects data ordering, not model weights.

**NeurIPS mirror:** `CLEAN/encoders/echojepa_l_pt50.pt` is the **Meta V-JEPA 2 init** checkpoint (see §2), NOT this run. Needs to be updated if using ImageNet init for the comparison.

---

## 2. EchoJEPA-L (V-JEPA 2.0) — Meta V-JEPA 2 Init

**Init:** `vitl.pt` (Meta V-JEPA 2 ViT-L, pretrained on internet video with JEPA objective, epoch 40)
**Data:** MIMIC-IV-Echo 525K clips
**Confirmed by:** Git commit `6d93d8a` (Jan 18): `force_load_pretrain: true`, `anneal_ckpt: vitl.pt`

**This is what was used for the ICML rebuttal 3-way comparison.** Gives JEPA a video-SSL head start over BYOL/MAE (which use ImageNet-21K). A confound for the NeurIPS controlled comparison.

| Epoch | Location | Date |
|-------|----------|------|
| e50 | EFS: `checkpoints/echojepa-l-pt50.pt` | Jan 21 |
| e70 | EFS: `checkpoints/echojepa-l-pt70.pt` | Jan 21 |
| e90 | EFS: `checkpoints/echojepa-l-pt90.pt` | Jan 22 |
| e110 | EFS: `checkpoints/echojepa-l-pt110.pt` | Jan 22 |
| e150 | EFS: `checkpoints/echojepa-l-pt150.pt` | Jan 23 |
| e180 | EFS: `checkpoints/echojepa-l-pt180.pt` | Jan 23 |
| e200 | EFS: `checkpoints/echojepa-l-pt200.pt` | Jan 23 |
| e220 | EFS: `checkpoints/echojepa-l-pt220.pt` | Jan 24 |
| e230+an10 | EFS: `checkpoints/echojepa-l-pt230-an10.pt` | Jan 24 |
| e230+an20 | EFS: `checkpoints/echojepa-l-pt230-an20.pt` | Jan 25 |
| e230+an30 | EFS: `checkpoints/echojepa-l-pt230-an30.pt` | Jan 25 |

Also in `checkpoints/pretrain/mimic/vjepa2_vitl_224px_16f_kinetics/`: e210, e215, e220 (Feb 12, separate run from same init).

**S3 mirror:** `CLEAN/encoders/echojepa_l_pt50.pt` (4.8 GB)

---

## 3. EchoBYOL-L — ImageNet-21K Init

**Init:** `vitl_in21k.pt` (ImageNet-21K supervised ViT-L, same as JEPA §1 and MAE §4)
**Data:** MIMIC-IV-Echo 525K clips
**Architecture:** ViT-L (304M), BYOL-Video, global mean-pool + cosine loss, EMA teacher

| Epoch | S3 Path | Run/Source | Date |
|-------|---------|-----------|------|
| e0-e50 (every 2ep) | `H100/checkpoints/byol-vitl-imagenet-v2/e{0,2,4,...,50}.pt` | Original H100 training | Mar 27-28 |
| e50 | EFS: `checkpoints/byol_vitl_imagenet_v2_e50.pt` | — | Mar 28 |
| e95 | `HYP/runs/byol_pretrain_resume_342/checkpoints/e95.pt` | Job 342 | Apr 4 |
| e100 | `HYP/runs/byol_pretrain_resume_342/checkpoints/e100.pt` | Job 342 | Apr 4 |
| e100-e200 | `HYP/runs/byol_pretrain_resume_e100_362/checkpoints/` | Job 362 (running) | Apr 4+ |

**Gap:** e52-e94 not saved to S3 (job 342 kept only last 3 on disk, manually uploaded e95/e100).

**S3 mirror:** `CLEAN/encoders/echobyol_l_pt50.pt` (2.3 GB)

---

## 4. EchoMAE-L (VideoMAE) — ImageNet-21K Init

**Init:** `vitl_raw.pth` = `vitl_in21k.pt` (confirmed identical via `torch.equal`)
**Data:** MIMIC-IV-Echo 525K clips
**Architecture:** ViT-L (304M), VideoMAE, pixel reconstruction, tube masking 90%

| Epoch | S3 Path | Run/Source | Date |
|-------|---------|-----------|------|
| e4-e54 (every 5ep) | `HYP/runs/videomae_matched_2n_245/training_folder/checkpoint-{4,9,...,54}.pth` | 2-node matched run | Mar 29 |
| e50 | `HYP/checkpoints/echomae_l_mimic_ep50.pth` | Standalone copy | Mar 29 |
| e58-e116 | `HYP/runs/videomae_resume_e54_354/training_folder/checkpoint-{58,59,64,...,116}.pth` | Job 354 | Apr 1 |
| e116-e124+ | `HYP/runs/videomae_resume_e116_363/training_folder/checkpoint-{116,119,124}.pth` | Job 363 (running) | Apr 4+ |

**No gaps.** Every 5 epochs from e4 to e124, with a few extras (e58, e59).

**S3 mirror:** `CLEAN/encoders/echomae_l_pt50.pth` (3.6 GB)

---

## 5. EchoJEPA-L (V-JEPA 2.1) — Meta V-JEPA 2.1 Init

**Init:** `vjepa2_1_vitl.pt` (Meta V-JEPA 2.1 ViT-L)
**Data:** MIMIC-IV-Echo 525K clips
**Architecture:** ViT-L (304M), V-JEPA 2.1 (dense hierarchical prediction, multi-layer heads)

| Epoch | Location | Date |
|-------|----------|------|
| e100 | EFS: `checkpoints/pretrain_21/mimic/vjepa2_1_vitl_224px_16f/e100_run125.pt` | Jan 25 |
| e109 | EFS: `checkpoints/pretrain_21/mimic/vjepa2_1_vitl_224px_16f/e109.pt` | Mar 26 |
| e114 | EFS: same dir `/e114.pt` | Mar 26 |
| e119 | EFS: same dir `/e119.pt` | Mar 26 |
| e125 | EFS: same dir `/e125_latest_backup.pt` | Mar 26 |
| e138 | EFS: same dir `/latest.pt` | Mar 27 |

Not part of the NeurIPS controlled comparison (different architecture version). Available for P1 experiments if needed.

---

## 6. SALT — Complete (v1 + v3 variants)

**Init:** S1 teacher: `vitl_in21k.pt` (ImageNet-21K). S2 student: random init (per SALT paper recipe).
**Code:** `app/salt/` (built 2026-04-04, DDP fix 2026-04-05)
**Configs:** `configs/train/vitl16/pretrain-salt-s{1,2}-mimic-224px-16f-hp.yaml`

### 🔒 Primary SALT checkpoint for final experiments: SALT v1 e79

All NeurIPS tables and figures use **`salt_s2_vitl_e79.pt`** (v1, hierarchical predictor, LR 1.75e-4 constant). Best test R²=0.414, MAE=6.66 on EchoNet-Dynamic. See `experiments/salt-comparison.md` for the full decision rationale. v1 e199 and v3 e79 remain as appendix robustness lines on END LVEF only — do not re-run them on other tasks.

| Variant | Encoder path | END LVEF probe | S2 epochs | Test R² | Test MAE | Role |
|---|---|---|---|---|---|---|
| **v1 e79** (primary) | EFS: `checkpoints/salt_s2_vitl_e79.pt` | `evals/vitl/icml/salt_s2_e79_end_lvef_224/.../best.pt` | 80 | **0.414** | **6.66** | Main §3 table row, all downstream experiments |
| v1 e199 | EFS: `checkpoints/salt_s2_vitl_e199.pt` | `evals/vitl/icml/salt_s2_e199_end_lvef_224/.../best.pt` | 200 | 0.360 | 7.02 | Appendix robustness (END LVEF only) |
| v3 e79 | S3: `HYP/runs/salt_s2v2_pretrain_446/checkpoints/e79.pt` | S3: `HYP/runs/salt_s2v3_echonet_lvef_454/probe/best.pt` | 80 | 0.348 | 7.03 | Appendix robustness (END LVEF only, paper-spec single-level predictor) |

### Stage 1 (V-Pixel teacher, 20 epochs, ImageNet init)

| Epoch | S3 Path | Run | Date |
|-------|---------|-----|------|
| e4, e9, e14, e19, latest (e20) | `HYP/runs/salt_s1_pretrain_379/checkpoints/` | Job 379 | Apr 5 |

### Stage 2 v1 (hierarchical predictor, constant LR, weak aug)

| Epoch | S3 Path | Run | Date |
|-------|---------|-----|------|
| e4–e79 (every 5ep) + latest (e80) | `HYP/runs/salt_s2_pretrain_388/checkpoints/` | Job 388 | Apr 5 |
| e80→e100 | `HYP/runs/salt_s2_resume_e80_391/checkpoints/` | Job 391 | Apr 5 |
| e100→e199 | `HYP/runs/salt_s2_resume_e100_392/checkpoints/` | Job 392 | Apr 5-6 |

### Stage 2 v3 (single-level predictor, paper-spec cosine LR + augmentation)

| Epoch | S3 Path | Run | Date |
|-------|---------|-----|------|
| e4–e79 (every 5ep) | `HYP/runs/salt_s2v2_pretrain_446/checkpoints/` | Job 446 | Apr 7 |

**Compute budget:** S1=20ep (21K steps) + S2=200ep (205K steps) = 226K total steps. Matches SALT paper's ~240K recommended budget. (Still ~10% of paper absolute budget due to batch-size difference — see `salt-comparison.md` for full deviation notes.)

**Implementation note:** S2 DDP fix applied — frozen teacher must not be wrapped in DDP (`app/salt/train.py:377`). Hierarchical `norms_block` layers in teacher were never trained (S1 uses `training_mode=False`); this is a known deviation from the paper but consistent across both v1 and v3. Both variants use `loss_exp: 1.0` (L1, matching paper Eq 2.1) — the earlier "v1 used L2" claim was retracted after config inspection (2026-04-07).

**Decision: RESOLVED (2026-04-08).** SALT underperforms all three EMA baselines by 0.03–0.24 R². Included as a single row in the §3 comparison table (v1 e79) plus two sentences in §4.5. The finding is robust across v1/v3 predictor architectures, HP regimes, and training lengths — all three variants land within ±0.03 R² of each other and all below MAE's 0.445. Full writeup: `experiments/salt-comparison.md`.

---

## 7. Initialization Checkpoints

| Name | File | Size | Type | Used By |
|------|------|------|------|---------|
| ImageNet-21K ViT-L | EFS: `checkpoints/vitl_in21k.pt` / S3: `HYP/checkpoints/vitl_raw.pth` | 1.3 GB | Flat state dict, 2D patch_embed, has cls_token | BYOL, MAE, JEPA§1, SALT (planned) |
| Meta V-JEPA 2 ViT-L | EFS: `checkpoints/vitl.pt` / S3: `HYP/checkpoints/vitl.pt` | 4.8 GB | V-JEPA checkpoint (encoder+predictor+target_encoder), epoch 40 | JEPA§2 (ICML rebuttal) |
| Meta V-JEPA 2.1 ViT-L | EFS: `checkpoints/vjepa2_1_vitl.pt` | ~4.8 GB | V-JEPA 2.1 checkpoint | JEPA 2.1§5 |

**`vitl_raw.pth` = `vitl_in21k.pt`** — confirmed identical weights via `torch.equal` on `blocks.0.attn.qkv.weight`.

---

## 8. NeurIPS Controlled Comparison Status

For the e100 controlled comparison (all ImageNet-21K init):

| Model | Init | e100 Checkpoint | Status |
|-------|------|----------------|--------|
| **JEPA** | ImageNet-21K | `HYP/runs/jepa_in21k_pretrain_376/.../e100.pt` | **Training** (job 376, ~e85, ~4h left) |
| **BYOL** | ImageNet-21K | `HYP/runs/byol_pretrain_resume_342/.../e100.pt` | **Available** |
| **MAE** | ImageNet-21K | `HYP/runs/videomae_resume_e54_354/.../checkpoint-99.pth` | **Available** (check exact filename) |
| **SALT S2** | Random student (ImageNet teacher) | `HYP/runs/salt_s2_pretrain_388/.../e79.pt` | **Available** (e200 in progress) |

**Training dynamics coverage (for appendix figure):**

| Model | Available Epochs | Gaps |
|-------|-----------------|------|
| JEPA IN21K (job 376) | e0-e85 (every 5ep), e100 expected | None (dense) |
| JEPA (old, same init) | e100, e110, e120, e130, e122-e238 | e0-e99 lost |
| BYOL | e0-e50 (every 2ep), e85, e90, e95, e100, e105 | e52-e84 |
| MAE | e4-e124 (every 5ep) | None |
| SALT S2 | e4-e79 (every 5ep), e100 expected, e200 in progress | None (dense) |

---

## 9. Probe Checkpoints (EchoNet-Dynamic LVEF)

These are the frozen probes used for frame shuffling and noise robustness. **Trained on JEPA§2 (Meta V-JEPA 2 init), not JEPA§1 (ImageNet init).** Will need new probes for the ImageNet-init controlled comparison.

| Model | S3 Path | Notes |
|-------|---------|-------|
| JEPA (Meta init) | `CLEAN/probes/end_lvef_pt50/echojepa_l_pt50/best.pt` | Head 3, d=4 attentive |
| BYOL | `CLEAN/probes/end_lvef_pt50/echobyol_l_pt50/best.pt` | Head 1, d=4 attentive |
| MAE | `CLEAN/probes/end_lvef_pt50/echomae_l_pt50/best.pt` | Head 5, d=4 attentive |
| JEPA (ImageNet init) | — | **Pending** (job 376 finishes ~Apr 6) |
| SALT S2 e79 | EFS: `evals/vitl/icml/salt_s2_e79_end_lvef_224/.../best.pt` | Val MAE 6.47 |
| SALT S2 e49 | EFS: `evals/vitl/icml/salt_s2_e49_end_lvef_224/.../best.pt` | Trained on other machine |

---

## 10. Key Decision: Initialization Confound

The ICML rebuttal used JEPA§2 (Meta V-JEPA 2 init) while BYOL and MAE used ImageNet-21K init. This is a confound — JEPA got a video-SSL head start.

For NeurIPS, all models should use **ImageNet-21K init** (§1/§3/§4). The JEPA§1 e100 checkpoint exists on S3. New probes will need to be trained on it.

**Action items:**
1. Verify JEPA§1 e100 loads correctly into the probe pipeline
2. Train LVEF/RVSP/CAMUS probes on JEPA§1 e100
3. Run frame shuffling + noise robustness with JEPA§1 probes
4. Compare results to ICML rebuttal numbers (JEPA§2) — if rankings hold despite different init, that's evidence the finding is robust
