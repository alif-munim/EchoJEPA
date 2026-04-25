# NeurIPS Checkpoint Inventory

Complete inventory of all pretraining checkpoints across objectives and initializations. Updated 2026-04-09.

> **WARNING (2026-04-09):** The JEPA encoder at `HYP/runs/vjepa_mimic_pretrain_125/.../e100.pt` (run 125) is **NOT** the canonical encoder used for probe training. The canonical encoder is from run 376 and lives at `CLEAN/encoders/jepa_in21k_vitl_e100.pt`. Using the wrong encoder with the correct probe produces near-random predictions. **Always use `CLEAN/encoders/` paths.** See `claude/neurips/canonical-checkpoints.md` for the definitive reference with md5 checksums.

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
| e125,150,175,195 | `HYP/runs/jepa_in21k_e200_280/training_folder/e{125,150,175,195}.pt` | 280 | Apr 21 | 234 |
| e200 | `HYP/runs/jepa_in21k_e200_280/training_folder/latest.pt` | 280 | Apr 22 | 234 |
| **Probes e125-e200** | `HYP/runs/jepa_ext_probes_332/jepa_e{125,150,175,200}_lvef/.../best.pt` | 332 | Apr 22 | — |
| e100 (old run, same init) | `HYP/runs/vjepa_mimic_pretrain_125/training_folder/e100.pt` | 125 | Jan 25 | 42 | **DO NOT USE for NeurIPS — different md5 from canonical. Probes trained on run 376.** |
| e110 | `HYP/runs/vjepa_mimic_pretrain_125/training_folder/e110.pt` | 125 | Jan 25 | 42 |
| e120 | `HYP/runs/vjepa_mimic_pretrain_125/training_folder/e120.pt` | 125 | Jan 25 | 42 |
| e130 | `HYP/runs/vjepa_mimic_pretrain_125/training_folder/e130.pt` | 125 | Jan 25 | 42 |
| e122-e154 | `HYP/runs/vjepa_mimic_pretrain_137/training_folder/e{122..154}.pt` | 137 | Jan 26 | 154 |
| e146-e168 | `HYP/runs/vjepa_mimic_pretrain_148/training_folder/e{146..168}.pt` | 148 | Jan 26-27 | 24 |
| e162-e238 | `HYP/runs/vjepa_mimic_pretrain_150/training_folder/e{162..238}.pt` | 150 | Jan 27-28 | 234 |

**Gaps:** Epochs 0-99 lost (trained on HyperPod NVMe, never synced to S3). Would need to retrain for training dynamics analysis at early epochs.

**Extended trajectory (2026-04-22):** Job 280 continued run 376 to e200 (`runs/jepa_in21k_e200_280/`). Job 332 trained EchoNet-Dynamic LVEF d=4 attentive probes at e125/150/175/200; val R² 0.685/0.700/0.717/0.715 — clean performance plateaus at e175-e200. Matched-frame inference on these 4 checkpoints is NOT YET RUN (extending the `tab:attn_traj` MAE-MF analysis to JEPA at extended training is pending).

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

### Extended-teacher experiments (2026-04-22 → 2026-04-24)

**Stage 1 extension:** Job 329 resumed `salt_s1` from e20 to e100, producing a longer-trained V-Pixel teacher for ablation.

| Epoch | S3 Path | Run | Date | Notes |
|-------|---------|-----|------|-------|
| e24, e29, ..., e99, latest | `HYP/runs/salt_s1_e100_resume_329/checkpoints/` | 329 | Apr 22 | Resumed from e20 via `pretrain-salt-s1-mimic-224px-16f-e100-resume.yaml`; loss 0.252 → 0.234 monotonic |

**Stage 2 with extended teachers:**

| Variant | Teacher | S3 Path | Run | State | Notes |
|---|---|---|---|---|---|
| V-Pixel teacher | salt_s1 e99 | `HYP/runs/salt_s2_vpixel_e99_teacher_330/checkpoints/` | 330 | **Complete** (80ep) | Loss 0.843→0.529 monotonic, 17 ckpts (e4…e79, latest). Probe job 349 queued. |
| JEPA teacher | jepa_in21k_e100 | `HYP/runs/salt_s2_jepa_teacher_335/checkpoints/` | 335 | **Running** (~e54/80) | Loss 0.685→0.416 at e54. Probe job 350 queued afterok:335. |

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
| JEPA IN21K extended (job 280) | e125, 150, 175, 195, 200 | probes trained in job 332 |
| JEPA (old, same init) | e100, e110, e120, e130, e122-e238 | e0-e99 lost |
| BYOL | e0-e50 (every 2ep), e85, e90, e95, e100, e105 | e52-e84 |
| MAE | e4-e124 (every 5ep) | None |
| SALT S2 | e4-e79 (every 5ep), e100 expected, e200 in progress | None (dense) |

---

## 9. CMR Cross-Modality (ViT-S, 2026-04-18 → 2026-04-24)

**Init:** `vits_in21k.pt` (ImageNet-21K supervised ViT-S, 2D, inflated to 3D)
**Data:** 21,840 SAX-only CMR clips from MnM + MnM2 + Sunnybrook + DSB2 + CMR-Multi
**Holdout:** ACDC (951 train / 538 test, 100/50 patients, EF + 5-class diagnosis labels)

| Experiment | S3 Path | Run | Date | Notes |
|---|---|---|---|---|
| CMR MAE ViT-S 800ep | `HYP/runs/mae_cmr_vits_183/training_folder/` | 183 | Apr 19 | Loss 1.12→0.27 monotonic; 800 ckpts |
| CMR MAE ACDC LVEF probe traj | `HYP/runs/cmr_probe_traj_209/` | 209 | Apr 19 | Best R²=0.133 at e800 |
| CMR MAE ACDC Dx probe + MF (ns=2) | `HYP/runs/cmr_dx_traj_281/` | 281 | Apr 20 | AUROC 0.759 at e800; matched-frame Δ ≈ 0 at all ckpts |
| CMR MAE ACDC Dx MF (ns=1) | `HYP/runs/cmr_dx_mf_ns1_282/` | 282 | Apr 20 | Confirms Δ ≈ 0 |
| CMR MAE ACDC LVEF MF (ns=1) | `HYP/runs/cmr_lvef_mf_ns1_283/` | 283 | Apr 20 | Confirms Δ ≈ 0 |
| CMR JEPA ViT-S 800ep (seed 234) | `HYP/runs/jepa_cmr_vits_333/training_folder/` | 333 | Apr 22-23 | Loss 0.60→0.35→0.505→0.43, non-monotonic rise |
| CMR JEPA resume e250 (seed 163) | `HYP/runs/jepa_cmr_vits_resume250_s163_344/training_folder/` | 344 | Apr 23-24 | Seed-independent rise — confirms 333 behavior |
| CMR JEPA ACDC LVEF probe traj | `HYP/runs/cmr_jepa_probe_traj_345/` | 345 | Apr 24 | Best R²=0.162 at e100, collapses to 0.069 at e800 |
| CMR JEPA slow-EMA (ema=0.99925) | `HYP/runs/jepa_cmr_vits_slowema_346/training_folder/` | 346 | Apr 24 (completed e295) | Loss rise muted (0.37→0.44) but present |
| CMR JEPA slow-EMA LVEF probes | `HYP/runs/cmr_jepa_probe_traj_slowema_375/` | 375 | Apr 24-25 (2h01m) | R² peaks 0.138 at e30, collapses to 0.089 at e295 |
| CMR JEPA slow-EMA Dx probes | `HYP/runs/cmr_jepa_dx_traj_slowema_376/` | 376 | Apr 25 (1h24m) | AUROC peaks 0.799 at e30, collapses to 0.766 at e295 — EMA-independent collapse confirmed |

See `experiments/cmr-cross-modality.md` for full results tables and interpretation.

---

## 9. Probe Checkpoints

### EchoNet-Dynamic LVEF (e100 init-matched, canonical)

| Model | S3 Path | EFS Path | Notes |
|-------|---------|----------|-------|
| JEPA IN21K e100 | `CLEAN/probes/end_lvef_e100/jepa_in21k_e100/best.pt` | TBD | R²=0.591 |
| BYOL e100 | `CLEAN/probes/end_lvef_e100/byol_e100/best.pt` | TBD | R²=0.468 |
| MAE e99 | `CLEAN/probes/end_lvef_e100/mae_e99/best.pt` | TBD | R²=0.445 |
| SALT v1 e79 | EFS: `evals/vitl/icml/salt_s2_e79_end_lvef_224/.../best.pt` | — | R²=0.414 |

### EchoNet-Dynamic LVEF (pt50 rebuttal, JEPA init confounded)

| Model | S3 Path | Notes |
|-------|---------|-------|
| JEPA (Meta init, confounded) | `CLEAN/probes/end_lvef_pt50/echojepa_l_pt50/best.pt` | Head 3, d=4 attentive. **Do not use for NeurIPS.** |
| BYOL pt50 | `CLEAN/probes/end_lvef_pt50/echobyol_l_pt50/best.pt` | Head 1, d=4 attentive |
| MAE pt50 | `CLEAN/probes/end_lvef_pt50/echomae_l_pt50/best.pt` | Head 5, d=4 attentive |

### CAMUS Segmentation (e100 init-matched)

| Model | S3 Path | EFS Path | Test Dice | Best Config |
|-------|---------|----------|-----------|-------------|
| JEPA IN21K e100 | `CLEAN/probes/camus_segmentation/jepa_in21k_e100/best_decoder.pt` | `results/segmentation/jepa_in21k_e100/lr5e-02_wd1e-04/best_decoder.pt` | 0.815 | lr5e-2, wd1e-4 |
| BYOL e100 | `CLEAN/probes/camus_segmentation/byol_e100/best_decoder.pt` | `results/segmentation/byol_e100/lr5e-02_wd1e-04/best_decoder.pt` | 0.825 | lr5e-2, wd1e-4 |
| MAE e99 | `CLEAN/probes/camus_segmentation/mae_e99/best_decoder.pt` | `results/segmentation/mae_e99/lr1e-02_wd1e-04/best_decoder.pt` | 0.827 | lr1e-2, wd1e-4 |

Grid summaries (7 HP configs per model) also on S3 at `CLEAN/probes/camus_segmentation/{model}/grid_summary.json`.

---

## 10. Key Decision: Initialization Confound

The ICML rebuttal used JEPA§2 (Meta V-JEPA 2 init) while BYOL and MAE used ImageNet-21K init. This was a confound — JEPA got a video-SSL head start. The ICML speckle probing claim ("23% less speckle") is **retracted** under init-matching (gap shrinks to 4%).

For NeurIPS, all models use **ImageNet-21K init** at e100. The surviving mechanism is **temporal structure encoding** (frame shuffling), not speckle filtering.

**Completed:**
1. ~~Verify JEPA§1 e100 loads correctly~~ — Done (job 376)
2. ~~Train LVEF probes on JEPA§1 e100~~ — Done (EchoNet-Dynamic R²=0.591)
3. ~~Train CAMUS probes on all e100 models~~ — Done (JEPA 0.815, BYOL 0.825, MAE 0.827)
4. Run frame shuffling + noise robustness with e100 probes — **Pending**
5. Run speckle probing with e100 encoders — **Done** (retracted: gap is 4% not 23%)
