# NeurIPS Canonical Checkpoints — Definitive Reference

**Created:** 2026-04-09
**Purpose:** Single source of truth for encoder + probe pairs. Prevents the encoder mismatch bug discovered on 2026-04-09.

---

## CRITICAL: Encoder Mismatch Bug (2026-04-09)

The JEPA encoder at `HYP/runs/vjepa_mimic_pretrain_125/training_folder/e100.pt` (run 125, seed 42) is **NOT** the same as the canonical `CLEAN/encoders/jepa_in21k_vitl_e100.pt` (run 376, seed 234). They have different md5 checksums:

- **WRONG:** `4c521e4b8464dd88b1c36ead69060f56` — run 125 (old run)
- **CORRECT:** `0893de1639fd61ff9df796ef18e144ff` — run 376 (canonical)

The END LVEF probes were trained with the run 376 encoder. Using the run 125 encoder with run 376 probes produces near-random predictions (Pearson 0.05 instead of 0.80).

**Rule: Always use `s3://echodata25/neurips/` (CLEAN) paths for all NeurIPS experiments.** Never use `HYP/runs/` paths for JEPA — those reference the old run 125.

The other three encoders (BYOL, MAE, SALT) have only one training run each, so there is no ambiguity. Checksums confirmed identical between HYP and CLEAN copies.

---

## Canonical Encoders

**S3 prefix:** `s3://echodata25/neurips/encoders/`

All encoders use ImageNet-21K init (controlled comparison). ViT-L, 304M params, patch 16, tubelet 2, 224px.

| Model | S3 Filename | Size | md5 | Training Run |
|-------|-------------|------|-----|-------------|
| **JEPA IN21K e100** | `jepa_in21k_vitl_e100.pt` | 4.8 GB | `0893de1639fd61ff9df796ef18e144ff` | Job 376, seed 234 |
| **BYOL e100** | `byol_vitl_e100.pt` | 2.3 GB | `b7c4a6d76945aebdddde9ecd4cbd0d4b` | Job 342 |
| **MAE e99** | `mae_vitl_e99.pth` | 3.6 GB | `2ff18369993ff34a4d84ae55a9166ce5` | Job 354 |
| **SALT S2v1 e79** | `salt_s2_vitl_e79.pt` | 3.7 GB | `1782f5e475a14954219dfe8ede181965` | Job 388 |

### Checkpoint format notes

| Model | Top-level keys | Encoder key | module_name |
|-------|---------------|-------------|-------------|
| JEPA | encoder, target_encoder, predictor, opt, scaler, epoch | `target_encoder` | `vit_encoder_multiclip` |
| BYOL | encoder, target_encoder, opt, scaler, epoch | `target_encoder` | `vit_encoder_multiclip` |
| MAE | model, optimizer, epoch, scaler, args | (use `videomae_encoder` module) | `videomae_encoder` |
| SALT | encoder, opt, scaler, epoch + extra keys | `encoder` (NOT target_encoder) | `vit_encoder_multiclip` |

### SALT loading warning

SALT loads with `_IncompatibleKeys(missing_keys=['norm.weight', 'norm.bias'], unexpected_keys=['img_mod_embed', 'video_mod_embed', 'norms_block.*'])`. This is expected — the `norms_block` layers exist in the SALT teacher but not in the standard ViT encoder. The probes were trained with this same loading behavior. Results are consistent.

---

## Canonical END LVEF Probes (d=4, attentive, 6-head HP grid)

**S3 prefix:** `s3://echodata25/neurips/probes/end_lvef_e100/`

All probes trained on EchoNet-Dynamic (7465 train / 1288 val), 20 epochs, 6-head HP grid, z-score normalization (mean=55.7776, std=12.4072 — embedded in checkpoint).

| Model | S3 Path | md5 | Best Head | Best Val MAE |
|-------|---------|-----|-----------|-------------|
| JEPA IN21K e100 | `jepa_in21k_e100/best.pt` | `b2a7351010c7a918847d3f63f6dcb7c2` | Head 3 | 5.219 |
| BYOL e100 | `byol_e100/best.pt` | (verify) | Head 1 | 5.898 |
| MAE e99 | `mae_e99/best.pt` | (verify) | Head 3 | 6.046 |
| SALT v1 e79 | `salt_v1_e79/best.pt` | (verify) | Head 2 | 6.473 |

### Probe loading note

The probe checkpoint contains 6 classifier state dicts (one per HP head). When loading with 1 `multihead_kwargs` entry, only head 0 is loaded via `zip()`. Head 0 is close to the best head for JEPA (5.275 vs 5.219) but the gap is larger for other models. For publication results, use 6 `multihead_kwargs` entries or modify `load_checkpoint` to select the best head.

---

## Verified END Test Set Results (2026-04-09)

1277 EchoNet-Dynamic test videos, single GPU, head 0 (not best head):

| Model | R² | Pearson | MAE | Expected (best head, documented) |
|-------|-----|---------|-----|----------------------------------|
| **JEPA** | **0.645** | **0.804** | **5.28** | R²=0.652, Pearson=0.808, MAE=5.32 |
| BYOL | 0.462 | 0.691 | 6.40 | R²=0.511, Pearson=0.720, MAE=6.18 |
| MAE | 0.402 | 0.657 | 6.89 | R²=0.447, Pearson=0.688, MAE=6.59 |
| SALT | 0.329 | 0.584 | 7.17 | R²=0.416, Pearson=0.659, MAE=6.66 |

Ranking preserved: JEPA > BYOL > MAE > SALT. All verified on 2026-04-09.

---

## Zero-Shot Pediatric LVEF Results (2026-04-09)

368 EchoNet-Pediatric test clips, END probes (no retraining), bootstrap 95% CIs (10K resamples):

| Model | R² [95% CI] | MAE [95% CI] | Pearson [95% CI] |
|-------|-------------|-------------|-----------------|
| **JEPA** | **0.342** [0.05, 0.52] | **7.08** [6.45, 7.73] | **0.719** [0.63, 0.79] |
| MAE | 0.307 [0.14, 0.41] | 7.40 [6.78, 8.04] | 0.620 [0.51, 0.71] |
| BYOL | -0.338 [-0.92, 0.02] | 10.34 [9.48, 11.23] | 0.561 [0.44, 0.66] |
| SALT | -0.018 [-0.27, 0.16] | 8.00 [7.16, 8.92] | 0.314 [0.19, 0.44] |

Note: Pediatric CSV labels are z-scored (scaler mean=61.03, std=10.44). The eval code re-normalizes using the probe's embedded END stats (55.78, 12.41), so `label_real` in the predictions CSV = pediatric z-scores and `pred_real` = real EF. The bootstrap script un-normalizes labels with the pediatric scaler.

---

## Training Dynamics Encoders

**S3 prefix:** `s3://echodata25/neurips/encoders/`

| Epoch | JEPA | BYOL | MAE | SALT |
|-------|------|------|-----|------|
| ~25 | `jepa_in21k_vitl_e25.pt` | `byol_vitl_e24.pt` | `mae_vitl_e24.pth` | `salt_s2_vitl_e29.pt` |
| ~50 | `jepa_in21k_vitl_e50.pt` | `byol_vitl_e50.pt` | `mae_vitl_e50.pth` | `salt_s2_vitl_e49.pt` |
| ~75 | `jepa_in21k_vitl_e75.pt` | `byol_vitl_e75.pt` | `mae_vitl_e74.pth` | `salt_s2_vitl_e79.pt` |
| ~100 | `jepa_in21k_vitl_e100.pt` | `byol_vitl_e100.pt` | `mae_vitl_e99.pth` | — |

---

## Download Template (for sbatch scripts)

```bash
CLEAN="s3://echodata25/neurips"
NVME="/opt/dlami/nvme"

# Encoders (use CLEAN, never HYP for JEPA)
aws s3 cp ${CLEAN}/encoders/jepa_in21k_vitl_e100.pt  ${NVME}/checkpoints/
aws s3 cp ${CLEAN}/encoders/byol_vitl_e100.pt         ${NVME}/checkpoints/
aws s3 cp ${CLEAN}/encoders/mae_vitl_e99.pth           ${NVME}/checkpoints/
aws s3 cp ${CLEAN}/encoders/salt_s2_vitl_e79.pt        ${NVME}/checkpoints/

# END LVEF probes
aws s3 cp ${CLEAN}/probes/end_lvef_e100/jepa_in21k_e100/best.pt ${NVME}/probes/end_lvef_e100/jepa_in21k_e100/
aws s3 cp ${CLEAN}/probes/end_lvef_e100/byol_e100/best.pt       ${NVME}/probes/end_lvef_e100/byol_e100/
aws s3 cp ${CLEAN}/probes/end_lvef_e100/mae_e99/best.pt         ${NVME}/probes/end_lvef_e100/mae_e99/
aws s3 cp ${CLEAN}/probes/end_lvef_e100/salt_v1_e79/best.pt     ${NVME}/probes/end_lvef_e100/salt_v1_e79/
```

### Verification after download

```bash
echo "0893de1639fd61ff9df796ef18e144ff" | md5sum -c <(echo "0893de1639fd61ff9df796ef18e144ff  ${NVME}/checkpoints/jepa_in21k_vitl_e100.pt")
```
