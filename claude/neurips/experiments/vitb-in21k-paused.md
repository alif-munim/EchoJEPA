# ViT-B MAE IN21K on MIMIC — Paused Run (Job 307)

## Status

**Paused at epoch ~33–34** on 2026-04-21 (job 307, cancelled via `scancel 307`).
Training was healthy and making normal progress at stop time. Kept on ice so the
ViT-L JEPA run (job 280) has the compute while we decide whether the ViT-B
controlled comparison is still needed for NeurIPS.

Last observed metrics (from `log.txt`, e0–e33):
- e0:  loss 0.500, lr 8.9e-6 (start of warmup)
- e28: loss 0.173, lr 2.86e-4 (warmup ongoing, peak at e40 = 1e-4 **after LR scaling**)
- e33: loss ≈ 0.167 (monotone decrease, normal MAE trajectory)
- AMP `loss_scale` swinging 4096 → 131072 with periodic NaN `grad_norm` spikes
  at ~e24, e28 (handled by scaler halving; training did not diverge)

## Purpose

Controlled comparison for the NeurIPS temporal-shortcut narrative: same MIMIC
data / same compute budget / same tube-masking schedule as EchoMAE-L, differing
only in model size. Tests whether the spatial-shortcut collapse scales with
capacity. Complements the ViT-S CMR run (`cmr-cross-modality.md`) as the
low-capacity boundary condition.

## Configuration

| Field | Value |
|---|---|
| Model | `pretrain_videomae_base_patch16_224` (ViT-B, 94M) |
| Init | ImageNet-21K (`vitb_in21k.pt`) |
| Dataset | MIMIC-IV-Echo, 525K clips, 16 frames @ 8 fps |
| Mask | Tube, 90% ratio, decoder depth 4 |
| Peak LR | **1.0e-4** (reduced from ViT-L's 1.5e-4 for stability) |
| Min LR | 1e-5 |
| Warmup | 40 epochs |
| Total epochs | 100 |
| Effective BS | 1024 (32/GPU × 8 GPUs × 4 accum) |
| Hardware | 1× p5.48xlarge (8× H100 80GB), cluster `echojepa-h100-neurips` |
| Compute node | ip-10-0-50-148 |

## Paths

**Sbatch script**
- `scripts/videomae_pretrain_mimic_vitb_in21k.sbatch`

**Local checkpoints** (compute node `/opt/dlami/nvme`, ephemeral — only guaranteed while node is alive)
- `/opt/dlami/nvme/mae_b_21k_307/output/checkpoint-{4,9,14,19,24,29,34}.pth`
- `/opt/dlami/nvme/mae_b_21k_307/output/checkpoint-latest.pth` (epoch 33)
- `/opt/dlami/nvme/mae_b_21k_307/output/log.txt` (per-iter loss; **not** synced to S3)

**S3 checkpoint mirror** (synced every ~15 min during training)
- `s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts/runs/mae_vitb_in21k_307/training_folder/`
- Contents: `checkpoint-{4,9,14,19,24,29,34}.pth`, `checkpoint-latest.pth`
- Note: `log.txt` is **not** in the sync filter (only `*.pth|*.pt|*.yaml|*.json|*.csv`).
  Pull it from the compute node before the node is reclaimed if we want the full loss curve.

**Initial weights** (used for from-scratch restart if checkpoints are lost)
- `s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts/checkpoints/vitb_in21k.pt`

## Resuming

The sbatch script supports `RESUME_FROM_JOB=<prev_id>`, which pulls the prior
run's S3 `training_folder/` into the new workdir before `auto_resume` picks up
`checkpoint-latest.pth` (epoch 33 → resumes at e34).

```bash
# On the HyperPod controller (echojepa-h100-neurips):
cd ~/EchoJEPA-repo
# (deploy latest code via S3 tarball if repo has changed since 2026-04-21)
RESUME_FROM_JOB=307 sbatch scripts/videomae_pretrain_mimic_vitb_in21k.sbatch
```

The new job will:
1. Sync `s3://.../runs/mae_vitb_in21k_307/training_folder/` → local `${OUTPUT_DIR}/`
2. `auto_resume` picks `checkpoint-latest.pth` (epoch 34) since its epoch is ≥
   the highest numbered file (`checkpoint-34.pth`). Training resumes at e35.
3. Continues cosine schedule from the current step count.

## Caveats on resume

- **NaN `grad_norm` spikes** appeared every ~4 epochs during warmup, absorbed by
  the AMP scaler (no divergence). If they persist post-warmup (e > 40), drop
  peak LR to 7.5e-5 or raise gradient clipping.
- **Warmup ends at e40.** Verification/probing should happen at e45+ once the
  schedule is post-peak; earlier probes only measure warmup behaviour.
- `checkpoint-latest.pth` is written every epoch (post-d91b4d4 fix). Numbered
  checkpoints every 5 epochs.
- S3 sync of `log.txt` is **not** enabled — consider adding `--include "*.txt"`
  to the `sync_ckpts` filter in the sbatch before relaunch.

## Decision log

- **Launched**: 2026-04-21, cluster `echojepa-h100-neurips`.
- **Paused**: 2026-04-21 at ~e33–34, ~14h wall-clock. Rationale: free the
  compute for ViT-L JEPA job 280 while we reassess whether ViT-B scaling is in
  NeurIPS scope. Checkpoints and init weights are preserved on S3 so the run
  is resumable at negligible cost.
