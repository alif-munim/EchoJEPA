#!/bin/bash
# Upload assets needed for Nature Medicine predavg inference to S3.
# Run this ONCE from SageMaker before submitting the HyperPod sbatch.
#
# What's already on S3 (no action needed):
#   - EchoJEPA-G encoder: s3://...vjepa2-artifacts/checkpoints/anneal/keep/pt-280-an81.pt (16.5GB)
#   - All 6 probe checkpoints: s3://...vjepa2-artifacts/checkpoints/probes/{lvef,mr_severity}/{echojepa-g,echoprime,panecho}/best.pt
#
# What this script uploads:
#   - EchoPrime encoder weights (133MB)
#   - PanEcho pretrained weights (downloaded from GitHub, ~170MB)
#   - Probe CSVs for LVEF and MR severity (~550MB total)
#   - zscore_params.json for LVEF regression
#
# Usage:
#   bash scripts/prep_nmed_predavg_s3.sh

set -euo pipefail

S3_BASE="s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts"
EFS="/mnt/custom-file-systems/efs/fs-0049217cdf69186d7_fsap-0fa7145b64eaa046b/vjepa2"

log() { echo "[$(date +%H:%M:%S)] $*"; }

# --- 1. EchoPrime encoder weights ---
log "Uploading EchoPrime encoder weights..."
aws s3 cp \
    "${EFS}/evals/video_classification_frozen/modelcustom/EchoPrime/model_data/weights/echo_prime_encoder.pt" \
    "${S3_BASE}/checkpoints/echoprime/echo_prime_encoder.pt"

# --- 2. PanEcho pretrained weights ---
PANECHO_URL="https://github.com/CarDS-Yale/PanEcho/releases/download/v1.0/panecho.pt"
PANECHO_LOCAL="/tmp/panecho_pretrained.pt"
if [ ! -f "$PANECHO_LOCAL" ]; then
    log "Downloading PanEcho weights from GitHub..."
    wget -q -O "$PANECHO_LOCAL" "$PANECHO_URL"
fi
log "Uploading PanEcho weights to S3..."
aws s3 cp "$PANECHO_LOCAL" "${S3_BASE}/checkpoints/panecho/panecho.pt"

# --- 3. Probe CSVs ---
for TASK in lvef mr_severity; do
    CSV_DIR="${EFS}/experiments/nature_medicine/uhn/probe_csvs/${TASK}"
    S3_CSV="${S3_BASE}/nmed_predavg/probe_csvs/${TASK}"
    log "Uploading ${TASK} CSVs..."
    for f in train_vf.csv test_vf.csv; do
        aws s3 cp "${CSV_DIR}/${f}" "${S3_CSV}/${f}"
    done
    # zscore_params.json (regression tasks only)
    if [ -f "${CSV_DIR}/zscore_params.json" ]; then
        aws s3 cp "${CSV_DIR}/zscore_params.json" "${S3_CSV}/zscore_params.json"
    fi
done

log "=== All assets uploaded to ${S3_BASE}/nmed_predavg/ ==="
log ""
log "S3 layout:"
log "  ${S3_BASE}/checkpoints/anneal/keep/pt-280-an81.pt          (G encoder, already there)"
log "  ${S3_BASE}/checkpoints/probes/{task}/{model}/best.pt        (probe ckpts, already there)"
log "  ${S3_BASE}/checkpoints/echoprime/echo_prime_encoder.pt      (uploaded)"
log "  ${S3_BASE}/checkpoints/panecho/panecho.pt                   (uploaded)"
log "  ${S3_BASE}/nmed_predavg/probe_csvs/{task}/{train_vf,test_vf}.csv  (uploaded)"
log ""
log "Next: connect to HyperPod controller, deploy code, and run:"
log "  sbatch scripts/nmed_lvef_mr_predavg.sbatch"
