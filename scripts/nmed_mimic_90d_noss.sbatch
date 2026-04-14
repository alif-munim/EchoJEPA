#!/bin/bash
#SBATCH --job-name=nmed-m90d-noss
#SBATCH --partition=ml-p5-48xlarge
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --output=/tmp/nmed_m90d_noss-%j.out
#SBATCH --error=/tmp/nmed_m90d_noss-%j.err
#SBATCH --time=1-00:00:00
#SBATCH --nodelist=ip-10-0-50-7

# Nature Medicine: 90d mortality attentive probe WITHOUT study_sampling
# Tests clip-level training (366K clips/epoch) vs study_sampling (3,205/epoch).
# Hypothesis: removing study_sampling should match or exceed CY sklearn Ensemble
# (0.883) by giving the attentive probe access to all clips, not just 1/study/epoch.
#
# Config:
#   EchoJEPA-G only, d=1 attentive probe, 15-head HP grid (5 LR x 3 WD)
#   15 epochs (vs 35 with study_sampling), batch_size=2 (ViT-G VRAM limit)
#   warmup=1.0 epoch (shorter — each epoch sees 366K clips, not 3,205)
#
# Estimated time:
#   ~3h/epoch (366K clips, BS=2, 8xH100) x 5 epochs = ~15h
#   + ~1h pred avg = ~16h total
#
# After training, runs prediction averaging on test set (79K clips).
# Results saved to NVMe, uploaded to S3, and symlinked to EFS.
#
# Baseline: mortality_90d with study_sampling = G 0.827 (test, pred avg)
# Target:   CY sklearn Ensemble = G 0.883 (test, pred avg)

set -eo pipefail

export PATH="/opt/vjepa2-312/bin:$PATH"
source /opt/vjepa2-312/bin/activate
set -u
cd /opt/vjepa2

# Fix cuBLAS 12.9 bf16 bug on H100
export LD_LIBRARY_PATH="/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/cublas/lib:/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/cudnn/lib:/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/nccl/lib:${LD_LIBRARY_PATH:-}"
export TORCH_HOME="/opt/dlami/nvme/torch_cache"

rm -rf /dev/shm/* 2>/dev/null || true

# ============================================================
# Paths
# ============================================================
S3_BASE="s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts"
S3_OUT="${S3_BASE}/nmed_mimic_noss/results"
LOCAL="/opt/dlami/nvme/nmed_m90d_noss"

CSV_DIR="${LOCAL}/csvs/mortality_90d"
ENCODER="${LOCAL}/encoder/pt-280-an81.pt"
OUT_DIR="${LOCAL}/output"
CKPT_DIR="${LOCAL}/checkpoints"

TASK="mortality_90d_noss"
MODEL_TAG="echojepa-g"
TAG="${TASK}-${MODEL_TAG}"
PA_TAG="${TASK}-predavg-${MODEL_TAG}"

ALL_DEVICES="cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7"
export MASTER_PORT=29500

log() { echo "[$(date +%H:%M:%S)] $*"; }

# ============================================================
# Step 0: Download assets from S3 to NVMe
# ============================================================
log "=== Downloading assets from S3 ==="
mkdir -p "${CSV_DIR}" "${LOCAL}/encoder" "${OUT_DIR}" "${CKPT_DIR}"

# EchoJEPA-G encoder (~16.5GB)
if [ ! -f "${ENCODER}" ]; then
    log "  Downloading G encoder..."
    aws s3 cp "${S3_BASE}/checkpoints/anneal/keep/pt-280-an81.pt" "${ENCODER}"
fi

# MIMIC mortality_90d CSVs
for f in train.csv val.csv test.csv; do
    if [ ! -f "${CSV_DIR}/${f}" ]; then
        log "  Downloading CSV: ${f}"
        aws s3 cp "${S3_BASE}/nmed_mimic_noss/probe_csvs/mortality_90d/${f}" "${CSV_DIR}/${f}"
    fi
done

log "=== All assets downloaded ==="

# ============================================================
# Validate inputs
# ============================================================
log "=== MIMIC 90d mortality — NO study_sampling ==="
log "  Train CSV: ${CSV_DIR}/train.csv ($(wc -l < "${CSV_DIR}/train.csv") clips)"
log "  Test CSV:  ${CSV_DIR}/test.csv ($(wc -l < "${CSV_DIR}/test.csv") clips)"
log "  Encoder:   ${ENCODER}"

if [ ! -f "${ENCODER}" ]; then
    log "ERROR: Encoder checkpoint not found"
    exit 1
fi

# ============================================================
# Step 1: Training config (study_sampling: false)
# ============================================================
TRAIN_CFG="/opt/dlami/nvme/nmed_m90d_noss_train.yaml"

cat > "${TRAIN_CFG}" <<YAML
app: vjepa
cpus_per_task: 32
folder: ${OUT_DIR}
mem_per_gpu: 80G
nodes: 1
tasks_per_node: 8
num_workers: 4

eval_name: video_classification_frozen
resume_checkpoint: true
tag: ${TAG}

experiment:
  classifier:
    task_type: classification
    num_heads: 16
    num_probe_blocks: 1
    num_targets: 2

  data:
    dataset_type: VideoDataset
    dataset_train: ${CSV_DIR}/train.csv
    dataset_val: ${CSV_DIR}/val.csv
    num_classes: 2
    resolution: 224
    frames_per_clip: 16
    frame_step: 2
    num_segments: 2
    num_views_per_segment: 1
    study_sampling: false

  optimization:
    batch_size: 2
    val_batch_size: 64
    num_epochs: 5
    use_bfloat16: true
    use_pos_embed: false
    multihead_kwargs:
    # 15-head grid: 5 LRs x 3 WDs (same grid as study_sampling runs)
    # Warmup reduced to 1.0 epoch (each epoch = 366K clips, not 3,205)
    # 5 epochs = 1.8M clips (16x the study_sampling budget of 112K)
    # LR=1e-3
    - {lr: 0.001,   start_lr: 0.0, warmup: 1.0, final_lr: 0.0, weight_decay: 0.001, final_weight_decay: 0.001}
    - {lr: 0.001,   start_lr: 0.0, warmup: 1.0, final_lr: 0.0, weight_decay: 0.01,  final_weight_decay: 0.01}
    - {lr: 0.001,   start_lr: 0.0, warmup: 1.0, final_lr: 0.0, weight_decay: 0.1,   final_weight_decay: 0.1}
    # LR=5e-4
    - {lr: 0.0005,  start_lr: 0.0, warmup: 1.0, final_lr: 0.0, weight_decay: 0.001, final_weight_decay: 0.001}
    - {lr: 0.0005,  start_lr: 0.0, warmup: 1.0, final_lr: 0.0, weight_decay: 0.01,  final_weight_decay: 0.01}
    - {lr: 0.0005,  start_lr: 0.0, warmup: 1.0, final_lr: 0.0, weight_decay: 0.1,   final_weight_decay: 0.1}
    # LR=1e-4
    - {lr: 0.0001,  start_lr: 0.0, warmup: 1.0, final_lr: 0.0, weight_decay: 0.001, final_weight_decay: 0.001}
    - {lr: 0.0001,  start_lr: 0.0, warmup: 1.0, final_lr: 0.0, weight_decay: 0.01,  final_weight_decay: 0.01}
    - {lr: 0.0001,  start_lr: 0.0, warmup: 1.0, final_lr: 0.0, weight_decay: 0.1,   final_weight_decay: 0.1}
    # LR=5e-5
    - {lr: 0.00005, start_lr: 0.0, warmup: 1.0, final_lr: 0.0, weight_decay: 0.001, final_weight_decay: 0.001}
    - {lr: 0.00005, start_lr: 0.0, warmup: 1.0, final_lr: 0.0, weight_decay: 0.01,  final_weight_decay: 0.01}
    - {lr: 0.00005, start_lr: 0.0, warmup: 1.0, final_lr: 0.0, weight_decay: 0.1,   final_weight_decay: 0.1}
    # LR=1e-5
    - {lr: 0.00001, start_lr: 0.0, warmup: 1.0, final_lr: 0.0, weight_decay: 0.001, final_weight_decay: 0.001}
    - {lr: 0.00001, start_lr: 0.0, warmup: 1.0, final_lr: 0.0, weight_decay: 0.01,  final_weight_decay: 0.01}
    - {lr: 0.00001, start_lr: 0.0, warmup: 1.0, final_lr: 0.0, weight_decay: 0.1,   final_weight_decay: 0.1}

model_kwargs:
  checkpoint: ${ENCODER}
  module_name: evals.video_classification_frozen.modelcustom.vit_encoder_multiclip
  pretrain_kwargs:
    encoder:
      checkpoint_key: target_encoder
      img_temporal_dim_size: null
      model_name: vit_giant_xformers
      patch_size: 16
      tubelet_size: 2
      uniform_power: true
      use_rope: true
  wrapper_kwargs:
    max_frames: 128
    use_pos_embed: false
YAML

log "Training config: ${TRAIN_CFG}"
log "  study_sampling: false"
log "  num_epochs: 5"
log "  warmup: 1.0 epoch"
log "  tag: ${TAG}"

# ============================================================
# Step 2: Run training
# ============================================================
log "=== Starting training ==="

# Kill orphaned workers
ps -eo pid,ppid,args | grep "multiprocessing.spawn" | grep -v grep | awk '$2 == 1 {print $1}' | xargs -r kill 2>/dev/null || true
ps -eo pid,ppid,args | grep "multiprocessing.resource_tracker" | grep -v grep | awk '$2 == 1 {print $1}' | xargs -r kill 2>/dev/null || true
sleep 2

TRAIN_START=$(date +%s)
rc=0
PYTHONUNBUFFERED=1 python -m evals.main --fname "${TRAIN_CFG}" --devices ${ALL_DEVICES} || rc=$?
TRAIN_END=$(date +%s)

if [ "$rc" -ne 0 ]; then
    log "TRAINING FAILED (exit ${rc}) after $(( (TRAIN_END - TRAIN_START) / 3600 ))h $(( ((TRAIN_END - TRAIN_START) % 3600) / 60 ))m"
    # Still try pred avg if best.pt exists (might have timed out mid-epoch)
fi

TRAIN_DIR="${OUT_DIR}/video_classification_frozen/${TAG}"
if [ -f "${TRAIN_DIR}/best.pt" ]; then
    log "Training complete. best.pt found."
    BEST_EPOCH=$(python3 -c "import torch; c=torch.load('${TRAIN_DIR}/best.pt', map_location='cpu'); print(c.get('epoch', '?'))" 2>/dev/null || echo "?")
    log "  Best epoch: ${BEST_EPOCH}"
    log "  Training time: $(( (TRAIN_END - TRAIN_START) / 3600 ))h $(( ((TRAIN_END - TRAIN_START) % 3600) / 60 ))m"

    # Archive checkpoint
    mkdir -p "${CKPT_DIR}/${MODEL_TAG}"
    cp "${TRAIN_DIR}/best.pt" "${CKPT_DIR}/${MODEL_TAG}/best.pt"
    cp "${TRAIN_DIR}/latest.pt" "${CKPT_DIR}/${MODEL_TAG}/latest.pt" 2>/dev/null || true
    cp "${TRAIN_DIR}/log_r0.csv" "${CKPT_DIR}/${MODEL_TAG}/log_r0.csv" 2>/dev/null || true
else
    log "ERROR: best.pt not found — training did not produce a checkpoint"
    exit 1
fi

# ============================================================
# Step 3: Prediction averaging config (test set)
# ============================================================
PA_CFG="/opt/dlami/nvme/nmed_m90d_noss_predavg.yaml"
PROBE_CKPT="${CKPT_DIR}/${MODEL_TAG}/best.pt"

cat > "${PA_CFG}" <<YAML
app: vjepa
cpus_per_task: 32
folder: ${OUT_DIR}
mem_per_gpu: 80G
nodes: 1
tasks_per_node: 8
num_workers: 2

eval_name: video_classification_frozen
val_only: true
resume_checkpoint: true
probe_checkpoint: ${PROBE_CKPT}
tag: ${PA_TAG}

experiment:
  classifier:
    task_type: classification
    num_heads: 16
    num_probe_blocks: 1
    num_targets: 2

  data:
    dataset_type: VideoDataset
    dataset_train: ${CSV_DIR}/train.csv
    dataset_val: ${CSV_DIR}/test.csv
    num_classes: 2
    resolution: 224
    frames_per_clip: 16
    frame_step: 2
    num_segments: 2
    num_views_per_segment: 1
    study_sampling: true
    prediction_averaging: true

  optimization:
    batch_size: 2
    val_batch_size: 128
    num_epochs: 1
    use_bfloat16: true
    use_pos_embed: false
    multihead_kwargs:
    # Must match training grid exactly (15 heads: 5 LR x 3 WD)
    - {lr: 0.001,   start_lr: 0.0, warmup: 1.0, final_lr: 0.0, weight_decay: 0.001, final_weight_decay: 0.001}
    - {lr: 0.001,   start_lr: 0.0, warmup: 1.0, final_lr: 0.0, weight_decay: 0.01,  final_weight_decay: 0.01}
    - {lr: 0.001,   start_lr: 0.0, warmup: 1.0, final_lr: 0.0, weight_decay: 0.1,   final_weight_decay: 0.1}
    - {lr: 0.0005,  start_lr: 0.0, warmup: 1.0, final_lr: 0.0, weight_decay: 0.001, final_weight_decay: 0.001}
    - {lr: 0.0005,  start_lr: 0.0, warmup: 1.0, final_lr: 0.0, weight_decay: 0.01,  final_weight_decay: 0.01}
    - {lr: 0.0005,  start_lr: 0.0, warmup: 1.0, final_lr: 0.0, weight_decay: 0.1,   final_weight_decay: 0.1}
    - {lr: 0.0001,  start_lr: 0.0, warmup: 1.0, final_lr: 0.0, weight_decay: 0.001, final_weight_decay: 0.001}
    - {lr: 0.0001,  start_lr: 0.0, warmup: 1.0, final_lr: 0.0, weight_decay: 0.01,  final_weight_decay: 0.01}
    - {lr: 0.0001,  start_lr: 0.0, warmup: 1.0, final_lr: 0.0, weight_decay: 0.1,   final_weight_decay: 0.1}
    - {lr: 0.00005, start_lr: 0.0, warmup: 1.0, final_lr: 0.0, weight_decay: 0.001, final_weight_decay: 0.001}
    - {lr: 0.00005, start_lr: 0.0, warmup: 1.0, final_lr: 0.0, weight_decay: 0.01,  final_weight_decay: 0.01}
    - {lr: 0.00005, start_lr: 0.0, warmup: 1.0, final_lr: 0.0, weight_decay: 0.1,   final_weight_decay: 0.1}
    - {lr: 0.00001, start_lr: 0.0, warmup: 1.0, final_lr: 0.0, weight_decay: 0.001, final_weight_decay: 0.001}
    - {lr: 0.00001, start_lr: 0.0, warmup: 1.0, final_lr: 0.0, weight_decay: 0.01,  final_weight_decay: 0.01}
    - {lr: 0.00001, start_lr: 0.0, warmup: 1.0, final_lr: 0.0, weight_decay: 0.1,   final_weight_decay: 0.1}

model_kwargs:
  checkpoint: ${ENCODER}
  module_name: evals.video_classification_frozen.modelcustom.vit_encoder_multiclip
  pretrain_kwargs:
    encoder:
      checkpoint_key: target_encoder
      img_temporal_dim_size: null
      model_name: vit_giant_xformers
      patch_size: 16
      tubelet_size: 2
      uniform_power: true
      use_rope: true
  wrapper_kwargs:
    max_frames: 128
    use_pos_embed: false
YAML

# ============================================================
# Step 4: Run prediction averaging on test set
# ============================================================
log "=== Starting prediction averaging (test set, 79K clips) ==="

# Clear stale output dir (Bug 012)
PA_OUT="${OUT_DIR}/video_classification_frozen/${PA_TAG}"
rm -rf "${PA_OUT}" 2>/dev/null || true

# Kill orphaned workers
ps -eo pid,ppid,args | grep "multiprocessing.spawn" | grep -v grep | awk '$2 == 1 {print $1}' | xargs -r kill 2>/dev/null || true
ps -eo pid,ppid,args | grep "multiprocessing.resource_tracker" | grep -v grep | awk '$2 == 1 {print $1}' | xargs -r kill 2>/dev/null || true
sleep 2

PA_START=$(date +%s)
pa_rc=0
PYTHONUNBUFFERED=1 python -m evals.main --fname "${PA_CFG}" --devices ${ALL_DEVICES} || pa_rc=$?
PA_END=$(date +%s)

if [ "$pa_rc" -ne 0 ]; then
    log "PRED AVG FAILED (exit ${pa_rc}) after $(( (PA_END - PA_START) / 60 ))m"
else
    log "PRED AVG DONE in $(( (PA_END - PA_START) / 60 ))m"
fi

# ============================================================
# Step 5: Upload results to S3
# ============================================================
log "=== Uploading results to S3 ==="

# Training checkpoint
aws s3 cp "${CKPT_DIR}/${MODEL_TAG}/best.pt" "${S3_OUT}/checkpoints/${TASK}/${MODEL_TAG}/best.pt" --quiet 2>/dev/null && \
    log "  Uploaded: best.pt" || log "  WARNING: best.pt upload failed"
aws s3 cp "${CKPT_DIR}/${MODEL_TAG}/log_r0.csv" "${S3_OUT}/checkpoints/${TASK}/${MODEL_TAG}/log_r0.csv" --quiet 2>/dev/null || true

# Pred avg results
if [ -d "${PA_OUT}" ]; then
    aws s3 sync "${PA_OUT}" "${S3_OUT}/${PA_TAG}/" --quiet 2>/dev/null && \
        log "  Uploaded: ${PA_TAG}/" || log "  WARNING: pred avg upload failed"
fi

# Training log
if [ -f "${TRAIN_DIR}/log_r0.csv" ]; then
    aws s3 cp "${TRAIN_DIR}/log_r0.csv" "${S3_OUT}/${TAG}/log_r0.csv" --quiet 2>/dev/null || true
fi

# ============================================================
# Step 6: Results are on NVMe + S3 (no EFS on compute node)
# ============================================================
# EFS copy must be done from SageMaker after job completes:
#   aws s3 sync s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts/nmed_mimic_noss/results/ \
#     $EFS/evals/vitg-384/nature_medicine/mimic/video_classification_frozen/
log "Results on NVMe + S3. Copy to EFS from SageMaker when complete."

# ============================================================
# Summary
# ============================================================
TOTAL_END=$(date +%s)
log "=== Job complete ==="
log "  Total time: $(( (TOTAL_END - TRAIN_START) / 3600 ))h $(( ((TOTAL_END - TRAIN_START) % 3600) / 60 ))m"
log "  Training: $(( (TRAIN_END - TRAIN_START) / 3600 ))h $(( ((TRAIN_END - TRAIN_START) % 3600) / 60 ))m"
log "  Pred avg: $(( (PA_END - PA_START) / 60 ))m"
log "  Results: ${PA_OUT}"
log "  S3: ${S3_OUT}/${PA_TAG}/"
log "  EFS: ${EFS_OUT}/${PA_TAG}/"
log ""
log "  Baseline (study_sampling=true): G 0.827 AUROC"
log "  Target (CY sklearn Ensemble):   G 0.883 AUROC"
log "  Check: python scripts/compute_predavg_stats.py --task mortality_90d_noss"
