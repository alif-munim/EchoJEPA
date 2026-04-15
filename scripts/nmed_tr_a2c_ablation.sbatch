#!/bin/bash
#SBATCH --job-name=nmed-tr-a2c
#SBATCH --partition=ml-p5-48xlarge
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --output=/tmp/nmed_tr_a2c-%j.out
#SBATCH --error=/tmp/nmed_tr_a2c-%j.err
#SBATCH --time=0-04:00:00
#SBATCH --nodelist=ip-10-0-50-7

# Nature Medicine: TR severity A2C cross-view ablation (inference only)
# Reuses existing multi-view TR severity probes, runs inference on A2C-only B-mode clips.
# TR severity training views: A4C, Subcostal, PLAX (B-mode) — A2C is excluded.
# Same probe, restricted input — isolates the cross-view generalization.
#
# 1 task × 3 models (G, EP, Pan) = 3 inference runs
#   TR severity (5-class): ~30-40 min each (88K A2C clips × 1B/34M/43M params)
#
# Estimated total: ~2h
#
# Then submits mortality_90d_noss training job.

set -eo pipefail

export PATH="/opt/vjepa2-312/bin:$PATH"
source /opt/vjepa2-312/bin/activate
set -u
cd /opt/vjepa2

# Fix cuBLAS 12.9 bf16 bug on H100
export LD_LIBRARY_PATH="/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/cublas/lib:/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/cudnn/lib:/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/nccl/lib:${LD_LIBRARY_PATH:-}"
export TORCH_HOME="/opt/dlami/nvme/torch_cache"

rm -rf /dev/shm/* 2>/dev/null || true

S3_BASE="s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts"
LOCAL="/opt/dlami/nvme/nmed_tr_a2c"
OUT_DIR="${LOCAL}/output"
S3_OUT="${S3_BASE}/nmed_tr_a2c/results/uhn"
ALL_DEVICES="cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7"
export MASTER_PORT=29500

log() { echo "[$(date +%H:%M:%S)] $*"; }

# ============================================================
# Step 1: Download assets from S3
# ============================================================
log "=== Downloading assets from S3 ==="
mkdir -p ${LOCAL}/{encoder,probes,csvs/tr_severity_a2c}

# EchoJEPA-G encoder (~16.5GB)
if [ ! -f "${LOCAL}/encoder/pt-280-an81.pt" ]; then
    log "  Downloading G encoder..."
    aws s3 cp "${S3_BASE}/checkpoints/anneal/keep/pt-280-an81.pt" "${LOCAL}/encoder/pt-280-an81.pt"
fi

# Probe checkpoints for all 3 models
for MODEL in echojepa-g echoprime panecho; do
    DEST="${LOCAL}/probes/tr_severity/${MODEL}"
    mkdir -p "$DEST"
    if [ ! -f "${DEST}/best.pt" ]; then
        log "  Downloading probe: tr_severity/${MODEL}"
        aws s3 cp "${S3_BASE}/checkpoints/probes/tr_severity/${MODEL}/best.pt" "${DEST}/best.pt"
    fi
done

# A2C-only probe CSVs
for f in train_vf.csv test_vf.csv; do
    if [ ! -f "${LOCAL}/csvs/tr_severity_a2c/${f}" ]; then
        aws s3 cp "${S3_BASE}/nmed_tr_a2c/probe_csvs/tr_severity_a2c/${f}" "${LOCAL}/csvs/tr_severity_a2c/${f}"
    fi
done

log "=== All assets downloaded ==="

# ============================================================
# Multihead kwargs block (12 heads: 4 LR × 3 WD)
# Must match the training grid exactly
# ============================================================
MULTIHEAD_BLOCK="    multihead_kwargs:
    - {lr: 0.0005, start_lr: 0.0, warmup: 2.0, final_lr: 0.0, weight_decay: 0.001, final_weight_decay: 0.001}
    - {lr: 0.0005, start_lr: 0.0, warmup: 2.0, final_lr: 0.0, weight_decay: 0.01,  final_weight_decay: 0.01}
    - {lr: 0.0005, start_lr: 0.0, warmup: 2.0, final_lr: 0.0, weight_decay: 0.1,   final_weight_decay: 0.1}
    - {lr: 0.0001, start_lr: 0.0, warmup: 2.0, final_lr: 0.0, weight_decay: 0.001, final_weight_decay: 0.001}
    - {lr: 0.0001, start_lr: 0.0, warmup: 2.0, final_lr: 0.0, weight_decay: 0.01,  final_weight_decay: 0.01}
    - {lr: 0.0001, start_lr: 0.0, warmup: 2.0, final_lr: 0.0, weight_decay: 0.1,   final_weight_decay: 0.1}
    - {lr: 0.00005, start_lr: 0.0, warmup: 2.0, final_lr: 0.0, weight_decay: 0.001, final_weight_decay: 0.001}
    - {lr: 0.00005, start_lr: 0.0, warmup: 2.0, final_lr: 0.0, weight_decay: 0.01,  final_weight_decay: 0.01}
    - {lr: 0.00005, start_lr: 0.0, warmup: 2.0, final_lr: 0.0, weight_decay: 0.1,   final_weight_decay: 0.1}
    - {lr: 0.00001, start_lr: 0.0, warmup: 2.0, final_lr: 0.0, weight_decay: 0.001, final_weight_decay: 0.001}
    - {lr: 0.00001, start_lr: 0.0, warmup: 2.0, final_lr: 0.0, weight_decay: 0.01,  final_weight_decay: 0.01}
    - {lr: 0.00001, start_lr: 0.0, warmup: 2.0, final_lr: 0.0, weight_decay: 0.1,   final_weight_decay: 0.1}"

# ============================================================
# Model kwargs templates
# ============================================================
MODEL_KWARGS_G="model_kwargs:
  checkpoint: ${LOCAL}/encoder/pt-280-an81.pt
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
    use_pos_embed: false"

MODEL_KWARGS_EP="model_kwargs:
  checkpoint: null
  module_name: evals.video_classification_frozen.modelcustom.echo_prime_encoder
  pretrain_kwargs: {}
  wrapper_kwargs:
    echo_prime_root: /opt/vjepa2/evals/video_classification_frozen/modelcustom/EchoPrime
    force_fp32: true
    bin_size: 50"

MODEL_KWARGS_PAN="model_kwargs:
  checkpoint: null
  module_name: evals.video_classification_frozen.modelcustom.panecho_encoder
  pretrain_kwargs: {}
  wrapper_kwargs: {}"

# Normalization block for EP/Pan (0-1 instead of ImageNet)
NORM_BLOCK="    normalization:
      - [0.0, 0.0, 0.0]
      - [1.0, 1.0, 1.0]"

# ============================================================
# Run inference for a single model
# ============================================================
run_single() {
    local MODEL_TAG="$1"
    local PROBE_CKPT="$2"
    local MODEL_KWARGS="$3"
    local NORM="$4"     # "yes" or "no"

    local TAG="tr_severity_a2c-predavg-${MODEL_TAG}"
    local CFG="/opt/dlami/nvme/nmed_tr_a2c_cfg_${MODEL_TAG}.yaml"
    local TRAIN_CSV="${LOCAL}/csvs/tr_severity_a2c/train_vf.csv"
    local TEST_CSV="${LOCAL}/csvs/tr_severity_a2c/test_vf.csv"

    local NORM_YAML=""
    if [ "$NORM" = "yes" ]; then
        NORM_YAML="${NORM_BLOCK}"
    fi

    log "--- ${TAG} ---"
    log "  Probe: ${PROBE_CKPT}"

    # Clear stale output
    rm -rf "${OUT_DIR}/video_classification_frozen/${TAG}" 2>/dev/null || true

    cat > "${CFG}" <<YAML
app: vjepa
cpus_per_task: 32
folder: ${OUT_DIR}
mem_per_gpu: 80G
nodes: 1
tasks_per_node: 8
num_workers: 4

eval_name: video_classification_frozen
val_only: true
resume_checkpoint: true
probe_checkpoint: ${PROBE_CKPT}
tag: ${TAG}

experiment:
  classifier:
    task_type: classification
    num_heads: 16
    num_probe_blocks: 1
    num_targets: 5

  data:
    dataset_type: VideoDataset
    dataset_train: ${TRAIN_CSV}
    dataset_val: ${TEST_CSV}
    num_classes: 5
    resolution: 224
    frames_per_clip: 16
    frame_step: 2
    num_segments: 2
    num_views_per_segment: 1
    study_sampling: true
    prediction_averaging: true
${NORM_YAML}

  optimization:
    batch_size: 2
    val_batch_size: 128
    num_epochs: 1
    use_bfloat16: true
    use_pos_embed: false
${MULTIHEAD_BLOCK}

${MODEL_KWARGS}
YAML

    # Kill orphaned workers
    ps -eo pid,ppid,args | grep "multiprocessing.spawn" | grep -v grep | awk '$2 == 1 {print $1}' | xargs -r kill 2>/dev/null || true
    ps -eo pid,ppid,args | grep "multiprocessing.resource_tracker" | grep -v grep | awk '$2 == 1 {print $1}' | xargs -r kill 2>/dev/null || true
    sleep 2

    local START=$(date +%s)
    local rc=0
    PYTHONUNBUFFERED=1 python -m evals.main --fname "${CFG}" --devices ${ALL_DEVICES} || rc=$?
    local END=$(date +%s)

    if [ "$rc" -ne 0 ]; then
        log "  FAILED (exit ${rc}) after $(( (END - START) / 60 ))m"
    else
        log "  DONE in $(( (END - START) / 60 ))m"
        # Upload to S3
        aws s3 sync "${OUT_DIR}/video_classification_frozen/${TAG}" \
            "${S3_OUT}/${TAG}/" --exclude "*.pt" --quiet 2>/dev/null && \
            log "  Uploaded to S3" || log "  WARNING: S3 upload failed"
    fi

    return $rc
}

# ============================================================
# Run all 3 models
# ============================================================
TOTAL_START=$(date +%s)
FAILURES=0

log "=== TR severity A2C cross-view ablation: 3 models ==="

# EchoJEPA-G (~40 min: 88K clips × 1B params)
run_single "echojepa-g" "${LOCAL}/probes/tr_severity/echojepa-g/best.pt" "$MODEL_KWARGS_G" "no" || FAILURES=$((FAILURES + 1))

# EchoPrime (~15 min: 88K clips × 34M params)
run_single "echoprime" "${LOCAL}/probes/tr_severity/echoprime/best.pt" "$MODEL_KWARGS_EP" "yes" || FAILURES=$((FAILURES + 1))

# PanEcho (~15 min: 88K clips × 43M params)
run_single "panecho" "${LOCAL}/probes/tr_severity/panecho/best.pt" "$MODEL_KWARGS_PAN" "yes" || FAILURES=$((FAILURES + 1))

TOTAL_END=$(date +%s)
log "=== Results: $((3 - FAILURES))/3 succeeded, ${FAILURES} failed ==="
log "  Total time: $(( (TOTAL_END - TOTAL_START) / 3600 ))h $(( ((TOTAL_END - TOTAL_START) % 3600) / 60 ))m"

# ============================================================
# Cleanup and chain mortality job
# ============================================================
log "=== Cleaning up ==="
rm -rf "${LOCAL}/encoder" "${LOCAL}/probes" "${LOCAL}/csvs" "${TORCH_HOME}" 2>/dev/null || true

log "=== Results at S3: ${S3_OUT}/ ==="
log "  Sync to EFS from SageMaker:"
log "    aws s3 sync ${S3_OUT}/ \$EFS/evals/vitg-384/nature_medicine/uhn/video_classification_frozen/ --exclude '*.pt'"

# Chain: submit mortality job (use /opt/vjepa2 path — deployed code on compute node)
log "=== Submitting mortality_90d_noss job ==="
sbatch /opt/vjepa2/scripts/nmed_mimic_90d_noss.sbatch && log "Mortality job submitted" || log "WARNING: mortality sbatch failed (submit manually from controller)"

log "=== All done ==="
