#!/bin/bash
#SBATCH --job-name=nmed-traj-pa
#SBATCH --partition=ml-p5-48xlarge
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --output=/tmp/nmed_traj_pa-%j.out
#SBATCH --error=/tmp/nmed_traj_pa-%j.err
#SBATCH --time=1-00:00:00
#SBATCH --nodelist=ip-10-0-50-7

# Trajectory onset prediction averaging (8 runs: 2 tasks × 4 models)
# Then submits mortality_90d_noss training job via sbatch dependency.
#
# Tasks: trajectory_tr_severity_onset, trajectory_tapse_onset
# Models: echojepa-g, echojepa-l-k, echoprime, panecho
#
# Probes from:
#   G/L-K: /opt/dlami/nvme/nmed_traj_onset/output/video_classification_frozen/
#   EP/Pan: /opt/dlami/nvme/nmed_fixup/output/video_classification_frozen/
#
# Estimated time: ~20-30 min per run × 8 = ~3-4h total

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
S3_OUT="${S3_BASE}/nmed_traj_onset/results/uhn"
PA_OUT_BASE="/opt/dlami/nvme/nmed_traj_pa/output"
ALL_DEVICES="cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7"
export MASTER_PORT=29500

log() { echo "[$(date +%H:%M:%S)] $*"; }

mkdir -p "${PA_OUT_BASE}"

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
G_ENCODER="/opt/dlami/nvme/nmed_traj_onset/encoder/pt-280-an81.pt"
LK_ENCODER="/opt/dlami/nvme/nmed_traj_onset/encoder/vitl-kinetics-pt220-an55.pt"

MODEL_KWARGS_G="model_kwargs:
  checkpoint: ${G_ENCODER}
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

MODEL_KWARGS_LK="model_kwargs:
  checkpoint: ${LK_ENCODER}
  module_name: evals.video_classification_frozen.modelcustom.vit_encoder_multiclip
  pretrain_kwargs:
    encoder:
      checkpoint_key: target_encoder
      img_temporal_dim_size: null
      model_name: vit_large
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
# Run predavg for a single task × model
# ============================================================
run_predavg() {
    local TASK="$1"
    local MODEL_TAG="$2"
    local PROBE_CKPT="$3"
    local MODEL_KWARGS="$4"
    local NORM="$5"     # "yes" or "no"

    local PA_TAG="${TASK}-predavg-${MODEL_TAG}"
    local CFG="/opt/dlami/nvme/nmed_traj_pa_cfg_${TASK}-${MODEL_TAG}.yaml"

    # CSV dir — G/L-K trained from nmed_traj_onset, EP/Pan from nmed_fixup
    # Both have the same CSVs, use whichever exists
    local CSV_DIR="/opt/dlami/nvme/nmed_traj_onset/csvs/${TASK}"
    if [ ! -d "$CSV_DIR" ]; then
        CSV_DIR="/opt/dlami/nvme/nmed_fixup/csvs/${TASK}"
    fi

    local NORM_YAML=""
    if [ "$NORM" = "yes" ]; then
        NORM_YAML="${NORM_BLOCK}"
    fi

    log "--- ${PA_TAG} ---"
    log "  Probe: ${PROBE_CKPT}"
    log "  CSV:   ${CSV_DIR}/test.csv"

    # Clear stale output (Bug 012)
    rm -rf "${PA_OUT_BASE}/video_classification_frozen/${PA_TAG}" 2>/dev/null || true

    cat > "${CFG}" <<YAML
app: vjepa
cpus_per_task: 32
folder: ${PA_OUT_BASE}
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
        aws s3 sync "${PA_OUT_BASE}/video_classification_frozen/${PA_TAG}" \
            "${S3_OUT}/${PA_TAG}/" --exclude "*.pt" --quiet 2>/dev/null && \
            log "  Uploaded to S3" || log "  WARNING: S3 upload failed"
    fi

    return $rc
}

# ============================================================
# Run all 8 predavg jobs
# ============================================================
TOTAL_START=$(date +%s)
FAILED=0

log "=== Trajectory onset prediction averaging: 2 tasks × 4 models ==="

for TASK in trajectory_tr_severity_onset trajectory_tapse_onset; do
    # EchoJEPA-G
    PROBE="/opt/dlami/nvme/nmed_traj_onset/output/video_classification_frozen/${TASK}-echojepa-g/best.pt"
    run_predavg "$TASK" "echojepa-g" "$PROBE" "$MODEL_KWARGS_G" "no" || FAILED=$((FAILED+1))

    # EchoJEPA-L-K
    PROBE="/opt/dlami/nvme/nmed_traj_onset/output/video_classification_frozen/${TASK}-echojepa-l-k/best.pt"
    run_predavg "$TASK" "echojepa-l-k" "$PROBE" "$MODEL_KWARGS_LK" "no" || FAILED=$((FAILED+1))

    # EchoPrime
    PROBE="/opt/dlami/nvme/nmed_fixup/output/video_classification_frozen/${TASK}-echoprime/best.pt"
    run_predavg "$TASK" "echoprime" "$PROBE" "$MODEL_KWARGS_EP" "yes" || FAILED=$((FAILED+1))

    # PanEcho
    PROBE="/opt/dlami/nvme/nmed_fixup/output/video_classification_frozen/${TASK}-panecho/best.pt"
    run_predavg "$TASK" "panecho" "$PROBE" "$MODEL_KWARGS_PAN" "yes" || FAILED=$((FAILED+1))
done

TOTAL_END=$(date +%s)
log "=== Predavg complete: $((8-FAILED))/8 succeeded, ${FAILED} failed ==="
log "  Total time: $(( (TOTAL_END - TOTAL_START) / 3600 ))h $(( ((TOTAL_END - TOTAL_START) % 3600) / 60 ))m"

# ============================================================
# Copy results to EFS-accessible S3 location
# ============================================================
log "=== All results at S3: ${S3_OUT}/ ==="
log "  Sync to EFS from SageMaker:"
log "    aws s3 sync ${S3_OUT}/ \$EFS/evals/vitg-384/nature_medicine/uhn/video_classification_frozen/ --exclude '*.pt'"

# ============================================================
# Chain: submit mortality_90d_noss job
# ============================================================
log "=== Submitting mortality_90d_noss job ==="
sbatch /tmp/vjepa2-ctrl/scripts/nmed_mimic_90d_noss.sbatch && log "Mortality job submitted" || log "WARNING: mortality sbatch failed"

log "=== All done ==="
