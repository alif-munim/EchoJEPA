#!/bin/bash
#SBATCH --job-name=nmed-fixup
#SBATCH --partition=ml-p5-48xlarge
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --output=/tmp/nmed_fixup-%j.out
#SBATCH --error=/tmp/nmed_fixup-%j.err
#SBATCH --time=0-03:00:00
#SBATCH --nodelist=ip-10-0-50-7

# Fix-up job: re-run failed runs from jobs 75 and 83
#
# Job 75 failures (ModuleNotFoundError — used nonexistent *_multiclip modules):
#   1. trajectory_tr_severity_onset-echoprime     (TRAINING, 15ep)
#   2. trajectory_tr_severity_onset-panecho        (TRAINING, 15ep)
#   3. trajectory_tapse_onset-echoprime            (TRAINING, 15ep)
#   4. trajectory_tapse_onset-panecho              (TRAINING, 15ep)
#
# Job 83 timeout (hit 2h limit after completing 6/8):
#   5. esv_plax-predavg-echojepa-l-k              (INFERENCE)
#   6. edv_plax-predavg-echojepa-l-k              (INFERENCE)
#
# Estimated: EP/Pan onset ~40 min + L-K PLAX ~20 min = ~1h

set -eo pipefail

export PATH="/opt/vjepa2-312/bin:$PATH"
source /opt/vjepa2-312/bin/activate
set -u
cd /opt/vjepa2

export LD_LIBRARY_PATH="/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/cublas/lib:/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/cudnn/lib:/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/nccl/lib:${LD_LIBRARY_PATH:-}"
export TORCH_HOME="/opt/dlami/nvme/torch_cache"

rm -rf /dev/shm/* 2>/dev/null || true

S3_BASE="s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts"
LOCAL="/opt/dlami/nvme/nmed_fixup"
OUT_DIR="/opt/dlami/nvme/nmed_fixup/output"
S3_OUT="${S3_BASE}/nmed_a4c/results"

log() { echo "[$(date +%H:%M:%S)] $*"; }

# ============================================================
# Step 1: Download assets from S3
# ============================================================
log "=== Downloading assets ==="

mkdir -p ${LOCAL}/{encoder,csvs/trajectory_tr_severity_onset,csvs/trajectory_tapse_onset,csvs/esv_plax,csvs/edv_plax}
mkdir -p ${LOCAL}/probes/{esv,edv}/echojepa-l-k

# EchoJEPA-L-K encoder
if [ ! -f "${LOCAL}/encoder/vitl-kinetics-pt220-an55.pt" ]; then
    log "  Downloading L-K encoder..."
    aws s3 cp "${S3_BASE}/checkpoints/anneal/keep/vitl-kinetics-pt220-an55.pt" "${LOCAL}/encoder/vitl-kinetics-pt220-an55.pt"
fi

# EchoPrime encoder weights
EP_WEIGHTS="/opt/vjepa2/evals/video_classification_frozen/modelcustom/EchoPrime/model_data/weights/echo_prime_encoder.pt"
if [ ! -f "$EP_WEIGHTS" ]; then
    log "  Downloading EchoPrime encoder weights..."
    sudo mkdir -p "$(dirname $EP_WEIGHTS)"
    sudo aws s3 cp "${S3_BASE}/setup/echo_prime_encoder.pt" "$EP_WEIGHTS"
fi

# PanEcho weights
PE_CACHE="${TORCH_HOME}/hub/checkpoints"
mkdir -p "$PE_CACHE"
if [ ! -f "${PE_CACHE}/panecho.pt" ]; then
    log "  Downloading PanEcho weights..."
    aws s3 cp "${S3_BASE}/setup/panecho.pt" "${PE_CACHE}/panecho.pt"
fi

# L-K probe checkpoints for ESV/EDV
for TASK in esv edv; do
    DEST="${LOCAL}/probes/${TASK}/echojepa-l-k"
    if [ ! -f "${DEST}/best.pt" ]; then
        log "  Downloading probe: ${TASK}/echojepa-l-k"
        aws s3 cp "${S3_BASE}/checkpoints/probes/${TASK}/echojepa-l-k/best.pt" "${DEST}/best.pt"
    fi
done

# Trajectory onset CSVs
for TASK in trajectory_tr_severity_onset trajectory_tapse_onset; do
    for f in train.csv val.csv test.csv; do
        if [ ! -f "${LOCAL}/csvs/${TASK}/${f}" ]; then
            aws s3 cp "${S3_BASE}/nmed_a4c/probe_csvs/${TASK}/${f}" "${LOCAL}/csvs/${TASK}/${f}"
        fi
    done
done

# PLAX view-filtered CSVs
for TASK in esv_plax edv_plax; do
    for f in train_vf.csv test_vf.csv; do
        if [ ! -f "${LOCAL}/csvs/${TASK}/${f}" ]; then
            aws s3 cp "${S3_BASE}/nmed_a4c/probe_csvs/${TASK}/${f}" "${LOCAL}/csvs/${TASK}/${f}"
        fi
    done
    aws s3 cp "${S3_BASE}/nmed_a4c/probe_csvs/${TASK}/zscore_params.json" "${LOCAL}/csvs/${TASK}/zscore_params.json" 2>/dev/null || true
done

log "=== All assets downloaded ==="

# ============================================================
# Step 2: Config generation
# ============================================================

HP_GRID='    - {lr: 0.0005, start_lr: 0.0, warmup: 2.0, final_lr: 0.0, weight_decay: 0.001, final_weight_decay: 0.001}
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
    - {lr: 0.00001, start_lr: 0.0, warmup: 2.0, final_lr: 0.0, weight_decay: 0.1,   final_weight_decay: 0.1}'

# --- EchoPrime onset training config ---
generate_config_ep_onset() {
    local TASK="$1"
    local TAG="$2"
    local OUTFILE="/opt/dlami/nvme/nmed_fixup_cfg_${TAG}.yaml"

    cat > "$OUTFILE" <<YAML
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

  data:
    dataset_type: VideoDataset
    dataset_train: ${LOCAL}/csvs/${TASK}/train.csv
    dataset_val: ${LOCAL}/csvs/${TASK}/test.csv
    num_classes: 2
    resolution: 224
    frames_per_clip: 16
    frame_step: 2
    num_segments: 2
    num_views_per_segment: 1
    study_sampling: true
    normalization:
      - [0.0, 0.0, 0.0]
      - [1.0, 1.0, 1.0]

  optimization:
    batch_size: 2
    val_batch_size: 128
    num_epochs: 15
    use_bfloat16: true
    use_pos_embed: false
    multihead_kwargs:
${HP_GRID}

model_kwargs:
  checkpoint: null
  module_name: evals.video_classification_frozen.modelcustom.echo_prime_encoder
  pretrain_kwargs: {}
  wrapper_kwargs:
    echo_prime_root: /opt/vjepa2/evals/video_classification_frozen/modelcustom/EchoPrime
    force_fp32: true
    bin_size: 50
YAML
    echo "$OUTFILE"
}

# --- PanEcho onset training config ---
generate_config_pan_onset() {
    local TASK="$1"
    local TAG="$2"
    local OUTFILE="/opt/dlami/nvme/nmed_fixup_cfg_${TAG}.yaml"

    cat > "$OUTFILE" <<YAML
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

  data:
    dataset_type: VideoDataset
    dataset_train: ${LOCAL}/csvs/${TASK}/train.csv
    dataset_val: ${LOCAL}/csvs/${TASK}/test.csv
    num_classes: 2
    resolution: 224
    frames_per_clip: 16
    frame_step: 2
    num_segments: 2
    num_views_per_segment: 1
    study_sampling: true
    normalization:
      - [0.0, 0.0, 0.0]
      - [1.0, 1.0, 1.0]

  optimization:
    batch_size: 2
    val_batch_size: 128
    num_epochs: 15
    use_bfloat16: true
    use_pos_embed: false
    multihead_kwargs:
${HP_GRID}

model_kwargs:
  checkpoint: null
  module_name: evals.video_classification_frozen.modelcustom.panecho_encoder
  pretrain_kwargs: {}
  wrapper_kwargs: {}
YAML
    echo "$OUTFILE"
}

# --- L-K inference config ---
generate_config_lk_infer() {
    local TASK_VF="$1"
    local TASK_BASE="$2"
    local OUTFILE="/opt/dlami/nvme/nmed_fixup_cfg_${TASK_VF}_lk.yaml"

    cat > "$OUTFILE" <<YAML
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
probe_checkpoint: ${LOCAL}/probes/${TASK_BASE}/echojepa-l-k/best.pt
tag: ${TASK_VF}-predavg-echojepa-l-k

experiment:
  classifier:
    task_type: regression
    num_heads: 16
    num_probe_blocks: 1
    num_targets: 1

  data:
    dataset_type: VideoDataset
    dataset_train: ${LOCAL}/csvs/${TASK_VF}/train_vf.csv
    dataset_val: ${LOCAL}/csvs/${TASK_VF}/test_vf.csv
    num_classes: 1
    resolution: 224
    frames_per_clip: 16
    frame_step: 2
    num_segments: 2
    num_views_per_segment: 1
    study_sampling: true

  optimization:
    batch_size: 2
    val_batch_size: 128
    num_epochs: 1
    use_bfloat16: true
    use_pos_embed: false
    multihead_kwargs:
${HP_GRID}

model_kwargs:
  checkpoint: ${LOCAL}/encoder/vitl-kinetics-pt220-an55.pt
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
    use_pos_embed: false
YAML
    echo "$OUTFILE"
}

# ============================================================
# Step 3: Run function
# ============================================================
run_single() {
    local CFG="$1"
    local TAG="$2"
    local DEVICES="$3"
    local PORT="$4"

    export MASTER_PORT="$PORT"
    local START=$(date +%s)

    ps -eo pid,ppid,args | grep "multiprocessing.spawn" | grep -v grep | awk '$2 == 1 {print $1}' | xargs -r kill 2>/dev/null || true
    sleep 2

    log "[${TAG}] Starting on ${DEVICES} (port ${PORT})"

    local OUT_TAG_DIR="${OUT_DIR}/video_classification_frozen/${TAG##*:}"
    rm -rf "$OUT_TAG_DIR" 2>/dev/null || true

    local rc=0
    PYTHONUNBUFFERED=1 python -m evals.main --fname "$CFG" --devices $DEVICES || rc=$?

    local END=$(date +%s)
    if [ "$rc" -ne 0 ]; then
        log "[${TAG}] FAILED (exit ${rc}) after $(( (END - START) / 60 )) min"
    else
        log "[${TAG}] DONE in $(( (END - START) / 60 )) min"
    fi
    return $rc
}

# ============================================================
# Step 4: Run all 6 fix-up jobs
# ============================================================
ALL_DEVICES="cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7"
FAILURES=0
TOTAL=6

log "=== Fix-up: 4 onset training + 2 L-K inference ==="

# --- Part A: Trajectory onset EP/Pan (training) ---
log "--- Part A: Trajectory onset EP/Pan ---"

for TASK in trajectory_tr_severity_onset trajectory_tapse_onset; do
    # EchoPrime
    TAG="${TASK}-echoprime"
    log "=== ${TAG} ==="
    CFG=$(generate_config_ep_onset "$TASK" "$TAG")
    run_single "$CFG" "train:${TAG}" "$ALL_DEVICES" "29500" || FAILURES=$((FAILURES + 1))
    rm -rf /dev/shm/* 2>/dev/null || true

    if [ -d "${OUT_DIR}/video_classification_frozen/${TAG}" ]; then
        log "  Uploading results..."
        aws s3 sync "${OUT_DIR}/video_classification_frozen/${TAG}" \
            "${S3_OUT}/video_classification_frozen/${TAG}/" --quiet \
            --exclude "epoch_*.pt"
    fi

    # PanEcho
    TAG="${TASK}-panecho"
    log "=== ${TAG} ==="
    CFG=$(generate_config_pan_onset "$TASK" "$TAG")
    run_single "$CFG" "train:${TAG}" "$ALL_DEVICES" "29500" || FAILURES=$((FAILURES + 1))
    rm -rf /dev/shm/* 2>/dev/null || true

    if [ -d "${OUT_DIR}/video_classification_frozen/${TAG}" ]; then
        log "  Uploading results..."
        aws s3 sync "${OUT_DIR}/video_classification_frozen/${TAG}" \
            "${S3_OUT}/video_classification_frozen/${TAG}/" --quiet \
            --exclude "epoch_*.pt"
    fi
done

# --- Part B: L-K PLAX inference (ESV + EDV) ---
log "--- Part B: L-K PLAX inference ---"

CFG=$(generate_config_lk_infer "esv_plax" "esv")
run_single "$CFG" "predavg:esv_plax-predavg-echojepa-l-k" "$ALL_DEVICES" "29500" || FAILURES=$((FAILURES + 1))

CFG=$(generate_config_lk_infer "edv_plax" "edv")
run_single "$CFG" "predavg:edv_plax-predavg-echojepa-l-k" "$ALL_DEVICES" "29500" || FAILURES=$((FAILURES + 1))

log "=== Results: $((TOTAL - FAILURES))/${TOTAL} succeeded ==="

# ============================================================
# Step 5: Upload L-K PLAX results
# ============================================================
log "=== Uploading L-K PLAX results ==="
for TASK in esv_plax edv_plax; do
    DIR="${OUT_DIR}/video_classification_frozen/${TASK}-predavg-echojepa-l-k"
    if [ -d "$DIR" ]; then
        aws s3 sync "$DIR" "${S3_OUT}/uhn/${TASK}-predavg-echojepa-l-k/"
        log "  Uploaded: ${TASK}-predavg-echojepa-l-k"
    fi
done

log "=== Fix-up complete. Failures: ${FAILURES}/${TOTAL} ==="
