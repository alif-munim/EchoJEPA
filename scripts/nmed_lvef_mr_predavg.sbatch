#!/bin/bash
#SBATCH --job-name=nmed-predavg
#SBATCH --partition=ml-p5-48xlarge
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --output=/tmp/nmed_predavg-%j.out
#SBATCH --error=/tmp/nmed_predavg-%j.err
#SBATCH --time=0-08:00:00
#SBATCH --nodelist=ip-10-0-50-241

# Nature Medicine: Prediction averaging for LVEF and MR severity
# Models: EchoJEPA-G, EchoPrime, PanEcho (3 models x 2 tasks = 6 runs)
#
# Prerequisites:
#   1. Run scripts/prep_nmed_predavg_s3.sh from SageMaker (once)
#   2. Deploy code: cd ~/EchoJEPA-repo && git pull && ~/deploy.sh
#   3. Submit: sbatch scripts/nmed_lvef_mr_predavg.sbatch
#
# Sequential execution, all 8 GPUs per run:
#   1. G-MR    (~1.5h)  — largest: 1.6M clips × 1B params
#   2. G-LVEF  (~25m)   — 400K clips × 1B params
#   3. EP-MR   (~10m)   — 1.6M clips × 34M params
#   4. EP-LVEF (~3m)    — 400K clips × 34M params
#   5. Pan-MR  (~8m)    — 1.6M clips × 43M params
#   6. Pan-LVEF(~2m)    — 400K clips × 43M params
# Total estimated time: ~2.5h (vs ~7h with 6-way parallel on 2 GPUs per G run)

set -eo pipefail

export PATH="/opt/vjepa2-312/bin:$PATH"
source /opt/vjepa2-312/bin/activate
set -u
cd /opt/vjepa2

# Fix cuBLAS 12.9 bf16 bug on H100
export LD_LIBRARY_PATH="/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/cublas/lib:/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/cudnn/lib:/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/nccl/lib:${LD_LIBRARY_PATH:-}"

# Set torch hub cache explicitly so PanEcho finds pre-downloaded weights
export TORCH_HOME="/tmp/torch_cache"

rm -rf /dev/shm/* 2>/dev/null || true

S3_BASE="s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts"
LOCAL="/tmp/nmed_predavg"
OUT_DIR="/opt/vjepa2/evals/vitg-384/nature_medicine/uhn"
S3_OUT="${S3_BASE}/nmed_predavg/results"

log() { echo "[$(date +%H:%M:%S)] $*"; }

# ============================================================
# Step 1: Download all assets from S3
# ============================================================
log "=== Downloading assets from S3 ==="
mkdir -p ${LOCAL}/{probes/lvef,probes/mr_severity,encoder,echoprime,csvs/lvef,csvs/mr_severity}

# Probe checkpoints (6 files, ~8.7GB total)
for TASK in lvef mr_severity; do
    for MODEL in echojepa-g echoprime panecho; do
        DEST="${LOCAL}/probes/${TASK}/${MODEL}"
        mkdir -p "$DEST"
        if [ ! -f "${DEST}/best.pt" ]; then
            log "  Downloading probe: ${TASK}/${MODEL}"
            aws s3 cp "${S3_BASE}/checkpoints/probes/${TASK}/${MODEL}/best.pt" "${DEST}/best.pt"
        fi
    done
done

# EchoJEPA-G encoder (~16.5GB)
if [ ! -f "${LOCAL}/encoder/pt-280-an81.pt" ]; then
    log "  Downloading G encoder..."
    aws s3 cp "${S3_BASE}/checkpoints/anneal/keep/pt-280-an81.pt" "${LOCAL}/encoder/pt-280-an81.pt"
fi

# EchoPrime encoder weights (~133MB)
EP_WEIGHT_DIR="/opt/vjepa2/evals/video_classification_frozen/modelcustom/EchoPrime/model_data/weights"
mkdir -p "$EP_WEIGHT_DIR"
if [ ! -f "${EP_WEIGHT_DIR}/echo_prime_encoder.pt" ]; then
    log "  Downloading EchoPrime encoder..."
    aws s3 cp "${S3_BASE}/checkpoints/echoprime/echo_prime_encoder.pt" "${EP_WEIGHT_DIR}/echo_prime_encoder.pt"
fi

# PanEcho pretrained weights -> torch hub cache
TORCH_CACHE="${TORCH_HOME}/hub/checkpoints"
mkdir -p "$TORCH_CACHE"
if [ ! -f "${TORCH_CACHE}/panecho.pt" ]; then
    log "  Downloading PanEcho weights..."
    aws s3 cp "${S3_BASE}/checkpoints/panecho/panecho.pt" "${TORCH_CACHE}/panecho.pt"
fi

# PanEcho content files (tasks.pkl, tasks.npy, etc.)
PANECHO_CONTENT="/opt/vjepa2/evals/video_classification_frozen/modelcustom/PanEcho/content"
mkdir -p "$PANECHO_CONTENT"
if [ ! -f "${PANECHO_CONTENT}/tasks.pkl" ]; then
    log "  Downloading PanEcho content files..."
    aws s3 sync "${S3_BASE}/nmed_predavg/panecho_content/" "${PANECHO_CONTENT}/"
fi

# Probe CSVs
for TASK in lvef mr_severity; do
    for f in train_vf.csv test_vf.csv; do
        if [ ! -f "${LOCAL}/csvs/${TASK}/${f}" ]; then
            aws s3 cp "${S3_BASE}/nmed_predavg/probe_csvs/${TASK}/${f}" "${LOCAL}/csvs/${TASK}/${f}"
        fi
    done
    # zscore_params (regression)
    aws s3 cp "${S3_BASE}/nmed_predavg/probe_csvs/${TASK}/zscore_params.json" "${LOCAL}/csvs/${TASK}/zscore_params.json" 2>/dev/null || true
done

log "=== All assets downloaded ==="

# ============================================================
# Step 2: Config generation function
# ============================================================
generate_config() {
    local TASK="$1"
    local TASK_TYPE="$2"
    local NUM_CLASSES="$3"
    local MODEL_TAG="$4"
    local MODULE_NAME="$5"
    local ENCODER_CKPT="$6"
    local PRETRAIN_KWARGS="$7"
    local WRAPPER_KWARGS="$8"
    local VAL_BS="$9"

    local PROBE_CKPT="${LOCAL}/probes/${TASK}/${MODEL_TAG}/best.pt"
    local TRAIN_CSV="${LOCAL}/csvs/${TASK}/train_vf.csv"
    local TEST_CSV="${LOCAL}/csvs/${TASK}/test_vf.csv"
    local OUTFILE="/tmp/nmed_cfg_${TASK}_${MODEL_TAG}.yaml"

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
probe_checkpoint: ${PROBE_CKPT}
tag: ${TASK}-predavg-${MODEL_TAG}

experiment:
  classifier:
    task_type: ${TASK_TYPE}
    num_heads: 16
    num_probe_blocks: 1
    num_targets: ${NUM_CLASSES}

  data:
    dataset_type: VideoDataset
    dataset_train: ${TRAIN_CSV}
    dataset_val: ${TEST_CSV}
    num_classes: ${NUM_CLASSES}
    resolution: 224
    frames_per_clip: 16
    frame_step: 2
    num_segments: 2
    num_views_per_segment: 1
    study_sampling: true

  optimization:
    batch_size: 2
    val_batch_size: ${VAL_BS}
    num_epochs: 1
    use_bfloat16: true
    use_pos_embed: false
    multihead_kwargs:
    # 12-head grid (4 LR x 3 WD) — must match training grid
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
    - {lr: 0.00001, start_lr: 0.0, warmup: 2.0, final_lr: 0.0, weight_decay: 0.1,   final_weight_decay: 0.1}

model_kwargs:
  checkpoint: ${ENCODER_CKPT}
  module_name: ${MODULE_NAME}
  pretrain_kwargs:
${PRETRAIN_KWARGS}
  wrapper_kwargs:
${WRAPPER_KWARGS}
YAML
    echo "$OUTFILE"
}

# ============================================================
# Step 3: Run a single model+task combination
# ============================================================
run_single() {
    local TASK="$1"
    local TASK_TYPE="$2"
    local NUM_CLASSES="$3"
    local MODEL="$4"
    local DEVICES="$5"
    local PORT="$6"

    export MASTER_PORT="$PORT"
    local TAG="${TASK}:${MODEL}"
    local START=$(date +%s)

    local CFG=""
    case "$MODEL" in
        echojepa-g)
            CFG=$(generate_config "$TASK" "$TASK_TYPE" "$NUM_CLASSES" "echojepa-g" \
                "evals.video_classification_frozen.modelcustom.vit_encoder_multiclip" \
                "${LOCAL}/encoder/pt-280-an81.pt" \
                "    encoder:
      checkpoint_key: target_encoder
      img_temporal_dim_size: null
      model_name: vit_giant_xformers
      patch_size: 16
      tubelet_size: 2
      uniform_power: true
      use_rope: true" \
                "    max_frames: 128
    use_pos_embed: false" \
                "128")
            ;;
        echoprime)
            CFG=$(generate_config "$TASK" "$TASK_TYPE" "$NUM_CLASSES" "echoprime" \
                "evals.video_classification_frozen.modelcustom.echo_prime_encoder" \
                "null" \
                "    {}" \
                "    echo_prime_root: /opt/vjepa2/evals/video_classification_frozen/modelcustom/EchoPrime
    force_fp32: true
    bin_size: 50" \
                "64")
            ;;
        panecho)
            CFG=$(generate_config "$TASK" "$TASK_TYPE" "$NUM_CLASSES" "panecho" \
                "evals.video_classification_frozen.modelcustom.panecho_encoder" \
                "null" \
                "    {}" \
                "    {}" \
                "64")
            ;;
    esac

    log "[${TAG}] Starting on ${DEVICES} (port ${PORT})"

    # Clear stale output dir (Bug 012)
    local OUT_TAG_DIR="${OUT_DIR}/video_classification_frozen/${TASK}-predavg-${MODEL}"
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
# Step 4: Run all 6 tasks sequentially on all 8 GPUs
# ============================================================
log "=== Starting sequential prediction averaging (8 GPUs per run) ==="

ALL_DEVICES="cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7"

# Detect MR severity num_classes from train CSV
MR_NUM_CLASSES=$(awk '{print $NF}' "${LOCAL}/csvs/mr_severity/train_vf.csv" | sort -u | wc -l)
log "MR severity: ${MR_NUM_CLASSES} classes"
log "All runs use 8 GPUs sequentially"

# Kill any orphaned workers before launching (Bug 011)
ps -eo pid,ppid,args | grep "multiprocessing.spawn" | grep -v grep | awk '$2 == 1 {print $1}' | xargs -r kill 2>/dev/null || true
sleep 2

FAILURES=0
TOTAL=6

# Run largest first (G-MR: 1.6M clips × 1B params)
run_single "mr_severity" "classification" "${MR_NUM_CLASSES}" "echojepa-g" "$ALL_DEVICES" "29500" || FAILURES=$((FAILURES + 1))

# Clean orphans between runs
ps -eo pid,ppid,args | grep "multiprocessing.spawn" | grep -v grep | awk '$2 == 1 {print $1}' | xargs -r kill 2>/dev/null || true
sleep 2

run_single "lvef"        "regression"     "1"                "echojepa-g" "$ALL_DEVICES" "29500" || FAILURES=$((FAILURES + 1))

ps -eo pid,ppid,args | grep "multiprocessing.spawn" | grep -v grep | awk '$2 == 1 {print $1}' | xargs -r kill 2>/dev/null || true
sleep 2

run_single "mr_severity" "classification" "${MR_NUM_CLASSES}" "echoprime"  "$ALL_DEVICES" "29500" || FAILURES=$((FAILURES + 1))

ps -eo pid,ppid,args | grep "multiprocessing.spawn" | grep -v grep | awk '$2 == 1 {print $1}' | xargs -r kill 2>/dev/null || true
sleep 2

run_single "lvef"        "regression"     "1"                "echoprime"  "$ALL_DEVICES" "29500" || FAILURES=$((FAILURES + 1))

ps -eo pid,ppid,args | grep "multiprocessing.spawn" | grep -v grep | awk '$2 == 1 {print $1}' | xargs -r kill 2>/dev/null || true
sleep 2

run_single "mr_severity" "classification" "${MR_NUM_CLASSES}" "panecho"    "$ALL_DEVICES" "29500" || FAILURES=$((FAILURES + 1))

ps -eo pid,ppid,args | grep "multiprocessing.spawn" | grep -v grep | awk '$2 == 1 {print $1}' | xargs -r kill 2>/dev/null || true
sleep 2

run_single "lvef"        "regression"     "1"                "panecho"    "$ALL_DEVICES" "29500" || FAILURES=$((FAILURES + 1))

log "=== Results: $((TOTAL - FAILURES))/${TOTAL} succeeded ==="

# ============================================================
# Step 5: Upload results to S3
# ============================================================
log "=== Uploading results to S3 ==="
for TASK in lvef mr_severity; do
    for MODEL in echojepa-g echoprime panecho; do
        NPZ="${OUT_DIR}/video_classification_frozen/${TASK}-predavg-${MODEL}/clip_outputs.npz"
        if [ -f "$NPZ" ]; then
            aws s3 cp "$NPZ" "${S3_OUT}/uhn/${TASK}-predavg-${MODEL}/clip_outputs.npz"
            log "  Uploaded: ${TASK}-predavg-${MODEL}"
        else
            log "  MISSING: ${TASK}-predavg-${MODEL}/clip_outputs.npz"
        fi
    done
done

log "=== Job complete ==="
log "Results on S3: ${S3_OUT}/uhn/"
log "Download to EFS with:"
log "  aws s3 sync ${S3_OUT}/uhn/ /mnt/custom-file-systems/efs/.../vjepa2/evals/vitg-384/nature_medicine/uhn/video_classification_frozen/"
