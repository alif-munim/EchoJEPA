#!/bin/bash
#SBATCH --job-name=nmed-plax-lvfn
#SBATCH --partition=ml-p5-48xlarge
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --output=/tmp/nmed_plax_lvfn-%j.out
#SBATCH --error=/tmp/nmed_plax_lvfn-%j.err
#SBATCH --time=0-03:00:00
#SBATCH --nodelist=ip-10-0-50-7

# Nature Medicine: PLAX-only single-view ablation
# Tests cross-view world model hypothesis on LV function (LVEF, ESV, EDV from PLAX)
# Probes trained on A4C+A2C only — PLAX excluded from training
# 3 tasks × 3 models (EchoJEPA-G, EchoPrime, PanEcho) = 9 runs
# All regression tasks (num_targets=1)

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
LOCAL="/opt/dlami/nvme/nmed_plax_lvfn"
OUT_DIR="/opt/dlami/nvme/nmed_plax_lvfn/output"
S3_OUT="${S3_BASE}/nmed_a4c/results"

log() { echo "[$(date +%H:%M:%S)] $*"; }

# ============================================================
# Step 1: Download assets from S3
# ============================================================
log "=== Downloading assets from S3 ==="

# Create directories
mkdir -p ${LOCAL}/{encoder,csvs/lvef_plax,csvs/esv_plax,csvs/edv_plax}
for TASK in lvef esv edv; do
    for MODEL in echojepa-g echoprime panecho; do
        mkdir -p "${LOCAL}/probes/${TASK}/${MODEL}"
    done
done

# EchoJEPA-G encoder
if [ ! -f "${LOCAL}/encoder/pt-280-an81.pt" ]; then
    log "  Downloading G encoder..."
    aws s3 cp "${S3_BASE}/checkpoints/anneal/keep/pt-280-an81.pt" "${LOCAL}/encoder/pt-280-an81.pt"
fi

# Probe checkpoints (3 models × 4 tasks)
for TASK in lvef esv edv; do
    for MODEL in echojepa-g echoprime panecho; do
        DEST="${LOCAL}/probes/${TASK}/${MODEL}"
        if [ ! -f "${DEST}/best.pt" ]; then
            log "  Downloading probe: ${TASK}/${MODEL}"
            aws s3 cp "${S3_BASE}/checkpoints/probes/${TASK}/${MODEL}/best.pt" "${DEST}/best.pt"
        fi
    done
done

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
    log "  Downloading PanEcho weights to torch hub cache..."
    aws s3 cp "${S3_BASE}/setup/panecho.pt" "${PE_CACHE}/panecho.pt"
fi

# View-filtered CSVs
for TASK in lvef_plax esv_plax edv_plax; do
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

# --- EchoJEPA-G configs ---
generate_config_g() {
    local TASK_VF="$1"     # e.g., lvef_plax
    local TASK_BASE="$2"   # e.g., lvef (for probe checkpoint)
    local VAL_BS="$3"

    local PROBE_CKPT="${LOCAL}/probes/${TASK_BASE}/echojepa-g/best.pt"
    local TRAIN_CSV="${LOCAL}/csvs/${TASK_VF}/train_vf.csv"
    local TEST_CSV="${LOCAL}/csvs/${TASK_VF}/test_vf.csv"
    local OUTFILE="/opt/dlami/nvme/nmed_plax_lvfn_cfg_${TASK_VF}_g.yaml"

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
tag: ${TASK_VF}-predavg-echojepa-g

experiment:
  classifier:
    task_type: regression
    num_heads: 16
    num_probe_blocks: 1
    num_targets: 1

  data:
    dataset_type: VideoDataset
    dataset_train: ${TRAIN_CSV}
    dataset_val: ${TEST_CSV}
    num_classes: 1
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
    use_pos_embed: false
YAML
    echo "$OUTFILE"
}

# --- EchoPrime configs ---
generate_config_echoprime() {
    local TASK_VF="$1"
    local TASK_BASE="$2"
    local VAL_BS="$3"

    local PROBE_CKPT="${LOCAL}/probes/${TASK_BASE}/echoprime/best.pt"
    local TRAIN_CSV="${LOCAL}/csvs/${TASK_VF}/train_vf.csv"
    local TEST_CSV="${LOCAL}/csvs/${TASK_VF}/test_vf.csv"
    local OUTFILE="/opt/dlami/nvme/nmed_plax_lvfn_cfg_${TASK_VF}_echoprime.yaml"

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
tag: ${TASK_VF}-predavg-echoprime

experiment:
  classifier:
    task_type: regression
    num_heads: 16
    num_probe_blocks: 1
    num_targets: 1

  data:
    dataset_type: VideoDataset
    dataset_train: ${TRAIN_CSV}
    dataset_val: ${TEST_CSV}
    num_classes: 1
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
    val_batch_size: ${VAL_BS}
    num_epochs: 1
    use_bfloat16: true
    use_pos_embed: false
    multihead_kwargs:
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

# --- PanEcho configs ---
generate_config_panecho() {
    local TASK_VF="$1"
    local TASK_BASE="$2"
    local VAL_BS="$3"

    local PROBE_CKPT="${LOCAL}/probes/${TASK_BASE}/panecho/best.pt"
    local TRAIN_CSV="${LOCAL}/csvs/${TASK_VF}/train_vf.csv"
    local TEST_CSV="${LOCAL}/csvs/${TASK_VF}/test_vf.csv"
    local OUTFILE="/opt/dlami/nvme/nmed_plax_lvfn_cfg_${TASK_VF}_panecho.yaml"

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
tag: ${TASK_VF}-predavg-panecho

experiment:
  classifier:
    task_type: regression
    num_heads: 16
    num_probe_blocks: 1
    num_targets: 1

  data:
    dataset_type: VideoDataset
    dataset_train: ${TRAIN_CSV}
    dataset_val: ${TEST_CSV}
    num_classes: 1
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
    val_batch_size: ${VAL_BS}
    num_epochs: 1
    use_bfloat16: true
    use_pos_embed: false
    multihead_kwargs:
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
  checkpoint: null
  module_name: evals.video_classification_frozen.modelcustom.panecho_encoder
  pretrain_kwargs: {}
  wrapper_kwargs: {}
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
# Step 4: Run inference — 12 runs (3 models × 4 tasks)
# ============================================================
ALL_DEVICES="cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7"
FAILURES=0
TOTAL=9

log "=== PLAX-only LV Function (3 models × 3 tasks) ==="

# --- LVEF from PLAX (probe trained on A4C+A2C, PLAX excluded) ---
log "--- LVEF PLAX: regression, 14978 test studies ---"

CFG=$(generate_config_g "lvef_plax" "lvef" "128")
run_single "$CFG" "predavg:lvef_plax-predavg-echojepa-g" "$ALL_DEVICES" "29500" || FAILURES=$((FAILURES + 1))

CFG=$(generate_config_echoprime "lvef_plax" "lvef" "128")
run_single "$CFG" "predavg:lvef_plax-predavg-echoprime" "$ALL_DEVICES" "29500" || FAILURES=$((FAILURES + 1))

CFG=$(generate_config_panecho "lvef_plax" "lvef" "128")
run_single "$CFG" "predavg:lvef_plax-predavg-panecho" "$ALL_DEVICES" "29500" || FAILURES=$((FAILURES + 1))

# --- ESV from PLAX (probe trained on A4C+A2C, PLAX excluded) ---
log "--- ESV PLAX: regression, 14961 test studies ---"

CFG=$(generate_config_g "esv_plax" "esv" "128")
run_single "$CFG" "predavg:esv_plax-predavg-echojepa-g" "$ALL_DEVICES" "29500" || FAILURES=$((FAILURES + 1))

CFG=$(generate_config_echoprime "esv_plax" "esv" "128")
run_single "$CFG" "predavg:esv_plax-predavg-echoprime" "$ALL_DEVICES" "29500" || FAILURES=$((FAILURES + 1))

CFG=$(generate_config_panecho "esv_plax" "esv" "128")
run_single "$CFG" "predavg:esv_plax-predavg-panecho" "$ALL_DEVICES" "29500" || FAILURES=$((FAILURES + 1))

# --- EDV from PLAX (probe trained on A4C+A2C, PLAX excluded) ---
log "--- EDV PLAX: regression, 15002 test studies ---"

CFG=$(generate_config_g "edv_plax" "edv" "128")
run_single "$CFG" "predavg:edv_plax-predavg-echojepa-g" "$ALL_DEVICES" "29500" || FAILURES=$((FAILURES + 1))

CFG=$(generate_config_echoprime "edv_plax" "edv" "128")
run_single "$CFG" "predavg:edv_plax-predavg-echoprime" "$ALL_DEVICES" "29500" || FAILURES=$((FAILURES + 1))

CFG=$(generate_config_panecho "edv_plax" "edv" "128")
run_single "$CFG" "predavg:edv_plax-predavg-panecho" "$ALL_DEVICES" "29500" || FAILURES=$((FAILURES + 1))

log "=== Results: $((TOTAL - FAILURES))/${TOTAL} succeeded ==="

# ============================================================
# Step 5: Upload results to S3
# ============================================================
log "=== Uploading results to S3 ==="
for TASK in lvef_plax esv_plax edv_plax; do
    for MODEL in echojepa-g echoprime panecho; do
        DIR="${OUT_DIR}/video_classification_frozen/${TASK}-predavg-${MODEL}"
        if [ -d "$DIR" ]; then
            aws s3 sync "$DIR" "${S3_OUT}/uhn/${TASK}-predavg-${MODEL}/"
            log "  Uploaded: ${TASK}-predavg-${MODEL}"
        fi
    done
done

log "=== All done. Failures: ${FAILURES}/${TOTAL} ==="
