#!/bin/bash
#SBATCH --job-name=nmed-traj-onset
#SBATCH --partition=ml-p5-48xlarge
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --output=/tmp/nmed_traj_onset-%j.out
#SBATCH --error=/tmp/nmed_traj_onset-%j.err
#SBATCH --time=0-08:00:00
#SBATCH --nodelist=ip-10-0-50-7

# Nature Medicine: Trajectory onset probe TRAINING
# TR severity onset (baseline <= mild -> future >= moderate) + TAPSE onset (>= 1.7 -> < 1.7)
# 4 models (EchoJEPA-G, EchoJEPA-L-K, EchoPrime, PanEcho) x 2 tasks = 8 runs
# d=1 attentive probes, 12-head HP grid (4 LR x 3 WD), 15 epochs, study_sampling
#
# Estimated time per run (train+val each epoch):
#   G-TR:     ~2h   (360K train clips, 125K test, 1B params, 15ep)
#   G-TAPSE:  ~30m  (22K train clips, 53K test, 1B params, 15ep)
#   LK-TR:    ~45m  (360K clips, 300M params)
#   LK-TAPSE: ~10m  (22K clips, 300M params)
#   EP-TR:    ~15m  (360K clips, 34M params)
#   EP-TAPSE: ~5m   (22K clips, 34M params)
#   Pan-TR:   ~15m  (360K clips, 43M params)
#   Pan-TAPSE:~5m   (22K clips, 43M params)
# Total: ~4.5h

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
LOCAL="/opt/dlami/nvme/nmed_traj_onset"
OUT_DIR="/opt/dlami/nvme/nmed_traj_onset/output"
S3_OUT="${S3_BASE}/nmed_a4c/results"

log() { echo "[$(date +%H:%M:%S)] $*"; }

# ============================================================
# Step 1: Download assets from S3
# ============================================================
log "=== Downloading assets from S3 ==="

mkdir -p ${LOCAL}/{encoder,csvs/trajectory_tr_severity_onset,csvs/trajectory_tapse_onset}

# EchoJEPA-G encoder (~16.5GB)
if [ ! -f "${LOCAL}/encoder/pt-280-an81.pt" ]; then
    log "  Downloading G encoder..."
    aws s3 cp "${S3_BASE}/checkpoints/anneal/keep/pt-280-an81.pt" "${LOCAL}/encoder/pt-280-an81.pt"
fi

# EchoJEPA-L-K encoder (~5.1GB)
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

# Onset CSVs
for TASK in trajectory_tr_severity_onset trajectory_tapse_onset; do
    for f in train.csv val.csv test.csv; do
        if [ ! -f "${LOCAL}/csvs/${TASK}/${f}" ]; then
            aws s3 cp "${S3_BASE}/nmed_a4c/probe_csvs/${TASK}/${f}" "${LOCAL}/csvs/${TASK}/${f}"
        fi
    done
done

log "=== All assets downloaded ==="

# ============================================================
# Step 2: Config generation functions
# ============================================================

# 12-head HP grid: 4 LR x 3 WD (same as existing onset probes)
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

# --- EchoJEPA-G config ---
generate_config_g() {
    local TASK="$1"    # e.g., trajectory_tr_severity_onset
    local TAG="$2"     # e.g., trajectory_tr_severity_onset-echojepa-g
    local OUTFILE="/opt/dlami/nvme/nmed_traj_onset_cfg_${TAG}.yaml"

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

  optimization:
    batch_size: 2
    val_batch_size: 128
    num_epochs: 15
    use_bfloat16: true
    use_pos_embed: false
    multihead_kwargs:
${HP_GRID}

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

# --- EchoJEPA-L-K config ---
generate_config_lk() {
    local TASK="$1"
    local TAG="$2"
    local OUTFILE="/opt/dlami/nvme/nmed_traj_onset_cfg_${TAG}.yaml"

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

  optimization:
    batch_size: 2
    val_batch_size: 128
    num_epochs: 15
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

# --- EchoPrime config ---
generate_config_echoprime() {
    local TASK="$1"
    local TAG="$2"
    local OUTFILE="/opt/dlami/nvme/nmed_traj_onset_cfg_${TAG}.yaml"

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

  optimization:
    batch_size: 2
    val_batch_size: 128
    num_epochs: 15
    use_bfloat16: true
    use_pos_embed: false
    multihead_kwargs:
${HP_GRID}

model_kwargs:
  module_name: evals.video_classification_frozen.modelcustom.echoprime_encoder_multiclip
  pretrain_kwargs:
    encoder: {}
  wrapper_kwargs:
    max_frames: 128
    use_pos_embed: false
YAML
    echo "$OUTFILE"
}

# --- PanEcho config ---
generate_config_panecho() {
    local TASK="$1"
    local TAG="$2"
    local OUTFILE="/opt/dlami/nvme/nmed_traj_onset_cfg_${TAG}.yaml"

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

  optimization:
    batch_size: 2
    val_batch_size: 128
    num_epochs: 15
    use_bfloat16: true
    use_pos_embed: false
    multihead_kwargs:
${HP_GRID}

model_kwargs:
  module_name: evals.video_classification_frozen.modelcustom.panecho_encoder_multiclip
  pretrain_kwargs:
    encoder: {}
  wrapper_kwargs:
    max_frames: 128
    use_pos_embed: false
YAML
    echo "$OUTFILE"
}

# ============================================================
# Step 3: Generate configs and run all 8 training jobs
# ============================================================

TASKS=(trajectory_tr_severity_onset trajectory_tapse_onset)
RUN=0
TOTAL=8

for TASK in "${TASKS[@]}"; do
    # --- EchoJEPA-G ---
    RUN=$((RUN+1))
    TAG="${TASK}-echojepa-g"
    log "=== Run ${RUN}/${TOTAL}: ${TAG} ==="
    CFG=$(generate_config_g "$TASK" "$TAG")
    python -m evals.main --fname "$CFG" --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7
    rm -rf /dev/shm/* 2>/dev/null || true

    # Upload results
    if [ -d "${OUT_DIR}/video_classification_frozen/${TAG}" ]; then
        log "  Uploading results..."
        aws s3 sync "${OUT_DIR}/video_classification_frozen/${TAG}" \
            "${S3_OUT}/video_classification_frozen/${TAG}/" --quiet \
            --exclude "epoch_*.pt"
    fi

    # --- EchoJEPA-L-K ---
    RUN=$((RUN+1))
    TAG="${TASK}-echojepa-l-k"
    log "=== Run ${RUN}/${TOTAL}: ${TAG} ==="
    CFG=$(generate_config_lk "$TASK" "$TAG")
    python -m evals.main --fname "$CFG" --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7
    rm -rf /dev/shm/* 2>/dev/null || true

    if [ -d "${OUT_DIR}/video_classification_frozen/${TAG}" ]; then
        log "  Uploading results..."
        aws s3 sync "${OUT_DIR}/video_classification_frozen/${TAG}" \
            "${S3_OUT}/video_classification_frozen/${TAG}/" --quiet \
            --exclude "epoch_*.pt"
    fi

    # --- EchoPrime ---
    RUN=$((RUN+1))
    TAG="${TASK}-echoprime"
    log "=== Run ${RUN}/${TOTAL}: ${TAG} ==="
    CFG=$(generate_config_echoprime "$TASK" "$TAG")
    python -m evals.main --fname "$CFG" --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7
    rm -rf /dev/shm/* 2>/dev/null || true

    if [ -d "${OUT_DIR}/video_classification_frozen/${TAG}" ]; then
        log "  Uploading results..."
        aws s3 sync "${OUT_DIR}/video_classification_frozen/${TAG}" \
            "${S3_OUT}/video_classification_frozen/${TAG}/" --quiet \
            --exclude "epoch_*.pt"
    fi

    # --- PanEcho ---
    RUN=$((RUN+1))
    TAG="${TASK}-panecho"
    log "=== Run ${RUN}/${TOTAL}: ${TAG} ==="
    CFG=$(generate_config_panecho "$TASK" "$TAG")
    python -m evals.main --fname "$CFG" --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7
    rm -rf /dev/shm/* 2>/dev/null || true

    if [ -d "${OUT_DIR}/video_classification_frozen/${TAG}" ]; then
        log "  Uploading results..."
        aws s3 sync "${OUT_DIR}/video_classification_frozen/${TAG}" \
            "${S3_OUT}/video_classification_frozen/${TAG}/" --quiet \
            --exclude "epoch_*.pt"
    fi
done

log "=== All ${TOTAL} training runs complete ==="
log "Results at: ${OUT_DIR}/video_classification_frozen/"
log "S3 backup at: ${S3_OUT}/video_classification_frozen/"
