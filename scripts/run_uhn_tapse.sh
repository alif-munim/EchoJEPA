#!/bin/bash
# Nature Medicine: TAPSE regression — 5 models, d=1 attentive probe, view-filtered training
# RV mechanics (Wendy's pillar 1)
# TAPSE: A4C view-filtered (18.4% of clips). ~25K train / 13K val studies, raw labels (z-scored at runtime)
# Every training clip is from an A4C view — no wasted gradient steps on uninformative views
set -euo pipefail

cd /mnt/custom-file-systems/efs/fs-0049217cdf69186d7_fsap-0fa7145b64eaa046b/vjepa2
export LD_LIBRARY_PATH=/opt/conda/lib:${LD_LIBRARY_PATH:-}

DEVICES="cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7"
TASK=tapse
CSV_DIR=experiments/nature_medicine/uhn/probe_csvs/${TASK}
OUT_DIR=evals/vitg-384/nature_medicine/uhn
SLEEP=90  # seconds between runs (TCPStore port cooldown)

START=$(date +%s)
log() { echo "[$(date '+%H:%M:%S')] $1"; }

run() {
    local tag="$1"
    local config="$2"
    log ">>> ${tag}..."
    PYTHONUNBUFFERED=1 python -m evals.main --fname "$config" --devices $DEVICES
    log ">>> DONE: ${tag}"
    echo ""
}

# -- Shared HP grid (written to temp files per model) --
# For regression: lower LR, no warmup, moderate WD
generate_config() {
    local MODEL_TAG="$1"
    local MODULE_NAME="$2"
    local CHECKPOINT="$3"
    local PRETRAIN_KWARGS="$4"
    local WRAPPER_KWARGS="$5"
    local OUTFILE="/tmp/nm_${TASK}_${MODEL_TAG}.yaml"

    cat > "$OUTFILE" <<YAML
# Nature Medicine: ${MODEL_TAG}, TAPSE regression, d=1 attentive probe
app: vjepa
cpus_per_task: 32
folder: ${OUT_DIR}
mem_per_gpu: 80G
nodes: 1
tasks_per_node: 8
num_workers: 8

eval_name: video_classification_frozen
resume_checkpoint: false
tag: ${TASK}-${MODEL_TAG}

experiment:
  classifier:
    task_type: regression
    num_heads: 16
    num_probe_blocks: 1
    num_targets: 1

  data:
    dataset_type: VideoDataset
    dataset_train: $(pwd)/${CSV_DIR}/train_vf.csv
    dataset_val: $(pwd)/${CSV_DIR}/val_vf.csv
    num_classes: 1
    resolution: 224
    frames_per_clip: 16
    frame_step: 2
    num_segments: 2
    num_views_per_segment: 1
    study_sampling: true

  optimization:
    batch_size: 1
    val_batch_size: 16
    num_epochs: 20
    use_bfloat16: true
    use_pos_embed: false
    multihead_kwargs:
    # LR=1e-3
    - {lr: 0.001, start_lr: 0.0, warmup: 2.0, final_lr: 0.0, weight_decay: 0.001, final_weight_decay: 0.001}
    - {lr: 0.001, start_lr: 0.0, warmup: 2.0, final_lr: 0.0, weight_decay: 0.01,  final_weight_decay: 0.01}
    - {lr: 0.001, start_lr: 0.0, warmup: 2.0, final_lr: 0.0, weight_decay: 0.1,   final_weight_decay: 0.1}
    - {lr: 0.001, start_lr: 0.0, warmup: 2.0, final_lr: 0.0, weight_decay: 0.4,   final_weight_decay: 0.4}
    # LR=5e-4
    - {lr: 0.0005, start_lr: 0.0, warmup: 2.0, final_lr: 0.0, weight_decay: 0.001, final_weight_decay: 0.001}
    - {lr: 0.0005, start_lr: 0.0, warmup: 2.0, final_lr: 0.0, weight_decay: 0.01,  final_weight_decay: 0.01}
    - {lr: 0.0005, start_lr: 0.0, warmup: 2.0, final_lr: 0.0, weight_decay: 0.1,   final_weight_decay: 0.1}
    - {lr: 0.0005, start_lr: 0.0, warmup: 2.0, final_lr: 0.0, weight_decay: 0.4,   final_weight_decay: 0.4}
    # LR=1e-4
    - {lr: 0.0001, start_lr: 0.0, warmup: 2.0, final_lr: 0.0, weight_decay: 0.001, final_weight_decay: 0.001}
    - {lr: 0.0001, start_lr: 0.0, warmup: 2.0, final_lr: 0.0, weight_decay: 0.01,  final_weight_decay: 0.01}
    - {lr: 0.0001, start_lr: 0.0, warmup: 2.0, final_lr: 0.0, weight_decay: 0.1,   final_weight_decay: 0.1}
    - {lr: 0.0001, start_lr: 0.0, warmup: 2.0, final_lr: 0.0, weight_decay: 0.4,   final_weight_decay: 0.4}
    # LR=5e-5
    - {lr: 0.00005, start_lr: 0.0, warmup: 2.0, final_lr: 0.0, weight_decay: 0.001, final_weight_decay: 0.001}
    - {lr: 0.00005, start_lr: 0.0, warmup: 2.0, final_lr: 0.0, weight_decay: 0.01,  final_weight_decay: 0.01}
    - {lr: 0.00005, start_lr: 0.0, warmup: 2.0, final_lr: 0.0, weight_decay: 0.1,   final_weight_decay: 0.1}
    - {lr: 0.00005, start_lr: 0.0, warmup: 2.0, final_lr: 0.0, weight_decay: 0.4,   final_weight_decay: 0.4}
    # LR=1e-5
    - {lr: 0.00001, start_lr: 0.0, warmup: 2.0, final_lr: 0.0, weight_decay: 0.001, final_weight_decay: 0.001}
    - {lr: 0.00001, start_lr: 0.0, warmup: 2.0, final_lr: 0.0, weight_decay: 0.01,  final_weight_decay: 0.01}
    - {lr: 0.00001, start_lr: 0.0, warmup: 2.0, final_lr: 0.0, weight_decay: 0.1,   final_weight_decay: 0.1}
    - {lr: 0.00001, start_lr: 0.0, warmup: 2.0, final_lr: 0.0, weight_decay: 0.4,   final_weight_decay: 0.4}

model_kwargs:
  checkpoint: ${CHECKPOINT}
  module_name: ${MODULE_NAME}
  pretrain_kwargs:
${PRETRAIN_KWARGS}
  wrapper_kwargs:
${WRAPPER_KWARGS}
YAML
    echo "$OUTFILE"
}

log "=== TAPSE REGRESSION: 5 models, d=1 attentive, study_sampling ==="

# 1. EchoJEPA-G
CFG=$(generate_config "echojepa-g" \
    "evals.video_classification_frozen.modelcustom.vit_encoder_multiclip" \
    "/mnt/custom-file-systems/efs/fs-0049217cdf69186d7_fsap-0fa7145b64eaa046b/vjepa2/checkpoints/anneal/keep/pt-280-an81.pt" \
    "    encoder:
      checkpoint_key: target_encoder
      img_temporal_dim_size: null
      model_name: vit_giant_xformers
      patch_size: 16
      tubelet_size: 2
      uniform_power: true
      use_rope: true" \
    "    max_frames: 128
    use_pos_embed: false")
run "[1/5] EchoJEPA-G" "$CFG"
sleep $SLEEP

# 2. EchoJEPA-L
CFG=$(generate_config "echojepa-l" \
    "evals.video_classification_frozen.modelcustom.vit_encoder_multiclip" \
    "/home/sagemaker-user/user-default-efs/vjepa2/checkpoints/anneal/keep/vitl-pt-210-an25.pt" \
    "    encoder:
      checkpoint_key: target_encoder
      img_temporal_dim_size: null
      model_name: vit_large
      patch_size: 16
      tubelet_size: 2
      uniform_power: true
      use_rope: true" \
    "    max_frames: 128
    use_pos_embed: false")
run "[2/5] EchoJEPA-L" "$CFG"
sleep $SLEEP

# 3. EchoMAE (VideoMAE architecture, echo-pretrained)
CFG=$(generate_config "echomae" \
    "evals.video_classification_frozen.modelcustom.videomae_encoder" \
    "/mnt/custom-file-systems/efs/fs-0049217cdf69186d7_fsap-0fa7145b64eaa046b/vjepa2/checkpoints/videomae-ep163.pth" \
    "    encoder:
      model_name: vit_large_patch16_224
      tubelet_size: 2" \
    "    {}")
run "[3/5] EchoMAE" "$CFG"
sleep $SLEEP

# 4. EchoPrime
CFG=$(generate_config "echoprime" \
    "evals.video_classification_frozen.modelcustom.echo_prime_encoder" \
    "null" \
    "    {}" \
    "    echo_prime_root: /home/sagemaker-user/user-default-efs/vjepa2/evals/video_classification_frozen/modelcustom/EchoPrime
    force_fp32: true
    bin_size: 50")
run "[4/5] EchoPrime" "$CFG"
sleep $SLEEP

# 5. PanEcho
CFG=$(generate_config "panecho" \
    "evals.video_classification_frozen.modelcustom.panecho_encoder" \
    "null" \
    "    {}" \
    "    {}")
run "[5/5] PanEcho" "$CFG"

END=$(date +%s)
log "=== ALL DONE in $(( (END - START) / 60 )) minutes ==="
