#!/bin/bash
# Overnight Node 2 — View classification + UHN probes (8×A100)
#
# Phase 1 (GPU): ViT-L view classification at d=1, d=4, and linear (~2-3h)
#   2 models (EchoJEPA-L, VideoMAE), 22K clips from S3
#
# Phase 2 (CPU): UHN linear probes, 53 tasks x 2 models (~2-4h)
#   Precomputed embeddings, no GPU needed
#
# NOTE: EchoNet-Dynamic LVEF skipped — video files not on this instance.
#
# Total: ~4-7 hours
set -euo pipefail

cd /mnt/custom-file-systems/efs/fs-0049217cdf69186d7_fsap-0fa7145b64eaa046b/vjepa2
DEVICES="cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7"

START=$(date +%s)
log() { echo "[$(date '+%H:%M:%S')] $1"; }
run() { log ">>> $1..."; PYTHONUNBUFFERED=1 python -m evals.main --fname "$2" --devices $DEVICES; log ">>> DONE"; echo ""; }

make_d4() {
    local src="$1"
    local dst="/tmp/$(basename "$src" .yaml)_d4.yaml"
    cp "$src" "$dst"
    sed -i 's/num_probe_blocks: 1/num_probe_blocks: 4/' "$dst"
    sed -i 's/tag: \(.*\)/tag: \1-d4/' "$dst"
    echo "$dst"
}

log "=== NODE 2 OVERNIGHT RUN ==="
echo ""

##############################################
# PHASE 1: ViT-L View Classification (S3)
##############################################

log "=== PHASE 1: VIEW CLASSIFICATION (ViT-L, 22K clips, S3) ==="

# Depth=1 (fresh copies with resume: false and unique tags)
cp configs/eval/vitl/view.yaml /tmp/view_echojepa_l_d1.yaml
sed -i 's/resume_checkpoint: true/resume_checkpoint: false/' /tmp/view_echojepa_l_d1.yaml
sed -i 's/tag:.*/tag: verify-echojepa-l-d1/' /tmp/view_echojepa_l_d1.yaml

cp configs/eval/vitg-384/view/videomae_view_classification.yaml /tmp/view_videomae_d1.yaml
sed -i 's/tag:.*/tag: verify-videomae-d1/' /tmp/view_videomae_d1.yaml

run "[1/6] EchoJEPA-L view d=1"         /tmp/view_echojepa_l_d1.yaml
run "[2/6] VideoMAE view d=1"            /tmp/view_videomae_d1.yaml

# Depth=4
D4_VIEW_L=$(make_d4 /tmp/view_echojepa_l_d1.yaml)
D4_VIEW_MAE=$(make_d4 /tmp/view_videomae_d1.yaml)

run "[3/6] EchoJEPA-L view d=4"         "$D4_VIEW_L"
run "[4/6] VideoMAE view d=4"            "$D4_VIEW_MAE"

# Linear
cp /tmp/view_echojepa_l_d1.yaml /tmp/view_echojepa_l_linear.yaml
grep -q "probe_type:" /tmp/view_echojepa_l_linear.yaml || \
    sed -i '/num_probe_blocks:/a\    probe_type: linear' /tmp/view_echojepa_l_linear.yaml
sed -i 's/probe_type: attentive/probe_type: linear/' /tmp/view_echojepa_l_linear.yaml
sed -i 's/tag:.*/tag: verify-echojepa-l-linear/' /tmp/view_echojepa_l_linear.yaml

cp /tmp/view_videomae_d1.yaml /tmp/view_videomae_linear.yaml
grep -q "probe_type:" /tmp/view_videomae_linear.yaml || \
    sed -i '/num_probe_blocks:/a\    probe_type: linear' /tmp/view_videomae_linear.yaml
sed -i 's/probe_type: attentive/probe_type: linear/' /tmp/view_videomae_linear.yaml
sed -i 's/tag:.*/tag: verify-videomae-linear/' /tmp/view_videomae_linear.yaml

run "[5/6] EchoJEPA-L view LINEAR"      /tmp/view_echojepa_l_linear.yaml
run "[6/6] VideoMAE view LINEAR"         /tmp/view_videomae_linear.yaml

GPU_END=$(date +%s)
log "GPU phase took $(( (GPU_END-START)/60 )) min"
echo ""

##############################################
# PHASE 2: UHN Linear Probes (CPU)
##############################################

log "=== PHASE 2: UHN LINEAR PROBES (CPU, precomputed embeddings) ==="

RESULTS_DIR=results/probes/nature_medicine/uhn
mkdir -p $RESULTS_DIR

for MODEL in echojepa_g echojepa_l; do
    log "--- $MODEL ---"
    SPLIT_DIR=experiments/nature_medicine/uhn/${MODEL}_splits

    for TASK_DIR in $SPLIT_DIR/*/; do
        TASK=$(basename "$TASK_DIR")
        TRAIN="$TASK_DIR/train.npz"
        TEST="$TASK_DIR/test.npz"
        OUT="$RESULTS_DIR/$MODEL/$TASK"

        [ -f "$OUT/metrics.json" ] && continue
        [ -f "$TRAIN" ] && [ -f "$TEST" ] || continue

        TASK_TYPE=$(python3 -c "
import numpy as np; d=np.load('$TRAIN'); labels=d['labels']; u=np.unique(labels)
print('classification' if np.all(labels==labels.astype(int)) and len(u)<20 else 'regression')
")

        log "  $MODEL/$TASK ($TASK_TYPE)"
        python -m evals.train_probe --train "$TRAIN" --val "$TEST" --task "$TASK_TYPE" --output_dir "$OUT" 2>&1 | tail -3
    done
done

END=$(date +%s)
echo ""
log "=== NODE 2 DONE: $(( (END-START)/60 )) min ==="
echo ""
echo "Results:"
echo "  View d=1/d=4/linear: evals/vitg-384/classifier/verify-{echojepa-l,videomae}-{d1,d1-d4,linear}/"
echo "  UHN probes: results/probes/nature_medicine/uhn/{echojepa_g,echojepa_l}/"
