#!/bin/bash
# Overnight run — Node 2 (CPU-only, no GPU needed)
# Train sklearn linear probes on ALL 53 UHN tasks for EchoJEPA-G and EchoJEPA-L
# Uses precomputed study-level embeddings (no video loading)
# Estimated: 2-4 hours total (sklearn Ridge/LogReg, CPU)
#
# These results populate the paper tables regardless of the verification outcome.
#
set -euo pipefail

cd /mnt/custom-file-systems/efs/fs-0049217cdf69186d7_fsap-0fa7145b64eaa046b/vjepa2

START=$(date +%s)
log() { echo "[$(date '+%H:%M:%S')] $1"; }

RESULTS_DIR=results/probes/nature_medicine/uhn
mkdir -p $RESULTS_DIR

log "=== UHN LINEAR PROBES: 53 tasks x 2 models ==="
echo ""

for MODEL in echojepa_g echojepa_l; do
    log "--- Model: $MODEL ---"
    SPLIT_DIR=experiments/nature_medicine/uhn/${MODEL}_splits

    for TASK_DIR in $SPLIT_DIR/*/; do
        TASK=$(basename "$TASK_DIR")
        TRAIN="$TASK_DIR/train.npz"
        TEST="$TASK_DIR/test.npz"
        OUT="$RESULTS_DIR/$MODEL/$TASK"

        # Skip if already done
        if [ -f "$OUT/metrics.json" ]; then
            echo "  SKIP $MODEL/$TASK (already done)"
            continue
        fi

        # Skip if files missing
        if [ ! -f "$TRAIN" ] || [ ! -f "$TEST" ]; then
            echo "  SKIP $MODEL/$TASK (missing train or test npz)"
            continue
        fi

        # Detect task type from labels
        TASK_TYPE=$(python3 -c "
import numpy as np
d = np.load('$TRAIN')
labels = d['labels']
unique = np.unique(labels)
# If labels are all integers and < 20 unique values, classification
if np.all(labels == labels.astype(int)) and len(unique) < 20:
    print('classification')
else:
    print('regression')
")

        log "  $MODEL/$TASK ($TASK_TYPE)..."
        python -m evals.train_probe \
            --train "$TRAIN" \
            --val "$TEST" \
            --task "$TASK_TYPE" \
            --output_dir "$OUT" \
            2>&1 | tail -3

    done
    echo ""
done

END=$(date +%s)
TOTAL_MIN=$(( (END - START) / 60 ))

log "=== COMPLETE: ${TOTAL_MIN} minutes ==="
echo ""
echo "Results at: $RESULTS_DIR/{echojepa_g,echojepa_l}/{task}/metrics.json"
echo ""
echo "Key results to check first (Wendy's pillars):"
echo "  MR severity:    $RESULTS_DIR/echojepa_g/mr_severity/metrics.json"
echo "  TAPSE:          $RESULTS_DIR/echojepa_g/tapse/metrics.json"
echo "  RV S':          $RESULTS_DIR/echojepa_g/rv_sp/metrics.json"
echo "  RV function:    $RESULTS_DIR/echojepa_g/rv_function/metrics.json"
echo "  AS severity:    $RESULTS_DIR/echojepa_g/as_severity/metrics.json"
echo "  HCM:            $RESULTS_DIR/echojepa_g/disease_hcm/metrics.json"
echo "  Amyloidosis:    $RESULTS_DIR/echojepa_g/disease_amyloidosis/metrics.json"
