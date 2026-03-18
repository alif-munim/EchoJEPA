#!/bin/bash
# Watcher: wait for PanEcho TR severity to finish, then archive.
# The main run_uhn_probe.sh for TR severity was started before archive_model() existed,
# so it won't archive automatically. This script fills the gap.
set -euo pipefail

REPO=/mnt/custom-file-systems/efs/fs-0049217cdf69186d7_fsap-0fa7145b64eaa046b/vjepa2
SRC="$REPO/evals/vitg-384/nature_medicine/uhn/video_classification_frozen/tr_severity-panecho"
DEST="$REPO/checkpoints/probes/tr_severity/panecho"
S3_DEST="s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts/checkpoints/probes/tr_severity/panecho"
LOG_CSV="$SRC/log_r0.csv"
TARGET_EPOCHS=15

echo "[$(date)] Watching for PanEcho TR severity completion (target: $TARGET_EPOCHS epochs)..."

while true; do
    # Check if training process is still running
    if ! pgrep -f "nm_tr_severity_panecho" > /dev/null 2>&1; then
        echo "[$(date)] PanEcho process no longer running."
        break
    fi

    # Check epoch count
    if [ -f "$LOG_CSV" ]; then
        n_epochs=$(grep -c "^[0-9]" "$LOG_CSV" 2>/dev/null || echo "0")
        if [ "$n_epochs" -ge "$TARGET_EPOCHS" ]; then
            echo "[$(date)] PanEcho completed $n_epochs epochs."
            break
        fi
        echo "[$(date)] PanEcho at epoch $n_epochs/$TARGET_EPOCHS..."
    fi

    sleep 120  # check every 2 minutes
done

# Archive
mkdir -p "$DEST"
for f in best.pt log_r0.csv latest.pt; do
    if [ -f "$SRC/$f" ]; then
        cp "$SRC/$f" "$DEST/$f"
    fi
done
echo "[$(date)] Archived locally: $DEST/"

# Push to S3
for f in best.pt log_r0.csv; do
    if [ -f "$DEST/$f" ]; then
        aws s3 cp "$DEST/$f" "$S3_DEST/$f" --quiet 2>/dev/null && \
            echo "[$(date)] Uploaded: $S3_DEST/$f" || \
            echo "[$(date)] WARNING: S3 upload failed for $f"
    fi
done

echo "[$(date)] PanEcho TR severity archiving complete."
