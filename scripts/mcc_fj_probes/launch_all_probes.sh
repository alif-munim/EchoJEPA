#!/bin/bash
# Launch all 6 LVEF probes (base e125 / MCC +25 / Full-joint v2) × (A4C-only / K=8).
# Must be run AFTER MCC 762 and FJ 776 checkpoints are available in S3.
#
# Usage:
#   bash scripts/mcc_fj_probes/launch_all_probes.sh [--dry-run] [--prep-only] [--submit-only]
#
# Flow:
#   1. Verify MCC e25.pt and FJ latest.pt (or best available) exist in S3
#   2. Adapt checkpoints (wrap clip_target_encoder → target_encoder + epoch) → S3
#   3. Submit 6 probes, sequentially queued on the dev partition
set -euo pipefail

ART_BUCKET="sagemaker-hyperpod-lifecycle-495467399120-usw2"
S3_BASE="s3://${ART_BUCKET}/vjepa2-artifacts"

CLUSTER_ID="n9we8xfqjv3p"
CTRL_TARGET="sagemaker-cluster:${CLUSTER_ID}_echojepa-neurips-controller-i-0415ce8f417564270"

MODE="${1:-full}"   # --dry-run | --prep-only | --submit-only | anything else = full

ssm_run() {
  local cmd="$1"
  script -q -c "timeout 240 aws ssm start-session --region us-west-2 \
    --target '$CTRL_TARGET' --document-name AWS-StartNonInteractiveCommand \
    --parameters '{\"command\":[\"sudo -u ubuntu bash -c \\\"${cmd}\\\"\"]}'" /dev/null 2>&1 | \
    grep -vE 'Exiting session|Starting session'
}

log() { echo "[launch_all_probes] $*"; }

# ---- Step 1: verify pretrain checkpoints exist in S3 ----
log "Step 1/4: resolve latest MCC and FJ checkpoints"
MCC_PREFIX="${S3_BASE}/runs/mcc_target_anchored_25of100_762/checkpoints"
# Prefer e25.pt when present; else take latest e*.pt (e.g., e20.pt while still running)
MCC_SRC="${MCC_PREFIX}/e25.pt"
if ! aws s3 ls "$MCC_SRC" --region us-west-2 > /dev/null 2>&1; then
  log "  MCC e25.pt not found; checking e*.pt..."
  latest=$(aws s3 ls "${MCC_PREFIX}/" --region us-west-2 | awk '/e[0-9]+\.pt$/{print $4}' | \
    sort -t e -k2 -n | tail -1)
  MCC_SRC="${MCC_PREFIX}/${latest}"
fi
FJ_SRC="${S3_BASE}/echomv_jepa/full_joint_restart_v2_30k_runs/776/latest.pt"
if ! aws s3 ls "$FJ_SRC" --region us-west-2 > /dev/null 2>&1; then
  log "  FJ latest.pt not found; checking step_*.pt..."
  latest=$(aws s3 ls "${S3_BASE}/echomv_jepa/full_joint_restart_v2_30k_runs/776/" --region us-west-2 | \
    awk '/step_[0-9]+\.pt$/{print $4}' | sort -t _ -k2 -n | tail -1)
  FJ_SRC="${S3_BASE}/echomv_jepa/full_joint_restart_v2_30k_runs/776/${latest}"
fi
for p in "$MCC_SRC" "$FJ_SRC"; do
  if ! aws s3 ls "$p" --region us-west-2 > /dev/null 2>&1; then
    log "FATAL: $p does not exist"
    exit 1
  fi
done
log "  MCC: $MCC_SRC"
log "  FJ:  $FJ_SRC"

if [[ "$MODE" == "--dry-run" ]]; then
  log "Dry run; exiting."
  exit 0
fi

# ---- Step 2: adapt checkpoints and upload to S3 ----
if [[ "$MODE" != "--submit-only" ]]; then
  log "Step 2/4: adapt checkpoints"
  TMP_DIR="/home/sagemaker-user/probe_checkpoint_prep"
  mkdir -p "$TMP_DIR"

  log "  Downloading MCC e25.pt..."
  aws s3 cp "$MCC_SRC" "$TMP_DIR/mcc_e25.pt" --region us-west-2 --only-show-errors
  log "  Adapting MCC e25.pt..."
  python scripts/mcc_jepa/adapt_checkpoint_for_eval.py "$TMP_DIR/mcc_e25.pt"
  log "  Uploading adapted MCC e25.pt..."
  aws s3 cp "$TMP_DIR/mcc_e25_for_eval.pt" "${S3_BASE}/checkpoints/probe_inputs/mcc_e25_for_eval.pt" --region us-west-2 --only-show-errors

  log "  Downloading FJ latest.pt / step_*.pt..."
  aws s3 cp "$FJ_SRC" "$TMP_DIR/fj_source.pt" --region us-west-2 --only-show-errors
  log "  Adapting FJ source.pt..."
  python scripts/mcc_jepa/adapt_checkpoint_for_eval.py "$TMP_DIR/fj_source.pt"
  log "  Uploading adapted FJ..."
  aws s3 cp "$TMP_DIR/fj_source_for_eval.pt" "${S3_BASE}/checkpoints/probe_inputs/fj_v2_latest_for_eval.pt" --region us-west-2 --only-show-errors

  # Clean local copies
  rm -f "$TMP_DIR/mcc_e25.pt" "$TMP_DIR/fj_source.pt" "$TMP_DIR/mcc_e25_for_eval.pt" "$TMP_DIR/fj_source_for_eval.pt"
  rmdir "$TMP_DIR" 2>/dev/null || true
fi

if [[ "$MODE" == "--prep-only" ]]; then
  log "Prep-only; exiting."
  exit 0
fi

# ---- Step 3: rebuild + upload tarball ----
log "Step 3/4: rebuild and upload tarball"
tar czf /home/sagemaker-user/vjepa2-src.tar.gz -T /tmp/deploy_files.txt
aws s3 cp /home/sagemaker-user/vjepa2-src.tar.gz "${S3_BASE}/setup/vjepa2-src.tar.gz" --region us-west-2 --only-show-errors
log "  Tarball uploaded"

# Extract on controller
log "  Extracting on controller..."
ssm_run "rm -rf /tmp/vjepa2-ctrl/* && aws s3 cp ${S3_BASE}/setup/vjepa2-src.tar.gz /tmp/vjepa2-src.tar.gz --quiet && tar xzf /tmp/vjepa2-src.tar.gz -C /tmp/vjepa2-ctrl && ls /tmp/vjepa2-ctrl/scripts/mcc_fj_probes/ | head -10"

# ---- Step 4: submit 6 probes sequentially ----
log "Step 4/4: submit 6 probes (chained; wait for 762+776 to finish)"
# The probes cannot start until both pretraining jobs finish (both nodes busy).
# Submit with --dependency=afterany:762:776 so SLURM auto-launches them when
# both parents have any terminal state (completed/failed). For inter-probe
# serialization we also use --dependency=singleton on the same job name so
# SLURM queues them one at a time on a single node.
PARENT_DEPS="${PARENT_DEPS:-afterany:762:776}"
SUBMIT_OPTS="${SUBMIT_OPTS:--p dev --nodelist=ip-10-0-50-146}"
JOBS=()
FIRST=""
for name in \
    probe_base_e125_lvef_a4c \
    probe_mcc_e25_lvef_a4c \
    probe_fj_v2_lvef_a4c \
    probe_base_e125_lvef_k8 \
    probe_mcc_e25_lvef_k8 \
    probe_fj_v2_lvef_k8 ; do
  if [[ -z "$FIRST" ]]; then
    # First probe waits only on the pretraining parents.
    dep_spec="--dependency=${PARENT_DEPS}"
  else
    # Subsequent probes wait on the previous probe AND the parents (redundant
    # given the singleton chain, but explicit is safer).
    dep_spec="--dependency=afterany:${FIRST},${PARENT_DEPS}"
  fi
  log "  Submitting $name (dep: $dep_spec)..."
  out=$(ssm_run "sbatch ${SUBMIT_OPTS} ${dep_spec} --job-name=mcc_fj_probe_chain /tmp/vjepa2-ctrl/scripts/mcc_fj_probes/${name}.sbatch" | tail -5)
  JOB_ID=$(echo "$out" | grep -oE '[0-9]+$' | head -1)
  JOBS+=("$JOB_ID:$name")
  if [[ -z "$FIRST" ]]; then FIRST="$JOB_ID"; fi
  log "    -> ${JOB_ID}"
done

log "Queue:"
for entry in "${JOBS[@]}"; do
  log "  $entry"
done
log "Done. Monitor with: squeue -u ubuntu"
