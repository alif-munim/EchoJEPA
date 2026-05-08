#!/usr/bin/env bash
# Submit the EchoMV-JEPA Stage-1 / 1b / 1m breadth run.
#
# Three SLURM jobs, same manifest + seed, differing only in the head/loss knob:
#   - stage1:  lambda_nce=0.0, num_modalities=1      (MVP)
#   - stage1b: lambda_nce=0.005, num_modalities=1    (tiny NCE)
#   - stage1m: lambda_nce=0.0, num_modalities=8      (per-modality projector)
#
# The three runs share the same K=8 sample manifest and cached c_clip, so
# §20.2 sampler-matched comparison is exact by construction. We start all
# three concurrently since the EchoJEPA-NeurIPS cluster has 8 H100s and each
# run uses 8 GPUs, but SLURM will queue them if the cluster is saturated.
#
# Usage:
#   scripts/echomv_jepa/launch_breadth.sh
#   scripts/echomv_jepa/launch_breadth.sh stage1 stage1m    # subset
#
# Run on the HyperPod controller (see CLAUDE.md HyperPod ops).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

STAGES=("${@:-stage1 stage1b stage1m}")

for stage in ${STAGES[@]}; do
  case "${stage}" in
    stage1|stage1b|stage1m) ;;
    *) echo "ERROR: unknown stage '${stage}'"; exit 1 ;;
  esac
done

echo "=== Submitting EchoMV-JEPA breadth run: ${STAGES[@]} ==="
for stage in ${STAGES[@]}; do
  echo "--- submit ${stage} ---"
  jid=$(sbatch --parsable --export=STAGE=${stage} scripts/echomv_jepa/pretrain.sbatch)
  echo "  jobid: ${jid}"
done

echo ""
echo "=== Submitted ==="
echo "Monitor:  squeue -u \$USER -l"
echo "Tail:     tail -f /opt/dlami/nvme/logs/echomv_pretrain-<JOBID>.out"
