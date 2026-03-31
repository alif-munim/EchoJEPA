#!/bin/bash
# RVSP noise robustness grid: 3 probes × 3 perturbation types × 3 severity levels = 27 runs
# Usage: bash scripts/rebuttal/run_rvsp_noise_grid.sh <gpu_id> <probe_type>
#   probe_type: multiview | a4c | psax
# Example: bash scripts/rebuttal/run_rvsp_noise_grid.sh 0 multiview

set -euo pipefail

GPU=${1:?Usage: $0 <gpu_id> <probe_type>}
PROBE_TYPE=${2:?Usage: $0 <gpu_id> <probe_type>}

EFS=/mnt/custom-file-systems/efs/fs-0049217cdf69186d7_fsap-0fa7145b64eaa046b/vjepa2

case $PROBE_TYPE in
    multiview)
        CONFIG=$EFS/configs/inference/vitl/icml/echojepa_l_pt50_rvsp_test.yaml
        TAG="multiview"
        ;;
    a4c)
        CONFIG=$EFS/configs/inference/vitl/icml/echojepa_l_pt50_rvsp_a4c_test.yaml
        TAG="a4c"
        ;;
    psax)
        CONFIG=$EFS/configs/inference/vitl/icml/echojepa_l_pt50_rvsp_psax_test.yaml
        TAG="psax"
        ;;
    *)
        echo "Unknown probe_type: $PROBE_TYPE (use multiview, a4c, or psax)"
        exit 1
        ;;
esac

PERTURBATION_TYPES="depth_attenuation gaussian_shadow haze"
SEVERITY_LEVELS="mild moderate severe"

OUTDIR=$EFS/predictions/rvsp_noise_grid
mkdir -p $OUTDIR

for PTYPE in $PERTURBATION_TYPES; do
    for SEV in $SEVERITY_LEVELS; do
        PRED_PATH=$OUTDIR/rvsp_${TAG}_${PTYPE}_${SEV}.csv
        echo "=== Running $TAG / $PTYPE / $SEV on cuda:$GPU ==="

        TMPDIR=/tmp LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH \
            PERTURBATION_TYPE=$PTYPE PERTURBATION_SEVERITY=$SEV \
            python -m evals.main \
                --fname $CONFIG \
                --devices cuda:$GPU \
                --val_only 2>&1 | tail -5

        # Copy predictions to grid output
        # The predictions_save_path in config is fixed, so we copy after each run
        echo "Done: $TAG / $PTYPE / $SEV"
    done
done

echo "=== All $TAG noise runs complete ==="
