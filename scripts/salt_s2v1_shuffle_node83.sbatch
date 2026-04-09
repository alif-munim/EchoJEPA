#!/bin/bash
#SBATCH --job-name=salt_shuf83
#SBATCH --partition=ml-p5-48xlarge
#SBATCH --nodes=1
#SBATCH --nodelist=ip-10-0-50-83
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --time=1-00:00:00
#SBATCH --output=/tmp/salt_shuf83-%j.out
#SBATCH --error=/tmp/salt_shuf83-%j.out

set -uo pipefail

export LD_LIBRARY_PATH="/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:/usr/local/cuda-12.3/compat:${LD_LIBRARY_PATH:-}"

REPO_DIR="/opt/vjepa2"
SCRATCH="/opt/dlami/nvme"
S3_DATA="s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts/data/EchoNet-Dynamic"
LOCAL_DATA="${SCRATCH}/echonet_dynamic"
PYTHON="/opt/vjepa2-312/bin/python3"

cd "${REPO_DIR}"

echo "[INFO] $(date) === SALT S2 v1 Frame Shuffling — Node 83 (e4 + e29) ==="

# ---------------------------------------------------------------
# Step 1: Download EchoNet-Dynamic test videos to NVMe
# ---------------------------------------------------------------
N_EXISTING=$(find ${LOCAL_DATA}/Videos/ -name '*.avi' 2>/dev/null | wc -l)
if [ "${N_EXISTING}" -lt 1200 ]; then
    echo "[INFO] Downloading EchoNet-Dynamic test data..."
    mkdir -p "${LOCAL_DATA}/Videos"
    aws s3 cp "${S3_DATA}/FileList.csv" "${LOCAL_DATA}/FileList.csv"

    # Strip \r, filter TEST split, download videos
    sed 's/\r$//' "${LOCAL_DATA}/FileList.csv" | awk -F',' 'NR>1 && $NF=="TEST" {print $1}' | while read fname; do
        if [ ! -f "${LOCAL_DATA}/Videos/${fname}.avi" ]; then
            aws s3 cp "${S3_DATA}/Videos/${fname}.avi" "${LOCAL_DATA}/Videos/${fname}.avi" --quiet
        fi
    done
    echo "[INFO] Downloaded $(find ${LOCAL_DATA}/Videos/ -name '*.avi' | wc -l) test videos"
else
    echo "[INFO] EchoNet data already present: ${N_EXISTING} videos"
fi

# ---------------------------------------------------------------
# Step 2: Create local test CSV (path label format)
# ---------------------------------------------------------------
TEST_CSV="${LOCAL_DATA}/test_local.csv"
echo "[INFO] Creating local test CSV..."
sed 's/\r$//' "${LOCAL_DATA}/FileList.csv" | \
    awk -F',' -v dir="${LOCAL_DATA}" 'NR>1 && $NF=="TEST" {print dir "/Videos/" $1 ".avi " $2}' \
    > "${TEST_CSV}"
echo "[INFO] Test CSV: $(wc -l < ${TEST_CSV}) entries"

# ---------------------------------------------------------------
# Step 3: Verify checkpoints and probes
# ---------------------------------------------------------------
echo "[INFO] e4 ckpt: $(ls ${REPO_DIR}/checkpoints/pretrain/mimic/salt_s2v1_e4.pt 2>/dev/null || echo MISSING)"
echo "[INFO] e29 ckpt: $(ls ${REPO_DIR}/checkpoints/pretrain/mimic/salt_s2v1_e29.pt 2>/dev/null || echo MISSING)"
echo "[INFO] e4 probe: $(ls ${REPO_DIR}/evals/vitl/neurips/salt_s2v1_e4_echonet_lvef_d4/video_classification_frozen/neurips-salt-s2v1-e4-echonet-lvef-d4/best.pt 2>/dev/null || echo MISSING)"
echo "[INFO] e29 probe: $(ls ${REPO_DIR}/evals/vitl/neurips/salt_s2v1_e29_echonet_lvef_d4/video_classification_frozen/neurips-salt-s2v1-e29-echonet-lvef-d4/best.pt 2>/dev/null || echo MISSING)"

# ---------------------------------------------------------------
# Step 4: Run severity gradient + 6-condition for e4 and e29
# ---------------------------------------------------------------
echo ""
echo "[INFO] $(date) === Severity Gradient: SALT-S2-e4 (cuda:0) ==="
${PYTHON} scripts/rebuttal/frame_shuffle_severity.py \
    --test_csv "${TEST_CSV}" \
    --models SALT-S2-e4 \
    --device cuda:0 \
    --output scripts/rebuttal/samples/severity_SALT_S2_e4.csv &
PID1=$!

echo "[INFO] $(date) === Severity Gradient: SALT-S2-e29 (cuda:1) ==="
${PYTHON} scripts/rebuttal/frame_shuffle_severity.py \
    --test_csv "${TEST_CSV}" \
    --models SALT-S2-e29 \
    --device cuda:1 \
    --output scripts/rebuttal/samples/severity_SALT_S2_e29.csv &
PID2=$!

echo "[INFO] $(date) === 6-Condition: SALT-S2-e4 (cuda:2) ==="
${PYTHON} scripts/rebuttal/frame_shuffle_6cond.py \
    --test_csv "${TEST_CSV}" \
    --models SALT-S2-e4 \
    --device cuda:2 \
    --output scripts/rebuttal/samples/6cond_SALT_S2_e4.csv &
PID3=$!

echo "[INFO] $(date) === 6-Condition: SALT-S2-e29 (cuda:3) ==="
${PYTHON} scripts/rebuttal/frame_shuffle_6cond.py \
    --test_csv "${TEST_CSV}" \
    --models SALT-S2-e29 \
    --device cuda:3 \
    --output scripts/rebuttal/samples/6cond_SALT_S2_e29.csv &
PID4=$!

echo "[INFO] Waiting for all 4 parallel runs..."
wait $PID1 && echo "[INFO] $(date) Severity e4 DONE" || echo "[WARN] Severity e4 FAILED"
wait $PID2 && echo "[INFO] $(date) Severity e29 DONE" || echo "[WARN] Severity e29 FAILED"
wait $PID3 && echo "[INFO] $(date) 6-condition e4 DONE" || echo "[WARN] 6-condition e4 FAILED"
wait $PID4 && echo "[INFO] $(date) 6-condition e29 DONE" || echo "[WARN] 6-condition e29 FAILED"

echo ""
echo "[INFO] $(date) === All Node 83 jobs complete ==="
ls -la scripts/rebuttal/samples/*SALT_S2_e4* scripts/rebuttal/samples/*SALT_S2_e29* 2>/dev/null
