#!/bin/bash
#SBATCH --job-name=salt_shuf184
#SBATCH --partition=ml-p5-48xlarge
#SBATCH --nodes=1
#SBATCH --nodelist=ip-10-0-50-184
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --time=1-00:00:00
#SBATCH --output=/tmp/salt_shuf184-%j.out
#SBATCH --error=/tmp/salt_shuf184-%j.out

set -uo pipefail

export LD_LIBRARY_PATH="/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:/usr/local/cuda-12.3/compat:${LD_LIBRARY_PATH:-}"

REPO_DIR="/opt/vjepa2"
SCRATCH="/opt/dlami/nvme"
S3_DATA="s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts/data/EchoNet-Dynamic"
LOCAL_DATA="${SCRATCH}/echonet_dynamic"
PYTHON="/opt/vjepa2-312/bin/python3"

cd "${REPO_DIR}"

echo "[INFO] $(date) === SALT S2 v1 Frame Shuffling — Node 184 (e54 + e79) ==="

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
# Step 3: Ensure checkpoints and probes are accessible
# ---------------------------------------------------------------
echo "[INFO] e54 ckpt: $(ls ${REPO_DIR}/checkpoints/pretrain/mimic/salt_s2v1_e54.pt 2>/dev/null || echo MISSING)"
echo "[INFO] e79 ckpt: $(ls ${REPO_DIR}/checkpoints/pretrain/mimic/salt_s2v1_e79.pt 2>/dev/null || echo MISSING)"

# e54 probe is on NVMe — ensure symlink to expected path
EVAL_NVME="${SCRATCH}/salt_evals/salt_s2v1_e54_echonet_lvef_d4"
EVAL_REPO="${REPO_DIR}/evals/vitl/neurips/salt_s2v1_e54_echonet_lvef_d4"
if [ -d "${EVAL_NVME}" ] && [ ! -e "${EVAL_REPO}" ]; then
    mkdir -p "$(dirname "${EVAL_REPO}")"
    ln -sf "${EVAL_NVME}" "${EVAL_REPO}"
    echo "[INFO] Symlinked e54 eval dir to repo path"
fi
echo "[INFO] e54 probe: $(ls ${REPO_DIR}/evals/vitl/neurips/salt_s2v1_e54_echonet_lvef_d4/video_classification_frozen/neurips-salt-s2v1-e54-echonet-lvef-d4/best.pt 2>/dev/null || echo MISSING)"

# e79: ensure checkpoint accessible at legacy path for ALL_CONFIGS
if [ ! -e "${REPO_DIR}/checkpoints/salt_s2_vitl_e79.pt" ] && [ -f "${REPO_DIR}/checkpoints/pretrain/mimic/salt_s2v1_e79.pt" ]; then
    ln -sf "${REPO_DIR}/checkpoints/pretrain/mimic/salt_s2v1_e79.pt" "${REPO_DIR}/checkpoints/salt_s2_vitl_e79.pt"
fi

# e79 probe: ensure accessible at icml path for ALL_CONFIGS
E79_PROBE_NEURIPS="${REPO_DIR}/evals/vitl/neurips/salt_s2v1_echonet_lvef_d4/video_classification_frozen/neurips-salt-s2v1-echonet-lvef-d4/best.pt"
E79_PROBE_ICML_DIR="${REPO_DIR}/evals/vitl/icml/salt_s2_e79_end_lvef_224/video_classification_frozen/icml-salt-s2-e79-end-lvef-d4"
if [ -f "${E79_PROBE_NEURIPS}" ] && [ ! -e "${E79_PROBE_ICML_DIR}/best.pt" ]; then
    mkdir -p "${E79_PROBE_ICML_DIR}"
    ln -sf "${E79_PROBE_NEURIPS}" "${E79_PROBE_ICML_DIR}/best.pt"
    echo "[INFO] Symlinked e79 probe to icml path"
fi
echo "[INFO] e79 probe: $(ls ${E79_PROBE_ICML_DIR}/best.pt 2>/dev/null || echo MISSING)"

# ---------------------------------------------------------------
# Step 4: Run severity gradient + 6-condition for e54 and e79
# ---------------------------------------------------------------
echo ""
echo "[INFO] $(date) === Severity Gradient: SALT-S2-e54 (cuda:0) ==="
${PYTHON} scripts/rebuttal/frame_shuffle_severity.py \
    --test_csv "${TEST_CSV}" \
    --models SALT-S2-e54 \
    --device cuda:0 \
    --output scripts/rebuttal/samples/severity_SALT_S2_e54.csv &
PID1=$!

echo "[INFO] $(date) === Severity Gradient: SALT-S2-e79 (cuda:1) ==="
${PYTHON} scripts/rebuttal/frame_shuffle_severity.py \
    --test_csv "${TEST_CSV}" \
    --models SALT-S2-e79 \
    --device cuda:1 \
    --output scripts/rebuttal/samples/severity_SALT_S2_e79.csv &
PID2=$!

echo "[INFO] $(date) === 6-Condition: SALT-S2-e54 (cuda:2) ==="
${PYTHON} scripts/rebuttal/frame_shuffle_6cond.py \
    --test_csv "${TEST_CSV}" \
    --models SALT-S2-e54 \
    --device cuda:2 \
    --output scripts/rebuttal/samples/6cond_SALT_S2_e54.csv &
PID3=$!

echo "[INFO] $(date) === 6-Condition: SALT-S2-e79 (cuda:3) ==="
${PYTHON} scripts/rebuttal/frame_shuffle_6cond.py \
    --test_csv "${TEST_CSV}" \
    --models SALT-S2-e79 \
    --device cuda:3 \
    --output scripts/rebuttal/samples/6cond_SALT_S2_e79.csv &
PID4=$!

echo "[INFO] Waiting for all 4 parallel runs..."
wait $PID1 && echo "[INFO] $(date) Severity e54 DONE" || echo "[WARN] Severity e54 FAILED"
wait $PID2 && echo "[INFO] $(date) Severity e79 DONE" || echo "[WARN] Severity e79 FAILED"
wait $PID3 && echo "[INFO] $(date) 6-condition e54 DONE" || echo "[WARN] 6-condition e54 FAILED"
wait $PID4 && echo "[INFO] $(date) 6-condition e79 DONE" || echo "[WARN] 6-condition e79 FAILED"

echo ""
echo "[INFO] $(date) === All Node 184 jobs complete ==="
ls -la scripts/rebuttal/samples/*SALT_S2_e54* scripts/rebuttal/samples/*SALT_S2_e79* 2>/dev/null
