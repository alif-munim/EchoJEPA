#!/bin/bash
#SBATCH -p ml-p5-48xlarge
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 0-04:00:00
#SBATCH -J ejg_mr_cmp
#SBATCH -o /tmp/ejg_mr_cmp-%j.out
#SBATCH -e /tmp/ejg_mr_cmp-%j.err
#SBATCH --signal=B:USR1@300
#SBATCH --nodelist=ip-10-0-50-184

set -euxo pipefail

# =========================
# GLOBAL CONFIG
# =========================
export AWS_DEFAULT_REGION="us-west-2"
export AWS_REGION="us-west-2"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export MASTER_PORT=$((12000 + $RANDOM % 20000))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

export AWS_SDK_LOAD_CONFIG=1
export HOME="/home/ubuntu"
export AWS_SHARED_CREDENTIALS_FILE="${HOME}/.aws/credentials"
export AWS_CONFIG_FILE="${HOME}/.aws/config"
unset AWS_EC2_METADATA_DISABLED 2>/dev/null || true
export AWS_METADATA_SERVICE_TIMEOUT=5
export AWS_METADATA_SERVICE_NUM_ATTEMPTS=3

export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export FI_EFA_USE_DEVICE_RDMA=1
export FI_PROVIDER=efa
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# cuBLAS fix for H100 bf16
export LD_LIBRARY_PATH="/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/cublas/lib:/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/cudnn/lib:/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/nccl/lib:/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:${LD_LIBRARY_PATH:-}"

# S3 paths
ART_BUCKET="sagemaker-hyperpod-lifecycle-495467399120-usw2"
ART_PREFIX="vjepa2-artifacts"
S3_BASE="s3://${ART_BUCKET}/${ART_PREFIX}"
RUN_S3="${S3_BASE}/runs/echojepa_g_mr_compare_${SLURM_JOB_ID}"
MIMIC_PROBE_S3="${S3_BASE}/runs/echojepa_g_mitral_regurg_436"
UHN_PROBE_S3="${S3_BASE}/runs/echojepa_g_mr_severity_uhn_443"

# =========================
# HOST SETUP
# =========================
echo "[INFO] host=$(hostname) job=${SLURM_JOB_ID}"
nvidia-smi -L || true

SCRATCH="/opt/dlami/nvme"
WORKDIR="${SCRATCH}/ejg_mr_cmp_${SLURM_JOB_ID}"
mkdir -p "$WORKDIR"

# =========================
# TRAPS
# =========================
CFG_FOLDER=""
upload_logs() {
  echo "[INFO] Uploading logs..."
  for f in "$WORKDIR"/*.log "$WORKDIR"/*.csv; do
    [ -f "$f" ] && aws s3 cp "$f" "${RUN_S3}/logs/$(basename $f)" --region us-west-2 2>/dev/null || true
  done
  if [ -n "$CFG_FOLDER" ] && [ -d "$CFG_FOLDER" ]; then
    aws s3 sync "$CFG_FOLDER" "${RUN_S3}/training_folder/" \
      --include "*.pt" --include "*.csv" --include "*.log" --include "*.json" --include "*.npz" \
      --exclude "*.pth" --region us-west-2 --only-show-errors 2>/dev/null || true
  fi
}
trap 'rc=$?; upload_logs; exit $rc' EXIT

# =========================
# ENVIRONMENT
# =========================
export PATH="/opt/vjepa2-312/bin:$PATH"

pip install scikit-learn 2>&1 | tail -3

# =========================
# CODE
# =========================
REPO_DIR="/opt/vjepa2"
cd "$REPO_DIR"
pip install -e . --no-deps --no-build-isolation 2>/dev/null || true
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# =========================
# DOWNLOAD INPUTS
# =========================
echo "[INFO] Downloading MIMIC MR CSVs..."
CSV_DIR="${SCRATCH}/data/csv"
mkdir -p "$CSV_DIR" data/csv
aws s3 cp s3://echodata25/nature_medicine/mimic/probe_csvs/mitral_regurg/train.csv "${CSV_DIR}/mimic_mitral_regurg_train.csv" --region us-west-2
aws s3 cp s3://echodata25/nature_medicine/mimic/probe_csvs/mitral_regurg/test.csv "${CSV_DIR}/mimic_mitral_regurg_test.csv" --region us-west-2
ln -sf "${CSV_DIR}/mimic_mitral_regurg_train.csv" data/csv/mimic_mitral_regurg_train.csv
ln -sf "${CSV_DIR}/mimic_mitral_regurg_test.csv" data/csv/mimic_mitral_regurg_test.csv

echo "[INFO] Downloading EchoJEPA-G encoder checkpoint to NVMe..."
CKPT_DIR="${SCRATCH}/checkpoints/anneal/keep"
mkdir -p "$CKPT_DIR"
if [ ! -f "${CKPT_DIR}/pt-280-an81.pt" ]; then
  aws s3 cp "${S3_BASE}/checkpoints/anneal/keep/pt-280-an81.pt" "${CKPT_DIR}/pt-280-an81.pt" --region us-west-2
fi
mkdir -p checkpoints/anneal/keep
ln -sf "${CKPT_DIR}/pt-280-an81.pt" checkpoints/anneal/keep/pt-280-an81.pt

echo "[INFO] Downloading MIMIC MR probe (job 436) to NVMe..."
PROBE_DIR="${SCRATCH}/probes"
mkdir -p "${PROBE_DIR}/echojepa-g-mitral-regurg"
aws s3 cp "${MIMIC_PROBE_S3}/training_folder/video_classification_frozen/echojepa-g-mitral-regurg/best.pt" \
  "${PROBE_DIR}/echojepa-g-mitral-regurg/best.pt" --region us-west-2
mkdir -p checkpoints/probes/echojepa-g-mitral-regurg
ln -sf "${PROBE_DIR}/echojepa-g-mitral-regurg/best.pt" checkpoints/probes/echojepa-g-mitral-regurg/best.pt

echo "[INFO] Downloading UHN MR severity probe (job 443) to NVMe..."
mkdir -p "${PROBE_DIR}/echojepa-g-mr-severity-uhn"
aws s3 cp "${UHN_PROBE_S3}/training_folder/video_classification_frozen/echojepa-g-mr-severity-uhn/best.pt" \
  "${PROBE_DIR}/echojepa-g-mr-severity-uhn/best.pt" --region us-west-2
mkdir -p checkpoints/probes/echojepa-g-mr-severity-uhn
ln -sf "${PROBE_DIR}/echojepa-g-mr-severity-uhn/best.pt" checkpoints/probes/echojepa-g-mr-severity-uhn/best.pt

# =========================
# CREDENTIALS
# =========================
unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN 2>/dev/null || true

export CFG_FOLDER="/opt/dlami/nvme/evals/vitg-384/nature_medicine"
mkdir -p "$CFG_FOLDER"

# =========================
# GPU SETUP
# =========================
GPU_COUNT=$(nvidia-smi -L | wc -l)
DEVICES=()
for ((i=0; i<GPU_COUNT; i++)); do DEVICES+=("cuda:${i}"); done
echo "[INFO] ${GPU_COUNT} GPUs: ${DEVICES[*]}"

# =========================
# PHASE 1: MIMIC probe → MIMIC test (pred-avg)
# =========================
echo "[INFO] === PHASE 1: MIMIC probe on MIMIC test (pred-avg) ==="
MIMIC_LOG="${WORKDIR}/mimic_probe_predavg.log"

set +e
python -m evals.main \
  --fname configs/eval/vitg-384/nature_medicine/echojepa_g_mitral_regurg_predavg.yaml \
  --devices "${DEVICES[@]}" \
  2>&1 | tee "$MIMIC_LOG"
MIMIC_RC=${PIPESTATUS[0]}
set -e
echo "[INFO] MIMIC probe pred-avg exit code: ${MIMIC_RC}"

# Save MIMIC probe study predictions
MIMIC_PREDS="${CFG_FOLDER}/video_classification_frozen/echojepa-g-mitral-regurg-predavg/study_predictions.csv"
if [ -f "$MIMIC_PREDS" ]; then
  cp "$MIMIC_PREDS" "${WORKDIR}/mimic_probe_study_predictions.csv"
fi

# =========================
# PHASE 2: UHN probe → MIMIC test (pred-avg)
# =========================
echo "[INFO] === PHASE 2: UHN probe on MIMIC test (pred-avg) ==="
UHN_LOG="${WORKDIR}/uhn_probe_predavg.log"

export MASTER_PORT=$((MASTER_PORT + 1))
set +e
python -m evals.main \
  --fname configs/eval/vitg-384/nature_medicine/echojepa_g_mr_uhn_on_mimic_predavg.yaml \
  --devices "${DEVICES[@]}" \
  2>&1 | tee "$UHN_LOG"
UHN_RC=${PIPESTATUS[0]}
set -e
echo "[INFO] UHN probe pred-avg exit code: ${UHN_RC}"

# Save UHN probe study predictions
UHN_PREDS="${CFG_FOLDER}/video_classification_frozen/echojepa-g-mr-uhn-on-mimic-predavg/study_predictions.csv"
if [ -f "$UHN_PREDS" ]; then
  cp "$UHN_PREDS" "${WORKDIR}/uhn_probe_study_predictions.csv"
fi

# =========================
# PHASE 3: sklearn comparison
# =========================
echo "[INFO] === PHASE 3: sklearn AUROC comparison ==="
python -c "
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, cohen_kappa_score, accuracy_score, classification_report
import os

results = {}
for name, path in [
    ('MIMIC probe (job 436)', '${WORKDIR}/mimic_probe_study_predictions.csv'),
    ('UHN probe (job 443)',   '${WORKDIR}/uhn_probe_study_predictions.csv'),
]:
    if not os.path.exists(path):
        print(f'[SKIP] {name}: {path} not found')
        continue
    df = pd.read_csv(path)
    labels = df['label'].values
    preds = df['predicted_class'].values
    probs = df[['prob_class_0','prob_class_1','prob_class_2','prob_class_3']].values

    acc = accuracy_score(labels, preds)
    bal_acc = balanced_accuracy_score(labels, preds)
    kappa = cohen_kappa_score(labels, preds, weights='quadratic')

    # macro AUROC (OvR)
    try:
        auroc = roc_auc_score(labels, probs, multi_class='ovr', average='macro')
    except Exception as e:
        auroc = float('nan')
        print(f'  AUROC error: {e}')

    results[name] = {'acc': acc, 'bal_acc': bal_acc, 'kappa': kappa, 'auroc': auroc, 'n': len(df)}
    print(f'\n=== {name} (n={len(df)} studies) ===')
    print(f'  Accuracy:          {acc:.4f}')
    print(f'  Balanced Accuracy: {bal_acc:.4f}')
    print(f'  Quadratic Kappa:   {kappa:.4f}')
    print(f'  Macro AUROC (OvR): {auroc:.4f}')
    print()
    print(classification_report(labels, preds, target_names=['None-Trivial','Mild','Moderate','Severe']))

# Save comparison CSV
if len(results) == 2:
    rows = []
    for name, m in results.items():
        rows.append({'probe': name, **m})
    pd.DataFrame(rows).to_csv('${WORKDIR}/mr_comparison.csv', index=False)
    print('\n[INFO] Comparison saved to mr_comparison.csv')
" 2>&1 | tee "${WORKDIR}/sklearn_comparison.log"

# =========================
# FINAL UPLOAD
# =========================
echo "[INFO] DONE. MIMIC RC=${MIMIC_RC}, UHN RC=${UHN_RC}"
