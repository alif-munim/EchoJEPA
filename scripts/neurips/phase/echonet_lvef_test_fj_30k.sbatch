#!/bin/bash
# EchoNet-Dynamic LVEF test inference — MCC-Anchored e25 (job 794 best.pt).
#
#SBATCH -p dev
#SBATCH --nodelist=ip-10-0-50-146
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -t 0-01:00:00
#SBATCH -J echonet_lvef_test_fj_30k
#SBATCH -o /opt/dlami/nvme/logs/echonet_lvef_test_fj_30k-%j.out
#SBATCH -e /opt/dlami/nvme/logs/echonet_lvef_test_fj_30k-%j.err
#SBATCH --chdir=/opt/dlami/nvme

set -euxo pipefail
export AWS_DEFAULT_REGION="us-west-2"; export AWS_REGION="us-west-2"
export OMP_NUM_THREADS=1; export MKL_NUM_THREADS=1; export PYTHONUNBUFFERED=1
export HOME="/home/ubuntu"
unset AWS_SHARED_CREDENTIALS_FILE 2>/dev/null || true
unset AWS_CONFIG_FILE 2>/dev/null || true
export PATH="/opt/vjepa2-312/bin:$PATH"
export LD_LIBRARY_PATH="/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/cublas/lib:/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
mkdir -p /opt/dlami/nvme/logs /opt/dlami/nvme/tmp
export TMPDIR="/opt/dlami/nvme/tmp"

PILOT_NAME="fj_30k"
TAG="neurips-${PILOT_NAME}-echonet-lvef-test"
FOLDER_REL="evals/vitl/neurips/${PILOT_NAME}_echonet_lvef_test"
ART_BUCKET="sagemaker-hyperpod-lifecycle-495467399120-usw2"
S3_BASE="s3://${ART_BUCKET}/vjepa2-artifacts"
SRC_TARBALL_S3="${S3_BASE}/setup/vjepa2-src-tokenrel-motion-v1.tar.gz"
CKPT_S3="${S3_BASE}/echomv_jepa/full_joint_restart_v2_30k_runs/776/latest.pt"
PROBE_BEST_S3="${S3_BASE}/runs/fj_30k_echonet_lvef_795/probe/video_classification_frozen/neurips-fj_30k-echonet-lvef/best.pt"
RUN_S3="${S3_BASE}/runs/${PILOT_NAME}_echonet_lvef_test_${SLURM_JOB_ID}"
WORKDIR="/opt/dlami/nvme/${PILOT_NAME}_echonet_lvef_test_${SLURM_JOB_ID}"
mkdir -p "$WORKDIR"

upload_artifacts() {
  [ -d "$FOLDER_REL" ] && aws s3 sync "$FOLDER_REL" "${RUN_S3}/test/" --only-show-errors 2>/dev/null || true
  PRED="${WORKDIR}/predictions/${PILOT_NAME}_echonet_lvef_test.csv"
  [ -f "$PRED" ] && aws s3 cp "$PRED" "${RUN_S3}/predictions/$(basename $PRED)" --only-show-errors || true
  for f in /opt/dlami/nvme/logs/echonet_lvef_test_fj_30k-${SLURM_JOB_ID}.{out,err}; do
    [ -f "$f" ] && aws s3 cp "$f" "${RUN_S3}/logs/$(basename "$f")" --only-show-errors || true
  done
}
trap 'upload_artifacts' EXIT

nvidia-smi -L || true
REPO_DIR="${WORKDIR}/code"
rm -rf "$REPO_DIR"; mkdir -p "$REPO_DIR"
aws s3 cp "${SRC_TARBALL_S3}" "${TMPDIR}/src.tar.gz" --only-show-errors
tar xzf "${TMPDIR}/src.tar.gz" -C "$REPO_DIR"
cd "$REPO_DIR"
pip install -e . --no-deps --no-build-isolation 2>/dev/null || true
export PYTHONPATH="$PWD:/opt/dlami/nvme/pyarrow_site:${PYTHONPATH:-}"
if ! python -c "import pyarrow" 2>/dev/null; then
  pip install --target=/opt/dlami/nvme/pyarrow_site --no-cache-dir pyarrow 2>&1 | tail -3
fi
pip install boto3 --no-cache-dir --quiet 2>&1 | tail -3

DATA_DIR="${WORKDIR}/data/csv"; mkdir -p "$DATA_DIR"
aws s3 cp "${S3_BASE}/data/csv/echonet_dynamic_test_s3_raw.csv" "$DATA_DIR/" --quiet

CKPT_DIR="${WORKDIR}/checkpoints"; mkdir -p "$CKPT_DIR"
aws s3 cp "$CKPT_S3" "${CKPT_DIR}/fj_raw.pt" --quiet
cat > /tmp/fj_adapt.py <<PYEOF
import torch
from pathlib import Path
src = Path("${CKPT_DIR}/fj_raw.pt")
dst = Path("${CKPT_DIR}/pretrain.pt")
sd = torch.load(src, map_location="cpu", weights_only=False)
top = set(sd.keys())
print("[adapt] source keys:", sorted(top))
if "clip_target_encoder" in top:
    adapted = {"target_encoder": sd["clip_target_encoder"], "encoder": sd.get("clip_encoder", sd["clip_target_encoder"]), "epoch": int(sd.get("step", 0)), "step": int(sd.get("step", 0))}
elif "target_encoder" in top and "epoch" in top:
    adapted = {"encoder": sd.get("encoder", sd["target_encoder"]), "target_encoder": sd["target_encoder"], "epoch": int(sd.get("epoch", 0))}
else:
    raise ValueError(f"don't know how to adapt keys: {sorted(top)}")
torch.save(adapted, dst)
print(f"[adapt] wrote {dst}")
PYEOF
python3 /tmp/fj_adapt.py
rm -f /tmp/fj_adapt.py "${CKPT_DIR}/fj_raw.pt"
aws s3 cp "$PROBE_BEST_S3" "${CKPT_DIR}/best.pt" --quiet

PRED_CSV="${WORKDIR}/predictions/${PILOT_NAME}_echonet_lvef_test.csv"
mkdir -p "$(dirname $PRED_CSV)" "$FOLDER_REL"
TEST_CFG="configs/inference/vitl/neurips/${PILOT_NAME}_echonet_lvef_test.yaml"
mkdir -p "$(dirname $TEST_CFG)"

cat > "$TEST_CFG" <<YAMLEOF
app: vjepa
cpus_per_task: 16
folder: ${FOLDER_REL}
mem_per_gpu: 80G
nodes: 1
tasks_per_node: 1
num_workers: 8
eval_name: video_classification_frozen
resume_checkpoint: true
tag: ${TAG}
val_only: true
predictions_save_path: ${PRED_CSV}
probe_checkpoint: ${CKPT_DIR}/best.pt
experiment:
  classifier:
    task_type: regression
    num_heads: 16
    num_probe_blocks: 4
    num_targets: 1
  data:
    dataset_type: VideoDataset
    dataset_train: ${DATA_DIR}/echonet_dynamic_test_s3_raw.csv
    dataset_val: ${DATA_DIR}/echonet_dynamic_test_s3_raw.csv
    target_mean: 55.7776
    target_std: 12.4072
    resolution: 224
    frames_per_clip: 16
    frame_step: 2
    num_segments: 2
    num_views_per_segment: 1
  optimization:
    batch_size: 1
    num_epochs: 1
    use_bfloat16: true
    use_pos_embed: false
    multihead_kwargs:
    - {final_lr: 0.0, final_weight_decay: 0.01, lr: 0.0, start_lr: 0.0, warmup: 0.0, weight_decay: 0.01}
model_kwargs:
  checkpoint: ${CKPT_DIR}/pretrain.pt
  module_name: evals.video_classification_frozen.modelcustom.vit_encoder_multiclip
  pretrain_kwargs:
    encoder:
      checkpoint_key: target_encoder
      model_name: vit_large
      patch_size: 16
      tubelet_size: 2
      uniform_power: true
      use_rope: true
  wrapper_kwargs:
    max_frames: 128
    use_pos_embed: false
YAMLEOF

python3 -m evals.main --fname "$TEST_CFG" --devices cuda:0 --val_only 2>&1 | tee "${WORKDIR}/test.log"
aws s3 sync "${FOLDER_REL}/" "${RUN_S3}/test/" --only-show-errors || true
[ -f "$PRED_CSV" ] && aws s3 cp "$PRED_CSV" "${RUN_S3}/predictions/$(basename $PRED_CSV)" --only-show-errors || true
echo "[INFO] ${PILOT_NAME} EchoNet-Dynamic test inference complete"
