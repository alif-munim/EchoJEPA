#!/bin/bash
#SBATCH -p ml.p5e.48xlarge
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH -t 4-00:00:00
#SBATCH -J videomae_mimic_matched
#SBATCH -o /tmp/videomae_%j.out
#SBATCH -e /tmp/videomae_%j.err
#SBATCH --signal=B:USR1@300

set -euxo pipefail

# === CONFIGURATION ===
export AWS_DEFAULT_REGION="us-west-2"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MASTER_PORT=$((12000 + $RANDOM % 20000))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

# S3 Paths
ART_BUCKET="sagemaker-hyperpod-lifecycle-495467399120-usw2"
S3_BASE="s3://${ART_BUCKET}/vjepa2-artifacts"
CODE_S3="${S3_BASE}/code/videomae-code.tar"
CSV_S3="${S3_BASE}/data/csv/mimic_annotations_s3.csv"
ENV_S3="${S3_BASE}/env/vjepa2-312-pack-env.tar"
RUN_S3="${S3_BASE}/runs/videomae_${SLURM_JOB_ID}"

# === LOGGING & SYNC ===
sync_ckpts() {
  aws s3 sync "output" "${RUN_S3}/training_folder/" --exclude "*" --include "*.pth" --include "*.pt" --include "*.yaml" --include "*.json" --include "*.csv" --only-show-errors || true
}
upload_logs() {
  aws s3 cp "/tmp/videomae_${SLURM_JOB_ID}.out" "${RUN_S3}/logs/job.out" --only-show-errors || true
  aws s3 cp "/tmp/videomae_${SLURM_JOB_ID}.err" "${RUN_S3}/logs/job.err" --only-show-errors || true
}
on_exit() {
  rc=$?
  upload_logs
  sync_ckpts
  exit $rc
}
trap on_exit EXIT
trap 'sync_ckpts; upload_logs' USR1

# === SETUP ===
nvidia-smi -L || true
SCRATCH="/opt/dlami/nvme"
[ ! -d "$SCRATCH" ] && SCRATCH="/tmp"
WORKDIR="${SCRATCH}/videomae_${SLURM_JOB_ID}"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

# === ENV & ARTIFACTS ===
echo "[INFO] Setting up environment..."
aws s3 cp "$ENV_S3" env.tar
mkdir -p env
tar -xf env.tar -C env
set +u
source env/bin/activate
set -u
env/bin/conda-unpack || true
pip install "numpy<2.0" timm==0.4.12 boto3 tensorboardX decord --no-cache-dir

echo "[INFO] Downloading code..."
aws s3 cp "$CODE_S3" code.tar
tar -xf code.tar
sed -i 's/from torch._six import inf/from torch import inf/g' utils.py

mkdir -p data
aws s3 cp "$CSV_S3" data/train.csv

# === S3 DATASET LOADER (Robust Version) ===
cat << 'EOF' > s3_dataset.py
import os
import boto3
import io
import numpy as np
import torch
import decord
from torch.utils.data import DistributedSampler, DataLoader
from torchvision import transforms

class TubeMaskingGenerator:
    def __init__(self, input_size, clip_size, window_size, mask_ratio):
        self.frames, self.height, self.width = input_size
        self.clip_size = clip_size
        self.window_size = window_size
        self.mask_ratio = mask_ratio
        
        self.tubelet_t = clip_size[0] # 2
        self.tubelet_h = clip_size[1] # 16
        self.tubelet_w = clip_size[2] # 16
        
        self.num_patches_per_frame = (self.height // self.tubelet_h) * (self.width // self.tubelet_w) 
        self.num_temporal_patches = self.frames // self.tubelet_t 
        self.total_patches = self.num_patches_per_frame * self.num_temporal_patches 

    def __call__(self):
        num_mask = int(self.mask_ratio * self.total_patches)
        mask = np.hstack([
            np.zeros(self.total_patches - num_mask),
            np.ones(num_mask),
        ])
        np.random.shuffle(mask)
        return mask 

class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, data_paths, frames_per_clip=16, frame_step=4, mask_ratio=0.9):
        self.samples = []
        if isinstance(data_paths, str): data_paths = [data_paths]
        for p in data_paths:
            with open(p, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 1: self.samples.append(parts[0])
        print(f"[INFO] Loaded {len(self.samples)} samples")
        self.frames_per_clip = frames_per_clip
        self.frame_step = frame_step
        self.s3_client = None
        self.masked_position_generator = TubeMaskingGenerator(
            input_size=(frames_per_clip, 224, 224),
            clip_size=(2, 16, 16),
            window_size=(16, 16),
            mask_ratio=mask_ratio
        )
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def _get_s3_client(self):
        if self.s3_client is None: self.s3_client = boto3.client('s3')
        return self.s3_client

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        try:
            vr = self.loadvideo_decord(path)
            images = self._sample_from_vr(vr)
            mask = self.masked_position_generator()
            return images, torch.from_numpy(mask).bool()
        except Exception as e:
            dummy = torch.zeros((3, self.frames_per_clip, 224, 224))
            mask = self.masked_position_generator()
            return dummy, torch.from_numpy(mask).bool()

    def loadvideo_decord(self, sample):
        if sample.startswith("s3://"):
            client = self._get_s3_client()
            bucket, key = sample.replace("s3://", "").split("/", 1)
            obj = client.get_object(Bucket=bucket, Key=key)
            f = io.BytesIO(obj['Body'].read())
            return decord.VideoReader(f, num_threads=1)
        return decord.VideoReader(sample, num_threads=1)

    def _sample_from_vr(self, vr):
        duration = len(vr)
        required = self.frames_per_clip * self.frame_step
        if duration < required: indices = np.arange(0, required, self.frame_step) % duration
        else:
            max_start = duration - required
            start = np.random.randint(0, max_start + 1)
            indices = np.arange(start, start + required, self.frame_step)
        
        images = vr.get_batch(indices).asnumpy()
        images = torch.from_numpy(images).float().permute(3, 0, 1, 2) / 255.0
        images = torch.nn.functional.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
        C, T, H, W = images.shape
        images = images.permute(1, 0, 2, 3) 
        images = self.normalize(images)
        images = images.permute(1, 0, 2, 3) 
        return images

def make_videodataset(data_paths, batch_size, frames_per_clip, frame_step, num_workers, pin_mem, rank, world_size, log_dir):
    dataset = VideoDataset(data_paths, frames_per_clip, frame_step)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_mem, drop_last=True)
    return dataset, loader, sampler
EOF

# === PATCH RUN SCRIPT ===
python - << 'PY'
import re
with open("run_mae_pretraining.py", "r") as f: code = f.read()
code = re.sub(r'from src\.datasets\..*', '', code) 
code = re.sub(r'from datasets import .*', '', code)
header = "from s3_dataset import make_videodataset\nimport utils\n"
if "from s3_dataset" not in code: code = header + code
pattern = r"dataset_train\s*=\s*build_pretraining_dataset\(args\)"
replacement = """
    dataset_train, data_loader_train, sampler_train = make_videodataset(
        data_paths=args.data_path, 
        batch_size=args.batch_size, 
        frames_per_clip=args.num_frames, 
        frame_step=args.sampling_rate,
        num_workers=args.num_workers,
        pin_mem=args.pin_mem,
        rank=utils.get_rank(),
        world_size=utils.get_world_size(),
        log_dir=args.log_dir
    )
"""
code = re.sub(pattern, replacement, code)
code = code.replace("data_loader_train = torch.utils.data.DataLoader", "# data_loader_train = torch.utils.data.DataLoader")
with open("run_mae_pretraining.py", "w") as f: f.write(code)
PY

# === RUN TRAINING ===
OUTPUT_DIR="${WORKDIR}/output"
mkdir -p "$OUTPUT_DIR"

( while true; do sleep 600; sync_ckpts; done ) &
SYNC_PID=$!

# COMPUTE MATCHED: 165 EPOCHS (~168k Steps) to match V-JEPA's 84k steps @ 1024 batch
srun --kill-on-bad-exit=1 \
  python -u run_mae_pretraining.py \
    --data_path "${WORKDIR}/data/train.csv" \
    --mask_type tube --mask_ratio 0.9 \
    --model pretrain_videomae_large_patch16_224 \
    --decoder_depth 4 \
    --batch_size 64 \
    --num_frames 16 \
    --sampling_rate 4 \
    --opt adamw \
    --opt_betas 0.9 0.95 \
    --lr 2.5e-5 \
    --weight_decay 0.04 \
    --warmup_epochs 20 \
    --save_ckpt_freq 10 \
    --epochs 165 \
    --log_dir "${OUTPUT_DIR}" \
    --output_dir "${OUTPUT_DIR}"
RC=$?

kill $SYNC_PID 2>/dev/null || true
exit $RC