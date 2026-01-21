#!/bin/bash
#SBATCH -p ml.p5e.48xlarge
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH -t 4-00:00:00
#SBATCH -J videomae_ram_load
#SBATCH -o /tmp/videomae_%j.out
#SBATCH -e /tmp/videomae_%j.err
#SBATCH --signal=B:USR1@300

set -euxo pipefail

# =========================
# GLOBAL CONFIG
# =========================
export AWS_DEFAULT_REGION="us-west-2"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export MASTER_PORT=$((12000 + $RANDOM % 20000))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

# Protocol
TARGET_FPS=8
NUM_FRAMES=16
BATCH_PER_GPU=48  
GLOBAL_BS=384     

# S3 Config
ART_BUCKET="sagemaker-hyperpod-lifecycle-495467399120-usw2"
S3_BASE="s3://${ART_BUCKET}/vjepa2-artifacts"
CODE_S3="${S3_BASE}/code/videomae-code.tar"
CSV_S3="${S3_BASE}/data/csv/mimic_annotations_s3.csv"
ENV_S3="${S3_BASE}/env/vjepa2-312-pack-env.tar"
VITL_RAW_S3="${S3_BASE}/checkpoints/vitl_raw.pth"
RUN_S3="${S3_BASE}/runs/videomae_${SLURM_JOB_ID}"

# =========================
# LOGGING
# =========================
sync_ckpts() {
  aws s3 sync "output" "${RUN_S3}/training_folder/" \
    --exclude "*" \
    --include "*.pth" --include "*.pt" --include "*.yaml" --include "*.json" --include "*.csv" \
    --only-show-errors || true
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

# =========================
# SETUP
# =========================
nvidia-smi -L || true
SCRATCH="/opt/dlami/nvme"
[ ! -d "$SCRATCH" ] && SCRATCH="/tmp"
export WORKDIR="${SCRATCH}/videomae_${SLURM_JOB_ID}"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

# Env
aws s3 cp "$ENV_S3" env.tar
mkdir -p env && tar -xf env.tar -C env
set +u && source env/bin/activate && set -u
env/bin/conda-unpack || true
pip install "numpy<2.0" timm==0.4.12 boto3 tensorboardX decord --no-cache-dir

# Code
echo "[INFO] Downloading code..."
aws s3 cp "$CODE_S3" code.tar
tar -xf code.tar

# =========================================================
# FIX: IN-MEMORY S3 DATASET LOADER (NO DISK I/O)
# =========================================================
cat << 'EOF' > s3_dataset.py
import boto3
import numpy as np
import torch
import decord
import os
import io
import time
from torch.utils.data import DistributedSampler, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F

class TubeMaskingGenerator:
    def __init__(self, input_size, clip_size, mask_ratio):
        self.frames, self.height, self.width = input_size
        self.tubelet_t, self.tubelet_h, self.tubelet_w = clip_size
        self.mask_ratio = mask_ratio
        self.num_patches_per_frame = (self.height // self.tubelet_h) * (self.width // self.tubelet_w)
        self.num_temporal_patches = self.frames // self.tubelet_t
        self.total_patches = self.num_patches_per_frame * self.num_temporal_patches

    def __call__(self):
        num_mask = int(self.mask_ratio * self.total_patches)
        mask = np.hstack([np.zeros(self.total_patches - num_mask), np.ones(num_mask)])
        np.random.shuffle(mask)
        return mask

def _parse_first_field(line: str) -> str:
    line = line.strip()
    if not line or line.startswith("#"): return ""
    if "," in line:
        return line.split(",")[0].strip()
    else:
        return line.split()[0].strip()

class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, data_paths, frames_per_clip=16, target_fps=8, crop_size=224, mask_ratio=0.9, rrc_scale=(0.5, 1.0), rrc_ratio=(0.9, 1.1)):
        self.samples = []
        if isinstance(data_paths, str): data_paths = [data_paths]
        for p in data_paths:
            with open(p, "r") as f:
                for line in f:
                    item = _parse_first_field(line)
                    if item: self.samples.append(item)
        print(f"[INFO] Loaded {len(self.samples)} samples from {data_paths}")
        self.frames_per_clip = int(frames_per_clip)
        self.target_fps = float(target_fps)
        self.crop_size = int(crop_size)
        self.rrc_scale = rrc_scale
        self.rrc_ratio = rrc_ratio
        self.s3_client = None
        self.masked_position_generator = TubeMaskingGenerator(
            input_size=(self.frames_per_clip, self.crop_size, self.crop_size),
            clip_size=(2, 16, 16),
            mask_ratio=mask_ratio
        )
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def _get_s3_client(self):
        if self.s3_client is None: self.s3_client = boto3.client("s3")
        return self.s3_client

    def __len__(self):
        return len(self.samples)

    def loadvideo_decord(self, sample):
        """Loads video directly from S3 to RAM (BytesIO) to avoid disk corruption/latency."""
        if sample.startswith("s3://"):
            client = self._get_s3_client()
            bucket, key = sample.replace("s3://", "").split("/", 1)
            
            for attempt in range(3):
                try:
                    # Download to RAM
                    obj = client.get_object(Bucket=bucket, Key=key)
                    video_bytes = obj['Body'].read()
                    
                    if len(video_bytes) == 0:
                        raise Exception("Empty file from S3")

                    # Wrap in BytesIO for decord
                    file_obj = io.BytesIO(video_bytes)
                    vr = decord.VideoReader(file_obj, num_threads=1)
                    return vr
                    
                except Exception as e:
                    # Exponential backoff
                    time.sleep(0.1 * (2 ** attempt))
                    
        return decord.VideoReader(sample, num_threads=1)

    def _sample_indices_by_fps(self, vr):
        T = self.frames_per_clip
        target_fps = self.target_fps
        try:
            duration = len(vr)
        except:
            return np.zeros((T,), dtype=np.int64)

        if duration <= 0: return np.zeros((T,), dtype=np.int64)
        
        try: src_fps = float(vr.get_avg_fps())
        except: src_fps = 0.0
        if not np.isfinite(src_fps) or src_fps <= 0: src_fps = 30.0

        clip_len_sec = T / target_fps
        vid_len_sec = duration / src_fps

        if vid_len_sec <= clip_len_sec:
            times = (np.arange(T, dtype=np.float32) / target_fps)
            idx = np.round(times * src_fps).astype(np.int64) % duration
            return idx

        max_start = vid_len_sec - clip_len_sec
        start_sec = np.random.uniform(0.0, max_start)
        times = start_sec + (np.arange(T, dtype=np.float32) / target_fps)
        idx = np.round(times * src_fps).astype(np.int64)
        idx = np.clip(idx, 0, duration - 1)
        return idx

    def __getitem__(self, idx):
        path = self.samples[idx]
        mask = self.masked_position_generator()
        try:
            vr = self.loadvideo_decord(path)
            indices = self._sample_indices_by_fps(vr)
            frames = vr.get_batch(indices).asnumpy()
            frames = torch.from_numpy(frames).float().permute(0, 3, 1, 2) / 255.0
            
            i, j, h, w = transforms.RandomResizedCrop.get_params(frames[0], scale=self.rrc_scale, ratio=self.rrc_ratio)
            frames = F.resized_crop(frames, i, j, h, w, size=(self.crop_size, self.crop_size))
            frames = self.normalize(frames)
            frames = frames.permute(1, 0, 2, 3).contiguous()
            return frames, torch.from_numpy(mask).bool()
        except Exception as e:
            # Short error log to avoid spamming
            print(f"[ERR] Load failed: {path} | {e}", flush=True) 
            dummy = torch.zeros((3, self.frames_per_clip, self.crop_size, self.crop_size))
            return dummy, torch.from_numpy(mask).bool()

def make_videodataset(data_paths, batch_size, frames_per_clip, target_fps, crop_size, num_workers, pin_mem, rank, world_size, log_dir=None):
    dataset = VideoDataset(data_paths, frames_per_clip=frames_per_clip, target_fps=target_fps, crop_size=crop_size)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_mem, drop_last=True, persistent_workers=(num_workers > 0))
    return dataset, loader, sampler
EOF
# =========================================================


# --- Safety Patch for Legacy PyTorch ---
if [ -f "utils.py" ]; then
    sed -i 's/from torch._six import inf/from torch import inf/g' utils.py
else
    find . -name utils.py -exec sed -i 's/from torch._six import inf/from torch import inf/g' {} +
fi

# Data
mkdir -p data
aws s3 cp "$CSV_S3" data/train.csv

# Checkpoint Init
echo "[INFO] Preparing Checkpoint..."
aws s3 cp "$VITL_RAW_S3" vitl_raw.pth

python - <<'PY'
import torch
from timm.models import create_model
import modeling_pretrain 

RAW = "vitl_raw.pth"
OUT = "pretrained_videomae_init.pth"
DECODER_DEPTH = 12 

ckpt = torch.load(RAW, map_location="cpu")
src = ckpt.get("model", ckpt.get("state_dict", ckpt))

# Clean
tmp = {k.replace("module.", ""): v for k, v in src.items() 
       if not k.startswith(("module.head.", "module.fc_norm.", "head.", "fc_norm."))}
src = tmp

model = create_model("pretrain_videomae_large_patch16_224", pretrained=False, decoder_depth=DECODER_DEPTH)
tgt_sd = model.state_dict()

def pick(*names):
    for n in names:
        if n in src: return src[n]
    return None

new_sd = {}
for k, v in tgt_sd.items():
    if k.startswith(("decoder.", "mask_token", "decoder_")):
        new_sd[k] = v
        continue

    sv = pick(k, k.replace("encoder.", ""), "encoder."+k)

    if sv is None and k.endswith("patch_embed.proj.weight"):
        sv = pick("patch_embed.proj.weight", "encoder.patch_embed.proj.weight")
    if sv is not None and k.endswith("patch_embed.proj.weight") and sv.ndim == 4 and v.ndim == 5:
        t = v.shape[2]
        sv = sv.unsqueeze(2).repeat(1, 1, t, 1, 1) / float(t)
        new_sd[k] = sv
        continue

    if sv is None and k.endswith("pos_embed"):
        sv = pick("pos_embed", "encoder.pos_embed")
    if sv is not None and k.endswith("pos_embed") and sv.ndim == 3 and v.ndim == 3:
        if sv.shape[1] == 197: sv_wo_cls = sv[:, 1:, :]
        else: sv_wo_cls = sv
        N = v.shape[1]
        if sv_wo_cls.shape[1] == 196 and N % 196 == 0:
            rep = N // 196
            new_sd[k] = sv_wo_cls.repeat(1, rep, 1)
            continue

    if sv is not None and sv.shape == v.shape:
        new_sd[k] = sv
    else:
        new_sd[k] = v

torch.save({"model": new_sd}, OUT)
print(f"[INFO] Wrote VideoMAE init ckpt: {OUT}")
PY

# Calculate Epochs
NUM_SAMPLES=525328
TARGET_TOTAL_CLIPS=73728000
EPOCHS=$(( (TARGET_TOTAL_CLIPS + NUM_SAMPLES - 1) / NUM_SAMPLES ))
WARMUP_EPOCHS=9

# =========================
# LAUNCH
# =========================
OUTPUT_DIR="${WORKDIR}/output"
mkdir -p "$OUTPUT_DIR"

srun --kill-on-bad-exit=1 \
  python -u run_mae_pretraining.py \
    --data_path "${WORKDIR}/data/train.csv" \
    --mask_type tube --mask_ratio 0.9 \
    --model pretrain_videomae_large_patch16_224 \
    --finetune "pretrained_videomae_init.pth" \
    --decoder_depth 12 \
    --batch_size "${BATCH_PER_GPU}" \
    --num_frames "${NUM_FRAMES}" \
    --sampling_rate "${TARGET_FPS}" \
    --opt adamw \
    --opt_betas 0.9 0.95 \
    --lr 1.875e-5 \
    --weight_decay 0.04 \
    --warmup_epochs "${WARMUP_EPOCHS}" \
    --save_ckpt_freq 10 \
    --epochs "${EPOCHS}" \
    --log_dir "${OUTPUT_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_workers 6
RC=$?

kill $SYNC_PID 2>/dev/null || true
exit $RC