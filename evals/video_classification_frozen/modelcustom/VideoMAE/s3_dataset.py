import boto3
import numpy as np
import torch
import decord
import os
import time
import sys
import tempfile
from pathlib import Path
from torch.utils.data import DistributedSampler, DataLoader, Dataset
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
    return line.split()[0].strip()

class VideoDataset(Dataset):
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
        
        default_tmp = "/dev/shm" if os.path.exists("/dev/shm") else "/tmp"
        self.tmp_dir = os.getenv("VIDEOMAE_TMP", default_tmp)
        Path(self.tmp_dir).mkdir(parents=True, exist_ok=True)

    def _get_s3_client(self):
        if self.s3_client is None: self.s3_client = boto3.client("s3")
        return self.s3_client

    def __len__(self):
        return len(self.samples)

    def loadvideo_decord(self, sample):
        if not sample.startswith("s3://"):
            return decord.VideoReader(sample, num_threads=1), None

        client = self._get_s3_client()
        bucket, key = sample.replace("s3://", "").split("/", 1)

        last_err = None
        for attempt in range(3):
            tmp_path = None
            try:
                suffix = os.path.splitext(key)[-1] or ".mp4"
                fd, tmp_path = tempfile.mkstemp(prefix="s3_", suffix=suffix, dir=self.tmp_dir)
                os.close(fd) 
                client.download_file(Bucket=bucket, Key=key, Filename=tmp_path)
                
                # Check if file is non-empty
                if os.path.getsize(tmp_path) < 100:
                    raise RuntimeError("File too small")

                vr = decord.VideoReader(tmp_path, num_threads=1)
                return vr, tmp_path
            except Exception as e:
                last_err = e
                if tmp_path and os.path.exists(tmp_path):
                    try: os.remove(tmp_path)
                    except: pass
                time.sleep(0.1 * (2 ** attempt))
        raise RuntimeError(f"Failed to load {sample}: {last_err}")

    def __getitem__(self, idx):
        path = self.samples[idx]
        mask = self.masked_position_generator()
        tmp_path = None
        try:
            vr, tmp_path = self.loadvideo_decord(path)
            duration = len(vr)
            if duration <= 0: raise RuntimeError("Zero duration")

            # Simple sampling
            indices = np.linspace(0, duration - 1, self.frames_per_clip).astype(int)
            frames = vr.get_batch(indices).asnumpy()
            
            # --- NUMERICAL SANITY CHECK ---
            if np.isnan(frames).any() or np.isinf(frames).any():
                raise RuntimeError("NaN in raw video pixels")
            
            frames = torch.from_numpy(frames).float().permute(0, 3, 1, 2) / 255.0
            
            # Augmentation
            i, j, h, w = transforms.RandomResizedCrop.get_params(frames[0], scale=self.rrc_scale, ratio=self.rrc_ratio)
            frames = F.resized_crop(frames, i, j, h, w, size=(self.crop_size, self.crop_size))
            frames = self.normalize(frames)
            frames = frames.permute(1, 0, 2, 3).contiguous()

            # --- FINAL TENSOR CHECK ---
            if torch.isnan(frames).any() or (torch.abs(frames) > 20).any():
                raise RuntimeError("Tensor out of bounds or NaN after normalization")

            return frames, torch.from_numpy(mask).bool(), path
            
        except Exception as e:
            print(f"[STRICT-RETRY] Skipping {path} due to: {e}", flush=True)
            new_idx = np.random.randint(0, len(self))
            return self.__getitem__(new_idx)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try: os.remove(tmp_path)
                except: pass

def make_videodataset(data_paths, batch_size, frames_per_clip, target_fps, crop_size, num_workers, pin_mem, rank, world_size, log_dir=None):
    dataset = VideoDataset(data_paths, frames_per_clip=frames_per_clip, target_fps=target_fps, crop_size=crop_size)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_mem, drop_last=True, persistent_workers=(num_workers > 0))
    return dataset, loader, sampler