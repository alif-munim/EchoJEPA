import boto3
import numpy as np
import torch
import decord
import os
import io
import uuid
import time
import sys
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
    # FIX: Robust split for "path 0" or "path,0" format
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

    def _get_s3_client(self):
        if self.s3_client is None: self.s3_client = boto3.client("s3")
        return self.s3_client

    def __len__(self):
        return len(self.samples)

    # Replace the entire loadvideo_decord method with this:
    def loadvideo_decord(self, sample):
        if sample.startswith("s3://"):
            client = self._get_s3_client()
            bucket, key = sample.replace("s3://", "").split("/", 1)
            
            for attempt in range(3):
                try:
                    # Download to RAM (Fix for moov atom error)
                    obj = client.get_object(Bucket=bucket, Key=key)
                    video_bytes = obj['Body'].read()
                    
                    if len(video_bytes) == 0:
                        raise Exception("Empty file from S3")

                    file_obj = io.BytesIO(video_bytes)
                    vr = decord.VideoReader(file_obj, num_threads=1)
                    return vr
                except Exception as e:
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
        max_retries = 50
        for attempt in range(max_retries):
            cur_idx = idx if attempt == 0 else np.random.randint(len(self.samples))
            path = self.samples[cur_idx]
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
                if attempt < 3 or attempt % 10 == 0:
                    print(f"[WARN] Retry {attempt}/{max_retries} for {path} | Error: {e}", flush=True)
        # Should never reach here with 525K samples, but just in case
        raise RuntimeError(f"Failed after {max_retries} retries (idx={idx})")

def make_videodataset(data_paths, batch_size, frames_per_clip, target_fps, crop_size, num_workers, pin_mem, rank, world_size, log_dir=None):
    dataset = VideoDataset(data_paths, frames_per_clip=frames_per_clip, target_fps=target_fps, crop_size=crop_size)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_mem, drop_last=True, persistent_workers=(num_workers > 0))
    return dataset, loader, sampler