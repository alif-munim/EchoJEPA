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
        
        # Prefer /dev/shm for speed (RAM), fallback to /tmp
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
                
                obj_meta = client.head_object(Bucket=bucket, Key=key)
                expected_bytes = obj_meta.get("ContentLength")
                if expected_bytes is not None:
                    written_bytes = os.path.getsize(tmp_path)
                    if written_bytes != expected_bytes:
                        raise RuntimeError(f"Incomplete download: {written_bytes}/{expected_bytes} bytes")

                vr = decord.VideoReader(tmp_path, num_threads=1)
                return vr, tmp_path

            except Exception as e:
                last_err = e
                if tmp_path is not None and os.path.exists(tmp_path):
                    try: os.remove(tmp_path)
                    except: pass
                time.sleep(0.1 * (2 ** attempt))

        raise RuntimeError(f"Failed to load {sample} after retries: {last_err}")
        
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
        tmp_path = None
        
        try:
            vr, tmp_path = self.loadvideo_decord(path)
            
            indices = self._sample_indices_by_fps(vr)
            frames = vr.get_batch(indices).asnumpy()
            frames = torch.from_numpy(frames).float().permute(0, 3, 1, 2) / 255.0
            
            i, j, h, w = transforms.RandomResizedCrop.get_params(frames[0], scale=self.rrc_scale, ratio=self.rrc_ratio)
            frames = F.resized_crop(frames, i, j, h, w, size=(self.crop_size, self.crop_size))
            frames = self.normalize(frames)
            frames = frames.permute(1, 0, 2, 3).contiguous()
            
            # --- CRITICAL SAFETY CHECK: NAN/INF ---
            if torch.isnan(frames).any() or torch.isinf(frames).any():
                raise RuntimeError(f"Frame tensor contains NaNs or Infs for {path}")
            # --------------------------------------

            return frames, torch.from_numpy(mask).bool()
            
        except Exception as e:
            # === CRITICAL CHANGE: RECURSIVE RETRY ===
            print(f"[WARN] Failed to load {path} (Error: {e}). Retrying with new sample...", flush=True)
            new_idx = np.random.randint(0, len(self))
            return self.__getitem__(new_idx)
            
        finally:
            if tmp_path is not None and os.path.exists(tmp_path):
                try: os.remove(tmp_path)
                except: pass

def make_videodataset(data_paths, batch_size, frames_per_clip, target_fps, crop_size, num_workers, pin_mem, rank, world_size, log_dir=None):
    dataset = VideoDataset(data_paths, frames_per_clip=frames_per_clip, target_fps=target_fps, crop_size=crop_size)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_mem, drop_last=True, persistent_workers=(num_workers > 0))
    return dataset, loader, sampler