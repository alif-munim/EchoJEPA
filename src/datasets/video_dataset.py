# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import os
import pathlib
import warnings
from logging import getLogger

## MODIFICATION: Import boto3 for S3 access and io to handle byte streams
import boto3
import io

import numpy as np
import pandas as pd
import torch
import torchvision
from decord import VideoReader, cpu

from src.datasets.utils.dataloader import ConcatIndices, MonitoredDataset, NondeterministicDataLoader
from src.datasets.utils.weighted_sampler import DistributedWeightedSampler

_GLOBAL_SEED = 0
logger = getLogger()


def make_videodataset(
    data_paths,
    batch_size,
    frames_per_clip=8,
    dataset_fpcs=None,
    frame_step=4,
    duration=None,
    fps=None,
    num_clips=1,
    random_clip_sampling=True,
    allow_clip_overlap=False,
    filter_short_videos=False,
    filter_long_videos=int(10**9),
    transform=None,
    shared_transform=None,
    rank=0,
    world_size=1,
    datasets_weights=None,
    collator=None,
    drop_last=True,
    num_workers=10,
    pin_mem=True,
    persistent_workers=True,
    deterministic=True,
    log_dir=None,
):
    dataset = VideoDataset(
        data_paths=data_paths,
        datasets_weights=datasets_weights,
        frames_per_clip=frames_per_clip,
        dataset_fpcs=dataset_fpcs,
        duration=duration,
        fps=fps,
        frame_step=frame_step,
        num_clips=num_clips,
        random_clip_sampling=random_clip_sampling,
        allow_clip_overlap=allow_clip_overlap,
        filter_short_videos=filter_short_videos,
        filter_long_videos=filter_long_videos,
        shared_transform=shared_transform,
        transform=transform,
    )

    log_dir = pathlib.Path(log_dir) if log_dir else None
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        # Worker ID will replace '%w'
        resource_log_filename = log_dir / f"resource_file_{rank}_%w.csv"
        dataset = MonitoredDataset(
            dataset=dataset,
            log_filename=str(resource_log_filename),
            log_interval=10.0,
            monitor_interval=5.0,
        )

    logger.info("VideoDataset dataset created")
    if datasets_weights is not None:
        dist_sampler = DistributedWeightedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    else:
        dist_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True
        )

    if deterministic:
        data_loader = torch.utils.data.DataLoader(
            dataset,
            collate_fn=collator,
            sampler=dist_sampler,
            batch_size=batch_size,
            drop_last=drop_last,
            pin_memory=pin_mem,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0) and persistent_workers,
            prefetch_factor=2
        )
    else:
        data_loader = NondeterministicDataLoader(
            dataset,
            collate_fn=collator,
            sampler=dist_sampler,
            batch_size=batch_size,
            drop_last=drop_last,
            pin_memory=pin_mem,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0) and persistent_workers,
            prefetch_factor=2
        )
    logger.info("VideoDataset unsupervised data loader created")

    return dataset, data_loader, dist_sampler


class VideoDataset(torch.utils.data.Dataset):
    """
    Video classification dataset that efficiently streams data from S3.
    """

    def __init__(
        self,
        data_paths,
        datasets_weights=None,
        frames_per_clip=16,
        fps=None,
        dataset_fpcs=None,
        frame_step=4,
        num_clips=1,
        transform=None,
        shared_transform=None,
        random_clip_sampling=True,
        allow_clip_overlap=False,
        filter_short_videos=False,
        filter_long_videos=int(10**9),
        duration=None,  # duration in seconds
    ):
        self.data_paths = data_paths
        self.datasets_weights = datasets_weights
        self.frame_step = frame_step
        self.num_clips = num_clips
        self.transform = transform
        self.shared_transform = shared_transform
        self.random_clip_sampling = random_clip_sampling
        self.allow_clip_overlap = allow_clip_overlap
        self.filter_short_videos = filter_short_videos
        self.filter_long_videos = filter_long_videos
        self.duration = duration
        self.fps = fps

        # Initialize s3_client to None. It will be created on-demand in each worker process
        # to prevent pickling errors with multiprocessing.
        self.s3_client = None
        
        if sum([v is not None for v in (fps, duration, frame_step)]) != 1:
            raise ValueError(f"Must specify exactly one of either {fps=}, {duration=}, or {frame_step=}.")

        if isinstance(data_paths, str):
            data_paths = [data_paths]

        if dataset_fpcs is None:
            self.dataset_fpcs = [frames_per_clip for _ in data_paths]
        else:
            if len(dataset_fpcs) != len(data_paths):
                raise ValueError("Frames per clip not properly specified for data paths")
            self.dataset_fpcs = dataset_fpcs

        if VideoReader is None:
            raise ImportError('Unable to import "decord" which is required to read videos.')

        # Load video paths and labels from the annotation file(s)
        samples, labels = [], []
        self.num_samples_per_dataset = []
        for data_path in self.data_paths:
            if data_path.endswith(".csv"):
                try:
                    data = pd.read_csv(data_path, header=None, delimiter=" ")
                except pd.errors.ParserError:
                    data = pd.read_csv(data_path, header=None, delimiter="::")
                samples.extend(list(data.values[:, 0]))
                labels.extend(list(data.values[:, 1]))
                self.num_samples_per_dataset.append(len(data))
            elif data_path.endswith(".npy"):
                data = np.load(data_path, allow_pickle=True)
                data = [repr(x)[1:-1] for x in data]
                samples.extend(data)
                labels.extend([0] * len(data))
                self.num_samples_per_dataset.append(len(data))
                
        self.per_dataset_indices = ConcatIndices(self.num_samples_per_dataset)

        self.sample_weights = None
        if self.datasets_weights is not None:
            self.sample_weights = []
            for dw, ns in zip(self.datasets_weights, self.num_samples_per_dataset):
                self.sample_weights.extend([dw / ns] * ns)

        self.samples = samples
        self.labels = labels

        print(f"Loaded {len(self.samples)} samples")  
        print(f"First 5 samples: {self.samples[:5]}")  
        print(f"Sample types: {[type(s) for s in self.samples[:5]]}")

    def __getitem__(self, index):  
        loaded_sample = False  
        retry_count = 0  
        max_retries = 10  # Prevent infinite loops  
          
        while not loaded_sample and retry_count < max_retries:  
            sample_path = self.samples[index]  
            if not isinstance(sample_path, str):  
                logger.warning("Invalid sample path.")  
            else:  
                is_image = sample_path.split(".")[-1].lower() in ("jpg", "png", "jpeg")  
                if is_image:  
                    loaded_sample = self.get_item_image(index)  
                else:  
                    loaded_sample = self.get_item_video(index)  
              
            if not loaded_sample:  
                warnings.warn(f"Retrying with new sample, failed to load: {self.samples[index]}")  
                index = np.random.randint(len(self))  
                retry_count += 1  
      
        if not loaded_sample:  
            raise RuntimeError(f"Failed to load any valid samples after {max_retries} retries")  
          
        return loaded_sample

    def get_item_video(self, index):    
        if self.s3_client is None:    
            try:    
                self.s3_client = boto3.client("s3")    
                # Test the connection immediately    
                self.s3_client.list_buckets()    
            except Exception as e:    
                logger.error(f"Failed to initialize S3 client: {e}")    
                return False  
      
        sample_uri = self.samples[index]  
        dataset_idx, _ = self.per_dataset_indices[index]  
        frames_per_clip = self.dataset_fpcs[dataset_idx]  
      
        buffer, clip_indices = self.loadvideo_decord(sample_uri, frames_per_clip)  
          
        # Check for None FIRST to avoid TypeError  
        if buffer is None:  
            logger.warning(f"Failed to load video: {sample_uri}")  
            return False  
              
        # Then check for empty buffer  
        if len(buffer) == 0:    
            return False    
          
        label = self.labels[index]  
      
        def split_into_clips(video):  
            fpc = frames_per_clip  
            nc = self.num_clips  
            return [video[i * fpc : (i + 1) * fpc] for i in range(nc)]  
      
        if self.shared_transform is not None:  
            buffer = self.shared_transform(buffer)  
        buffer = split_into_clips(buffer)  
        if self.transform is not None:  
            buffer = [self.transform(clip) for clip in buffer]  
      
        return buffer, label, clip_indices
    
    def get_item_image(self, index):
        """
        Handles the logic for loading and processing a single image sample.
        """
        # Lazily initialize the S3 client in the worker process
        if self.s3_client is None:
            self.s3_client = boto3.client("s3")

        sample_uri = self.samples[index]
        dataset_idx, _ = self.per_dataset_indices[index]
        fpc = self.dataset_fpcs[dataset_idx]
        
        try:
            if not sample_uri.startswith("s3://"):
                raise ValueError(f"Image path is not a valid S3 URI: {sample_uri}")
            
            bucket_name, key = sample_uri.replace("s3://", "").split("/", 1)
            response = self.s3_client.get_object(Bucket=bucket_name, Key=key)
            image_bytes = response['Body'].read()
            image_tensor = torchvision.io.decode_image(torch.from_numpy(np.frombuffer(image_bytes, np.uint8)), mode=torchvision.io.ImageReadMode.RGB)
        except Exception as e:
            logger.warning(f"Failed to load image {sample_uri}: {e}")
            return False

        label = self.labels[index]
        clip_indices = [np.arange(start=0, stop=fpc, dtype=np.int32)]
        buffer = image_tensor.unsqueeze(dim=0).repeat((fpc, 1, 1, 1))
        buffer = buffer.permute((0, 2, 3, 1))

        if self.shared_transform is not None:
            buffer = self.shared_transform(buffer)
        if self.transform is not None:
            buffer = [self.transform(buffer)]

        return buffer, label, clip_indices

    def debug_sample_loading(self, index):  
        sample_uri = self.samples[index]  
        print(f"Attempting to load sample {index}: {sample_uri}")  
        print(f"Sample type: {type(sample_uri)}")  
        print(f"Is string: {isinstance(sample_uri, str)}")  
        print(f"Starts with s3://: {sample_uri.startswith('s3://') if isinstance(sample_uri, str) else False}")  
          
        if self.s3_client is None:  
            print("S3 client not initialized")  
            return  
              
        try:  
            bucket_name, key = sample_uri.replace("s3://", "").split("/", 1)  
            print(f"Bucket: {bucket_name}, Key: {key}")  
            response = self.s3_client.head_object(Bucket=bucket_name, Key=key)  
            print(f"Object exists, size: {response['ContentLength']}")  
        except Exception as e:  
            print(f"S3 error: {e}")

    def loadvideo_decord(self, sample_uri, fpc):  
        if not sample_uri.startswith("s3://"):  
            logger.warning(f"Invalid S3 URI: {sample_uri}")  
            return [], None  # Match original return format  
          
        try:  
            bucket_name, key = sample_uri.replace("s3://", "").split("/", 1)  
              
            # More specific exception handling  
            try:  
                response = self.s3_client.head_object(Bucket=bucket_name, Key=key)  
                _fsize = response['ContentLength']  
            except self.s3_client.exceptions.NoSuchKey:  
                logger.warning(f"S3 object not found: {sample_uri}")  
                return [], None  
            except self.s3_client.exceptions.ClientError as e:  
                logger.warning(f"S3 access error for {sample_uri}: {e}")  
                return [], None
      
            if _fsize > self.filter_long_videos:  
                logger.warning(f"Skipping long video {sample_uri} of size {_fsize} bytes")  
                return None, None  
      
            # Download and process video  
            video_object = self.s3_client.get_object(Bucket=bucket_name, Key=key)  
            video_bytes = io.BytesIO(video_object['Body'].read())  
              
            vr = VideoReader(video_bytes, num_threads=-1, ctx=cpu(0))  
              
        except Exception as e:  
            logger.warning(f"Failed to load video {sample_uri}. Error: {e}")  
            return None, None  
            
        fstp = self.frame_step
        if self.duration is not None or self.fps is not None:
            try:
                video_fps = math.ceil(vr.get_avg_fps())
            except Exception as e:
                logger.warning(e)
                video_fps = 30 # Default if FPS can't be read

            if self.duration is not None:
                fstp = int(self.duration * video_fps / fpc)
            else:
                fstp = video_fps // self.fps
        
        fstp = max(1, fstp)
        clip_len = int(fpc * fstp)

        if self.filter_short_videos and len(vr) < clip_len:
            warnings.warn(f"Skipping short video {sample_uri} of length {len(vr)}")
            return None, None

        vr.seek(0)
        partition_len = len(vr) // self.num_clips
        all_indices, clip_indices = [], []

        for i in range(self.num_clips):
            if partition_len > clip_len:
                end_indx = clip_len
                if self.random_clip_sampling:
                    end_indx = np.random.randint(clip_len, partition_len)
                start_indx = end_indx - clip_len
                indices = np.linspace(start_indx, end_indx, num=fpc, dtype=np.int64)
                indices = np.clip(indices, start_indx, end_indx - 1)
                indices = indices + i * partition_len
            else:
                if not self.allow_clip_overlap:
                    indices = np.linspace(0, partition_len, num=partition_len // fstp, dtype=np.int64)
                    indices = np.concatenate((indices, np.ones(fpc - len(indices), dtype=np.int64) * (partition_len-1)))
                    indices = np.clip(indices, 0, partition_len - 1)
                    indices = indices + i * partition_len
                else:
                    sample_len = min(clip_len, len(vr))
                    indices = np.linspace(0, sample_len-1, num=fpc, dtype=np.int64)
                    clip_step = 0
                    if len(vr) > clip_len and self.num_clips > 1:
                       clip_step = (len(vr) - clip_len) // (self.num_clips - 1)
                    indices = indices + i * clip_step
            
            clip_indices.append(indices)
            all_indices.extend(list(indices))

        buffer = vr.get_batch(all_indices).asnumpy()
        return buffer, clip_indices
    
    def __len__(self):
        return len(self.samples)