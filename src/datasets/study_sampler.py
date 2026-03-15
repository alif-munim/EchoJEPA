"""
Study-aware distributed sampler for echocardiographic video datasets.

Each epoch, selects exactly 1 random clip per study, then distributes
across ranks. This gives cross-view augmentation (different echo views
per epoch) while ensuring each study is seen exactly once.

Drop-in replacement for DistributedSampler in the training dataloader.
"""

import math
import os
import re
from collections import defaultdict
from logging import getLogger
from typing import Iterator, List, Optional

import torch
from torch.utils.data import Dataset, Sampler

logger = getLogger()


class DistributedStudySampler(Sampler[int]):
    """Distributed sampler that selects 1 random clip per study per epoch.

    Groups dataset rows by study_id (extracted from S3 path), then each epoch:
    1. For each study, randomly selects 1 clip index
    2. Shuffles the selected indices
    3. Distributes across ranks (same as DistributedSampler)

    Args:
        dataset: Dataset with a .samples attribute (list of S3 paths).
        study_ids: Optional pre-computed list of study IDs per sample.
            If None, study IDs are extracted from S3 paths automatically.
        num_replicas: Number of distributed processes (default: world_size).
        rank: Rank of current process (default: current rank).
        seed: Random seed for reproducibility.
        drop_last: If True, drop tail samples that don't fit evenly across ranks.
    """

    def __init__(
        self,
        dataset: Dataset,
        study_ids: Optional[List] = None,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        seed: int = 0,
        drop_last: bool = False,
        class_balance_ratio: Optional[float] = None,
    ):
        if num_replicas is None:
            if not torch.distributed.is_available() or not torch.distributed.is_initialized():
                num_replicas = 1
            else:
                num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available() or not torch.distributed.is_initialized():
                rank = 0
            else:
                rank = torch.distributed.get_rank()

        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        # Build study -> [row_indices] mapping
        self.study_to_indices = defaultdict(list)

        if study_ids is not None:
            for idx, sid in enumerate(study_ids):
                self.study_to_indices[sid].append(idx)
        else:
            samples = dataset.samples if hasattr(dataset, "samples") else dataset.dataset.samples
            for idx, path in enumerate(samples):
                sid = self._extract_study_id(str(path))
                self.study_to_indices[sid].append(idx)

        self.study_keys = sorted(self.study_to_indices.keys())
        self.num_studies = len(self.study_keys)

        # Class balancing: cap each class at ratio × minority_class_count
        if class_balance_ratio is not None and class_balance_ratio > 0:
            labels = dataset.labels if hasattr(dataset, "labels") else dataset.dataset.labels
            # Map study -> label (use first clip's label)
            study_labels = {}
            for sid in self.study_keys:
                study_labels[sid] = labels[self.study_to_indices[sid][0]]
            # Group studies by class
            class_studies = defaultdict(list)
            for sid, label in study_labels.items():
                class_studies[label].append(sid)
            # Cap at ratio × minority count
            min_count = min(len(sids) for sids in class_studies.values())
            cap = int(min_count * class_balance_ratio)
            # Deterministic downsample
            g = torch.Generator()
            g.manual_seed(seed)
            balanced_keys = []
            for label in sorted(class_studies.keys()):
                sids = class_studies[label]
                if len(sids) > cap:
                    perm = torch.randperm(len(sids), generator=g).tolist()
                    sids = [sids[i] for i in perm[:cap]]
                balanced_keys.extend(sids)
            # Remove dropped studies
            dropped = set(self.study_keys) - set(balanced_keys)
            for sid in dropped:
                del self.study_to_indices[sid]
            self.study_keys = sorted(balanced_keys)
            orig = self.num_studies
            self.num_studies = len(self.study_keys)
            logger.info(
                f"Class balancing (ratio={class_balance_ratio}): {orig} -> {self.num_studies} studies "
                f"(cap={cap}/class, {len(class_studies)} classes, min_class={min_count})"
            )

        # Compute per-rank sizes (same logic as DistributedSampler)
        if self.drop_last and self.num_studies % self.num_replicas != 0:
            self.num_samples = math.ceil((self.num_studies - self.num_replicas) / self.num_replicas)
        else:
            self.num_samples = math.ceil(self.num_studies / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    @staticmethod
    def _extract_study_id(path: str) -> str:
        """Extract study ID from S3 path.

        Handles:
          MIMIC: .../s90001295/90001295_0054.mp4 -> "90001295"
          UHN:   .../StudyUID/clip.mp4 -> "StudyUID"

        Falls back to parent directory name.
        """
        # MIMIC pattern: /s{digits}/{digits}_{clip}.mp4
        match = re.search(r"/s(\d+)/\d+_\d+\.mp4$", path)
        if match:
            return match.group(1)
        # Fallback: parent directory of the file
        parent = os.path.basename(os.path.dirname(path))
        return parent

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # For each study, pick 1 random clip
        indices = []
        for key in self.study_keys:
            clip_indices = self.study_to_indices[key]
            pick = torch.randint(len(clip_indices), (1,), generator=g).item()
            indices.append(clip_indices[pick])

        # Shuffle studies
        perm = torch.randperm(len(indices), generator=g).tolist()
        indices = [indices[i] for i in perm]

        # Pad to make evenly divisible by num_replicas
        if len(indices) < self.total_size:
            padding = self.total_size - len(indices)
            indices += indices[:padding]
        elif len(indices) > self.total_size:
            indices = indices[: self.total_size]

        # Subsample for this rank
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
