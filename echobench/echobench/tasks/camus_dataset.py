"""CAMUS echocardiography segmentation dataset for EchoBench.

Loads half-sequence videos and dense ground truth segmentation masks.
Returns 16-frame video clips with ED/ES masks for segmentation evaluation.

Labels: 0=background, 1=LV cavity, 2=myocardium, 3=left atrium

Ported from evals/segmentation_frozen/camus_dataset.py with an added
``normalize`` flag for noise injection before ImageNet normalization.

Reference:
    Leclerc et al. "Deep Learning for Segmentation using an Open Large-Scale
    Dataset in 2D Echocardiography" IEEE TMI 2019.
"""

import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

LABEL_NAMES = {0: "BG", 1: "LV", 2: "MYO", 3: "LA"}
NUM_CLASSES = 4

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def load_split(camus_root, split="training"):
    """Load patient IDs for a given split."""
    split_file = os.path.join(camus_root, "database_split", f"subgroup_{split}.txt")
    with open(split_file) as f:
        return [line.strip() for line in f if line.strip()]


class CAMUSSegDataset(Dataset):
    """CAMUS dataset returning video clips with per-frame segmentation masks.

    Args:
        patient_ids: list of patient identifiers
        camus_root: path to CAMUS_public directory
        views: tuple of views to use ("4CH", "2CH", or both)
        resolution: target spatial resolution
        num_frames: number of frames to uniformly sample
        augment: enable random horizontal flips
        normalize: if True (default), apply ImageNet normalization to video.
            Set to False for EchoBench noise injection (noise is applied
            to unnormalized [0,1] video, then normalized manually).
    """

    def __init__(
        self,
        patient_ids,
        camus_root,
        views=("4CH",),
        resolution=224,
        num_frames=16,
        augment=False,
        normalize=True,
    ):
        self.camus_root = camus_root
        self.resolution = resolution
        self.num_frames = num_frames
        self.augment = augment
        self.normalize = normalize

        self.samples = []
        db_dir = os.path.join(camus_root, "database_nifti")
        for pid in patient_ids:
            for view in views:
                seq_path = os.path.join(db_dir, pid, f"{pid}_{view}_half_sequence.nii.gz")
                if os.path.exists(seq_path):
                    self.samples.append((pid, view))

    def __len__(self):
        return len(self.samples)

    def _load_config(self, patient_dir, view):
        cfg_path = os.path.join(patient_dir, f"Info_{view}.cfg")
        config = {}
        with open(cfg_path) as f:
            for line in f:
                parts = line.strip().split(": ", 1)
                if len(parts) == 2:
                    config[parts[0]] = parts[1]
        return config

    def __getitem__(self, idx):
        import nibabel as nib

        patient_id, view = self.samples[idx]
        patient_dir = os.path.join(self.camus_root, "database_nifti", patient_id)

        config = self._load_config(patient_dir, view)
        ed_frame = int(config["ED"]) - 1  # 1-indexed -> 0-indexed
        es_frame = int(config["ES"]) - 1

        seq = nib.load(os.path.join(patient_dir, f"{patient_id}_{view}_half_sequence.nii.gz")).get_fdata()
        gt = nib.load(os.path.join(patient_dir, f"{patient_id}_{view}_half_sequence_gt.nii.gz")).get_fdata()
        n_frames = seq.shape[2]

        frame_indices = np.linspace(0, n_frames - 1, self.num_frames, dtype=int)
        ed_sampled_idx = int(np.argmin(np.abs(frame_indices - ed_frame)))
        es_sampled_idx = int(np.argmin(np.abs(frame_indices - es_frame)))

        frames = seq[:, :, frame_indices]  # [H, W, T']
        gt_frames = gt[:, :, frame_indices]

        frames = torch.from_numpy(frames.copy()).float().permute(2, 0, 1)  # [T, H, W]
        gt_frames = torch.from_numpy(gt_frames.copy()).long().permute(2, 0, 1)

        fmax = frames.max()
        if fmax > 0:
            frames = frames / fmax

        frames = F.interpolate(
            frames.unsqueeze(1), size=(self.resolution, self.resolution),
            mode="bilinear", align_corners=False,
        ).squeeze(1)

        gt_frames = F.interpolate(
            gt_frames.float().unsqueeze(1), size=(self.resolution, self.resolution),
            mode="nearest",
        ).squeeze(1).long()

        # Grayscale -> RGB [C, T, H, W]
        video = frames.unsqueeze(1).expand(-1, 3, -1, -1).permute(1, 0, 2, 3).contiguous()

        if self.augment and torch.rand(1).item() < 0.5:
            video = video.flip(-1)
            gt_frames = gt_frames.flip(-1)

        ed_temporal_token = ed_sampled_idx // 2
        es_temporal_token = es_sampled_idx // 2

        result = {
            "video_raw": video,  # [3, T, H, W] in [0, 1], always provided
            "gt_all": gt_frames,
            "ed_mask": gt_frames[ed_sampled_idx],
            "es_mask": gt_frames[es_sampled_idx],
            "ed_temporal_token": ed_temporal_token,
            "es_temporal_token": es_temporal_token,
            "patient_id": patient_id,
            "view": view,
        }

        if self.normalize:
            result["video"] = (video - IMAGENET_MEAN.unsqueeze(1)) / IMAGENET_STD.unsqueeze(1)
        else:
            result["video"] = None

        return result
