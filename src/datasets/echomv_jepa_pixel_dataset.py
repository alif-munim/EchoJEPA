"""Per-study pixel dataset for EchoMV-JEPA Option A.

The pooled-cache dataset (``src/datasets/echomv_jepa_dataset.py``) reads
precomputed 1024-d ``c_clip`` vectors from NVMe. Option A instead reads raw
clips at training time so the study transformer can consume per-clip
**tokens** (not pooled vectors) from the frozen V-JEPA encoder.

This dataset:

  - Reads the same K-sample manifest EchoSet v1 and Stage-1 use
    (``study_clip_sample_K8_seed0_train.parquet``).
  - Groups clips by study_id.
  - For each study, loads K raw clip tensors from S3 via the existing
    ``VideoDataset`` machinery (handles boto3 + Decord + sampling +
    transform).
  - Yields per-study dicts with the same pad/meta/mask layout as the
    pooled dataset **except** the visual content lives in
    ``ctx_clips`` / ``tgt_clips`` of shape ``(M_*, 3, T_frames, H, W)``
    instead of ``ctx_elements`` / ``tgt_elements`` of shape ``(M_*, d_clip)``.

The collate stacks per-study batches into fixed (B, max_M, 3, T, H, W)
tensors with padding masks.

Element grouping still happens at the clip level — multiple clips in the
same ``(view_family, modality, phase_bucket)`` key are aggregated **after**
the encoder by mean-pooling their token outputs. For Stage-1 Option A we
treat each K-sample clip as its own element (no grouping) to keep M as
large as possible: M = number of valid clips returned by the loader.
This simplifies the dataset and lets the token-level attention have as
much cross-element structure as possible. If the smoke succeeds, a
token-aware grouping step can be added later.
"""

from __future__ import annotations

import hashlib
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.datasets.echoset_jepa_dataset import (
    DEFAULT_MASK_STRATEGY_WEIGHTS,
    pick_mask_indices,
)
from src.datasets.video_dataset import VideoDataset
from src.models.meta_embeddings import MetaEmbeddings

logger = logging.getLogger(__name__)


def _quality_bucket_from_score(q: float) -> str:
    if q != q:  # NaN
        return "unknown"
    if q >= 0.75:
        return "high"
    if q >= 0.55:
        return "med"
    return "low"


class EchoMVJEPAPixelDataset(Dataset):
    """One study per __getitem__. Loads K raw clips via VideoDataset.

    Parameters
    ----------
    k_sample_manifest_path : str
        Parquet path with columns study_id, clip_id, s3_uri, view_family,
        modality, phase_bucket, measurement_site, quality_score.
    meta : MetaEmbeddings
        For vocabulary lookups (view/modality/phase/quality).
    frames_per_clip : int, default 16
    frame_step : int, default 2
    resolution : int, default 224
    transform : callable or None
        Per-clip transform applied inside VideoDataset.
    strategy_weights : optional mask-strategy weights dict.
    seed : int, default 0
    permute_every_step : bool, default True
    """

    def __init__(
        self,
        k_sample_manifest_path: str,
        meta: MetaEmbeddings,
        *,
        frames_per_clip: int = 16,
        frame_step: int = 2,
        resolution: int = 224,
        transform=None,
        strategy_weights: Optional[Dict[str, float]] = None,
        seed: int = 0,
        permute_every_step: bool = True,
    ) -> None:
        self.meta = meta
        self.strategy_weights = strategy_weights or DEFAULT_MASK_STRATEGY_WEIGHTS
        self._rng = random.Random(seed)
        self.permute_every_step = permute_every_step
        self.frames_per_clip = int(frames_per_clip)
        self.frame_step = int(frame_step)
        self.resolution = int(resolution)

        df = pd.read_parquet(k_sample_manifest_path)
        # Preserve the manifest row order inside each study (sample_K.py emits
        # a deterministic K-slot ordering — we keep it).
        self._studies: List[str] = sorted(df["study_id"].astype(str).unique().tolist())
        self._rows_by_study: Dict[str, List[Dict[str, Any]]] = {
            sid: grp.to_dict("records") for sid, grp in df.groupby(df["study_id"].astype(str))
        }

        # Build a flat s3_uri list — one per manifest row — and a study→clip-
        # indices map. VideoDataset reads the CSV on disk; we write it once.
        self._flat_uris: List[str] = []
        self._study_to_clip_idx: Dict[str, List[int]] = {}
        self._study_to_rows: Dict[str, List[Dict[str, Any]]] = {}
        for sid in self._studies:
            rows = self._rows_by_study[sid]
            self._study_to_clip_idx[sid] = []
            self._study_to_rows[sid] = rows
            for r in rows:
                self._study_to_clip_idx[sid].append(len(self._flat_uris))
                self._flat_uris.append(str(r["s3_uri"]))

        # Write the CSV that VideoDataset needs (uri + dummy label).
        self._csv_path = Path(f"/tmp/echomv_pixel_dataset_{abs(hash(k_sample_manifest_path)) % 10**12}.csv")
        self._csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._csv_path, "w") as f:
            for uri in self._flat_uris:
                f.write(f"{uri} 0\n")

        # Single underlying VideoDataset shared by all workers.
        self._vds = VideoDataset(
            data_paths=[str(self._csv_path)],
            frames_per_clip=self.frames_per_clip,
            frame_step=self.frame_step,
            num_clips=1,
            transform=transform,
            shared_transform=None,
            allow_clip_overlap=True,
            filter_short_videos=False,
            duration=None,
        )
        # Expected clip tensor shape from VideoDataset.get_item_video: list of
        # (C, T, H, W) tensors (one per num_clips).  We request num_clips=1.

    def __len__(self) -> int:
        return len(self._studies)

    @staticmethod
    def _study_id_int(sid: str) -> int:
        return int(hashlib.sha256(str(sid).encode()).hexdigest()[:15], 16)

    def _load_clip(self, clip_idx: int) -> Optional[torch.Tensor]:
        """Return (3, T_frames, H, W) or None on decode failure."""
        try:
            out = self._vds.get_item_video(clip_idx)
        except Exception as e:  # noqa: BLE001
            logger.warning("clip %d load failed: %s", clip_idx, e)
            return None
        if out is None:
            return None
        buffer, _label, _clip_indices, _uri, _phase = out
        # buffer is a list with one tensor per num_clips; num_clips=1.
        if not buffer or buffer[0] is None:
            return None
        t = buffer[0]
        if isinstance(t, list):
            t = t[0]
        # Ensure (C, T, H, W). VideoDataset transforms normally yield CTHW.
        if t.dim() == 4 and t.shape[0] == 3:
            return t
        if t.dim() == 4 and t.shape[1] == 3:
            return t.permute(1, 0, 2, 3).contiguous()
        raise RuntimeError(f"unexpected clip tensor shape {tuple(t.shape)}")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sid = self._studies[idx]
        rows = self._study_to_rows[sid]
        clip_indices = self._study_to_clip_idx[sid]

        clip_tensors: List[torch.Tensor] = []
        keys: List[Tuple[str, str, str]] = []
        qs: List[float] = []
        for r, cidx in zip(rows, clip_indices):
            t = self._load_clip(cidx)
            if t is None:
                continue
            clip_tensors.append(t)
            keys.append(
                (
                    r.get("view_family", "unknown"),
                    r.get("modality", "b_mode"),
                    r.get("phase_bucket", "unknown"),
                )
            )
            qs.append(float(r.get("quality_score", 0.5)))

        if not clip_tensors:
            # Degenerate: no clips loaded. Emit a single all-zero placeholder.
            T, H, W = self.frames_per_clip, self.resolution, self.resolution
            zero_clip = torch.zeros(3, T, H, W)
            return self._empty_study_batch(sid, zero_clip)

        # Option A treats each loaded clip as its own element (no grouping).
        M = len(clip_tensors)

        # Permute element order every step (matches EchoSet v1 policy).
        if self.permute_every_step and M > 1:
            perm = list(range(M))
            self._rng.shuffle(perm)
            clip_tensors = [clip_tensors[i] for i in perm]
            keys = [keys[i] for i in perm]
            qs = [qs[i] for i in perm]

        # Mask strategy — same as pooled path.
        if M < 2:
            ctx, tgt, strategy = list(range(M)), [], "none"
        else:
            ctx, tgt, strategy = pick_mask_indices(M, keys, self.strategy_weights, rng=self._rng)

        ctx_clips = (
            torch.stack([clip_tensors[i] for i in ctx], dim=0)
            if ctx
            else torch.zeros(0, 3, self.frames_per_clip, self.resolution, self.resolution)
        )
        tgt_clips = (
            torch.stack([clip_tensors[i] for i in tgt], dim=0)
            if tgt
            else torch.zeros(0, 3, self.frames_per_clip, self.resolution, self.resolution)
        )

        def _meta_ids(idxs: List[int]) -> Dict[str, torch.Tensor]:
            k = [keys[i] for i in idxs]
            q_l = [qs[i] for i in idxs]
            view = torch.tensor([self.meta.view_id(x[0]) for x in k], dtype=torch.long)
            mod = torch.tensor([self.meta.modality_id(x[1]) for x in k], dtype=torch.long)
            phase = torch.tensor([self.meta.phase_id(x[2]) for x in k], dtype=torch.long)
            qual = torch.tensor([self.meta.quality_id(_quality_bucket_from_score(q)) for q in q_l], dtype=torch.long)
            return {"view": view, "modality": mod, "phase": phase, "quality": qual}

        ctx_meta = _meta_ids(ctx)
        tgt_meta = _meta_ids(tgt)

        return {
            "ctx_clips": ctx_clips,  # (M_ctx, 3, T, H, W)
            "tgt_clips": tgt_clips,  # (M_tgt, 3, T, H, W)
            "ctx_meta_view": ctx_meta["view"],
            "ctx_meta_modality": ctx_meta["modality"],
            "ctx_meta_phase": ctx_meta["phase"],
            "ctx_meta_quality": ctx_meta["quality"],
            "tgt_meta_view": tgt_meta["view"],
            "tgt_meta_modality": tgt_meta["modality"],
            "tgt_meta_phase": tgt_meta["phase"],
            "tgt_meta_quality": tgt_meta["quality"],
            "mask_strategy": strategy,
            "study_id": sid,
            "study_id_int": self._study_id_int(sid),
            "n_elements": M,
        }

    def _empty_study_batch(self, sid: str, zero_clip: torch.Tensor) -> Dict[str, Any]:
        return {
            "ctx_clips": zero_clip.unsqueeze(0),  # one placeholder element
            "tgt_clips": torch.zeros_like(zero_clip).unsqueeze(0)[:0],
            "ctx_meta_view": torch.tensor([self.meta.view_id("unknown")], dtype=torch.long),
            "ctx_meta_modality": torch.tensor([self.meta.modality_id("unknown")], dtype=torch.long),
            "ctx_meta_phase": torch.tensor([self.meta.phase_id("unknown")], dtype=torch.long),
            "ctx_meta_quality": torch.tensor([self.meta.quality_id("unknown")], dtype=torch.long),
            "tgt_meta_view": torch.zeros(0, dtype=torch.long),
            "tgt_meta_modality": torch.zeros(0, dtype=torch.long),
            "tgt_meta_phase": torch.zeros(0, dtype=torch.long),
            "tgt_meta_quality": torch.zeros(0, dtype=torch.long),
            "mask_strategy": "empty",
            "study_id": sid,
            "study_id_int": self._study_id_int(sid),
            "n_elements": 0,
        }


def echomv_pixel_collate(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Pad variable-length study pixel batches.

    Produces the EchoMV batch contract expected by the token-mode training
    step (``training_step_echomv_tokens`` in ``app/echomv_jepa/train.py``):

      ctx_clips        (B, max_ctx, 3, T, H, W)
      tgt_clips        (B, max_tgt, 3, T, H, W)
      full_clips       (B, max_ctx+max_tgt, 3, T, H, W)   — concat along elem
      ctx_pad_mask     (B, max_ctx)  bool
      tgt_pad_mask     (B, max_tgt)  bool
      full_pad_mask    (B, max_ctx+max_tgt) bool
      ctx_meta_*       (B, max_ctx) long
      tgt_meta_*       (B, max_tgt) long
      full_meta_*      (B, max_ctx+max_tgt) long
      target_idx_in_full  (B, max_tgt) long
      context_idx_in_full (B, max_ctx) long
      study_id_int, n_elements, mask_strategies
    """
    B = len(batch)
    # Determine per-batch max M_ctx / M_tgt.
    max_ctx = max(1, max(b["ctx_clips"].shape[0] for b in batch))
    max_tgt = max(1, max(b["tgt_clips"].shape[0] for b in batch))
    # Spatial / temporal dims taken from first non-empty clip.
    for b in batch:
        if b["ctx_clips"].numel() > 0:
            _, C, T, H, W = b["ctx_clips"].shape
            break
    else:
        C, T, H, W = 3, 16, 224, 224  # safe fallback for empty-batch edge case

    ctx_clips = torch.zeros(B, max_ctx, C, T, H, W)
    tgt_clips = torch.zeros(B, max_tgt, C, T, H, W)
    ctx_pad = torch.ones(B, max_ctx, dtype=torch.bool)
    tgt_pad = torch.ones(B, max_tgt, dtype=torch.bool)

    def _zeros_meta(M: int) -> torch.Tensor:
        return torch.zeros(B, M, dtype=torch.long)

    ctx_meta = {k: _zeros_meta(max_ctx) for k in ("view", "modality", "phase", "quality")}
    tgt_meta = {k: _zeros_meta(max_tgt) for k in ("view", "modality", "phase", "quality")}
    study_id_int = torch.zeros(B, dtype=torch.long)
    n_elements = torch.zeros(B, dtype=torch.long)
    strategies: List[str] = []

    for i, b in enumerate(batch):
        mc = int(b["ctx_clips"].shape[0])
        mt = int(b["tgt_clips"].shape[0])
        if mc > 0:
            ctx_clips[i, :mc] = b["ctx_clips"]
            ctx_pad[i, :mc] = False
            ctx_meta["view"][i, :mc] = b["ctx_meta_view"]
            ctx_meta["modality"][i, :mc] = b["ctx_meta_modality"]
            ctx_meta["phase"][i, :mc] = b["ctx_meta_phase"]
            ctx_meta["quality"][i, :mc] = b["ctx_meta_quality"]
        if mt > 0:
            tgt_clips[i, :mt] = b["tgt_clips"]
            tgt_pad[i, :mt] = False
            tgt_meta["view"][i, :mt] = b["tgt_meta_view"]
            tgt_meta["modality"][i, :mt] = b["tgt_meta_modality"]
            tgt_meta["phase"][i, :mt] = b["tgt_meta_phase"]
            tgt_meta["quality"][i, :mt] = b["tgt_meta_quality"]
        study_id_int[i] = b["study_id_int"]
        n_elements[i] = b["n_elements"]
        strategies.append(b["mask_strategy"])

    out: Dict[str, Any] = {
        "ctx_clips": ctx_clips,
        "tgt_clips": tgt_clips,
        "ctx_pad_mask": ctx_pad,
        "tgt_pad_mask": tgt_pad,
        "ctx_meta_view": ctx_meta["view"],
        "ctx_meta_modality": ctx_meta["modality"],
        "ctx_meta_phase": ctx_meta["phase"],
        "ctx_meta_quality": ctx_meta["quality"],
        "tgt_meta_view": tgt_meta["view"],
        "tgt_meta_modality": tgt_meta["modality"],
        "tgt_meta_phase": tgt_meta["phase"],
        "tgt_meta_quality": tgt_meta["quality"],
        "study_id_int": study_id_int,
        "n_elements": n_elements,
        "mask_strategies": strategies,
    }

    # Full-study tensors (ctx ∥ tgt).
    out["full_clips"] = torch.cat([ctx_clips, tgt_clips], dim=1)
    out["full_meta_view"] = torch.cat([ctx_meta["view"], tgt_meta["view"]], dim=1)
    out["full_meta_modality"] = torch.cat([ctx_meta["modality"], tgt_meta["modality"]], dim=1)
    out["full_meta_phase"] = torch.cat([ctx_meta["phase"], tgt_meta["phase"]], dim=1)
    out["full_meta_quality"] = torch.cat([ctx_meta["quality"], tgt_meta["quality"]], dim=1)
    out["full_pad_mask"] = torch.cat([ctx_pad, tgt_pad], dim=1)

    ctx_idx = torch.arange(max_ctx, dtype=torch.long).unsqueeze(0).expand(B, -1).contiguous()
    tgt_idx = (torch.arange(max_tgt, dtype=torch.long).unsqueeze(0).expand(B, -1) + max_ctx).contiguous()
    out["context_idx_in_full"] = ctx_idx
    out["target_idx_in_full"] = tgt_idx
    return out


__all__ = ["EchoMVJEPAPixelDataset", "echomv_pixel_collate"]
