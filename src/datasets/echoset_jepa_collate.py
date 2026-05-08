"""DDP collate for EchoSet-JEPA (PR-N4).

Pads variable-length study elements into rectangular tensors with boolean
padding masks. Reads cached c_clip ``.npy`` files referenced by the
study-level sampler manifest.

The dataset yields one study per ``__getitem__`` call (dict with keys matching
``app.echoset_jepa.train.training_step`` expectations); the collate pads the
batch.

Expected per-row schema in the K-sample manifest (produced by
``experiments.echoset_jepa.sample_K``):
  study_id, clip_id, view_family, modality, phase_bucket,
  measurement_site, quality_score, cached_cclip_s3

The ``cached_cclip_s3`` column may be a local path or s3://; the loader
resolves it via ``cache_prefix`` if a local mirror exists.
"""

from __future__ import annotations

import hashlib
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from experiments.echoset_jepa.quality_proxy import quality_buckets_from_train_tertiles
from src.datasets.echoset_jepa_dataset import (
    DEFAULT_MASK_STRATEGY_WEIGHTS,
    group_into_elements,
    pick_mask_indices,
)
from src.models.meta_embeddings import MetaEmbeddings


# ---------------------------------------------------------------------------
# Per-study dataset
# ---------------------------------------------------------------------------


class EchoSetStudyDataset(Dataset):
    """One study per item. Loads per-clip c_clip vectors from local cache.

    Expected behavior:
      - Read the K-sample manifest (``study_clip_sample_K8_seed0_train.parquet``).
      - For each unique study, load the ≤K cached c_clip vectors.
      - Group into elements at ``__getitem__`` time (so mask sampling picks
        different targets across epochs).
    """

    def __init__(
        self,
        k_sample_manifest_path: str,
        cache_prefix: str,
        meta: MetaEmbeddings,
        element_agg: str = "mean",
        strategy_weights: Optional[Dict[str, float]] = None,
        seed: int = 0,
        permute_every_step: bool = True,
    ) -> None:
        import pandas as pd

        self.cache_prefix = cache_prefix
        self.meta = meta
        self.element_agg = element_agg
        self.strategy_weights = strategy_weights or DEFAULT_MASK_STRATEGY_WEIGHTS
        self._rng = random.Random(seed)
        self.permute_every_step = permute_every_step

        df = pd.read_parquet(k_sample_manifest_path)
        self._studies: List[str] = sorted(df["study_id"].unique().tolist())
        self._by_study: Dict[str, List[Dict]] = {
            sid: grp.to_dict("records") for sid, grp in df.groupby("study_id")
        }

    def __len__(self) -> int:
        return len(self._studies)

    def _load_c_clip(self, study_id: str, clip_id: str) -> Optional[np.ndarray]:
        path = Path(self.cache_prefix) / str(study_id) / f"{clip_id}.npy"
        if not path.exists():
            return None
        return np.load(path)

    def _study_id_int(self, study_id: str) -> int:
        return int(hashlib.sha256(str(study_id).encode()).hexdigest()[:15], 16)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sid = self._studies[idx]
        rows = self._by_study[sid]
        clip_keys: List[Tuple[str, str, str]] = []
        clip_vecs: List[np.ndarray] = []
        clip_qs: List[float] = []
        for r in rows:
            v = self._load_c_clip(sid, r["clip_id"])
            if v is None:
                continue
            clip_keys.append((r.get("view_family", "unknown"),
                              r.get("modality", "b_mode"),
                              r.get("phase_bucket", "unknown")))
            clip_vecs.append(v)
            clip_qs.append(float(r.get("quality_score", 0.5)))

        if not clip_vecs:
            # Return a placeholder with a single all-zero element
            d = 1024
            return self._empty_study_batch(sid, d)

        clip_vecs_arr = np.stack(clip_vecs, axis=0)
        clip_qs_arr = np.asarray(clip_qs, dtype=np.float32)

        elem_keys, elem_vecs, elem_q = group_into_elements(
            clip_keys, clip_vecs_arr, clip_qs_arr, element_agg=self.element_agg
        )
        M = len(elem_keys)
        if self.permute_every_step and M > 1:
            perm = list(range(M))
            self._rng.shuffle(perm)
            elem_keys = [elem_keys[i] for i in perm]
            elem_vecs = elem_vecs[perm]
            elem_q = elem_q[perm]

        # Pick mask partition
        if M < 2:
            # Degenerate study — skip target loss by emitting zero target slots
            ctx, tgt, strategy = list(range(M)), [], "none"
        else:
            ctx, tgt, strategy = pick_mask_indices(
                M, elem_keys, self.strategy_weights, rng=self._rng
            )

        def _meta_ids(keys: List[Tuple[str, str, str]], qs: List[float]) -> Dict[str, torch.Tensor]:
            view = torch.tensor([self.meta.view_id(k[0]) for k in keys], dtype=torch.long)
            mod = torch.tensor([self.meta.modality_id(k[1]) for k in keys], dtype=torch.long)
            phase = torch.tensor([self.meta.phase_id(k[2]) for k in keys], dtype=torch.long)
            q_bucket = torch.tensor(
                [self.meta.quality_id(_quality_bucket_from_score(q)) for q in qs], dtype=torch.long
            )
            return {"view": view, "modality": mod, "phase": phase, "quality": q_bucket}

        ctx_keys = [elem_keys[i] for i in ctx]
        ctx_qs = [float(elem_q[i]) for i in ctx]
        tgt_keys = [elem_keys[i] for i in tgt]
        tgt_qs = [float(elem_q[i]) for i in tgt]

        ctx_meta = _meta_ids(ctx_keys, ctx_qs)
        tgt_meta = _meta_ids(tgt_keys, tgt_qs)

        return {
            "ctx_elements": torch.from_numpy(elem_vecs[ctx].astype(np.float32)) if ctx else torch.zeros(0, elem_vecs.shape[1]),
            "tgt_elements": torch.from_numpy(elem_vecs[tgt].astype(np.float32)) if tgt else torch.zeros(0, elem_vecs.shape[1]),
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

    def _empty_study_batch(self, sid: str, d: int) -> Dict[str, torch.Tensor]:
        return {
            "ctx_elements": torch.zeros(1, d),
            "tgt_elements": torch.zeros(0, d),
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


def _quality_bucket_from_score(q: float) -> str:
    if q != q:
        return "unknown"
    if q >= 0.75:
        return "high"
    if q >= 0.55:
        return "med"
    return "low"


# ---------------------------------------------------------------------------
# Collate: pad variable-length elements into rectangular tensors
# ---------------------------------------------------------------------------


def echoset_collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Pad per-study dicts into a padded batch."""
    B = len(batch)
    d_clip = batch[0]["ctx_elements"].shape[1] if batch[0]["ctx_elements"].numel() else 1024
    max_ctx = max(1, max(b["ctx_elements"].shape[0] for b in batch))
    max_tgt = max(1, max(b["tgt_elements"].shape[0] for b in batch))

    ctx_elements = torch.zeros(B, max_ctx, d_clip)
    tgt_elements = torch.zeros(B, max_tgt, d_clip)
    ctx_pad = torch.ones(B, max_ctx, dtype=torch.bool)    # True = pad
    tgt_pad = torch.ones(B, max_tgt, dtype=torch.bool)

    def _zeros_meta(B, M):
        return torch.zeros(B, M, dtype=torch.long)

    ctx_meta = {k: _zeros_meta(B, max_ctx) for k in ["view", "modality", "phase", "quality"]}
    tgt_meta = {k: _zeros_meta(B, max_tgt) for k in ["view", "modality", "phase", "quality"]}

    study_id_int = torch.zeros(B, dtype=torch.long)
    n_elements = torch.zeros(B, dtype=torch.long)
    strategies = []

    for i, b in enumerate(batch):
        mc = b["ctx_elements"].shape[0]
        mt = b["tgt_elements"].shape[0]
        if mc > 0:
            ctx_elements[i, :mc] = b["ctx_elements"]
            ctx_pad[i, :mc] = False
            ctx_meta["view"][i, :mc] = b["ctx_meta_view"]
            ctx_meta["modality"][i, :mc] = b["ctx_meta_modality"]
            ctx_meta["phase"][i, :mc] = b["ctx_meta_phase"]
            ctx_meta["quality"][i, :mc] = b["ctx_meta_quality"]
        if mt > 0:
            tgt_elements[i, :mt] = b["tgt_elements"]
            tgt_pad[i, :mt] = False
            tgt_meta["view"][i, :mt] = b["tgt_meta_view"]
            tgt_meta["modality"][i, :mt] = b["tgt_meta_modality"]
            tgt_meta["phase"][i, :mt] = b["tgt_meta_phase"]
            tgt_meta["quality"][i, :mt] = b["tgt_meta_quality"]
        study_id_int[i] = b["study_id_int"]
        n_elements[i] = b["n_elements"]
        strategies.append(b["mask_strategy"])

    return {
        "ctx_elements": ctx_elements,
        "tgt_elements": tgt_elements,
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


__all__ = ["EchoSetStudyDataset", "echoset_collate"]
