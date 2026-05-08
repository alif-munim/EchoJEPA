"""Dataset for EchoSet-JEPA Stage-2.

Yields, per study, a bag of study elements with meta ids and a mask assignment.
All cross-method fairness is enforced by reading a shared K-matched manifest
produced by ``experiments/echoset_jepa/sample_K.py``.

This module is intentionally read-oriented: it assumes cached ``c_clip``
vectors already live in S3 (or mirrored to local disk) and that the element
manifest + K-sample manifest have been built offline. It does **not** invoke
the V-JEPA clip encoder at train time.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.models.meta_embeddings import MetaEmbeddings


MaskStrategyWeights = Dict[str, float]

DEFAULT_MASK_STRATEGY_WEIGHTS: MaskStrategyWeights = {
    "random_element": 0.50,
    "whole_view_family": 0.25,
    "whole_modality": 0.15,
    "apical_holdout": 0.033,
    "doppler_holdout": 0.033,
    "bmode_holdout": 0.034,
}


@dataclass
class StudyRecord:
    study_id: str
    patient_id: str
    clip_keys: List[Tuple[str, str, str]]  # (view_family, modality, phase_bucket) per clip
    clip_embeddings: np.ndarray             # (K, d_clip) — the K sampled clips' cached c_clip
    clip_quality: np.ndarray                # (K,) — clip quality scores


def group_into_elements(
    clip_keys: Sequence[Tuple[str, str, str]],
    clip_embeddings: np.ndarray,
    clip_quality: np.ndarray,
    element_agg: str = "mean",
    tau_quality: float = 0.5,
) -> Tuple[List[Tuple[str, str, str]], np.ndarray, np.ndarray]:
    """Aggregate K clips into M_elements by ``(view_family, modality, phase_bucket)``.

    Returns ``(element_keys, element_embeddings [M, d_clip], element_quality [M])``.

    Element quality is the **mean** quality score of member clips regardless of
    ``element_agg``.
    """
    buckets: Dict[Tuple[str, str, str], List[int]] = {}
    for idx, key in enumerate(clip_keys):
        buckets.setdefault(key, []).append(idx)

    element_keys: List[Tuple[str, str, str]] = []
    element_vecs: List[np.ndarray] = []
    element_q: List[float] = []
    for key, idxs in buckets.items():
        vecs = clip_embeddings[idxs]
        qs = clip_quality[idxs]
        if element_agg == "mean":
            agg = vecs.mean(axis=0)
        elif element_agg == "quality_weighted":
            logits = qs / tau_quality
            w = np.exp(logits - logits.max())
            w = w / w.sum()
            agg = (vecs * w[:, None]).sum(axis=0)
        else:
            raise ValueError(f"unknown element_agg={element_agg!r}")
        element_keys.append(key)
        element_vecs.append(agg)
        element_q.append(float(qs.mean()))

    return element_keys, np.stack(element_vecs, axis=0), np.asarray(element_q, dtype=np.float32)


def pick_mask_indices(
    M: int,
    element_keys: Sequence[Tuple[str, str, str]],
    strategy_weights: MaskStrategyWeights = None,
    rng: Optional[random.Random] = None,
    mask_ratio_cap: float = 0.6,
    min_context: int = 1,
    min_target: int = 1,
) -> Tuple[List[int], List[int], str]:
    """Partition ``range(M)`` into context and target indices per a sampled strategy.

    Guarantees ``len(context) >= min_context`` and ``len(target) >= min_target``
    and ``len(target) / M <= mask_ratio_cap``.

    Returns ``(context_idx, target_idx, strategy_name)``.
    """
    rng = rng or random
    weights = dict(strategy_weights or DEFAULT_MASK_STRATEGY_WEIGHTS)

    def _apical(i: int) -> bool:
        return element_keys[i][0] == "apical"

    def _doppler(i: int) -> bool:
        return element_keys[i][1] in {"color_doppler", "pw_doppler", "cw_doppler"}

    def _bmode(i: int) -> bool:
        return element_keys[i][1] == "b_mode"

    # Disqualify stratified strategies whose filter would produce empty / all targets
    def _valid_stratified(filter_fn) -> bool:
        tgt = [i for i in range(M) if filter_fn(i)]
        if not tgt:
            return False
        if M - len(tgt) < min_context:
            return False
        if len(tgt) / max(M, 1) > mask_ratio_cap:
            return False
        return True

    if not _valid_stratified(_apical):
        weights.pop("apical_holdout", None)
    if not _valid_stratified(_doppler):
        weights.pop("doppler_holdout", None)
    if not _valid_stratified(_bmode):
        weights.pop("bmode_holdout", None)

    # whole_view_family / whole_modality: drop if only one distinct key
    views = {k[0] for k in element_keys}
    modalities = {k[1] for k in element_keys}
    if len(views) < 2:
        weights.pop("whole_view_family", None)
    if len(modalities) < 2:
        weights.pop("whole_modality", None)

    # Fallback: ensure at least random_element remains
    if not weights:
        weights = {"random_element": 1.0}

    names = list(weights.keys())
    probs = np.asarray([weights[n] for n in names], dtype=np.float64)
    probs = probs / probs.sum()
    strategy = rng.choices(names, weights=probs.tolist(), k=1)[0]

    if strategy == "apical_holdout":
        target = [i for i in range(M) if _apical(i)]
    elif strategy == "doppler_holdout":
        target = [i for i in range(M) if _doppler(i)]
    elif strategy == "bmode_holdout":
        target = [i for i in range(M) if _bmode(i)]
    elif strategy == "whole_view_family":
        chosen = rng.choice(sorted(views))
        target = [i for i in range(M) if element_keys[i][0] == chosen]
    elif strategy == "whole_modality":
        chosen = rng.choice(sorted(modalities))
        target = [i for i in range(M) if element_keys[i][1] == chosen]
    else:  # random_element
        n_tgt = max(min_target, min(int(0.4 * M), int(mask_ratio_cap * M)))
        n_tgt = min(n_tgt, M - min_context)
        target = rng.sample(range(M), n_tgt)

    # Enforce invariants (clamp)
    if len(target) >= M - min_context + 1:
        target = rng.sample(range(M), max(min_target, M - min_context))
    if len(target) < min_target:
        target = rng.sample(range(M), min_target)

    target_set = set(target)
    context = [i for i in range(M) if i not in target_set]
    return context, sorted(target_set), strategy


class EchoSetJEPADataset(Dataset):
    """A list of :class:`StudyRecord` with on-the-fly element grouping and masking.

    The dataset delegates the expensive V-JEPA forward pass to an offline cache;
    `__getitem__` only does numpy + small tensor ops.
    """

    def __init__(
        self,
        records: Sequence[StudyRecord],
        meta: MetaEmbeddings,
        element_agg: str = "mean",
        strategy_weights: Optional[MaskStrategyWeights] = None,
        seed: int = 0,
        permute_every_step: bool = True,
    ) -> None:
        self.records = list(records)
        self.meta = meta
        self.element_agg = element_agg
        self.strategy_weights = strategy_weights or DEFAULT_MASK_STRATEGY_WEIGHTS
        self._rng = random.Random(seed)
        self.permute_every_step = permute_every_step

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.records[idx]
        elem_keys, elem_vecs, elem_q = group_into_elements(
            rec.clip_keys, rec.clip_embeddings, rec.clip_quality, element_agg=self.element_agg
        )
        M = len(elem_keys)
        if self.permute_every_step and M > 1:
            perm = list(range(M))
            self._rng.shuffle(perm)
            elem_keys = [elem_keys[i] for i in perm]
            elem_vecs = elem_vecs[perm]
            elem_q = elem_q[perm]

        ctx, tgt, strategy = pick_mask_indices(M, elem_keys, self.strategy_weights, rng=self._rng)

        def _meta_ids(keys: Sequence[Tuple[str, str, str]], qs: Sequence[float]) -> Dict[str, torch.Tensor]:
            view = torch.tensor([self.meta.view_id(k[0]) for k in keys], dtype=torch.long)
            mod = torch.tensor([self.meta.modality_id(k[1]) for k in keys], dtype=torch.long)
            phase = torch.tensor([self.meta.phase_id(k[2]) for k in keys], dtype=torch.long)
            q_bucket = torch.tensor(
                [self.meta.quality_id(_quality_bucket(q)) for q in qs], dtype=torch.long
            )
            return {"view": view, "modality": mod, "phase": phase, "quality": q_bucket}

        ctx_keys = [elem_keys[i] for i in ctx]
        ctx_qs = [float(elem_q[i]) for i in ctx]
        tgt_keys = [elem_keys[i] for i in tgt]
        tgt_qs = [float(elem_q[i]) for i in tgt]

        ctx_meta = _meta_ids(ctx_keys, ctx_qs)
        tgt_meta = _meta_ids(tgt_keys, tgt_qs)

        return {
            "ctx_elements": torch.from_numpy(elem_vecs[ctx].astype(np.float32)),
            "tgt_elements": torch.from_numpy(elem_vecs[tgt].astype(np.float32)),
            "ctx_meta_view": ctx_meta["view"],
            "ctx_meta_modality": ctx_meta["modality"],
            "ctx_meta_phase": ctx_meta["phase"],
            "ctx_meta_quality": ctx_meta["quality"],
            "tgt_meta_view": tgt_meta["view"],
            "tgt_meta_modality": tgt_meta["modality"],
            "tgt_meta_phase": tgt_meta["phase"],
            "tgt_meta_quality": tgt_meta["quality"],
            "mask_strategy": strategy,
            "study_id": rec.study_id,
        }


def _quality_bucket(q: float) -> str:
    if q >= 0.66:
        return "high"
    if q >= 0.33:
        return "med"
    return "low"


__all__ = [
    "StudyRecord",
    "EchoSetJEPADataset",
    "group_into_elements",
    "pick_mask_indices",
    "DEFAULT_MASK_STRATEGY_WEIGHTS",
]
