"""Control C — Shuffled-study pretraining (plan §7).

Same EchoSet-JEPA architecture and loss, but **context elements come from a
different random study** (matched on M_elements and view-family mix). Target
element still comes from the true study.

The only thing this module needs to change from the main training loop is the
batch construction: swap each row's context with a size-and-mix-matched
shuffled-study context. Everything else — model, loss, optimizer, EMA,
diagnostics — is identical.

Implementation shape:
  - Load K-manifest + element manifest.
  - Precompute per-(M_elements, view_family_mix_bucket) index of studies.
  - At dataloader time, for each row, sample a matched-bucket donor study and
    replace context elements with that donor's elements (keeping donor's meta).
  - Keep the target untouched.

This is a scaffold; the matcher (bucket-key construction + donor sampling)
lives here so main ``app/echoset_jepa/train.py`` stays clean.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


def view_family_mix_bucket(view_families: List[str]) -> Tuple[str, ...]:
    """Build a coarse bucket key from the context element view families.

    Canonicalized as a sorted tuple so ``("apical","b_mode")`` and
    ``("b_mode","apical")`` hash to the same bucket.
    """
    return tuple(sorted(set(view_families)))


def build_donor_index(
    studies: Dict[str, List[Tuple[int, List[str]]]],
) -> Dict[Tuple[int, Tuple[str, ...]], List[str]]:
    """Build a ``(M_elements, view_family_bucket) -> [study_id]`` index.

    ``studies`` maps ``study_id -> [(M_elements, [view_family_per_element])]``.
    """
    idx: Dict[Tuple[int, Tuple[str, ...]], List[str]] = {}
    for sid, rows in studies.items():
        for M, vfs in rows:
            key = (M, view_family_mix_bucket(vfs))
            idx.setdefault(key, []).append(sid)
    return idx


def sample_shuffled_donor(
    target_study_id: str,
    M_elements: int,
    view_families: List[str],
    donor_index: Dict[Tuple[int, Tuple[str, ...]], List[str]],
    rng: random.Random,
) -> str:
    """Pick a donor study that matches (M_elements, view_family_mix) and ≠ target."""
    key = (M_elements, view_family_mix_bucket(view_families))
    pool = [s for s in donor_index.get(key, ()) if s != target_study_id]
    if not pool:
        # Fallback: same M_elements, ignore view mix
        pool = [
            s
            for (M, _), sids in donor_index.items()
            if M == M_elements
            for s in sids
            if s != target_study_id
        ]
    if not pool:
        raise RuntimeError(f"no donor available for M={M_elements} and key={key}")
    return rng.choice(pool)


__all__ = ["view_family_mix_bucket", "build_donor_index", "sample_shuffled_donor"]
