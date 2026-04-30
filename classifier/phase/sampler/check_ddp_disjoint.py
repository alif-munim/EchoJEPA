"""Simulate a 2-rank DDP setup on a single process.

Instantiates two PhaseMatchedStudySampler objects with identical
``seed`` and different ``rank``, calls ``set_epoch(0)`` on each, and
verifies that the emitted clip_b row indices (and the full
MatchRecord identities) are disjoint.

This is a logical disjointness check — we don't spin up real
``torch.distributed``. The per-rank slicing inside the sampler is the
only DDP-dependent path; the sampler's ``build_records`` does exactly
the same rank-slicing regardless of whether it learned ``rank`` /
``num_replicas`` from ``torch.distributed`` or from constructor args.

Usage:
    python check_ddp_disjoint.py \\
        --parquet phase_annotations/phase_annotations.parquet \\
        --world-size 2 --epoch 0
"""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# decord stub for import-time only; no video decoding here.
decord_stub = types.ModuleType("decord")
decord_stub.VideoReader = object
decord_stub.cpu = lambda *a, **kw: None
sys.modules.setdefault("decord", decord_stub)

from phase_matched_sampler import PhaseMatchedStudySampler  # noqa: E402


def _ids(records):
    """Convert records to a hashable identity per record (the study_id +
    per-clip dicom_ids are unique enough for disjointness testing)."""
    out = []
    for r in records:
        out.append((r.study_id, r.clip_a.dicom_id, r.clip_b.dicom_id,
                    r.clip_a.anchor_frame, r.clip_b.anchor_frame,
                    round(float(r.target_phi_a), 6)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", type=Path, required=True)
    ap.add_argument("--world-size", type=int, default=2)
    ap.add_argument("--epoch", type=int, default=0)
    ap.add_argument("--sampling-mode", default="uniform_phase")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    per_rank_records = []
    per_rank_row_idxs = []
    for rank in range(args.world_size):
        s = PhaseMatchedStudySampler(
            parquet_path=args.parquet,
            tiers=("high",), rr_filter_mode="strict",
            sampling_mode=args.sampling_mode, phase_tolerance=0.15,
            frames_per_clip=16, frame_step=1, pairs_per_study=1,
            seed=args.seed, num_replicas=args.world_size, rank=rank,
        )
        s.set_epoch(args.epoch)
        recs = s.build_records()
        per_rank_records.append(recs)
        per_rank_row_idxs.append([r.clip_b.row_idx for r in recs])
        print(f"rank={rank}: {len(recs)} records; first 3 row_idx={per_rank_row_idxs[-1][:3]}")

    # Disjointness: no MatchRecord identity shared across ranks.
    sets = [set(_ids(recs)) for recs in per_rank_records]
    overlap = set.intersection(*sets) if len(sets) > 1 else set()
    print(f"\nper-rank record id sets: sizes={[len(s) for s in sets]}")
    print(f"shared records across ranks: {len(overlap)}")
    if overlap:
        print(f"  sample overlap: {list(overlap)[:3]}")

    # Row-index disjointness at the DataLoader level.
    idx_sets = [set(idxs) for idxs in per_rank_row_idxs]
    idx_overlap = set.intersection(*idx_sets) if len(idx_sets) > 1 else set()
    print(f"per-rank row_idx sets: sizes={[len(s) for s in idx_sets]}; shared={len(idx_overlap)}")

    # Union equals a single-rank run's full record count (sans padding).
    # Build a rank=0 world=1 reference and compare lengths.
    ref = PhaseMatchedStudySampler(
        parquet_path=args.parquet,
        tiers=("high",), rr_filter_mode="strict",
        sampling_mode=args.sampling_mode, phase_tolerance=0.15,
        frames_per_clip=16, frame_step=1, pairs_per_study=1,
        seed=args.seed, num_replicas=1, rank=0,
    )
    ref.set_epoch(args.epoch)
    ref_recs = ref.build_records()
    ref_ids = set(_ids(ref_recs))
    union_ids = set.union(*sets)
    missing = ref_ids - union_ids
    extras = union_ids - ref_ids
    print(f"\nreference (world=1): {len(ref_recs)} records")
    print(f"  missing from union of ranks: {len(missing)}")
    print(f"  extra in union (from padding): {len(extras)}")

    status = "PASS" if (not overlap and not idx_overlap) else "FAIL"
    print(f"\nDDP disjointness: {status}")


if __name__ == "__main__":
    main()
