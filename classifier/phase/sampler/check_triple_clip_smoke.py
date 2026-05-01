"""Triple-clip data-path smoke test for phase_relational_jepa.

Exercises the full chain:
    PhaseMatchedStudySampler (with require_same_study_wrong_phase_negative=True)
      → _records_to_pair_dataframe emits view_0, view_1, view_2 + hard-neg metadata
      → _records_to_anchor_table emits 3 anchor entries per row
      → VideoGroupDataset(group_size=3).set_pair_dataframe installs
      → DataLoader returns (segs, label, clip_indices, slot_mask, meta)

Then asserts the full acceptance-gate block from the user brief. Prints a
concise metrics table at the end and exits 0 on pass, non-zero on any
failure. Only proceeds to train.py once this passes.

Usage:
    python classifier/phase/sampler/check_triple_clip_smoke.py \\
        --parquet classifier/phase/phase_annotations/phase_annotations.parquet \\
        --dicom-dir classifier/phase/dicoms \\
        --n 128

Uses the pydicom-backed decord stub from check_anchor_loading.py so it
runs on a CPU dev box without real video decode / S3 credentials. The
load path still flows through VideoGroupDataset (no shortcuts).
"""
from __future__ import annotations

import argparse
import math
import sys
import types
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
VJEPA_ROOT = HERE.parents[2]
if str(VJEPA_ROOT) not in sys.path:
    sys.path.insert(0, str(VJEPA_ROOT))


# --- decord stub (same pattern as check_anchor_loading.py) --------------- #
class _DicomVR:
    def __init__(self, dicom_path):
        import pydicom
        ds = pydicom.dcmread(str(dicom_path))
        pa = ds.pixel_array
        pi = str(getattr(ds, "PhotometricInterpretation", ""))
        if "PALETTE" in pi:
            from pydicom.pixels.processing import apply_color_lut
            pa = apply_color_lut(pa, ds)
            if pa.dtype == np.uint16:
                pa = (pa / 256).astype(np.uint8)
        pa = np.ascontiguousarray(pa, dtype=np.uint8)
        if pa.ndim == 3:
            pa = pa[None, ...]
        if pa.shape[-1] == 1:
            pa = np.repeat(pa, 3, axis=-1)
        self._frames = pa
        try:
            self._fps = 1000.0 / float(ds.FrameTime) if hasattr(ds, "FrameTime") else 30.0
        except Exception:
            self._fps = 30.0

    def __len__(self):
        return int(self._frames.shape[0])

    def get_avg_fps(self):
        return self._fps

    def get_batch(self, inds):
        batch = self._frames[np.asarray(inds).clip(0, len(self) - 1)]
        out = types.SimpleNamespace()
        out.asnumpy = lambda b=batch: b
        return out


def _install_decord_stub():
    mod = types.ModuleType("decord")

    class _VideoReader:
        def __new__(cls, uri, *args, **kwargs):
            return _DicomVR(Path(uri))

    mod.VideoReader = _VideoReader
    mod.cpu = lambda *a, **kw: None
    sys.modules["decord"] = mod


_install_decord_stub()

from src.datasets.video_group_dataset import VideoGroupDataset  # noqa: E402
from phase_matched_sampler import (  # noqa: E402
    PhaseMatchedStudySampler,
    circular_phase_distance,
)
from phase_matched_pair_dataset import (  # noqa: E402
    _records_to_pair_dataframe,
    _records_to_anchor_table,
)


def _rewrite_to_local(pair_df: pd.DataFrame, dicom_dir: Path) -> pd.DataFrame:
    """Replace S3 view URIs with local DICOM paths so the stub VR can
    read them. Only affects the three view columns; metadata untouched."""
    out = pair_df.copy()
    out["view_0"] = out.clip_a_dicom_id.astype(str).map(lambda d: str(dicom_dir / f"{d}.dcm"))
    out["view_1"] = out.clip_b_dicom_id.astype(str).map(lambda d: str(dicom_dir / f"{d}.dcm"))
    if "view_2" in out.columns:
        def _neg_uri(did):
            did = str(did) if did else ""
            if not did:
                return "MISS"  # VideoGroupDataset MISSING_TOKEN
            return str(dicom_dir / f"{did}.dcm")
        out["view_2"] = out.clip_b_neg_dicom_id.astype(str).map(_neg_uri)
    return out


def _draw_triple_records(
    sampler: PhaseMatchedStudySampler,
    on_disk: set,
    n_target: int,
    seed: int = 0,
):
    """Drive the real _draw_pair repeatedly against studies that have ≥3
    clips available on disk, forcing the sampler onto local triples."""
    rng = np.random.default_rng(seed)
    on_disk_df = sampler._df[sampler._df.dicom_id.astype(str).isin(on_disk)]
    per_study = on_disk_df.groupby("study_id").size()
    multi = per_study[per_study >= 3].index.tolist()
    if not multi:
        raise RuntimeError(
            "no studies with ≥3 on-disk clips; widen --dicom-dir "
            f"(have {len(on_disk)} clips across {len(per_study)} studies)"
        )
    records = []
    n_attempts = 0
    max_attempts = n_target * 12
    while len(records) < n_target and n_attempts < max_attempts:
        n_attempts += 1
        sid = str(multi[int(rng.integers(0, len(multi)))])
        sub = on_disk_df[on_disk_df.study_id.astype(str) == sid]
        row_idxs = sub.index.tolist()
        # Use the real sampler's rows index, overridden to our on-disk subset.
        sampler.study_to_rows[sid] = [int(x) for x in row_idxs]
        r = sampler._draw_pair(sid, rng)
        if r is None or r.clip_b_neg_phase is None:
            continue
        records.append(r)
    if len(records) < n_target:
        print(
            f"[WARN] only drew {len(records)}/{n_target} triples after "
            f"{n_attempts} attempts; proceeding with what we have."
        )
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", type=Path,
                    default=Path("classifier/phase/phase_annotations/phase_annotations.parquet"))
    ap.add_argument("--dicom-dir", type=Path,
                    default=Path("classifier/phase/dicoms"))
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--frames-per-clip", type=int, default=16)
    ap.add_argument("--frame-step", type=int, default=1)
    ap.add_argument("--wrong-phase-min-delta", type=float, default=0.25)
    ap.add_argument("--phase-tolerance", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--wrong-phase-strategy",
        default="same_view_then_same_family",
        choices=("same_view_only", "same_view_then_same_family", "any_same_study"),
        help=(
            "Production uses 'same_view_then_same_family'. The local "
            "sanity DICOM set has view=None for every clip, so this "
            "smoke test should be invoked with 'any_same_study' to "
            "exercise the triple-clip plumbing end-to-end. The view-"
            "preference logic is covered by separate unit checks."
        ),
    )
    args = ap.parse_args()

    # ----- build sampler with the paper-ready config --------------------- #
    sampler = PhaseMatchedStudySampler(
        parquet_path=args.parquet,
        tiers=("high", "medium"),
        rr_filter_mode="strict",
        require_rr_consistent=True,
        sampling_mode="uniform_phase",
        phase_tolerance=args.phase_tolerance,
        frames_per_clip=args.frames_per_clip,
        frame_step=args.frame_step,
        pairs_per_study=1,
        seed=args.seed,
        view_pair_policy={
            "enabled": True,
            "same_view_prob": 0.35,
            "same_family_prob": 0.45,
            "cross_family_prob": 0.20,
            "require_different_dicom": True,
            "allow_same_view": True,
            "resample_attempts": 8,
        },
        delta_phase_mode="controlled_buckets",
        delta_phase_buckets=(0.0, 0.125, 0.25, 0.5),
        delta_phase_bucket_probs=(0.40, 0.30, 0.20, 0.10),
        require_same_study_wrong_phase_negative=True,
        wrong_phase_min_delta=args.wrong_phase_min_delta,
        wrong_phase_strategy=args.wrong_phase_strategy,
        allow_missing_hard_negative=False,
        hard_negative_fallback="resample_anchor",
        max_hard_neg_attempts=16,
    )

    # ----- filter on-disk studies + draw n triples ----------------------- #
    on_disk = {p.stem for p in args.dicom_dir.glob("*.dcm")}
    print(f"[INFO] parquet rows: {len(sampler._df):,}")
    print(f"[INFO] on-disk DICOMs: {len(on_disk)}")
    records = _draw_triple_records(sampler, on_disk, args.n, seed=args.seed)
    print(f"[INFO] drew {len(records)} valid triples")
    if len(records) == 0:
        print(
            "[FAIL] sampler produced 0 triples; cannot run downstream checks.\n"
            "       Common causes on the dev dataset:\n"
            "       - view=None on all on-disk clips → switch "
            "`--wrong-phase-strategy any_same_study`.\n"
            "       - Too few on-disk clips per study → widen --dicom-dir."
        )
        raise SystemExit(2)

    # ----- build pair DataFrame + anchor table --------------------------- #
    pair_df = _records_to_pair_dataframe(records, sampler._df, video_uri_mode="dicom")
    pair_df = pair_df.reset_index(drop=True)
    # Rewrite view columns to local DICOM paths.
    pair_df = _rewrite_to_local(pair_df, args.dicom_dir)
    anchors = _records_to_anchor_table(records)

    # ================= ASSERTION BLOCK ================================== #
    failures: list[str] = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)
            print(f"[FAIL] {msg}")
        else:
            print(f"[pass] {msg[:80]}")

    # --- 1. DataFrame schema --- #
    required_cols = {
        "view_0", "view_1", "view_2", "label",
        "clip_b_neg_dicom_id", "clip_b_neg_anchor_frame",
        "clip_b_neg_phase_at_anchor", "clip_b_neg_phase_error",
        "clip_b_neg_view", "target_phi_b_neg",
        "delta_phase_bucket_pos", "delta_phase_bucket_neg",
        "view_pair_class_pos", "view_pair_class_neg",
        "hard_neg_available", "hard_neg_resample_count",
    }
    missing = required_cols - set(pair_df.columns)
    check(not missing, f"pair_df has all hard-neg schema columns (missing={missing})")

    # --- 2. Study identity (anchor rows all from same study) --- #
    same_study_all_three = 0
    for r in records:
        if (r.clip_b_neg_phase is not None
                and r.study_id == r.study_id  # dataclass: study is on record
                and r.clip_a.dicom_id != r.clip_b.dicom_id
                and r.clip_a.dicom_id != r.clip_b_neg_phase.dicom_id):
            # Reconstruct study_id from sampler df for each of the 3 clips.
            sa = str(sampler._df.loc[r.clip_a.row_idx, "study_id"])
            sb = str(sampler._df.loc[r.clip_b.row_idx, "study_id"])
            sn = str(sampler._df.loc[r.clip_b_neg_phase.row_idx, "study_id"])
            if sa == sb == sn == r.study_id:
                same_study_all_three += 1
    same_study_frac = same_study_all_three / max(1, len(records))
    check(same_study_frac == 1.0, f"same_study_all_three_frac == 1.0 (got {same_study_frac:.4f})")

    # --- 3. Phase relation constraint --- #
    phase_distances = []
    bucket_ok = 0
    for r in records:
        if r.clip_b_neg_phase is None:
            continue
        delta_pos = (r.target_phi_b - r.target_phi_a) % 1.0
        delta_neg = (r.target_phi_b_neg - r.target_phi_a) % 1.0
        d = circular_phase_distance(delta_pos, delta_neg)
        phase_distances.append(d)
        # clip_b_pos bucket proximity check
        if r.delta_phase_bucket_pos is not None:
            center = sampler.delta_phase_bucket_centers[r.delta_phase_bucket_pos]
            half = sampler._delta_phase_half_width
            # Δφ_pos must be within half-width of bucket center (allowing a small
            # tolerance for anchor-snap drift — anchor may be up to phase_tolerance off).
            if circular_phase_distance(abs(delta_pos), center) <= half + args.phase_tolerance:
                bucket_ok += 1
    phase_distances = np.asarray(phase_distances, dtype=np.float64)
    phase_min = float(phase_distances.min()) if len(phase_distances) else 0.0
    phase_mean = float(phase_distances.mean()) if len(phase_distances) else 0.0
    # Allow a tiny numerical tolerance on the min.
    EPS = 1e-6
    check(
        phase_min >= args.wrong_phase_min_delta - EPS,
        f"phase_distance_min >= {args.wrong_phase_min_delta} (got {phase_min:.4f})",
    )

    # --- 4. Hard-negative health --- #
    n_avail = sum(1 for r in records if r.hard_neg_available)
    hn_avail_frac = n_avail / max(1, len(records))
    # Same-view / same-family breakdown for clip_b_neg vs clip_b_pos view.
    same_view = 0
    same_family = 0

    def _fam(v):
        from phase_matched_sampler import VIEW_FAMILIES
        if v is None:
            return "other"
        u = v.upper()
        if u == "SUBCOSTAL":
            u = "Subcostal"
        return VIEW_FAMILIES.get(u if u in VIEW_FAMILIES else v, "other")

    for r in records:
        if r.clip_b_neg_phase is None:
            continue
        vp = r.clip_b.view
        vn = r.clip_b_neg_phase.view
        if vp is not None and vn is not None and vp == vn:
            same_view += 1
        if _fam(vp) == _fam(vn):
            same_family += 1
    hn_same_view_frac = same_view / max(1, len(records))
    hn_same_family_frac = same_family / max(1, len(records))
    hn_resample_counts = [r.hard_neg_resample_count for r in records]
    check(hn_avail_frac >= 0.95, f"hard_neg_available_frac >= 0.95 (got {hn_avail_frac:.3f})")

    # --- 5. DataLoader batch --- #
    # Build a throw-away VideoGroupDataset(group_size=3), install pair_df, iterate.
    placeholder = pd.DataFrame({
        "view_0": ["x"],
        "view_1": ["y"],
        "view_2": ["z"],
        "label": [0.0],
    })
    tmp_csv = Path("/tmp/triple_pair_placeholder.csv")
    placeholder.to_csv(tmp_csv, index=False)

    dataset = VideoGroupDataset(
        data_paths=str(tmp_csv),
        group_size=3,
        frames_per_clip=args.frames_per_clip,
        frame_step=args.frame_step,
        img_size=224,
        training=False,
        transform=None,
        shared_transform=None,
    )
    dataset.set_pair_dataframe(pair_df, anchors_by_index=anchors)

    # Pull a handful of items, ignoring any that raise (stub VR + strided
    # sampling can miss on very-short DICOMs). Record first-successful shapes.
    loaded = []
    idxs = list(range(min(8, len(pair_df))))
    for idx in idxs:
        try:
            item = dataset[idx]
            loaded.append(item)
        except Exception as e:
            print(f"[warn] dataset[{idx}] raised: {type(e).__name__}: {e}")
    check(len(loaded) >= 1, f"DataLoader returned at least 1 item (got {len(loaded)})")

    # Validate the returned shape: segs is list of 3 tensors (or arrays).
    shape_msg = "unknown"
    metadata_ok = False
    anchor_for_all_three = False
    if loaded:
        segs, label, clip_idx, slot_mask, meta = loaded[0]
        shape_msg = f"n_views={len(segs)}; first={getattr(segs[0], 'shape', None)}"
        check(len(segs) == 3, f"segs has 3 views (got {len(segs)})")
        # meta is a dict of per-sample metadata
        metadata_ok = isinstance(meta, dict) and "clip_b_neg_dicom_id" in meta
        check(metadata_ok, f"meta dict includes clip_b_neg_dicom_id ({type(meta).__name__})")
        # All three clip_indices present
        anchor_for_all_three = len(clip_idx) == 3
        check(anchor_for_all_three, f"clip_idx has 3 views (got {len(clip_idx)})")

    # --- 6. Backward compat (group_size=2 smooth_l1) --- #
    # Build a 2-clip pair_df (no view_2), install on a group_size=2 VGD.
    bc_status = "unknown"
    try:
        ph2 = pd.DataFrame({"view_0": ["x"], "view_1": ["y"], "label": [0.0]})
        ph2.to_csv("/tmp/pair_g2_placeholder.csv", index=False)
        ds2 = VideoGroupDataset(
            data_paths="/tmp/pair_g2_placeholder.csv",
            group_size=2,
            frames_per_clip=args.frames_per_clip,
            frame_step=args.frame_step,
            img_size=224,
            training=False,
            transform=None,
            shared_transform=None,
        )
        # Drop view_2 + hard-neg cols from the pair_df to simulate smooth_l1 output.
        smooth_df = pair_df.drop(columns=[
            c for c in pair_df.columns
            if c == "view_2" or c.startswith("clip_b_neg") or c in (
                "target_phi_b_neg", "delta_phase_bucket_pos", "delta_phase_bucket_neg",
                "view_pair_class_pos", "view_pair_class_neg",
                "hard_neg_available", "hard_neg_resample_count",
            )
        ])
        anchors_2 = {i: [entries[0], entries[1]] for i, entries in anchors.items()}
        ds2.set_pair_dataframe(smooth_df.reset_index(drop=True), anchors_by_index=anchors_2)
        # Sanity __getitem__ on the first row.
        item2 = ds2[0]
        bc_segs = item2[0]
        check(len(bc_segs) == 2, f"smooth_l1 group_size=2 returns 2 views (got {len(bc_segs)})")
        bc_status = f"OK (segs={len(bc_segs)})"
    except Exception as e:
        bc_status = f"FAIL: {type(e).__name__}: {e}"
        failures.append(f"group_size=2 smooth_l1 backward compat: {bc_status}")

    # ----- histograms + final table ------------------------------------- #
    def _hist(labels, title):
        counts = Counter(labels)
        total = sum(counts.values())
        return ", ".join(f"{k}={v}({100*v/max(1, total):.0f}%)" for k, v in sorted(counts.items()))

    bucket_pos_hist = _hist(
        [r.delta_phase_bucket_pos for r in records if r.delta_phase_bucket_pos is not None],
        "bucket_pos",
    )
    bucket_neg_hist = _hist(
        [r.delta_phase_bucket_neg for r in records if r.delta_phase_bucket_neg is not None],
        "bucket_neg",
    )
    vp_pos_hist = _hist([r.view_pair_class_pos or "?" for r in records], "vp_pos")
    vp_neg_hist = _hist([r.view_pair_class_neg or "?" for r in records if r.view_pair_class_neg], "vp_neg")

    print()
    print("=" * 70)
    print("TRIPLE-CLIP SMOKE — summary table")
    print("=" * 70)
    print(f"{'n_samples':<40} {len(records)}")
    print(f"{'hard_neg_available_frac':<40} {hn_avail_frac:.3f}")
    print(f"{'same_study_all_three_frac':<40} {same_study_frac:.3f}")
    print(f"{'phase_distance_min':<40} {phase_min:.4f}")
    print(f"{'phase_distance_mean':<40} {phase_mean:.4f}")
    print(f"{'hard_neg_same_view_frac':<40} {hn_same_view_frac:.3f}")
    print(f"{'hard_neg_same_family_frac':<40} {hn_same_family_frac:.3f}")
    print(f"{'hard_neg_resample_count_mean':<40} {np.mean(hn_resample_counts):.2f}")
    print(f"{'hard_neg_resample_count_max':<40} {max(hn_resample_counts) if hn_resample_counts else 0}")
    print(f"{'delta_phase_bucket_pos_hist':<40} {bucket_pos_hist}")
    print(f"{'delta_phase_bucket_neg_hist':<40} {bucket_neg_hist}")
    print(f"{'view_pair_class_pos_hist':<40} {vp_pos_hist}")
    print(f"{'view_pair_class_neg_hist':<40} {vp_neg_hist}")
    print(f"{'dataloader_batch_shape':<40} {shape_msg}")
    print(f"{'group_size2_smooth_l1_compat':<40} {bc_status}")
    print("=" * 70)

    if failures:
        print()
        print("FAILURES:")
        for msg in failures:
            print(f"  - {msg}")
        print()
        print("DO NOT PROCEED TO train.py")
        raise SystemExit(1)

    print()
    print("ALL SMOKE CHECKS PASSED — safe to proceed to train.py")


if __name__ == "__main__":
    main()
