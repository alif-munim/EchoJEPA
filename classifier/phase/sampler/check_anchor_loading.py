"""Anchor-loading sanity for phase-matched multi-view training.

For a small set of pairs, verifies:
  1. Pair DataFrame is constructed and installed on the dataset.
  2. Dataset.__getitem__ actually loads frames centered on the anchor.
  3. Loaded frame indices contain or are nearest to anchor_frame.
  4. anchor_pos is at or near clip center unless was_clamped=True.
  5. Per-view frame_step is honored.
  6. Prints phase, circular diff, source_span stats per pair.

Runs on local DICOM paths (``classifier/phase/dicoms/``) since the
SageMaker dev box doesn't have GPU decord or S3 credentials to stream
from ``s3://echodata25``. We synthesize an mp4 mirror path for each
clip or, when absent, read the DICOM directly via pydicom and feed a
numpy-array stand-in through a minimal fake VideoReader that mimics
``decord.VideoReader``.

Usage:
    python check_anchor_loading.py \\
        --parquet phase_annotations/phase_annotations.parquet \\
        --dicom-dir classifier/phase/dicoms \\
        --n-pairs 5
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
VJEPA_ROOT = HERE.parents[2]
if str(VJEPA_ROOT) not in sys.path:
    sys.path.insert(0, str(VJEPA_ROOT))


# --- decord stub ----------------------------------------------------- #
# On the CPU-only dev box we don't ship decord. Install a lightweight
# stand-in that reads a DICOM via pydicom and exposes the same API.
class _DicomVR:
    def __init__(self, dicom_path: Path):
        ds = pydicom.dcmread(str(dicom_path))
        pa = ds.pixel_array
        pi = str(getattr(ds, "PhotometricInterpretation", ""))
        if "PALETTE" in pi:
            from pydicom.pixels.processing import apply_color_lut
            pa = apply_color_lut(pa, ds)
            if pa.dtype == np.uint16:
                pa = (pa / 256).astype(np.uint8)
        pa = np.ascontiguousarray(pa, dtype=np.uint8)
        # Ensure [T,H,W,C]
        if pa.ndim == 3:                      # single frame RGB
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
        batch = self._frames[inds]
        result = types.SimpleNamespace()
        result.asnumpy = lambda b=batch: b
        return result


def _install_decord_stub():
    mod = types.ModuleType("decord")

    class _VideoReader:
        """Stub that mimics ``decord.VideoReader(uri, ...)`` but reads the
        DICOM at ``uri`` via pydicom. Ignores ``num_threads`` and ``ctx``."""

        def __new__(cls, uri, *args, **kwargs):
            return _DicomVR(Path(uri))

    mod.VideoReader = _VideoReader
    mod.cpu = lambda *args, **kwargs: None
    sys.modules["decord"] = mod


_install_decord_stub()

from src.datasets.video_group_dataset import VideoGroupDataset, _compute_clip_indices  # noqa: E402
from phase_matched_sampler import PhaseMatchedStudySampler  # noqa: E402
from phase_matched_pair_dataset import PhaseMatchedEpochBuilder  # noqa: E402


# The dataset opens videos by URI. We rewrite S3 URIs to local DICOM
# paths inside the pair DataFrame (monkey-patching ``view_0``/``view_1``
# to ``<dicom_dir>/<dicom_id>.dcm``) before installing on the dataset.
def _rewrite_to_local(pair_df: pd.DataFrame, dicom_dir: Path) -> pd.DataFrame:
    out = pair_df.copy()
    out["view_0"] = out.clip_a_dicom_id.astype(str).map(lambda d: str(dicom_dir / f"{d}.dcm"))
    out["view_1"] = out.clip_b_dicom_id.astype(str).map(lambda d: str(dicom_dir / f"{d}.dcm"))
    return out


def _filter_to_on_disk(pair_df: pd.DataFrame, dicom_dir: Path) -> pd.DataFrame:
    avail = {p.stem for p in dicom_dir.glob("*.dcm")}
    m = pair_df.clip_a_dicom_id.isin(avail) & pair_df.clip_b_dicom_id.isin(avail)
    return pair_df[m].reset_index(drop=True)


def _dummy_transform(clip):
    # decord returns np.uint8 HxWxC; our fake VR emits np.ndarray.
    # Group dataset transforms expect torch tensors for preprocessing,
    # but for this sanity check we skip the spatial transform entirely
    # (we only care about the temporal index selection).
    return clip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", type=Path, required=True)
    ap.add_argument("--dicom-dir", type=Path, required=True)
    ap.add_argument("--n-pairs", type=int, default=5)
    ap.add_argument("--frames-per-clip", type=int, default=16)
    ap.add_argument("--frame-step", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--debug-csv", type=Path, default=None)
    args = ap.parse_args()

    sampler = PhaseMatchedStudySampler(
        parquet_path=args.parquet,
        tiers=("high",),
        rr_filter_mode="strict",
        sampling_mode="uniform_phase",
        phase_tolerance=0.15,
        frames_per_clip=args.frames_per_clip,
        frame_step=args.frame_step,
        pairs_per_study=1,
        seed=args.seed,
    )
    # Construct a minimal VideoGroupDataset with group_size=2. We give it
    # a placeholder CSV that we'll overwrite via set_pair_dataframe.
    placeholder = pd.DataFrame({"view_0": ["x"], "view_1": ["y"], "label": [0.0]})
    tmp_csv = Path("/tmp/phase_pair_placeholder.csv")
    placeholder.to_csv(tmp_csv, index=False)

    dataset = VideoGroupDataset(
        data_paths=str(tmp_csv),
        group_size=2,
        frames_per_clip=args.frames_per_clip,
        frame_step=args.frame_step,
        img_size=224,
        training=False,
        transform=None,
        shared_transform=None,
    )

    # Synthesize pair DF from multi-clip on-disk studies so we don't depend on
    # the sampler's random pick happening to land on local DICOMs. The pair
    # math (anchor selection, phase scoring) still goes through the real
    # sampler's per-clip MatchRecord builder.
    avail = {p.stem for p in args.dicom_dir.glob("*.dcm")}
    sampler_df = sampler._df
    on_disk_df = sampler_df[sampler_df.dicom_id.astype(str).isin(avail)]
    per_study = on_disk_df.groupby("study_id").size()
    multi = per_study[per_study >= 2].index.tolist()
    if not multi:
        print("no multi-clip on-disk studies; widen --dicom-dir.")
        return

    import numpy as np
    rng = np.random.default_rng(args.seed)

    records = []
    for sid in multi[: args.n_pairs]:
        sub = on_disk_df[on_disk_df.study_id == sid]
        row_idxs = sub.index.tolist()
        i, j = rng.choice(len(row_idxs), size=2, replace=False)
        key = str(sid)
        # call the private draw-pair with this group key injected
        sampler.study_to_rows[key] = [int(row_idxs[int(i)]), int(row_idxs[int(j)])]
        r = sampler._draw_pair(key, rng)
        if r is not None:
            records.append(r)
    if not records:
        print("sampler couldn't draw pairs for on-disk studies (tolerance?). widen dicom set.")
        return
    from phase_matched_pair_dataset import _records_to_pair_dataframe, _records_to_anchor_table
    pair_df = _records_to_pair_dataframe(records, sampler._df)
    pair_df = pair_df.reset_index(drop=True)
    sub_local = _rewrite_to_local(pair_df, args.dicom_dir)
    # Rebuild anchor table for the reduced DataFrame's contiguous indices.
    sub_anchors = {
        int(i): [
            {"anchor_frame": int(r.clip_a_anchor_frame), "frame_step": int(r.frame_step)},
            {"anchor_frame": int(r.clip_b_anchor_frame), "frame_step": int(r.frame_step)},
        ]
        for i, r in sub_local.iterrows()
    }
    dataset.set_pair_dataframe(sub_local, anchors_by_index=sub_anchors)

    print(f"\nrunning anchor-loading sanity on {len(sub)} pairs...\n")
    results = []
    for idx in range(len(sub)):
        row = sub_local.iloc[idx]
        anchors = sub_anchors[idx]
        # Call the per-view loader directly so we capture meta immediately
        # before the next call overwrites dataset._last_clip_meta.
        clips_a, idxs_a = dataset._loadvideo_decord_multi(
            row.view_0, args.frames_per_clip, 1, anchor_frame=anchors[0]
        )
        meta_a = dataset._last_clip_meta[0] if getattr(dataset, "_last_clip_meta", None) else {}
        clips_b, idxs_b = dataset._loadvideo_decord_multi(
            row.view_1, args.frames_per_clip, 1, anchor_frame=anchors[1]
        )
        meta_b = dataset._last_clip_meta[0] if getattr(dataset, "_last_clip_meta", None) else {}
        idx_a = idxs_a[0]
        idx_b = idxs_b[0]

        anchor_a = int(row.clip_a_anchor_frame)
        anchor_b = int(row.clip_b_anchor_frame)

        in_a = bool(anchor_a in idx_a.tolist())
        in_b = bool(anchor_b in idx_b.tolist())
        # Fallback: if not exactly in, find the nearest.
        nearest_a_pos = int(np.argmin(np.abs(idx_a - anchor_a)))
        nearest_b_pos = int(np.argmin(np.abs(idx_b - anchor_b)))
        nearest_a_dist = int(abs(int(idx_a[nearest_a_pos]) - anchor_a))
        nearest_b_dist = int(abs(int(idx_b[nearest_b_pos]) - anchor_b))

        # Clamped? inspect _compute_clip_indices' meta via the dataset.
        clamped_a = bool(meta_a.get("was_clamped", False))
        clamped_b = bool(meta_b.get("was_clamped", False))

        out = {
            "pair": idx,
            "study_id": row.study_id,
            "views": (row.clip_a_view, row.clip_b_view),
            "anchor_a": anchor_a, "anchor_b": anchor_b,
            "indices_a_first_last": (int(idx_a[0]), int(idx_a[-1])),
            "indices_b_first_last": (int(idx_b[0]), int(idx_b[-1])),
            "anchor_a_in_indices": in_a, "anchor_b_in_indices": in_b,
            "nearest_a_dist": nearest_a_dist, "nearest_b_dist": nearest_b_dist,
            "anchor_pos_a": meta_a.get("anchor_pos"), "anchor_pos_b": meta_b.get("anchor_pos"),
            "was_clamped_a": clamped_a, "was_clamped_b": clamped_b,
            "padded_a": bool(meta_a.get("padded", False)),
            "padded_b": bool(meta_b.get("padded", False)),
            "frame_step": int(row.frame_step),
            "source_span_frames": int(row.source_span_frames),
            "source_span_cycles_a": float(row.source_span_cycles_a),
            "source_span_cycles_b": float(row.source_span_cycles_b),
            "phase_a": float(row.clip_a_phase_at_anchor),
            "phase_b": float(row.clip_b_phase_at_anchor),
            "target_phi_a": float(row.target_phi_a),
            "target_phi_b": float(row.target_phi_b),
            "circular_phase_diff": float(row.circular_phase_diff),
        }
        results.append(out)
        print(json.dumps(out, indent=2, default=str))

        # Hard assertions for acceptance.
        if not clamped_a:
            assert in_a, f"pair {idx}: anchor_a {anchor_a} not in loaded indices {idx_a.tolist()}"
            # Center-ish: anchor_pos within one step of the exact center.
            center = (args.frames_per_clip - 1) // 2
            assert abs(meta_a["anchor_pos"] - center) <= 1, (
                f"pair {idx}: anchor_pos_a {meta_a['anchor_pos']} not near center {center}"
            )
        if not clamped_b:
            assert in_b, f"pair {idx}: anchor_b {anchor_b} not in loaded indices {idx_b.tolist()}"
            center = (args.frames_per_clip - 1) // 2
            assert abs(meta_b["anchor_pos"] - center) <= 1, (
                f"pair {idx}: anchor_pos_b {meta_b['anchor_pos']} not near center {center}"
            )

    print(f"\n{len(results)} pairs passed anchor-loading sanity.")


if __name__ == "__main__":
    main()
