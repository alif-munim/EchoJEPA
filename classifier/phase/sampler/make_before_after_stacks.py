"""Generate before/after phase-alignment stacked GIFs for a list of study
pairs. For each pair emits two GIFs::

    examples/phase_aligned/<study>/<a>_<b>_stacked_prealign.gif  (naive)
    examples/phase_aligned/<study>/<a>_<b>_stacked.gif           (phase-aligned)

Native-temporal-sampling version: each panel plays the source clip's raw
frames at the source fps, so the heart beats at its real rate. The two
panels may have different RR lengths (when HR differs between acquisitions);
we pad the shorter one with held frames at its final position so the
single stacked GIF has a well-defined frame count. Panel timing is matched
to ``min(fps_a, fps_b)`` so a single `duration` per frame applies to both.

Reuses frame-pick and masking logic from ``make_phase_aligned_gifs``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from PIL import Image, ImageDraw

import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from make_phase_aligned_gifs import (  # noqa: E402
    _decode_pixels,
    _mask_below_sector,
    _nearest_confident_frame,
    _pick_rr_interval,
    write_gif,
)


def _annotate(frame: np.ndarray, label: str, phi: float | None, frame_idx: int, n_src: int) -> np.ndarray:
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    phi_str = f"{phi:.2f}" if (phi is not None and np.isfinite(phi)) else "--"
    txt = f"{label}  phi={phi_str}  f={frame_idx}/{n_src - 1}"
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        draw.text((8 + dx, 8 + dy), txt, fill=(0, 0, 0))
    draw.text((8, 8), txt, fill=(255, 255, 255))
    return np.array(img)


def load_clip_context(parquet: Path, dicom_dir: Path, clip_id: str):
    """Return (pixels, n_src, phase, confident, view_row)."""
    df = pd.read_parquet(
        parquet,
        columns=[
            "dicom_id", "n_video_frames", "fps_video", "hr_metadata",
            "r_peaks_video_json", "per_frame_phase_json", "confident_mask_json",
        ],
    )
    row = df[df.dicom_id == clip_id].iloc[0]
    phase = np.array(
        [np.nan if v is None else float(v) for v in json.loads(row.per_frame_phase_json)],
        dtype=np.float64,
    )
    confident = np.array(json.loads(row.confident_mask_json), dtype=bool)
    ds = pydicom.dcmread(str(dicom_dir / f"{clip_id}.dcm"))
    pixels = _decode_pixels(ds)
    pixels = _mask_below_sector(pixels, ds)
    return pixels, int(row.n_video_frames), phase, confident, row


def _frames_for_indices(pixels, phase, confident, label, idxs, n_src):
    """Render the given source frame indices with annotations."""
    out = []
    for i in idxs:
        i = int(i)
        ph = phase[i] if (i < len(phase) and confident[i]) else float("nan")
        out.append(_annotate(pixels[i].copy(), label, ph, i, n_src))
    return out


def prealign_indices(n_src, length):
    """Native-rate: first ``length`` source frames (or all if shorter)."""
    return list(range(min(length, n_src)))


def aligned_indices(row):
    """Native-rate: source-frame indices inside the picked RR interval."""
    restrict = _pick_rr_interval(row, strategy="nearest_meta")
    if restrict is None or len(restrict) == 0:
        return None
    return list(map(int, restrict))


def _pad_to_length(frames, target_len):
    """Pad with the last frame so both panels share frame count."""
    if len(frames) >= target_len:
        return frames[:target_len]
    return frames + [frames[-1]] * (target_len - len(frames))


def stack_v(frames_top, frames_bot, gap_px=4):
    """Vertically stack equal-length frame lists; resize to min width."""
    assert len(frames_top) == len(frames_bot)
    w = min(frames_top[0].shape[1], frames_bot[0].shape[1])

    def resize(frames):
        out = []
        for f in frames:
            h, wf = f.shape[:2]
            new_h = int(round(h * w / wf))
            out.append(np.array(Image.fromarray(f).resize((w, new_h), Image.BILINEAR)))
        return out

    a = resize(frames_top)
    b = resize(frames_bot)
    gap = np.zeros((gap_px, w, 3), dtype=np.uint8)
    return [np.vstack([fa, gap, fb]) for fa, fb in zip(a, b)]


def process_pair(
    parquet: Path,
    dicom_dir: Path,
    out_root: Path,
    study: str,
    clip_a: str,
    clip_b: str,
    label_a: str,
    label_b: str,
    prealign_seconds: float = 2.0,
    loop_cycles: int = 3,
):
    """Render both stacks at *source frame rate* so the heart beats at its
    true speed.

    - Pre-align: first ``prealign_seconds`` worth of each source clip.
    - Aligned: extract the source frames inside each clip's picked RR
      interval; pad the shorter RR with held frames; optionally repeat
      ``loop_cycles`` times so the rhythm is obvious when the viewer
      doesn't auto-loop.

    GIF frame duration = ``1000 / min(fps_a, fps_b)`` ms. Slight temporal
    skew (a few %) between panels is acceptable; the alternative is
    resampling one clip which hides the HR story.
    """
    pa, na, ph_a, c_a, row_a = load_clip_context(parquet, dicom_dir, clip_a)
    pb, nb, ph_b, c_b, row_b = load_clip_context(parquet, dicom_dir, clip_b)

    fps_a = float(row_a.fps_video)
    fps_b = float(row_b.fps_video)
    fps_out = min(fps_a, fps_b)
    duration_ms = int(round(1000.0 / fps_out))

    # --- pre-align: native rate, ~prealign_seconds of each clip ---
    pre_len = int(round(prealign_seconds * fps_out))
    pre_idx_a = prealign_indices(na, pre_len)
    pre_idx_b = prealign_indices(nb, pre_len)
    pre_a = _frames_for_indices(pa, ph_a, c_a, label_a, pre_idx_a, na)
    pre_b = _frames_for_indices(pb, ph_b, c_b, label_b, pre_idx_b, nb)
    T = max(len(pre_a), len(pre_b))
    pre_stack = stack_v(_pad_to_length(pre_a, T), _pad_to_length(pre_b, T))
    pre_path = out_root / study / f"{clip_a}_{clip_b}_stacked_prealign.gif"
    write_gif_ms(pre_stack, pre_path, duration_ms)
    print(f"wrote {pre_path}  ({T}f @ {duration_ms}ms, ~{1000/duration_ms:.1f} fps)")

    # --- aligned: each clip's own RR at native rate, padded to match length ---
    aln_idx_a = aligned_indices(row_a)
    aln_idx_b = aligned_indices(row_b)
    if aln_idx_a is None or aln_idx_b is None:
        print(f"  skipped aligned stack: no valid RR interval")
        return
    aln_a = _frames_for_indices(pa, ph_a, c_a, label_a, aln_idx_a, na)
    aln_b = _frames_for_indices(pb, ph_b, c_b, label_b, aln_idx_b, nb)
    T = max(len(aln_a), len(aln_b))
    # Repeat the whole sequence loop_cycles times so the rhythm is obvious.
    aln_a_loop = (_pad_to_length(aln_a, T)) * loop_cycles
    aln_b_loop = (_pad_to_length(aln_b, T)) * loop_cycles
    aln_stack = stack_v(aln_a_loop, aln_b_loop)
    aln_path = out_root / study / f"{clip_a}_{clip_b}_stacked.gif"
    write_gif_ms(aln_stack, aln_path, duration_ms)
    rr_a_ms = len(aln_a) * 1000.0 / fps_a
    rr_b_ms = len(aln_b) * 1000.0 / fps_b
    print(f"wrote {aln_path}  (RR_a={len(aln_a)}f/{rr_a_ms:.0f}ms ~{60000/rr_a_ms:.0f}bpm, "
          f"RR_b={len(aln_b)}f/{rr_b_ms:.0f}ms ~{60000/rr_b_ms:.0f}bpm, "
          f"T={T}x{loop_cycles})")


def write_gif_ms(frames, path, duration_ms):
    """Thin wrapper around PIL save with explicit per-frame duration."""
    path.parent.mkdir(parents=True, exist_ok=True)
    imgs = [Image.fromarray(f) for f in frames]
    # Browsers enforce a ~20ms minimum; keep above that.
    duration_ms = max(20, int(duration_ms))
    imgs[0].save(
        path,
        save_all=True,
        append_images=imgs[1:],
        duration=duration_ms,
        loop=0,
        disposal=2,
        optimize=False,
    )


def main():
    phase_dir = HERE.parent  # classifier/phase
    parquet = phase_dir / "phase_annotations" / "phase_annotations.parquet"
    dicom_dir = phase_dir / "dicoms"
    out_root = phase_dir / "examples" / "phase_aligned"

    # All pairs pass RR-vs-metadata sanity (median RR within 25% of
    # 60*fps/HR_metadata). The naive "high tier with rpeak_ratio_dist <
    # 0.10" gate can let through clips where the detector marks every
    # 2nd or 4th beat — avoid those for visualization.
    # Pairs below all pass two sanity layers:
    #   (a) median RR within 25% of metadata cycle (60*fps/HR_metadata)
    #   (b) max(RR) / min(RR) <= 1.4    (reject clips with missed beats)
    # Alignment walker uses strategy="nearest_meta" to pick the RR closest
    # to the metadata cycle, so a missed-beat interval never gets chosen.
    pairs = [
        # (study, clip_a, label_a, clip_b, label_b)
        # Original three (non-apical and low-dispersion apical):
        ("94712615", "94712615_0007", "PLAX",    "94712615_0047", "PLAX"),
        ("96166542", "96166542_0014", "PSAX-AV", "96166542_0043", "A5C"),
        ("95624795", "95624795_0056", "A4C",     "95624795_0063", "Exclude"),
        # Top-dispersion apical-mixed pairs spanning every view-combo,
        # pulled from the full-cohort ranking (pre |Δφ| >= 0.488 each):
        ("95462184", "95462184_0046", "A4C", "95462184_0067", "A2C"),
        ("92900930", "92900930_0075", "A2C", "92900930_0079", "A3C"),
        ("91722876", "91722876_0007", "A2C", "91722876_0028", "A5C"),
        ("94330184", "94330184_0058", "A4C", "94330184_0071", "A3C"),
        ("95373991", "95373991_0070", "A5C", "95373991_0006", "A3C"),
        ("93083042", "93083042_0042", "A4C", "93083042_0055", "A5C"),
    ]
    for study, a, la, b, lb in pairs:
        print(f"\n== study {study}: {a} ({la}) + {b} ({lb}) ==")
        process_pair(parquet, dicom_dir, out_root, study, a, b, la, lb)


if __name__ == "__main__":
    main()
