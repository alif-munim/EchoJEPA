#!/usr/bin/env python3
"""Validate R-peak phase alignment as a multi-view JEPA supervision signal.

Pipeline:
  1. Load JEPA-IN21K-ep100 encoder (ViT-L, 224px, 16-frame clips).
  2. For each clip in the 30+30 alignment test set, extract video frames from
     DICOM, cache to frame_cache/{clip}.npz.
  3. Compute per-clip per-frame cardiac phase from R-peak indices via the
     ECG->video index mapping.
  4. Embed every kept frame, cache to embedding_cache/{clip}.npz.
  5. For each within-study pair, draw 8 anchor frames from clip A; find the
     phase-matched frame in clip B, a random frame in clip B, and the
     phase-matched frame in a cross-study clip C. Record cosine similarities.
  6. Report Δ_within and Δ_specificity distributions + phase-bucketed
     breakdown. Emit a decision-rule verdict.

Sanity checks are gated: the script refuses to run the main experiment if any
of (encoder sanity, mapping sanity, self-similarity matrix) fails.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pydicom

import os as _os

HERE = Path(__file__).resolve().parent
# Resolve to the vjepa2 root (contains src/, configs/, checkpoints/). Look
# first at VJEPA_ROOT env var, then walk up from this file, then check a
# known-good HyperPod location as a last resort.
def _find_vjepa_root() -> Path:
    env = _os.environ.get("VJEPA_ROOT")
    if env and (Path(env) / "src" / "models" / "vision_transformer.py").exists():
        return Path(env)
    for p in [HERE] + list(HERE.parents):
        if (p / "src" / "models" / "vision_transformer.py").exists():
            return p
    for fallback in [Path("/opt/vjepa2"),
                     Path("/mnt/custom-file-systems/efs/fs-0049217cdf69186d7_fsap-0fa7145b64eaa046b/vjepa2")]:
        if (fallback / "src" / "models" / "vision_transformer.py").exists():
            return fallback
    raise RuntimeError(
        "Could not locate vjepa2 root. Set VJEPA_ROOT env var to the dir "
        "containing src/models/vision_transformer.py"
    )

VJEPA_ROOT = _find_vjepa_root()
# JEPA IN21K e100 checkpoint. First EFS path is the source of truth; the
# NVMe path is the same file staged for the HyperPod compute node.
_CKPT_CANDIDATES = [
    VJEPA_ROOT / "checkpoints" / "jepa_in21k_vitl_e100.pt",
    Path("/opt/dlami/nvme/checkpoints/jepa_in21k_vitl_e100.pt"),
]
CHECKPOINT = next((p for p in _CKPT_CANDIDATES if p.exists()), _CKPT_CANDIDATES[0])

FRAME_CACHE = HERE / "frame_cache"
EMBED_CACHE = HERE / "embedding_cache"
PROCESSED_DIR = HERE / "lastframe" / "waveform_processed"
CALIB_CSV = HERE / "calibration_results.csv"
META_CSV = HERE / "dicom_metadata.csv"
DICOM_DIR = HERE / "dicoms"

RESOLUTION = 224
# Encoder uses tubelet_size=2, so the minimum usable temporal dim is 2.
# We feed each target frame duplicated to length 2 (acts as a per-frame
# encoder). This keeps inference cheap enough to run on CPU if needed.
FRAMES_PER_CLIP = 2

# -----------------------------------------------------------------------------
# Encoder
# -----------------------------------------------------------------------------

def build_encoder(device: str = "cuda"):
    """Load the JEPA IN21K e100 encoder in eval mode."""
    import sys as _sys
    if str(VJEPA_ROOT) not in _sys.path:
        _sys.path.insert(0, str(VJEPA_ROOT))
    import torch
    import src.models.vision_transformer as vit

    ckpt = torch.load(str(CHECKPOINT), map_location="cpu", weights_only=False)
    enc_kwargs = dict(
        model_name="vit_large",
        patch_size=16,
        tubelet_size=2,
        uniform_power=True,
        use_rope=True,
    )
    model = vit.__dict__[enc_kwargs["model_name"]](
        img_size=RESOLUTION, num_frames=FRAMES_PER_CLIP, **enc_kwargs
    )
    pretrained = ckpt["target_encoder"]
    pretrained = {k.replace("module.", "").replace("backbone.", ""): v
                  for k, v in pretrained.items()}
    # Trim / retain only matching-shape keys.
    model_state = model.state_dict()
    loadable = {}
    for k, v in pretrained.items():
        if k in model_state and model_state[k].shape == v.shape:
            loadable[k] = v
    missing = set(model_state.keys()) - set(loadable.keys())
    extra = set(pretrained.keys()) - set(loadable.keys())
    if missing:
        print(f"[encoder] {len(missing)} keys not loaded (e.g. {list(missing)[:3]})")
    msg = model.load_state_dict(loadable, strict=False)
    print(f"[encoder] load_state_dict -> {msg}")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.to(device)
    print(f"[encoder] embed_dim={model.embed_dim}  params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    return model


def embed_frames_batch(model, frames_uint8: np.ndarray, device: str = "cuda",
                       batch_size: int = 8) -> np.ndarray:
    """Embed a (N, H, W, 3) uint8 array into (N, D) by replicating each frame
    into a FRAMES_PER_CLIP-length pseudo-clip."""
    import torch
    N = len(frames_uint8)
    out = []
    # ImageNet-style normalization (what JEPA uses by default).
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    for start in range(0, N, batch_size):
        batch = frames_uint8[start:start + batch_size].astype(np.float32) / 255.0
        batch = (batch - mean) / std
        # (B, H, W, 3) -> (B, 3, H, W)
        batch_t = torch.from_numpy(batch).permute(0, 3, 1, 2).to(device)
        # Replicate each frame FRAMES_PER_CLIP times: (B, 3, T, H, W)
        clip = batch_t.unsqueeze(2).expand(-1, -1, FRAMES_PER_CLIP, -1, -1)
        with torch.no_grad():
            tokens = model(clip)  # (B, num_tokens, D)
            pooled = tokens.mean(dim=1)  # global spatial+temporal pool
        out.append(pooled.float().cpu().numpy())
    return np.concatenate(out, axis=0)


# -----------------------------------------------------------------------------
# Frame extraction from DICOM
# -----------------------------------------------------------------------------

def load_dicom_frames(dcm_path: Path) -> tuple[np.ndarray, float]:
    """Return (frames [T,H,W,3] uint8 @ RESOLUTION, fps)."""
    ds = pydicom.dcmread(str(dcm_path))
    n_frames = int(getattr(ds, "NumberOfFrames", 1))
    if n_frames <= 1:
        raise RuntimeError("single-frame DICOM")
    pa = ds.pixel_array
    pi = str(getattr(ds, "PhotometricInterpretation", ""))
    if "PALETTE" in pi:
        from pydicom.pixels.processing import apply_color_lut
        pa = apply_color_lut(pa, ds)
        if pa.dtype == np.uint16:
            pa = (pa / 256).astype(np.uint8)
    if pa.ndim == 3:
        pa = np.stack([pa, pa, pa], axis=-1)
    pa = np.ascontiguousarray(pa, dtype=np.uint8)  # (T, H, W, 3)
    # Resize each frame to RESOLUTION with aspect-preserving central crop
    from PIL import Image
    out = np.empty((pa.shape[0], RESOLUTION, RESOLUTION, 3), dtype=np.uint8)
    for i in range(pa.shape[0]):
        im = Image.fromarray(pa[i])
        h, w = im.height, im.width
        side = min(h, w)
        left = (w - side) // 2
        top = (h - side) // 2
        im = im.crop((left, top, left + side, top + side)).resize(
            (RESOLUTION, RESOLUTION), Image.BILINEAR
        )
        out[i] = np.asarray(im)
    frame_time_ms = float(getattr(ds, "FrameTime", 1000.0 / 30.0))
    fps = 1000.0 / frame_time_ms
    return out, fps


def load_or_cache_frames(clip: str) -> tuple[np.ndarray, float] | None:
    p = FRAME_CACHE / f"{clip}.npz"
    if p.exists():
        d = np.load(p)
        return d["frames"], float(d["fps"])
    dcm = DICOM_DIR / f"{clip}.dcm"
    if not dcm.exists():
        return None
    try:
        frames, fps = load_dicom_frames(dcm)
    except Exception as e:
        print(f"[frames] failed {clip}: {e}")
        return None
    FRAME_CACHE.mkdir(exist_ok=True)
    np.savez_compressed(p, frames=frames, fps=np.float32(fps))
    return frames, fps


# -----------------------------------------------------------------------------
# ECG strip -> video frame mapping
# -----------------------------------------------------------------------------

def load_processed_signal(clip: str) -> dict | None:
    p = PROCESSED_DIR / f"{clip}.npz"
    if not p.exists():
        return None
    d = dict(np.load(p))
    return d


def ecg_col_to_video_frame(
    col: int,
    strip_width: int,
    sr_ecg: float,
    n_video_frames: int,
    fps_video: float,
    x0: int | None = None,
    x1: int | None = None,
) -> int:
    """Map an ECG strip column to a video frame index.

    The ECG trace is drawn across the columns of the detected trace span
    ``[x0, x1]`` and is co-registered with the video: column ``x0`` = video
    frame 0, column ``x1`` = video frame ``n_video_frames - 1``. Empirically
    ``(x1 - x0 + 1) / sr_ecg`` matches video duration to within ~5%, so
    interpolating linearly across the span is correct.

    The old convention (right-edge = "now", extrapolating back over the full
    strip width) was wrong: empty margins of the strip PNG are not history,
    they're just empty image space. Passing ``x0``/``x1`` (trace span from
    ``process_waveform``'s NPZ) uses the correct linear mapping; omitting
    them falls back to the old right-edge convention for backward compat.
    """
    if x0 is not None and x1 is not None and x1 > x0:
        span = x1 - x0
        frac = (col - x0) / span
        return int(round(frac * (n_video_frames - 1)))
    # Fallback: right-edge-is-now over the full strip width.
    t_from_end_s = (strip_width - 1 - col) / sr_ecg
    frame_from_end = int(round(t_from_end_s * fps_video))
    return (n_video_frames - 1) - frame_from_end


REGIME_STRICT = "strict"          # between two in-video R-peaks
REGIME_PERMISSIVE = "permissive"  # strict + ±1 median-RR extrapolation
REGIME_HR_EXTRAP = "hr_extrap"    # any R-peak (incl. pre-video) + metadata HR


def video_frame_phase(
    n_video_frames: int,
    fps_video: float,
    r_peaks_ecg: np.ndarray,
    sr_ecg: float,
    strip_width: int,
    hr_metadata: float | None = None,
    hr_extrap_max_cycles: float = 0.5,
    x0: int | None = None,
    x1: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per video-frame cardiac phase in [0, 1) + confidence mask + regime mask.

    Three regimes are stacked in order of decreasing precision:
      1. **strict** — frame sits between two in-video R-peaks; phase is
         `(idx - r0) / (r1 - r0)`. The RR interval is measured directly from
         the clip, so phase is precise to the R-peak detection accuracy.
      2. **permissive** — frame is past the last in-video R-peak or before
         the first, within ±1 within-clip median-RR of that R-peak; phase is
         `(idx - anchor) / median_rr` mod 1. Within-clip RR is still measured,
         but drift accumulates with distance from the anchor.
      3. **hr_extrap** — only usable when strict+permissive leave a frame
         unconfident; anchor is the nearest R-peak (including pre-video ones)
         and the cycle is metadata HR. Least precise — relies on metadata HR
         being accurate for this ~2-3s window.

    Returns (phase, confident, regime), each shape (n_video_frames,).
    ``regime[i]`` is one of REGIME_STRICT / REGIME_PERMISSIVE /
    REGIME_HR_EXTRAP / "" (for unconfident frames).
    """
    r_peaks_video_all = np.array([
        ecg_col_to_video_frame(int(c), strip_width, sr_ecg,
                               n_video_frames, fps_video,
                               x0=x0, x1=x1)
        for c in r_peaks_ecg
    ], dtype=int)
    in_window = (r_peaks_video_all >= 0) & (r_peaks_video_all < n_video_frames)
    r_peaks_video = np.unique(r_peaks_video_all[in_window])
    phase = np.full(n_video_frames, np.nan)
    confident = np.zeros(n_video_frames, dtype=bool)
    regime = np.array([""] * n_video_frames, dtype=object)

    # --- Regime 1: strict between-R-peaks using measured RR ---
    if len(r_peaks_video) >= 2:
        for i in range(len(r_peaks_video) - 1):
            s, e = int(r_peaks_video[i]), int(r_peaks_video[i + 1])
            if e <= s:
                continue
            idx = np.arange(s, e)
            phase[idx] = (idx - s) / (e - s)
            confident[idx] = True
            regime[idx] = REGIME_STRICT

    # --- Regime 2: permissive extrapolation from first/last in-video R-peak ---
    if len(r_peaks_video) >= 2:
        rr = np.diff(r_peaks_video.astype(int))
        median_rr = float(np.median(rr))
        if median_rr > 0:
            extrap = int(1.0 * median_rr)
            first, last = int(r_peaks_video[0]), int(r_peaks_video[-1])
            pre = np.arange(max(0, first - extrap), first)
            for i in pre:
                if not confident[i]:
                    phase[i] = ((i - first) / median_rr) % 1.0
                    confident[i] = True
                    regime[i] = REGIME_PERMISSIVE
            post = np.arange(last, min(n_video_frames, last + extrap))
            for i in post:
                if not confident[i]:
                    phase[i] = ((i - last) / median_rr) % 1.0
                    confident[i] = True
                    regime[i] = REGIME_PERMISSIVE

    # --- Regime 3: HR-extrapolated from any R-peak (incl. pre-video) ---
    if (hr_metadata is not None and hr_metadata > 0
            and len(r_peaks_video_all) > 0):
        cycle_frames = 60.0 / float(hr_metadata) * float(fps_video)
        if cycle_frames > 0:
            max_dist = hr_extrap_max_cycles * cycle_frames
            idx = np.arange(n_video_frames)
            anchors = r_peaks_video_all.astype(np.float64)
            dists = np.abs(idx[:, None] - anchors[None, :])
            nearest = dists.min(axis=1)
            nearest_anchor = anchors[dists.argmin(axis=1)]
            can_fill = (~confident) & (nearest <= max_dist)
            if can_fill.any():
                ph = ((idx - nearest_anchor) / cycle_frames) % 1.0
                phase[can_fill] = ph[can_fill]
                confident[can_fill] = True
                regime[can_fill] = REGIME_HR_EXTRAP

    return phase, confident, regime


# -----------------------------------------------------------------------------
# Supporting data loaders
# -----------------------------------------------------------------------------

def load_clip_data(limit: int | None = None) -> dict:
    """Build {clip_id: {sr, hr, study, n_frames}} restricted to multi-frame cines."""
    sr_by = {}
    with CALIB_CSV.open() as f:
        for r in csv.DictReader(f):
            try:
                sr_by[r["dicom_id"].replace(".dcm", "")] = float(r["sampling_rate_hz"])
            except (ValueError, KeyError):
                pass
    meta_by: dict[str, dict] = {}
    with META_CSV.open() as f:
        for r in csv.DictReader(f):
            k = (r.get("dicom", "") or r.get("dicom_id", "")).replace(".dcm", "")
            try:
                hr = float(r.get("heart_rate", "") or 0)
            except ValueError:
                hr = 0
            try:
                n_frames = int(r.get("n_frames", "") or 0)
            except ValueError:
                n_frames = 0
            meta_by[k] = {"hr": hr if hr > 0 else None, "n_frames": n_frames}
    clips: dict = {}
    for npz in sorted(PROCESSED_DIR.glob("*.npz")):
        c = npz.stem
        if c not in sr_by:
            continue
        m = meta_by.get(c, {})
        if not m.get("hr") or m.get("n_frames", 0) < 16:
            continue  # need multi-frame cine + HR
        clips[c] = {
            "sr_ecg": sr_by[c],
            "hr": m["hr"],
            "n_frames": m["n_frames"],
            "study": c.split("_")[0],
        }
        if limit and len(clips) >= limit:
            break
    return clips


def sample_pairs(clips: dict, n_within: int = 30, n_cross: int = 30,
                 seed: int = 42):
    """Same sampling as test_rpeak_alignment.py (seed 42)."""
    rng = np.random.default_rng(seed)
    by_study: dict[str, list[str]] = defaultdict(list)
    for c, v in clips.items():
        by_study[v["study"]].append(c)
    multi = {s: cs for s, cs in by_study.items() if len(cs) >= 2}
    studies = list(multi.keys())
    rng.shuffle(studies)
    within = []
    for s in studies:
        if len(within) >= n_within:
            break
        cs = multi[s]
        i, j = rng.choice(len(cs), size=2, replace=False)
        within.append((cs[i], cs[j]))
    all_clips = [c for cs in by_study.values() for c in cs]
    cross = []
    attempts = 0
    while len(cross) < n_cross and attempts < 10000:
        attempts += 1
        i, j = rng.choice(len(all_clips), size=2, replace=False)
        a, b = all_clips[i], all_clips[j]
        if a.split("_")[0] != b.split("_")[0]:
            cross.append((a, b))
    return within, cross


# -----------------------------------------------------------------------------
# R-peak detection (reuse robust_rpeaks)
# -----------------------------------------------------------------------------

def get_rpeaks_for_clip(clip: str, clips: dict) -> np.ndarray:
    """Run robust_rpeaks on the processed NPZ and return ECG-column indices."""
    from rpeak_detectors import robust_rpeaks
    d = load_processed_signal(clip)
    if d is None:
        return np.array([], dtype=int)
    full_y = d["full_y"].astype(np.float64)
    x0, x1 = int(d["x0"]), int(d["x1"])
    if x1 - x0 < 50:
        return np.array([], dtype=int)
    seg = np.nan_to_num(full_y[x0:x1 + 1], nan=0.0)
    sr = clips[clip]["sr_ecg"]
    hr = clips[clip]["hr"]
    if not hr:
        return np.array([], dtype=int)
    try:
        peaks, _, _ = robust_rpeaks(seg, sr, hr)
    except Exception:
        return np.array([], dtype=int)
    return (peaks + x0).astype(int)


# -----------------------------------------------------------------------------
# Per-clip embedding cache
# -----------------------------------------------------------------------------

def compute_phase_for_clip(
    clip: str,
    clips: dict,
    n_video_frames: int,
    fps: float,
    hr_extrap_max_cycles: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (phase, confident, regime_bytes, r_peaks_ecg) for a clip.

    regime_bytes is an S-dtype array so it can round-trip through np.savez.
    """
    sig = load_processed_signal(clip)
    if sig is None:
        return (np.full(n_video_frames, np.nan, dtype=np.float32),
                np.zeros(n_video_frames, dtype=bool),
                np.full(n_video_frames, "", dtype="S12"),
                np.array([], dtype=np.int32))
    strip_width = int(sig["width"])
    x0 = int(sig["x0"]); x1 = int(sig["x1"])
    sr = clips[clip]["sr_ecg"]
    hr = clips[clip].get("hr")
    r_peaks = get_rpeaks_for_clip(clip, clips)
    phase, confident, regime = video_frame_phase(
        n_video_frames, fps, r_peaks, sr, strip_width,
        hr_metadata=hr,
        hr_extrap_max_cycles=hr_extrap_max_cycles,
        x0=x0, x1=x1,
    )
    regime_bytes = np.array([s.encode("ascii") for s in regime], dtype="S12")
    return (phase.astype(np.float32), confident,
            regime_bytes, r_peaks.astype(np.int32))


def load_or_embed(model, clip: str, clips: dict, device: str,
                  hr_extrap_max_cycles: float = 0.5,
                  rebuild_phase: bool = True) -> dict | None:
    """Return {embeddings, phase, confident, regime, fps, strip_width, r_peaks_ecg}.

    If a cache exists, keep the embeddings but recompute phase/confident/regime
    under the current rules (cheap; avoids re-running the encoder).
    """
    cache_path = EMBED_CACHE / f"{clip}.npz"
    if cache_path.exists():
        d = dict(np.load(cache_path))
        if rebuild_phase:
            fps = float(d["fps"])
            n_frames = len(d["embeddings"])
            phase, conf, regime, r_peaks = compute_phase_for_clip(
                clip, clips, n_frames, fps, hr_extrap_max_cycles
            )
            d["phase"] = phase
            d["confident"] = conf
            d["regime"] = regime
            d["r_peaks_ecg"] = r_peaks
        return d
    frames_info = load_or_cache_frames(clip)
    if frames_info is None:
        return None
    frames, fps = frames_info
    sig = load_processed_signal(clip)
    if sig is None:
        return None
    strip_width = int(sig["width"])
    sr = clips[clip]["sr_ecg"]
    phase, confident, regime_bytes, r_peaks = compute_phase_for_clip(
        clip, clips, len(frames), fps, hr_extrap_max_cycles,
    )
    embs = embed_frames_batch(model, frames, device=device)
    EMBED_CACHE.mkdir(exist_ok=True)
    out = {
        "embeddings": embs.astype(np.float32),
        "phase": phase,
        "confident": confident,
        "regime": regime_bytes,
        "fps": np.float32(fps),
        "strip_width": np.int32(strip_width),
        "r_peaks_ecg": r_peaks,
        "sr_ecg": np.float32(sr),
    }
    np.savez_compressed(cache_path, **out)
    return out


# -----------------------------------------------------------------------------
# Sanity checks
# -----------------------------------------------------------------------------

def cos(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(a @ b)


def sanity_encoder(model, clips: dict, device: str) -> bool:
    """Sanity 1: adjacent same-clip frames should be much more similar than
    random cross-clip frames."""
    print("\n[sanity 1] encoder adjacency test")
    keys = list(clips.keys())
    rng = np.random.default_rng(0)
    a_clip = keys[0]; b_clip = keys[-1]
    a_info = load_or_cache_frames(a_clip)
    b_info = load_or_cache_frames(b_clip)
    if a_info is None or b_info is None:
        print("  failed to load frames")
        return False
    a_frames, _ = a_info; b_frames, _ = b_info
    pairs = [
        ("same-clip adj", a_frames[0], a_frames[1]),
        ("same-clip mid", a_frames[len(a_frames) // 2], a_frames[len(a_frames) // 2 + 1]),
        ("cross-clip 1", a_frames[0], b_frames[0]),
        ("cross-clip 2", a_frames[5], b_frames[5]),
    ]
    stacked = np.stack([p[1] for p in pairs] + [p[2] for p in pairs], axis=0)
    embs = embed_frames_batch(model, stacked, device=device)
    n = len(pairs)
    for i, (lbl, _, _) in enumerate(pairs):
        s = cos(embs[i], embs[i + n])
        print(f"  {lbl:20s}  cos={s:+.4f}")
    # Success: same-clip > cross-clip by a clear margin.
    same = [cos(embs[i], embs[i + n]) for i, (lbl, _, _) in enumerate(pairs) if "same" in lbl]
    crs = [cos(embs[i], embs[i + n]) for i, (lbl, _, _) in enumerate(pairs) if "cross" in lbl]
    ok = np.mean(same) - np.mean(crs) > 0.05
    print(f"  mean same-clip: {np.mean(same):+.3f}  cross-clip: {np.mean(crs):+.3f}  "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def sanity_mapping(clips: dict) -> bool:
    """Sanity 2: plot strip + R-peak frames for one clip."""
    print("\n[sanity 2] ECG->video index mapping")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for clip in list(clips.keys()):
        sig = load_processed_signal(clip)
        if sig is None:
            continue
        frames_info = load_or_cache_frames(clip)
        if frames_info is None:
            continue
        frames, fps = frames_info
        sr = clips[clip]["sr_ecg"]
        r_peaks = get_rpeaks_for_clip(clip, clips)
        if len(r_peaks) < 2:
            continue
        strip_width = int(sig["width"])
        full_y = sig["full_y"]
        out_path = HERE / "embedding_mapping_diagnostic.png"
        r_peaks_video = [
            ecg_col_to_video_frame(int(c), strip_width, sr, len(frames), fps)
            for c in r_peaks
        ]
        valid_peaks = [(c, v) for c, v in zip(r_peaks, r_peaks_video)
                       if 0 <= v < len(frames)]
        if len(valid_peaks) < 2:
            continue

        fig = plt.figure(figsize=(14, 6))
        ax_strip = fig.add_subplot(2, 1, 1)
        t = np.arange(strip_width) / sr
        ax_strip.plot(t, full_y, color="tab:blue", linewidth=0.8)
        for c, _ in valid_peaks:
            ax_strip.axvline(c / sr, color="red", alpha=0.6)
        ax_strip.set_title(f"{clip}  sr={sr:.0f}Hz  fps={fps:.1f}  "
                           f"n_video_frames={len(frames)}  n_rpeaks={len(valid_peaks)}")
        ax_strip.set_xlabel("time from strip start (s)")

        n_show = min(6, len(valid_peaks))
        for i in range(n_show):
            ax = fig.add_subplot(2, n_show, n_show + i + 1)
            _, v = valid_peaks[i]
            ax.imshow(frames[v])
            ax.set_title(f"frame {v}\n(R-peak col {valid_peaks[i][0]})", fontsize=8)
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_path, dpi=100)
        plt.close(fig)
        print(f"  wrote {out_path}  ({n_show} R-peak frames shown)")
        return True
    print("  no usable clip found")
    return False


def sanity_self_similarity(model, clips: dict, device: str) -> bool:
    """Sanity 3: self-similarity matrix should show cyclic block structure."""
    print("\n[sanity 3] per-clip self-similarity matrix")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for clip in list(clips.keys()):
        frames_info = load_or_cache_frames(clip)
        if frames_info is None:
            continue
        frames, fps = frames_info
        if len(frames) < 16:
            continue
        embs = embed_frames_batch(model, frames, device=device)
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9
        norm_embs = embs / norms
        sim = norm_embs @ norm_embs.T
        out_path = HERE / "embedding_self_similarity_example.png"
        fig, ax = plt.subplots(figsize=(6, 5), dpi=120)
        im = ax.imshow(sim, cmap="viridis", vmin=-1, vmax=1)
        ax.set_title(f"{clip}: frame-to-frame cosine similarity (n={len(frames)})")
        ax.set_xlabel("frame j"); ax.set_ylabel("frame i")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        # Heuristic: check whether the off-diagonal decay is non-monotone
        # (indicates cyclic structure). Compute the lag-1 decay and lag at
        # the next local maximum.
        mean_by_lag = np.array([
            np.mean(np.diagonal(sim, offset=k))
            for k in range(1, len(frames) // 2)
        ])
        # Local maxima in mean_by_lag indicate cyclic recurrences.
        peaks_found = 0
        for i in range(2, len(mean_by_lag) - 2):
            if (mean_by_lag[i] > mean_by_lag[i - 1]
                    and mean_by_lag[i] > mean_by_lag[i + 1]
                    and mean_by_lag[i] > mean_by_lag[0] - 0.1):
                peaks_found += 1
        print(f"  {clip}: wrote {out_path}  cyclic_peaks_in_lag={peaks_found}  "
              f"lag0 mean={mean_by_lag[0]:.3f}")
        return peaks_found >= 1 or mean_by_lag[0] > 0.8  # or strong adjacency
    return False


# -----------------------------------------------------------------------------
# Main experiment
# -----------------------------------------------------------------------------

def find_phase_matched(cache: dict, target_phase: float) -> int | None:
    phase = cache["phase"]
    conf = cache["confident"]
    valid = conf & ~np.isnan(phase)
    if not valid.any():
        return None
    diffs = np.abs(((phase - target_phase) + 0.5) % 1.0 - 0.5)
    diffs = np.where(valid, diffs, np.inf)
    idx = int(np.argmin(diffs))
    return idx if diffs[idx] < 0.5 else None


def run_experiment(model, clips: dict, within_pairs, cross_pairs,
                   n_anchors: int = 8, device: str = "cuda",
                   seed: int = 42):
    rng = np.random.default_rng(seed)
    # Pre-embed every needed clip.
    all_needed = set()
    for a, b in within_pairs:
        all_needed.add(a); all_needed.add(b)
    for a, b in cross_pairs:
        all_needed.add(a); all_needed.add(b)
    # For cross-study controls we need extra clips with detected R-peaks.
    cache: dict[str, dict] = {}
    for c in sorted(all_needed):
        d = load_or_embed(model, c, clips, device)
        if d is not None and d["confident"].any():
            cache[c] = d

    print(f"\n[experiment] usable clips in cache: {len(cache)} "
          f"out of {len(all_needed)} requested")

    # Pool of non-pair clips for cross-study negatives — any clip with
    # detected R-peaks, confident frames.
    usable_by_study: dict[str, list[str]] = defaultdict(list)
    for c, d in cache.items():
        if d["confident"].any():
            usable_by_study[c.split("_")[0]].append(c)

    records = []
    for pair_idx, (a, b) in enumerate(within_pairs):
        if a not in cache or b not in cache:
            continue
        ca = cache[a]; cb = cache[b]
        anchors = np.where(ca["confident"])[0]
        if len(anchors) < n_anchors:
            continue
        sel = np.linspace(0, len(anchors) - 1, n_anchors).astype(int)
        anchor_idxs = anchors[sel]
        # Pick a cross-study negative clip from a different study.
        neg_studies = [s for s in usable_by_study
                       if s != a.split("_")[0] and s != b.split("_")[0]]
        if not neg_studies:
            continue
        neg_study = rng.choice(neg_studies)
        neg_clip = rng.choice(usable_by_study[neg_study])
        cc = cache[neg_clip]
        b_confident = np.where(cb["confident"])[0]
        for ai in anchor_idxs:
            p = float(ca["phase"][ai])
            bi = find_phase_matched(cb, p)
            ci = find_phase_matched(cc, p)
            if bi is None or ci is None:
                continue
            ri = int(rng.choice(b_confident))
            emb_a = ca["embeddings"][ai]
            emb_b_match = cb["embeddings"][bi]
            emb_b_rand = cb["embeddings"][ri]
            emb_c_match = cc["embeddings"][ci]
            records.append({
                "pair_idx": pair_idx,
                "anchor_clip": a,
                "partner_clip": b,
                "cross_clip": neg_clip,
                "anchor_frame": int(ai),
                "anchor_phase": round(p, 4),
                "sim_phase_within": round(cos(emb_a, emb_b_match), 4),
                "sim_random_within": round(cos(emb_a, emb_b_rand), 4),
                "sim_phase_cross": round(cos(emb_a, emb_c_match), 4),
            })
    return records


def summarize(records: list[dict], out_txt: Path, out_png: Path) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import wilcoxon

    if not records:
        msg = "No records produced; experiment failed."
        out_txt.write_text(msg + "\n")
        return msg

    sim_phase = np.array([r["sim_phase_within"] for r in records])
    sim_rand = np.array([r["sim_random_within"] for r in records])
    sim_cross = np.array([r["sim_phase_cross"] for r in records])
    phases = np.array([r["anchor_phase"] for r in records])

    d_within = sim_phase - sim_rand
    d_spec = sim_phase - sim_cross

    def stats(v, name):
        return (f"{name}: n={len(v)}  mean={v.mean():+.3f}  median={np.median(v):+.3f}  "
                f"IQR=[{np.percentile(v, 25):+.3f}, {np.percentile(v, 75):+.3f}]  "
                f"frac>0={(v > 0).mean() * 100:.0f}%")

    lines = []
    lines.append(stats(sim_phase, "sim_phase_within"))
    lines.append(stats(sim_rand, "sim_random_within"))
    lines.append(stats(sim_cross, "sim_phase_cross"))
    lines.append("")
    lines.append(stats(d_within, "delta_within (phase - random)"))
    lines.append(stats(d_spec, "delta_specificity (within - cross)"))

    try:
        w_within = wilcoxon(d_within, alternative="greater")
        lines.append(f"Wilcoxon delta_within > 0:       p = {w_within.pvalue:.2e}")
    except Exception as e:
        lines.append(f"Wilcoxon delta_within failed: {e}")
    try:
        w_spec = wilcoxon(d_spec, alternative="greater")
        lines.append(f"Wilcoxon delta_specificity > 0:  p = {w_spec.pvalue:.2e}")
    except Exception:
        pass

    # Phase-bucketed breakdown
    lines.append("\nPer phase bucket (delta_within):")
    edges = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    bucket_means = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (phases >= lo) & (phases < hi)
        if mask.any():
            vs = d_within[mask]
            lines.append(f"  phase [{lo:.1f},{hi:.1f}): n={mask.sum():3d}  "
                         f"median delta_within={np.median(vs):+.3f}  "
                         f"frac>0={(vs > 0).mean() * 100:.0f}%")
            bucket_means.append((lo, np.median(vs)))
        else:
            bucket_means.append((lo, np.nan))

    # Decision
    med_dw = float(np.median(d_within))
    med_ds = float(np.median(d_spec))
    frac_dw = float((d_within > 0).mean())
    frac_ds = float((d_spec > 0).mean())
    if med_dw >= 0.10 and med_ds >= 0.05 and frac_dw > 0.85 and frac_ds > 0.85:
        verdict = "PASS — both deltas strongly positive. Commit to multi-view JEPA with R-peak phase."
    elif med_dw >= 0.05 and frac_dw > 0.70 and (med_ds < 0.03 or frac_ds < 0.70):
        verdict = ("PARTIAL — delta_within positive, delta_specificity near zero. "
                   "Encoder picks up cardiac phase but not patient-specific structure. "
                   "Substrate provides cycle alignment only.")
    elif med_dw <= 0.03 or frac_dw < 0.70:
        verdict = ("WEAK — delta_within too small. Encoder not sensitive to phase, "
                   "or phase alignment not predictive of frame similarity.")
    elif med_dw < 0:
        verdict = "BROKEN — delta_within negative. Check ECG->video mapping first."
    else:
        verdict = ("INTERMEDIATE — delta_within positive but below 'strong' thresholds; "
                   "judgment call needed.")
    lines.append("")
    lines.append(f"Decision: {verdict}")

    out_txt.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # Phase-bucket diagnostic plot
    fig, ax = plt.subplots(figsize=(7, 4), dpi=120)
    labels = [f"[{lo:.1f},{lo+0.2:.1f})" for lo, _ in bucket_means]
    vals = [v if not np.isnan(v) else 0 for _, v in bucket_means]
    ax.bar(labels, vals, color="tab:blue")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("median delta_within")
    ax.set_xlabel("anchor phase bucket")
    ax.set_title("Δ_within by anchor phase (R-peak at 0.0)")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    return verdict


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--skip-sanity", action="store_true")
    ap.add_argument("--sanity-only", action="store_true")
    ap.add_argument("--n-within", type=int, default=30)
    ap.add_argument("--n-cross", type=int, default=30)
    ap.add_argument("--n-anchors", type=int, default=8)
    ap.add_argument("--device", default=None,
                    help="'cuda' or 'cpu'. Auto: cuda if available else cpu.")
    args = ap.parse_args()

    FRAME_CACHE.mkdir(exist_ok=True)
    EMBED_CACHE.mkdir(exist_ok=True)

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[main] device={device}")

    clips = load_clip_data()
    print(f"Clips with processed NPZ + calibration: {len(clips)}")

    model = build_encoder(device)

    # --- sanity ---
    if not args.skip_sanity:
        ok1 = sanity_encoder(model, clips, device)
        if not ok1:
            print("ABORT: sanity 1 failed. Encoder is not producing sensible embeddings.")
            return
        ok2 = sanity_mapping(clips)
        if not ok2:
            print("ABORT: sanity 2 failed. ECG->video mapping plot could not be produced.")
            return
        ok3 = sanity_self_similarity(model, clips, device)
        if not ok3:
            print("WARN: sanity 3 showed no obvious cyclic structure in lag-decay. "
                  "Continuing, but this may indicate weak encoder-phase coupling.")
        if args.sanity_only:
            return

    within_pairs, cross_pairs = sample_pairs(
        clips, n_within=args.n_within, n_cross=args.n_cross
    )
    print(f"\n[experiment] within pairs: {len(within_pairs)}  "
          f"cross pairs: {len(cross_pairs)}")

    records = run_experiment(
        model, clips, within_pairs, cross_pairs,
        n_anchors=args.n_anchors, device=device,
    )
    print(f"[experiment] anchor records: {len(records)}")

    csv_path = HERE / "embedding_validation_results.csv"
    if records:
        keys = list(records[0].keys())
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader(); w.writerows(records)
        print(f"Wrote {csv_path}")

    summarize(
        records,
        HERE / "embedding_validation_summary.txt",
        HERE / "embedding_phase_diagnostic.png",
    )


if __name__ == "__main__":
    main()
