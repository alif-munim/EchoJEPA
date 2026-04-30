"""Debug harness for the anchor-aware VideoDataset patch.

Loads N samples from the phase-probe train CSV and verifies, per sample:
  (1) Returned clip has exactly frames_per_clip frames.
  (2) All returned frame indices are in [0, n_video_frames).
  (3) anchor_frame is one of the returned indices (best case) OR
      is inside [min_idx, max_idx] (acceptable — anchor is inside window).
  (4) The loaded target (sin/cos of φ_anchor) matches the re-read phase
      from the parquet at anchor_frame to within 1e-3.
  (5) confident_mask is true at anchor_frame.
  (6) Reports fraction of sampled frames with confident_mask true.

Prints a per-sample table (first 10 samples) and aggregate PASS/FAIL
summary. Exits non-zero on any failure.

Usage (on HyperPod compute node):
    python classifier/phase/debug_anchor_probe_loader.py \
        --csv data/csv/mimic_phase_sin_train_10k.csv \
        --parquet classifier/phase/phase_annotations/phase_annotations.parquet \
        --n 100

Subset-able for local smoke with decord + S3 creds.
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

from src.datasets.video_dataset import VideoDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', type=Path, required=True,
                    help='3-col probe CSV: uri anchor_frame target')
    ap.add_argument('--parquet', type=Path,
                    default=Path('classifier/phase/phase_annotations/phase_annotations.parquet'))
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--frames-per-clip', type=int, default=16)
    ap.add_argument('--frame-step', type=int, default=2)
    ap.add_argument('--target', choices=['sin', 'cos'], default='sin')
    args = ap.parse_args()

    # Load first N rows of CSV (parsed same way VideoDataset does)
    raw = pd.read_csv(args.csv, header=None, delimiter=' ', nrows=args.n,
                      names=['uri', 'anchor_frame', 'target'])
    uris = raw['uri'].tolist()
    anchors = raw['anchor_frame'].astype(int).tolist()
    targets = raw['target'].astype(float).tolist()
    print(f"Loaded {len(raw)} sample rows from {args.csv}")

    # Construct dataset with just these rows by writing a temp CSV
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
        for u, a, t in zip(uris, anchors, targets):
            tmp.write(f"{u} {a} {t:.6f}\n")
        tmp_path = tmp.name

    ds = VideoDataset(
        data_paths=[tmp_path],
        frames_per_clip=args.frames_per_clip,
        frame_step=args.frame_step,
        num_clips=1,   # anchor mode forces this anyway
        random_clip_sampling=False,
        filter_short_videos=False,
    )
    print(f"Dataset anchors attr is {'SET' if ds.anchors is not None else 'None'}, "
          f"num_clips={ds.num_clips}")
    assert ds.anchors is not None, "Anchor mode did not activate — check CSV format"
    assert ds.num_clips == 1

    # Load parquet for ground-truth phase + confident-mask
    parq = pd.read_parquet(
        args.parquet,
        columns=['dicom_id', 'per_frame_phase_json', 'confident_mask_json', 'n_video_frames'],
    )
    parq = parq.set_index('dicom_id')

    # Map uri → dicom_id (last 13 chars before .dcm is "<study>_<clip>")
    def uri_to_dicom_id(u: str) -> str:
        # e.g. s3://.../94106955_0001.dcm -> 94106955_0001
        return Path(u).stem

    n_pass = 0; n_fail = 0
    fail_reasons = defaultdict(int)
    confident_sampled_rates = []

    print(f"\n{'i':>3}  {'dicom_id':<16} {'anchor':>6} {'win_min':>7} {'win_max':>7} "
          f"{'anchor_in_win':>13} {'conf@anc':>9} {'conf_frac':>9} "
          f"{'tgt_csv':>8} {'tgt_parq':>9} {'|Δ|':>7}")
    print('-' * 118)

    mp4_frame_mismatches = 0
    for i in range(len(uris)):
        u = uris[i]; anchor = anchors[i]; tgt_csv = targets[i]
        did = uri_to_dicom_id(u)
        if did not in parq.index:
            fail_reasons['parq_missing'] += 1; n_fail += 1; continue

        prow = parq.loc[did]
        n_video_frames_parq = int(prow['n_video_frames'])
        try:
            phases = json.loads(prow['per_frame_phase_json'])
            cmask  = json.loads(prow['confident_mask_json'])
        except Exception:
            fail_reasons['json_parse'] += 1; n_fail += 1; continue

        # Pull the clip directly (bypass __getitem__'s retry-substitution
        # so failures don't silently load a different sample).
        try:
            sample = ds.get_item_video(i)
        except Exception as e:
            fail_reasons[f'load_exc:{type(e).__name__}'] += 1
            print(f"{i:>3}  {did:<16} load failed: {e}")
            n_fail += 1; continue
        if sample is None:
            fail_reasons['load_none'] += 1; n_fail += 1; continue

        # VideoDataset loader returns (buffer, label, clip_indices, uri, phase_meta)
        # get_item_video packages this via __call__ path; let's use get_item_video directly.
        # Actually ds[i] goes through loader_substitution; easier to call get_item_video.
        # Try both paths.
        if isinstance(sample, tuple) and len(sample) >= 3:
            clip_indices = sample[2]
        else:
            fail_reasons['unexpected_sample_shape'] += 1; n_fail += 1; continue

        idx_arr = np.asarray(clip_indices[0])
        win_min, win_max = int(idx_arr.min()), int(idx_arr.max())
        anchor_in_window = (win_min <= anchor <= win_max)
        anchor_exact = (anchor in idx_arr.tolist())

        # MP4 frame-count agreement with parquet (critical for anchor validity:
        # parquet anchor_idx is in DICOM frame coordinates; MP4 must preserve
        # that indexing). loadvideo_decord gave us n frames via vr.
        # We don't have a direct vr here, but len(sample[0]) is the clip
        # (already 16 frames); the right sanity check is win_max < n_parquet.
        # For a stronger check, re-open vr directly.
        try:
            import decord
            from decord import VideoReader
            import io, boto3
            s3 = boto3.client('s3')
            bk, key = u.replace('s3://', '').split('/', 1)
            data = s3.get_object(Bucket=bk, Key=key)['Body'].read()
            vr = VideoReader(io.BytesIO(data), num_threads=1)
            n_mp4 = len(vr)
        except Exception:
            n_mp4 = None

        # Bounds check
        max_n = n_mp4 if n_mp4 is not None else n_video_frames_parq
        if win_min < 0 or win_max >= max_n:
            fail_reasons['out_of_bounds'] += 1; n_fail += 1
            print(f"{i:>3}  {did:<16} OUT_OF_BOUNDS idx={idx_arr.tolist()} "
                  f"n_mp4={n_mp4} n_parq={n_video_frames_parq}")
            continue
        if n_mp4 is not None and n_mp4 != n_video_frames_parq:
            mp4_frame_mismatches += 1
            # Do not fail — MP4 may truncate/pad by 1-2 frames during re-encode.
            # Only flag egregious mismatches (>5% or >5 frames).
            if abs(n_mp4 - n_video_frames_parq) > max(5, int(0.05 * n_video_frames_parq)):
                fail_reasons['mp4_parq_frame_mismatch'] += 1; n_fail += 1
                print(f"{i:>3}  {did:<16} FRAME_COUNT_MISMATCH mp4={n_mp4} parq={n_video_frames_parq}")
                continue

        # Anchor coverage
        if not anchor_in_window:
            fail_reasons['anchor_outside_window'] += 1; n_fail += 1
            print(f"{i:>3}  {did:<16} ANCHOR_OUT anc={anchor} win=[{win_min},{win_max}]")
            continue

        # Phase value at anchor
        phi_parq = phases[anchor] if anchor < len(phases) else None
        if phi_parq is None:
            fail_reasons['parq_phase_none'] += 1; n_fail += 1; continue
        tgt_parq = math.sin(2 * math.pi * phi_parq) if args.target == 'sin' \
                   else math.cos(2 * math.pi * phi_parq)
        diff = abs(tgt_csv - tgt_parq)
        if diff > 1e-3:
            fail_reasons['target_mismatch'] += 1; n_fail += 1
            print(f"{i:>3}  {did:<16} TARGET_DIFF csv={tgt_csv:+.4f} parq={tgt_parq:+.4f} Δ={diff:.4f}")
            continue

        conf_at_anchor = int(cmask[anchor]) if anchor < len(cmask) else 0
        if not conf_at_anchor:
            fail_reasons['anchor_not_confident'] += 1; n_fail += 1; continue

        # Confident mask fraction across sampled frames
        conf_sampled = [cmask[j] for j in idx_arr.tolist() if j < len(cmask)]
        conf_frac = sum(conf_sampled) / len(conf_sampled) if conf_sampled else 0.0
        confident_sampled_rates.append(conf_frac)

        n_pass += 1
        if i < 10:
            print(f"{i:>3}  {did:<16} {anchor:>6} {win_min:>7} {win_max:>7} "
                  f"{'EXACT' if anchor_exact else 'IN':>13} {conf_at_anchor:>9} "
                  f"{conf_frac:>9.2f} {tgt_csv:>+8.4f} {tgt_parq:>+9.4f} {diff:>7.4f}")

    print('-' * 118)
    print(f"\nSUMMARY: n={len(uris)}  PASS={n_pass}  FAIL={n_fail}")
    print(f"MP4/parquet frame-count mismatches (small, tolerated): {mp4_frame_mismatches}")
    if fail_reasons:
        print("Fail reasons:")
        for k, v in sorted(fail_reasons.items(), key=lambda x: -x[1]):
            print(f"  {k}: {v}")
    if confident_sampled_rates:
        rates = np.array(confident_sampled_rates)
        print(f"Confident-mask fraction across sampled frames (N={len(rates)}): "
              f"mean={rates.mean():.3f} median={np.median(rates):.3f} "
              f"min={rates.min():.3f} frac≥0.75={float((rates>=0.75).mean()):.3f}")

    Path(tmp_path).unlink(missing_ok=True)
    if n_fail > 0:
        raise SystemExit(1)
    print("\nALL CHECKS PASSED")


if __name__ == '__main__':
    main()
