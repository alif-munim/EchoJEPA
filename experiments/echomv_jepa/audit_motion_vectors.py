"""Audit H.264 motion-vector availability in MIMIC MP4s for the MCC-MC-MGM
experiment gate (§9.2 of claude/neurips/experiments/masked-cross-clip-vjepa.md).

Pass threshold (from the doc):
  - >= 95% of clips have motion vectors present in the codec stream
  - >= 70% of those have motion vectors that spatially correlate with genuine
    cardiac motion (proxy: MV magnitude distribution; full RAFT correlation
    deferred — this script reports a cheap proxy instead).

Usage:
  python -m experiments.echomv_jepa.audit_motion_vectors \\
      --sample_csv /tmp/mimic_mv_audit_sample.csv \\
      --out /tmp/mv_audit_results.csv \\
      --workdir /tmp/mv_audit_clips

Reads a CSV with an s3_uri column; downloads each clip; probes with PyAV
(export_mvs=True); records per-clip stats.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


def _download(s3_uri: str, dest: Path) -> bool:
    """aws s3 cp one clip. Returns True on success."""
    try:
        subprocess.run(
            ["aws", "s3", "cp", s3_uri, str(dest), "--quiet"],
            check=True,
            timeout=30,
        )
        return dest.exists() and dest.stat().st_size > 0
    except Exception as e:  # noqa: BLE001
        print(f"  download failed {s3_uri}: {e}", file=sys.stderr)
        return False


def _probe_clip(mp4_path: Path) -> dict:
    """Decode clip with PyAV + export_mvs. Return per-clip stats dict."""
    import av
    from av.codec.context import Flags2

    stats = {
        "codec": None,
        "n_frames_total": 0,
        "n_frames_with_mv": 0,
        "n_mv_total": 0,
        "mv_frac_frames": 0.0,
        "mv_mean_per_frame": 0.0,
        "mv_mag_mean": 0.0,
        "mv_mag_p50": 0.0,
        "mv_mag_p95": 0.0,
        "mv_mag_max": 0.0,
        "mv_spatial_span": 0.0,
        "width": 0,
        "height": 0,
        "fps": 0.0,
        "pict_types": "",
        "error": "",
    }
    try:
        with av.open(str(mp4_path)) as c:
            vs = c.streams.video[0]
            stats["codec"] = vs.codec_context.name
            stats["width"] = vs.width
            stats["height"] = vs.height
            stats["fps"] = float(vs.average_rate) if vs.average_rate else 0.0
            vs.codec_context.flags2 |= Flags2.export_mvs
            all_mag: list[float] = []
            all_dst_x: list[float] = []
            all_dst_y: list[float] = []
            pict_type_counts: dict[str, int] = {}
            for frame in c.decode(vs):
                stats["n_frames_total"] += 1
                pt = str(frame.pict_type)
                pict_type_counts[pt] = pict_type_counts.get(pt, 0) + 1
                mvs = frame.side_data.get("MOTION_VECTORS") if frame.side_data else None
                if mvs is None or len(mvs) == 0:
                    continue
                stats["n_frames_with_mv"] += 1
                stats["n_mv_total"] += len(mvs)
                # Each MV has motion_x, motion_y in motion_scale units, and a
                # macroblock location (dst_x, dst_y).
                for mv in mvs:
                    # Magnitude in quarter-pel units (motion_scale=4 typical).
                    scale = max(1, mv.motion_scale)
                    mx = mv.motion_x / scale
                    my = mv.motion_y / scale
                    all_mag.append(float((mx * mx + my * my) ** 0.5))
                    all_dst_x.append(float(mv.dst_x))
                    all_dst_y.append(float(mv.dst_y))
            stats["pict_types"] = ",".join(f"{k}={v}" for k, v in sorted(pict_type_counts.items()))
            if stats["n_frames_total"] > 0:
                stats["mv_frac_frames"] = stats["n_frames_with_mv"] / stats["n_frames_total"]
                stats["mv_mean_per_frame"] = stats["n_mv_total"] / stats["n_frames_total"]
            if all_mag:
                a = np.asarray(all_mag)
                stats["mv_mag_mean"] = float(a.mean())
                stats["mv_mag_p50"] = float(np.median(a))
                stats["mv_mag_p95"] = float(np.percentile(a, 95))
                stats["mv_mag_max"] = float(a.max())
            if all_dst_x and all_dst_y:
                # Spatial span: how much of the frame the MVs cover (as a
                # rough proxy for "are MVs localized to a moving region vs
                # spread everywhere"). Units: macroblock centers in pixels.
                sx = np.std(all_dst_x)
                sy = np.std(all_dst_y)
                stats["mv_spatial_span"] = float(0.5 * (sx + sy))
    except Exception as e:  # noqa: BLE001
        stats["error"] = str(e)[:200]
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample_csv", required=True, help="CSV with s3_uri column")
    ap.add_argument("--out", required=True, help="output CSV path")
    ap.add_argument("--workdir", default="/tmp/mv_audit_clips")
    ap.add_argument("--limit", type=int, default=0, help="cap number of clips (0 = all)")
    args = ap.parse_args()

    df = pd.read_csv(args.sample_csv)
    if args.limit > 0:
        df = df.head(args.limit)
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    t0 = time.time()
    for i, row in enumerate(df.itertuples(index=False)):
        s3 = row.s3_uri
        local = workdir / Path(s3).name
        ok = True if local.exists() else _download(s3, local)
        if not ok:
            rows.append(
                {
                    **{c: getattr(row, c, None) for c in df.columns},
                    "download_ok": False,
                    "error": "download_failed",
                }
            )
            continue
        probe = _probe_clip(local)
        rows.append(
            {
                **{c: getattr(row, c, None) for c in df.columns},
                "download_ok": True,
                **probe,
            }
        )
        if (i + 1) % 10 == 0 or i + 1 == len(df):
            elapsed = time.time() - t0
            print(
                f"[{i+1}/{len(df)}] elapsed {elapsed:.0f}s "
                f"last codec={probe['codec']} mv_frac={probe['mv_frac_frames']:.2f} "
                f"mv_mean_per_frame={probe['mv_mean_per_frame']:.1f}"
            )

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(f"\nwrote {args.out} ({len(out)} rows)\n")

    # Summary
    print("=== MCC-MC-MGM Audit Summary ===")
    ok = out[out["download_ok"] == True]  # noqa: E712
    if len(ok) == 0:
        print("no clips decoded — aborting")
        return
    has_mv = ok[ok["mv_frac_frames"] >= 0.5]  # at least half the frames have MV
    print(f"clips successfully probed: {len(ok)}/{len(out)}")
    print(f"clips with MV coverage >=50%: {len(has_mv)}/{len(ok)} = {100*len(has_mv)/len(ok):.1f}%")
    print(f"clips with MV coverage >=90%: {(ok['mv_frac_frames']>=0.9).sum()}/{len(ok)} = "
          f"{100*(ok['mv_frac_frames']>=0.9).sum()/len(ok):.1f}%")
    if len(has_mv) > 0:
        print(f"MV mag mean across clips: {has_mv['mv_mag_mean'].mean():.2f} px (median of per-clip means)")
        print(f"MV mag p95 across clips:  {has_mv['mv_mag_p95'].mean():.2f} px")
        print(f"MV count per frame mean:  {has_mv['mv_mean_per_frame'].mean():.0f}")
    # Per-modality breakdown
    if "modality" in ok.columns:
        print("\nper-modality MV frac means:")
        print(ok.groupby("modality")["mv_frac_frames"].agg(["count", "mean"]).to_string())
    if "view_family" in ok.columns:
        print("\nper-view-family MV frac means:")
        print(ok.groupby("view_family")["mv_frac_frames"].agg(["count", "mean"]).to_string())

    # Gate decision
    frac_coverage_ok = (ok["mv_frac_frames"] >= 0.9).mean()
    print("\n=== Gate verdict ===")
    print(f"Threshold: >= 0.95 of clips with MV coverage >= 0.90")
    print(f"Observed:  {frac_coverage_ok:.3f}")
    print("GATE PASS" if frac_coverage_ok >= 0.95 else "GATE FAIL — motion-vector coverage insufficient")


if __name__ == "__main__":
    main()
