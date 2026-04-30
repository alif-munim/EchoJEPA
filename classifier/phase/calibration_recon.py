#!/usr/bin/env python3
"""Reconnaissance script: characterize calibration-source availability across a
DICOM directory. Output is purely diagnostic — no sampling rates are computed.

Checks, in priority order:
  1. `PhysicalDeltaX` on an ECG region inside `SequenceOfUltrasoundRegions`
     (ideal — manufacturer-provided, per-clip, HR-independent).
  2. `WaveformSequence` at top level (digital ECG, bypasses visual extraction).
  3. Scanner identification (`Manufacturer`, `ManufacturerModelName`) so
     we know whether per-scanner defaults are a viable fallback.
  4. Cross-metadata consistency (HeartRate × NominalInterval ≈ 60000,
     FrameTime × CineRate ≈ 1000) — a sanity check that the scanner is
     writing coherent metadata.

Ultrasound region spatial format codes (DICOM PS3.3 C.8.5.5):
  1 = 2D image, 2 = M-Mode, 3 = Spectral Doppler, 4 = Waveform, 5 = Graphics.
An ECG waveform region will typically have ``RegionSpatialFormat=4`` and
``PhysicalUnitsXDirection=4`` (seconds).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pydicom

ECG_SR_RANGE_HZ = (100.0, 2000.0)  # sanity range for implied sampling rates


def _inspect_regions(ds) -> list[dict]:
    """Summarize each ultrasound region in the dataset."""
    out: list[dict] = []
    if "SequenceOfUltrasoundRegions" not in ds:
        return out
    for i, r in enumerate(ds.SequenceOfUltrasoundRegions):
        info = {
            "idx": i,
            "spatial_format": getattr(r, "RegionSpatialFormat", None),
            "data_type": getattr(r, "RegionDataType", None),
            "units_x": getattr(r, "PhysicalUnitsXDirection", None),
            "units_y": getattr(r, "PhysicalUnitsYDirection", None),
            "delta_x": getattr(r, "PhysicalDeltaX", None),
            "delta_y": getattr(r, "PhysicalDeltaY", None),
            "x0": getattr(r, "RegionLocationMinX0", None),
            "y0": getattr(r, "RegionLocationMinY0", None),
            "x1": getattr(r, "RegionLocationMaxX1", None),
            "y1": getattr(r, "RegionLocationMaxY1", None),
        }
        out.append(info)
    return out


def _region_is_ecg_candidate(region: dict) -> bool:
    """A region is an ECG candidate if it's marked as waveform/spectral OR
    if its X units are seconds. RegionSpatialFormat==4 is the cleanest
    indicator; PhysicalUnitsXDirection==4 (seconds) is also accepted."""
    sf = region.get("spatial_format")
    ux = region.get("units_x")
    if sf == 4:
        return True
    if ux == 4:
        return True
    return False


def analyze_calibration_availability(dicom_dir: Path) -> dict:
    dcms = sorted(dicom_dir.glob("*.dcm"))
    n_total = len(dcms)

    px_n_ecg_region = 0
    px_n_populated = 0
    delta_x_values: list[float] = []
    implied_sr: list[float] = []
    region_samples: list[dict] = []

    wf_n = 0
    wf_channel_defs: list[dict] = []

    manu = Counter()
    models = Counter()

    hr_nominal_ok = 0
    hr_nominal_total = 0
    frame_cine_ok = 0
    frame_cine_total = 0

    per_scanner_px = Counter()

    errors: list[tuple[str, str]] = []

    for p in dcms:
        try:
            ds = pydicom.dcmread(str(p), stop_before_pixels=True)
        except Exception as e:
            errors.append((p.name, str(e)[:120]))
            continue

        manufacturer = str(getattr(ds, "Manufacturer", "") or "")
        model = str(getattr(ds, "ManufacturerModelName", "") or "")
        manu[manufacturer] += 1
        models[model] += 1

        # --- PhysicalDeltaX ---
        regions = _inspect_regions(ds)
        has_ecg_region = False
        for r in regions:
            if _region_is_ecg_candidate(r):
                has_ecg_region = True
                dx = r.get("delta_x")
                if dx is not None and dx > 0:
                    px_n_populated += 1
                    dx_f = float(dx)
                    delta_x_values.append(dx_f)
                    implied_sr.append(1.0 / dx_f)
                    per_scanner_px[f"{manufacturer}|{model}"] += 1
                    if len(region_samples) < 10:
                        region_samples.append({"dicom": p.name, **r})
                    break
        if has_ecg_region:
            px_n_ecg_region += 1

        # --- WaveformSequence ---
        if "WaveformSequence" in ds:
            wf_n += 1
            try:
                for w in ds.WaveformSequence:
                    fs = getattr(w, "SamplingFrequency", None)
                    nch = getattr(w, "NumberOfWaveformChannels", None)
                    nsamp = getattr(w, "NumberOfWaveformSamples", None)
                    channels = []
                    if "ChannelDefinitionSequence" in w:
                        for ch in w.ChannelDefinitionSequence:
                            src = getattr(ch, "ChannelSourceSequence", None)
                            ch_name = ""
                            if src and len(src) > 0:
                                ch_name = getattr(src[0], "CodeMeaning", "") or ""
                            channels.append(ch_name)
                    wf_channel_defs.append({
                        "dicom": p.name,
                        "sampling_frequency": fs,
                        "n_channels": nch,
                        "n_samples": nsamp,
                        "channels": channels,
                    })
            except Exception as e:
                errors.append((p.name, f"WaveformSeq: {e}"))

        # --- consistency checks ---
        hr = getattr(ds, "HeartRate", None)
        nominal_ms = getattr(ds, "NominalInterval", None)
        if hr and nominal_ms:
            hr_nominal_total += 1
            try:
                product = float(hr) * float(nominal_ms)
                if 50000 < product < 72000:  # ±10% of 60000
                    hr_nominal_ok += 1
            except Exception:
                pass

        frame_ms = getattr(ds, "FrameTime", None)
        cine = getattr(ds, "CineRate", None)
        if frame_ms and cine:
            frame_cine_total += 1
            try:
                product = float(frame_ms) * float(cine)
                if 900 < product < 1100:
                    frame_cine_ok += 1
            except Exception:
                pass

    result: dict[str, Any] = {
        "n_total": n_total,
        "physical_delta_x": {
            "n_with_ecg_region": px_n_ecg_region,
            "n_with_populated_delta_x": px_n_populated,
            "delta_x_values": delta_x_values,
            "implied_sampling_rates_hz": implied_sr,
            "sample_regions": region_samples,
            "per_scanner": dict(per_scanner_px),
        },
        "waveform_sequence": {
            "n_with_waveform": wf_n,
            "channel_definitions": wf_channel_defs[:20],
        },
        "scanner_distribution": {
            "manufacturers": dict(manu),
            "models": dict(models),
        },
        "metadata_consistency": {
            "n_hr_nominal_total": hr_nominal_total,
            "n_hr_nominal_consistent": hr_nominal_ok,
            "n_frame_cine_total": frame_cine_total,
            "n_frame_cine_consistent": frame_cine_ok,
        },
        "read_errors": errors,
    }
    return result


def _summarize(r: dict) -> None:
    print(f"\n=== Calibration reconnaissance over {r['n_total']} DICOMs ===\n")

    px = r["physical_delta_x"]
    print(f"PATH 1 — PhysicalDeltaX (preferred):")
    print(f"  clips with any ECG/waveform region:  {px['n_with_ecg_region']:>4d} / {r['n_total']}")
    print(f"  clips with populated PhysicalDeltaX: {px['n_with_populated_delta_x']:>4d} / {r['n_total']}")
    if px["implied_sampling_rates_hz"]:
        import numpy as np
        sr = np.asarray(px["implied_sampling_rates_hz"])
        in_range = ((sr >= ECG_SR_RANGE_HZ[0]) & (sr <= ECG_SR_RANGE_HZ[1])).sum()
        print(f"  implied SR range (Hz): min={sr.min():.1f}  med={np.median(sr):.1f}  max={sr.max():.1f}")
        print(f"  SR in sane 100–2000 Hz range: {in_range} / {len(sr)}")
        print(f"  DeltaX range (s/px): min={min(px['delta_x_values']):.6f}  med={np.median(px['delta_x_values']):.6f}  max={max(px['delta_x_values']):.6f}")
    if px["sample_regions"]:
        print(f"  sample regions (first 3):")
        for s in px["sample_regions"][:3]:
            print(f"    {s['dicom']}  sf={s['spatial_format']} ux={s['units_x']} dx={s['delta_x']}")

    wf = r["waveform_sequence"]
    print(f"\nPATH 2 — WaveformSequence:")
    print(f"  clips with WaveformSequence: {wf['n_with_waveform']:>4d} / {r['n_total']}")
    if wf["channel_definitions"]:
        print("  first few channel definitions:")
        for c in wf["channel_definitions"][:3]:
            print(f"    {c}")

    sc = r["scanner_distribution"]
    print(f"\nPATH 3 — Scanner identification:")
    print(f"  Manufacturers: {dict(sorted(sc['manufacturers'].items(), key=lambda x: -x[1])[:5])}")
    print(f"  Top 5 models: {dict(sorted(sc['models'].items(), key=lambda x: -x[1])[:5])}")

    mc = r["metadata_consistency"]
    print(f"\nMetadata consistency checks:")
    print(f"  HeartRate×NominalInterval within ±10% of 60000 ms: "
          f"{mc['n_hr_nominal_consistent']} / {mc['n_hr_nominal_total']}")
    print(f"  FrameTime×CineRate within ±10% of 1000 ms: "
          f"{mc['n_frame_cine_consistent']} / {mc['n_frame_cine_total']}")

    if r["read_errors"]:
        print(f"\nRead errors on {len(r['read_errors'])} files (first 5):")
        for e in r["read_errors"][:5]:
            print(f"  {e}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dicom-dir", type=Path,
                    default=Path(__file__).resolve().parent / "dicoms")
    ap.add_argument("-o", "--out-json", type=Path, default=None,
                    help="Optional path to dump the full result as JSON.")
    args = ap.parse_args()

    result = analyze_calibration_availability(args.dicom_dir)
    _summarize(result)
    if args.out_json:
        args.out_json.write_text(json.dumps(result, indent=2, default=str))
        print(f"\nWrote {args.out_json}")


if __name__ == "__main__":
    main()
