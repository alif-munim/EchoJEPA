#!/usr/bin/env python3
"""Determine per-clip ECG sampling rate (Hz) from DICOM metadata.

Calibration sources, tried in priority order:

  1. **WaveformSequence** (high confidence) — if present, reads
     `SamplingFrequency`. Bypasses the image entirely. Empirically absent
     in MIMIC-IV-Echo but cheap to check.

  2. **Scanner default** (medium confidence) — lookup by
     `(Manufacturer, ManufacturerModelName)` against `scanner_defaults.json`.
     Defaults were derived empirically from Phase 3 validation across 142
     clips by back-solving SR from HR matching, then taking the modal
     scanner-wise value. This *is* circular for HR-based validation on the
     same clips used to derive it, but independent on held-out clips and
     on non-HR quality signals (RR CV, R-wave count, hand-label F1).

  3. **Fallback** (low confidence) — returns None.

Note: `PhysicalDeltaX` on `SequenceOfUltrasoundRegions` was considered but
is **not** a valid source for the burned-in ECG strip. Phase 1 recon found
PhysicalDeltaX is populated only on spectral-Doppler stills, where it
describes the *spectrum's* time axis, not the ECG overlay's. Using it as
the ECG sampling rate gave a systematic ~1.5–2× HR mismatch on a first
pass. The cleanest signal in MIMIC-IV-Echo is the empirical scanner
default derived from HR back-solve.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pydicom

HERE = Path(__file__).resolve().parent
SCANNER_DEFAULTS_PATH = HERE / "scanner_defaults.json"


# --- helpers ----------------------------------------------------------------

def _try_physical_delta_x(ds) -> dict | None:
    """Path 1 — look for a seconds-domain region in SequenceOfUltrasoundRegions."""
    if "SequenceOfUltrasoundRegions" not in ds:
        return None
    for i, r in enumerate(ds.SequenceOfUltrasoundRegions):
        sf = getattr(r, "RegionSpatialFormat", None)
        ux = getattr(r, "PhysicalUnitsXDirection", None)
        dx = getattr(r, "PhysicalDeltaX", None)
        if (sf == 4 or ux == 4) and dx and dx > 0:
            return {
                "sampling_rate_hz": 1.0 / float(dx),
                "source": "physical_delta_x",
                "confidence": "high",
                "details": {
                    "region_index": i,
                    "spatial_format": sf,
                    "units_x": ux,
                    "delta_x_s_per_px": float(dx),
                },
            }
    return None


def _try_waveform_sequence(ds) -> dict | None:
    """Path 2 — read SamplingFrequency from a WaveformSequence item if present."""
    if "WaveformSequence" not in ds:
        return None
    for i, w in enumerate(ds.WaveformSequence):
        fs = getattr(w, "SamplingFrequency", None)
        if fs and fs > 0:
            return {
                "sampling_rate_hz": float(fs),
                "source": "waveform_sequence",
                "confidence": "high",
                "details": {
                    "waveform_index": i,
                    "n_channels": getattr(w, "NumberOfWaveformChannels", None),
                    "n_samples": getattr(w, "NumberOfWaveformSamples", None),
                },
            }
    return None


def _try_scanner_default(ds, defaults: dict) -> dict | None:
    """Path 3 — look up a per-scanner default Hz."""
    manu = str(getattr(ds, "Manufacturer", "") or "")
    model = str(getattr(ds, "ManufacturerModelName", "") or "")
    key = f"{manu}|{model}"
    entry = defaults.get(key) or defaults.get(f"{manu}|*")
    if not entry:
        return None
    return {
        "sampling_rate_hz": float(entry["sampling_rate_hz"]),
        "source": "scanner_default",
        "confidence": "medium",
        "details": {
            "manufacturer": manu,
            "model": model,
            "key": key,
            "n_supporting": entry.get("n_supporting"),
            "notes": entry.get("notes"),
        },
    }


def load_scanner_defaults(path: Path = SCANNER_DEFAULTS_PATH) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}


# --- main API ---------------------------------------------------------------

def calibrate_sampling_rate(
    dicom_path: Path,
    scanner_defaults: dict | None = None,
) -> dict:
    """Determine sampling rate for a single clip. See module docstring."""
    if scanner_defaults is None:
        scanner_defaults = load_scanner_defaults()

    try:
        ds = pydicom.dcmread(str(dicom_path), stop_before_pixels=True)
    except Exception as e:
        return {
            "sampling_rate_hz": None,
            "source": "fallback",
            "confidence": "low",
            "details": {"error": str(e)[:200]},
        }

    r = _try_waveform_sequence(ds)
    if r is not None:
        return r

    r = _try_scanner_default(ds, scanner_defaults)
    if r is not None:
        return r

    return {
        "sampling_rate_hz": None,
        "source": "fallback",
        "confidence": "low",
        "details": {
            "manufacturer": str(getattr(ds, "Manufacturer", "") or ""),
            "model": str(getattr(ds, "ManufacturerModelName", "") or ""),
        },
    }


# --- scanner-defaults-from-recon helper -------------------------------------

def build_scanner_defaults_from_dir(
    dicom_dir: Path,
    out_path: Path = SCANNER_DEFAULTS_PATH,
) -> dict:
    """Scan a DICOM directory, extract PhysicalDeltaX for every clip that has
    a seconds-domain region, and write per-scanner modal sampling rate to
    ``scanner_defaults.json``. Intended for bootstrapping from Phase 1 recon.
    """
    from collections import Counter, defaultdict

    by_key: dict[str, list[float]] = defaultdict(list)
    for p in sorted(dicom_dir.glob("*.dcm")):
        try:
            ds = pydicom.dcmread(str(p), stop_before_pixels=True)
        except Exception:
            continue
        if "SequenceOfUltrasoundRegions" not in ds:
            continue
        manu = str(getattr(ds, "Manufacturer", "") or "")
        model = str(getattr(ds, "ManufacturerModelName", "") or "")
        key = f"{manu}|{model}"
        for r in ds.SequenceOfUltrasoundRegions:
            sf = getattr(r, "RegionSpatialFormat", None)
            ux = getattr(r, "PhysicalUnitsXDirection", None)
            dx = getattr(r, "PhysicalDeltaX", None)
            if (sf == 4 or ux == 4) and dx and dx > 0:
                by_key[key].append(1.0 / float(dx))
                break

    defaults: dict[str, dict] = {}
    for key, srs in by_key.items():
        # Modal rounded to integer Hz.
        counts = Counter(round(s) for s in srs)
        modal_hz, modal_n = counts.most_common(1)[0]
        defaults[key] = {
            "sampling_rate_hz": float(modal_hz),
            "n_supporting": modal_n,
            "n_total_for_scanner": len(srs),
            "all_unique_hz": sorted(set(round(s, 1) for s in srs)),
            "notes": "derived from same-scanner spectral-still PhysicalDeltaX "
                     "(modal). Applies to cines under the assumption that the "
                     "scanner uses the same time-per-pixel for ECG display in "
                     "both spectral and cine modes.",
        }

    out_path.write_text(json.dumps(defaults, indent=2))
    return defaults


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Bootstrap scanner_defaults.json from same-scanner spectral "
                    "PhysicalDeltaX. NOTE: this is a first-pass estimate; the final "
                    "values in scanner_defaults.json should be refined via Phase 3 "
                    "HR-match validation — see run_calibration_batch.py + "
                    "validate_pipeline.py.")
    ap.add_argument("--dicom-dir", type=Path,
                    default=HERE / "dicoms")
    ap.add_argument("-o", "--out", type=Path, default=SCANNER_DEFAULTS_PATH)
    args = ap.parse_args()

    defaults = build_scanner_defaults_from_dir(args.dicom_dir, args.out)
    print(f"Wrote scanner defaults for {len(defaults)} scanners → {args.out}")
    print(json.dumps(defaults, indent=2))


if __name__ == "__main__":
    main()
