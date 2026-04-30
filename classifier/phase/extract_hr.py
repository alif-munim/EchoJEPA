#!/usr/bin/env python3
"""Extract all available DICOM metadata for every .dcm in phase/dicoms/.

Writes `phase/dicom_metadata.csv` with one row per DICOM and one column per
DICOM keyword seen across the input files. `heart_rate` is the primary
column of interest (tag 0018,1088, in bpm) — kept up front for easy
comparison against waveform-inferred HR later — but every scalar tag is
included so we can slice by manufacturer, view, resolution, etc.

Design choices
--------------
- Pixel data is skipped (`stop_before_pixels=True`) so this is fast.
- Sequence-valued tags (VR == 'SQ') are flattened to a short summary of
  child keys (e.g. `SequenceOfUltrasoundRegions=[RegionSpatialFormat=1,...]`)
  since a CSV can't faithfully hold nested structure.
- Byte-valued tags are omitted from the CSV (they're useless as text).
- Columns are the union of keywords across all files; missing values are
  left empty in the CSV.
"""

import argparse
import csv
from pathlib import Path

import pydicom

HERE = Path(__file__).resolve().parent
DICOM_DIR = HERE / "dicoms"
OUT_CSV = HERE / "dicom_metadata.csv"

# Columns we want pinned to the left of the CSV for readability.
LEADING_COLUMNS = [
    "dicom",
    "heart_rate",
    "n_frames",
    "frame_time_ms",
    "cine_rate",
    "nominal_interval_ms",
    "manufacturer",
    "model",
    "modality",
    "sop_class_uid",
    "rows",
    "columns",
    "study_date",
    "study_time",
    "series_number",
    "instance_number",
    "photometric_interpretation",
    "error",
]


def _elem_value(elem) -> str:
    """Render a single DICOM element's value as a CSV-safe string."""
    if elem.VR == "SQ":
        # Summarize a sequence with the set of keys in its first item.
        try:
            if len(elem.value) == 0:
                return "<empty SQ>"
            first = elem.value[0]
            kv = ";".join(
                f"{sub.keyword or sub.name}={_elem_value(sub)}"
                for sub in first
                if sub.VR != "SQ" and sub.VR not in ("OB", "OW", "UN")
            )
            return f"[{kv}]" if len(elem.value) == 1 else f"[{kv}]*{len(elem.value)}"
        except Exception:
            return "<SQ>"

    if elem.VR in ("OB", "OW", "UN"):
        return ""  # binary blob

    v = elem.value
    if isinstance(v, bytes):
        return ""
    if isinstance(v, (list, tuple, pydicom.multival.MultiValue)):
        return "\\".join(str(x) for x in v)
    return str(v)


def read_metadata(dcm_path: Path) -> dict:
    try:
        ds = pydicom.dcmread(str(dcm_path), stop_before_pixels=True)
    except Exception as e:
        return {"dicom": dcm_path.name, "error": str(e)[:200]}

    row: dict = {"dicom": dcm_path.name}

    # Pinned columns with friendly names (null-safe getattr).
    row["heart_rate"] = getattr(ds, "HeartRate", "")
    row["n_frames"] = int(getattr(ds, "NumberOfFrames", 1))
    row["frame_time_ms"] = getattr(ds, "FrameTime", "")
    row["cine_rate"] = getattr(ds, "CineRate", "")
    row["nominal_interval_ms"] = getattr(ds, "NominalInterval", "")
    row["manufacturer"] = getattr(ds, "Manufacturer", "")
    row["model"] = getattr(ds, "ManufacturerModelName", "")
    row["modality"] = getattr(ds, "Modality", "")
    row["sop_class_uid"] = str(getattr(ds, "SOPClassUID", ""))
    row["rows"] = getattr(ds, "Rows", "")
    row["columns"] = getattr(ds, "Columns", "")
    row["study_date"] = getattr(ds, "StudyDate", "")
    row["study_time"] = getattr(ds, "StudyTime", "")
    row["series_number"] = getattr(ds, "SeriesNumber", "")
    row["instance_number"] = getattr(ds, "InstanceNumber", "")
    row["photometric_interpretation"] = getattr(ds, "PhotometricInterpretation", "")

    # All remaining top-level tags, keyed by DICOM keyword (camelCase).
    # Any collisions with pinned columns are preserved under the keyword
    # name — the pinned column is the "clean" version for humans.
    for elem in ds:
        kw = elem.keyword
        if not kw:
            continue
        if kw in row:
            continue
        row[kw] = _elem_value(elem)

    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dicom-dir", type=Path, default=DICOM_DIR)
    ap.add_argument("-o", "--out", type=Path, default=OUT_CSV)
    args = ap.parse_args()

    dcms = sorted(args.dicom_dir.glob("*.dcm"))
    print(f"Reading metadata from {len(dcms)} DICOMs → {args.out}")

    rows = [read_metadata(p) for p in dcms]

    # Build the union of columns across all rows, with leading columns pinned.
    all_keys = set().union(*(r.keys() for r in rows))
    trailing = sorted(k for k in all_keys if k not in LEADING_COLUMNS)
    fieldnames = [k for k in LEADING_COLUMNS if k in all_keys] + trailing

    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    n_hr = sum(1 for r in rows if r.get("heart_rate") not in ("", None))
    n_err = sum(1 for r in rows if r.get("error"))
    print(f"Done: {len(rows)} rows × {len(fieldnames)} columns, "
          f"{n_hr} with HR, {n_err} read errors")


if __name__ == "__main__":
    main()
