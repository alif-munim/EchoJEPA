"""Build ``study_clip_manifest.parquet`` from MIMIC sources (plan §11.1, PR-N1).

NeurIPS EchoSet-JEPA is MIMIC-only. This module joins:

- ``classifier/phase/phase_annotations/phase_annotations.parquet`` — per-clip
  DICOM-derived metadata (525,422 rows: s3_uri, study_id, subject_id,
  n_video_frames, fps_video, video_duration_s, manufacturer, model, quality_tier
  = ECG-trace reliability, per_frame_phase_json, ...).
- ``user-default-efs/vjepa2/classifier/output/mimic_classifications.csv`` —
  per-clip view + color predictions (525,328 rows: s3_uri pointing at
  .mp4 224px, view, view_confidence, color, color_confidence).

Join key: ``dicom_id`` (extracted from each source's s3_uri). The two sources
do NOT share s3_uri directly (raw .dcm vs 224px .mp4); the dicom_id is the
stable identifier.

Filters:
  - Drop ``view ∈ {Exclude, TEE}``.
  - Drop ``quality_tier == reject`` (ECG-trace uninterpretable → phase labels
    untrustworthy; bad seed for a phase-aware pretraining objective).
  - Drop rows with ``n_video_frames <= 0`` or missing fps.

Output schema (plan §11.1, this implementation):
  patient_id, study_id, clip_id, s3_uri, dicom_series_uid, view_label,
  view_family, view_conf, modality, measurement_site, phase_label,
  phase_bucket, phase_conf, quality_score, quality_proxy_version,
  frame_rate_hz, clip_duration_s, n_frames, pixel_spacing_cm_per_px,
  acquisition_ts, site_id, vendor, cached_cclip_s3, n_duplicates,
  is_duplicate_of

Fields not available from MIMIC sources emit explicit placeholders:
  - ``dicom_series_uid``, ``pixel_spacing_cm_per_px``, ``acquisition_ts``,
    ``phase_conf`` → NaN or empty string.
  - ``n_duplicates``, ``is_duplicate_of`` → set by dedup.py (PR-1b).
  - ``cached_cclip_s3`` → set by cache_cclip.py (PR-N3).
  - ``quality_bucket`` → computed by :func:`add_quality_buckets` after a
    train/val/test split is available.

``phase_label`` is set to ``unknown`` at this stage; real per-clip phase
assignment (systolic / diastolic / full_cycle) requires parsing
``per_frame_phase_json``. That derivation happens in a follow-up because it
depends on the downstream sampler's clip-window selection (which frames of the
clip get used) — not all clips have a unique phase label.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Optional

from experiments.echoset_jepa.phase_bucket import derive_phase_buckets_batch
from experiments.echoset_jepa.quality_proxy import (
    QUALITY_PROXY_VERSION,
    quality_scores,
    quality_buckets_from_train_tertiles,
)
from experiments.echoset_jepa.taxonomy import (
    infer_measurement_site,
    is_excluded_view,
    normalize_modality,
    normalize_phase_bucket,
    normalize_view_family,
)

logger = logging.getLogger(__name__)


MANIFEST_COLUMNS = [
    "patient_id",
    "study_id",
    "clip_id",
    "s3_uri",
    "dicom_series_uid",
    "view_label",
    "view_family",
    "view_conf",
    "modality",
    "measurement_site",
    "phase_label",
    "phase_bucket",
    "phase_conf",
    "quality_score",
    "quality_proxy_version",
    "frame_rate_hz",
    "clip_duration_s",
    "n_frames",
    "pixel_spacing_cm_per_px",
    "acquisition_ts",
    "site_id",
    "vendor",
    "cached_cclip_s3",
    "n_duplicates",
    "is_duplicate_of",
]

_MIMIC_PATH_RE = re.compile(
    r"/(?P<patient>p\d+)/(?P<study>s\d+)/(?P<clip>[^/]+)\.(mp4|dcm)$"
)


def _parse_mimic_path(s3_uri: str) -> dict:
    """Pull ``patient_id`` and ``study_id`` out of a MIMIC path.

    MIMIC paths look like ``s3://echodata25/mimic-echo-224px/files/p10/p10002221/s94106955/94106955_0001.mp4``.
    """
    m = _MIMIC_PATH_RE.search(s3_uri)
    if not m:
        return {"patient_id": "", "study_id": ""}
    # The per-clip basename already includes the MIMIC study id as its prefix
    # (e.g. '94106955_0001' under 's94106955/'); use the 's'-prefixed form as
    # the canonical hashed study id.
    return {"patient_id": m.group("patient"), "study_id": m.group("study")}


def build_mimic_manifest(
    phase_annotations_parquet: str,
    mimic_classifications_csv: str,
    out_path: str,
    drop_reject_tier: bool = True,
    drop_exclude_tee: bool = True,
    site_id: str = "mimic",
) -> None:
    import pandas as pd

    logger.info("loading phase_annotations: %s", phase_annotations_parquet)
    pa = pd.read_parquet(
        phase_annotations_parquet,
        columns=[
            "dicom_id",
            "subject_id",
            "study_id",
            "s3_uri",
            "n_video_frames",
            "fps_video",
            "video_duration_s",
            "manufacturer",
            "model",
            "quality_tier",
            "sop_class_uid",
            "per_frame_phase_json",
            "confident_mask_json",
            "r_peaks_video_json",
        ],
    )
    # phase_annotations has 103 dicom_id duplicates; drop them deterministically.
    pa = pa.drop_duplicates(subset=["dicom_id"], keep="first").reset_index(drop=True)
    logger.info("phase_annotations: %d unique clips", len(pa))

    logger.info("loading mimic_classifications: %s", mimic_classifications_csv)
    mv = pd.read_csv(mimic_classifications_csv)
    mv["dicom_id"] = mv["s3_uri"].str.rsplit("/", n=1).str[-1].str.replace(".mp4", "", regex=False)
    logger.info("mimic_classifications: %d rows", len(mv))

    # Inner join on dicom_id (pa → metadata, mv → view+color+mp4 s3_uri)
    df = pa.merge(mv, on="dicom_id", how="inner", suffixes=("_dcm", ""))
    logger.info("inner join: %d rows", len(df))

    # Filters
    if drop_exclude_tee:
        before = len(df)
        df = df[~df["view"].map(is_excluded_view)].reset_index(drop=True)
        logger.info("dropped Exclude/TEE: %d → %d", before, len(df))

    if drop_reject_tier:
        before = len(df)
        df = df[df["quality_tier"].str.lower() != "reject"].reset_index(drop=True)
        logger.info("dropped quality_tier=reject: %d → %d", before, len(df))

    df = df[df["n_video_frames"].fillna(0) > 0].reset_index(drop=True)
    df = df[df["fps_video"].fillna(0) > 0].reset_index(drop=True)
    logger.info("after frame/fps sanity: %d rows", len(df))

    # --- taxonomy normalization --------------------------------------------
    df["view_label"] = df["view"].astype(str)
    df["view_conf"] = df["view_confidence"].astype(float)
    df["view_family"] = [
        normalize_view_family(v, c) for v, c in zip(df["view_label"], df["view_conf"])
    ]
    df["modality"] = [
        normalize_modality(color_flag=c, filename=f, dicom_sop_uid=s)
        for c, f, s in zip(df["color"], df["s3_uri"], df["sop_class_uid"])
    ]
    df["measurement_site"] = [
        infer_measurement_site(raw_view_label=v, modality=m, filename=f)
        for v, m, f in zip(df["view_label"], df["modality"], df["s3_uri"])
    ]
    # Per-clip phase bucket from per_frame_phase_json + confident mask + R-peaks
    # (plan §3.1; see experiments.echoset_jepa.phase_bucket).
    derived_phase = derive_phase_buckets_batch(
        df["per_frame_phase_json"].tolist(),
        df["confident_mask_json"].tolist(),
        df["r_peaks_video_json"].tolist(),
    )
    df["phase_label"] = derived_phase
    df["phase_conf"] = float("nan")   # no per-clip confidence from the derivation
    # Apply the modality gate: Doppler / M-mode / TDI ignore phase labels.
    df["phase_bucket"] = [
        normalize_phase_bucket(p, m) for p, m in zip(df["phase_label"], df["modality"])
    ]

    # --- quality proxy -----------------------------------------------------
    df["quality_score"] = quality_scores(
        view_confidences=df["view_conf"].tolist(),
        durations_s=df["video_duration_s"].tolist(),
        fps=df["fps_video"].tolist(),
        n_frames=df["n_video_frames"].astype("Int64").tolist(),
        ecg_tiers=df["quality_tier"].tolist(),
    )
    df["quality_proxy_version"] = QUALITY_PROXY_VERSION

    # --- stable ids --------------------------------------------------------
    # After the merge, the mp4-side s3_uri is `s3_uri`; the dcm-side is
    # `s3_uri_dcm`. Parse patient_id from the mp4 path; study_id comes from
    # the phase_annotations parquet (already in the canonical 's94106955' form).
    parsed = df["s3_uri"].map(_parse_mimic_path)
    df["patient_id"] = [p.get("patient_id", "") for p in parsed]
    # study_id is already present from the parquet side of the merge.
    df["study_id"] = df["study_id"].astype(str)
    df["clip_id"] = df["dicom_id"].astype(str)

    # --- misc --------------------------------------------------------------
    df["dicom_series_uid"] = ""
    df["frame_rate_hz"] = df["fps_video"].astype(float)
    df["clip_duration_s"] = df["video_duration_s"].astype(float)
    df["n_frames"] = df["n_video_frames"].astype("Int64")
    df["pixel_spacing_cm_per_px"] = float("nan")
    df["acquisition_ts"] = ""
    df["site_id"] = site_id
    df["vendor"] = df["manufacturer"].fillna("").astype(str)
    df["cached_cclip_s3"] = ""
    df["n_duplicates"] = 0
    df["is_duplicate_of"] = ""

    out = df[MANIFEST_COLUMNS].copy()
    logger.info("writing %d rows to %s", len(out), out_path)
    out.to_parquet(out_path, index=False)
    _emit_schema_report(out, str(out_path).replace(".parquet", ".schema.json"))


def _emit_schema_report(df, path: str) -> None:
    """Write a small JSON report next to the parquet: row counts + coverage."""
    report = {
        "n_rows": int(len(df)),
        "n_unique_patients": int(df["patient_id"].nunique()),
        "n_unique_studies": int(df["study_id"].nunique()),
        "view_family_counts": df["view_family"].value_counts().to_dict(),
        "modality_counts": df["modality"].value_counts().to_dict(),
        "measurement_site_counts": df["measurement_site"].value_counts().to_dict(),
        "phase_bucket_counts": df["phase_bucket"].value_counts().to_dict(),
        "site_id_counts": df["site_id"].value_counts().to_dict(),
        "vendor_counts": df["vendor"].value_counts().head(10).to_dict(),
        "quality_score_stats": {
            "mean": float(df["quality_score"].mean()),
            "std": float(df["quality_score"].std()),
            "min": float(df["quality_score"].min()),
            "max": float(df["quality_score"].max()),
        },
        "clips_per_study": {
            "median": float(df.groupby("study_id").size().median()),
            "p75": float(df.groupby("study_id").size().quantile(0.75)),
            "p95": float(df.groupby("study_id").size().quantile(0.95)),
            "max": float(df.groupby("study_id").size().max()),
        },
    }
    Path(path).write_text(json.dumps(report, indent=2))
    logger.info("schema report: %s", path)


def add_quality_buckets(manifest_path: str, train_study_ids: list[str], out_path: str) -> None:
    """Post-pass that adds a ``quality_bucket`` column from train tertiles.

    Train/val/test split is decided outside this script; callers pass in the
    list of train study_ids. Tertile thresholds are computed on rows whose
    study_id is in train_study_ids, then applied to every row.
    """
    import pandas as pd

    df = pd.read_parquet(manifest_path)
    train_set = set(train_study_ids)
    mask = df["study_id"].isin(train_set).tolist()
    df["quality_bucket"] = quality_buckets_from_train_tertiles(
        df["quality_score"].tolist(), mask
    )
    df.to_parquet(out_path, index=False)
    logger.info("wrote %d rows with quality_bucket column → %s", len(df), out_path)


def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase_annotations", required=True)
    ap.add_argument("--classifications_csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep_reject_tier", action="store_true")
    ap.add_argument("--keep_tee_exclude", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build_mimic_manifest(
        phase_annotations_parquet=args.phase_annotations,
        mimic_classifications_csv=args.classifications_csv,
        out_path=args.out,
        drop_reject_tier=not args.keep_reject_tier,
        drop_exclude_tee=not args.keep_tee_exclude,
    )


if __name__ == "__main__":
    _main()


__all__ = [
    "MANIFEST_COLUMNS",
    "build_mimic_manifest",
    "add_quality_buckets",
]
