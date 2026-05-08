"""Taxonomy normalization for EchoSet-JEPA.

Separates three orthogonal axes that the v2 plan conflated:

- ``view_family``       — where the probe sits (anatomic window)
- ``modality``          — how the image is formed (B-mode / Doppler / M-mode / TDI / contrast)
- ``measurement_site``  — what anatomic structure the clip targets (TR / MV / LVOT / etc.)

Previously ``element_grouping.view_family()`` mapped spectral Doppler / M-mode /
TDI into ``view_family``; those are *modalities*, not views. A TR CW Doppler
clip should be ``view_family=apical, modality=cw_doppler, measurement_site=TR``.
"""

from __future__ import annotations

from typing import Optional, Tuple

# ---------------------------------------------------------------------------
# Vocabularies — keep in sync with src/models/meta_embeddings.py.
# ---------------------------------------------------------------------------

VIEW_FAMILIES: Tuple[str, ...] = (
    "apical",
    "parasternal_long",
    "parasternal_short",
    "rv_focused",
    "subcostal",
    "suprasternal",
    "unknown",
)

MODALITIES: Tuple[str, ...] = (
    "b_mode",
    "color_doppler",
    "pw_doppler",
    "cw_doppler",
    "m_mode",
    "tdi",
    "contrast",
    "unknown",
)

MEASUREMENT_SITES: Tuple[str, ...] = (
    "none",
    "TR",
    "MV_inflow",
    "LVOT",
    "AV",
    "MV",
    "TV",
    "septal_annulus",
    "lateral_annulus",
    "IVC",
    "unknown",
)

PHASE_BUCKETS: Tuple[str, ...] = (
    "systolic",
    "diastolic",
    "full_cycle",
    "not_applicable",
    "unknown",
)

# ---------------------------------------------------------------------------
# View family normalization.
# ---------------------------------------------------------------------------

# Maps raw labels from the MIMIC view classifier (see
# /home/sagemaker-user/user-default-efs/vjepa2/classifier/output/mimic_classifications.csv)
# to the canonical view_family vocabulary. Any label not in this map falls back
# to ``unknown``.
_VIEW_LABEL_TO_FAMILY = {
    "A2C": "apical",
    "A3C": "apical",
    "A4C": "apical",
    "A5C": "apical",
    "APICAL": "apical",
    "PLAX": "parasternal_long",
    "PSAX": "parasternal_short",
    "PSAX-AP": "parasternal_short",
    "PSAX-AV": "parasternal_short",
    "PSAX-MV": "parasternal_short",
    "PSAX-PM": "parasternal_short",
    "PSAX-PML": "parasternal_short",
    "SUBCOSTAL": "subcostal",
    "SUBCOSTAL-IVC": "subcostal",
    "IVC": "subcostal",
    "SSN": "suprasternal",
    "RV-FOCUSED": "rv_focused",
    "RVIT": "rv_focused",  # RV inflow tract
}

# Labels we explicitly drop rather than route to ``unknown``. MIMIC's view
# classifier emits ``Exclude`` for non-cardiac frames and ``TEE`` for
# transesophageal; EchoSet-JEPA v1 is transthoracic only.
EXCLUDED_VIEW_LABELS = frozenset({"EXCLUDE", "TEE"})


def normalize_view_family(raw_view_label: Optional[str], view_conf: Optional[float] = None,
                          conf_high: float = 0.7, conf_low: float = 0.4) -> str:
    """Return a canonical view_family.

    Rules (plan §11.5):
      - ``view_conf >= conf_high``    → use the label
      - ``conf_low <= view_conf <  conf_high`` → use the label (dataset-side
        probabilistic dropout to ``<unknown>`` is a separate concern, handled
        at the dataloader via ``MetaDropout``, not here)
      - ``view_conf <  conf_low``     → ``unknown``

    Excluded labels (EXCLUDE, TEE) return ``unknown`` — callers are expected to
    have already dropped those rows before this function sees them. They still
    map to ``unknown`` as a defensive default.
    """
    if raw_view_label is None:
        return "unknown"
    up = str(raw_view_label).strip().upper()
    if not up or up in EXCLUDED_VIEW_LABELS:
        return "unknown"
    if view_conf is not None and view_conf < conf_low:
        return "unknown"
    return _VIEW_LABEL_TO_FAMILY.get(up, "unknown")


def is_excluded_view(raw_view_label: Optional[str]) -> bool:
    """True iff the row should be dropped before building the manifest."""
    if raw_view_label is None:
        return False
    return str(raw_view_label).strip().upper() in EXCLUDED_VIEW_LABELS


# ---------------------------------------------------------------------------
# Modality normalization.
# ---------------------------------------------------------------------------


def normalize_modality(
    raw_modality: Optional[str] = None,
    color_flag: Optional[str] = None,
    dicom_sop_uid: Optional[str] = None,
    filename: Optional[str] = None,
) -> str:
    """Return a canonical modality.

    Decision order:
      1. If ``raw_modality`` is already a canonical value, return it.
      2. If ``color_flag == 'Yes'`` (the MIMIC color classifier output),
         return ``color_doppler``. This is coarse but all we have without
         DICOM-level modality tags for every clip.
      3. If ``filename`` contains Doppler/M-mode/TDI hints, return that.
      4. Default to ``b_mode``.

    Note: the MIMIC classifier distinguishes only B-mode vs color Doppler.
    Spectral/M-mode/TDI are not currently separated by the classifier, so
    unless a downstream source supplies DICOM tags explicitly, we cannot
    reliably populate those modality classes. The manifest will largely show
    ``b_mode`` and ``color_doppler`` until a better modality source lands.
    """
    if raw_modality is not None:
        m = str(raw_modality).strip().lower()
        if m in MODALITIES:
            return m

    if color_flag is not None:
        c = str(color_flag).strip().lower()
        if c == "yes":
            return "color_doppler"
        if c == "no":
            return "b_mode"

    if filename is not None:
        f = str(filename).lower()
        if "mmode" in f or "m_mode" in f or "mmodeclip" in f:
            return "m_mode"
        if "tdi" in f or "tissuedoppler" in f:
            return "tdi"
        if "cwd" in f or "cwdoppler" in f:
            return "cw_doppler"
        if "pwd" in f or "pwdoppler" in f:
            return "pw_doppler"

    # SOP Class UID can sometimes help disambiguate (e.g.
    # 1.2.840.10008.5.1.4.1.1.3.1 is US Multi-frame, not a modality tag, so
    # this is a weak signal; we leave the hook here for future extension).
    _ = dicom_sop_uid

    return "b_mode"


# ---------------------------------------------------------------------------
# Measurement site inference.
# ---------------------------------------------------------------------------


def infer_measurement_site(
    raw_view_label: Optional[str] = None,
    modality: Optional[str] = None,
    dicom_tags: Optional[dict] = None,
    filename: Optional[str] = None,
    text: Optional[str] = None,
) -> str:
    """Heuristic for anatomic measurement site.

    For MIMIC without reliable OCR or DICOM text annotations, almost every row
    will return ``none`` (B-mode) or ``unknown`` (Doppler). That is correct —
    the manifest honestly encodes "we don't know the measurement site" rather
    than guessing. Downstream code is expected to tolerate ``unknown``.
    """
    mod = (modality or "").lower()
    if mod == "b_mode":
        return "none"
    if mod == "m_mode":
        # M-mode sweep location is usually mitral/aortic; without a text cue
        # we can't tell.
        return "unknown"

    # Filename hints for Doppler measurement sites.
    f = (filename or "").lower()
    t = (text or "").lower()
    combined = f + " " + t
    if any(k in combined for k in ("tr_", "_tr", "tricuspid regurg", "tricuspid_regurg")):
        return "TR"
    if "lvot" in combined:
        return "LVOT"
    if any(k in combined for k in ("mv_inflow", "mitral_inflow", "mvinflow")):
        return "MV_inflow"
    if "aortic_valve" in combined or "_av_" in combined:
        return "AV"
    if "mitral_valve" in combined or "_mv_" in combined:
        return "MV"
    if "ivc" in combined:
        return "IVC"
    if "septal_annul" in combined or "medial_annul" in combined:
        return "septal_annulus"
    if "lateral_annul" in combined:
        return "lateral_annulus"

    return "unknown"


# ---------------------------------------------------------------------------
# Phase bucket normalization.
# ---------------------------------------------------------------------------


def normalize_phase_bucket(phase_label: Optional[str], modality: str) -> str:
    """Return a canonical phase_bucket.

    Rules (plan §3.1):
      - Spectral Doppler / M-mode / TDI / contrast → ``not_applicable``.
      - B-mode / color Doppler: use the phase label if in {systolic,
        diastolic, full_cycle}; otherwise ``unknown``.
    """
    mod = (modality or "").lower()
    if mod in {"pw_doppler", "cw_doppler", "m_mode", "tdi", "contrast"}:
        return "not_applicable"
    if phase_label is None:
        return "unknown"
    p = str(phase_label).strip().lower()
    if p in {"systolic", "diastolic", "full_cycle"}:
        return p
    return "unknown"


__all__ = [
    "VIEW_FAMILIES",
    "MODALITIES",
    "MEASUREMENT_SITES",
    "PHASE_BUCKETS",
    "EXCLUDED_VIEW_LABELS",
    "normalize_view_family",
    "is_excluded_view",
    "normalize_modality",
    "infer_measurement_site",
    "normalize_phase_bucket",
]
