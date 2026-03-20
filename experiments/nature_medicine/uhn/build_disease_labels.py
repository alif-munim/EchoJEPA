#!/usr/bin/env python3
"""Build disease detection label NPZ files for UHN probe evaluation.

Rebuilds the 9 binary disease detection datasets with full provenance.
Each disease has a positive cohort (disease present) and a hard negative
cohort (clinically similar but different diagnosis). Labels are derived
from three independent sources:

    1. Syngo structured observations (syngo_observations)
    2. HeartLab free-text findings (heartlab_finding_intersects.NOTE)
    3. Syngo free-text fields (syngo_analytics_study + syngo_study_details)

All sources are queried separately, then unioned with provenance tags.
Negation filtering is applied to exclude rule-out and indication-only mentions.

Data sources:
    - echo.db: syngo_observations, heartlab tables, syngo free-text tables
    - data/aws/aws_syngo.csv: StudyRef -> DeidentifiedStudyID mapping
    - study_deid_map (in echo.db): HeartLab STUDY_INSTANCE_UID -> DeidentifiedStudyID
    - patient_split.json: patient_id -> train/val/test split

Usage:
    python build_disease_labels.py --all
    python build_disease_labels.py --task disease_hcm
    python build_disease_labels.py --all --validate
    python build_disease_labels.py --all --validate --output_dir labels_v2
    python build_disease_labels.py --all --validate --exclude_indication --output_dir labels_v3
    python build_disease_labels.py --all --validate --exclude_indication --require_heartlab --output_dir labels_v4

Output:
    - NPZ files: {output_dir}/disease_{name}.npz
    - Provenance JSON: {output_dir}/provenance_{name}.json
"""

import argparse
import json
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
ECHO_DB = REPO_ROOT / "uhn_echo" / "nature_medicine" / "data_exploration" / "echo.db"
AWS_SYNGO_CSV = REPO_ROOT / "data" / "aws" / "aws_syngo.csv"
PATIENT_SPLIT = SCRIPT_DIR / "patient_split.json"
EXISTING_LABELS = SCRIPT_DIR / "labels"

DISEASES = [
    "disease_hcm",
    "disease_amyloidosis",
    "disease_takotsubo",
    "disease_stemi",
    "disease_dcm",
    "disease_bicuspid_av",
    "disease_myxomatous_mv",
    "disease_rheumatic_mv",
]

# ---------------------------------------------------------------------------
# Standard negation patterns
# ---------------------------------------------------------------------------
NEGATION_PREFIXES = [
    "no {term}",
    "r/o {term}",
    "rule out {term}",
    "rule-out {term}",
    "no evidence of {term}",
    "unlikely {term}",
    "to exclude {term}",
    "no definite {term}",
    "no obvious {term}",
    "without {term}",
    "absent {term}",
]

# Negation words for proximity-based negation checking.
# Unlike NEGATION_PREFIXES (which require "{neg} {term}" as adjacent substring),
# proximity negation fires when any trigger word appears BEFORE the disease term
# anywhere in the same text unit. Catches patterns like "R/O vegetation / endocarditis"
# where "R/O" and "endocarditis" are separated by other content.
PROXIMITY_NEGATION_TRIGGERS = [
    "r/o ", "r/o\t", "r/o\n",
    "rule out ", "rule-out ", "to exclude ", "to rule out ",
    "no evidence of ", "no definite ", "no obvious ",
]


def _term_needs_word_boundary(term):
    """Check if a search term needs word-boundary validation.

    Short uppercase acronyms (e.g., STEMI, ATTR, DCM) can falsely match as
    substrings of common words (e.g., 'STEMI' in 'systemic'). Returns True
    for terms that need regex \\b validation after SQL LIKE.
    """
    return len(term) <= 6 and term.replace("-", "").isupper()


def _validate_term_match(text, term):
    """Validate that a term genuinely matches in text, not as a substring of another word.

    For short uppercase acronyms, uses word-boundary regex (\\bSTEMI\\b rejects
    'systemic'). For longer/mixed-case terms, returns True since SQL LIKE already
    confirmed the substring presence.
    """
    if not text:
        return False
    if not _term_needs_word_boundary(term):
        return True
    return bool(re.search(r'\b' + re.escape(term) + r'\b', text, re.IGNORECASE))


def _has_proximity_negation(text, term):
    """Check if any negation trigger appears BEFORE the disease term in text.

    Catches non-adjacent negation like 'R/O vegetation / endocarditis' where
    the exact pattern 'r/o endocarditis' is not a contiguous substring.

    Only triggers when negation appears before the term (not after), to avoid
    false flags like 'Patient has endocarditis. No fever.'
    """
    if not text:
        return False
    text_lower = text.lower()
    term_lower = term.lower()
    pos = text_lower.find(term_lower)
    if pos < 0:
        return False
    prefix = text_lower[:pos]
    for trigger in PROXIMITY_NEGATION_TRIGGERS:
        if trigger in prefix:
            return True
    return False


def build_negation_clauses(column, term, extra_negations=None):
    """Build SQL NOT LIKE clauses for negation filtering.

    Returns a list of SQL fragments like "AND col NOT LIKE '%no term%' COLLATE NOCASE".
    """
    clauses = []
    for prefix in NEGATION_PREFIXES:
        pattern = prefix.format(term=term)
        clauses.append(f"AND {column} NOT LIKE '%{pattern}%' COLLATE NOCASE")
    if extra_negations:
        for neg in extra_negations:
            clauses.append(f"AND {column} NOT LIKE '%{neg}%' COLLATE NOCASE")
    return clauses


def negation_check(text, term, extra_negations=None):
    """Check if a text mention is negated (Python-side filtering).

    Returns True if the mention is negated (should be excluded).
    """
    if text is None:
        return True
    text_lower = text.lower()
    term_lower = term.lower()
    for prefix in NEGATION_PREFIXES:
        pattern = prefix.format(term=term_lower)
        if pattern in text_lower:
            return True
    if extra_negations:
        for neg in extra_negations:
            if neg.lower() in text_lower:
                return True
    return False


# ---------------------------------------------------------------------------
# Shared data loading (mirrors build_labels.py)
# ---------------------------------------------------------------------------
_studyref_cache = {}
_deid_patient_cache = {}
_patient_split_cache = {}
_hl_stud_to_deid_cache = {}


def load_studyref_mapping():
    """Load StudyRef -> (DeidentifiedStudyID, patient_id) from aws_syngo.csv."""
    if _studyref_cache:
        return _studyref_cache["ref_to_deid"], _studyref_cache["ref_to_patient"]
    df = pd.read_csv(AWS_SYNGO_CSV, usecols=["STUDY_REF", "PATIENT_ID", "DeidentifiedStudyID"])
    ref_to_deid = dict(zip(df["STUDY_REF"].astype(str), df["DeidentifiedStudyID"]))
    ref_to_patient = dict(zip(df["STUDY_REF"].astype(str), df["PATIENT_ID"].astype(str)))
    _studyref_cache["ref_to_deid"] = ref_to_deid
    _studyref_cache["ref_to_patient"] = ref_to_patient
    print(f"  Loaded {len(ref_to_deid):,} StudyRef->DeidID mappings from aws_syngo.csv")
    return ref_to_deid, ref_to_patient


def load_deid_to_patient():
    """Build DeidentifiedStudyID -> patient_id from Syngo + HeartLab."""
    if _deid_patient_cache:
        return _deid_patient_cache["map"]
    conn = sqlite3.connect(str(ECHO_DB))
    rows = conn.execute("SELECT DeidentifiedStudyID, OriginalPatientID FROM study_deid_map").fetchall()
    conn.close()
    hl_map = {r[0]: str(r[1]) for r in rows}
    df = pd.read_csv(AWS_SYNGO_CSV, usecols=["DeidentifiedStudyID", "PATIENT_ID"])
    syngo_map = dict(zip(df["DeidentifiedStudyID"], df["PATIENT_ID"].astype(str)))
    combined = {**hl_map, **syngo_map}
    _deid_patient_cache["map"] = combined
    print(f"  Loaded {len(combined):,} DeidID->patient_id mappings")
    return combined


def load_patient_splits():
    """Load patient_id -> split mapping."""
    if _patient_split_cache:
        return _patient_split_cache["map"]
    with open(PATIENT_SPLIT) as f:
        splits = json.load(f)
    _patient_split_cache["map"] = splits
    print(f"  Loaded {len(splits):,} patient splits")
    return splits


def load_hl_stud_to_deid():
    """Build HeartLab STUD_ID -> DeidentifiedStudyID mapping.

    Join: heartlab_studies.STUDY_INSTANCE_UID = study_deid_map.OriginalStudyID.
    """
    if _hl_stud_to_deid_cache:
        return _hl_stud_to_deid_cache["map"]
    conn = sqlite3.connect(str(ECHO_DB))
    rows = conn.execute(
        """
        SELECT hs.ID as stud_id, dm.DeidentifiedStudyID, dm.OriginalPatientID
        FROM heartlab_studies hs
        JOIN study_deid_map dm ON hs.STUDY_INSTANCE_UID = dm.OriginalStudyID
        """
    ).fetchall()
    conn.close()
    stud_to_deid = {}
    stud_to_patient = {}
    for stud_id, deid, patient in rows:
        stud_to_deid[str(stud_id)] = deid
        stud_to_patient[str(stud_id)] = str(patient)
    _hl_stud_to_deid_cache["map"] = stud_to_deid
    _hl_stud_to_deid_cache["patient"] = stud_to_patient
    print(f"  Loaded {len(stud_to_deid):,} HeartLab STUD_ID->DeidID mappings")
    return stud_to_deid


def get_hl_stud_to_patient():
    """Get HeartLab STUD_ID -> patient_id (must call load_hl_stud_to_deid first)."""
    return _hl_stud_to_deid_cache.get("patient", {})


# ---------------------------------------------------------------------------
# Source query functions
# ---------------------------------------------------------------------------


def query_syngo_obs(conn, obs_name, values, exclude_values=None):
    """Query syngo_observations for structured observation values.

    Returns: dict of {StudyRef: value} for matching studies.
    """
    placeholders = ",".join("?" for _ in values)
    query = f"""
        SELECT DISTINCT StudyRef, Value
        FROM syngo_observations
        WHERE Name = ?
          AND Value IN ({placeholders})
    """
    rows = conn.execute(query, [obs_name] + list(values)).fetchall()
    result = {}
    for ref, val in rows:
        if exclude_values and val in exclude_values:
            continue
        result[str(ref)] = val
    return result


def query_syngo_obs_like(conn, obs_name, value_patterns, exclude_patterns=None):
    """Query syngo_observations using LIKE patterns on Value.

    Returns: dict of {StudyRef: value} for matching studies.
    """
    conditions = " OR ".join(f"Value LIKE '{p}' COLLATE NOCASE" for p in value_patterns)
    query = f"""
        SELECT DISTINCT StudyRef, Value
        FROM syngo_observations
        WHERE Name = ? AND ({conditions})
    """
    rows = conn.execute(query, [obs_name]).fetchall()
    result = {}
    for ref, val in rows:
        if exclude_patterns:
            skip = False
            for ep in exclude_patterns:
                if re.search(ep, val, re.IGNORECASE):
                    skip = True
                    break
            if skip:
                continue
        result[str(ref)] = val
    return result


def query_heartlab_freetext(conn, search_terms, negation_terms=None, extra_negations=None,
                            exclude_finding_groups=None):
    """Query HeartLab for disease mentions via TWO paths:

    1. SENTENCE path: heartlab_findings.SENTENCE matched via fi.FIN_ID (primary — most findings)
    2. NOTE path: heartlab_finding_intersects.NOTE free-text (supplementary)

    Both use the correct FK chain: fi.REP_ID -> reports.ID -> series.ID -> studies.ID.

    Args:
        exclude_finding_groups: list of finding group labels to skip (e.g.,
            ["VALVE/ROOT  DISEASE", "CONGENITAL HEART DISEASE"]). Matches from
            these HL groups are treated as indication/referral entries, not findings.

    Returns: dict of {STUD_ID: [list of matched texts]}.
    Also returns negated: dict of {STUD_ID: [negated texts]}.
    """
    results = defaultdict(list)
    negated = defaultdict(list)

    all_neg_patterns = list(negation_terms or []) + list(extra_negations or [])

    def _is_negated(text):
        if not text:
            return False
        text_lower = text.lower()
        for pat in all_neg_patterns:
            if pat.lower() in text_lower:
                return True
        return False

    _excl_groups = set(g.lower() for g in (exclude_finding_groups or []))

    for term in search_terms:
        # Path 1: SENTENCE-level match (fi.FIN_ID -> heartlab_findings.SENTENCE)
        query = f"""
            SELECT DISTINCT se.STUD_ID, f.SENTENCE, fi.NOTE, hfg.LABEL
            FROM heartlab_finding_intersects fi
            JOIN heartlab_findings f ON fi.FIN_ID = f.ID
            JOIN heartlab_reports hr ON fi.REP_ID = hr.ID
            JOIN heartlab_series se ON hr.SERI_ID = se.ID
            LEFT JOIN heartlab_finding_groups hfg ON f.FGRP_ID = hfg.ID
            WHERE f.SENTENCE LIKE '%{term}%' COLLATE NOCASE
        """
        rows = conn.execute(query).fetchall()
        for stud_id, sentence, note, group_label in rows:
            # Skip indication-like finding groups if requested
            if _excl_groups and group_label and group_label.lower() in _excl_groups:
                continue
            # Fix 1: Validate word boundary for short acronyms
            # (e.g., reject "STEMI" matching "systemic")
            if not _validate_term_match(sentence, term):
                continue
            # Fix 2: Check exact negation patterns + proximity negation
            # (e.g., catch "R/O vegetation / endocarditis" where R/O is non-adjacent)
            if (_is_negated(sentence) or _is_negated(note)
                    or _has_proximity_negation(sentence, term)
                    or _has_proximity_negation(note, term)):
                negated[str(stud_id)].append(sentence)
            else:
                results[str(stud_id)].append(sentence)

        # Path 2: NOTE-level match (supplementary — catches free-text mentions not in SENTENCE)
        query = f"""
            SELECT DISTINCT se.STUD_ID, fi.NOTE
            FROM heartlab_finding_intersects fi
            JOIN heartlab_reports hr ON fi.REP_ID = hr.ID
            JOIN heartlab_series se ON hr.SERI_ID = se.ID
            WHERE fi.NOTE LIKE '%{term}%' COLLATE NOCASE
        """
        rows = conn.execute(query).fetchall()
        for stud_id, note in rows:
            stud_str = str(stud_id)
            if stud_str in results:
                continue  # Already found via SENTENCE
            # Fix 1: Validate word boundary for short acronyms
            if not _validate_term_match(note, term):
                continue
            # Fix 2: Check exact negation + proximity negation
            if _is_negated(note) or _has_proximity_negation(note, term):
                negated[stud_str].append(note)
            else:
                results[stud_str].append(note)

    return dict(results), dict(negated)


def query_heartlab_negated_only(conn, search_terms, negation_terms):
    """Query HeartLab for studies where the term is ONLY mentioned in negated context.

    Searches both SENTENCE and NOTE paths. These are studies where endocarditis/etc
    was ruled out — useful as hard negatives.
    Returns: set of STUD_IDs.
    """
    negated_studs = set()

    def _is_negated(text):
        if not text:
            return False
        text_lower = text.lower()
        for nt in negation_terms:
            if nt.lower() in text_lower:
                return True
        return False

    for term in search_terms:
        # SENTENCE path
        query = f"""
            SELECT DISTINCT se.STUD_ID, f.SENTENCE, fi.NOTE
            FROM heartlab_finding_intersects fi
            JOIN heartlab_findings f ON fi.FIN_ID = f.ID
            JOIN heartlab_reports hr ON fi.REP_ID = hr.ID
            JOIN heartlab_series se ON hr.SERI_ID = se.ID
            WHERE f.SENTENCE LIKE '%{term}%' COLLATE NOCASE
        """
        rows = conn.execute(query).fetchall()
        for stud_id, sentence, note in rows:
            if not _validate_term_match(sentence, term):
                continue
            if (_is_negated(sentence) or _is_negated(note)
                    or _has_proximity_negation(sentence, term)
                    or _has_proximity_negation(note, term)):
                negated_studs.add(str(stud_id))

        # NOTE path
        query = f"""
            SELECT DISTINCT se.STUD_ID, fi.NOTE
            FROM heartlab_finding_intersects fi
            JOIN heartlab_reports hr ON fi.REP_ID = hr.ID
            JOIN heartlab_series se ON hr.SERI_ID = se.ID
            WHERE fi.NOTE LIKE '%{term}%' COLLATE NOCASE
        """
        rows = conn.execute(query).fetchall()
        for stud_id, note in rows:
            if not _validate_term_match(note, term):
                continue
            if _is_negated(note) or _has_proximity_negation(note, term):
                negated_studs.add(str(stud_id))

    return negated_studs


def query_syngo_freetext(conn, search_terms, negation_terms=None, extra_negations=None,
                         exclude_indication=False):
    """Query Syngo free-text fields for mentions.

    Searches both syngo_analytics_study (Indication, StudyComments, PatientHistory)
    and syngo_study_details (STUDY_COMMENTS, PATIENT_HISTORY, REASON_FOR_STUDY).

    Args:
        exclude_indication: If True, skip Indication and REASON_FOR_STUDY fields
            (referral reasons, not diagnostic findings). Keeps StudyComments,
            PatientHistory, STUDY_COMMENTS, PATIENT_HISTORY.

    Returns: dict of {StudyRef: {"field": field_name, "text": matched_text, "negated": bool}}.
    Also returns negated: dict of same structure for negated mentions.
    """
    results = {}
    negated = {}

    # Table 1: syngo_analytics_study
    if exclude_indication:
        analytics_fields = ["StudyComments", "PatientHistory"]
    else:
        analytics_fields = ["Indication", "StudyComments", "PatientHistory"]
    for term in search_terms:
        conditions = " OR ".join(f'"{f}" LIKE \'%{term}%\' COLLATE NOCASE' for f in analytics_fields)
        query = f"""
            SELECT StudyRef, Indication, StudyComments, PatientHistory
            FROM syngo_analytics_study
            WHERE {conditions}
        """
        rows = conn.execute(query).fetchall()
        for ref, indication, comments, history in rows:
            ref = str(ref)
            for field_name, text in [("Indication", indication), ("StudyComments", comments), ("PatientHistory", history)]:
                if text and term.lower() in text.lower():
                    # Fix 1: Validate word boundary for short acronyms
                    if not _validate_term_match(text, term):
                        continue
                    is_neg = False
                    if negation_terms:
                        for nt in negation_terms:
                            if nt.lower() in text.lower():
                                is_neg = True
                                break
                    if extra_negations and not is_neg:
                        for en in extra_negations:
                            if en.lower() in text.lower():
                                is_neg = True
                                break
                    # Fix 2: Proximity negation
                    if not is_neg and _has_proximity_negation(text, term):
                        is_neg = True
                    entry = {"field": field_name, "text": text[:200], "negated": is_neg}
                    if is_neg:
                        if ref not in negated:
                            negated[ref] = entry
                    else:
                        if ref not in results:
                            results[ref] = entry

    # Table 2: syngo_study_details
    if exclude_indication:
        details_fields = ["STUDY_COMMENTS", "PATIENT_HISTORY"]
    else:
        details_fields = ["STUDY_COMMENTS", "PATIENT_HISTORY", "REASON_FOR_STUDY"]
    for term in search_terms:
        conditions = " OR ".join(f'"{f}" LIKE \'%{term}%\' COLLATE NOCASE' for f in details_fields)
        query = f"""
            SELECT STUDY_REF, STUDY_COMMENTS, PATIENT_HISTORY, REASON_FOR_STUDY
            FROM syngo_study_details
            WHERE {conditions}
        """
        rows = conn.execute(query).fetchall()
        for ref, comments, history, reason in rows:
            ref = str(ref)
            for field_name, text in [
                ("STUDY_COMMENTS", comments),
                ("PATIENT_HISTORY", history),
                ("REASON_FOR_STUDY", reason),
            ]:
                if text and term.lower() in text.lower():
                    # Fix 1: Validate word boundary for short acronyms
                    if not _validate_term_match(text, term):
                        continue
                    is_neg = False
                    if negation_terms:
                        for nt in negation_terms:
                            if nt.lower() in text.lower():
                                is_neg = True
                                break
                    if extra_negations and not is_neg:
                        for en in extra_negations:
                            if en.lower() in text.lower():
                                is_neg = True
                                break
                    # Fix 2: Proximity negation
                    if not is_neg and _has_proximity_negation(text, term):
                        is_neg = True
                    entry = {"field": field_name, "text": text[:200], "negated": is_neg}
                    if is_neg:
                        if ref not in negated:
                            negated[ref] = entry
                    else:
                        if ref not in results:
                            results[ref] = entry

    return results, negated


# ---------------------------------------------------------------------------
# Mapping helpers
# ---------------------------------------------------------------------------


def syngo_refs_to_deid(studyrefs, ref_to_deid, ref_to_patient):
    """Map Syngo StudyRefs to (DeidentifiedStudyID, patient_id) pairs."""
    mapped = {}
    for ref in studyrefs:
        deid = ref_to_deid.get(str(ref))
        patient = ref_to_patient.get(str(ref))
        if deid and patient:
            mapped[deid] = str(patient)
    return mapped


def hl_studs_to_deid(stud_ids, stud_to_deid):
    """Map HeartLab STUD_IDs to DeidentifiedStudyIDs."""
    stud_to_patient = get_hl_stud_to_patient()
    mapped = {}
    for stud_id in stud_ids:
        deid = stud_to_deid.get(str(stud_id))
        patient = stud_to_patient.get(str(stud_id))
        if deid and patient:
            mapped[deid] = str(patient)
    return mapped


# ---------------------------------------------------------------------------
# HeartLab requirement filter
# ---------------------------------------------------------------------------


def filter_syngo_text_only(all_pos, confirmed_deids, disease_name):
    """Remove positives that only come from Syngo free-text (no HeartLab/obs confirmation).

    A "Syngo-text-only" positive means the only evidence is from StudyComments or
    PatientHistory — the ordering physician's context, not the echocardiographer's
    report. These are real patients but noisier labels for a model learning from video.

    Args:
        all_pos: dict of {DeidID: patient_id} — modified in place
        confirmed_deids: set of DeidIDs confirmed by HeartLab SENTENCE/NOTE or Syngo obs
        disease_name: for logging

    Returns:
        Number of removed studies.
    """
    syngo_only = {d for d in all_pos if d not in confirmed_deids}
    for d in syngo_only:
        del all_pos[d]
    if syngo_only:
        print(f"  --require_heartlab: removed {len(syngo_only):,} Syngo-text-only positives from {disease_name}")
    return len(syngo_only)


# ---------------------------------------------------------------------------
# Provenance reporting
# ---------------------------------------------------------------------------


def build_provenance_report(
    disease_name,
    syngo_obs_count,
    heartlab_count,
    syngo_text_count,
    heartlab_negated_count,
    syngo_text_negated_count,
    total_pos,
    total_neg,
    indication_only_count=0,
    syngo_text_only_removed=0,
):
    """Print a provenance report for a disease cohort."""
    total_raw = syngo_obs_count + heartlab_count + syngo_text_count
    print(f"\n  === Provenance Report: {disease_name} ===")
    print(f"  Positive: {total_pos:,} studies | Negative: {total_neg:,} studies")
    print(f"  Source breakdown (positive, pre-dedup):")
    if total_raw > 0:
        print(f"    Syngo structured obs: {syngo_obs_count:,} ({100*syngo_obs_count/max(total_raw,1):.1f}%)")
        print(f"    HeartLab free-text:   {heartlab_count:,} ({100*heartlab_count/max(total_raw,1):.1f}%)")
        print(f"    Syngo free-text:      {syngo_text_count:,} ({100*syngo_text_count/max(total_raw,1):.1f}%)")
    print(f"  Negation filtering:")
    print(f"    HeartLab negated (excluded): {heartlab_negated_count:,}")
    print(f"    Syngo text negated (excluded): {syngo_text_negated_count:,}")
    total_negated = heartlab_negated_count + syngo_text_negated_count
    total_before_filter = total_raw + total_negated
    if total_before_filter > 0:
        print(f"    Overall negation rate: {100*total_negated/total_before_filter:.1f}%")
    if indication_only_count > 0:
        print(f"  Indication-only mentions: {indication_only_count:,}")
    if syngo_text_only_removed > 0:
        print(f"  Syngo-text-only removed (--require_heartlab): {syngo_text_only_removed:,}")

    # Confidence assessment
    neg_rate = total_negated / max(total_before_filter, 1)
    if neg_rate > 0.30:
        confidence = "LOW"
        note = "High negation rate suggests many rule-out mentions"
    elif heartlab_count > 0.8 * total_raw and syngo_obs_count < 0.05 * total_raw:
        confidence = "MEDIUM"
        note = "Almost entirely free-text derived; no structured validation"
    elif syngo_obs_count > 0.1 * total_raw:
        confidence = "HIGH"
        note = "Substantial structured observation support"
    else:
        confidence = "MEDIUM"
        note = "Mixed sources"
    print(f"  Confidence: {confidence} — {note}")

    return {
        "disease": disease_name,
        "total_positive": total_pos,
        "total_negative": total_neg,
        "sources": {
            "syngo_obs": syngo_obs_count,
            "heartlab_freetext": heartlab_count,
            "syngo_freetext": syngo_text_count,
        },
        "negation": {
            "heartlab_negated": heartlab_negated_count,
            "syngo_text_negated": syngo_text_negated_count,
            "overall_rate": round(neg_rate, 4),
        },
        "indication_only": indication_only_count,
        "syngo_text_only_removed": syngo_text_only_removed,
        "confidence": confidence,
        "note": note,
    }


# ---------------------------------------------------------------------------
# Per-disease builders
# ---------------------------------------------------------------------------


def build_disease_hcm(conn, ref_to_deid, ref_to_patient, stud_to_deid, patient_splits, deid_to_patient,
                      exclude_indication=False, require_heartlab=False):
    """Build HCM (positive) vs Concentric LVH (negative)."""
    print("\n--- Building disease_hcm ---")

    # === POSITIVE COHORT ===
    # Source 1: Syngo obs — Cardiomyopathy_obs
    syngo_obs_hits = query_syngo_obs(conn, "Cardiomyopathy_obs", ["HOCM", "HCM", "Hypertrophic", "hypertrophic"])
    # Also check for free-text obs values mentioning HCM
    syngo_obs_like = query_syngo_obs_like(conn, "Cardiomyopathy_obs", ["%HOCM%", "%hypertrophic%"])
    syngo_obs_hits.update(syngo_obs_like)
    syngo_pos = syngo_refs_to_deid(syngo_obs_hits.keys(), ref_to_deid, ref_to_patient)
    print(f"  Syngo obs: {len(syngo_obs_hits)} StudyRefs -> {len(syngo_pos)} mapped")

    # Source 2: HeartLab free-text
    hl_terms = ["hypertrophic cardiomyopath", "HCM", "HOCM", "hypertrophic obstructive"]
    hl_neg_terms = [
        "no HCM", "no HOCM", "r/o HCM", "r/o HOCM",
        "rule out HCM", "rule out HOCM",
        "no evidence of HCM", "no evidence of HOCM",
        "unlikely HCM", "unlikely HOCM",
        "no hypertrophic", "rule out hypertrophic",
    ]
    hl_positive, hl_negated = query_heartlab_freetext(conn, hl_terms, hl_neg_terms)
    hl_pos = hl_studs_to_deid(hl_positive.keys(), stud_to_deid)
    print(f"  HeartLab: {len(hl_positive)} studies -> {len(hl_pos)} mapped, {len(hl_negated)} negated")

    # HeartLab SAM — only include when co-occurring with HCM/HOCM terms
    sam_results, sam_negated = query_heartlab_freetext(
        conn,
        ["SAM", "systolic anterior motion"],
        ["no SAM", "chordal SAM", "no systolic anterior motion", "no evidence of SAM"],
        extra_negations=["without SAM"],
    )
    # Filter SAM: only keep if the same NOTE also mentions HCM/HOCM/hypertrophic
    sam_with_hcm = {}
    hcm_pattern = re.compile(r"(HCM|HOCM|hypertrophic cardiomyopath)", re.IGNORECASE)
    for stud_id, notes in sam_results.items():
        for note in notes:
            if hcm_pattern.search(note or ""):
                sam_with_hcm[stud_id] = notes
                break
    sam_pos = hl_studs_to_deid(sam_with_hcm.keys(), stud_to_deid)
    print(f"  SAM+HCM co-occurring: {len(sam_with_hcm)} studies -> {len(sam_pos)} mapped")

    # HeartLab ASH
    ash_results, ash_negated = query_heartlab_freetext(
        conn,
        ["asymmetric septal hypertrophy"],
        ["no asymmetric septal", "r/o asymmetric", "no evidence of asymmetric"],
    )
    ash_pos = hl_studs_to_deid(ash_results.keys(), stud_to_deid)
    print(f"  ASH: {len(ash_results)} studies -> {len(ash_pos)} mapped")

    # Source 3: Syngo free-text
    syngo_text_terms = ["hypertrophic cardiomyopathy", "HOCM", "HCM"]
    syngo_text_neg = ["no HCM", "no HOCM", "r/o HCM", "rule out HCM", "no hypertrophic cardiomyopath"]
    syngo_text_pos, syngo_text_negated = query_syngo_freetext(conn, syngo_text_terms, syngo_text_neg,
                                                              exclude_indication=exclude_indication)
    syngo_text_mapped = syngo_refs_to_deid(syngo_text_pos.keys(), ref_to_deid, ref_to_patient)
    print(f"  Syngo text: {len(syngo_text_pos)} StudyRefs -> {len(syngo_text_mapped)} mapped, {len(syngo_text_negated)} negated")

    # Count indication-only
    indication_only = sum(1 for v in syngo_text_pos.values() if v.get("field") == "Indication")

    # Union all positive sources
    all_pos = {}
    for deid, pid in syngo_pos.items():
        all_pos[deid] = pid
    for deid, pid in hl_pos.items():
        all_pos[deid] = pid
    for deid, pid in sam_pos.items():
        all_pos[deid] = pid
    for deid, pid in ash_pos.items():
        all_pos[deid] = pid
    for deid, pid in syngo_text_mapped.items():
        all_pos[deid] = pid
    print(f"  Total positive (pre-filter): {len(all_pos):,}")

    # Filter: require HeartLab/obs confirmation
    syngo_text_only_removed = 0
    if require_heartlab:
        confirmed = set(syngo_pos.keys()) | set(hl_pos.keys()) | set(sam_pos.keys()) | set(ash_pos.keys())
        syngo_text_only_removed = filter_syngo_text_only(all_pos, confirmed, "disease_hcm")

    # === NEGATIVE COHORT: Concentric LVH without HCM ===
    # Syngo obs: LV_Thickness_2_0_obs with concentric values
    lvh_obs = query_syngo_obs(
        conn,
        "LV_Thickness_2_0_obs",
        ["mild_concentric", "moderate_concentric", "severe_concentric"],
    )
    lvh_neg = syngo_refs_to_deid(lvh_obs.keys(), ref_to_deid, ref_to_patient)
    print(f"  Concentric LVH (Syngo obs): {len(lvh_obs)} StudyRefs -> {len(lvh_neg)} mapped")

    # HeartLab: concentric hypertrophy/LVH mentions
    hl_lvh_results, _ = query_heartlab_freetext(
        conn,
        ["concentric hypertrophy", "concentric LVH", "concentric remodel", "concentric left ventricular"],
        None,
    )
    hl_lvh_neg = hl_studs_to_deid(hl_lvh_results.keys(), stud_to_deid)
    print(f"  Concentric LVH (HeartLab): {len(hl_lvh_results)} studies -> {len(hl_lvh_neg)} mapped")

    # Union negatives, exclude any study in positive set
    all_neg = {}
    for deid, pid in lvh_neg.items():
        if deid not in all_pos:
            all_neg[deid] = pid
    for deid, pid in hl_lvh_neg.items():
        if deid not in all_pos:
            all_neg[deid] = pid

    # v7.1 Fix 3: Exclude negatives that mention HCM in Syngo indication/comments.
    # These are HCM patients with concentric phenotype — their HCM diagnosis appears
    # in referral fields even though exclude_indication hides it from positives.
    hcm_crosscheck_terms = ["hypertrophic cardiomyopathy", "HOCM", "HCM"]
    hcm_crosscheck_neg = ["no HCM", "no HOCM", "r/o HCM", "rule out HCM", "no hypertrophic cardiomyopath"]
    hcm_in_syngo, _ = query_syngo_freetext(conn, hcm_crosscheck_terms, hcm_crosscheck_neg,
                                            exclude_indication=False)  # include ALL fields
    hcm_syngo_deids = set(syngo_refs_to_deid(hcm_in_syngo.keys(), ref_to_deid, ref_to_patient).keys())
    neg_hcm_removed = 0
    for deid in list(all_neg.keys()):
        if deid in hcm_syngo_deids:
            del all_neg[deid]
            neg_hcm_removed += 1
    if neg_hcm_removed:
        print(f"  v7.1: Removed {neg_hcm_removed} negatives with HCM in Syngo indication/comments")

    print(f"  Total negative (pre-patient-filter): {len(all_neg):,}")

    provenance = build_provenance_report(
        "disease_hcm",
        syngo_obs_count=len(syngo_pos),
        heartlab_count=len(hl_pos) + len(sam_pos) + len(ash_pos),
        syngo_text_count=len(syngo_text_mapped),
        heartlab_negated_count=len(hl_negated) + len(sam_negated),
        syngo_text_negated_count=len(syngo_text_negated),
        total_pos=len(all_pos),
        total_neg=len(all_neg),
        indication_only_count=indication_only,
        syngo_text_only_removed=syngo_text_only_removed,
    )
    return all_pos, all_neg, provenance


def build_disease_amyloidosis(conn, ref_to_deid, ref_to_patient, stud_to_deid, patient_splits, deid_to_patient,
                              exclude_indication=False, require_heartlab=False):
    """Build Cardiac Amyloidosis (positive) vs HCM (negative)."""
    print("\n--- Building disease_amyloidosis ---")

    # === POSITIVE: Amyloidosis ===
    # Source 1: Syngo obs (Cardiomyopathy_obs — likely no amyloid values but check)
    syngo_obs_hits = query_syngo_obs_like(conn, "Cardiomyopathy_obs", ["%amyloid%", "%infiltrative%"])
    syngo_pos = syngo_refs_to_deid(syngo_obs_hits.keys(), ref_to_deid, ref_to_patient)
    print(f"  Syngo obs: {len(syngo_obs_hits)} StudyRefs -> {len(syngo_pos)} mapped")

    # Source 2: HeartLab
    hl_terms = ["amyloid", "cardiac amyloid", "ATTR", "transthyretin"]
    hl_neg_terms = [
        "no amyloid", "r/o amyloid", "rule out amyloid",
        "no evidence of amyloid", "unlikely amyloid",
        "renal amyloid", "hepatic amyloid",
        # Fix 2: Catch screening referrals (systemic amyloid → r/o cardiac involvement)
        "rule out cardiac involvement", "r/o cardiac amyloid",
        "rule out heart involvement", "rule out cardiac amyloid",
        # v7.2: Family history is not patient diagnosis
        "family history of amyloid", "family hx of amyloid",
        "family hx amyloid", "family history amyloid",
    ]
    hl_positive, hl_negated = query_heartlab_freetext(conn, hl_terms, hl_neg_terms)
    hl_pos = hl_studs_to_deid(hl_positive.keys(), stud_to_deid)
    print(f"  HeartLab: {len(hl_positive)} studies -> {len(hl_pos)} mapped, {len(hl_negated)} negated")

    # Source 3: Syngo free-text
    syngo_text_terms = ["amyloid", "cardiac amyloid", "ATTR", "transthyretin"]
    syngo_text_neg = [
        "no amyloid", "r/o amyloid", "rule out amyloid", "no evidence of amyloid",
        "renal amyloid", "hepatic amyloid",
        # Fix 2: Catch screening referrals
        "rule out cardiac involvement", "r/o cardiac amyloid",
        "rule out heart involvement", "rule out cardiac amyloid",
        # v7.2: Family history is not patient diagnosis
        "family history of amyloid", "family hx of amyloid",
        "family hx amyloid", "family history amyloid",
    ]
    syngo_text_pos, syngo_text_negated = query_syngo_freetext(conn, syngo_text_terms, syngo_text_neg,
                                                              exclude_indication=exclude_indication)
    syngo_text_mapped = syngo_refs_to_deid(syngo_text_pos.keys(), ref_to_deid, ref_to_patient)
    print(f"  Syngo text: {len(syngo_text_pos)} StudyRefs -> {len(syngo_text_mapped)} mapped, {len(syngo_text_negated)} negated")

    indication_only = sum(1 for v in syngo_text_pos.values() if v.get("field") == "Indication")

    all_pos = {}
    for src in [syngo_pos, hl_pos, syngo_text_mapped]:
        all_pos.update(src)
    print(f"  Total positive (pre-filter): {len(all_pos):,}")

    # Filter: require HeartLab/obs confirmation
    syngo_text_only_removed = 0
    if require_heartlab:
        confirmed = set(syngo_pos.keys()) | set(hl_pos.keys())
        syngo_text_only_removed = filter_syngo_text_only(all_pos, confirmed, "disease_amyloidosis")

    # === NEGATIVE: HCM cohort ===
    hcm_pos, _, _ = build_disease_hcm(conn, ref_to_deid, ref_to_patient, stud_to_deid, patient_splits, deid_to_patient,
                                      exclude_indication=exclude_indication, require_heartlab=require_heartlab)
    all_neg = {deid: pid for deid, pid in hcm_pos.items() if deid not in all_pos}
    print(f"  HCM negative (excluding amyloidosis positives): {len(all_neg):,}")

    provenance = build_provenance_report(
        "disease_amyloidosis",
        syngo_obs_count=len(syngo_pos),
        heartlab_count=len(hl_pos),
        syngo_text_count=len(syngo_text_mapped),
        heartlab_negated_count=len(hl_negated),
        syngo_text_negated_count=len(syngo_text_negated),
        total_pos=len(all_pos),
        total_neg=len(all_neg),
        indication_only_count=indication_only,
        syngo_text_only_removed=syngo_text_only_removed,
    )
    return all_pos, all_neg, provenance


def build_disease_takotsubo(conn, ref_to_deid, ref_to_patient, stud_to_deid, patient_splits, deid_to_patient,
                            exclude_indication=False, require_heartlab=False):
    """Build Takotsubo (positive) vs STEMI (negative)."""
    print("\n--- Building disease_takotsubo ---")

    # === POSITIVE ===
    # Source 1: Syngo obs — not expected
    syngo_obs_hits = query_syngo_obs_like(conn, "Cardiomyopathy_obs", ["%takotsubo%", "%tako%", "%apical balloon%"])
    syngo_pos = syngo_refs_to_deid(syngo_obs_hits.keys(), ref_to_deid, ref_to_patient)

    # Source 2: HeartLab
    hl_terms = ["takotsubo", "tako-tsubo", "apical ballooning", "stress cardiomyopath"]
    hl_neg_terms = ["no takotsubo", "r/o takotsubo", "no evidence of takotsubo", "unlikely takotsubo",
                    "no apical ballooning", "r/o apical ballooning"]
    hl_positive, hl_negated = query_heartlab_freetext(conn, hl_terms, hl_neg_terms)
    hl_pos = hl_studs_to_deid(hl_positive.keys(), stud_to_deid)
    print(f"  HeartLab: {len(hl_positive)} studies -> {len(hl_pos)} mapped, {len(hl_negated)} negated")

    # Source 3: Syngo free-text
    syngo_text_terms = ["takotsubo", "tako-tsubo", "apical ballooning", "stress cardiomyopathy"]
    syngo_text_neg = ["no takotsubo", "r/o takotsubo", "no evidence of takotsubo", "unlikely takotsubo"]
    syngo_text_pos, syngo_text_negated = query_syngo_freetext(conn, syngo_text_terms, syngo_text_neg,
                                                              exclude_indication=exclude_indication)
    syngo_text_mapped = syngo_refs_to_deid(syngo_text_pos.keys(), ref_to_deid, ref_to_patient)
    print(f"  Syngo text: {len(syngo_text_pos)} -> {len(syngo_text_mapped)} mapped, {len(syngo_text_negated)} negated")

    indication_only = sum(1 for v in syngo_text_pos.values() if v.get("field") == "Indication")

    all_pos = {}
    for src in [syngo_pos, hl_pos, syngo_text_mapped]:
        all_pos.update(src)
    print(f"  Total positive (pre-filter): {len(all_pos):,}")

    # Filter: require HeartLab/obs confirmation
    syngo_text_only_removed = 0
    if require_heartlab:
        confirmed = set(syngo_pos.keys()) | set(hl_pos.keys())
        syngo_text_only_removed = filter_syngo_text_only(all_pos, confirmed, "disease_takotsubo")

    # === NEGATIVE: STEMI ===
    stemi_pos, _, _ = build_disease_stemi(conn, ref_to_deid, ref_to_patient, stud_to_deid, patient_splits, deid_to_patient,
                                          exclude_indication=exclude_indication, require_heartlab=require_heartlab)
    all_neg = {deid: pid for deid, pid in stemi_pos.items() if deid not in all_pos}
    print(f"  STEMI negative (excluding takotsubo): {len(all_neg):,}")

    provenance = build_provenance_report(
        "disease_takotsubo",
        syngo_obs_count=len(syngo_pos),
        heartlab_count=len(hl_pos),
        syngo_text_count=len(syngo_text_mapped),
        heartlab_negated_count=len(hl_negated),
        syngo_text_negated_count=len(syngo_text_negated),
        total_pos=len(all_pos),
        total_neg=len(all_neg),
        indication_only_count=indication_only,
        syngo_text_only_removed=syngo_text_only_removed,
    )
    return all_pos, all_neg, provenance


def build_disease_stemi(conn, ref_to_deid, ref_to_patient, stud_to_deid, patient_splits, deid_to_patient,
                        exclude_indication=False, require_heartlab=False):
    """Build STEMI (positive) vs NSTEMI (negative)."""
    print("\n--- Building disease_stemi ---")

    # === POSITIVE: STEMI ===
    # Source 1: Syngo obs — DISABLED in v7
    # "Cardiomyopathy_obs = ischemic" is chronic ischemic cardiomyopathy,
    # NOT acute STEMI. Manual audit found 2/10 false positives from this source.
    syngo_obs_hits = {}
    syngo_pos = {}
    print(f"  Syngo obs: DISABLED (ischemic CMP != acute STEMI)")

    # Source 2: HeartLab — STEMI, but NOT NSTEMI
    hl_terms = ["STEMI", "ST elevation myocardial", "ST-elevation myocardial"]
    hl_neg_terms = ["NSTEMI", "non-STEMI", "non STEMI", "non-ST elevation",
                    # v7: "False STEMI" = ECG mimic, not real STEMI
                    "false STEMI", "False STEMI"]
    hl_positive, hl_negated = query_heartlab_freetext(conn, hl_terms, hl_neg_terms)
    hl_pos = hl_studs_to_deid(hl_positive.keys(), stud_to_deid)
    print(f"  HeartLab STEMI: {len(hl_positive)} studies -> {len(hl_pos)} mapped, {len(hl_negated)} negated")

    # Source 3: Syngo free-text
    syngo_text_terms = ["STEMI", "ST elevation myocardial", "ST-elevation myocardial"]
    syngo_text_neg = ["NSTEMI", "non-STEMI", "non STEMI", "non-ST elevation",
                      "false STEMI", "False STEMI"]
    syngo_text_pos, syngo_text_negated = query_syngo_freetext(conn, syngo_text_terms, syngo_text_neg,
                                                              exclude_indication=exclude_indication)
    syngo_text_mapped = syngo_refs_to_deid(syngo_text_pos.keys(), ref_to_deid, ref_to_patient)
    print(f"  Syngo text STEMI: {len(syngo_text_pos)} -> {len(syngo_text_mapped)} mapped, {len(syngo_text_negated)} negated")

    indication_only = sum(1 for v in syngo_text_pos.values() if v.get("field") == "Indication")

    all_pos = {}
    for src in [syngo_pos, hl_pos, syngo_text_mapped]:
        all_pos.update(src)
    print(f"  Total STEMI positive (pre-filter): {len(all_pos):,}")

    # Filter: require HeartLab/obs confirmation
    syngo_text_only_removed = 0
    if require_heartlab:
        confirmed = set(syngo_pos.keys()) | set(hl_pos.keys())
        syngo_text_only_removed = filter_syngo_text_only(all_pos, confirmed, "disease_stemi")

    # === NEGATIVE: NSTEMI ===
    hl_nstemi_terms = ["NSTEMI", "non-STEMI", "non-ST elevation", "NSTE-ACS"]
    hl_nstemi_neg = ["STEMI"]  # Exclude if also mentions STEMI (not non-STEMI)
    # Special handling: we need NSTEMI mentions but not those that also say "STEMI" without "N"
    hl_nstemi, hl_nstemi_negated = query_heartlab_freetext(conn, hl_nstemi_terms, None)
    hl_nstemi_mapped = hl_studs_to_deid(hl_nstemi.keys(), stud_to_deid)

    syngo_nstemi_terms = ["NSTEMI", "non-STEMI", "non-ST elevation"]
    syngo_nstemi_pos, _ = query_syngo_freetext(conn, syngo_nstemi_terms, None,
                                               exclude_indication=exclude_indication)
    syngo_nstemi_mapped = syngo_refs_to_deid(syngo_nstemi_pos.keys(), ref_to_deid, ref_to_patient)

    all_neg = {}
    for src in [hl_nstemi_mapped, syngo_nstemi_mapped]:
        for deid, pid in src.items():
            if deid not in all_pos:
                all_neg[deid] = pid
    print(f"  NSTEMI negative (excluding STEMI): {len(all_neg):,}")

    provenance = build_provenance_report(
        "disease_stemi",
        syngo_obs_count=len(syngo_pos),
        heartlab_count=len(hl_pos),
        syngo_text_count=len(syngo_text_mapped),
        heartlab_negated_count=len(hl_negated),
        syngo_text_negated_count=len(syngo_text_negated),
        total_pos=len(all_pos),
        total_neg=len(all_neg),
        indication_only_count=indication_only,
        syngo_text_only_removed=syngo_text_only_removed,
    )
    return all_pos, all_neg, provenance


def build_disease_endocarditis(conn, ref_to_deid, ref_to_patient, stud_to_deid, patient_splits, deid_to_patient,
                               exclude_indication=False, require_heartlab=False):
    """Build Endocarditis (positive) vs No-Endocarditis (negative).

    CRITICAL: 66% of endocarditis mentions are rule-outs. Aggressive negation filtering required.
    """
    print("\n--- Building disease_endocarditis ---")

    # === POSITIVE ===
    # Source 1: Syngo obs — EXCLUDED
    # "Endocarditis_possible" is a workflow/screening flag, not a confirmed diagnosis.
    # 0/160 mapped studies have any HeartLab endocarditis/vegetation mention.
    # These are "screen for endocarditis" studies, not "has endocarditis" studies.
    syngo_obs_hits = {}  # Disabled — Endocarditis_possible is a screening flag
    syngo_pos = {}
    print(f"  Syngo obs: DISABLED (Endocarditis_possible is a screening flag, not diagnosis)")

    # Source 2: HeartLab — with aggressive negation filtering
    hl_terms = ["endocarditis", "vegetation", "infective endocarditis", "bacterial endocarditis"]
    hl_neg_terms = [
        "rule out endocarditis", "r/o endocarditis", "no endocarditis",
        "no vegetation", "r/o vegetation", "rule out vegetation",
        "no evidence of vegetation", "no evidence of endocarditis",
        "to exclude endocarditis", "to exclude vegetation",
        "no definite vegetation", "no obvious vegetation",
        "R/O vegetation", "R/O endocarditis",
        "rule-out endocarditis", "rule-out vegetation",
        "no vegetations", "no evidence of vegetations",
        "without vegetation", "without endocarditis",
    ]
    hl_positive, hl_negated = query_heartlab_freetext(conn, hl_terms, hl_neg_terms)
    hl_pos = hl_studs_to_deid(hl_positive.keys(), stud_to_deid)
    print(f"  HeartLab: {len(hl_positive)} studies -> {len(hl_pos)} mapped, {len(hl_negated)} negated")

    # Source 3: Syngo free-text
    syngo_text_terms = ["endocarditis", "vegetation"]
    syngo_text_neg = [
        "rule out endocarditis", "r/o endocarditis", "no endocarditis",
        "rule out vegetation", "r/o vegetation", "no vegetation",
        "to exclude endocarditis", "to exclude vegetation",
        "no evidence of endocarditis", "no evidence of vegetation",
    ]
    syngo_text_pos, syngo_text_negated = query_syngo_freetext(conn, syngo_text_terms, syngo_text_neg,
                                                              exclude_indication=exclude_indication)
    syngo_text_mapped = syngo_refs_to_deid(syngo_text_pos.keys(), ref_to_deid, ref_to_patient)
    print(f"  Syngo text: {len(syngo_text_pos)} -> {len(syngo_text_mapped)} mapped, {len(syngo_text_negated)} negated")

    indication_only = sum(1 for v in syngo_text_pos.values() if v.get("field") == "Indication")

    all_pos = {}
    for src in [syngo_pos, hl_pos, syngo_text_mapped]:
        all_pos.update(src)
    print(f"  Total positive (pre-filter): {len(all_pos):,}")

    # Filter: require HeartLab/obs confirmation
    syngo_text_only_removed = 0
    if require_heartlab:
        confirmed = set(syngo_pos.keys()) | set(hl_pos.keys())
        syngo_text_only_removed = filter_syngo_text_only(all_pos, confirmed, "disease_endocarditis")

    # === NEGATIVE: Studies where endocarditis was explicitly ruled out ===
    hl_negated_studs = query_heartlab_negated_only(conn, hl_terms, hl_neg_terms)
    hl_ruled_out = hl_studs_to_deid(hl_negated_studs, stud_to_deid)

    all_neg = {deid: pid for deid, pid in hl_ruled_out.items() if deid not in all_pos}
    print(f"  Ruled-out negative: {len(all_neg):,}")

    provenance = build_provenance_report(
        "disease_endocarditis",
        syngo_obs_count=len(syngo_pos),
        heartlab_count=len(hl_pos),
        syngo_text_count=len(syngo_text_mapped),
        heartlab_negated_count=len(hl_negated) + len(hl_negated_studs),
        syngo_text_negated_count=len(syngo_text_negated),
        total_pos=len(all_pos),
        total_neg=len(all_neg),
        indication_only_count=indication_only,
        syngo_text_only_removed=syngo_text_only_removed,
    )
    return all_pos, all_neg, provenance


def build_disease_dcm(conn, ref_to_deid, ref_to_patient, stud_to_deid, patient_splits, deid_to_patient,
                      exclude_indication=False, require_heartlab=False):
    """Build DCM (positive) vs HF-without-DCM (negative)."""
    print("\n--- Building disease_dcm ---")

    # === POSITIVE ===
    # Source 1: Syngo obs
    syngo_obs_hits = query_syngo_obs(conn, "Cardiomyopathy_obs", ["dilated", "idiopathic"])
    syngo_pos = syngo_refs_to_deid(syngo_obs_hits.keys(), ref_to_deid, ref_to_patient)
    print(f"  Syngo obs: {len(syngo_obs_hits)} StudyRefs -> {len(syngo_pos)} mapped")

    # Source 2: HeartLab
    hl_terms = ["dilated cardiomyopath", "non-ischemic dilated"]
    hl_neg_terms = [
        "no DCM", "no dilated cardiomyopath", "r/o DCM", "no evidence of DCM",
        "r/o dilated", "rule out dilated",
        # Fix 2: Catch "at risk" mentions (not yet diagnosed)
        "at risk for DCM", "at risk for dilated", "risk of DCM", "risk of dilated",
    ]
    hl_positive, hl_negated = query_heartlab_freetext(conn, hl_terms, hl_neg_terms)

    # Also search DCM but require cardiac context to avoid false matches
    hl_dcm, hl_dcm_neg = query_heartlab_freetext(
        conn,
        ["DCM"],
        ["no DCM", "r/o DCM", "rule out DCM",
         "at risk for DCM", "risk of DCM"],
    )
    # Filter DCM: keep only if note also mentions cardiomyopathy or heart
    dcm_filtered = {}
    cardiac_pattern = re.compile(r"(cardiomyopath|heart|cardiac|ventricul|ejection)", re.IGNORECASE)
    for stud_id, notes in hl_dcm.items():
        for note in notes:
            if cardiac_pattern.search(note or ""):
                dcm_filtered[stud_id] = notes
                break

    hl_positive.update(dcm_filtered)
    hl_pos = hl_studs_to_deid(hl_positive.keys(), stud_to_deid)
    print(f"  HeartLab: {len(hl_positive)} studies -> {len(hl_pos)} mapped, {len(hl_negated)} negated")

    # NICM search
    hl_nicm, _ = query_heartlab_freetext(conn, ["NICM"], ["no NICM"])
    nicm_mapped = hl_studs_to_deid(hl_nicm.keys(), stud_to_deid)
    print(f"  HeartLab NICM: {len(hl_nicm)} studies -> {len(nicm_mapped)} mapped")

    # Source 3: Syngo free-text
    syngo_text_terms = ["dilated cardiomyopathy", "DCM", "NICM"]
    syngo_text_neg = [
        "no DCM", "no dilated", "r/o DCM", "no evidence of DCM",
        # Fix 2: Catch "at risk" mentions
        "at risk for DCM", "at risk for dilated", "risk of DCM", "risk of dilated",
    ]
    syngo_text_pos, syngo_text_negated = query_syngo_freetext(conn, syngo_text_terms, syngo_text_neg,
                                                              exclude_indication=exclude_indication)
    syngo_text_mapped = syngo_refs_to_deid(syngo_text_pos.keys(), ref_to_deid, ref_to_patient)
    print(f"  Syngo text: {len(syngo_text_pos)} -> {len(syngo_text_mapped)} mapped, {len(syngo_text_negated)} negated")

    indication_only = sum(1 for v in syngo_text_pos.values() if v.get("field") == "Indication")

    all_pos = {}
    for src in [syngo_pos, hl_pos, nicm_mapped, syngo_text_mapped]:
        all_pos.update(src)
    print(f"  Total positive (pre-filter): {len(all_pos):,}")

    # Filter: require HeartLab/obs confirmation
    syngo_text_only_removed = 0
    if require_heartlab:
        confirmed = set(syngo_pos.keys()) | set(hl_pos.keys()) | set(nicm_mapped.keys())
        syngo_text_only_removed = filter_syngo_text_only(all_pos, confirmed, "disease_dcm")

    # === NEGATIVE: HF without DCM ===
    hl_hf_terms = ["heart failure", "HFrEF", "HFpEF", "reduced ejection fraction",
                   "systolic dysfunction", "severely reduced"]
    hl_hf, _ = query_heartlab_freetext(conn, hl_hf_terms, None)
    hl_hf_mapped = hl_studs_to_deid(hl_hf.keys(), stud_to_deid)

    syngo_hf_terms = ["heart failure", "HFrEF", "HFpEF", "reduced ejection fraction"]
    syngo_hf_pos, _ = query_syngo_freetext(conn, syngo_hf_terms, None,
                                           exclude_indication=exclude_indication)
    syngo_hf_mapped = syngo_refs_to_deid(syngo_hf_pos.keys(), ref_to_deid, ref_to_patient)

    all_neg = {}
    for src in [hl_hf_mapped, syngo_hf_mapped]:
        for deid, pid in src.items():
            if deid not in all_pos:
                all_neg[deid] = pid
    print(f"  HF-without-DCM negative: {len(all_neg):,}")

    provenance = build_provenance_report(
        "disease_dcm",
        syngo_obs_count=len(syngo_pos),
        heartlab_count=len(hl_pos) + len(nicm_mapped),
        syngo_text_count=len(syngo_text_mapped),
        heartlab_negated_count=len(hl_negated),
        syngo_text_negated_count=len(syngo_text_negated),
        total_pos=len(all_pos),
        total_neg=len(all_neg),
        indication_only_count=indication_only,
        syngo_text_only_removed=syngo_text_only_removed,
    )
    return all_pos, all_neg, provenance


def build_disease_bicuspid_av(conn, ref_to_deid, ref_to_patient, stud_to_deid, patient_splits, deid_to_patient,
                              exclude_indication=False, require_heartlab=False):
    """Build Bicuspid AV (positive) vs Tricuspid AV (negative).

    HIGH confidence — structural finding with strong Syngo structured labels.
    """
    print("\n--- Building disease_bicuspid_av ---")

    # === POSITIVE ===
    # Source 1: Syngo obs — AoV_structure_uhn_obs (primary) + AoV_structure_sD_obs
    bicuspid_values = [
        "bicuspid", "bicuspid_10_4", "bicuspid_1_7", "bicuspid_9_3",
        "functional_bicuspid",
    ]
    syngo_obs_uhn = query_syngo_obs(conn, "AoV_structure_uhn_obs", bicuspid_values)
    syngo_obs_sd = query_syngo_obs(conn, "AoV_structure_sD_obs", ["bicuspid", "bicuspid_w_doming"])
    # Also catch free-text bicuspid values in the obs
    syngo_obs_like_uhn = query_syngo_obs_like(conn, "AoV_structure_uhn_obs", ["%bicuspid%"])
    syngo_obs_like_sd = query_syngo_obs_like(conn, "AoV_structure_sD_obs", ["%bicuspid%"])

    all_syngo_refs = {}
    for d in [syngo_obs_uhn, syngo_obs_sd, syngo_obs_like_uhn, syngo_obs_like_sd]:
        all_syngo_refs.update(d)
    syngo_pos = syngo_refs_to_deid(all_syngo_refs.keys(), ref_to_deid, ref_to_patient)
    print(f"  Syngo obs bicuspid: {len(all_syngo_refs)} StudyRefs -> {len(syngo_pos)} mapped")

    # Source 2: HeartLab
    hl_terms = ["bicuspid", "BAV"]
    hl_neg_terms = ["no bicuspid", "r/o bicuspid", "unlikely bicuspid", "not bicuspid"]
    hl_positive, hl_negated = query_heartlab_freetext(conn, hl_terms, hl_neg_terms)
    hl_pos = hl_studs_to_deid(hl_positive.keys(), stud_to_deid)
    print(f"  HeartLab: {len(hl_positive)} studies -> {len(hl_pos)} mapped, {len(hl_negated)} negated")

    # Source 3: Syngo free-text
    syngo_text_terms = ["bicuspid", "BAV"]
    syngo_text_neg = ["no bicuspid", "r/o bicuspid", "unlikely bicuspid"]
    syngo_text_pos, syngo_text_negated = query_syngo_freetext(conn, syngo_text_terms, syngo_text_neg,
                                                              exclude_indication=exclude_indication)
    syngo_text_mapped = syngo_refs_to_deid(syngo_text_pos.keys(), ref_to_deid, ref_to_patient)
    print(f"  Syngo text: {len(syngo_text_pos)} -> {len(syngo_text_mapped)} mapped, {len(syngo_text_negated)} negated")

    indication_only = sum(1 for v in syngo_text_pos.values() if v.get("field") == "Indication")

    all_pos = {}
    for src in [syngo_pos, hl_pos, syngo_text_mapped]:
        all_pos.update(src)
    print(f"  Total positive (pre-filter): {len(all_pos):,}")

    # Filter: require HeartLab/obs confirmation
    syngo_text_only_removed = 0
    if require_heartlab:
        confirmed = set(syngo_pos.keys()) | set(hl_pos.keys())
        syngo_text_only_removed = filter_syngo_text_only(all_pos, confirmed, "disease_bicuspid_av")

    # === NEGATIVE: Tricuspid AV ===
    tricuspid_obs_uhn = query_syngo_obs(conn, "AoV_structure_uhn_obs", ["tricuspid"])
    tricuspid_obs_sd = query_syngo_obs(conn, "AoV_structure_sD_obs", ["tricuspid", "normal"])
    all_tricuspid_refs = {}
    all_tricuspid_refs.update(tricuspid_obs_uhn)
    all_tricuspid_refs.update(tricuspid_obs_sd)
    tricuspid_neg = syngo_refs_to_deid(all_tricuspid_refs.keys(), ref_to_deid, ref_to_patient)

    # HeartLab tricuspid AV — SENTENCE matches "The aortic valve is tricuspid/trileaflet."
    hl_tri, _ = query_heartlab_freetext(conn, ["aortic valve is tricuspid", "trileaflet"], None)
    hl_tri_mapped = hl_studs_to_deid(hl_tri.keys(), stud_to_deid)

    all_neg = {}
    for src in [tricuspid_neg, hl_tri_mapped]:
        for deid, pid in src.items():
            if deid not in all_pos:
                all_neg[deid] = pid
    print(f"  Tricuspid AV negative: {len(all_neg):,}")

    provenance = build_provenance_report(
        "disease_bicuspid_av",
        syngo_obs_count=len(syngo_pos),
        heartlab_count=len(hl_pos),
        syngo_text_count=len(syngo_text_mapped),
        heartlab_negated_count=len(hl_negated),
        syngo_text_negated_count=len(syngo_text_negated),
        total_pos=len(all_pos),
        total_neg=len(all_neg),
        indication_only_count=indication_only,
        syngo_text_only_removed=syngo_text_only_removed,
    )
    return all_pos, all_neg, provenance


def build_disease_myxomatous_mv(conn, ref_to_deid, ref_to_patient, stud_to_deid, patient_splits, deid_to_patient,
                                exclude_indication=False, require_heartlab=False):
    """Build Myxomatous MV (positive) vs Non-Myxomatous MR (negative)."""
    print("\n--- Building disease_myxomatous_mv ---")

    # === POSITIVE ===
    # Source 1: Syngo obs
    syngo_obs_hits = query_syngo_obs(conn, "MV_Structure_functionuhn_obs", ["myxomatous"])
    syngo_pos = syngo_refs_to_deid(syngo_obs_hits.keys(), ref_to_deid, ref_to_patient)
    print(f"  Syngo obs myxomatous: {len(syngo_obs_hits)} StudyRefs -> {len(syngo_pos)} mapped")

    # Source 2: HeartLab
    hl_terms = ["myxomatous", "mitral valve prolapse", "MVP", "Barlow"]
    hl_neg_terms = ["no MVP", "no prolapse", "r/o MVP", "no myxomatous",
                    "no mitral valve prolapse", "r/o mitral valve prolapse",
                    "without prolapse",
                    # v7: Clinical denial of MVP criteria
                    "does not meet criteria for MVP",
                    "does not meet criteria for prolapse",
                    "does not meet criteria for mitral valve prolapse",
                    "borderline mitral valve prolapse",
                    # v7.1: Intervening-adjective negation (audit: "No definite MVP")
                    "no definite mitral valve prolapse",
                    "no definite prolapse",
                    "no obvious mitral valve prolapse",
                    "no obvious prolapse",
                    "no clear mitral valve prolapse",
                    "no significant prolapse"]
    # v7.2: Exclude VALVE/ROOT DISEASE group (indication/referral entries like
    # "Assess mitral valve prolapse" that match "prolapse" but are not findings)
    hl_positive, hl_negated = query_heartlab_freetext(
        conn, hl_terms, hl_neg_terms,
        exclude_finding_groups=["VALVE/ROOT  DISEASE"])
    hl_pos = hl_studs_to_deid(hl_positive.keys(), stud_to_deid)
    print(f"  HeartLab: {len(hl_positive)} studies -> {len(hl_pos)} mapped, {len(hl_negated)} negated")

    # Source 3: Syngo free-text
    syngo_text_terms = ["myxomatous", "mitral valve prolapse", "MVP", "Barlow"]
    syngo_text_neg = ["no MVP", "no prolapse", "r/o MVP", "no myxomatous"]
    syngo_text_pos, syngo_text_negated = query_syngo_freetext(conn, syngo_text_terms, syngo_text_neg,
                                                              exclude_indication=exclude_indication)
    syngo_text_mapped = syngo_refs_to_deid(syngo_text_pos.keys(), ref_to_deid, ref_to_patient)
    print(f"  Syngo text: {len(syngo_text_pos)} -> {len(syngo_text_mapped)} mapped, {len(syngo_text_negated)} negated")

    indication_only = sum(1 for v in syngo_text_pos.values() if v.get("field") == "Indication")

    all_pos = {}
    for src in [syngo_pos, hl_pos, syngo_text_mapped]:
        all_pos.update(src)
    print(f"  Total positive (pre-filter): {len(all_pos):,}")

    # Filter: require HeartLab/obs confirmation
    syngo_text_only_removed = 0
    if require_heartlab:
        confirmed = set(syngo_pos.keys()) | set(hl_pos.keys())
        syngo_text_only_removed = filter_syngo_text_only(all_pos, confirmed, "disease_myxomatous_mv")

    # === NEGATIVE: Non-myxomatous MR ===
    # Use MR severity source CSVs to find studies with MR (any grade > none)
    mr_findings = REPO_ROOT / "data" / "aws" / "aws_syngo_findings_v2.csv"
    mr_hl_findings = REPO_ROOT / "data" / "aws" / "aws_heartlab_findings_v2.csv"

    mr_classes = ["MR_trace", "MR_mild", "MR_moderate", "MR_severe"]
    deid_to_pat = load_deid_to_patient()

    mr_studies = set()
    if mr_findings.exists():
        syngo_f = pd.read_csv(mr_findings)
        mr_syngo = syngo_f[syngo_f["common_label"].isin(mr_classes)]["DeidentifiedStudyID"].unique()
        mr_studies.update(mr_syngo)
    if mr_hl_findings.exists():
        hl_f = pd.read_csv(mr_hl_findings)
        mr_hl = hl_f[hl_f["common_label"].isin(mr_classes)]["DeidentifiedStudyID"].unique()
        mr_studies.update(mr_hl)

    all_neg = {}
    for deid in mr_studies:
        if deid not in all_pos:
            pid = deid_to_pat.get(deid)
            if pid:
                all_neg[deid] = pid
    print(f"  Non-myxomatous MR negative: {len(all_neg):,}")

    provenance = build_provenance_report(
        "disease_myxomatous_mv",
        syngo_obs_count=len(syngo_pos),
        heartlab_count=len(hl_pos),
        syngo_text_count=len(syngo_text_mapped),
        heartlab_negated_count=len(hl_negated),
        syngo_text_negated_count=len(syngo_text_negated),
        total_pos=len(all_pos),
        total_neg=len(all_neg),
        indication_only_count=indication_only,
        syngo_text_only_removed=syngo_text_only_removed,
    )
    return all_pos, all_neg, provenance


def build_disease_rheumatic_mv(conn, ref_to_deid, ref_to_patient, stud_to_deid, patient_splits, deid_to_patient,
                               exclude_indication=False, require_heartlab=False):
    """Build Rheumatic MV (positive) vs Non-Rheumatic MS (negative)."""
    print("\n--- Building disease_rheumatic_mv ---")

    # === POSITIVE ===
    # Source 1: Syngo obs
    syngo_obs_hits = query_syngo_obs(conn, "MV_Structure_functionuhn_obs", ["rheumatic", "mild_rheumatic_no_ms"])
    syngo_pos = syngo_refs_to_deid(syngo_obs_hits.keys(), ref_to_deid, ref_to_patient)
    print(f"  Syngo obs rheumatic: {len(syngo_obs_hits)} StudyRefs -> {len(syngo_pos)} mapped")

    # Source 2: HeartLab
    hl_terms = ["rheumatic", "RHD", "rheumatic heart disease", "rheumatic mitral"]
    hl_neg_terms = ["no rheumatic", "non-rheumatic", "r/o rheumatic", "unlikely rheumatic",
                    "no evidence of rheumatic"]
    hl_positive, hl_negated = query_heartlab_freetext(conn, hl_terms, hl_neg_terms)
    hl_pos = hl_studs_to_deid(hl_positive.keys(), stud_to_deid)
    print(f"  HeartLab: {len(hl_positive)} studies -> {len(hl_pos)} mapped, {len(hl_negated)} negated")

    # Source 3: Syngo free-text
    syngo_text_terms = ["rheumatic", "RHD", "rheumatic heart disease"]
    syngo_text_neg = ["no rheumatic", "non-rheumatic", "r/o rheumatic"]
    syngo_text_pos, syngo_text_negated = query_syngo_freetext(conn, syngo_text_terms, syngo_text_neg,
                                                              exclude_indication=exclude_indication)
    syngo_text_mapped = syngo_refs_to_deid(syngo_text_pos.keys(), ref_to_deid, ref_to_patient)
    print(f"  Syngo text: {len(syngo_text_pos)} -> {len(syngo_text_mapped)} mapped, {len(syngo_text_negated)} negated")

    indication_only = sum(1 for v in syngo_text_pos.values() if v.get("field") == "Indication")

    all_pos = {}
    for src in [syngo_pos, hl_pos, syngo_text_mapped]:
        all_pos.update(src)
    print(f"  Total positive (pre-filter): {len(all_pos):,}")

    # Filter: require HeartLab/obs confirmation
    syngo_text_only_removed = 0
    if require_heartlab:
        confirmed = set(syngo_pos.keys()) | set(hl_pos.keys())
        syngo_text_only_removed = filter_syngo_text_only(all_pos, confirmed, "disease_rheumatic_mv")

    # === NEGATIVE: Non-rheumatic mitral stenosis ===
    # CRITICAL: Must filter negation — "There is no mitral stenosis" is the most common
    # SENTENCE containing "mitral stenosis" and must NOT be included as a negative.
    # v7.1: Also exclude VALVE/ROOT DISEASE and CONGENITAL HEART DISEASE finding groups —
    # these are indication/referral entries ("Mitral stenosis" as reason for echo) not
    # diagnostic findings. Studies with only indication-group matches often have
    # "There is no mitral stenosis" as the actual finding.
    hl_ms_terms = ["mitral stenosis"]
    hl_ms_neg_terms = ["no mitral stenosis", "no evidence of mitral stenosis",
                       "without mitral stenosis", "r/o mitral stenosis",
                       "rule out mitral stenosis"]
    hl_ms, _ = query_heartlab_freetext(conn, hl_ms_terms, hl_ms_neg_terms,
                                       exclude_finding_groups=["VALVE/ROOT  DISEASE",
                                                               "CONGENITAL HEART DISEASE"])
    hl_ms_mapped = hl_studs_to_deid(hl_ms.keys(), stud_to_deid)

    syngo_ms_terms = ["mitral stenosis"]
    syngo_ms_neg = ["no mitral stenosis", "no evidence of mitral stenosis",
                    "without mitral stenosis"]
    syngo_ms_pos, _ = query_syngo_freetext(conn, syngo_ms_terms, syngo_ms_neg,
                                           exclude_indication=exclude_indication)
    syngo_ms_mapped = syngo_refs_to_deid(syngo_ms_pos.keys(), ref_to_deid, ref_to_patient)

    all_neg = {}
    for src in [hl_ms_mapped, syngo_ms_mapped]:
        for deid, pid in src.items():
            if deid not in all_pos:
                all_neg[deid] = pid
    print(f"  Non-rheumatic MS negative: {len(all_neg):,}")

    provenance = build_provenance_report(
        "disease_rheumatic_mv",
        syngo_obs_count=len(syngo_pos),
        heartlab_count=len(hl_pos),
        syngo_text_count=len(syngo_text_mapped),
        heartlab_negated_count=len(hl_negated),
        syngo_text_negated_count=len(syngo_text_negated),
        total_pos=len(all_pos),
        total_neg=len(all_neg),
        indication_only_count=indication_only,
        syngo_text_only_removed=syngo_text_only_removed,
    )
    return all_pos, all_neg, provenance


# ---------------------------------------------------------------------------
# NPZ assembly
# ---------------------------------------------------------------------------


def propagate_to_patient(study_dict, deid_to_patient, patient_to_studies):
    """Propagate study-level labels to all studies for the same patient.

    If any study for a patient is positive, all that patient's studies become positive.
    This matches the old notebook's behavior (patient-level labeling).

    Args:
        study_dict: {DeidentifiedStudyID: patient_id} for source studies
        deid_to_patient: {DeidentifiedStudyID: patient_id} for all studies
        patient_to_studies: {patient_id: [list of DeidentifiedStudyIDs]}

    Returns:
        Expanded dict of {DeidentifiedStudyID: patient_id} including propagated studies.
    """
    # Find all patient IDs from source studies
    source_patients = set()
    for deid, pid in study_dict.items():
        canonical = deid_to_patient.get(deid, pid)
        source_patients.add(str(canonical))

    # Propagate to all studies for those patients
    propagated = {}
    n_source = len(study_dict)
    for pid in source_patients:
        for deid in patient_to_studies.get(pid, []):
            propagated[deid] = pid

    n_propagated = len(propagated) - n_source
    if n_propagated > 0:
        print(f"  Patient propagation: {n_source:,} source -> {len(propagated):,} total (+{n_propagated:,})")
    return propagated


def build_patient_to_studies(deid_to_patient):
    """Build reverse mapping: patient_id -> [list of DeidentifiedStudyIDs]."""
    patient_to_studies = defaultdict(list)
    for deid, pid in deid_to_patient.items():
        patient_to_studies[str(pid)].append(deid)
    return dict(patient_to_studies)


def cohorts_to_npz(all_pos, all_neg, patient_splits, deid_to_patient, patient_propagation=False):
    """Convert positive/negative deid dicts to NPZ arrays with patient split filtering.

    If patient_propagation=True, positive labels are propagated to all studies
    for the same patient (matches old notebook behavior).
    """
    if patient_propagation:
        patient_to_studies = build_patient_to_studies(deid_to_patient)
        all_pos = propagate_to_patient(all_pos, deid_to_patient, patient_to_studies)

    records = []
    for deid, pid in all_pos.items():
        # Use deid_to_patient for canonical patient_id if available
        canonical_pid = deid_to_patient.get(deid, pid)
        if str(canonical_pid) in patient_splits:
            records.append({
                "study_id": deid,
                "patient_id": str(canonical_pid),
                "label": 1,
                "split": patient_splits[str(canonical_pid)],
            })
    for deid, pid in all_neg.items():
        canonical_pid = deid_to_patient.get(deid, pid)
        if str(canonical_pid) in patient_splits:
            records.append({
                "study_id": deid,
                "patient_id": str(canonical_pid),
                "label": 0,
                "split": patient_splits[str(canonical_pid)],
            })

    if not records:
        print("  WARNING: No studies passed patient filter!")
        return {
            "study_ids": np.array([], dtype=object),
            "patient_ids": np.array([], dtype=object),
            "labels": np.array([], dtype=np.int32),
            "splits": np.array([], dtype=object),
        }

    df = pd.DataFrame(records)
    # Deduplicate: same study_id should not appear in both pos and neg
    # If it does, positive label wins (disease mention overrides)
    df = df.sort_values("label", ascending=False).drop_duplicates(subset="study_id", keep="first")

    n_pos = (df["label"] == 1).sum()
    n_neg = (df["label"] == 0).sum()
    n_train = (df["split"] == "train").sum()
    n_val = (df["split"] == "val").sum()
    n_test = (df["split"] == "test").sum()
    print(f"  Final: {len(df):,} studies (pos={n_pos:,}, neg={n_neg:,}) | train={n_train:,}, val={n_val:,}, test={n_test:,}")

    return {
        "study_ids": df["study_id"].values.astype(object),
        "patient_ids": df["patient_id"].values.astype(object),
        "labels": df["label"].values.astype(np.int32),
        "splits": df["split"].values.astype(object),
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_against_existing(task, arrays, existing_dir):
    """Compare new NPZ against existing."""
    existing_path = existing_dir / f"{task}.npz"
    if not existing_path.exists():
        print(f"  [VALIDATE] No existing NPZ at {existing_path}")
        return

    existing = np.load(str(existing_path), allow_pickle=True)
    n_new = len(arrays["study_ids"])
    n_old = len(existing["study_ids"])

    old_sids = set(existing["study_ids"])
    new_sids = set(arrays["study_ids"])
    overlap = old_sids & new_sids
    in_old_not_new = old_sids - new_sids
    in_new_not_old = new_sids - old_sids

    print(f"  [VALIDATE] Existing: {n_old:,}, New: {n_new:,}")
    print(f"  [VALIDATE] Overlap: {len(overlap):,} ({100*len(overlap)/max(n_old,1):.1f}% of old)")
    print(f"  [VALIDATE] In old not new: {len(in_old_not_new):,}, In new not old: {len(in_new_not_old):,}")

    # Label comparison for overlapping studies
    if overlap:
        old_dict = dict(zip(existing["study_ids"], existing["labels"]))
        new_dict = dict(zip(arrays["study_ids"], arrays["labels"]))
        label_match = sum(1 for s in overlap if int(old_dict[s]) == int(new_dict[s]))
        label_mismatch = len(overlap) - label_match
        print(f"  [VALIDATE] Label match: {label_match:,}/{len(overlap):,} ({100*label_match/len(overlap):.1f}%)")
        if label_mismatch > 0:
            # Show some mismatches
            mismatches = [(s, int(old_dict[s]), int(new_dict[s])) for s in overlap if int(old_dict[s]) != int(new_dict[s])]
            print(f"  [VALIDATE] Label mismatches (first 5): {mismatches[:5]}")

    # Split comparison
    if overlap:
        old_splits = dict(zip(existing["study_ids"], existing["splits"]))
        new_splits = dict(zip(arrays["study_ids"], arrays["splits"]))
        split_match = sum(1 for s in overlap if old_splits[s] == new_splits[s])
        print(f"  [VALIDATE] Split match: {split_match:,}/{len(overlap):,}")

    # Check positive/negative count changes
    old_pos = int((existing["labels"] == 1).sum())
    old_neg = int((existing["labels"] == 0).sum())
    new_pos = int((arrays["labels"] == 1).sum())
    new_neg = int((arrays["labels"] == 0).sum())
    pos_pct = abs(new_pos - old_pos) / max(old_pos, 1) * 100
    neg_pct = abs(new_neg - old_neg) / max(old_neg, 1) * 100
    print(f"  [VALIDATE] Pos: {old_pos:,} -> {new_pos:,} ({'+' if new_pos >= old_pos else ''}{new_pos-old_pos:,}, {pos_pct:.1f}% change)")
    print(f"  [VALIDATE] Neg: {old_neg:,} -> {new_neg:,} ({'+' if new_neg >= old_neg else ''}{new_neg-old_neg:,}, {neg_pct:.1f}% change)")

    if pos_pct > 20 or neg_pct > 20:
        print(f"  [VALIDATE] WARNING: >20% count change detected!")


def save_npz(task, arrays, output_dir):
    """Save arrays to NPZ file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{task}.npz"
    np.savez(str(path), **arrays)
    print(f"  Saved: {path} ({len(arrays['study_ids']):,} studies)")


def save_provenance(task, provenance, output_dir):
    """Save provenance JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"provenance_{task.replace('disease_', '')}.json"
    with open(path, "w") as f:
        json.dump(provenance, f, indent=2)
    print(f"  Provenance: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

BUILDERS = {
    "disease_hcm": build_disease_hcm,
    "disease_amyloidosis": build_disease_amyloidosis,
    "disease_takotsubo": build_disease_takotsubo,
    "disease_stemi": build_disease_stemi,
    "disease_dcm": build_disease_dcm,
    "disease_bicuspid_av": build_disease_bicuspid_av,
    "disease_myxomatous_mv": build_disease_myxomatous_mv,
    "disease_rheumatic_mv": build_disease_rheumatic_mv,
}


def main():
    parser = argparse.ArgumentParser(description="Build disease detection label NPZ files")
    parser.add_argument("--task", type=str, choices=DISEASES, help="Build a single disease")
    parser.add_argument("--all", action="store_true", help="Build all 9 diseases")
    parser.add_argument("--output_dir", type=str, default="labels_v2", help="Output directory (relative to script dir)")
    parser.add_argument("--validate", action="store_true", help="Compare against existing NPZs in labels/")
    parser.add_argument(
        "--patient_propagation", action="store_true",
        help="Propagate positive labels to all studies for the same patient "
        "(matches old notebook behavior). Without this flag, only studies where "
        "the disease term was directly found are labeled positive.",
    )
    parser.add_argument(
        "--exclude_indication", action="store_true",
        help="Exclude Indication and REASON_FOR_STUDY fields from Syngo free-text "
        "queries. These fields contain referral reasons, not diagnostic findings.",
    )
    parser.add_argument(
        "--require_heartlab", action="store_true",
        help="Require HeartLab SENTENCE/NOTE or Syngo structured obs confirmation "
        "for positive labels. Removes Syngo-text-only positives (studies where "
        "only StudyComments/PatientHistory mention the disease).",
    )
    args = parser.parse_args()

    if not args.task and not args.all:
        parser.error("Specify --task or --all")

    tasks = DISEASES if args.all else [args.task]
    output_dir = SCRIPT_DIR / args.output_dir

    print("=" * 60)
    print("Disease Detection Label Builder")
    print(f"  Patient propagation: {'ON' if args.patient_propagation else 'OFF (study-level only)'}")
    print(f"  Exclude indication:  {'ON' if args.exclude_indication else 'OFF (all Syngo fields)'}")
    print(f"  Require HeartLab:    {'ON' if args.require_heartlab else 'OFF (Syngo text accepted)'}")
    print("=" * 60)

    # Load shared data
    print("\nLoading shared data...")
    patient_splits = load_patient_splits()
    ref_to_deid, ref_to_patient = load_studyref_mapping()
    deid_to_patient = load_deid_to_patient()
    stud_to_deid = load_hl_stud_to_deid()

    conn = sqlite3.connect(str(ECHO_DB))

    # Track which diseases have already been built (for cross-references)
    built_cache = {}
    all_provenances = []

    for task in tasks:
        builder = BUILDERS[task]
        all_pos, all_neg, provenance = builder(
            conn, ref_to_deid, ref_to_patient, stud_to_deid, patient_splits, deid_to_patient,
            exclude_indication=args.exclude_indication,
            require_heartlab=args.require_heartlab,
        )
        built_cache[task] = (all_pos, all_neg)

        arrays = cohorts_to_npz(all_pos, all_neg, patient_splits, deid_to_patient, args.patient_propagation)
        save_npz(task, arrays, output_dir)
        save_provenance(task, provenance, output_dir)
        all_provenances.append(provenance)

        if args.validate:
            validate_against_existing(task, arrays, EXISTING_LABELS)

    conn.close()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for task in tasks:
        path = output_dir / f"{task}.npz"
        npz = np.load(str(path), allow_pickle=True)
        labels = npz["labels"]
        splits = npz["splits"]
        n_pos = int((labels == 1).sum())
        n_neg = int((labels == 0).sum())
        n_train = int((splits == "train").sum())
        n_val = int((splits == "val").sum())
        n_test = int((splits == "test").sum())
        print(f"  {task}: {len(labels):,} total (pos={n_pos:,}, neg={n_neg:,}) | train={n_train:,}, val={n_val:,}, test={n_test:,}")

    # Save combined provenance
    combined_path = output_dir / "provenance_all.json"
    with open(combined_path, "w") as f:
        json.dump(all_provenances, f, indent=2)
    print(f"\nCombined provenance: {combined_path}")


if __name__ == "__main__":
    main()
