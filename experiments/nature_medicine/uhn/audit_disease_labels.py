#!/usr/bin/env python3
"""Audit disease labels by sampling random positives and retrieving matched text.

For each disease, samples N random positive studies and retrieves all
HeartLab SENTENCE/NOTE text and Syngo free-text that contain the disease
search terms. Outputs a TSV for manual review.

Usage:
    python audit_disease_labels.py --n 50 --output_dir audit_v3
"""

import argparse
import json
import random
import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
ECHO_DB = REPO_ROOT / "uhn_echo" / "nature_medicine" / "data_exploration" / "echo.db"
AWS_SYNGO_CSV = REPO_ROOT / "data" / "aws" / "aws_syngo.csv"

# Disease search terms (matching build_disease_labels.py)
DISEASE_TERMS = {
    "disease_hcm": {
        "hl_terms": ["hypertrophic cardiomyopath", "HCM", "HOCM", "hypertrophic obstructive",
                      "asymmetric septal hypertrophy", "SAM", "systolic anterior motion"],
        "syngo_terms": ["hypertrophic cardiomyopathy", "HOCM", "HCM"],
        "syngo_obs": [("Cardiomyopathy_obs", ["HOCM", "HCM", "Hypertrophic", "hypertrophic"])],
    },
    "disease_amyloidosis": {
        "hl_terms": ["amyloid", "cardiac amyloid", "ATTR", "transthyretin"],
        "syngo_terms": ["amyloid", "cardiac amyloid", "ATTR", "transthyretin"],
        "syngo_obs": [("Cardiomyopathy_obs", ["%amyloid%", "%infiltrative%"])],
    },
    "disease_takotsubo": {
        "hl_terms": ["takotsubo", "tako-tsubo", "apical ballooning", "stress cardiomyopath"],
        "syngo_terms": ["takotsubo", "tako-tsubo", "apical ballooning", "stress cardiomyopathy"],
        "syngo_obs": [],
    },
    "disease_stemi": {
        "hl_terms": ["STEMI", "ST elevation myocardial", "ST-elevation myocardial"],
        "syngo_terms": ["STEMI", "ST elevation myocardial", "ST-elevation myocardial"],
        "syngo_obs": [("Cardiomyopathy_obs", ["ischemic"])],
    },
    "disease_endocarditis": {
        "hl_terms": ["endocarditis", "vegetation", "infective endocarditis", "bacterial endocarditis"],
        "syngo_terms": ["endocarditis", "vegetation"],
        "syngo_obs": [("UHN_Endocarditis_obs", ["Endocarditis_possible"])],
    },
    "disease_dcm": {
        "hl_terms": ["dilated cardiomyopath", "non-ischemic dilated", "DCM", "NICM"],
        "syngo_terms": ["dilated cardiomyopathy", "DCM", "NICM"],
        "syngo_obs": [("Cardiomyopathy_obs", ["dilated", "idiopathic"])],
    },
    "disease_bicuspid_av": {
        "hl_terms": ["bicuspid", "BAV"],
        "syngo_terms": ["bicuspid", "BAV"],
        "syngo_obs": [("AoV_structure_uhn_obs", ["%bicuspid%"]), ("AoV_structure_sD_obs", ["%bicuspid%"])],
    },
    "disease_myxomatous_mv": {
        "hl_terms": ["myxomatous", "mitral valve prolapse", "MVP", "Barlow"],
        "syngo_terms": ["myxomatous", "mitral valve prolapse", "MVP", "Barlow"],
        "syngo_obs": [("MV_Structure_functionuhn_obs", ["myxomatous"])],
    },
    "disease_rheumatic_mv": {
        "hl_terms": ["rheumatic", "RHD", "rheumatic heart disease", "rheumatic mitral"],
        "syngo_terms": ["rheumatic", "RHD", "rheumatic heart disease"],
        "syngo_obs": [("MV_Structure_functionuhn_obs", ["rheumatic", "mild_rheumatic_no_ms"])],
    },
}


def load_mappings():
    """Load deid->StudyRef and deid->STUD_ID reverse mappings."""
    # aws_syngo: StudyRef -> DeidID, we need reverse
    df = pd.read_csv(AWS_SYNGO_CSV, usecols=["STUDY_REF", "DeidentifiedStudyID"])
    deid_to_refs = {}
    for _, row in df.iterrows():
        deid = str(row["DeidentifiedStudyID"])
        ref = str(row["STUDY_REF"])
        deid_to_refs.setdefault(deid, []).append(ref)

    # study_deid_map: STUD_ID -> DeidID, we need reverse
    conn = sqlite3.connect(str(ECHO_DB))
    rows = conn.execute("""
        SELECT hs.ID as STUD_ID, sdm.DeidentifiedStudyID
        FROM heartlab_studies hs
        JOIN study_deid_map sdm ON hs.STUDY_INSTANCE_UID = sdm.OriginalStudyID
    """).fetchall()
    conn.close()
    deid_to_studs = {}
    for stud_id, deid in rows:
        deid = str(deid)
        deid_to_studs.setdefault(deid, []).append(stud_id)

    return deid_to_refs, deid_to_studs


def get_heartlab_text(conn, stud_ids, search_terms):
    """Get all HeartLab findings text for given study IDs matching search terms.

    Uses a batch approach: fetch ALL findings for these studies, then filter in Python.
    Much faster than per-term LIKE queries on a 5.7GB database.
    """
    if not stud_ids:
        return []
    results = []
    placeholders = ",".join("?" * len(stud_ids))

    # Batch: get ALL SENTENCE + group for these studies
    query = f"""
        SELECT DISTINCT hs.ID as STUD_ID, hf.SENTENCE, hfg.LABEL as GROUP_LABEL
        FROM heartlab_studies hs
        JOIN heartlab_series hse ON hse.STUD_ID = hs.ID
        JOIN heartlab_reports hr ON hr.SERI_ID = hse.ID
        JOIN heartlab_finding_intersects fi ON fi.REP_ID = hr.ID
        JOIN heartlab_findings hf ON fi.FIN_ID = hf.ID
        LEFT JOIN heartlab_finding_groups hfg ON hf.FGRP_ID = hfg.ID
        WHERE hs.ID IN ({placeholders})
    """
    rows = conn.execute(query, stud_ids).fetchall()
    for stud_id, sentence, group_label in rows:
        if not sentence:
            continue
        s_lower = sentence.lower()
        for term in search_terms:
            if term.lower() in s_lower:
                results.append({
                    "source": "HL_SENTENCE",
                    "group": group_label or "unknown",
                    "text": sentence,
                    "matched_term": term,
                })
                break  # One match per sentence is enough

    # Batch: get ALL NOTEs for these studies
    query = f"""
        SELECT DISTINCT hs.ID as STUD_ID, fi.NOTE
        FROM heartlab_studies hs
        JOIN heartlab_series hse ON hse.STUD_ID = hs.ID
        JOIN heartlab_reports hr ON hr.SERI_ID = hse.ID
        JOIN heartlab_finding_intersects fi ON fi.REP_ID = hr.ID
        WHERE hs.ID IN ({placeholders})
          AND fi.NOTE IS NOT NULL
    """
    rows = conn.execute(query, stud_ids).fetchall()
    for stud_id, note in rows:
        if not note:
            continue
        n_lower = note.lower()
        for term in search_terms:
            if term.lower() in n_lower:
                results.append({
                    "source": "HL_NOTE",
                    "group": "",
                    "text": note[:300],
                    "matched_term": term,
                })
                break

    return results


def get_syngo_text(conn, study_refs, search_terms):
    """Get Syngo free-text for given StudyRefs matching search terms."""
    if not study_refs:
        return []
    results = []
    placeholders = ",".join("?" * len(study_refs))

    for term in search_terms:
        # syngo_analytics_study
        query = f"""
            SELECT StudyRef, Indication, StudyComments, PatientHistory
            FROM syngo_analytics_study
            WHERE StudyRef IN ({placeholders})
              AND (StudyComments LIKE '%{term}%' COLLATE NOCASE
                   OR PatientHistory LIKE '%{term}%' COLLATE NOCASE
                   OR Indication LIKE '%{term}%' COLLATE NOCASE)
        """
        rows = conn.execute(query, study_refs).fetchall()
        for ref, indication, comments, history in rows:
            for field_name, text in [("Indication", indication), ("StudyComments", comments), ("PatientHistory", history)]:
                if text and term.lower() in text.lower():
                    results.append({
                        "source": f"Syngo_{field_name}",
                        "group": "",
                        "text": text[:300],
                        "matched_term": term,
                    })

        # syngo_study_details
        query = f"""
            SELECT STUDY_REF, STUDY_COMMENTS, PATIENT_HISTORY, REASON_FOR_STUDY
            FROM syngo_study_details
            WHERE STUDY_REF IN ({placeholders})
              AND (STUDY_COMMENTS LIKE '%{term}%' COLLATE NOCASE
                   OR PATIENT_HISTORY LIKE '%{term}%' COLLATE NOCASE
                   OR REASON_FOR_STUDY LIKE '%{term}%' COLLATE NOCASE)
        """
        rows = conn.execute(query, study_refs).fetchall()
        for ref, comments, history, reason in rows:
            for field_name, text in [("STUDY_COMMENTS", comments), ("PATIENT_HISTORY", history), ("REASON_FOR_STUDY", reason)]:
                if text and term.lower() in text.lower():
                    results.append({
                        "source": f"Syngo_{field_name}",
                        "group": "",
                        "text": text[:300],
                        "matched_term": term,
                    })

    return results


def get_syngo_obs(conn, study_refs, obs_specs):
    """Get structured observations for given StudyRefs."""
    if not study_refs or not obs_specs:
        return []
    results = []
    placeholders = ",".join("?" * len(study_refs))

    for obs_name, values in obs_specs:
        if any("%" in v for v in values):
            # LIKE query
            for val in values:
                query = f"""
                    SELECT StudyRef, Name, Value
                    FROM syngo_observations
                    WHERE StudyRef IN ({placeholders})
                      AND Name = ?
                      AND Value LIKE ? COLLATE NOCASE
                """
                rows = conn.execute(query, study_refs + [obs_name, val]).fetchall()
                for ref, name, value in rows:
                    results.append({
                        "source": "Syngo_obs",
                        "group": name,
                        "text": f"{name} = {value}",
                        "matched_term": value,
                    })
        else:
            val_placeholders = ",".join("?" * len(values))
            query = f"""
                SELECT StudyRef, Name, Value
                FROM syngo_observations
                WHERE StudyRef IN ({placeholders})
                  AND Name = ?
                  AND Value IN ({val_placeholders})
            """
            rows = conn.execute(query, study_refs + [obs_name] + values).fetchall()
            for ref, name, value in rows:
                results.append({
                    "source": "Syngo_obs",
                    "group": name,
                    "text": f"{name} = {value}",
                    "matched_term": value,
                })

    return results


def _term_needs_word_boundary(term):
    """Check if a search term needs word-boundary validation."""
    return len(term) <= 6 and term.replace("-", "").isupper()


def _validate_term_match(text, terms):
    """Check that at least one term genuinely matches with word boundaries.

    For short uppercase acronyms (e.g., STEMI), validates that the match is not
    a substring of another word (e.g., 'systemic'). Returns the matching term
    or None if no valid match found.
    """
    if not text:
        return None
    text_lower = text.lower()
    for term in terms:
        if term.lower() not in text_lower:
            continue
        if not _term_needs_word_boundary(term):
            return term
        if re.search(r'\b' + re.escape(term) + r'\b', text, re.IGNORECASE):
            return term
    return None


def classify_mention(text, disease, search_terms=None):
    """Auto-classify a mention as DEFINITIVE, HEDGED, HISTORY, INDICATION, or CONTAMINATED.

    If search_terms is provided, also checks for false substring matches
    (e.g., 'STEMI' in 'systemic') and proximity negation.
    """
    t = (text or "").lower()

    # Fix 1: Check for false substring matches (e.g., "STEMI" in "systemic")
    if search_terms:
        valid_term = _validate_term_match(text, search_terms)
        if valid_term is None:
            return "FALSE_MATCH"

    # Fix 2: Proximity negation — R/O or "rule out" anywhere before the disease term
    if search_terms:
        proximity_triggers = ["r/o ", "r/o\t", "r/o\n", "rule out ", "rule-out ",
                              "to exclude ", "to rule out ", "no evidence of ",
                              "no definite ", "no obvious "]
        for term in (search_terms or []):
            term_lower = term.lower()
            pos = t.find(term_lower)
            if pos > 0:
                prefix = t[:pos]
                if any(trigger in prefix for trigger in proximity_triggers):
                    return "RULE_OUT_LEAK"

    # Family history
    if any(p in t for p in ["family h", "fam hx", "family history", "fhx"]):
        return "FAMILY_HX"

    # Rule-out / query (exact substring patterns — kept for backward compat)
    if any(p in t for p in ["r/o ", "rule out", "rule-out", "to exclude", "no evidence of",
                             "no definite", "no obvious", "unlikely", "cannot rule out"]):
        return "RULE_OUT_LEAK"

    # Hedged / uncertain
    if any(p in t for p in ["possible ", "probable ", "suspect", "cannot exclude",
                             "suggestive of", "consistent with but", "consider ",
                             "question of", "questionable", "equivocal",
                             "features of", "may represent", "could represent"]):
        return "HEDGED"

    # History / prior known
    if any(p in t for p in ["history of", "known ", "previous ", "prior ", "h/o ",
                             "diagnosed with", "established "]):
        return "HISTORY"

    # Definitive
    return "DEFINITIVE"


def main():
    parser = argparse.ArgumentParser(description="Audit disease labels")
    parser.add_argument("--n", type=int, default=50, help="Number of random positives to sample per disease")
    parser.add_argument("--label_dir", type=str, default="labels_v3_study_level", help="Label directory to audit")
    parser.add_argument("--output_dir", type=str, default="audit_v3", help="Output directory for audit TSVs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    label_dir = SCRIPT_DIR / args.label_dir
    output_dir = SCRIPT_DIR / args.output_dir
    output_dir.mkdir(exist_ok=True)

    print("Loading mappings...")
    deid_to_refs, deid_to_studs = load_mappings()
    conn = sqlite3.connect(str(ECHO_DB))

    summary_rows = []

    for disease, terms in DISEASE_TERMS.items():
        npz_path = label_dir / f"{disease}.npz"
        if not npz_path.exists():
            print(f"  SKIP {disease} — NPZ not found")
            continue

        npz = np.load(str(npz_path), allow_pickle=True)
        study_ids = npz["study_ids"]
        labels = npz["labels"]
        pos_ids = [str(sid) for sid, lab in zip(study_ids, labels) if lab == 1]

        n_sample = min(args.n, len(pos_ids))
        sampled = random.sample(pos_ids, n_sample)

        print(f"\n{'='*60}")
        print(f"  {disease}: sampling {n_sample}/{len(pos_ids)} positives")
        print(f"{'='*60}")

        audit_rows = []

        for i, deid in enumerate(sampled):
            # Get all matched text from all sources
            stud_ids = deid_to_studs.get(deid, [])
            study_refs = deid_to_refs.get(deid, [])

            hl_matches = get_heartlab_text(conn, stud_ids, terms["hl_terms"])
            syngo_matches = get_syngo_text(conn, study_refs, terms["syngo_terms"])
            obs_matches = get_syngo_obs(conn, study_refs, terms.get("syngo_obs", []))

            all_matches = obs_matches + hl_matches + syngo_matches

            if not all_matches:
                # No matches found — might be from patient propagation or mapping gap
                audit_rows.append({
                    "study_id": deid,
                    "source": "NONE",
                    "finding_group": "",
                    "matched_term": "",
                    "text": "[NO MATCHING TEXT FOUND IN DATABASE]",
                    "auto_class": "NO_EVIDENCE",
                })
                continue

            # Deduplicate by text
            seen_texts = set()
            for match in all_matches:
                text_key = match["text"][:100]
                if text_key in seen_texts:
                    continue
                seen_texts.add(text_key)

                # Only apply word-boundary false-match detection to free-text,
                # not to structured obs (which use different vocab, e.g., "ischemic" for STEMI)
                st = terms["hl_terms"] if match["source"] != "Syngo_obs" else None
                auto_class = classify_mention(match["text"], disease, search_terms=st)
                audit_rows.append({
                    "study_id": deid,
                    "source": match["source"],
                    "finding_group": match["group"],
                    "matched_term": match["matched_term"],
                    "text": match["text"].replace("\n", " ").replace("\t", " ")[:300],
                    "auto_class": auto_class,
                })

        # Save per-disease TSV
        if audit_rows:
            tsv_path = output_dir / f"audit_{disease}.tsv"
            with open(tsv_path, "w") as f:
                headers = ["study_id", "source", "finding_group", "matched_term", "text", "auto_class"]
                f.write("\t".join(headers) + "\n")
                for row in audit_rows:
                    f.write("\t".join(str(row[h]) for h in headers) + "\n")
            print(f"  Saved: {tsv_path}")

        # Compute summary stats
        classes = [r["auto_class"] for r in audit_rows]
        n_total = len(set(r["study_id"] for r in audit_rows))
        # Per-study classification: take the "best" class per study
        # (if any match is DEFINITIVE, the study is DEFINITIVE)
        study_classes = {}
        priority = {"DEFINITIVE": 0, "HISTORY": 1, "HEDGED": 2, "RULE_OUT_LEAK": 3, "FAMILY_HX": 4, "NO_EVIDENCE": 5, "FALSE_MATCH": 6}
        for r in audit_rows:
            sid = r["study_id"]
            cls = r["auto_class"]
            if sid not in study_classes or priority.get(cls, 99) < priority.get(study_classes[sid], 99):
                study_classes[sid] = cls

        class_counts = {}
        for cls in study_classes.values():
            class_counts[cls] = class_counts.get(cls, 0) + 1

        print(f"\n  Per-study classification (n={n_total}):")
        for cls in ["DEFINITIVE", "HISTORY", "HEDGED", "RULE_OUT_LEAK", "FAMILY_HX", "NO_EVIDENCE", "FALSE_MATCH"]:
            count = class_counts.get(cls, 0)
            pct = 100 * count / n_total if n_total > 0 else 0
            if count > 0:
                print(f"    {cls}: {count} ({pct:.1f}%)")

        summary_rows.append({
            "disease": disease,
            "n_sampled": n_total,
            "DEFINITIVE": class_counts.get("DEFINITIVE", 0),
            "HISTORY": class_counts.get("HISTORY", 0),
            "HEDGED": class_counts.get("HEDGED", 0),
            "RULE_OUT_LEAK": class_counts.get("RULE_OUT_LEAK", 0),
            "FAMILY_HX": class_counts.get("FAMILY_HX", 0),
            "NO_EVIDENCE": class_counts.get("NO_EVIDENCE", 0),
            "FALSE_MATCH": class_counts.get("FALSE_MATCH", 0),
        })

    conn.close()

    # Print overall summary
    print(f"\n{'='*80}")
    print("OVERALL AUDIT SUMMARY")
    print(f"{'='*80}")
    print(f"{'Disease':<25} {'N':>4} {'Defntv':>7} {'Hstry':>6} {'Hdged':>6} {'ROLeak':>7} {'FamHx':>6} {'NoEvd':>6} {'FalMt':>6} {'Clean%':>7}")
    print("-" * 85)
    for row in summary_rows:
        n = row["n_sampled"]
        clean = row["DEFINITIVE"] + row["HISTORY"]
        clean_pct = 100 * clean / n if n > 0 else 0
        print(f"{row['disease']:<25} {n:>4} {row['DEFINITIVE']:>7} {row['HISTORY']:>6} {row['HEDGED']:>6} "
              f"{row['RULE_OUT_LEAK']:>7} {row['FAMILY_HX']:>6} {row['NO_EVIDENCE']:>6} {row['FALSE_MATCH']:>6} {clean_pct:>6.1f}%")

    # Save summary
    summary_path = output_dir / "audit_summary.tsv"
    with open(summary_path, "w") as f:
        headers = ["disease", "n_sampled", "DEFINITIVE", "HISTORY", "HEDGED", "RULE_OUT_LEAK", "FAMILY_HX", "NO_EVIDENCE", "FALSE_MATCH"]
        f.write("\t".join(headers) + "\n")
        for row in summary_rows:
            f.write("\t".join(str(row[h]) for h in headers) + "\n")
    print(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
