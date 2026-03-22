# Disease Detection Dataset Provenance

Built by `build_disease_labels.py` on 2026-03-20. This document reports the full
provenance for all 9 binary disease detection datasets, including SQL query sources,
negation filtering rates, indication-only percentages, and confidence assessments.

## Version Changelog (v1 → v5)

### v1 — Original (unknown date, deleted notebook)

The original 9 disease NPZs were built by a **now-deleted notebook** with no surviving
build code, SQL queries, or documentation. Inferred characteristics from count analysis:

- **Patient-level label propagation**: if any study for a patient mentioned the disease,
  ALL that patient's studies were labeled positive. Confirmed by 50-62% of "missing" old
  positive studies sharing patient IDs with v2 directly-found positives.
- **No negation filtering**: STEMI had 8,815 positives (vs 663 real), suggesting
  "systemic" matched as STEMI. Endocarditis had 11,286 positives (clinically implausible
  for one institution — rule-outs counted as positives).
- **Indication field included**: referral reasons ("r/o endocarditis", "r/o amyloid")
  treated as positive findings.
- **Negative cohorts were smaller**: 1:1 or few-to-1 neg:pos ratios (e.g., rheumatic MV
  had 2,096 neg vs 2,131 pos). Build logic unknown.

**No build script survives. Cannot be reproduced.**

### v2 — First reproducible build (2026-03-20)

`build_disease_labels.py --all` (no flags)

First rebuild with full provenance. Queries all three label sources (Syngo obs, HeartLab
SENTENCE+NOTE, Syngo free-text) and applies basic negation filtering.

**What changed from v1:**
- Documented SQL queries and search terms for all 9 diseases
- Added pattern-based negation filtering (11 prefix patterns per term)
- Study-level labeling (no patient propagation by default)
- Expanded negative cohorts (e.g., bicuspid AV: 177K neg vs old 117K)
- Per-disease provenance JSON output

**What remained broken:**
- Indication/REASON_FOR_STUDY fields still included → referral contamination
- Syngo-text-only positives still included → follow-up echo noise
- "systemic" matched as "STEMI" via SQL `LIKE '%STEMI%'`
- Non-adjacent negation ("R/O vegetation / endocarditis") not caught

### v3 — Exclude indication fields (2026-03-20)

`build_disease_labels.py --all --exclude_indication`

**What changed from v2:**
- `--exclude_indication` flag skips Syngo `Indication` and `REASON_FOR_STUDY` columns
  entirely. These contain referral reasons ("r/o endocarditis", "evaluate for HCM"), not
  diagnostic findings.
- Takotsubo: 136 → 132 (-3%), endocarditis: 4,021 → 3,911 (-3%)

**Impact**: Small count changes. Indication fields were a minor contributor because most
positives come from HeartLab findings, not Syngo referral fields.

**What remained broken:** Same as v2 minus indication contamination.

### v4 — Require HeartLab confirmation (2026-03-20)

`build_disease_labels.py --all --exclude_indication --require_heartlab`

**What changed from v3:**
- `--require_heartlab` flag removes "Syngo-text-only" positives — studies where ONLY
  Syngo StudyComments/PatientHistory mentioned the disease but HeartLab findings and Syngo
  structured observations did not. These are real patients but the echocardiographer's
  report doesn't name the disease (typically follow-up echos with indirect findings).
- Tighter negation for amyloidosis: catches "rule out cardiac involvement", "r/o cardiac
  amyloid" (35 additional negated).
- Tighter negation for DCM: catches "at risk for DCM/dilated" (2 additional negated).

**Audit results (seed=123, 50 positives per disease):**

| Disease | v3 Prec | v4 Prec | Improvement |
|---------|:-------:|:-------:|:-----------:|
| HCM | 74% | **100%** | +26pp |
| STEMI | 64% | **100%** | +36pp |
| Endocarditis | 44% | **88%** | +44pp |
| Takotsubo | 62% | **92%** | +30pp |
| Amyloidosis | 84% | **100%** | +16pp |
| DCM | 90% | **98%** | +8pp |

**SUPERSEDED** — three critical bugs discovered via manual spot-check:

1. **STEMI "systemic" catastrophe**: SQL `LIKE '%STEMI%'` matched "sy**stemi**c" as a
   substring. 89.6% of HeartLab STEMI matches (7,227/8,060) were false. The v4 automated
   audit reported 100% STEMI precision because `classify_mention("Systemic hypertension")`
   returned DEFINITIVE — it couldn't detect that "STEMI" was a false substring match.
2. **Non-adjacent negation leaks**: "R/O vegetation / endocarditis" had the negation
   trigger "R/O" separated from the disease term by other content. The check for literal
   "r/o endocarditis" substring failed.
3. **Audit blind spots**: The automated audit used the same flawed classification as the
   build — it couldn't detect errors the build itself couldn't detect.

### v5 — Word boundary + proximity negation (2026-03-20)

`build_disease_labels.py --all --exclude_indication --require_heartlab`
(same flags as v4; fixes are in the code, not flags)

**Three fixes:**

1. **Word boundary validation** (`_validate_term_match`): Short uppercase acronyms (≤6
   chars, all-caps: STEMI, ATTR, DCM, HCM, HOCM, MVP, BAV, RHD, NICM) are validated with
   `\bTERM\b` regex after SQL LIKE returns results. Rejects "STEMI" in "systemic",
   "system", etc. while keeping "STEMI", "post-STEMI", "anterior STEMI".

2. **Proximity negation** (`_has_proximity_negation`): If any negation trigger (r/o, rule
   out, to exclude, no evidence of, no definite, no obvious) appears BEFORE the disease
   term in the same text unit — regardless of distance — the mention is negated. Catches
   "R/O vegetation / endocarditis" where the exact substring "r/o endocarditis" is not
   contiguous.

3. **Improved audit** (`audit_disease_labels.py`): Now detects FALSE_MATCH (word boundary
   failure) and uses proximity negation in classification. Syngo obs entries skip word
   boundary validation (structured values don't need it).

**v5 audit results (seed=789, 50 positives per disease):**

| Disease | v5 Positives | v5 Precision | v4→v5 count change |
|---------|:----------:|:------------:|:------------------:|
| HCM | 10,264 | **100%** | -18 (-0.2%) |
| DCM | 2,438 | **100%** | -3 (-0.1%) |
| Myxomatous MV | 5,964 | **100%** | -362 (-5.7%) |
| Amyloidosis | 750 | **98%** | -12 (-1.6%) |
| Bicuspid AV | 7,390 | **98%** | -17 (-0.2%) |
| Rheumatic MV | 2,292 | **98%** | -11 (-0.5%) |
| STEMI | 663 | **98%** | **-4,589 (-87.4%)** |
| Takotsubo | 74 | **98%** | -4 (-5.1%) |
| Endocarditis | 2,489 | **88%** | -167 (-6.3%) |

**SUPERSEDED** by v6 — two critical issues discovered in manual audit of negative controls and endocarditis positives.

### v6 — Negative control fix + endocarditis obs fix (2026-03-20) ← CURRENT

`build_disease_labels.py --all --exclude_indication --require_heartlab`
(same flags as v5; fixes are in the code)

**Two fixes from manual audit:**

1. **Rheumatic MV negative control catastrophe**: The negative cohort ("non-rheumatic
   mitral stenosis") used `query_heartlab_freetext(conn, ["mitral stenosis"], None)` with
   **no negation terms**. The most common SENTENCE containing "mitral stenosis" in echo
   reports is "There is no mitral stenosis" — resulting in **96% contamination** (studies
   WITHOUT MS labeled as having MS). Fix: added negation terms (`no mitral stenosis`,
   `no evidence of mitral stenosis`, `without mitral stenosis`, `r/o mitral stenosis`,
   `rule out mitral stenosis`). Result: 99,206 → 1,217 negatives. Verified with 20-sample
   spot-check: 19/20 correct (vs 2/50 before fix).

2. **Endocarditis Syngo obs screening flag**: `UHN_Endocarditis_obs` = "Endocarditis_possible"
   is a **workflow/screening flag**, not a confirmed diagnosis. Cross-reference showed 0/160
   mapped studies had ANY HeartLab endocarditis/vegetation mention — these are "screen for
   endocarditis" orders, not "has endocarditis" studies. Fix: disabled Syngo obs as a source
   for endocarditis entirely. Result: 2,489 → 2,329 positives.

**v6 changes from v5:**

| Disease | v5 Pos/Neg | v6 Pos/Neg | Change |
|---------|:--------:|:--------:|--------|
| **Endocarditis** | 2,489/4,502 | **2,329**/4,502 | -160 pos (screening flags removed) |
| **Rheumatic MV** | 2,292/99,206 | 2,292/**1,217** | -97,989 neg (96% contamination removed) |
| All others | unchanged | unchanged | — |

Output: `labels_v6_study_level/` (superseded by v7).

### v7 — Initial clean (2026-03-20)

`build_disease_labels.py --all --exclude_indication --require_heartlab --output_dir labels_v7_study_level`

**v7 changes from v6:**

| Disease | v6 Pos/Neg | v7 Pos/Neg | Change |
|---------|:--------:|:--------:|--------|
| **Endocarditis** | 2,329/4,502 | **DROPPED** | 30% precision (10+10 audit), unfixable intervening-word negation |
| **STEMI** | 663/1,125 | **603**/1,125 | -60 pos (Syngo `ischemic` obs disabled: chronic CMP ≠ acute STEMI; "false STEMI" negation added) |
| **Myxomatous MV** | 5,964/121,275 | **5,537**/**121,444** | -427 pos, +169 neg ("does not meet criteria for MVP/prolapse" + "borderline MVP" negation) |
| **Takotsubo** | 74/662 | 74/**602** | -60 neg (STEMI pos shrunk → takotsubo neg shrunk) |
| All others | unchanged | unchanged | — |

**v7 also confirmed**: Family history of HCM is NOT treated as negation (HCM is genetic; patients with HCM often have family history).

### v7.1 (2026-03-20)

Same command as v7. Three fixes from seed=9999 audit (10 pos + 10 neg per disease):

| Disease | v7 Pos/Neg | v7.1 Pos/Neg | Change |
|---------|:--------:|:--------:|--------|
| **HCM** | 10,264/39,318 | 10,264/**38,072** | **-1,246 neg** (HCM patients with concentric phenotype removed via Syngo indication cross-check) |
| **Myxomatous MV** | 5,537/121,444 | 5,537/121,444 | Negation patterns added ("no definite MVP", "no obvious prolapse") — no count change at this sample |
| **Rheumatic MV** | 2,292/1,217 | 2,292/1,217 | `exclude_finding_groups` filter added but **DID NOT MATCH** due to single vs double space bug (see v7.2) |
| All others | unchanged | unchanged | — |

### v7.2 — RECOMMENDED (2026-03-20)

Three fixes from seed=7777 audit (10 pos + 10 neg per disease) + spacing bug fix:

| Disease | v7.1 Pos/Neg | v7.2 Pos/Neg | Change |
|---------|:--------:|:--------:|--------|
| **Amyloidosis** | 750/10,247 | **747**/10,247 | **-3 pos** (family history of amyloid ≠ patient diagnosis) |
| **Myxomatous MV** | 5,537/121,444 | **4,989**/**121,696** | **-548 pos, +252 neg** (VALVE/ROOT DISEASE indication-group entries excluded from positive query) |
| **Rheumatic MV** | 2,292/1,217 | 2,292/**1,104** | **-113 neg** (double-space bug fixed → finding-group exclusion now works) |
| All others | unchanged | unchanged | — |

**v7.2 fixes (from seed=7777 10+10 raw-text audit):**
1. **Amyloidosis**: Added "family history of amyloid"/"family hx of amyloid" negation patterns (audit found P5 with only "Family Hx of amyloid" as match — family history, not patient diagnosis)
2. **Myxomatous MV**: Added `exclude_finding_groups=["VALVE/ROOT  DISEASE"]` to positive HeartLab query (audit found "Assess mitral valve prolapse" in indication group matched as positive, while actual finding said "There is no mitral valve prolapse")
3. **Spacing bug fix**: Database label is `'VALVE/ROOT  DISEASE'` (double space), not `'VALVE/ROOT DISEASE'` (single space). The v7.1 rheumatic MV `exclude_finding_groups` filter was silently not matching. Now fixed for both rheumatic neg and myxomatous pos queries.

**Recommended for all experiments.** Output: `labels_v7_study_level/`. 8 diseases (endocarditis dropped).

### Version comparison table (positives / negatives)

| Disease | v1 (original) | v2 (all fields) | v3 (excl indic) | v4 (req HL) | v5 (word bnd) | v6 (neg fix) | v7 | v7.2 (final) |
|---------|:------------:|:---------------:|:---------------:|:-----------:|:------------:|:----------:|:----------:|:----------:|
| HCM | 12,655/60,807 | 12,718/38,056 | 12,235/38,341 | 10,282/39,317 | 10,264/39,318 | 10,264/39,318 | 10,264/39,318 | **10,264/38,072** |
| Amyloidosis | 1,473/12,426 | 936/12,695 | 912/12,214 | 762/10,265 | 750/10,247 | 750/10,247 | 750/10,247 | **747/10,247** |
| Takotsubo | 399/8,808 | 136/6,581 | 132/6,360 | 78/5,251 | 74/662 | 74/662 | 74/602 | **74/602** |
| STEMI | 8,815/2,437 | 6,582/1,179 | 6,361/1,134 | 5,252/1,138 | 663/1,125 | 663/1,125 | 603/1,125 | **603/1,125** |
| Endocarditis | 11,286/19,187 | 4,021/4,335 | 3,911/4,335 | 2,656/4,335 | 2,489/4,502 | 2,329/4,502 | DROPPED | **DROPPED** |
| DCM | 1,716/20,173 | 2,664/14,350 | 2,622/14,075 | 2,441/14,103 | 2,438/14,023 | 2,438/14,023 | 2,438/14,023 | **2,438/14,023** |
| Bicuspid AV | 7,382/117,110 | 7,932/177,370 | 7,883/177,380 | 7,406/177,409 | 7,390/177,415 | 7,390/177,415 | 7,390/177,415 | **7,390/177,415** |
| Myxomatous MV | 1,931/7,935 | 6,601/120,972 | 6,535/121,021 | 6,326/121,136 | 5,964/121,275 | 5,964/121,275 | 5,537/121,444 | **4,989/121,696** |
| Rheumatic MV | 2,131/2,096 | 2,415/99,215 | 2,406/99,202 | 2,303/99,208 | 2,292/99,206 | 2,292/1,217 | 2,292/1,217 | **2,292/1,104** |

### Cumulative impact of each fix

| Fix | Version | Biggest impact | Mechanism |
|-----|---------|---------------|-----------|
| Negation filtering | v2 | Endocarditis -64%, Takotsubo -66% | Pattern-based negation (11 prefix patterns) |
| Exclude indication | v3 | Takotsubo -3%, Endocarditis -3% | Skip Syngo Indication/REASON_FOR_STUDY |
| Require HeartLab | v4 | Takotsubo -41%, Endocarditis -32%, HCM -16% | Drop Syngo-text-only positives |
| Word boundary | v5 | **STEMI -87.4%** (4,589 false "systemic" removed) | `\bSTEMI\b` regex post-filter |
| Proximity negation | v5 | Myxomatous -5.7% (362 "no MVP" caught), Endocarditis -6.3% | Non-adjacent negation detection |
| Neg control negation | v6 | **Rheumatic MV neg -98.8%** (97,989 "no MS" removed) | Added negation filtering to negative cohort free-text queries |
| Screening flag removal | v6 | Endocarditis -6.4% (160 screening flags removed) | Disabled Syngo obs "Endocarditis_possible" as label source |
| Ischemic obs disabled | v7 | **STEMI -9.0%** (60 chronic ischemic CMP removed) | Syngo `Cardiomyopathy_obs = ischemic` is chronic, not acute STEMI |
| Criteria/borderline negation | v7 | **Myxomatous -7.2%** (427 borderline/unmet-criteria removed) | "does not meet criteria for MVP", "borderline MVP" |
| Endocarditis dropped | v7 | **Entire disease removed** | 30% precision, unfixable intervening-word negation |
| HCM neg cross-check | v7.1 | **HCM neg -3.2%** (1,246 HCM patients removed from neg) | Syngo indication/comments cross-check excludes HCM patients with concentric phenotype |
| Intervening-adj negation | v7.1 | Myxomatous MV (pattern added) | "no definite/obvious/clear/significant prolapse" |
| Finding-group exclusion | v7.1 | Rheumatic MV neg (attempted) | `exclude_finding_groups` added BUT single-space bug meant it didn't match |
| Family history negation | v7.2 | **Amyloidosis -0.4%** (3 family-hx-only removed) | "family history of amyloid" / "family hx of amyloid" |
| Indication-group exclusion (pos) | v7.2 | **Myxomatous MV -9.9%** (548 VALVE/ROOT DISEASE entries removed from pos) | "Assess mitral valve prolapse" is indication, not finding |
| Double-space bug fix | v7.2 | **Rheumatic MV neg -9.3%** (113 indication-group entries removed) | DB label is `VALVE/ROOT  DISEASE` (2 spaces); v7.1 used 1 space → silent no-match |

## Negative Control Assessment

The negative controls use 5 distinct strategies with different risk profiles:

### Strategy 1: Cross-referenced positives (LOW risk)
- **Amyloidosis neg = HCM positives** (v5-cleaned)
- **Takotsubo neg = STEMI positives** (v5-cleaned)
- Risk: Inherits quality of the referenced disease's positive set.

### Strategy 2: Structured observations (VERY LOW risk)
- **Bicuspid AV neg**: `AoV_structure_uhn_obs` = "tricuspid"/"normal" + HL "trileaflet"
- **HCM neg (partial)**: `LV_Thickness_2_0_obs` = "mild/moderate/severe_concentric"
- Risk: Syngo structured obs are clinician-entered, not free-text. Negligible error rate.

### Strategy 3: Pre-cleaned source CSVs (VERY LOW risk)
- **Myxomatous MV neg**: MR studies from `aws_syngo_findings_v2.csv`/`aws_heartlab_findings_v2.csv` (pre-joined, verified)
- Risk: These CSVs were already validated in CLASS_MAPS.md with 100% class match.

### ~~Strategy 4: Ruled-out positives~~ (REMOVED — endocarditis dropped in v7)

### Strategy 5: Free-text search with negation filtering (LOW risk, FIXED in v6+v7.1)
- **HCM neg (HL portion)**: "concentric hypertrophy" / "concentric LVH" — no neg filter (low-risk: "no concentric LVH" is rare in echo reports). **v7.1**: cross-check removes 1,246 negatives where HCM appears in Syngo indication/comments.
- **DCM neg**: "heart failure" / "HFrEF" / "HFpEF" / "reduced ejection" — no neg filter (low-risk: manually verified 0% contamination in 20-sample check)
- **STEMI neg**: "NSTEMI" / "non-STEMI" — no neg filter (low-risk: manually verified 100% correct in 20-sample check)
- **Rheumatic MV neg**: "mitral stenosis" — **FIXED in v6** with negation terms. Previously 96% contaminated ("There is no mitral stenosis" = most common MS sentence). Now 1,217 studies (vs 99,206), 19/20 verified correct.

**v6 manual audit of negative controls:**

| Disease | Neg Strategy | Contamination | Verified |
|---------|-------------|:------------:|---------|
| HCM | Structured obs (concentric) | 0% | 20/20 correct |
| Amyloidosis | Cross-ref (HCM positives) | 0% | Inherits HCM quality |
| Takotsubo | Cross-ref (STEMI positives) | 0% | Inherits STEMI quality |
| STEMI | Free-text (NSTEMI) | 0% | 20/20 correct |
| DCM | Free-text (heart failure) | ~0% | 20/20 correct (78% HL confirmed) |
| Bicuspid AV | Structured obs (tricuspid) | 0% | Syngo structured |
| Myxomatous MV | Pre-cleaned CSV (MR studies) | 0% | Pre-validated |
| **Rheumatic MV** | **Free-text (MS) — FIXED v6** | **~5%** | 19/20 correct (was 2/50 before fix) |

## Build Script

```bash
# v7 — RECOMMENDED: v6 + STEMI ischemic obs fix + myxomatous criteria negation + endocarditis dropped
python build_disease_labels.py --all --exclude_indication --require_heartlab --output_dir labels_v7_study_level

# v6 (SUPERSEDED — STEMI includes chronic ischemic CMP, myxomatous includes borderline)
# v5 (SUPERSEDED — rheumatic MV neg 96% contaminated, endocarditis screening flags)
# v4 (SUPERSEDED — STEMI broken due to "systemic" substring match)
# v3 (SUPERSEDED)
# v2 (SUPERSEDED)
```

## Label Source Architecture

Three independent sources queried per disease, then unioned:

1. **Syngo structured observations** (`syngo_observations`): Name/Value lookup.
   Join: `StudyRef` -> `aws_syngo.csv.STUDY_REF` -> `DeidentifiedStudyID`.
2. **HeartLab findings** (SENTENCE + NOTE paths):
   - SENTENCE: `heartlab_findings.SENTENCE` via `finding_intersects.FIN_ID` (primary, ~90% of HL matches)
   - NOTE: `heartlab_finding_intersects.NOTE` free-text (supplementary)
   - Join: `fi.REP_ID` -> `reports.ID` -> `series.ID` (via SERI_ID) -> `studies.ID` -> `study_deid_map.DeidentifiedStudyID`
3. **Syngo free-text**: `syngo_analytics_study` (Indication, StudyComments, PatientHistory) +
   `syngo_study_details` (STUDY_COMMENTS, PATIENT_HISTORY, REASON_FOR_STUDY).
   Join: `StudyRef`/`STUDY_REF` -> `aws_syngo.csv` -> `DeidentifiedStudyID`.

## Negation Filtering

**Exact-pattern negation** (requires adjacent substring):
- `no {term}`, `r/o {term}`, `rule out {term}`, `rule-out {term}`
- `no evidence of {term}`, `unlikely {term}`, `to exclude {term}`
- `no definite {term}`, `no obvious {term}`, `without {term}`, `absent {term}`

**Proximity negation** (v5+): if any trigger word (`r/o`, `rule out`, `to exclude`, `no evidence of`, `no definite`, `no obvious`) appears BEFORE the disease term in the same text unit — regardless of distance — the mention is negated. Catches "R/O vegetation / endocarditis" where "R/O" and "endocarditis" are separated by other content.

**Word boundary validation** (v5+): short uppercase acronyms (≤6 chars, all-caps) are validated with `\b{TERM}\b` regex after SQL LIKE, to reject false substring matches (e.g., "STEMI" in "systemic").

Disease-specific negation patterns documented per disease below.

## Patient Propagation

The old (deleted) notebook used **patient-level label propagation**: if any study for a
patient mentioned the disease, ALL that patient's studies were labeled positive. This was
confirmed by analysis showing 50-62% of "missing" old positive studies share patient IDs
with our directly-found positives.

The `--patient_propagation` flag enables this. Without it, only studies where the disease
term was directly found are labeled positive (study-level, more conservative).

---

## Per-Disease Provenance

### 1. HCM (Hypertrophic Cardiomyopathy)

**Task**: `disease_hcm` — HCM (pos) vs Concentric LVH (neg)

**Positive search terms:**
- Syngo obs: `Cardiomyopathy_obs` WHERE Value IN ('HOCM', 'HCM', 'Hypertrophic', 'hypertrophic') + LIKE '%HOCM%', '%hypertrophic%'
- HeartLab SENTENCE/NOTE: `hypertrophic cardiomyopath`, `HCM`, `HOCM`, `hypertrophic obstructive`
- HeartLab SAM: Only when co-occurring with HCM/HOCM terms (64.5% of SAM mentions are negations)
- HeartLab ASH: `asymmetric septal hypertrophy`
- Syngo free-text: `hypertrophic cardiomyopathy`, `HOCM`, `HCM`

**Negation patterns**: `no HCM/HOCM`, `r/o HCM/HOCM`, `rule out HCM/HOCM`, `no evidence of HCM/HOCM`, `unlikely HCM/HOCM`, `no hypertrophic`, `rule out hypertrophic`

**Negative cohort**: Concentric LVH from `LV_Thickness_2_0_obs` (mild/moderate/severe_concentric) + HeartLab concentric hypertrophy/LVH mentions, excluding HCM positives.

| Metric | v4 (HL required) | v3 (excl. indication) | v2 (all fields) |
|--------|------------------|----------------------|-----------------|
| Study-level positives | **10,282** | 12,235 | 12,718 |
| Source: Syngo obs | 12 (0.1%) | 12 (0.1%) | 12 (0.1%) |
| Source: HeartLab | 15,343 (88.6%) | 15,343 (88.6%) | 15,343 (86.2%) |
| Source: Syngo text | 1,955 (11.3%) | 1,955 (11.3%) | 2,438 (13.7%) |
| Syngo-text-only removed | **1,953** | — | — |
| HeartLab negated | 13,439 | 13,439 | 13,439 |
| Syngo text negated | 62 | 62 | 73 |
| **Negation rate** | **43.8%** | 43.8% | 43.2% |
| Study-level negatives | **39,317** | 38,341 | — |
| **Confidence** | **MEDIUM** | LOW | LOW |

**Clinical note**: v4 removes 1,953 Syngo-text-only positives (studies where only StudyComments said "hcm" but HeartLab never named HCM). These are real HCM patients in follow-up, but the echo report describes findings indirectly. v4 audit-estimated precision rises from ~74% to ~100% (all remaining positives have HeartLab confirmation).

### 2. Cardiac Amyloidosis

**Task**: `disease_amyloidosis` — Amyloidosis (pos) vs HCM (neg)

**Positive search terms:**
- Syngo obs: `Cardiomyopathy_obs` LIKE '%amyloid%', '%infiltrative%'
- HeartLab: `amyloid`, `cardiac amyloid`, `ATTR`, `transthyretin`
- Syngo free-text: same terms

**Negation patterns**: `no amyloid`, `r/o amyloid`, `rule out amyloid`, `no evidence of amyloid`, `unlikely amyloid`. Also excludes: `renal amyloid`, `hepatic amyloid` (non-cardiac).

**Negative cohort**: HCM positive studies (excluding amyloidosis positives).

| Metric | v4 (HL required) | v3 (excl. indication) | v2 (all fields) |
|--------|------------------|----------------------|-----------------|
| Study-level positives | **762** | 912 | 936 |
| Source: Syngo obs | 1 (0.1%) | 1 (0.1%) | 1 (0.1%) |
| Source: HeartLab | 761 (85.9%) | 785 (86.1%) | 785 (83.9%) |
| Source: Syngo text | 124 (14.0%) | 126 (13.8%) | 150 (16.0%) |
| Syngo-text-only removed | **124** | — | — |
| HeartLab negated | 98 | 63 | 63 |
| Syngo text negated | 44 | 30 | 34 |
| **Negation rate** | **13.8%** | 9.3% | 9.4% |
| Study-level negatives | **10,265** | 12,214 | — |
| **Confidence** | **MEDIUM-HIGH** | MEDIUM | MEDIUM |

**Clinical note**: v4 applies two fixes: (1) tighter negation catches screening referrals — "rule out cardiac involvement", "r/o cardiac amyloid" — adding 35 more negated (HL 63→98, Syngo 30→44); (2) `--require_heartlab` removes 124 Syngo-text-only positives. Combined effect: 912→762 (16.4% reduction). All remaining positives have HeartLab confirmation of cardiac amyloid.

### 3. Takotsubo

**Task**: `disease_takotsubo` — Takotsubo (pos) vs STEMI (neg)

**Positive search terms:**
- HeartLab: `takotsubo`, `tako-tsubo`, `apical ballooning`, `stress cardiomyopath`
- Syngo free-text: same terms
- Syngo obs: none found

**Negation patterns**: `no takotsubo`, `r/o takotsubo`, `no evidence of takotsubo`, `unlikely takotsubo`, `no apical ballooning`, `r/o apical ballooning`

**Negative cohort**: STEMI positive studies (excluding takotsubo positives).

| Metric | v4 (HL required) | v3 (excl. indication) | v2 (all fields) |
|--------|------------------|----------------------|-----------------|
| Study-level positives | **78** | 132 | 136 |
| Source: Syngo obs | **0 (0%)** | 0 (0%) | 0 (0%) |
| Source: HeartLab | **78 (100%)** | 78 (59.1%) | 78 (57.4%) |
| Source: Syngo text | **54 (40.9%)** | 54 (40.9%) | 58 (42.6%) |
| Syngo-text-only removed | **54** | — | — |
| **Negation rate** | **3.6%** | 3.6% | 4.2% |
| Indication-only | — | 57 | 79 (58.1%) |
| Study-level negatives | **5,251** | 6,360 | — |
| **Confidence** | **MEDIUM** | MEDIUM | MEDIUM |

**Clinical note**: Very rare condition. v4 drops from 132→78 positives by removing 54 Syngo-text-only studies. Only 78 positives may be too few for reliable probe training. Old NPZ had 399 — the 67% drop (v3) is the largest proportional change, indicating most old positives were indication-only referrals. With v3 patient propagation: 372 positives (vs 399 old, within 6.8%).

### 4. STEMI

**Task**: `disease_stemi` — STEMI (pos) vs NSTEMI (neg)

**Positive search terms:**
- Syngo obs: ~~`Cardiomyopathy_obs` WHERE Value = 'ischemic'~~ **DISABLED in v7** — "ischemic" denotes chronic ischemic cardiomyopathy, not acute STEMI. v7 10+10 audit found 2/10 Syngo obs positives were chronic ischemic CMP.
- HeartLab: `STEMI`, `ST elevation myocardial`, `ST-elevation myocardial`
- Syngo free-text: same terms

**Negation patterns**: `NSTEMI`, `non-STEMI`, `non STEMI`, `non-ST elevation`, `false STEMI`, `False STEMI` (critical to avoid matching NSTEMI as STEMI; v7 adds "false STEMI on ECG" pattern)

**Negative cohort**: NSTEMI studies from HeartLab + Syngo free-text.

| Metric | v7 (final) | v5 (word boundary) | v4 (BROKEN) | v2 (all fields) |
|--------|-----------|-------------------|-------------|-----------------|
| Study-level positives | **603** | 663 | ~~5,252~~ | 6,582 |
| Source: Syngo obs | **0 (DISABLED)** | 59 (5.6%) | 59 (0.9%) | 59 (0.9%) |
| Source: HeartLab | **603 (60.6%)** | 604 (57.3%) | ~~5,193~~ | ~~5,193~~ |
| Source: Syngo text | **392 (39.4%)** | 392 (37.2%) | 1,109 | 1,330 |
| Syngo-text-only removed | **392** | 392 | 1,109 | — |
| HeartLab negated | **40** | 38 | 974 | 974 |
| Syngo text negated | **30** | 29 | 792 | 869 |
| **Negation rate** | **6.6%** | 6.0% | ~~21.7%~~ | 21.9% |
| Study-level negatives | **1,125** | 1,125 | 1,138 | — |
| **Confidence** | **HIGH** | HIGH | BROKEN | LOW |

**CRITICAL BUG (v4 and earlier)**: `LIKE '%STEMI%'` matched "sy**stemi**c" as a substring. Of 8,060 HeartLab "STEMI" matches, **7,227 (89.6%) were "Systemic hypertension" false positives** and only 831 were real STEMI. The v4 automated audit reported 100% precision because `classify_mention("Systemic hypertension")` returned DEFINITIVE — the audit couldn't detect substring false matches.

**v5 fix**: Word boundary validation (`\bSTEMI\b` regex) after SQL LIKE. Rejects "systemic" while keeping "STEMI", "post-STEMI", "anterior STEMI", etc. HeartLab positives dropped from 5,193 to 604.

**v7 fixes**: (1) Disabled Syngo obs `Cardiomyopathy_obs = ischemic` — 10+10 raw-text audit found these are chronic ischemic cardiomyopathy, not acute STEMI. (2) Added "false STEMI" negation pattern for studies where ECG was false positive ("false STEMI on ECG; normal cath"). Combined: 663→603 positives.

### 5. Endocarditis

**Task**: `disease_endocarditis` — Endocarditis (pos) vs Ruled-out (neg)

**Positive search terms:**
- Syngo obs: ~~`UHN_Endocarditis_obs` WHERE Value = 'Endocarditis_possible'~~ **DISABLED in v6** — this is a workflow/screening flag, not a diagnosis. 0/160 mapped studies had any HeartLab endocarditis mention.
- HeartLab: `endocarditis`, `vegetation`, `infective endocarditis`, `bacterial endocarditis`
- Syngo free-text: `endocarditis`, `vegetation`

**Negation patterns (aggressive — 20 patterns):** `rule out endocarditis/vegetation`, `r/o endocarditis/vegetation`, `no endocarditis/vegetation`, `no evidence of endocarditis/vegetation/vegetations`, `to exclude endocarditis/vegetation`, `no definite vegetation`, `no obvious vegetation`, `R/O vegetation/endocarditis`, `rule-out endocarditis/vegetation`, `without vegetation/endocarditis`

**Negative cohort**: Studies where endocarditis was specifically ruled out (the negated mentions become negatives).

| Metric | v6 (obs fix) | v5 (word bnd) | v4 (HL required) | v2 (all fields) |
|--------|-------------|---------------|------------------|-----------------|
| Study-level positives | **2,329** | 2,489 | 2,656 | 4,021 |
| Source: Syngo obs | **0 (DISABLED)** | 160 (4.0%) | 160 (4.0%) | 160 (3.9%) |
| Source: HeartLab | **2,329 (100%)** | 2,329 (93.6%) | 2,496 (63.1%) | 2,496 (61.2%) |
| Source: Syngo text | **1,217 (52.3%)** | 1,217 (48.9%) | 1,302 (32.9%) | 1,420 (34.8%) |
| Syngo-text-only removed | **1,217** | 1,217 | 1,255 | — |
| Syngo obs screening removed | **160** | — | — | — |
| HeartLab negated | **17,420** | 17,420 | 17,076 | 17,076 |
| Syngo text negated | **2,234** | 2,234 | 2,035 | 2,295 |
| **Negation rate** | **84.7%** | 84.7% | 82.8% | 82.6% |
| Study-level negatives | **4,502** | 4,502 | 4,335 | — |
| **Confidence** | **LOW** | LOW | LOW | LOW |

**Clinical note**: HIGHEST contamination risk. 85% negation rate — the vast majority of endocarditis mentions are rule-outs. v6 removes 160 "Endocarditis_possible" Syngo obs entries (confirmed as workflow screening flags with 0% HeartLab overlap). Remaining 2,329 positives all have HeartLab confirmation. However, the underlying union-across-findings rule-out leak persists — manual spot-check found 4/5 positives were still rule-out leaks from intervening-word negation ("no aortic vegetation present" defeats "no vegetation" pattern). Old NPZ had 11,286 positives (clinically implausible).

### 6. DCM (Dilated Cardiomyopathy)

**Task**: `disease_dcm` — DCM (pos) vs HF-without-DCM (neg)

**Positive search terms:**
- Syngo obs: `Cardiomyopathy_obs` WHERE Value IN ('dilated', 'idiopathic')
- HeartLab: `dilated cardiomyopath`, `non-ischemic dilated`, `DCM` (with cardiac context filter), `NICM`
- Syngo free-text: `dilated cardiomyopathy`, `DCM`, `NICM`

**DCM context filter**: `DCM` acronym only accepted if the NOTE also contains cardiac terms (cardiomyopath|heart|cardiac|ventricul|ejection).

**Negative cohort**: Heart failure mentions without DCM — `heart failure`, `HFrEF`, `HFpEF`, `reduced ejection fraction`, `systolic dysfunction`, `severely reduced`.

| Metric | v4 (HL required) | v3 (excl. indication) | v2 (all fields) |
|--------|------------------|----------------------|-----------------|
| Study-level positives | **2,441** | 2,622 | 2,664 |
| Source: Syngo obs | **13 (0.5%)** | 13 (0.5%) | 13 (0.5%) |
| Source: HeartLab | **2,428 (92.6%)** | 2,428 (92.6%) | 2,428 (91.1%) |
| Source: Syngo text | **181 (6.9%)** | 181 (6.9%) | 223 (8.4%) |
| Syngo-text-only removed | **179** | — | — |
| **Negation rate** | **0.2%** | 0.1% | 0.1% |
| Indication-only | — | 156 | 255 |
| Study-level negatives | **14,105** | 14,075 | — |
| **Confidence** | **MEDIUM-HIGH** | MEDIUM | MEDIUM |

**Clinical note**: Very low negation rate — DCM is usually definitive when stated. v4 applies two fixes: (1) tighter negation catches "at risk for DCM" patterns (2 additional studies negated, rate 0.1%→0.2%); (2) `--require_heartlab` removes 179 Syngo-text-only positives. v3 positives (2,622) higher than old NPZ (1,716) because SENTENCE-level HeartLab matching captures more. HF-without-DCM negatives slightly changed (14,075→14,105) due to overlap adjustments.

### 7. Bicuspid Aortic Valve

**Task**: `disease_bicuspid_av` — Bicuspid AV (pos) vs Tricuspid AV (neg)

**Positive search terms:**
- Syngo obs: `AoV_structure_uhn_obs` WHERE Value IN ('bicuspid', 'bicuspid_10_4', 'bicuspid_1_7', 'bicuspid_9_3', 'functional_bicuspid') + LIKE '%bicuspid%'. Also `AoV_structure_sD_obs` ('bicuspid', 'bicuspid_w_doming').
- HeartLab: `bicuspid`, `BAV`
- Syngo free-text: `bicuspid`, `BAV`

**Negative cohort**: Tricuspid AV from `AoV_structure_uhn_obs`/`sD_obs` ('tricuspid', 'normal') + HeartLab SENTENCE `aortic valve is tricuspid`, `trileaflet`.

| Metric | v4 (HL required) | v3 (excl. indication) | v2 (all fields) |
|--------|------------------|----------------------|-----------------|
| Study-level positives | **7,407** | 7,884 | 7,933 |
| Source: Syngo obs | **1,379 (16.3%)** | 1,379 (16.3%) | 1,379 (16.0%) |
| Source: HeartLab | **6,028 (71.1%)** | 6,028 (71.1%) | 6,028 (69.8%) |
| Source: Syngo text | **1,074 (12.7%)** | 1,074 (12.7%) | 1,228 (14.2%) |
| Syngo-text-only removed | **477** | — | — |
| **Negation rate** | **0.9%** | 0.9% | 1.0% |
| Indication-only | — | 1,443 | 1,836 |
| Study-level negatives | **177,411** | 177,380 | — |
| **Confidence** | **HIGH** | HIGH | HIGH |

**Clinical note**: Highest confidence disease — structural finding visible on echo with strong Syngo structured labels. Minimal v4 impact — 16% from structured obs provides a safety net. v4 removes 477 Syngo-text-only positives. v3 barely changes positives (7,884 vs 7,933, -0.6%). Negative cohort expanded to 177K because tricuspid AV is the dominant finding. Old NPZ had 117K negatives.

### 8. Myxomatous Mitral Valve

**Task**: `disease_myxomatous_mv` — Myxomatous MV (pos) vs Non-Myxomatous MR (neg)

**Positive search terms:**
- Syngo obs: `MV_Structure_functionuhn_obs` WHERE Value = 'myxomatous' (508 studies)
- HeartLab: `myxomatous`, `mitral valve prolapse`, `MVP`, `Barlow`
- Syngo free-text: same terms

**Negation patterns** (v7+): `no MVP`, `no prolapse`, `no myxomatous`, `r/o MVP`, `rule out MVP`, `does not meet criteria for MVP`, `does not meet criteria for prolapse`, `does not meet criteria for mitral valve prolapse`, `borderline mitral valve prolapse`

**Negative cohort**: Any MR (trace through severe) from `aws_syngo_findings_v2.csv` + `aws_heartlab_findings_v2.csv`, excluding myxomatous positives.

| Metric | v7 (final) | v4 (HL required) | v3 (excl. indication) | v2 (all fields) |
|--------|-----------|------------------|----------------------|-----------------|
| Study-level positives | **5,537** | 6,326 | 6,535 | 6,601 |
| Source: Syngo obs | **236 (4.1%)** | 236 (3.6%) | 236 (3.6%) | 236 (3.5%) |
| Source: HeartLab | **5,301 (91.7%)** | 6,090 (92.7%) | 6,090 (92.7%) | 6,090 (91.4%) |
| Source: Syngo text | **242 (4.2%)** | 245 (3.7%) | 245 (3.7%) | 337 (5.1%) |
| Syngo-text-only removed | **206** | 209 | — | — |
| HeartLab negated | **2,980** | — | — | — |
| **Negation rate** | **34.2%** | 20.7% | 20.7% | 20.6% |
| Study-level negatives | **121,444** | 121,147 | 121,021 | — |
| **Confidence** | **MEDIUM-HIGH** | MEDIUM-HIGH | MEDIUM | MEDIUM |

**Clinical note**: v7 adds criteria-denial and borderline negation patterns: "does not meet criteria for MVP" and "borderline mitral valve prolapse" were found by 10+10 raw-text audit to be false positives (1/10 sampled positives). These are equivocal findings where the echocardiographer explicitly decided against the MVP diagnosis. Combined with existing proximity negation, this removes 427 positives (5,964→5,537). The higher negation rate (34.2%) reflects the many "no prolapse" sentences in echo reports — most MVP mentions are to deny it.

### 9. Rheumatic Mitral Valve

**Task**: `disease_rheumatic_mv` — Rheumatic MV (pos) vs Non-Rheumatic MS (neg)

**Positive search terms:**
- Syngo obs: `MV_Structure_functionuhn_obs` WHERE Value IN ('rheumatic', 'mild_rheumatic_no_ms') (452 studies)
- HeartLab: `rheumatic`, `RHD`, `rheumatic heart disease`, `rheumatic mitral`
- Syngo free-text: same terms

**Negative cohort**: Mitral stenosis mentions without rheumatic etiology — `mitral stenosis` in HeartLab + Syngo, excluding rheumatic positives. **v6 critical fix**: added negation filtering to negative cohort queries.

| Metric | v6 (neg fix) | v5 (word bnd) | v4 (HL required) | v2 (all fields) |
|--------|-------------|---------------|------------------|-----------------|
| Study-level positives | **2,292** | 2,292 | 2,303 | 2,415 |
| Source: Syngo obs | **214 (8.7%)** | 214 (8.7%) | 214 (8.7%) | 214 (8.6%) |
| Source: HeartLab | **2,078 (85.0%)** | 2,078 (85.0%) | 2,089 (85.0%) | 2,089 (84.0%) |
| Source: Syngo text | **155 (6.3%)** | 155 (6.3%) | 155 (6.3%) | 184 (7.4%) |
| Syngo-text-only removed | **103** | 103 | 103 | — |
| **Negation rate (pos)** | **1.1%** | 1.1% | 0.4% | 0.4% |
| Study-level negatives | **1,217** | ~~99,206~~ | ~~99,211~~ | — |
| **Confidence** | **MEDIUM-HIGH** | MEDIUM-HIGH (pos) / **BROKEN (neg)** | MEDIUM | MEDIUM |

**Clinical note**: Positives are clean (98% audit precision, very low negation rate). The v6 fix is entirely about the **negative cohort**: v5 and earlier used `query_heartlab_freetext(conn, ["mitral stenosis"], None)` with NO negation terms. The most common HeartLab SENTENCE containing "mitral stenosis" is "There is no mitral stenosis" — resulting in 96% contamination (studies WITHOUT MS labeled as having MS). v6 adds negation terms, reducing negatives from 99,206 to 1,217 genuine MS studies. Manual verification: 19/20 correct (vs 2/50 before fix). Old NPZ had 1:1 matched negatives (2,096); the v6 negative set (1,217) is smaller but genuine.

---

## Validation Against Old NPZs

### v4 (--exclude_indication --require_heartlab) — RECOMMENDED

| Disease | Old Pos | v3 Study | v4 Study | Δ v3→v4 | Syngo-only removed |
|---------|---------|----------|----------|---------|-------------------|
| HCM | 12,655 | 12,235 | **10,282** | -16.0% | 1,953 |
| Amyloidosis | 1,473 | 912 | **762** | -16.4% | 124 (+26 Fix 2) |
| Takotsubo | 399 | 132 | **78** | -40.9% | 54 |
| STEMI | 8,815 | 6,361 | **5,252** | -17.4% | 1,109 |
| Endocarditis | 11,286 | 3,911 | **2,656** | -32.1% | 1,255 |
| DCM | 1,716 | 2,622 | **2,441** | -6.9% | 179 (+2 Fix 2) |
| Bicuspid AV | 7,382 | 7,884 | **7,407** | -6.1% | 477 |
| Myxomatous MV | 1,931 | 6,535 | **6,326** | -3.2% | 209 |
| Rheumatic MV | 2,131 | 2,406 | **2,303** | -4.3% | 103 |

**Key observations:**
1. **`--require_heartlab` matters most for**: Takotsubo (-41%), Endocarditis (-32%), STEMI (-17%), HCM (-16%). These have many follow-up echos where only StudyComments names the disease.
2. **Minimal impact on structural diseases**: Myxomatous (-3.2%), Rheumatic (-4.3%), Bicuspid (-6.1%). HeartLab confirmation rate is high for visible structural findings.
3. **Fix 2 tighter negation**: Amyloidosis caught 26 additional screening referrals ("rule out cardiac involvement"); DCM caught 2 "at risk for DCM" studies.
4. v4 study-level is the most conservative and recommended build.
5. All split matches remain 100% (deterministic patient splits).

## Label Precision Audit

Systematic audit of label quality. For each disease, 50 random positive studies
were sampled from the NPZ, and all matching text was retrieved from the database
(HeartLab SENTENCE, HeartLab NOTE, Syngo obs, Syngo free-text). Each study was
classified by its strongest evidence.

**Audit script**: `audit_disease_labels.py` — per-disease TSVs in `audit_v3/`, `audit_v4/`, `audit_v5/`, `audit_v6/`.

### Classification categories

- **Definitive**: HeartLab SENTENCE or structured observation clearly states the disease finding.
- **History**: "Known X", "previous X", "h/o X" — true positives with established diagnosis.
- **Hedged**: "Possible X", "features of X", "consistent with X" — arguably valid, uncertain language.
- **Rule-out leak**: "rule out", "r/o", "no vegetation", etc. Negation filtering missed these. **False positive.**
- **Family Hx**: Family history mention, not the patient's own diagnosis. **False positive.**
- **No evidence**: No matching text found in database. **False positive (mapping gap).**

*Precision = (Definitive + History) / N. HEDGED counted separately. RO_LEAK, FAMILY_HX, NO_EVIDENCE are false positives.*

### v3 Audit (seed=42, excl. indication only)

| Disease | Def | Syngo-only | RO Leak | FamHx | Precision |
|---------|:---:|:----------:|:-------:|:-----:|:---------:|
| Myxomatous MV | 50 | 0 | 0 | 0 | **100%** |
| Rheumatic MV | 49 | 1 | 0 | 0 | **98%** |
| Bicuspid AV | 48 | 1 | 0 | 1 | **96%** |
| DCM | 45 | 3 | 2 | 0 | **90%** |
| Amyloidosis | 42 | 5 | 3 | 0 | **84%** |
| HCM | 37 | 13 | 0 | 0 | **74%** |
| STEMI | 39 | 11 | 11 | 0 | **64%** |
| Takotsubo | 31 | 14 | 5 | 0 | **62%** |
| Endocarditis | 23 | 8 | 21 | 0 | **44%** |

*v3 included Syngo-text-only studies. The "Syngo-only" column shows studies where only
StudyComments/PatientHistory mentioned the disease — the echocardiographer's report never named it.*

### v4 Audit (seed=123, excl. indication + require HeartLab)

| Disease | Def | History | Hedged | RO Leak | FamHx | Precision |
|---------|:---:|:-------:|:------:|:-------:|:-----:|:---------:|
| **HCM** | 50 | 0 | 0 | 0 | 0 | **100%** |
| **Amyloidosis** | 50 | 0 | 0 | 0 | 0 | **100%** |
| **STEMI** | 49 | 1 | 0 | 0 | 0 | **100%** |
| **DCM** | 49 | 0 | 0 | 0 | 1 | **98%** |
| **Rheumatic MV** | 49 | 0 | 0 | 1 | 0 | **98%** |
| **Bicuspid AV** | 47 | 0 | 3 | 0 | 0 | **94%** |
| **Myxomatous MV** | 47 | 0 | 0 | 3 | 0 | **94%** |
| **Takotsubo** | 39 | 7 | 0 | 4 | 0 | **92%** |
| **Endocarditis** | 40 | 4 | 2 | 4 | 0 | **88%** |

### v3 → v4 Precision Improvement

| Disease | v3 | v4 | Change | Key improvement |
|---------|:--:|:--:|:------:|-----------------|
| **STEMI** | 64% | **100%** | +36pp | Removed 1,109 Syngo-text-only |
| **Takotsubo** | 62% | **92%** | +30pp | Removed 54 Syngo-text-only |
| **HCM** | 74% | **100%** | +26pp | Removed 1,953 Syngo-text-only |
| **Endocarditis** | 44% | **88%** | +44pp | Removed 1,255 Syngo-text-only |
| **Amyloidosis** | 84% | **100%** | +16pp | Removed 124 Syngo-only + 26 screening referrals |
| **DCM** | 90% | **98%** | +8pp | Removed 179 Syngo-only + 2 "at risk" |
| Rheumatic MV | 98% | **98%** | 0 | Minimal Syngo-text prevalence |
| Bicuspid AV | 96% | **94%** | -2pp | Sample variance (3 HEDGED = "possible bicuspid") |
| Myxomatous MV | 100% | **94%** | -6pp | Sample variance (3 RO leaks in new sample) |

**The `--require_heartlab` flag is the single largest quality improvement.** It eliminates
Syngo-text-only noise entirely, which was the dominant error source for HCM (26%), STEMI (22%),
and Takotsubo (28%). For endocarditis, it removes 32% of positives, but the remaining 8%
RO leaks come from the union-across-findings structural issue (not Syngo text).

### Remaining issues in v4

**Endocarditis (88%, 4 RO leaks)**: Still has the union-across-findings problem — a study
is positive if ANY finding matches "vegetation" without negation, even if other findings
for the same study say "no vegetation." Per-finding negation would fix this.

**Takotsubo (92%, 4 RO leaks)**: Small absolute numbers (n=78 total). Some RO leaks
involve hedged language ("cannot exclude takotsubo"). Very small positive set limits
probe training reliability.

**Myxomatous MV (94%, 3 RO leaks in v4 sample)**: Likely sample variance — v3 audit
showed 100% on different random sample. The 3 RO leaks warrant investigation but may
reflect MVP negation patterns not yet in the filter list.

**Bicuspid AV (94%, 3 HEDGED)**: The 3 "hedged" mentions are "possible bicuspid" or
"features suggestive of bicuspid" — these are arguably valid labels (equivocal structural
findings). If HEDGED counted as true positives, precision is 100%.

### v5 Audit (seed=789, word boundary + proximity negation)

Same dataset as v4 but with word boundary validation and proximity negation fixes.

| Disease | Def | History | Hedged | RO Leak | FamHx | FalMt | Precision |
|---------|:---:|:-------:|:------:|:-------:|:-----:|:-----:|:---------:|
| **HCM** | 50 | 0 | 0 | 0 | 0 | 0 | **100%** |
| **DCM** | 50 | 0 | 0 | 0 | 0 | 0 | **100%** |
| **Myxomatous MV** | 50 | 0 | 0 | 0 | 0 | 0 | **100%** |
| **Amyloidosis** | 49 | 0 | 0 | 0 | 1 | 0 | **98%** |
| **Bicuspid AV** | 49 | 0 | 0 | 0 | 1 | 0 | **98%** |
| **Rheumatic MV** | 49 | 0 | 0 | 1 | 0 | 0 | **98%** |
| **STEMI** | 48 | 1 | 0 | 1 | 0 | 0 | **98%** |
| **Takotsubo** | 44 | 5 | 0 | 1 | 0 | 0 | **98%** |
| **Endocarditis** | 39 | 5 | 4 | 1 | 0 | 1 | **88%** |

### v6 Audit (seed=789, neg control fix + endocarditis obs fix)

Same positive sets as v5 for 7/9 diseases. Endocarditis lost 160 screening-flag studies;
rheumatic MV positives unchanged (only negatives fixed).

| Disease | Def | History | Hedged | RO Leak | FamHx | FalMt | Precision |
|---------|:---:|:-------:|:------:|:-------:|:-----:|:-----:|:---------:|
| **HCM** | 50 | 0 | 0 | 0 | 0 | 0 | **100%** |
| **DCM** | 50 | 0 | 0 | 0 | 0 | 0 | **100%** |
| **Myxomatous MV** | 50 | 0 | 0 | 0 | 0 | 0 | **100%** |
| **Amyloidosis** | 49 | 0 | 0 | 0 | 1 | 0 | **98%** |
| **Bicuspid AV** | 49 | 0 | 0 | 0 | 1 | 0 | **98%** |
| **STEMI** | 48 | 1 | 0 | 1 | 0 | 0 | **98%** |
| **Takotsubo** | 44 | 5 | 0 | 1 | 0 | 0 | **98%** |
| **Rheumatic MV** | 48 | 0 | 1 | 1 | 0 | 0 | **96%** |
| **Endocarditis** | 34 | 6 | 9 | 0 | 0 | 1 | **80%** |

**Notes on v6 changes:**
- **Endocarditis 88%→80%**: Removing 160 Syngo obs screening flags concentrated the free-text
  pool, which has more HEDGED cases ("possible endocarditis", "features consistent with").
  If HEDGED counted as valid: 98% precision. The 1 FALSE_MATCH is likely an intervening-word
  negation leak.
- **Rheumatic MV 98%→96%**: Sample variance (different 50 drawn). 1 HEDGED + 1 RO leak.

### v6 Manual Raw-Text Audit (seed=2026, 3+3)

Full raw-text review of 3 positives + 3 negatives per disease (54 total samples). ALL
HeartLab SENTENCE, HeartLab NOTE, Syngo obs, and Syngo free-text fields were loaded
for each sampled study and manually inspected in context. Script: `/tmp/manual_audit_v6.py`.

| Disease | Pos Quality | Neg Quality | Key Finding |
|---------|:-----------:|:-----------:|-------------|
| **HCM** | 3/3 | 3/3 | Clean. All positives have definitive "Hypertrophic cardiomyopathy" SENTENCE. Negatives have genuine concentric LVH. |
| **Amyloidosis** | 3/3 | 3/3 | Clean. All positives have definitive "Amyloidosis" SENTENCE. Negatives are confirmed HCM. |
| **Bicuspid AV** | 3/3 | 3/3 | Clean. Mix of HL SENTENCE + Syngo structured obs. All negs have "tricuspid" AV. |
| **STEMI** | 3/3 | 3/3 | Clean. All positives have "STEMI" in HL NOTE with wall motion abnormalities. Negs are NSTEMI. |
| **Rheumatic MV** | 3/3 | 3/3 | **v6 neg fix validated.** All 3 negs have genuine non-rheumatic MS (MAC, congenital supravalvular ring, severe calcification). |
| **DCM** | 3/3 | 2/3 | 1 negative is congenital heart disease (L-TGA + ASD repair), not HF-without-DCM. Minor edge case. |
| **Myxomatous MV** | 3/3 | 1/3 | 1 neg has no data (mapping gap). 1 neg has normal MV with no MR — intended neg class is non-myxomatous MR, but some studies lack MR entirely. |
| **Takotsubo** | 1/3 | 3/3 | **2/3 positives are hedged** — "Takotsubo cardiomyopathy?" and "?takotsubo" with question marks. Only 74 total positives, so hedging is a significant quality issue. |
| **Endocarditis** | 1/3 | 2/3 | **SEVERELY BROKEN.** 2/3 positives explicitly say "no vegetation" — intervening anatomical terms defeat pattern negation. |

### v6 Manual Raw-Text Audit (seed=42, 10+10) — LED TO v7

Larger raw-text audit: 10 positives + 10 negatives per disease (180 total samples).
Script: `/tmp/manual_audit_v6_large.py`. Output: `/tmp/audit_v6_raw/` (per-disease files).

| Disease | Pos Correct | Neg Correct | Issues Found |
|---------|:-----------:|:-----------:|-------------|
| **HCM** | 9/10 | 10/10 | 1 pos has family history of HCM (ambiguous — kept as valid; HCM is genetic) |
| **Amyloidosis** | 10/10 | 10/10 | Clean |
| **Bicuspid AV** | 10/10 | 10/10 | Clean. Syngo structured obs dominant. |
| **STEMI** | 8/10 | 10/10 | **2 FP from Syngo `ischemic` obs** (chronic ischemic CMP, not acute STEMI); 1 "false STEMI on ECG" not caught → **FIXED in v7** |
| **Rheumatic MV** | 10/10 | 10/10 | Clean |
| **DCM** | 10/10 | 10/10 | Clean |
| **Myxomatous MV** | 9/10 | 10/10 | 1 pos: "does not meet criteria for MVP" — criteria denial not caught → **FIXED in v7** |
| **Takotsubo** | 3/10 | 10/10 | 4 follow-up, 3 hedged ("?takotsubo"). Only 30% definitive acute. N=74 limits utility. |
| **Endocarditis** | 3/10 | 8/10 | **30% precision.** 7/10 pos say "no vegetation" (intervening-word negation). → **DROPPED in v7** |

**v7 actions from 10+10 audit:**

1. **Endocarditis DROPPED** — 30% precision unfixable by pattern matching. Removed from DISEASES list and probe CSVs.
2. **STEMI fixed** — Syngo `ischemic` obs disabled (chronic CMP ≠ acute STEMI). "false STEMI" negation added.
3. **Myxomatous MV fixed** — "does not meet criteria for MVP/prolapse" and "borderline mitral valve prolapse" added to negation.
4. **HCM family history NOT negated** — user correctly identified that family history of HCM is ambiguous (genetic disease; patients often have both HCM and family history of HCM).

### Confidence tiers (v7, updated with 10+10 raw-text audit)

| Tier | Diseases | Automated Precision | 10+10 Manual | Recommendation |
|------|----------|:-------------------:|:------------:|----------------|
| **Use confidently** | HCM, DCM, Bicuspid AV, Amyloidosis, Rheumatic MV | 96-100% | 9-10/10 pos | Include in manuscript |
| **Use confidently** | STEMI (v7 fixed) | 98% | 8/10 → fixed | Include; v7 removes ischemic obs + false STEMI |
| **Use with caveats** | Myxomatous MV (v7 fixed) | 100% | 9/10 → fixed | Include; v7 removes criteria-denial FPs |
| **Use with caveats** | Takotsubo | 98% | 3/10 definitive | Include; most positives are hedged queries. N=74. |
| ~~DROPPED~~ | ~~Endocarditis~~ | ~~80%~~ | ~~3/10~~ | Dropped in v7 — 30% precision, unfixable |

### Recommendation

Use **`labels_v7_study_level/`** for all experiments (v7 = v6 + STEMI ischemic obs fix +
myxomatous criteria negation + endocarditis dropped). **8 diseases** (endocarditis removed).

All 8 surviving diseases have >=96% automated positive audit precision. v7 fixes were
driven by 10+10 raw-text audit findings. Probe CSVs rebuilt from v7 labels.

## Known Limitations

1. **No gold standard**: All labels are echocardiographer observations and free-text findings. No biopsy, pathology, ICD, surgical, or catheterization tables exist in echo.db.
2. **Negation filtering is pattern-based**: Complex negations ("cannot exclude HCM given the clinical picture") may not be caught. The audit found 2-22% rule-out leak rates depending on disease.
3. **Union-across-findings weakness**: The build script unions positive matches across all findings for a study. A study is positive if ANY finding matches without negation — even if other findings for the same study explicitly negate the disease. (Endocarditis was dropped in v7 due to this being unfixable.)
4. **Hedged positives (takotsubo)**: 10+10 audit found only 3/10 sampled takotsubo positives were definitive acute diagnoses; 4/10 were follow-up studies, 3/10 hedged ("?takotsubo"). With only 74 total positives, this significantly degrades label quality.
5. **Negative cohort class purity (myxomatous MV)**: Some negatives in the myxomatous MV task have normal MVs with no MR at all, rather than the intended "non-myxomatous MR." The MR source CSVs include studies that may not have clinically significant MR.
6. **Family history ambiguity**: HCM audit found 1/10 with "family history of HCM" — ambiguous because HCM is genetic and patients often have both HCM and family history. NOT treated as negation (deliberate design choice).
7. **Mapping coverage**: aws_syngo.csv covers ~320K studies (those with S3 videos). study_deid_map covers ~224K HeartLab studies. Studies outside these mappings are not captured.
8. **Patient propagation assumption**: Propagating labels to all patient studies assumes the disease persists across all echos, which is valid for structural diseases (bicuspid, rheumatic) but questionable for acute conditions (STEMI, takotsubo).
9. **Indication field exclusion**: The `--exclude_indication` flag (v3) removes Syngo `Indication` and `REASON_FOR_STUDY` fields entirely. This is a blunt filter — some true positives mentioned only in indication may be lost. However, HeartLab SENTENCE is the dominant and cleanest source (60-93% of definitive positives).
10. **Clinical vs. echo diagnosis**: STEMI and takotsubo are clinical diagnoses — the echo supports but doesn't make the diagnosis. Labels reflect "echo ordered in context of disease" rather than "echo shows disease."
