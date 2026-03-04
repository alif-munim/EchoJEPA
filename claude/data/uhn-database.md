# UHN Echocardiography Database Reference

Detailed context for UHN database exploration lives in:
- `uhn_echo/nature_medicine/data_exploration/CLAUDE.md` — sqlite3 rules, echo.db schema, documentation system, query conventions

## Quick Reference

- All data is in `uhn_echo/nature_medicine/data_exploration/echo.db` (single sqlite3 database)
- **Syngo** tables (primary, 2005-2019): 26M measures, 6.6M observations, 390K studies
- **HeartLab** tables (legacy, 2002-2014): 6.4M findings, 2.6M measurements, 432K studies
- 12-year overlap (2005-2014): deduplicate by accession number
- 314K patients across both systems

## Key Rules When Working in data_exploration/

- Always use `sqlite3 echo.db "SELECT ..."` — never grep/awk on CSVs
- Value columns are TEXT — always `CAST(Value AS REAL)`
- syngo_study_details uses UPPERCASE column names (STUDY_REF not StudyRef)
- Read `docs/` files before querying (schemas, guides, reference tables documented there)

## Rare Disease Cohorts (deduped)

HCM 12,291 / Amyloidosis 1,174 / Endocarditis 5,236 / Takotsubo 186 / Constrictive 376

## Key Outcome

No mortality, outcome, MRN, OHIP, or external linkage in UHN data. Outcome prediction experiments are MIMIC-only.
