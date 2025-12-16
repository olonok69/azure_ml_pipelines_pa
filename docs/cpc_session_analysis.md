# CPC Session Catalogue Overview

_Date generated: 2025-10-13_

This note compares the current CPC session exports for 2024–2025 across the three provided files:

- `PA/data/cpcn/CPC2025_session_export.csv`
- `PA/data/cpcn/CPCN24_session_export.csv`
- `PA/data/cpcn/CPCN25_session_export.csv`

Only the columns `theatre__name`, `title`, `stream`, and `synopsis_stripped` were analysed. Records were de-duplicated on these four columns to avoid double counting individual sessions that appear multiple times for different speakers.

## Snapshot summary

| Event | Total sessions | Venues represented | Unique streams | Synopsis available | Synopsis missing | Sessions tagged with ≥2 streams |
| --- | ---:| --- | ---:| ---:| ---:| ---:|
| CPC 2025 | 196 | 16 theatres (see below) | 11 | 163 (83.2%) | 33 (16.8%) | 47 (24.0%) |
| CPC North 2024 | 80 | 7 theatres | _None supplied_ | 65 (81.3%) | 15 (18.7%) | 0 |
| CPC North 2025 | 77 | 6 theatres | 5 | 31 (40.3%) | 46 (59.7%) | 3 (3.9%) |

Percentages are relative to the total session counts per event.

## CPC 2025 (London)

- **Theatre footprint (top five by count)**
  1. Strategy & Policy Forum — 19 sessions (9.7%)
  2. Integrated Care & Primary Care Theatre — 16 (8.2%)
  3. MORPh Theatre — 16 (8.2%)
  4. Technology & Innovation Theatre — 16 (8.2%)
  5. Technical Services Theatre — 15 (7.7%)

- **Stream coverage** (11 distinct values; counts shown): Prescribing (38), Senior Pharmacist (21), Primary Care (26), Secondary Care (24), Leaders (17), Pharmacy Technician (20), Pharmacist (19), Specialist Pharmacist (15), Community (5), Technical Services (3), Home Care (1).

- **Synopsis completeness**: 163 sessions include narrative abstracts, while 33 lack any synopsis. Missing examples include `CPC Awards`, `Community pharmacy Independent Prescribing Pathfinders; integration with neighbourhood health`, and `Ready to administer, ready to save: enhancing safety and value with RTA injectables`.

- **Multi-stream tagging**: 47 sessions carry more than one stream label. Representative examples:
  - `An update on asthma and new national guidelines - a paradigm shift` (Prescribing; Primary Care; Secondary Care)
  - `Navigating the cardiovascular renal metabolic space` (Prescribing; Primary Care; Secondary Care)
  - `Bumps, babies & bonding: the pharmacist’s role in perinatal mental health` (Pharmacist; Specialist Pharmacist)

## CPC North 2024

- **Theatre footprint**: Innovation & Integration Theatre (16 sessions), Strategy Theatre (16), Keynote Theatre (13), Clinical Theatre 1 (13), Clinical Theatre 2 (13), Practical Skill Zone (5), Room 8 (4).

- **Stream metadata**: No stream tags are populated in this export, so stream-level segmentation is not possible.

- **Synopsis completeness**: 65 sessions have descriptive copy; 15 are blank. Missing examples include `An exciting time for ATMPs: what do you need to know?`, `Pharmacy Technician Professional Practice - enabling pharmacy reforms`, and `Developing an integrated pharmacy workforce`.

- **Multi-stream tagging**: None of the rows include multiple stream values (consistent with the absence of stream data).

## CPC North 2025

- **Theatre footprint**: Innovation & Integration Theatre (16 sessions), Strategy Theatre (16), Clinical Theatre 1 (15), Keynote Theatre (13), Clinical Theatre 2 (13), Room 4 (bookable sessions) (4).

- **Stream coverage** (5 distinct values): Clinical Practice (23), Leadership & Management (14), Education (4), Education & Training (1), Research (1).

- **Synopsis completeness**: Only 31 sessions provide a synopsis, leaving 46 (59.7%) blank. Missing examples include `Pharmacy Technician professional practice`, `Session to be confirmed`, and `Chief Pharmaceutical Officer for England's keynote address`.

- **Multi-stream tagging**: 3 sessions carry multiple stream labels, notably `The future of ICBs and ICSs within the current NHS England landscape` (Leadership & Management ×3) and `The path to credentialling` (Leadership & Management; Research).

## Observations & follow-ups

- **Data quality divergence**: Stream values are fully absent in CPC North 2024 and sparse for CPC North 2025 compared with the richer tagging in CPC 2025. If consistent taxonomy is required, upstream enrichment will be needed for the northern datasets.
- **Synopsis coverage drop**: CPC North 2025 shows a marked decline in synopsis completion (40%) relative to the 2024 export (81%). Consider prioritising copy collection for keynote and flagship sessions called out above.
- **Deduplication**: Multiple rows per session (e.g., per speaker) exist in the raw exports. The counts above normalise this by deduplicating on theatre, title, stream, and synopsis before aggregation.

_Data prepared via `scripts/summarize_sessions.py`._
