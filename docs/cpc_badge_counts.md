# CPC Registration and Attendance Summary

_Date generated: 2025-10-13_

This report summarises badge registration and attendance counts for the CPC North 2024 and CPC London 2025 events, using the specified registration and demographics exports. All counts are restricted to badge holders whose `badge_type` matches the approved list:

- Colleague Delegate
- Commercial Delegate HCP
- Commercial Delegate ORDM
- Delegate HCP
- Delegate ORDM
- Guest
- Guest HCP
- Guest Non-HCP
- International Delegate
- International Delegate HCP

Badge identifiers (`badge_id`) are treated as the visitor identifier, and attendance is derived from the `attended` field (`"1"` = attended). All counts remove duplicate `badge_id` entries before measurement.

## Data Sources

| Event | Registration File | Demographics File |
| --- | --- | --- |
| CPC North (2024) | `PA/data/cpcn/20251006_eventregistration_CPCN24_25_graphql.json` | `PA/data/cpcn/20251007_CPCN24_25_demographics_graphql.json` |
| CPC London (2025) | `PA/data/cpcn/20251006_eventregistration_CPC25_graphql.json` | `PA/data/cpcn/20251006_eventregistration_CPC25_demographics_graphql.json` |

## Filters Applied

- **CPC North 2024:**
  - `metadata_source_filename == "20251006_eventregistration_CPCN24.json"`
  - `show_ref` in {`CPCN24`, `CPCN2024`}

- **CPC London 2025:**
  - `metadata_source_filename == "20251006_eventregistration_CPC25.json"`
  - `show_ref` in {`CPC2025`, `CPC25`}

- `badge_type` in the valid badge list above
- Attendance counted where `attended == "1"`

## Results

### CPC North 2024

- Unique badge IDs (registered, deduped): **2,071**
- Unique badge IDs with `attended = "1"`: **871**
- Demographics linkage:
  - Unique badge IDs in demographics (CPCN24/CPCN2024): **2,315**
  - Overlap with filtered registrations: **1,561** badge IDs (67.4%)

### CPC London 2025

- Unique badge IDs (registered, deduped): **5,347**
- Unique badge IDs with `attended = "1"`: **2,790**
- Demographics linkage:
  - Unique badge IDs in demographics (CPC2025/CPC25): **5,239**
  - Overlap with filtered registrations: **4,837** badge IDs (92.3%)

## Notes

- Percentages are calculated as overlap ÷ demographics unique badge IDs.
- Counts are deduplicated by `badge_id` only; no additional household/company collapsing performed.
- All figures are based solely on the files listed above and the filters described.
