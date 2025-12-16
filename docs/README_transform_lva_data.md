# LVS / LVA Registration Schema Adapter

`PA/transform_lva_data.py` normalises the latest LVS (London Vet Show) registration and demographic exports so they align with the schema expected by the Personal Agendas pipeline.

## What it does

- Converts the 2024 and 2025 LVS registration feeds into the GraphQL-style schema used by downstream steps.
- Performs the equivalent transformation for the demographic question feeds.
- Emits a combined 24/25 dataset alongside the year-specific outputs to simplify pipeline wiring.
- Optionally rewrites all datasets into the legacy/"old" schema required when `old_format: true` is configured (mirroring `data/cpcn/reference/`).

## Requirements

- Python 3.9+ (the project virtual environment in `.venv` works fine).
- Raw JSON inputs placed under `PA/data/lva/`:
  - `20251104_eventregistration_LVS24.json`
  - `20251104_eventregistration_LVS25.json`
  - `20251104_eventregistration_LVS24_demographics.json`
  - `20251104_eventregistration_LVS25_demographics.json`
- Reference GraphQL exports already present in `PA/data/cpcn/reference/` (the script reuses them to determine the target schema fields).

## Usage

Run the script from the repository root:

```powershell
# Emit GraphQL-format outputs (default)
python PA/transform_lva_data.py

# Emit GraphQL + legacy-format outputs (for old_format=true runs)
python PA/transform_lva_data.py --legacy
```

All artefacts are written next to the source files in `PA/data/lva/`. For each dataset you will see:

- Year-specific GraphQL files, for example:
  - `20251104_eventregistration_LVS24_graphql.json`
  - `20251104_eventregistration_LVS24_demographics_graphql.json`
- Combined 2024/2025 GraphQL files:
  - `20251104_eventregistration_LVS24_25_graphql.json`
  - `20251104_eventregistration_LVS24_25_demographics_graphql.json`
- When run with `--legacy`, equivalent `_legacy.json` companions for every GraphQL export (used when `old_format: true`).

Point the pipeline configuration (`config/config_vet_lva.yaml`) at the `_graphql.json` outputs by default. Switch to the `_legacy.json` variants when running in legacy mode.

## Integration tips

- The script is idempotent—rerun it whenever updated registration/demographic drops arrive.
- Update `SHOW_DETAILS` inside `transform_lva_data.py` if new show codes or event IDs are introduced.
- Keep the raw filenames unchanged; the script relies on their current naming to build combined outputs.
- If additional schema fields appear in future drops, refresh the reference GraphQL files or extend the transformation logic accordingly.
