# CPCN / CPC Registration Schema Adapter

This helper script normalises the latest CPCN / CPC registration and demographic JSON feeds so they can be consumed by the existing Personal Agendas pipeline.

## What it does

* Merges the 2024 and 2025 CPCN drops for both registration and demographic data.
* Maps every record onto the GraphQL-style schema the modern pipeline expects (default behaviour).
* Optionally rewrites the same data into the legacy/"old" schema used when `old_format: true` is configured (matching the structure of the reference files in `data/cpcn/reference/`).

## Requirements

* Python 3.9+ (the project virtual environment in `.venv` works fine).
* The raw input files described in the project brief stored under `PA/data/cpcn/`.

## Usage

Run the script from the repository root:

```powershell
# Emit GraphQL-format outputs (default)
python PA/transform_cpc_data.py

# Emit GraphQL + legacy-format outputs (for old_format=true runs)
python PA/transform_cpc_data.py --legacy
```

Outputs are written beside the raw inputs in `PA/data/cpcn/`. The legacy artefacts follow the same naming pattern with a `_legacy` suffix, for example:

* `20251006_eventregistration_CPCN24_25_graphql.json`
* `20251006_eventregistration_CPCN24_25_legacy.json`

When `old_format` is set to `true` in `config/config_cpcn.yaml`, point the pipeline to the `_legacy.json` files to keep downstream processing unchanged. Leave the default GraphQL outputs in place for `old_format: false`.

## Integration tips

* The script is idempotent—rerun it whenever new source JSON files arrive.
* Update `SHOW_DETAILS` in `transform_cpc_data.py` if future shows require different `event_id` or `metadata_show_date` values.
* Install `python-Levenshtein` in the virtual environment to silence fuzzywuzzy's optional optimisation warning.
