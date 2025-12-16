# Restoring Recommendation Graph Data

This document explains how to replay a historical recommendations export back into Neo4j using `scripts/restore_recommendations_from_export.py`. Use it whenever the recommendation pipeline (step 10) cannot be rerun but you still have the archived output files.

## What the Script Does

- Reads the JSON export produced by `session_recommendation_processor.py` (and optionally its CSV twin).
- For every `Visitor_this_year` node present in the export, resets:
  - `has_recommendation`
  - `recommendations_generated_at`
  - control-group flag (always `0` for CPCN because control groups were disabled).
- Recreates each `IS_RECOMMENDED` relationship from the visitor to the corresponding `Sessions_this_year` node, copying the stored similarity score and timestamp.

> The config file (`config/config_cpcn.yaml`) keeps `session_recommendation_processing: false`, so this helper is the only step touching Neo4j during the rebuild.

## Prerequisites

1. **Environment variables** – `keys/.env` must contain the `NEO4J_URI_TEST`, `NEO4J_USERNAME`, and `NEO4J_PASSWORD` entries. The script loads the env automatically based on the config `env_file` entry.
2. **Config file** – default is `config/config_cpcn.yaml`. Update the `--config` flag if you run it for another show.
3. **Recommendation exports** – JSON is required, CSV is optional but recommended:
   - `data/cpcn/recommendations/visitor_recommendations_cpcn_20251113_120352.json`
   - `data/cpcn/recommendations/visitor_recommendations_cpcn_20251113_120352.csv`
4. **Python dependencies** – whatever is already used for the pipeline (`poetry`/`pip install -r PA/requirements.txt`).

## Basic Usage

Run from the repo root (`PA/`):

```powershell
python scripts/restore_recommendations_from_export.py `
  --config config/config_cpcn.yaml `
  --json data/cpcn/recommendations/visitor_recommendations_cpcn_20251113_120352.json `
  --csv data/cpcn/recommendations/visitor_recommendations_cpcn_20251113_120352.csv
```

### Recommended Flow

1. **Dry run first** – counts visitors/relationships without touching Neo4j:
   ```powershell
   python scripts/restore_recommendations_from_export.py --dry-run ...
   ```
2. **Live run** – rerun without `--dry-run` to apply updates. The script deletes existing `IS_RECOMMENDED` edges for each visitor (unless you pass `--no-replace`) and recreates them from the export.
3. **Validate** – spot-check the graph, or rerun downstream reports (e.g., `deep_agents_reports/reporting_pipeline.py`) to ensure they see the restored relationships.

## CLI Flags

| Flag | Description |
| --- | --- |
| `--config` | YAML config path. Defaults to `config/config_cpcn.yaml`. |
| `--json` | **Required.** Path to the recommendation JSON export. |
| `--csv` | Optional CSV export. If provided, the script warns if CSV/JSON sessions diverge. |
| `--batch-size` | Visitor batch size per transaction (default `100`). Increase if the DB is fast and stable; decrease if you hit timeouts. |
| `--dry-run` | Parse files and report counts without writing to Neo4j. |
| `--no-replace` | Skip deletion of existing `IS_RECOMMENDED` relationships; only adds missing ones. Use only if you explicitly want to keep the current graph state. |
| `--log-level` | Python logging level (`INFO`, `DEBUG`, ...). |

## Output & Logging

- The script logs visitors or sessions missing from Neo4j. Missing visitors are skipped; missing sessions remove only the affected relationships.
- A show mismatch warning appears if the export metadata (`metadata.show`) disagrees with the config’s `neo4j.show_name`; the config wins in that case.
- Final stats summarize how many visitors were updated and how many `IS_RECOMMENDED` relationships were created.

## Troubleshooting

- **Missing visitors** – rerun `neo4j_visitor_processor.py` / `neo4j_visitor_relationship_processor.py` before restoring recommendations.
- **Missing sessions** – rerun `session_processor.py` and `session_embedding_processor.py` to push `Sessions_this_year` nodes back into the graph.
- **Credential errors** – confirm the `keys/.env` entries or override `neo4j.uri`, `neo4j.username`, `neo4j.password` in the YAML.

## Sample Verification Query

After the import, you can verify counts directly in Neo4j Browser or Cypher-shell:

```cypher
MATCH (v:Visitor_this_year)-[r:IS_RECOMMENDED]->(s:Sessions_this_year)
WHERE v.show = 'cpcn'
RETURN count(DISTINCT v) AS visitors_with_recs,
       count(r) AS total_recommendations,
       count(DISTINCT s) AS sessions_referenced;
```

This should roughly match the `metadata` block inside the JSON export (e.g., `1904` visitors and `19,023` relationships for the 2025-11-13 CPCN snapshot).
