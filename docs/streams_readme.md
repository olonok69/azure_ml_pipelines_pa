# Streams Processing Overview

This guide explains how session stream metadata is generated, enriched, and consumed across the Personal Agendas pipeline. It captures the legacy behaviour and the new backfill logic introduced in October 2025.

## 1. Stream Lifecycle

1. **Raw data ingestion**
   - `PA/session_processor.py` loads the current-year and historical session CSVs defined under `session_files` in each event configuration.
   - The processor filters placeholder sessions, normalises key fields, and writes cleaned outputs to `data/<event>/output/` (file names come from `session_output_files.processed_sessions`).

2. **Unique stream extraction**
   - While preparing CSV outputs, the processor gathers all distinct stream values (splitting on `;`) to form the canonical stream list for the run (`SessionProcessor.unique_streams`).

3. **Catalog generation & caching**
   - A stream description catalog (default `streams.json`) is produced in `output/`. The processor:
     - Reuses a cached JSON file if `stream_processing.use_cached_descriptions` is `true` and the cache exists.
     - Otherwise calls the configured LLM (Azure OpenAI or OpenAI) to generate concise descriptions per stream.
     - Stores the responses in both memory and `streams_cache.json` for future runs.
   - Enhanced catalogs can be produced by supplying alternate filenames (for example, `streams_enhanced.json`).

## 2. New Stream Backfill Functionality

The October 2025 update introduces an optional backfill pass that uses the LLM to classify sessions missing stream assignments.

### 2.1 Configuration Flags

`config/<event>.yaml` includes a `stream_processing` block:

```yaml
stream_processing:
  use_cached_descriptions: false
  create_missing_streams: true
  use_enhanced_streams_catalog: true
```

- `create_missing_streams`: enables the backfill workflow after the cleaned CSV files are saved.
- `use_enhanced_streams_catalog`: forces the classifier to prefer enhanced catalogs (e.g., `streams_enhanced.json`); falls back to the regular catalog otherwise.
- `use_cached_descriptions`: unchanged behaviour controlling whether the LLM is called for catalog generation.

### 2.2 Backfill Workflow (Session Processor)

Triggered at the end of `SessionProcessor.process()`:

1. **File backups** – each processed CSV (`session_this_filtered_valid_cols.csv`, etc.) is copied to a timestamped `*.bak` file prior to modification.
2. **Catalog loading** – the processor loads the enhanced or baseline stream catalog to supply stream names and descriptions.
3. **Candidate selection** – for every session lacking a stream value (blank, null, or sentinel values):
   - If the theatre is known, the processor prefers streams already linked to that theatre across all session CSVs.
   - Otherwise it falls back to the full stream catalog (capped at 60 entries per prompt to keep requests efficient).
4. **LLM classification** – the processor sends the theatre-specific stream descriptions, session title, and synopsis to the same LLM endpoint used for catalog generation. The prompt enforces a semicolon-delimited response with one to three stream names exactly matching the catalog.
5. **Persisting results** – rows updated with new stream assignments are written back to the original CSV file. A theatre-level cache is updated so subsequent sessions benefit from newly inferred streams.
6. **Metrics capture** – the processor tracks counts such as `sessions_backfilled`, `sessions_skipped_empty_synopsis`, and `sessions_failed_llm`. These metrics are exposed to:
   - `utils/mlflow_utils.py` for MLflow logging (`session_missing_streams_*`).
   - `utils/summary_utils.py` where they appear under `session.missing_stream_backfill` in `logs/processing_summary.json`.

If the LLM is unavailable or no catalog can be loaded, the backfill step logs a warning and exits without mutating files.

## 3. Where Streams Are Used

| Component | Usage of Streams |
|-----------|------------------|
| **Session Processor** (`PA/session_processor.py`) | Generates stream lists, descriptions, and now backfills missing entries. Outputs the CSVs consumed downstream. |
| **Neo4j Session Processor** (`PA/neo4j_session_processor.py`) | Loads the processed session CSVs, creates `Stream` nodes, and links sessions to streams via `HAS_STREAM` relationships. |
| **Neo4j Job / Specialization Stream Processors** | Optionally map visitor job roles or specialisations to streams, depending on event configuration flags. |
| **Session Embedding Processor** | Embeds sessions and can incorporate stream descriptions where configured. |
| **Session Recommendation Processor** | Relies on stream metadata to score similarity and filter candidate sessions. |
| **Reports & Exports** | Stream data appears in recommendation CSVs and any analytics derived from `streams.json` or processed session files. |

## 4. Key Artifacts

- `data/<event>/output/streams.json` (and optional enhanced variants): stream descriptions keyed by stream name.
- `data/<event>/output/session_*.csv`: cleaned session datasets enriched with stream values (including any backfilled entries).
- `data/<event>/output/streams_cache.json`: reusable description cache when `use_cached_descriptions` is enabled.
- `logs/processing_summary.json`: includes `session.missing_stream_backfill` metrics summarising the most recent run.
- MLflow metrics (`session_missing_streams_*`) for long-term monitoring of backfill effectiveness.

## 5. Operational Notes

- **Authentication** – both catalog generation and backfill reuse the environment variables loaded from `keys/.env` (`OPENAI_API_KEY` or Azure equivalents). Missing credentials will disable LLM calls but the pipeline continues.
- **Prompt determinism** – while the classifier enforces strict formatting, responses can still vary slightly between runs. Backups ensure a roll-back path when needed.
- **Performance** – large events may trigger many LLM calls; consider enabling caching (`use_cached_descriptions`) and reviewing token quotas.
- **Extensibility** – additional heuristics (e.g., keyword-based matching) can be inserted ahead of the LLM call if deterministic assignment is required.

For implementation references, see:
- `PA/session_processor.py` (methods `extract_unique_streams`, `save_streams_catalog`, `backfill_missing_streams`).
- `PA/utils/mlflow_utils.py` (logging backfill metrics).
- `PA/utils/summary_utils.py` (surfacing metrics in summary reports).
- Event configuration files (e.g., `PA/config/config_ecomm.yaml`) for examples of the new stream settings.
