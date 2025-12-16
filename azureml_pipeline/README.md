# Azure Machine Learning Pipeline (Personal Agendas)

This directory contains the Azure Machine Learning (Azure ML) production pipeline implementation for the Personal Agendas (event recommendation) platform. It operationalizes the core on-prem / local pipeline (`PA/`) inside Azure ML as a reproducible, scheduled, and monitored multi-step workflow.

## Objectives
- Ingest & preprocess raw event data (registration, scans, sessions)
- Prepare Neo4j graph entities & relationships
- Generate semantic embeddings for current-year sessions
- Produce personalized session recommendations
- Persist artifacts & metrics for downstream activation (e.g., email, CRM)

## Components Overview
| File | Purpose |
|------|---------|
| `pipeline_config.yaml` | Declarative definition of Azure ML pipeline metadata, compute, steps, IO contracts & parameters. |
| `azureml_step1_data_prep.py` | Step 1: Data ingestion & normalization (registration, scan, session processing) + standard outputs. |
| `azureml_step2_neo4j_prep.py` | Step 2: Creates / updates Neo4j nodes & relationships (visitors, sessions, streams, cross-year links). |
| `azureml_step3_session_embedding.py` | Step 3: Generates sentence-transformer embeddings for sessions (GPU optional). |
| `azureml_step4_recommendations.py` | Step 4: Runs recommendation engine using embeddings + historical / similarity logic. |
| `archive/` | Previous or experimental versions (kept for reference, not executed). |

## Step Breakdown
### 1. Data Preparation (`azureml_step1_data_prep.py`)
Processes raw source files (referenced via datastore `landing`) using the core processors:
- `RegistrationProcessor`
- `ScanProcessor`
- `SessionProcessor`

Outputs (mounted as `uri_folder`):
- `registration_output` (normalized current & past year registration + demographics)
- `scan_output` (attendance / scan join outputs)
- `session_output` (filtered current & past sessions + streams catalog)
- `metadata_output` (auxiliary logs / metadata)

Key features:
- Optional incremental flag (currently mostly full-refresh semantics)
- Key Vault secret resolution for environment (.env) when running inside Azure ML
- Standardized file naming derived from event config (e.g., vet vs ecomm)

### 2. Neo4j Preparation (`azureml_step2_neo4j_prep.py`)
Consumes Step 1 outputs and loads / updates Neo4j graph:
- Visitor nodes (this year + past years)
- Session nodes (this + past)
- Stream nodes & HAS_STREAM links
- Same_Visitor cross-year identity linking
- Optional job_to_stream & specialization_to_stream (config driven; disabled for ecomm)

Incremental mode mapping: `incremental=True` translates to `create_only_new=True` in underlying processors (preserves existing sessions & recommendations when configured).

### 3. Session Embeddings (`azureml_step3_session_embedding.py`)
Generates embeddings for `Sessions_this_year` using Sentence Transformers (default: `all-MiniLM-L6-v2`).
- Can target GPU compute (`gpu-cluster`) if configured.
- Skips re-embedding unchanged sessions when incremental mode is enabled.
- Writes embedding artifact folder for Step 4.

### 4. Recommendations (`azureml_step4_recommendations.py`)
Produces personalized recommendations combining:
- Past-year attendance (returning visitors)
- Similar visitor matching (attribute similarity vectors defined in config)
- Content similarity via embeddings (cosine similarity)
- Fallback: popular past sessions (randomized selection within a top slice)

Artifacts:
- JSON: `visitor_recommendations_<show>_<timestamp>.json`
- CSV: Flattened recommendations with visitor & session attributes (now enriched with Email, name & contact fields if configured)
- Neo4j updates: `IS_RECOMMENDED` relationships + `has_recommendation` flag
- Metrics: counts, processing time, error summaries

## Configuration (`pipeline_config.yaml`)
Key sections:
- `azure_ml.compute`: Cluster name, min/max autoscale, VM size
- `steps.*`: Enabled flag, script name, environment, IO bindings, step parameters
- `monitoring`: MLflow toggle & log retention
- `error_handling`: Retry policy
- `schedule`: Optional CRON-based triggering (disabled by default)
- `tags`: Propagated to Azure ML runs for governance

Environment variable placeholders (e.g., `${EVENT_TYPE}`, `${SUBSCRIPTION_ID}`) are resolved externally (CLI, orchestrator, or Azure DevOps / GitHub Actions).

## Secrets & Key Vault
Each step attempts to load Neo4j & other secrets via:
1. Existing environment variables (`NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`)
2. Azure Key Vault (if running in Azure ML context & `use_keyvault=True`)
3. Writes / updates `.env` at `PA/keys/.env` for downstream processors

## Execution Patterns
You typically orchestrate the Azure ML pipeline using a submission notebook or a Python client script.

Notebooks (in `notebooks/`):
- `submit_pipeline.ipynb`
- `submit_pipeline_step2.ipynb`
- `submit_pipeline_with_step3.ipynb`
- `submit_pipeline_complete.ipynb`

These notebooks:
- Authenticate to Azure ML workspace
- Load or register Data assets / Environments
- Resolve `EVENT_TYPE` (e.g., `vet` or `ecomm`)
- Submit the multi-step pipeline run & monitor status

## Data Interfaces
| Step | Consumes | Produces |
|------|----------|----------|
| 1 | Datastore: landing raw folders | Normalized CSV/JSON folders (registration/scan/session/metadata) |
| 2 | Step 1 folders | Graph update side-effects + Neo4j export snapshots (if any) |
| 3 | Step 1 sessions + Neo4j | Embedding vectors (folder) |
| 4 | Neo4j + Embeddings | Recommendation JSON/CSV + Neo4j relationships + metrics |

## Incremental vs Full Refresh
- Full refresh: (default) rebuilds / reprocesses all eligible entities
- Incremental (set `incremental: true` / step param):
  - Step 2 (`create_only_new=True`) preserves existing sessions & recommendations
  - Step 3 can skip unchanged session embeddings (logic in processor)
  - Step 4 processes only visitors without `has_recommendation="1"` when configured

## Monitoring & Observability
- MLflow (if enabled) logs parameters (event type, thresholds), metrics (counts, timings)
- Per-step log files stored in each step working directory
- Azure ML Run UI provides lineage & artifact browsing

## Error Handling
Configured retries (`retry_count`, `retry_delay_seconds`). If `continue_on_step_failure=false`, pipeline aborts on first failing mandatory step.

## Scheduling
`pipeline_config.yaml` includes an optional CRON schedule block—enable to automate periodic runs (e.g., weekly refresh). External orchestrators (Azure DevOps, GitHub Actions) may also invoke submission notebooks or a CLI wrapper.

## Extensibility
To add a new step:
1. Create a new `azureml_stepX_<name>.py` script
2. Add a block under `steps:` in `pipeline_config.yaml`
3. Define inputs referencing previous step outputs (`<previous>.outputs.alias`)
4. Register / mirror environment configuration
5. Update notebooks if selective submission logic is required

## Environments
Two logical environments (example):
- `personal_agendas_env` (CPU) – general processing & recommendations
- `personal_agendas_gpu_env` (GPU) – embeddings acceleration

Ensure they contain dependencies from root `requirements.txt` plus any Azure ML SDK & sentence-transformers packages.

## Common Customizations
| Goal | Change |
|------|-------|
| Increase recommendation depth | Adjust `steps.recommendations.parameters.top_k` & pipeline config thresholds |
| Switch model | Modify embeddings config in event YAML + environment dep (if new model) |
| Enable job/specialization stream logic | Toggle flags in event YAML (`neo4j.job_stream_mapping.enabled`) |
| Add enrichment fields to CSV | Update `recommendation.export_additional_visitor_fields` in event config |

## Quick Validation Checklist
- Config file path resolves inside step container
- Secrets loaded (log lines confirm Neo4j URI)
- Step 1 outputs contain expected CSV counts
- Step 2 preserves sessions if incremental
- Step 3 embedding count == session count (filtered)
- Step 4 CSV has Email / Forename / Surname when enrichment configured

## Troubleshooting
| Symptom | Possible Cause | Action |
|---------|----------------|-------|
| Neo4j auth failure | Missing Key Vault secret mapping | Verify Key Vault name & secret names, check logs |
| Empty recommendations | No embeddings or no past sessions | Confirm Step 3 success, inspect JSON metadata block |
| Pipeline stuck provisioning | Insufficient quota / wrong VM size | Adjust `vm_size` or region capacity |
| Missing enrichment columns | Not configured in event YAML | Add to `export_additional_visitor_fields` and rerun Step 4 |

## Next Enhancements (Suggested)
- Add data asset registration for outputs (for lineage)
- Add SLA metrics push (App Insights / Log Analytics)
- Introduce drift detection on embedding distributions
- Parameterize similarity thresholds per segment

---
For submission usage, refer to the notebooks in `notebooks/` which demonstrate end-to-end pipeline registration and execution.
