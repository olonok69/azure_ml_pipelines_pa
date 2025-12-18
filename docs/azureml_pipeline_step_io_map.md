# Azure ML Pipeline I/O Map

This document mirrors the local Personal Agendas step map but focuses on the four Azure ML pipeline steps defined in [azureml_pipeline/pipeline_config.yaml](azureml_pipeline/pipeline_config.yaml). It highlights, per step, the required inputs, generated outputs, and how Step 1 artifacts are transferred downstream so the Azure ML pipeline behaves identically to the local pipeline for both vet (LVA/BVA) and non-vet (CPCN) shows.

## Step 1 – Data Preparation (`azureml_step1_data_prep.py`)
- **Purpose**: Run `RegistrationProcessor`, `ScanProcessor`, and `SessionProcessor` inside Azure ML, hydrated with files downloaded from the landing datastore.
- **Key behaviors**:
  - `download_blob_data()` builds a lookup from the selected show config (e.g., [PA/config/config_vet_lva.yaml](PA/config/config_vet_lva.yaml) vs [PA/config/config_cpcn.yaml](PA/config/config_cpcn.yaml)) so every declared `input_files`, `scan_files`, and `session_files` entry is placed under `data/<event>/...`, regardless of whether the show is vet or not.
  - `copy_outputs_to_azure_ml()` copies the normalized CSVs from `data/<event>/output/` into the Azure ML output mounts so downstream steps see the same filenames the local processors expect.

| Input | Source | Vet-specific notes | Non-vet notes |
| --- | --- | --- | --- |
| `input_data` (uri_folder) | `azureml://datastores/landing_pa/paths/landing/azureml/` | Includes vet scans (LVS/BVA) and theatre capacity CSVs; the config file ensures they land under the correct `data/lva/...` tree. | Includes CPCN/CPC entry scans and pharmacy session exports; config switches `event.name` so they land under `data/cpcn/...`. |
| `config_file` (uri_file) | `PA/config/config_${EVENT_TYPE}.yaml` | Enables vet logic (`enable_filtering`, theatre capacities, job/specialization mappings). | For CPCN, filtering/ capacity limits disabled and practice-specific mappings omitted. |

| Output | Contents | Used by |
| --- | --- | --- |
| `registration_output` | `df_reg_demo_*.csv`, registration_with_demographic CSVs, raw exports | Step 2 (Neo4j visitor + relationship processors) |
| `scan_output` | `scan_*` CSVs, `sessions_visited_*`, entry scan references | Step 2 relationship processor |
| `session_output` | `session_*_filtered_valid_cols.csv`, `streams.json`, theatre catalog | Step 2 Neo4j session processor + Step 4 capacity enforcement |
| `metadata_output` | Logs, JSON summaries, MLflow artifacts | All steps for audit |

**Transfer guarantees**: Vet and non-vet configs enumerate every file consumed later (e.g., `recommendation.theatre_capacity_limits.capacity_file`). Step 1 records missing files before finishing, so any parity issue is surfaced immediately.

## Step 2 – Neo4j Preparation (`azureml_step2_neo4j_prep.py`)
- **Purpose**: Load Step 1 outputs back into the canonical folder structure and run the five Neo4j processors.
- **Key behaviors**:
  - `copy_input_data()` rebuilds `data/<event>/output` from the mounted Step 1 folders, copying each expected CSV. It also propagates support files (job/specialization mapping CSVs) if the config references them, so vet-only features activate exactly as they do locally.
  - The processors read the same config file as Step 1, so toggles such as `neo4j.job_stream_mapping.enabled` or `pipeline_steps.session_recommendation_processing` remain show-specific.

| Input | Source | Notes |
| --- | --- | --- |
| `registration_data` | `data_preparation.outputs.registration_output` | Provides `df_reg_demo_*` CSVs. |
| `scan_data` | `data_preparation.outputs.scan_output` | Provides attendance and `scan_bva_past.csv` / `scan_lva_past.csv`. |
| `session_data` | `data_preparation.outputs.session_output` | Provides session catalogs and `streams.json`. |
| `config_file` | `PA/config/config_${EVENT_TYPE}.yaml` | Same config ensures vet vs non-vet logic. |

| Output | Contents | Notes |
| --- | --- | --- |
| `metadata_output` | Neo4j processor summaries (`*_statistics.json`, logs) | Used only for diagnostics; data lives in Neo4j. |

**Transfer guarantees**: Because Step 2 recreates the exact `data/<event>/output` tree, Steps 3–4 (and any reruns) see identical file paths as the local run, keeping vet-specific CSVs (e.g., `session_last_filtered_valid_cols_lva.csv`) available.

## Step 3 – Session Embeddings (`azureml_step3_session_embedding.py`)
- **Purpose**: Reuse the Neo4j state populated in Step 2 to embed `Sessions_this_year` via `SessionEmbeddingProcessor`.
- **Inputs**:
  - `config_file`: selects show-specific embedding flags (same YAML as earlier steps).
  - Environment / Key Vault: provides Neo4j credentials so the processor queries the same graph targeted locally.
- **Outputs**:
  - `metadata_output`: embedding summaries and completion markers (no additional data assets; embeddings are written directly to Neo4j nodes).

**Dependency guarantee**: By depending on `neo4j_preparation`, Step 3 only starts after Neo4j nodes mirror the Step 1 outputs, ensuring deterministic embeddings for both vet and non-vet runs.

## Step 4 – Recommendations (`azureml_step4_recommendations.py`)
- **Purpose**: Run `SessionRecommendationProcessor`, honoring filtering, capacity limits, control groups, and overlap resolution exactly as in local execution.
- **Key behaviors**:
  - Accepts both the config file and `session_data` mount from Step 1. `_remap_session_support_files()` rewrites config paths (e.g., `recommendation.theatre_capacity_limits.capacity_file`) to point at the mounted CSVs, so theatre capacity enforcement works even when filtering is disabled.
  - `_parse_input_data_uri()` + `upload_pipeline_artifacts()` push the final `data/<event>/output` tree back to the landing datastore, matching local artifact locations.

| Input | Source | Vet-specific notes | Non-vet notes |
| --- | --- | --- | --- |
| `config_file` | `PA/config/config_${EVENT_TYPE}.yaml` | Enables filtering rules, control group, theatre capacities. | Keeps filtering and capacity disabled, but still writes recommendations when step enabled. |
| `session_data` | `data_preparation.outputs.session_output` | Supplies theatre CSVs (`teatres.csv`) and current-year session export for capacity/overlap metadata. | Supplies CPCN session exports when vet features are off; `_remap_session_support_files` simply leaves original config paths intact if no files are mounted. |

| Output | Contents | Notes |
| --- | --- | --- |
| `metadata_output` | Recommendation stats, JSON summary of outputs copied to datastore | Azure ML artifact for monitoring. |
| Datastore upload | `data/<event>/output/recommendations/...` | Mirrors local folder structure (JSON, CSV, optional control-group copies). |

**Parity guarantee**: Recommendation logic uses only configuration-driven paths and Step 1 mounts. Since Step 1 already organized files according to the show YAML, Step 4 never hardcodes vet vs non-vet differences, ensuring local vs Azure ML behavior stays aligned.

## End-to-end File Flow Checklist
1. **Landing datastore ➜ Step 1**: Config-driven lookup ensures every declared `input_files` entry (registration, scans, session exports, theatre capacities) is placed under `data/<event>/...`.
2. **Step 1 ➜ Step 2**: `registration_output`, `scan_output`, and `session_output` mounts supply the normalized CSVs; Step 2 copies them into `data/<event>/output`, recreating local folder expectations.
3. **Step 2 ➜ Step 3**: Neo4j processors write directly to the graph; Step 3 depends on Step 2 to guarantee embeddings see the updated sessions regardless of show type.
4. **Step 1/3 ➜ Step 4**: Recommendations reuse Step 1 session artifacts (for theatre limits) and Step 3 embeddings (stored in Neo4j). The artifact upload step sends the same `data/<event>/output/recommendations/...` tree back to the landing datastore, mirroring local runs.

By matching the folder structure and config references at every step, the Azure ML pipeline now guarantees that vet and non-vet shows behave identically across local and cloud executions, with all necessary files transferred between steps.