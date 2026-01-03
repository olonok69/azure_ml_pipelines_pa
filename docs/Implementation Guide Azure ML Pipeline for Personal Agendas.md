# Implementation Guide – Azure ML Pipeline for Personal Agendas

This guide explains how the Azure Machine Learning (Azure ML) pipeline operationalizes the Personal Agendas (PA) processors, how each Azure ML step maps to the legacy 10-step PA pipeline, and how to submit, monitor, and troubleshoot end-to-end runs by using the shared notebooks.

---

## 1. Purpose and Scope
- Provide an authoritative description of the four-step Azure ML pipeline that replaces the on-prem PA automation.
- Document the bindings between Azure ML command components and the corresponding PA modules so future updates stay aligned.
- Capture the notebooks and configuration surfaces that orchestrate the pipeline in non-production, staging, and production environments.

---

## 2. Architecture Overview
```
PA Local Pipeline (10 Steps) ──> Azure ML Managed Pipeline (4 Steps)
├── Step 1: Data Preparation          → azureml_pipeline/azureml_step1_data_prep.py
├── Step 2: Neo4j Preparation         → azureml_pipeline/azureml_step2_neo4j_prep.py
├── Step 3: Session Embeddings        → azureml_pipeline/azureml_step3_session_embedding.py
└── Step 4: Recommendations           → azureml_pipeline/azureml_step4_recommendations.py
```
![Azure ML Pipeline Flow](azureml_pipeline_flow.svg)
- Every Azure ML step wraps the same processor classes invoked inside [PA/pipeline.py](../PA/pipeline.py); we run those processors in hosted compute, copy artifacts between steps with managed outputs, and persist run metadata for reproducibility.
- The pipeline is parameterized via [azureml_pipeline/pipeline_config.yaml](../azureml_pipeline/pipeline_config.yaml), while event-specific options remain in the existing PA configuration files (for example `PA/config/config_vet_lva.yaml`).

---

## 3. Repository Surfaces
- **PA/** – original processors, utilities, configuration files, and local orchestration code (no modifications required for Azure ML runs).
- **azureml_pipeline/** – Azure ML wrappers, shared helpers (`step_input_sync.py`, `neo4j_env_utils.py`), and the declarative pipeline configuration.
- **notebooks/** – authenticated submission entry points: the `submit_pipeline_complete_*.ipynb` series register inputs/environments (if needed) and submit the full four-step pipeline.
- **docs/** – reference materials (I/O maps, key vault mappings, this implementation guide, etc.) that inform operational hand-offs.

---

## 4. End-to-End Flow at a Glance
| Azure ML Step | Bound PA Processors | Primary Responsibilities | Azure ML Inputs | Azure ML Outputs |
| --- | --- | --- | --- | --- |
| Step 1 – Data Preparation | `RegistrationProcessor`, `ScanProcessor`, `SessionProcessor` | Hydrate `.env` from Key Vault, download landing payload, run processors, and stage canonical CSV/JSON artifacts. | `pipeline_input_data` (landing datastore), config file URI, incremental flag | `registration_output`, `scan_output`, `session_output`, `metadata_output` |
| Step 2 – Neo4j Preparation | `Neo4jVisitorProcessor`, `Neo4jSessionProcessor`, `Neo4jJobStreamProcessor`, `Neo4jSpecializationStreamProcessor`, `Neo4jVisitorRelationshipProcessor` | Recreate `data/<event>/output`, push entities/relationships to Neo4j, honor incremental mode (`create_only_new`). | Step 1 outputs + config file + optional Neo4j environment override | `metadata_output` (processor stats, logs) |
| Step 3 – Session Embeddings | `SessionEmbeddingProcessor` | Select Neo4j environment, skip unchanged sessions when incremental, emit embedding summary + completion markers. | Step 1 outputs (for support files), config file, depends_on Step 2 | `metadata_output` (statistics, completion marker) |
| Step 4 – Recommendations | `SessionRecommendationProcessor` | Load embeddings + Neo4j state, build visitor similarity matrices, export CSV/JSON deliverables, copy artifacts back to datastore. | Step 1 outputs, config file, metadata dependency on Step 3 | `metadata_output` (includes recommendation exports) |

---

## 5. Detailed Step Narratives

### 5.1 Step 1 – Data Preparation
- **Entry point**: [azureml_pipeline/azureml_step1_data_prep.py](../azureml_pipeline/azureml_step1_data_prep.py).
- **Key bindings to PA**: imports `RegistrationProcessor`, `ScanProcessor`, and `SessionProcessor` directly from `PA/`. No code forks exist; improvements in PA automatically flow here.
- **Secret management**: `_load_secrets_from_keyvault()` uses `KEYVAULT_NAME` to materialize `PA/keys/.env` so processors read Neo4j/OpenAI credentials exactly as they do locally.
- **Input landing payload**: `download_blob_data()` arranges files according to the `config.event` expectations; it indexes the landing folder and maps every file to the canonical `data/<event>/input` layout.
- **Outputs**: `copy_outputs_to_azure_ml()` fans out the processor results into the three Azure ML outputs plus metadata. Files are categorized by prefix (registration / scan / session) and any unknowns default to session output so downstream steps can still access them.
- **Incremental semantics**: the `incremental` flag is passed to each processor; for example, scan processing can skip unchanged records while session processing still recomputes derived features if configs demand it.

### 5.2 Step 2 – Neo4j Preparation
- **Entry point**: [azureml_pipeline/azureml_step2_neo4j_prep.py](../azureml_pipeline/azureml_step2_neo4j_prep.py).
- **Data hydration**: `stage_step1_outputs()` (shared helper) rebuilds the PA `data/<event>/output` tree from the mounted Azure ML inputs and makes sure the optional support files (job-to-stream, specialization-to-stream, theatre capacity files) are restored under the right relative paths.
- **Processor coverage**: five PA processors are invoked sequentially; each honors `create_only_new` which is set to `incremental=True` so previously created nodes/relationships stay untouched when running incremental refreshes.
- **Neo4j environment overrides**: `select_neo4j_environment()` allows changing the database target per run (for example `neo4j_environment=staging`) without editing YAML.
- **Outputs**: only metadata is emitted because durable artifacts live either inside Neo4j or in the re-created `data/<event>/output`. The metadata contains statistics (counts, durations) for quick validations.

### 5.3 Step 3 – Session Embeddings
- **Entry point**: [azureml_pipeline/azureml_step3_session_embedding.py](../azureml_pipeline/azureml_step3_session_embedding.py).
- **Processor binding**: wraps `SessionEmbeddingProcessor`, so all modeling logic (model name, text recipe, chunking) is inherited from PA configs.
- **Dependencies**: even though embeddings read from Neo4j, the step depends on Step 2 to guarantee ordering and uses Stage 1 outputs for support CSVs when needed.
- **Compute**: defaults to CPU but can be redirected to a GPU cluster by changing the `compute` field inside `pipeline_config.yaml` for `session_embeddings` if/when a GPU SKU is available.
- **Outputs**: summary JSON, per-processor statistics, and a completion marker file ensure Step 4 does not start prematurely.

### 5.4 Step 4 – Recommendations
- **Entry point**: [azureml_pipeline/azureml_step4_recommendations.py](../azureml_pipeline/azureml_step4_recommendations.py).
- **Processor binding**: wraps `SessionRecommendationProcessor`, which calculates multi-signal relevance (attendance history, demographic similarity, embeddings, and popularity fallbacks).
- **Artifacts**: the step copies all recommendation exports (`visitor_recommendations_*.json`, flattened CSV, Neo4j summaries) back into Azure ML outputs and optionally syncs them to the landing datastore for downstream activation teams.
- **Secrets**: reuses the Key Vault bootstrapping logic from Step 1 to avoid manual secret provisioning inside the container.
- **MLflow**: `configure_mlflow()` ensures runs log into the Azure ML experiment when invoked from a hosted job, but leaves Databricks/other URIs untouched when executed locally.

---

## 6. Pipeline Configuration Surfaces
- **`pipeline_config.yaml`** defines the authoritative step list, default compute, and contracts. Adjustments (for example enabling GPU, changing datastore paths, tweaking retry counts) should happen here so notebooks stay thin.
  - `steps.<name>.inputs.*.source` expresses dependencies (`data_preparation.outputs.registration_output`, etc.).
  - `parameters` support run-time toggles such as `incremental`, `model_type`, or `top_k`.
  - `azure_ml.compute` describes the default cluster so DevOps/SRE can align quota and autoscale policies.
- **Event configs (`PA/config/*.yaml`)** still own business logic (segment cut-offs, enrichment fields, recommendation limits). Azure ML does not fork or duplicate those settings.

---

## 7. Notebook Submission & Automation
| Notebook | Typical Use | Notes |
| --- | --- | --- |
| `submit_pipeline_complete_lva.ipynb` | Full four-step run for Vet LVA | Sets `pipeline_config_type="vet_lva"`; reuses the shared `personal_agendas_complete_pipeline` function. |
| `submit_pipeline_complete_bva.ipynb` | Full run for Vet BVA | Mirrors the LVA notebook but points at `config_vet.yaml`. |
| `submit_pipeline_complete_ecomm.ipynb` | Full eCommerce run | Demonstrates switching to `config_ecomm.yaml` and different landing folders. |
| `submit_pipeline_complete_cpcn.ipynb` | CPC North show (proof of concept) | Useful reference when cloning for future shows. |

All notebooks follow the same pattern:
1. Authenticate through `DefaultAzureCredential` and initialize `MLClient`.
2. (Optionally) register or reuse the `pa-env` Azure ML environment defined in `env/pa_env_v12_conda.yaml`.
3. Build the multi-step pipeline by referencing the scripts in `azureml_pipeline/` and wiring parameters/inputs from `pipeline_config.yaml`.
4. Submit the job, watch for completion, and capture run IDs for audit tracking.

For scheduled or CI-triggered runs, replicate the notebook logic inside a thin Python CLI or Azure DevOps/GitHub Actions workflow that imports the same pipeline factory.

---

## 8. Binding Back to `PA/pipeline.py`
- The Azure ML steps import the exact same run functions defined in [PA/pipeline.py](../PA/pipeline.py); there is no duplicated business logic.
- Registration, scan, and session processors in Step 1 still leverage vet-specific overrides (see `run_registration_processing` for BVA/LVA behavior) because the processors detect event names from the YAML config.
- Neo4j steps use the same helper functions to orchestrate Steps 4–8 from the legacy pipeline, ensuring graph schemas stay consistent between local and hosted executions.
- Embeddings and recommendations reuse the shared utilities under `PA/utils/`, so any upgrade (for example changing the embedding model) only needs to be reviewed once.

---

## 9. Operations, Monitoring, and Troubleshooting
- **Tracking & Metrics**: Enable MLflow via `monitoring.enable_mlflow` in `pipeline_config.yaml` to capture run parameters, timing, and counts. Each step also writes log files under `azureml_pipeline/logs/` inside the run container.
- **Data validation**: Inspect the `metadata_output` folders after each step—these contain JSON summaries (counts, missing files, Neo4j statistics) and are the fastest way to confirm success without downloading large artifacts.
- **Incremental strategy**: Pass `pipeline_incremental="true"` from the notebooks to keep `create_only_new=True` on Neo4j and recommendation processors. Combine this with the `has_recommendation` flags in Neo4j to avoid re-issuing recommendations unnecessarily.
- **Common faults**:
  - Missing support files → ensure Step 1 outputs include job/specialization mappings; Step 2 logs call out missing filenames via `stage_step1_outputs`.
  - Neo4j auth errors → confirm Key Vault contains `neo4j-uri`, `neo4j-username`, `neo4j-password` secrets and that the managed identity or service principal has `get` permissions.
  - Datastore permission issues → pipeline notebooks require the `landing_pa` datastore to be accessible to the submitting identity; otherwise Step 1 cannot mount inputs.

---

## 10. Validation Checklist Before Go-Live
1. `pa-env` environment published in Azure ML and referenced by every step.
2. Landing datastore contains the expected folder hierarchy for the target show (`landing/azureml/<event>`).
3. Key Vault secrets validated via `ensure_env_file` inside Step 1 logs.
4. Step outputs show non-zero file counts; metadata summaries align with historical baselines (registrations, sessions, visitor counts).
5. Recommendation CSV includes enrichment fields configured in the event YAML, and the JSON payload is consumable by downstream activation tools.

---

## 11. Next Enhancements (Backlog)
- Register Step 1/4 artifacts as named Azure ML Data assets for better lineage tracking.
- Parameterize Neo4j environment per pipeline schedule so staging and production can run from the same notebook without code edits.
- Optional GPU path for embeddings to shrink run time once GPU quota is granted.
- Extend notebooks or CLI wrapper with notification hooks (Teams/Webhook) on job completion.

---

By consolidating the knowledge captured above, new team members can quickly understand how Azure ML orchestrates the Personal Agendas processors, how each step exchanges data, and where to intervene when onboarding new shows or responding to production incidents.