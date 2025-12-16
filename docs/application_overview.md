# Personal Agendas Platform – Comprehensive Guide

> This document as the single source of truth to explain the application Personal Agendas. Each section maps to the operating procedures, configuration, and cloud orchestration that power the Personal Agendas (PA) platform._

---

## 1. What the Application Does

The Personal Agendas application ingests multi-show registration, session, and scan data to build a knowledge graph in Neo4j and deliver personalized session recommendations. The same processors can be executed in three operational modes (engagement, personal agendas, post-analysis) to support the entire event lifecycle:

- **Before the show** – curate audiences, refresh visitor profiles, and generate agenda recommendations aligned to current-year sessions.
- **During the show** – keep the Neo4j graph synchronized with newly published sessions and incremental registration updates.
- **After the show** – replay badge scans, capture factual attendance (`assisted_session_this_year`), and benchmark algorithm performance.

The system is configuration-driven, enabling any conference (B2B, veterinary, e-commerce, CPCN, etc.) to opt in by dropping the right YAML file and raw exports.

---

## 2. Technologies in Use

| Layer | Technologies | Notes |
| --- | --- | --- |
| Programming & Tooling | Python 3.10+, Poetry/`requirements.txt`, Jupyter Notebooks | Main package lives in `PA/`; notebooks under `bva/`, `cpc/`, `notebooks/` cover submissions & analyses. |
| Data Processing | Pandas, NumPy, custom processors (registration, scan, session) | Harmonize JSON/CSV inputs and normalize show-specific schemas. |
| Graph Database | Neo4j 4.4+ via official Python driver | Stores visitors, sessions, streams, and all recommendation/attendance relationships. |
| ML/AI | `sentence-transformers` (default: `all-MiniLM-L6-v2`), optional GPT/Claude for stream descriptions | Generates embeddings and natural-language enrichments. |
| Observability | MLflow tracking, structured logging utilities (`utils/logging_utils.py`) | Parameters, metrics, and artifacts are logged per run (local + Azure ML). |
| Cloud Orchestration | Azure Machine Learning Pipelines, Azure Key Vault, Azure Storage | Executes the identical 4-step cloud pipeline (`azureml_pipeline/`). |
| Control & Reporting | Deep Agents workflow (`deep_agents_reports/`), custom Markdown reports | Automates pre/post-show intelligence packs using the same Neo4j graph. |

---

## 3. Data Processing Pipeline

![Generic Data Pipeline](../docs/generic_pipeline_map.svg)

| Step | Processor | Purpose | Key Outputs |
| --- | --- | --- | --- |
| 1 | `RegistrationProcessor` | Ingest registration + demographics files, align survey questions, detect returning visitors. | `df_reg_demo_this.csv`, `Registration_data_*`, job/specialization dictionaries. |
| 2 | `ScanProcessor` | Normalize session scans across current/past shows, join with visitor demographics. | `scan_*_filtered_valid_cols.csv`, `sessions_visited_*`, `scan_this_year_post.csv` (post-analysis). |
| 3 | `SessionProcessor` | Clean programme exports, remove placeholder sessions, enrich streams. | `session_this_filtered_valid_cols.csv`, `streams.json`. |
| 4 | `Neo4jVisitorProcessor` | Create/update visitor nodes (mode-aware create-only safeguards). | `Visitor_this_year`, `Visitor_last_year_*` nodes. |
| 5 | `Neo4jSessionProcessor` | Merge sessions + streams, maintain HAS_STREAM edges. | `Sessions_this_year` & `Sessions_past_year` nodes, HAS_STREAM relationships. |
| 6 | `Neo4jJobStreamProcessor` | Map job roles to relevant content tracks. | `job_to_stream` edges. |
| 7 | `Neo4jSpecializationStreamProcessor` | Link practice specializations to streams per visitor cohort. | `specialization_to_stream` edges. |
| 8 | `Neo4jVisitorRelationshipProcessor` | Cross-year linking, attendance wiring, entry-scan shows. | `Same_Visitor`, `attended_session`, `assisted_session_this_year`, `registered_to_show`. |
| 9 | `SessionEmbeddingProcessor` | Encode session text + (optional) stream descriptions for semantic similarity. | 384-dim embeddings stored on `Sessions_this_year`. |
| 10 | `SessionRecommendationProcessor` | Generate and persist personalized agendas + control groups. | Recommendation JSON/CSV, `IS_RECOMMENDED` edges, `has_recommendation` flag.

_All steps are callable individually through CLI flags (`--only-steps`) or the Azure ML pipeline, ensuring reproducibility whether you run locally or in the cloud._

---

## 4. Configuration Model

The entire platform is config-driven. YAML files (e.g., `config/config_cpcn.yaml`, `config/config_ecomm.yaml`, `config/config_vet_lva.yaml`) dictate:

1. **Event metadata** – show codes, paired events, legacy filenames, and survey mappings.
2. **Mode switches** – `mode: personal_agendas | engagement | post_analysis`, plus mode-specific blocks.
3. **Input/Output wiring** – registration, scan, session file paths and their normalized output filenames.
4. **Neo4j schema knobs** – node labels, relationship names, unique identifiers, and optional processors (job/specialization mapping).
5. **Recommendation behaviour** – similarity weights, control groups, theatre capacity limits, LangChain filters.
6. **Runtime controls** – `pipeline_steps` toggles, `create_only_new`, logging verbosity, ML/AI settings.

For a new show:

```bash
cp config/config_vet_lva.yaml config/config_<show>.yaml
# edit event, input_files, recommendation, neo4j blocks
python PA/main.py --config config/config_<show>.yaml
```

[`readme_config.pdf`](readme_config.pdf) contains per-field descriptions; refer to it whenever you introduce new survey questions, badge types, or Azure ML parameters.

---

## 5. Neo4j Database Schema

![Neo4j Schema](../docs/neo4j_database_map.svg)

### Core Node Types

- `Visitor_this_year`, `Visitor_last_year_bva`, `Visitor_last_year_lva` – visitor cohorts with badge IDs, show codes, survey answers, control flags.
- `Sessions_this_year`, `Sessions_past_year` – session catalogues enriched with streams, embeddings, sponsor data.
- `Stream` – curated specialization/interest taxonomies shared across events.
- `Show_this_year` – dynamically generated nodes for entry-scans in post-analysis mode.

### Relationship Highlights

- `HAS_STREAM`, `job_to_stream`, `specialization_to_stream` – taxonomy edges driving personalization.
- `IS_RECOMMENDED`, `has_recommendation` property on visitors – algorithm outputs.
- `attended_session`, `assisted_session_this_year` – historic vs. live attendance comparisons.
- `registered_to_show` – presence tracking from gate scans.
- `Same_Visitor` – cross-year identity stitching powering engagement/post-analysis reporting.

Processors respect `create_only_new` semantics to avoid overwriting historical graphs, especially in post-analysis where we only append facts.

---

## 6. Operational Modes

| Mode | When to Use | Behavioural Changes | Key References |
| --- | --- | --- | --- |
| **Personal Agendas** | Default pre-show pipeline. | Visitors come from current-year registration, all steps run end-to-end. | `readme.md` (Overview). |
| **Engagement** | Re-engage prior-year attendees with upcoming sessions before new registrations arrive. | `Visitor_this_year` draws from last year's show codes; scans/relationships replay historic attendance; filenames remain unchanged for downstream systems. | [`engagement_mode.pdf`](engagement_mode.pdf) |
| **Post-Analysis** | After-show replay to compare predictions vs reality. | Adds raw session exports + live scan files; steps 4 & 5 only insert missing nodes; step 8 creates `assisted_session_this_year` + `registered_to_show`; embeddings/recs optional. | [`post_analysis_mode.pdf`](post_analysis_mode.pdf) |

Switch by editing the `mode` field, then provide the extra blocks required by each mode (`engagement_mode`, `post_analysis_mode`).

---

## 7. How the Application Works (Execution Paths)

1. **Local / On-Prem Runs**
   - Execute `python PA/main.py --config config/config_<show>.yaml` from repo root.
   - Use `--only-steps` to limit execution, `--create-only-new` to preserve existing Neo4j nodes, and CLI overrides for experiment toggles.
   - Logs land under `logs/`, MLflow can be activated via `mlflow ui`.

2. **Notebook-driven Operations**
   - `bva/`, `cpc/`, `notebooks/` host exploratory Jupyter notebooks for EDA, validation, and Azure ML submissions.
   - Notebook outputs align with the same file/folder layout used by the CLI.

3. **Deep Agents Reporting**
   - `deep_agents_reports/deep_agents_workflow.py` automates 13-task pre-show or post-show analysis packs.
   - Workflow prebuilds TODO manifests, schema captures, and recommended queries; agents consume repo-relative paths for deterministic execution.

4. **Artifacts & Restorations**
   - All processors write structured CSV/JSON under `data/<show>/output/` and `data/<show>/recommendations/`.
   - `scripts/restore_recommendations_from_export.py` can rehydrate Neo4j from stored recommendation files, respecting control-group metadata where available.

---

## 8. Azure ML Pipeline

The cloud implementation in `azureml_pipeline/` mirrors the ten-step local flow but packages them into four Azure ML steps:

| Azure ML Step | Script | Maps to Local Steps | Responsibilities |
| --- | --- | --- | --- |
| `step1` | `azureml_step1_data_prep.py` | 1–3 | Data ingestion, normalization, writing standardized outputs. |
| `step2` | `azureml_step2_neo4j_prep.py` | 4–8 | Neo4j node/relationship updates, incremental safeguards. |
| `step3` | `azureml_step3_session_embedding.py` | 9 | Session embeddings (CPU/GPU). |
| `step4` | `azureml_step4_recommendations.py` | 10 | Recommendation engine, exports, Neo4j updates. |

Key artifacts:
- `pipeline_config.yaml` – declarative pipeline definition (compute, IO bindings, schedules, retries, MLflow toggles).
- Submission notebooks (`notebooks/submit_pipeline*.ipynb`) – authenticate, register data/environments, and launch runs.
- Azure Key Vault integration for secrets (`NEO4J_URI`, etc.).
- Incremental flags propagate from pipeline parameters to processors (`incremental=True` → `create_only_new=True`).

**Operational checklist:**
1. Update `EVENT_TYPE` parameters and data asset paths.
2. Submit via notebook/CLI; monitor run lineage in Azure ML Studio.
3. Validate outputs/metrics per step (registration counts, Neo4j insertions, embedding totals, recommendation coverage).
4. Trigger downstream activation (email/CRM, PDF exports) once artifacts land in the configured data lake containers.

---

## 9. Appendices & References

- [`streams_readme.pdf`](streams_readme.pdf) – stream catalog governance.
- [`control_group_personal_agendas.pdf`](control_group_personal_agendas.pdf) – control-group toggles and Neo4j impacts.
- [`README_transform_cpc_data.pdf`](README_transform_cpc_data.pdf), [`README_transform_lva_data.pdf`](README_transform_lva_data.pdf) – upstream ETL notes for raw exports.
- [`README_theatre_capacity_limits.pdf`](README_theatre_capacity_limits.pdf) – seat-cap enforcement for recommendation throttling.
- [`Implementation Guide Azure ML Pipeline for Personal Agendas.pdf`](Implementation%20Guide%20Azure%20ML%20Pipeline%20for%20Personal%20Agendas.pdf) – extended cloud deployment walkthrough.
- [`post_analysis_mode.pdf`](post_analysis_mode.pdf), [`engagement_mode.pdf`](engagement_mode.pdf) – mode-specific operations.
- `docs/neo4j_vet_database_map.svg`, `docs/generic_pipeline_map.svg` – diagrams included above in the document.

With this guide you can describe, operate, and audit every facet of the PA platform when presenting to stakeholders or migrating to new shows.
