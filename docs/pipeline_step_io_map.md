# Personal Agendas Pipeline I/O Map

Mode focus: **personal_agendas**. Inputs/outputs come from the show configs [PA/config/config_vet_lva.yaml](PA/config/config_vet_lva.yaml) (vet/personal agendas) and [PA/config/config_cpcn.yaml](PA/config/config_cpcn.yaml) (pharmacy/post-analysis). Where CPCN currently runs in `post_analysis` mode, toggle `mode` + `pipeline_steps.session_recommendation_processing` before reusing this table.

## Step 1 – Registration Processing (`RegistrationProcessor` in [PA/registration_processor.py](PA/registration_processor.py))
- **Purpose**: Normalize registration + demographic payloads, derive profession lists, create combined CSVs consumed downstream.
- **Inputs**:

  | Config key | LVA path | CPCN path | Notes |
  | --- | --- | --- | --- |
  | `input_files.main_event_registration` | `data/lva/20251127_eventregistration_LVS24_25_legacy.json` | `data/cpcn/20251201_eventregistration_CPCN24_25_graphql.json` | `old_format` true for LVA, false for CPCN. |
  | `input_files.secondary_event_registration` | `data/lva/20250922_registration_BVA_25.json` | `data/cpcn/20251006_eventregistration_CPC25_graphql.json` | Secondary stream for returning visitor linkage. |
  | `input_files.main_event_demographic` | `data/lva/20251127_eventregistration_LVS24_25_demographics_legacy.json` | `data/cpcn/20251201_CPCN24_25_demographics_graphql.json` | Used to build `_with_demographic` artifacts. |
  | `input_files.secondary_event_demographic` | `data/lva/20250922_demographics_BVA_25.json` | `data/cpcn/20251006_eventregistration_CPC25_demographics_graphql.json` | |
  | `input_files.practices` | `data/lva/practices_missing.csv` | `data/cpcn/practices_missing.csv` | Drives `practice_matching` reconciliation. |
  | `map_vets`, `titles_to_remove`, `practice_matching` | Vet-specific dictionaries | Generic minimal mappings | Only LVA uses the rich veterinary dictionaries. |

- **Outputs** (all written under `output_dir/output` using names in `output_files`):

  | Config key | LVA filename | CPCN filename | Notes |
  | --- | --- | --- | --- |
  | `output_files.raw_data.*` | `Registration_data_bva.csv`, etc. | Same defaults | Raw extracts for auditing. |
  | `output_files.processed_data.*` | `Registration_data_bva_25_only_valid.csv`, etc. | Same defaults | Clean registration per cohort. |
  | `output_files.demographic_data.*` | `Registration_demographicdata_bva_25_raw.csv`, etc. | Same defaults | Clean demographics. |
  | `output_files.registration_with_demographic.*` | `Registration_data_with_demographicdata_bva_this.csv`, etc. | Same defaults | Master files used in Neo4j steps. |
  | `output_files.combined_demographic_registration.*` | `df_reg_demo_this.csv`, `df_reg_demo_last_bva.csv`, `df_reg_demo_last_lva.csv` | Same defaults | Feed scan + relationship steps. |
  | `output_files.recommendations.*` | Folder `recommendations/visitor_recommendations_{timestamp}` | Same defaults | Path template consumed by Step 10. |

- **Conditions**: `mode` drives engagement/post-analysis overrides; `create_only_new` is `false` for LVA (full rebuild) and `true` for CPCN (only new visitors). Vet-only enrichments (`map_vets`, control groups) activate when `event.main_event_name` is `lva/bva`.

## Step 2 – Scan Processing (`ScanProcessor` in [PA/scan_processor.py](PA/scan_processor.py))
- **Purpose**: Clean seminar scans, align them with registration cohorts, and persist attendance CSVs.
- **Inputs**:

  | Config key | LVA path | CPCN path | Notes |
  | --- | --- | --- | --- |
  | `scan_files.session_this` | `data/lva/LVS25_session_export.csv` | `data/cpcn/CPCN25_session_export.csv` | Base schedule for current year. |
  | `scan_files.session_past_main` | `data/lva/LVS24_session_export.csv` | `data/cpcn/CPCN24_session_export.csv` | Past-year show for Same_Visitor linking. |
  | `scan_files.session_past_secondary` | `data/lva/BVA25_session_export.csv` | `data/cpcn/CPC2025_session_export.csv` | Secondary show. |
  | `scan_files.seminars_*` | Vet files under `data/lva/...` | Pharmacy files under `data/cpcn/...` | Provide badge-level attendance. |
  | `output_files.combined_demographic_registration.*` | Step 1 outputs | Step 1 outputs | Required to enrich scan exports with visitor metadata. |
  | `post_analysis_mode.*` | Not used (mode=personal_agendas) | Provides `scan_files` + `entry_scan_files` for CPCN when mode switched. |

- **Outputs** (under `output_dir/output` per `scan_output_files`):

  | Config key | LVA filename | CPCN filename | Notes |
  | --- | --- | --- | --- |
  | `scan_output_files.processed_scans.this_year` | `scan_this_filtered_valid_cols.csv` | Same defaults | Filtered this-year scans. |
  | `scan_output_files.processed_scans.last_year_main` | `scan_last_filtered_valid_cols_bva.csv` | Same defaults | Past-year scans for main show. |
  | `scan_output_files.sessions_visited.*` | `sessions_visited_last_bva.csv`, etc. | Same defaults | Reference for Step 8. |
  | `scan_output_files.attended_session_inputs.*` | `scan_bva_past.csv`, etc. | Same defaults | Hard-coded filenames consumed in relationship builder. |
  | `scan_output_files.processed_scans.this_year_post` | `scan_this_year_post.csv` | Only populated when `mode=post_analysis`. |

- **Conditions**: `mode=post_analysis` enables additional “this-year” seminar merges plus entry scan outputs. Works for both vet + pharmacy; file list switches automatically via config.

## Step 3 – Session Processing (`SessionProcessor` in [PA/session_processor.py](PA/session_processor.py))
- **Purpose**: Clean and deduplicate session catalogs, optionally backfill streams via LLM.
- **Inputs**:

  | Config key | LVA value | CPCN value | Notes |
  | --- | --- | --- | --- |
  | `session_files.session_this` | `data/lva/LVS25_session_export.csv` | `data/cpcn/CPCN25_session_export.csv` | Same file as scans for schedule context. |
  | `session_files.session_past_bva` | `data/lva/LVS24_session_export.csv` | `data/cpcn/CPCN24_session_export.csv` | Past-year main. |
  | `session_files.session_past_lva` | `data/lva/BVA25_session_export.csv` | `data/cpcn/CPC2025_session_export.csv` | Past-year secondary. |
  | `titles_to_remove`, `map_vets` | Vet-only curation lists | Minimal set for CPCN | Controls filtering and sponsor normalization. |
  | `stream_processing.*` | `use_cached_descriptions=false`, `create_missing_streams=true`, etc. | Same | Tells processor whether to call LLM/backfill. |
  | `env_file` | `keys/.env` | `keys/.env` | Supplies OpenAI/Azure OpenAI credentials for synopsis filling. |

- **Outputs**:

  | Config key | LVA filename | CPCN filename | Notes |
  | --- | --- | --- | --- |
  | `session_output_files.processed_sessions.this_year` | `session_this_filtered_valid_cols.csv` | Same defaults | Source for Neo4j sessions. |
  | `session_output_files.processed_sessions.last_year_main/secondary` | `session_last_filtered_valid_cols_bva.csv`, etc. | Same defaults | Past-year catalogs. |
  | `session_output_files.streams_catalog` | `streams.json` | Same defaults | Backfilled stream metadata. |
  | Streams cache | `output/streams_cache.json` | Same | Only written when `use_cached_descriptions=true`. |

- **Conditions**: Vet-specific abbreviations only applied when `map_vets` populated (LVA). LLM enrichment requires valid credentials; disable by toggling `stream_processing.create_missing_streams`.

## Step 4 – Neo4j Visitor Processing (`Neo4jVisitorProcessor` in [PA/neo4j_visitor_processor.py](PA/neo4j_visitor_processor.py))
- **Purpose**: Create `Visitor_this_year` and historical visitor nodes from Step 1 CSVs.
- **Inputs**:

  | Artifact | Source | Notes |
  | --- | --- | --- |
  | `output/Registration_data_with_demographicdata_bva_this.csv` (and `..._last_*.csv`) | `output_files.registration_with_demographic.*` | Primary CSVs merged with demographics. |
  | `output/df_reg_demo_*.csv` | `output_files.combined_demographic_registration.*` | Used for additional relationships/filters. |
  | Neo4j config | `neo4j.node_labels`, `neo4j.unique_identifiers.visitor` | Controls labels + key field (`BadgeId`). |
  | Credentials | `env_file` + `neo4j.environment` | Determine URI + authentication. |

- **Outputs**: Nodes in Neo4j (`Visitor_this_year`, `Visitor_last_year_bva`, `Visitor_last_year_lva`) with `show=<show_name>`. Counts logged; no new files.
- **Conditions**: Respects `pipeline_steps.neo4j_visitor_processing`; `create_only_new` toggles MERGE vs update (false for LVA, true for CPCN). Show-specific behaviors derive from `neo4j.show_name`.

## Step 5 – Neo4j Session Processing (`Neo4jSessionProcessor` in [PA/neo4j_session_processor.py](PA/neo4j_session_processor.py))
- **Purpose**: Create session + stream nodes from Step 3 exports.
- **Inputs**:

  | Config key | Files | Notes |
  | --- | --- | --- |
  | `session_output_files.processed_sessions.*` | `output/session_this_filtered_valid_cols.csv`, etc. | CSVs loaded into Neo4j. |
  | `session_output_files.streams_catalog` | `output/streams.json` | Used to seed `Stream` nodes. |
  | Neo4j config | `node_labels.session_this_year`, `relationships.session_stream`, `show_name` | Controls labels and HAS_STREAM relationship names. |

- **Outputs**: Neo4j nodes (`Sessions_this_year`, `Sessions_past_year`, `Stream`) and `HAS_STREAM` edges. No filesystem output beyond logs.
- **Conditions**: Requires Step 3 outputs. `create_only_new` defaults to config-level flag; ensures past sessions are not recreated when `create_only_new=true` (CPCN).

## Step 6 – Neo4j Job→Stream Mapping (`Neo4jJobStreamProcessor` in [PA/neo4j_job_stream_processor.py](PA/neo4j_job_stream_processor.py))
- **Purpose**: Connect visitor nodes to streams using job-role mappings.
- **Inputs**:

  | Config key | LVA value | CPCN value | Notes |
  | --- | --- | --- | --- |
  | `neo4j.job_stream_mapping.enabled` | `true` | `false` | CPCN skips this step unless enabled. |
  | `neo4j.job_stream_mapping.file` | `job_to_stream.csv` (looked up under `output_dir`, repo root, or `data/bva`) | Same path | Mapping table. |
  | Visitor data | `output/Registration_data_with_demographicdata_bva_this.csv` | Same default name | Provides `job_role` column referenced in relationships. |
  | Neo4j config | `node_labels.visitor_this_year`, `relationships.job_stream` | Controls labels + relationship name. |

- **Outputs**: `job_to_stream` (or custom name) relationships between visitors and stream nodes. Stats logged.
- **Conditions**: Entire step skipped when `job_stream_mapping.enabled=false` or when mapping CSV missing. Vet-only adjustments happen via mapping content.

## Step 7 – Neo4j Specialization→Stream Mapping (`Neo4jSpecializationStreamProcessor` in [PA/neo4j_specialization_stream_processor.py](PA/neo4j_specialization_stream_processor.py))
- **Purpose**: Link visitors’ specialization answers to streams.
- **Inputs**:

  | Config key | LVA value | CPCN value | Notes |
  | --- | --- | --- | --- |
  | `neo4j.specialization_stream_mapping.enabled` | `true` | `false` | CPCN skips by default. |
  | `neo4j.specialization_stream_mapping.file` | `spezialization_to_stream.csv` | Same | Mapping file (searched under `output_dir/output`, repo root, etc.). |
  | `specialization_field_*` keys | Vet config maps BVA/LVA question names | CPCN uses generic question strings | Determines which CSV columns to read. |
  | `specialization_map_*` overrides | Extensive vet mappings for BVA vs LVA | Mostly empty | Controls normalization before matching to streams. |

- **Outputs**: `specialization_to_stream` relationships. Statistics persisted in logs.
- **Conditions**: Step auto-skips if mapping disabled or file missing. Vet-specific per-cohort mapping ensures only relevant shows use the data.

## Step 8 – Neo4j Visitor Relationships (`Neo4jVisitorRelationshipProcessor` in [PA/neo4j_visitor_relationship_processor.py](PA/neo4j_visitor_relationship_processor.py))
- **Purpose**: Build Same_Visitor, attended_session, and (post-analysis) assisted relationships.
- **Inputs**:

  | Source | Key Config | Notes |
  | --- | --- | --- |
  | Combined registration outputs | `output_files.combined_demographic_registration.*` | Provide last-year vs this-year visitor cohorts. |
  | Scan enrichments | `scan_output_files.sessions_visited.*`, `scan_output_files.attended_session_inputs.*` | Supply attendance for linking visitors to sessions. |
  | Post-analysis extras (CPCN) | `post_analysis_mode.entry_scan_files.*` | Enables `registered_to_show` + `assisted_session_this_year` relationships when `mode=post_analysis`. |
  | Relationship config | `neo4j.visitor_relationship.same_visitor_properties`, `relationships.attended_session`, etc. | Controls labels + metadata applied on edges. |

- **Outputs**: Neo4j relationships (`Same_Visitor`, `attended_session`, optionally `assisted_session_this_year`, `registered_to_show`). Optionally show nodes in post-analysis mode.
- **Conditions**: Requires Steps 1–2 outputs in `output/`. Additional entry-scan edges only appear when `mode=post_analysis` (CPCN default). Honors `create_only_new` when re-running.

## Step 9 – Session Embedding Processing (`SessionEmbeddingProcessor` in [PA/session_embedding_processor.py](PA/session_embedding_processor.py))
- **Purpose**: Compute sentence-transformer embeddings for session nodes directly in Neo4j.
- **Inputs**:

  | Config key | LVA value | CPCN value | Notes |
  | --- | --- | --- | --- |
  | `neo4j.show_name` | `lva` | `cpcn` | Filters session nodes by show. |
  | `embeddings.model` | `all-MiniLM-L6-v2` | Same | HuggingFace model pulled at runtime. |
  | `embeddings.batch_size` | `100` | Same | Controls chunking when querying Neo4j. |
  | `embeddings.include_stream_descriptions` | `false` | `false` | If set true, Step 3’s stream catalog is pulled for extra text. |
  | Credentials | `.env` + `neo4j` section | Provide DB access. |

- **Outputs**: Embedding vectors stored on `Sessions_*` nodes (`s.embedding` property). Stats only logged to console. No filesystem artifacts.
- **Conditions**: Requires Steps 5 outputs already ingested into Neo4j. Obeys `pipeline_steps.session_embedding_processing` flag; CPCN currently keeps this `true` even though recommendations are off.

## Step 10 – Session Recommendation Processing (`SessionRecommendationProcessor` in [PA/session_recommendation_processor.py](PA/session_recommendation_processor.py))
- **Purpose**: Generate personalized agendas + optional control-group exports.
- **Inputs**:

  | Config key | LVA value | CPCN value | Notes |
  | --- | --- | --- | --- |
  | `pipeline_steps.session_recommendation_processing` | `true` | `false` | CPCN disables this step unless personal_agendas mode is required. |
  | `recommendation.*` | Vet config enables filtering, control group, theatre capacities | CPCN disables filtering/control group, sets capacity limits to `false` | Drives similarity weights, filtering, control cohorts. |
  | `recommendation.theatre_capacity_limits.capacity_file` | `data/lva/teatres.csv` | `false` / not set | Supplies per-theatre seat counts; must exist whenever capacity enforcement is enabled, regardless of filtering toggles. |
  | `recommendation.theatre_capacity_limits.session_file` | `data/lva/LVS25_session_export.csv` | `false` / not set | Reuses the current-year session export to map theatre/date slots for capacity trimming. |
  | `recommendation.theatre_capacity_limits.capacity_multiplier` | `3.0` | `1.0` | Multiplier applied to each theatre’s capacity before pruning overlapping attendees. |
  | `output_files.recommendations.*` | `recommendations/visitor_recommendations_{timestamp}` | Same default | Target folder for JSON/CSV outputs. |
  | `output_files.registration_with_demographic.this_year` | `output/Registration_data_with_demographicdata_bva_this.csv` | Same default | Used to enrich recommendation exports with visitor contact fields. |
  | Neo4j data | Sessions + visitors with embeddings + relationships | Same | Processor queries Neo4j for all show-specific nodes. |
  | Embeddings | Stored on session nodes by Step 9 | Same | Required for cosine similarity scoring. |

- **Outputs**: Files under `output/recommendations/` (JSON plus optional CSV when `save_csv=true`). If control group enabled, duplicates land under `recommendations/control/` with suffix `_control`.
- **Conditions**: Honors `create_only_new` (only write missing visitor recs when true). Vet-specific behavior triggered by `recommendation.enable_filtering` and custom rules; CPCN leaves these off. Theatre capacity enforcement checks only `recommendation.theatre_capacity_limits.enabled`, so its CSV inputs are required even when filtering is disabled.

---
**Usage notes**
- Steps 4–10 rely on artifacts from Steps 1–3 located under each show’s `output_dir`. Ensure the folders named in the tables exist in the Azure ML datastore mount (`azureml_pipeline/data/<event>/...`).
- To run the pharmacy show (`config_cpcn.yaml`) in `personal_agendas` mode, set `mode: "personal_agendas"`, enable `session_recommendation_processing`, and consider disabling post-analysis-only outputs (entry scans, assisted relationships) if not required. Otherwise, Steps 9–10 will be skipped by design.
