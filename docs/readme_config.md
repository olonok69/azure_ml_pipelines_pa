# Configuration Playbook for PA Pipeline

This document explains every section of the YAML files under `PA/config`. It maps each setting to the processor that consumes it and highlights subtle differences between the three canonical examples:

- `config/config_cpcn.yaml` → engagement run for Clinical Pharmacy Congress North (CPCN)
- `config/config_ecomm.yaml` → personal agendas for ECOMM with legacy registration exports
- `config/config_vet_lva.yaml` → personal agendas for London Vet Show (LVA) paired with BVA

Use this playbook when onboarding a new event or troubleshooting an existing configuration.

---

## 1. Global Pipeline Controls

| Key | Description | Processors | Example Values |
| --- | --- | --- | --- |
| `mode` | Controls visitor cohort logic. `"personal_agendas"` (default) processes current-year registrations. `"engagement"` reuses last year's attendees and recommends upcoming sessions to them. | RegistrationProcessor, SessionRecommendationProcessor | CPCN uses `engagement`; ECOMM and LVA keep `personal_agendas`. |
| `create_only_new` | When `true`, recommendation step skips visitors who already have recommendations in Neo4j. Leave `false` for full rebuilds. | SessionRecommendationProcessor | All three examples set `false`. |
| `old_format` | Tells registration processor whether the input JSON uses legacy field names. When `true`, the loader expects flattened CamelCase headers. When `false`, it performs GraphQL-style field remapping. | RegistrationProcessor | CPCN & LVA set `false`; ECOMM sets `true` because the transformed JSON mimics legacy exports. |
| `pipeline_steps` | Enables/disables processors 1-10. Useful for partial reruns. Each flag corresponds to a stage exposed by `main.py` (steps 1-10). | pipeline orchestrator | CPCN enables all steps; ECOMM disables nothing; LVA can toggle job/specialization processors independently. |

**Processor map (steps 1-10):**
1. `RegistrationProcessor`
2. `ScanProcessor`
3. `SessionProcessor`
4. `Neo4jVisitorProcessor`
5. `Neo4jSessionProcessor`
6. `Neo4jJobStreamProcessor`
7. `Neo4jSpecializationStreamProcessor`
8. `Neo4jVisitorRelationshipProcessor`
9. `SessionEmbeddingProcessor`
10. `SessionRecommendationProcessor`

`main.py` reads `pipeline_steps` before invoking each runner, so disabling a step in the config is the same as skipping it via `--only-steps` on the CLI. The CLI still wins if you explicitly request a subset (e.g., `--only-steps 4,8`).

---

## 2. Event Metadata (`event` block)

Defines how each processor labels visitors, sessions, and file outputs.

| Field | Usage | CPCN | ECOMM | LVA |
| --- | --- | --- | --- | --- |
| `name` / `main_event` | Base show code; becomes the `show` property for Neo4j nodes. | `"cpcn"` | `"ecomm"` | `"lva"` |
| `main_event_name` | Display-friendly alias used in logging and filenames. For veterinarian shows it also toggles vet-specific enhancements. | `"cpcn"` | `"ecomm"` | `"lva"` (vet) |
| `secondary_event_name` | Paired event (e.g. CPC). Processors still output legacy `bva`/`lva` filenames, but this value appears in stats. | `"cpc"` | `"tfm"` | `"bva"` |
| `year` | Annotates reports and fallback filenames. | `2025` | `2025` | `2025` |
| `shows_this_year` | Show codes to treat as "current" registration in personal agendas. In engagement mode these will be overridden. | `['CPCN25']` | `['ECE25','TFM25']` | `['LVA2025']` |
| `shows_this_year_exclude` | Optional blacklist of codes to drop. | CPCN lists past CPC variants to exclude. | ECOMM excludes `TFM24`. | LVA leaves empty. |
| `shows_last_year_main` / `shows_last_year_secondary` | Upstream codes that identify last year's attendees. | `['CPCN24','CPCN2024']` / `['CPC2025','CPC25']` | `ECE24` / `TFM24` | `BVA2024` / `LVA2024` |

**Vet-specific logic:** when `main_event_name` contains `bva`, `veterinary`, or `vet`, `RegistrationProcessor` imports extra helper functions (`utils.vet_specific_functions`). That is why vet configs keep `main_event_name: "lva"` but the code still injects vet rules.

### Badge and practice metadata

The veterinary configuration also defines auxiliary blocks that control how returning visitors are linked and how practice types are harmonised:

- `badge_history_columns`: tells registration processing which columns hold last-year badge IDs for the paired events. The Neo4j visitor relationship step uses these fields when creating `Same_Visitor` edges.
- `practice_type_columns`: lists survey question keys for current vs. historic shows. The recommendation filters and stream specialisation mappings rely on these names to line up practice-type answers across editions.
- `practice_matching`: drives the fuzzy matching helper that normalises practice/company names. When `missing_companies_output` is set, the registration step writes an exceptions CSV to the configured `output_dir`.
- `job_role_question`: pinpoints the canonical question label the registration processor should treat as the job-role column before handing data to the recommendation engine.

---

## 3. Engagement Overrides (`engagement_mode`)

Only CPCN sets `mode: "engagement"`; this block guides how we replay last year's cohort.

| Field | Processor | Purpose |
| --- | --- | --- |
| `registration_shows.this_year_main` | RegistrationProcessor | List of show codes from the **previous** cycle that should be treated as current-year registrants. |
| `registration_shows.this_year_secondary` | RegistrationProcessor | Same for the paired event; ensures last-year sister show attendees stay linked. |
| `registration_shows.last_year_*` | RegistrationProcessor | Optional look-back to an even older year; usually empty for engagement runs. |
| `drop_last_year_when_missing` | RegistrationProcessor | Omits placeholder rows when there's no matching last-year badge ID. |
| `reset_returning_flags` | RegistrationProcessor | Resets boolean columns like `is_returning` so the pipeline doesn't assume visitors already received recommendations. |

ECOMM and LVA omit this section because they stay in personal agendas mode.

---

## 4. Input File Paths (`input_files`)

Consumed only by the registration processor.

| Key | Expected file | Example |
| --- | --- | --- |
| `main_event_registration` | Combined current+past registration JSON. | `data/cpcn/20251006_eventregistration_CPCN24_25_graphql.json` |
| `secondary_event_registration` | Optional additional show (if running dual events). | `data/ecomm/20250722_registration_TFM24.json` |
| `main_event_demographic` | Survey responses for the same cohort as the registration file. | `data/vet_lva/LVS24_demographics.json` |
| `secondary_event_demographic` | Paired event demographics. | Provided in all three configs. |
| `practices` | (Optional) CSV of unmatched practice names. | CPCN references `practices_missing.csv`. |

`old_format: true` switches the registration loader into legacy header mode; `main.py` passes this straight through to the processor. Use it when the upstream JSON mimics the flattened exports rather than GraphQL payloads.

Make sure the paths live under the event-specific data folder to keep the repo tidy.

---

## 5. Scan Configuration (`scan_files` and `scan_output_files`)

### Inputs (`scan_files`)

Used by `ScanProcessor` to gather raw badge scans and seminar lookups.

- `session_this`: current show scans (CPCN25, ECE25, LVA2025).
- `session_past_main`: main last-year scans.
- `session_past_secondary`: partner event scans.
- `seminars_scan_reference_*` / `seminars_scans_*`: optional lookups for conference sessions vs stands.

### Outputs (`scan_output_files`)

The processor writes cleaned datasets and the visitor relationship processor consumes them. Keep legacy filenames identical to avoid code changes.

```yaml
scan_output_files:
  processed_scans:
    this_year: "scan_this_filtered_valid_cols.csv"
    last_year_main: "scan_last_filtered_valid_cols_bva.csv"
    last_year_secondary: "scan_last_filtered_valid_cols_lva.csv"
  attended_session_inputs:
    past_year_main_scan: "scan_bva_past.csv"
    past_year_secondary_scan: "scan_lva_past.csv"
    past_year_main_filter: "df_reg_demo_last_bva.csv"
    past_year_secondary_filter: "df_reg_demo_last_lva.csv"
```

All three configs reuse this shape. Even though CPCN is not a veterinary show, keeping `bva`/`lva` filenames maintains 100% compatibility with legacy processors.

`scan_output_files.attended_session_inputs` was added to make the relationship builder configurable. The Neo4j visitor relationship processor now looks up these filenames instead of relying on hard-coded defaults when creating `attended_session` and `assisted_session_this_year` edges.

---

## 6. Session Configuration (`session_files` and `session_output_files`)

### Inputs (`session_files`)

| Key | Description | Example |
| --- | --- | --- |
| `session_this` | Current programme (CPCN25, ECE25, LVA2025). | `data/cpcn/CPCN25_session_export.csv` |
| `session_past_bva` | Primary past-year sessions. | CPCN reuses `CPCN24_session_export.csv`; LVA uses `BVA24_session_export.csv`. |
| `session_past_lva` | Secondary show sessions. | CPCN points to `CPC2025_session_export.csv`; vet version points to `LVA24_session_export.csv`. |

### Outputs (`session_output_files`)

`SessionProcessor` generates cleaned CSVs and a `streams.json` file. The Neo4j Session Processor expects fixed names regardless of the event:

```yaml
session_output_files:
  processed_sessions:
    this_year: "session_this_filtered_valid_cols.csv"
    last_year_main: "session_last_filtered_valid_cols_bva.csv"
    last_year_secondary: "session_last_filtered_valid_cols_lva.csv"
  streams_catalog: "streams.json"
```

This naming convention is uniform across CPCN, ECOMM, and VET/LVA.

The `stream_processing` block now includes:

- `create_missing_streams`: when `true`, the session processor auto-generates stream buckets if they are missing from the source export.
- `use_enhanced_streams_catalog`: toggles the extended enrichment routine that produces richer metadata for embeddings and Neo4j upload.

---

## 7. Stream Descriptions and Sponsor Maps

Optional enrichment sections consumed by `SessionProcessor`:

- `stream_processing.use_cached_descriptions`: toggles reuse of `streams_cache.json`. CPCN disables caching (`false`) to force regeneration.
- `titles_to_remove`: event-specific session titles to drop. All three configs include variations like "Session title to be confirmed".
- `map_vets`: sponsor abbreviation replacement. Non-veterinary shows (ECOMM, CPCN) still populate this map for compatibility; keys reflect event-specific vendors.

When `create_missing_streams` is enabled, the processor also consults `titles_to_remove` to ensure auto-generated streams do not resurrect placeholder sessions.

---

## 8. Registration Outputs (`output_files`)

Nested block describing every CSV/JSON the registration processor emits. Even if a downstream consumer only needs a subset, populate the entire structure to satisfy legacy code paths.

Important subsections:

- `professions_list`, `specializations`, `job_roles`: lightweight enrichment artefacts used by dashboards and the Neo4j stream processors. They are generated whenever the registration step runs.
- `raw_data`: raw CSVs exported after initial load.
- `processed_data`: filtered and validated registrants (used by scans).
- `returning_demographic_data`: demographic joins specifically for the past/current visitor overlap cohort.
- `demographic_data` & `registration_with_demographic`: join tables leveraged by analytics.
- `processed_demographic_data`: JSON serialisations consumed by MLflow artifact logging and QA scripts.
- `combined_demographic_registration`: the `df_reg_demo_*` files referenced by scan processing.
- `concatenated_registration_data`: JSON snapshots of current and historic cohorts; useful for diffing in notebooks.
- `recommendations`: controls where final recommendation exports are saved and whether CSV copies are produced. All configs set `save_csv: true` so the processor writes both JSON and CSV.

Example from CPCN:
```yaml
output_files:
  recommendations:
    output_directory: "data/cpcn"
    filename_pattern: "visitor_recommendations_{timestamp}"
    save_csv: true
```

`SessionRecommendationProcessor` takes `output_directory`, assigns `recommendations/<pattern>_{show}.json`, and mirrors to CSV when `save_csv` is true.

`main.py` also reads `output_dir` (top-level key) when logging MLflow artifacts, so make sure this path matches the on-disk location of the files listed above.

---

## 9. Recommendation Logic (`recommendation` block)

Defines behaviour for the collaborative filtering step.

| Field | Usage | Example |
| --- | --- | --- |
| `min_similarity_score` | Numeric cutoff for visitor similarity. | CPCN uses `0.3`; ECOMM `0.5`; LVA `0.3`. |
| `max_recommendations` | Limit per visitor (10 default). | CPCN: `10`; ECOMM: `20`. |
| `similar_visitors_count` | Number of similar visitors to aggregate when generating recommendations. | CPCN & LVA: `3`. |
| `use_langchain` | Enables LLM-based post-filtering (currently off everywhere). |
| `enable_filtering` | Toggles domain-specific filters. LVA sets `true`; CPCN & ECOMM keep `false`. |
| `export_additional_visitor_fields` | Columns to add to exported recommendation CSVs. |
| `similarity_attributes` | Weighted fields used for similarity computations. Vet configs include job role, specialization; CPCN lists pharmacy-specific questions (e.g., `you_are_a`). |
| `field_mappings` & `rules_config` | Additional filtering metadata for vet shows. |

When switching to engagement mode, these parameters remain applicable; the only difference is the visitor cohort being processed.

Additional controls introduced in the vet config:

- `resolve_overlapping_sessions_by_similarity`: ensures only the highest-similarity sessions survive when clashes occur.
- `returning_without_history`: boosts confidence for returning visitors lacking scan history by exponentiating similarity scores.
- `theatre_capacity_limits`: enforces per-session caps. The session recommendation processor loads the supplied capacity CSV and session file to throttle output volume.
- `role_groups`: pre-clustered job-role categories. These feed directly into `rules_config` and allow simpler maintenance of nurse/vet/business logic.
- `field_mappings`: centralises field-name lookups so the filtering code can reference consistent keys regardless of survey wording.
- `rules_config.custom_rules`: flag-controlled micro-rules. The processors read the `apply_to_events` list before applying vet-specific heuristics.

---

## 10. Neo4j Configuration (`neo4j` block)

Shared across processors 4-7.

| Subsection | Key Fields | Description |
| --- | --- | --- |
| Connection | `uri`, `username`, `password`, `show_name` | Direct credentials used by every Neo4j processor. `show_name` becomes a property on created nodes. |
| `node_labels` | Mappings for visitor/session/stream labels. | Keep consistent across configs to avoid schema drift. |
| `unique_identifiers` | Primary key fields for each node type. | Usually `BadgeId`, `session_id`, `stream`. |
| `relationships` | Names for edges (HAS_STREAM, attended_session, etc.). | Must align with existing graph schema. |
| `job_stream_mapping` & `specialization_stream_mapping` | Optional toggles for steps 6 and 7; vet configs enable them, ECOMM disables. |
| `visitor_relationship.same_visitor_properties` | Defines legacy keys (`bva`, `lva`) and metadata attached to `Same_Visitor` edges. CPCN keeps keys as `bva/lva` but sets the `type` field to `cpcn` / `cpc`. |

**Tip:** Do not change relationship names unless you also migrate Neo4j data. A mismatch triggers `UnknownRelationshipType` warnings during recommendations.

New relationship keys such as `assisted_session_this_year` and `registered_to_show` surface in the Neo4j visitor relationship processor. If you remove them from the config, make sure the underlying processor stops referencing those properties.

Both `job_stream_mapping` and `specialization_stream_mapping` now accept an `enabled` flag. `run_neo4j_processing` respects that flag and skips the corresponding step unless both the pipeline flag **and** the mapping-specific flag are `true`.

---

## 11. Default Visitor Properties

`default_visitor_properties` supplies fallback values when registration fields are missing. The registration processor merges these defaults before writing to Neo4j. Customize per event to match tone and email domain (e.g., CPCN uses `Email_domain: "cpcn.co.uk"`).

---

## 12. Logging & Miscellaneous

- `env_file`: path to `.env` file containing API keys. All configs use `keys/.env`.
- `language_model`: model parameters for session stream descriptions. The `deployment` or `model` names differ per tenant (CPCN uses `gpt-4o-mini`, ECOMM uses `gpt-4.1-mini`).
- `embeddings`: controls embedding generation; typically `all-MiniLM-L6-v2`, `batch_size: 100`, `include_stream_descriptions: false`.
- `logging`: sets log level and output file for processors (e.g., `data_processing.log`).
- `output_dir`: root directory used by multiple processors (and MLflow artifact logging) when persisting intermediate CSV/JSON exports.
- `event_date_this_year` / `event_date_last_year`: used by reporting helpers to flag session clashes relative to the show dates.
- `valid_badge_types`: whitelist applied during registration cleaning; only visitors with these badge types feed into downstream steps.
- Top-level `shows_this_year`: convenience alias for processors that do not look inside the nested `event` block. Keep it aligned with `event.shows_this_year`.
- `questions_to_keep`: narrows demographic columns retained for vet analytics; the registration processor uses the `current`/`past` lists when building demographic exports.

---

## 13. Putting It All Together (Implementation Flow)

1. **RegistrationProcessing** (step 1)
   - Reads `mode`, `event.*`, `input_files`, `engagement_mode` and `output_files`.
   - Writes cleaned CSVs to directories defined in `output_files`.

2. **ScanProcessing** (step 2)
   - Uses `scan_files` (inputs) + merged registration/demographic outputs.
   - Writes to `scan_output_files` for downstream steps.

3. **SessionProcessing** (step 3)
   - Reads `session_files`, `stream_processing`, `map_vets`, `titles_to_remove`.
   - Outputs to `session_output_files` and `streams.json`.

4. **Neo4j Steps (4-7)**
   - Share connection info from `neo4j` block and reuse the processed CSVs.
   - Step 8 also references `scan_output_files.attended_session_inputs` to create `attended_session` edges.

5. **Embedding & Recommendation (9-10)**
   - Embedding step reads `embeddings` config and session outputs.
   - Recommendation step uses `recommendation`, `mode`, `output_files.recommendations`, and `similarity_attributes` to generate JSON/CSV exports.

---

## 14. Checklist for New Config Creation

1. Copy the closest example config and rename it.
2. Update `event` block (names, year, show codes).
3. Wire `input_files`, `scan_files`, and `session_files` to new raw datasets.
4. Confirm `session_output_files` and `scan_output_files` stick to legacy filenames.
5. If you need an engagement pass, set `mode: "engagement"` and populate `engagement_mode.registration_shows` with prior-year codes.
6. Verify Neo4j credentials and show name.
7. Adjust `recommendation.similarity_attributes` to the survey questions present in the registration data.
8. Run `python main.py --config config/<new_file>.yaml` and monitor `logs/data_processing.log` for warnings.

Following this guide ensures each processor receives the data it expects and keeps cross-event behaviour consistent.
