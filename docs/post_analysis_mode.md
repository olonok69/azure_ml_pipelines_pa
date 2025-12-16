# Post-Analysis Mode

Post-analysis mode lets you rerun the personal agenda pipeline **after the show** to measure how well the recommendation engine performed. It reuses the full processor stack, adds new data sources for actual badge scans, and writes fresh Neo4j relationships so you can compare delivered recommendations with the sessions visitors really attended.

---

## 📌 How post-analysis differs from personal agendas

| Area | Personal agendas (default) | Post-analysis mode |
| --- | --- | --- |
| Config flag | `mode: "personal_agendas"` | `mode: "post_analysis"` |
| Session source | Processed CSV from step 3 | Direct raw session export (`scan_files.session_this`), with processed fallback |
| Scan inputs | Past-year seminar scans only | Adds this-year seminar scans (`post_analysis_mode.scan_files.*`) |
| Visitor upload | Creates visitors from current-year registration CSV | Only creates missing `Visitor_this_year` badges; existing nodes are left untouched |
| Neo4j relationships | `attended_session` (past-year) only | Adds `assisted_session_this_year` between `Visitor_this_year` and `Sessions_this_year` **and** `registered_to_show` edges to `Show_this_year` nodes sourced from entry scans; leaves other edges untouched |
| Extra outputs | None | `scan_this_year_post.csv`, `sessions_visited_this_year.csv` with live attendance |
| MLflow metrics | Nodes/relationships created by default steps | Adds visitors updated + assisted-this-year relationship counts |

Although you usually care about a subset of the pipeline (steps 1–3 and Neo4j updates), the **entire 10-step flow remains available**. You can restrict execution via the CLI `--only-steps` flag if you only need the post-show pass.

---

## 🧭 End-to-end flow in post-analysis mode

1. **Registration processing**
   - Same inputs as personal agendas: multi-year registration & demographic JSON for main and paired shows.
   - Outputs keep legacy filenames (e.g., `Registration_data_with_demographicdata_bva_this.csv`). Post-analysis consumes these to anchor the visitor nodes.

2. **Scan processing**
   - Loads the usual past-year seminar data.
   - Additionally reads the two post-show files configured under `post_analysis_mode.scan_files` (reference + raw scans for this year).
   - Produces four key outputs in `data/<event>/output/`:
     - `scan_bva_past.csv` / `scan_lva_past.csv` *(unchanged)*
     - `scan_this_year_post.csv` *(new)*
     - `sessions_visited_this_year.csv` *(new; enriched with demographics for this-year scans)*
   - Logs schema comparisons to ensure the new attendance export matches the personal-agenda structure.

3. **Session processing**
   - Still generates filtered CSVs for this-year and past-year sessions.
   - Post-analysis mode tells `Neo4jSessionProcessor` to source current sessions directly from `scan_files.session_this`. If the raw export is missing, it falls back to the processed CSV.

4. **Neo4j visitor upload (step 4)**
   - Only inserts `Visitor_this_year` nodes that aren't already in Neo4j, regardless of `create_only_new`.
   - Existing visitor nodes are preserved exactly as-is so the graph remains a faithful snapshot of show-day data.

5. **Neo4j session upload (step 5)**
   - Sessions from the raw export are merged; existing sessions keep their edges but receive refreshed field values.
   - New `Sessions_this_year` nodes created in this pass are automatically linked back to visitors who scanned them.
   - Stream nodes/relationships are left untouched in post-analysis to avoid rippling changes.

6. **Neo4j visitor relationships (step 8)**
   - Existing logic recreates `Same_Visitor` and `attended_session` links for past years (with zero-count placeholders if skipped).
   - `_create_assisted_session_this_year_relationships` reads `sessions_visited_this_year.csv` and creates `assisted_session_this_year` edges from `Visitor_this_year` to `Sessions_this_year`, storing scan metadata (date/file/seminar name).
   - `_create_entry_scan_show_relationships` loads entry scan CSVs, merges `Show_this_year` nodes, and links visitors to those shows via `registered_to_show` edges with scan timestamps.

7. **Embeddings & recommendations (steps 9–10)**
   - Optional in post-analysis. You can skip them if you just need the attendance view; otherwise they operate as in other modes.

---

## 📂 New artifacts & metrics

| Artifact/Metrics | Location | Purpose |
| --- | --- | --- |
| `scan_this_year_post.csv` | `data/<event>/output/` | Raw this-year scan rows joined with session metadata |
| `sessions_visited_this_year.csv` | `data/<event>/output/` | Scan data enriched with visitor demographics (mirrors personal-agenda outputs) |
| `assisted_session_this_year` relationship | Neo4j | Connects visitors to sessions they actually attended in the current show |
| `Show_this_year` + `registered_to_show` | Neo4j | Captures entry scan show presence with scan metadata |
| MLflow visitor inserts metric | MLflow run metrics (step 4) | Count of new visitor nodes inserted (existing ones remain untouched) |
| MLflow session refresh metric | MLflow run metrics (step 5) | Number of existing sessions refreshed with post-show data |
| MLflow assisted relationship metric | MLflow run metrics (step 8) | Number of attendance edges written |

Use these to compare recommendations vs reality and to drive downstream analytics.

---

## 🛠️ Configuration checklist

Minimum additions to your event config (example from `config/config_ecomm.yaml`):

```yaml
mode: "post_analysis"

post_analysis_mode:
  registration_shows:
    this_year_main:
      - ["ECE25", "TFM25"]   # Shows to keep in the this-year cohort
  scan_files:
    seminars_scan_reference_this: "data/ecomm/post/ecetfm2025 seminar scans reference.csv"
    seminars_scans_this: "data/ecomm/post/ecetfm2025 seminar scans.csv"
   entry_scan_files:
      entry_scans_this:
         - "data/ecomm/post/ece25_unfiltered_entry_scans.csv"
         - "data/ecomm/post/tfm25_unfiltered_entry_scans.csv"

scan_files:
  session_this: "data/ecomm/ECE_TFM_25_session_export.csv"
  session_past_main: "data/ecomm/ECE24_session_export.csv"
  session_past_secondary: "data/ecomm/TFM24_session_export.csv"
  # ... past seminar files omitted for brevity ...

scan_output_files:
  processed_scans:
    this_year: "scan_this_filtered_valid_cols.csv"
    last_year_main: "scan_last_filtered_valid_cols_bva.csv"
    last_year_secondary: "scan_last_filtered_valid_cols_lva.csv"
    this_year_post: "scan_this_year_post.csv"
  sessions_visited:
    this_year: "sessions_visited_last_bva.csv"
    last_year_main: "sessions_visited_last_bva.csv"
    last_year_secondary: "sessions_visited_last_lva.csv"
    this_year_post: "sessions_visited_this_year.csv"
```

Keep the legacy filenames for backwards compatibility—only the new post-analysis ones should be added. All other processor settings (filters, titles_to_remove, etc.) remain unchanged.

---

## ▶️ Running a post-analysis pass

1. Update the event config with the checklist above.
2. Ensure the new scan files and raw session export are present at the configured paths.
3. (Optional) Archive or clear `data/<event>/output/` so the post-analysis outputs are easy to identify.
4. Run the pipeline with the subset of steps you need:

    - **Full post-analysis including data prep and Neo4j graph (no new recommendations):**

       ```powershell
       python .\PA\main.py --config PA\config\config_ecomm.yaml --only-steps 1,2,3,4,5,8
       ```

    - **Only push existing CSVs into Neo4j (you already ran steps 1–3 and outputs haven't changed):**

       ```powershell
       python .\PA\main.py --config PA\config\config_ecomm.yaml --only-steps 4,5,8
       ```

    - **Include embeddings and recommendations as well (full 1–10 flow in post-analysis):**

       ```powershell
       python .\PA\main.py --config PA\config\config_ecomm.yaml --only-steps 1,2,3,4,5,8,9,10
       ```

    In post-analysis, the key Neo4j relationships for "what actually happened" are:
    - `assisted_session_this_year` and `registered_to_show`, both created in **step 8**.
    - These are **not** used by the recommendation engine in step 10; they are for analysis/comparison only.

5. Inspect MLflow (if enabled) for the new metrics and archive the generated CSVs/Neo4j reports for analysis.

---

## ✅ Verification checklist

- `data/<event>/output/sessions_visited_this_year.csv` exists and mirrors the column schema of `sessions_visited_last_bva.csv`.
- `MATCH ()-[r:assisted_session_this_year]->() RETURN COUNT(r)` in Neo4j returns a positive value.
- `MATCH ()-[r:registered_to_show]->() RETURN COUNT(r)` highlights entry scan connections.
- MLflow run shows updated visitor/session counts and `assisted_session_this_year` relationship metrics.
- Visitor nodes in Neo4j show refreshed properties (e.g., latest registration answers).

If any check fails, re-run the relevant processor step with the same config.

---

## 🧰 Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `sessions_visited_this_year.csv` missing | Post-analysis scan files not provided or paths incorrect | Verify `post_analysis_mode.scan_files.*` paths and rerun step 2 |
| `assisted_session_this_year` relationships not created | Sessions file lacked `key_text`/Seminar mapping or visitors missing | Ensure session export contains session titles; confirm `sessions_visited_this_year.csv` has `key_text` populated |
| `registered_to_show` relationships missing | Entry scan CSV missing badge IDs or show references | Verify `post_analysis_mode.entry_scan_files` paths and column headers (`Badge Id`, `Show Ref`) |
| Visitor nodes not updating | Config still using `create_only_new=False` or missing registration data | Run post-analysis with default `create_only_new=True`; validate `df_reg_demo_this.csv` exists |
| Session merge falls back to processed CSV | `scan_files.session_this` path missing | Provide the raw export or update the path; rerun step 5 |
| Schema mismatch warning | Additional columns in this-year scans | Review warning details; update analytics scripts to accept new fields if needed |

---

## 🔄 Returning to personal agendas

1. Switch `mode` back to `"personal_agendas"`.
2. Remove (or leave inactive) the `post_analysis_mode` block.
3. If you ran only a subset of steps, rerun the full pipeline to rebuild recommendations for the upcoming cycle.
4. Archive the post-analysis outputs and export Neo4j results for postmortem reporting.

With these pieces in place, you can run a one-time (or periodic) post-show analysis to quantify recommendation performance without disrupting the standard personal agenda workflow.

---

## 📊 Post-Analysis Report Types

Post-analysis mode supports **four report types** that can be generated to communicate results to stakeholders. These reports are organized by scope (conversion campaign only vs. full journey) and detail level (full vs. summary).

### Report Matrix

| Report Type | Scope | Detail Level | Example Filename | Primary Audience |
|-------------|-------|--------------|------------------|------------------|
| **Full Report** | Conversion Campaign | Detailed | `report_cpcn_post_02122025.md` | Data Team, Technical Stakeholders |
| **Summary Report** | Conversion Campaign | Executive | `report_cpcn_post_summary_02122025.md` | Management, Quick Reference |
| **Enriched Full Report** | Registration + Conversion Campaign | Detailed | `report_cpcn_post_enriched_04122025.md` | Data Team, Strategy Team |
| **Enriched Summary Report** | Registration + Conversion Campaign | Executive | `report_cpcn_post_summary_enriched_04122025.md` | Management, Marketing Team |

---

### 📄 Report Type 1: Full Report (Conversion Campaign Only)

**Filename Pattern:** `report_<event>_post_<date>.md`  
**Example:** `report_cpcn_post_02122025.md`

**Purpose:** Comprehensive post-show analysis focusing on conversion campaign (personal agendas) recommendations versus actual session attendance.

**Key Sections:**
- Executive Summary with key performance metrics
- Task 0: Special Events Identification (sessions excluded from analysis)
- Task 1: Event Overview & Visitor Funnel
- Task 2: Hit Rate Analysis (session-level and visitor-level)
- Task 3: Session Performance Analysis
- Task 4-15: Detailed breakdowns (conversion rates, under/over-recommendation, gaps)
- Appendix: Configuration reference and methodology notes

**Data Sources:**
- Neo4j Post-Show Database (`neo4j-test`)
- `IS_RECOMMENDED` relationships (conversion campaign recommendations)
- `assisted_session_this_year` relationships (actual attendance)
- `registered_to_show` relationships (venue attendance)

**Key Metrics Covered:**
| Metric | Description |
|--------|-------------|
| Session-Level Hit Rate | % of attended sessions that were recommended |
| Visitor-Level Hit Rate | % of session attendees who attended ≥1 recommended session |
| Recommendation Coverage | % of registered visitors who received recommendations |
| Venue Attendance Rate | % of registered visitors who attended the venue |
| Session Engagement Rate | % of registered visitors who attended ≥1 session |
| Conversion Rate per Session | (Actual Attendance / Recommendations) × 100 |

**When to Use:**
- Standard post-show performance review
- When registration campaign data is not available or not relevant
- For technical deep-dives into recommendation algorithm performance

---

### 📄 Report Type 2: Summary Report (Conversion Campaign Only)

**Filename Pattern:** `report_<event>_post_summary_<date>.md`  
**Example:** `report_cpcn_post_summary_02122025.md`

**Purpose:** Executive-level summary of conversion campaign performance with key metrics and actionable insights.

**Key Sections:**
- Report Metadata
- Executive Summary: Visitor Journey Funnel (table format)
- Key Metrics: Coverage & Reach, Recommendation Effectiveness, Funnel Conversion
- Key Insights: Strengths, Areas for Improvement, Recommendations
- Data Quality Notes

**Visitor Journey Funnel Table:**
```markdown
|  | **Recommendations Sent** | **No Recommendations** | **Total** |
|:--|:-----------------------:|:---------------------:|:---------:|
| **Registered** | X,XXX | XXX | X,XXX |
| **Attended show** | X,XXX | XXX | X,XXX |
| **Attended any sessions (≥1)** | X,XXX | XXX | X,XXX |
| **Attended recommended sessions** | X,XXX | — | X,XXX |
```

**Key Metrics Covered:**
| Metric Category | Metrics Included |
|-----------------|------------------|
| Coverage & Reach | Recommendation Coverage, Show Attendance Rate, Session Engagement Rate |
| Recommendation Effectiveness | Visitor-Level Hit Rate, Session Engagement (With/Without Recs), Engagement Lift |
| Funnel Conversion | Stage-to-stage conversion rates with/without recommendations |

**When to Use:**
- Management presentations and executive briefings
- Quick performance snapshots
- Cross-event comparisons

---

### 📄 Report Type 3: Enriched Full Report (Registration + Conversion Campaign)

**Filename Pattern:** `report_<event>_post_enriched_<date>.md`  
**Example:** `report_cpcn_post_enriched_04122025.md`

**Purpose:** Complete visitor journey analysis from registration campaign (re-engaging prior-year attendees) through conversion campaign recommendations to actual attendance.

**Key Sections:**
- Executive Summary with complete journey metrics
- **NEW Section E1:** Registration Campaign Funnel (prior-year → re-registration → conversion recs → attendance)
- **NEW Section E2:** Returning vs New Visitor Performance Comparison
- **NEW Section E3:** Session Journey (Registration Campaign → Conversion Campaign → Attendance)
- **NEW Section E4:** Lost Registration Campaign Signal Analysis
- **NEW Section E5:** Sessions Cancelled After Registration Campaign Recommendations
- Original Tasks 0-15 from Full Report
- Registration Campaign Effectiveness Summary
- Recommendations for Future Events

**Data Sources:**
- Neo4j Post-Show Database (`neo4j-test`) - Conversion campaign and attendance data
- Neo4j Registration Campaign Database (`neo4j-dev`) - Registration campaign recommendations
- `Same_Visitor` relationships - Links returning visitors across years

**New Metrics Covered:**
| Metric | Description |
|--------|-------------|
| Registration Pool | Prior-year attendees targeted in registration campaign |
| Re-registration Rate | % of registration pool who registered for current year |
| Registration Campaign Recs | Recommendations generated during registration campaign |
| Conversion Campaign Recs | Recommendations generated during conversion campaign |
| Signal Lost | % change in recommendations between registration and conversion campaigns |
| Returning Visitor Hit Rate | Hit rate for visitors from registration campaign cohort |
| New Visitor Hit Rate | Hit rate for visitors not in registration campaign |

**Session Journey Table:**
| Session | Reg Campaign Recs | Conv Campaign Recs | Actual Attendance | Conv Conversion | Journey Status |
|---------|-------------------|--------------------|--------------------|-----------------|----------------|
| Session A | 796 | 3 | 84 | 2,800% | 🔴 CRITICAL: Lost registration signal |
| Session B | 748 | 1,020 | 120 | 11.8% | ✅ Well recommended |

**When to Use:**
- Full campaign lifecycle analysis
- Validating registration campaign ROI
- Identifying algorithm drift between campaign modes
- Strategy team deep-dives

---

### 📄 Report Type 4: Enriched Summary Report (Registration + Conversion Campaign)

**Filename Pattern:** `report_<event>_post_summary_enriched_<date>.md`  
**Example:** `report_cpcn_post_summary_enriched_04122025.md`

**Purpose:** Executive-level summary of the complete visitor journey from registration campaign through conversion campaign to actual attendance.

**Key Sections:**
- Report Metadata (includes both databases)
- Executive Summary: Complete Visitor Journey Funnel
- **NEW:** Registration Campaign to Post-Show Funnel
- Key Metrics by Visitor Type (Returning vs New)
- Registration Campaign Signal Analysis: Lost Opportunities
- Key Insights with registration campaign context
- Summary Statistics across all campaign modes

**Complete Visitor Journey Funnel Table:**
```markdown
|  | **Registration Pool** | **Re-registered** | **Conv Campaign Recs** | **No Conv Recs** | **Total** |
|:--|:---------------------:|:-----------------:|:----------------------:|:----------------:|:---------:|
| **Target/Registered** | 1,576 | 671 | 1,904 | 228 | 2,132 |
| **Attended show** | — | 590 | 1,659 | 204 | 1,863 |
| **Attended any sessions (≥1)** | — | 229 | 695 | 83 | 778 |
| **Attended recommended sessions** | — | 94 | 291 | — | 291 |
```

**Key Comparison Metrics:**
| Metric | Returning (Registered) | New Visitors | Delta |
|--------|:----------------------:|:------------:|:-----:|
| Session-Level Hit Rate | 16.8% | 14.9% | +1.9 pp |
| Visitor-Level Hit Rate | 41.0% | 35.9% | +5.1 pp |

**When to Use:**
- Management presentations requiring full campaign context
- Justifying registration campaign investment
- Marketing team briefings on campaign effectiveness
- Quick reference for cross-campaign performance

---

### 🔧 Generating Reports

Reports are generated using the prompt templates and Neo4j queries defined in:
- `prompt_post_show_generic.md` - Full report generation prompts
- `prompt_post_show_summary_generic.md` - Summary report generation prompts

**Enriched Report Generation Requirements:**

To generate enriched reports (Types 3 & 4), you need access to **both databases**:

| Database | Contains | Purpose |
|----------|----------|---------|
| `neo4j-dev` | Registration campaign data | `IS_RECOMMENDED` from engagement mode run |
| `neo4j-test` | Post-show data | `IS_RECOMMENDED` (conversion), `assisted_session_this_year`, `registered_to_show` |

**Key Queries for Enriched Reports:**

```cypher
// Registration campaign recommendations (neo4j-dev)
MATCH (v:Visitor_this_year)-[r:IS_RECOMMENDED]->(s:Sessions_this_year)
WHERE v.show = '[SHOW_CODE]'
RETURN s.session_id, s.title, count(r) AS reg_campaign_recs

// Returning visitor performance (neo4j-test)
MATCH (v:Visitor_this_year)-[sv:Same_Visitor]->(lv)
WHERE v.show = '[SHOW_CODE]'
OPTIONAL MATCH (v)-[att:assisted_session_this_year]->(s:Sessions_this_year)
OPTIONAL MATCH (v)-[rec:IS_RECOMMENDED]->(s)
WITH v, count(DISTINCT att) AS sessions_attended,
     count(DISTINCT CASE WHEN rec IS NOT NULL THEN att END) AS hits
WHERE sessions_attended > 0
RETURN sum(sessions_attended) AS total_attendances,
       sum(hits) AS total_hits,
       round(100.0 * sum(hits) / sum(sessions_attended), 1) AS session_hit_rate
```

---

### 📋 Report Selection Guide

| Scenario | Recommended Report |
|----------|-------------------|
| Standard post-show review | Full Report (Type 1) |
| Executive presentation (conversion only) | Summary Report (Type 2) |
| Full campaign lifecycle analysis | Enriched Full Report (Type 3) |
| Executive presentation (full journey) | Enriched Summary Report (Type 4) |
| Validating registration campaign ROI | Enriched Full Report (Type 3) |
| Cross-event comparison | Summary Reports (Types 2 or 4) |
| Algorithm debugging | Full Reports (Types 1 or 3) |
| Marketing team briefing | Enriched Summary Report (Type 4) |

---

### 📁 Report Archive Structure

Recommended folder structure for archiving post-analysis reports:

```
reports/
└── <event>/
    └── <year>/
        ├── post_show/
        │   ├── report_<event>_post_<date>.md           # Full Report
        │   ├── report_<event>_post_summary_<date>.md   # Summary Report
        │   ├── report_<event>_post_enriched_<date>.md  # Enriched Full
        │   └── report_<event>_post_summary_enriched_<date>.md  # Enriched Summary
        ├── pre_show/
        │   └── report_<event>_consolidated_<date>.md
        └── registration_campaign/
            └── report_<event>_engagement_<date>.md
```

**Example for CPCN 2025:**
```
reports/
└── cpcn/
    └── 2025/
        ├── post_show/
        │   ├── report_cpcn_post_02122025.md
        │   ├── report_cpcn_post_summary_02122025.md
        │   ├── report_cpcn_post_enriched_04122025.md
        │   └── report_cpcn_post_summary_enriched_04122025.md
        ├── pre_show/
        │   └── report_cpcn_consolidated_13112025.md
        └── registration_campaign/
            └── report_cpcn_engagement_08102025.md
```