# Engagement Mode

Engagement mode lets you reuse the personal-agenda pipeline to re-engage last year's visitors with this year's programme without rebuilding any processors. It runs the **exact same 10-step pipeline**, but tweaks the inputs and configuration so that visitors from the prior show are treated as "this year's" cohort. The resulting recommendations focus on promoting upcoming sessions to known attendees before you switch back to the standard *personal_agendas* cycle.

---

## 📌 How engagement mode differs from personal agendas

| Area | Personal agendas (default) | Engagement mode |
| --- | --- | --- |
| Config flag | `mode: "personal_agendas"` | `mode: "engagement"` |
| Visitor cohort in `Visitor_this_year` | Current registration/demographic files | Prior-year main-event attendees (treated as current) |
| Same-visitor matching | Links current visitors to prior-year main/secondary records | Links prior-year attendees (now `Visitor_this_year`) back to their own historical records |
| Recommendation target | Current-year visitors | Last-year main-event visitors |
| Input files | 4 JSONs (current & past, main & secondary) | Same files; multi-year JSON for main event must include both years |
| Output filenames | Legacy BVA/LVA naming retained | Identical names to avoid downstream refactors |
| Extra config | None | `engagement_mode.registration_shows`, `drop_last_year_when_missing`, `reset_returning_flags` |

Under the hood, **every processor still runs**:

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

Engagement mode simply rewires the show filters and returning-visitor flags so that this flow targets last year's audience.

---

## 🧭 End-to-end data flow in engagement mode

1. **Registration processing**
   - Reads the same multi-year JSON files as personal agendas.
   - `engagement_mode.registration_shows` overrides the show codes so that "this-year" filters point at last year's main-event codes (e.g. `CPCN24`, `CPCN2024`).
   - Optional switches:
     - `drop_last_year_when_missing`: removes past-year placeholder rows when matching fails.
     - `reset_returning_flags`: zeros out the returning-visitor booleans so downstream processors treat everyone as "new" for this pass.
   - Outputs (CSV/JSON) keep the legacy names (`Registration_data_bva_25_only_valid.csv`, etc.) but now contain prior-year attendees in the `*_this` slots.

2. **Scan processing**
   - Uses the same scan exports and demographic-enriched registration CSVs produced above.
   - Outputs `scan_bva_past.csv`, `scan_lva_past.csv`, and `sessions_visited_last_*.csv` with engagement-adjusted content, ready for relationship creation.

3. **Session processing**
   - Filters/cleans the session exports exactly as before.
   - Keeps legacy filenames (`session_this_filtered_valid_cols.csv`, `session_last_filtered_valid_cols_bva.csv`, etc.).
   - Because the outputs are neutral, the Neo4j stage can be rerun later for personal agendas without conflicts.

4. **Neo4j visitor/session uploads (steps 4–7)**
   - Load the adjusted visitor cohort into Neo4j.
   - The `show_name` remains the live-event identifier (`cpcn`, `ecomm`, etc.), so recommendations are still created under the upcoming show namespace.
   - Session nodes/streams come from the current-year session exports, so recommendations point at this year’s content.

5. **Neo4j visitor relationships (step 8)**
   - Because registration outputs now hold last-year visitors as "this year", the `same_visitor` logic connects them back to their own prior-year records (`Visitor_last_year_bva`/`Visitor_last_year_lva`).
   - `attended_session` edges are recreated from the engagement run’s scan outputs, allowing recommendation logic to see what those visitors attended last year.

6. **Embeddings and recommendations (steps 9–10)**
   - Embedding generation is unchanged; it operates on the current-year session catalogue.
   - Recommendation logic finds last-year attendees (now in `Visitor_this_year`) and promotes similar sessions scheduled for the upcoming event. New visitors (if any slipped through) fall back to popular sessions, just like the standard mode.

The produced recommendations land under `data/<event>/recommendations/` using the same timestamped filenames you already consume.

---

## 🛠️ Configuration checklist

Minimum settings for engagement mode:

```yaml
# config/config_<event>.yaml
mode: "engagement"

engagement_mode:
  registration_shows:
    this_year_main:
      - ["CPCN24", "CPCN2024"]   # Prior-year main-event codes
    this_year_secondary:
      - ["CPC2025", "CPC25"]     # Prior-year secondary/pair event codes
    last_year_main: []              # Optional: keep empty if no older history
    last_year_secondary: []
    drop_last_year_when_missing: true
  reset_returning_flags: true

# Leave event/output filenames untouched so legacy processors still work
session_output_files:
  processed_sessions:
    this_year: "session_this_filtered_valid_cols.csv"
    last_year_main: "session_last_filtered_valid_cols_bva.csv"
    last_year_secondary: "session_last_filtered_valid_cols_lva.csv"
  streams_catalog: "streams.json"

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

Additional tips:

- **Input files**: the main-event registration/demographic JSON must bundle *both* the current and prior years (as shown in `config_ecomm.yaml`). Engagement mode filters those internally; you don’t need to pre-split files.
- **Output directory hygiene**: ensure `data/<event>/output/` is either empty or archived before the engagement run so the results are easy to distribute.
- **Neo4j credentials**: unchanged; engagement mode writes into the same database/labels.

---

## ▶️ Running the engagement pass

1. Update the event config with the checklist above.
2. (Optional) Clear `PA/data/<event>/output/` if you want a clean snapshot.
3. From the repo root, run the pipeline the same way you do for personal agendas, for example:

```powershell
.\.venv\Scripts\python.exe main.py --config PA\config\config_cpcn.yaml
```

4. Deliver the generated recommendation exports to the engagement team.

Because this is typically a one-off send, keep the outputs available but **do not overwrite them** when you switch back to personal-agenda mode.

---

## 🔁 Switching back to personal agendas

1. Change `mode` back to `"personal_agendas"` in the config.
2. Remove or comment out the `engagement_mode` overrides if they are no longer needed.
3. Refresh any cached outputs (especially registration/scan CSVs) before running the standard cycle.
4. Re-run the pipeline as usual to rebuild Neo4j nodes and recommendations for the live show.

Because filenames never changed, downstream dashboards and exports continue to function with no additional code changes.

---

## ✅ Verification checklist after an engagement run

- `data/<event>/output/Registration_data_with_demographicdata_bva_this.csv` contains last-year attendees.
- `data/<event>/output/scan_bva_past.csv` reflects historical scan activity for the main event.
- Neo4j has `attended_session` edges for the show (`MATCH ()-[r:attended_session]->() RETURN COUNT(r)` > 0).
- Recommendation CSVs under `data/<event>/recommendations/` include last-year visitor badge IDs.

If any of the above checks fail, re-run the specific processor step (e.g., relationship processor) or review the config overrides.

---

## 🧰 Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `Neo.ClientNotification.Statement.UnknownRelationshipTypeWarning` for `attended_session` | Relationship processor skipped creation because legacy keys (`bva`/`lva`) weren’t present | Ensure `visitor_relationship.same_visitor_properties` keeps `bva`/`lva` keys even if the `type` values change (e.g. `bva: { type: "cpcn" }`). Re-run step 8. |
| Recommendations log "No popular sessions found" | No `attended_session` edges present | Same as above—rebuild relationships so the recommender sees last-year attendance. |
| "No valid similarity attributes" warnings | Registration outputs missing survey fields referenced in `similarity_attributes` | Either include those questions in the transformation or adjust weights to available columns before running recommendations. |
| Engagement outputs overwrote personal-agenda data | Both modes share filenames by design | Archive engagement outputs before switching modes, or run engagement on a branch copy of the `data/<event>/output` folder. |

---

## ℹ️ Additional resources

- `PA/config/config_ecomm.yaml`: canonical example with `mode: "engagement"`.
- `PA/config/config_cpcn.yaml`: CPCN adaptation keeping legacy file names.
- Processor implementations in `PA/` (e.g., `registration_processor.py`, `neo4j_visitor_relationship_processor.py`) for deeper behaviour details.

With this setup, you can run the engagement pass once, distribute the recommendations, and immediately pivot back to the regular personal agendas workflow with zero refactoring.
