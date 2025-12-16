# Control Group Implementation for Personal Agendas

This document explains how the recommendation pipeline handles control-group visitors, how to enable it via configuration, and what downstream artifacts (files, Neo4j properties) are affected.

---

## Overview

The control group randomly withholds a slice of visitors from **receiving** recommendations (via exports) so their onsite behaviour can be compared with those who *did* receive suggestions. The logic is implemented inside `session_recommendation_processor.SessionRecommendationProcessor` and is active only when the show runs in **personal_agendas** mode.

### Key Design Principles

1. **Recommendations are computed for ALL visitors** (control and non-control)
2. **IS_RECOMMENDED relationships are created for ALL visitors** in Neo4j
3. **File exports are separated**: main files exclude control visitors, control files contain only control visitors
4. **Control visitors are marked** with `control_group = 1` property for identification
5. **All visitors get `has_recommendation = "1"`** to prevent reprocessing in incremental runs

### Why Keep IS_RECOMMENDED for Control Visitors?

The IS_RECOMMENDED relationships are intentionally preserved for control group visitors to enable **post-show A/B analysis**:

| Analysis | Description |
|----------|-------------|
| **Treatment Hit Rate** | % of sessions attended that were recommended (control_group = 0) |
| **Organic Hit Rate** | % of sessions attended that were recommended but NOT sent (control_group = 1) |
| **Recommendation Lift** | Treatment Hit Rate - Organic Hit Rate = true recommendation effectiveness |

Without IS_RECOMMENDED edges for control visitors, we lose the ability to measure organic attendance of sessions we *would have* recommended.

---

## Configuration

Control groups are configured under the `recommendation.control_group` section of each show's YAML file (e.g., `config/config_vet_lva.yaml`):

```yaml
recommendation:
  control_group:
    enabled: true               # flip to true to activate
    percentage: 5               # percent of eligible visitors to withhold
    random_seed: 42             # optional deterministic seed for repeatability
    file_suffix: "_control"     # appended to control-group output filenames
    output_directory: "recommendations/control"  # optional target dir
    neo4j_property: "control_group"
```

### Configuration Settings

| Setting | Description | Default |
|---------|-------------|---------|
| **enabled** | Must be `true` to activate the logic | `false` |
| **percentage** | Accepts 0-1 fractions or 0-100 percentages; internally normalised to 0–1. Applied to visitors who produced ≥1 recommendation | `5` |
| **random_seed** | Optional; set to reproduce the same control sample across runs | `null` |
| **file_suffix** | Appended to control-group output filenames | `"_control"` |
| **output_directory** | Optional target directory for control files | `null` |
| **neo4j_property** | Name of the flag property stored on `Visitor_this_year` nodes | `"control_group"` |

---

## Pipeline Behaviour

### Step 1: Generate Recommendations

All visitors are processed identically—recommendations are computed for everyone, including those who will be assigned to the control group.

### Step 2: Split Control Group

The `_split_control_group` method divides visitors with recommendations:

```
┌─────────────────────────────────────────────────────────────┐
│           All Visitors with Recommendations                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌─────────────────────┐    ┌─────────────────────────┐   │
│   │   Main Group (95%)  │    │  Control Group (5%)     │   │
│   │   control_group = 0 │    │  control_group = 1      │   │
│   │                     │    │                         │   │
│   │   → Main CSV/JSON   │    │  → Control CSV/JSON     │   │
│   │   → Sent to team    │    │  → Withheld from team   │   │
│   └─────────────────────┘    └─────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Step 3: Persist Outputs

| Output | Content | Purpose |
|--------|---------|---------|
| `visitor_recommendations_<show>_<timestamp>.json` | Non-control visitors only | Send to team for personal agendas |
| `visitor_recommendations_<show>_<timestamp>.csv` | Non-control visitors only | CRM/marketing integration |
| `visitor_recommendations_<show>_<timestamp>_control.json` | Control visitors only | Analysis & audit |
| `visitor_recommendations_<show>_<timestamp>_control.csv` | Control visitors only | Analysis & audit |

### Step 4: Neo4j Updates

**All visitors** (control and non-control) receive:

| Update | Control Visitors | Non-Control Visitors |
|--------|-----------------|---------------------|
| `has_recommendation` property | `"1"` | `"1"` |
| `control_group` property | `1` | `0` |
| `IS_RECOMMENDED` relationships | ✅ Created | ✅ Created |
| `recommendations_generated_at` | Timestamp | Timestamp |

**Important**: `has_recommendation = "1"` for control visitors ensures they are **not reprocessed** in incremental runs.

---

## Neo4j State After Processing

### Expected State

```cypher
// Verify control group implementation
MATCH (v:Visitor_this_year)
WHERE v.show = 'lva'
RETURN 
  v.control_group AS control_flag,
  v.has_recommendation AS has_rec,
  count(*) AS visitors
ORDER BY control_flag DESC
```

Expected result:
| control_flag | has_rec | visitors |
|--------------|---------|----------|
| 1 | "1" | ~5% of total |
| 0 | "1" | ~95% of total |

### Verify IS_RECOMMENDED Relationships

```cypher
// Both control and non-control should have IS_RECOMMENDED
MATCH (v:Visitor_this_year)
WHERE v.show = 'lva'
OPTIONAL MATCH (v)-[r:IS_RECOMMENDED]->(s:Sessions_this_year)
WITH v.control_group AS control_flag, count(r) AS rec_count
WHERE rec_count > 0
RETURN control_flag, count(*) AS visitors_with_recs
ORDER BY control_flag DESC
```

Both control_flag = 1 and control_flag = 0 should have visitors with recommendations.

---

## Incremental Runs

When running with `create_only_new: true`:

1. **Query**: Get visitors where `has_recommendation IS NULL OR has_recommendation = "0"`
2. **Control visitors are SKIPPED**: They already have `has_recommendation = "1"`
3. **Only new registrants** are processed

This prevents control visitors from being reprocessed and potentially moved to the non-control group.

---

## Post-Analysis Mode Integration

In `post_analysis` mode, the control group enables A/B testing analysis:

### Key Relationships

| Relationship | Source | Purpose |
|--------------|--------|---------|
| `IS_RECOMMENDED` | personal_agendas mode | What we recommended |
| `assisted_session_this_year` | post_analysis mode | What they actually attended |
| `registered_to_show` | post_analysis mode | Who entered the venue |

### Control Group Analysis Queries

#### Query 1: Treatment vs Control Hit Rates

```cypher
// Compare session-level hit rates between treatment and control groups
MATCH (v:Visitor_this_year)-[a:assisted_session_this_year]->(s:Sessions_this_year)
WHERE v.show = 'lva' AND (s.show = 'lva' OR s.show IS NULL)
OPTIONAL MATCH (v)-[r:IS_RECOMMENDED]->(s)
WITH v.control_group AS control_flag,
     count(a) AS total_attended,
     count(CASE WHEN r IS NOT NULL THEN 1 END) AS attended_recommended
RETURN 
  CASE control_flag WHEN 1 THEN 'Control (not sent)' ELSE 'Treatment (sent)' END AS group_name,
  sum(total_attended) AS total_sessions_attended,
  sum(attended_recommended) AS recommended_sessions_attended,
  round(100.0 * sum(attended_recommended) / sum(total_attended), 2) AS hit_rate_pct
ORDER BY control_flag
```

#### Query 2: Visitor-Level Comparison

```cypher
// Visitors who attended at least one recommended session
MATCH (v:Visitor_this_year)-[a:assisted_session_this_year]->(s:Sessions_this_year)
WHERE v.show = 'lva' AND (s.show = 'lva' OR s.show IS NULL)
WITH v, collect(DISTINCT s.session_id) AS attended_sessions
OPTIONAL MATCH (v)-[r:IS_RECOMMENDED]->(rec:Sessions_this_year)
WHERE rec.session_id IN attended_sessions
WITH v.control_group AS control_flag,
     v,
     CASE WHEN count(rec) > 0 THEN 1 ELSE 0 END AS had_hit
RETURN 
  CASE control_flag WHEN 1 THEN 'Control' ELSE 'Treatment' END AS group_name,
  count(v) AS visitors_attended,
  sum(had_hit) AS visitors_with_hits,
  round(100.0 * sum(had_hit) / count(v), 2) AS visitor_hit_rate_pct
ORDER BY control_flag
```

#### Query 3: Recommendation Lift Calculation

```cypher
// Calculate the lift from sending recommendations
MATCH (v:Visitor_this_year)-[a:assisted_session_this_year]->(s:Sessions_this_year)
WHERE v.show = 'lva' AND (s.show = 'lva' OR s.show IS NULL)
OPTIONAL MATCH (v)-[r:IS_RECOMMENDED]->(s)
WITH v.control_group AS control_flag,
     count(a) AS attended,
     count(CASE WHEN r IS NOT NULL THEN 1 END) AS hits
WITH control_flag,
     sum(attended) AS total_attended,
     sum(hits) AS total_hits,
     100.0 * sum(hits) / sum(attended) AS hit_rate
WITH collect({flag: control_flag, rate: hit_rate}) AS rates
WITH 
  [r IN rates WHERE r.flag = 0][0].rate AS treatment_rate,
  [r IN rates WHERE r.flag = 1][0].rate AS control_rate
RETURN 
  round(treatment_rate, 2) AS treatment_hit_rate,
  round(control_rate, 2) AS organic_hit_rate,
  round(treatment_rate - control_rate, 2) AS lift_percentage_points,
  round(100.0 * (treatment_rate - control_rate) / control_rate, 2) AS relative_lift_pct
```

---

## Downstream System Requirements

### Critical: Filtering for Personal Agenda Delivery

Any system that sends personal agendas **MUST** filter by `control_group`:

```cypher
// For sending personal agendas - EXCLUDE control group
MATCH (v:Visitor_this_year)-[r:IS_RECOMMENDED]->(s:Sessions_this_year)
WHERE v.show = 'lva' 
  AND (v.control_group = 0 OR v.control_group IS NULL)  // CRITICAL FILTER
RETURN v.BadgeId, v.Email, collect(s.title) AS recommendations
```

### Safe: Using CSV Exports

If using the CSV/JSON exports, control visitors are automatically excluded from the main files. Only `*_control.csv` files contain control visitors.

---

## Summary: What Goes Where

| Item | Main Export | Control Export | Neo4j |
|------|------------|----------------|-------|
| Visitor record | ✅ (control=0 only) | ✅ (control=1 only) | ✅ All |
| Recommendations | ✅ (control=0 only) | ✅ (control=1 only) | ✅ All |
| `has_recommendation` | "1" | "1" | "1" for all |
| `control_group` property | 0 | 1 | 0 or 1 |
| `IS_RECOMMENDED` edges | — | — | ✅ All |

---

## Validation Checklist

After running personal_agendas with control group enabled:

- [ ] Main CSV/JSON contains only `control_group = 0` visitors
- [ ] Control CSV/JSON contains only `control_group = 1` visitors  
- [ ] Neo4j: All visitors have `has_recommendation = "1"`
- [ ] Neo4j: ~5% of visitors have `control_group = 1`
- [ ] Neo4j: Both groups have `IS_RECOMMENDED` relationships
- [ ] Incremental run skips all previously processed visitors

---

## Report Type Integration

| Report Type | Uses Control Group? | Analysis |
|-------------|--------------------|-----------| 
| Pre-show (Conversion) | ❌ Not yet relevant | Standard metrics |
| Post-show Summary | ✅ Optional | Can segment by control_group |
| Post-show Full | ✅ Recommended | Treatment vs Control comparison |
| Post-show Enriched | ✅ Recommended | Full A/B analysis with lift calculation |

---

*Document Version: 2.0 | Updated: December 2025*
*Previous version incorrectly stated IS_RECOMMENDED should not be created for control visitors*