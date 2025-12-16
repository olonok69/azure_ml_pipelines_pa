# Theatre Capacity Enforcement Overview

This document explains the end-to-end implementation of the `theatre_capacity_limits` feature that now governs the session recommendation pipeline. The addition protects each theatre/slot from being over-suggested by trimming outgoing recommendations against an event-specific capacity plan.

## High-Level Goals

- Respect per-theatre visitor capacity when building personal agendas.
- Provide a non-destructive post-processing step that can be toggled per show via configuration.
- Surface detailed telemetry (counts, affected slots, notes per visitor) so event teams can validate that the limits behaved as expected.

## Configuration

The feature is configured under the `recommendation` section of the event YAML (example: `PA/config/config_vet_lva.yaml`).

```yaml
recommendation:
  theatre_capacity_limits:
    enabled: true
    capacity_file: "data/lva/teatres.csv"
    session_file: "data/lva/LVS25_session_export.csv"
    capacity_multiplier: 1.0
```

**Fields**
- `enabled`: Feature flag. When `false` none of the logic executes.
- `capacity_file`: CSV path with theatre names and their `number_visitors` (or analogous) capacities.
- `session_file`: CSV path with the slot schedule (must contain `session_id` and a theatre column).
- `capacity_multiplier`: Optional scalar used to be more conservative (e.g. `0.9` to ensure 10% buffer).

Changing the capacity source or session export only requires updating these paths. No code changes are needed as long as the CSVs contain the expected columns.

## Data Requirements

1. **Capacity CSV**
   - Must include a theatre column (`theatre__name`, `theatre_name`, or `theatre`).
   - Must include a capacity column (`number_visitors`, `capacity`, or `max_visitors`).
   - Empty capacities are ignored; such theatres will behave as “no cap”.
2. **Session CSV**
   - Must include `session_id`.
   - Must include the theatre column listed above.
   - Optional columns: `date`, `start_time`. If present they are folded into the slot key; otherwise the session ID alone defines the slot.

Column discovery is case insensitive. The processor will log a warning and disable the feature if the expected columns are missing or if the files cannot be read.

## Processor Changes (`PA/session_recommendation_processor.py`)

### Initialization

- New attributes:
  - `self.theatre_limits_enabled`, `self.theatre_capacity_multiplier`.
  - `self.theatre_capacity_map`: normalized theatre name → capacity.
  - `self.session_slot_index`: session ID → slot metadata (theatre, date, start time, slot key, capacity).
  - `self._theatre_capacity_stats`: dictionary for telemetry.
- `_initialize_theatre_capacity_limits()` is executed during `__init__`.
  - Loads CSVs, performs column discovery, normalizes theatre strings, removes duplicate session IDs, and builds slot keys.
  - Normalization uses `_normalize_theatre_name()` (lower-case, trimmed, whitespace collapsed).
  - `_parse_capacity_value()` converts values to integers, ignoring empties.
  - `_build_session_slot_key()` composes `theatre|date|start` (falls back to `theatre|session_id`).
  - Feature automatically disables itself (with informative logs) if files are missing or no valid entries are found.

### Enforcement

- `_enforce_theatre_capacity_limits(recommendations_dict)` runs once after all visitors are processed but before statistics are finalized.
- The method:
  1. Buckets recommended sessions per slot (`slot_key`). Each bucket stores visitor payload, recommendation record, similarity score, and slot metadata.
  2. Ignores sessions without known metadata or capacity, but counts them in diagnostics.
  3. Applies the allowance `floor(capacity * multiplier)`. When the bucket size exceeds that limit, entries are sorted by similarity descending.
  4. Keeps the top-N per slot and removes the remainder from `filtered_recommendations`.
  5. Adds a note to each affected visitor: `Removed session <id> due to theatre capacity limit (<slot label>, limit <allowed>)`.
  6. Updates per-visitor `metadata.filtered_count` to match the trimmed list.

- The adjustments summary (`self._theatre_capacity_stats`) tracks:
  - `recommendations_removed`
  - `slots_limited`
  - `sessions_missing_metadata`
  - `sessions_without_capacity`
  - `slots_processed`

### Statistics & Output

- Post-enforcement statistics recompute totals (`visitors_with_recommendations`, `total_recommendations_generated`, etc.).
- `self.statistics['theatre_capacity_limits']` carries the summary metrics, so they surface in the final JSON and in summary/MLflow logging.
- Visitor payloads retain the shortened recommendation list plus the appended notes for auditability.

## Logging & Diagnostics

- During initialization, the processor emits warnings when configuration files are missing or malformed and automatically disables the feature instead of failing the whole run.
- At runtime, after trimming, it logs how many recommendations were removed and how many slots were limited.
- Any recommendation removed for capacity reasons is annotated directly in the JSON export under `metadata.notes`.

## Validation Workflow

1. **Dry Run**
   - Run `python PA/session_recommendation_processor.py --config PA/config/config_vet_lva.yaml --create-only-new` (adjust flags as needed).
   - Confirm no warnings appear about missing columns or disabled enforcement.

2. **Review Statistics**
   - Inspect the console log for: `Applied theatre capacity limits: removed X recommendations across Y slot(s)`.
   - Open the output JSON inside `data/lva/recommendations/visitor_recommendations_<show>_<timestamp>.json` and locate `"theatre_capacity_limits"` under `statistics`.

3. **Spot Check Visitors**
   - For any trimmed slot, locate a visitor in the JSON and confirm the `metadata.notes` entry reflects the removal reason.
   - Ensure `metadata.filtered_count` aligns with the final recommendation count.

4. **Scenario Testing**
   - Adjust `capacity_multiplier` to a low value (e.g. `0.5`), rerun, and verify more recommendations get trimmed.
   - Modify the capacity CSV to remove a theatre’s capacity, rerun, and confirm the processor reports increased `sessions_without_capacity` while leaving those sessions untouched.

## Failure Modes & Safeguards

- **Missing files/columns**: feature auto-disables and logs a warning. Recommendation processing continues unchanged.
- **Empty capacity map**: treated as disable; no caps, explicit log entry.
- **Duplicate session IDs**: only the first entry is used, ensuring stable slot mappings.
- **Unexpected data types**: capacity parser ignores non-numeric values instead of raising.

## Integration Points

- No changes to the Neo4j import path were required. The limiter operates entirely within the recommendation processor’s workflow.
- Summary metrics and MLflow logging automatically consume the new statistics without additional wiring.
- Configuration defaults maintain backward compatibility; other events remain unaffected until they opt-in via YAML.

## Next Steps & Enhancements

- Add automated tests to cover CSV parsing edge cases and trimming logic.
- Extend slot identification to include end times to support back-to-back sessions in the same theatre.
- Consider exporting a dedicated CSV report summarizing which sessions hit capacity, aiding on-site scheduling decisions.

By following the guide above, teams can understand, operate, and extend the theatre capacity enforcement with confidence.
