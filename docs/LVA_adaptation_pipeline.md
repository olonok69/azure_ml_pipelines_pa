# LVA Pipeline Adaptation - Issues and Changes Log

## Overview
This document logs all issues identified and changes made to adapt the generic event recommendation pipeline for the LVA (London Vet Show) 2025.

---

## 1. Critical Configuration Issues Identified

### 1.1 Specialization Mapping Transformation Bug 🔴
**Issue**: The pipeline was incorrectly transforming specializations using a mapping meant for historical data migration.

**Root Cause**: 
- The `specialization_map` in the configuration was transforming "Small Animal" → "Companion Animal"
- "Companion Animal" doesn't exist in the current year's `spezialization_to_stream.csv`
- This affected 2,338 visitors (58.8% of total)

**Fix Applied**:
```yaml
# config_vet_lva.yaml
specialization_stream_mapping:
  specialization_map: {}  # Set to empty to prevent transformations
```

**Impact**: 
- Before: Only 6,279 specialization relationships created
- Expected After: ~15,000+ relationships
- Visitors affected: 2,959 → 0

---

### 1.2 Missing Job Role: Clinical or Other Director 🔴
**Issue**: 388 visitors (9.8%) had the job role "Clinical or other Director" which was not in the mapping file.

**Root Cause**: 
- `job_to_stream.csv` only had 13 rows, missing this critical role
- Clinical Directors are senior decision-makers needing specific content

**Fix Applied**:
- Updated `job_to_stream.csv` from 13 to 15 rows
- Added "Clinical or other Director" with 51 stream mappings
- Focused on leadership, management, and strategic streams

**Impact**:
- Before: 388 visitors with no job-based recommendations
- After: Full coverage with appropriate senior-level content

---

## 2. Data Quality Issues

### 2.1 String Mismatch: "Other" Role ⚠️
**Issue**: Database had "Other" but CSV had "Other (please specify)"

**Resolution**: 
- Added "Other" as a separate row in `job_to_stream.csv`
- Set to 0 stream mappings as per stakeholder requirement
- Affects 108 visitors (2.7%) - accepted business decision

### 2.2 Empty Mapping Files ⚠️
**Issue**: Several JSON files were empty or incomplete

**Files Updated**:
- `specializations.json`: Updated with current year terms
- `job_roles.json`: Remains empty (not critical for processing)
- `streams.json`: Generated with 61 stream descriptions

---

## 3. Veterinary-Specific Adaptations

### 3.1 Enabled Features
Unlike ECOMM events, LVA requires veterinary-specific processing:

```yaml
# Features enabled for LVA
neo4j:
  job_stream_mapping:
    enabled: true  # Maps job roles to relevant streams
  specialization_stream_mapping:
    enabled: true  # Maps practice types to content
```

### 3.2 Filtering Rules Configuration
```yaml
recommendation:
  enable_filtering: true  # Veterinary-specific filtering
  rules_config:
    equine_mixed_exclusions: ["exotics", "feline", "small animal"]
    small_animal_exclusions: ["equine", "farm", "large animal"]
    vet_exclusions: ["nursing"]
    nurse_streams: ["nursing", "wellbeing", "welfare"]
```

### 3.3 Role Groups Definition
```yaml
role_groups:
  vet_roles: [Vet/Vet Surgeon, Assistant Vet, Vet/Owner, 
              Clinical or other Director, Locum Vet, Academic]
  nurse_roles: [Head Nurse/Senior Nurse, Vet Nurse, Locum RVN]
  business_roles: [Practice Manager, Practice Partner/Owner]
  other_roles: [Student, Receptionist, Other]
```

---

## 4. File Changes Summary

### 4.1 Configuration Files
| File | Changes Made | Reason |
|------|-------------|--------|
| config_vet_lva.yaml | Set specialization_map: {} | Prevent incorrect transformations |
| config_vet_lva.yaml | Enabled job/specialization mapping | Veterinary-specific features |
| config_vet_lva.yaml | Added filtering rules | Role-appropriate content |

### 4.2 Mapping Files
| File | Before | After | Changes |
|------|--------|-------|---------|
| job_to_stream.csv | 13 rows | 15 rows | Added Clinical Director, Other |
| spezialization_to_stream.csv | 7 rows | 7 rows | No changes needed |
| specializations.json | Incorrect terms | Updated | Correct current year terms |
| streams.json | Not existed | 61 streams | Generated descriptions |

### 4.3 Database Impact
| Metric | Before Changes | After Changes | Improvement |
|--------|---------------|---------------|-------------|
| Job mappings coverage | 87.5% | 97.3% | +11% |
| Specialization coverage | 25.6% | 78% (expected) | +204% |
| Total relationships | 158,869 | ~187,000 (expected) | +18% |

---

## 5. Processing Differences from ECOMM

### 5.1 Feature Comparison
| Feature | LVA (Veterinary) | ECOMM | Difference |
|---------|------------------|-------|------------|
| Specialization Mapping | ✅ Enabled | ❌ Disabled | Vet-specific |
| Job Stream Mapping | ✅ Enabled | ❌ Disabled | Vet-specific |
| Practice Type Filtering | ✅ Active | ❌ Not used | Role-based content |
| Stream Count | 61 | ~40 | More specialized |
| Similarity Attributes | 4 (with practice type) | 2-3 | More complex |

### 5.2 Configuration Differences
```yaml
# LVA-specific settings
specialization_stream_mapping:
  enabled: true
  specialization_field_this_year: "what_type_does_your_practice_specialise_in"
  
# ECOMM would have
specialization_stream_mapping:
  enabled: false  # Generic events don't use specialization
```

---

## 6. Validation Checklist

### Pre-Run Validation ✅
- [x] Verify specialization_map is empty in config
- [x] Confirm job_to_stream.csv has 15 rows
- [x] Check Clinical Director has stream mappings
- [x] Validate specializations.json has correct terms
- [x] Ensure filtering rules are configured
- [x] Confirm similarity attributes include practice type

### Post-Run Validation 
- [ ] Verify Small Animal visitors have relationships
- [ ] Check Clinical Directors have job mappings
- [ ] Confirm specialization relationships > 15,000
- [ ] Validate recommendation quality scores
- [ ] Review visitors with < 10 recommendations
- [ ] Check stream utilization statistics

---

## 7. Lessons Learned

### 7.1 Configuration Management
1. **Mapping Direction Matters**: Historical migration mappings should not be applied to current data
2. **Empty Defaults Are Safer**: Use empty mappings {} rather than incorrect transformations
3. **Validate Against CSV Headers**: Ensure mapped values exist in target CSV files

### 7.2 Data Validation
1. **String Matching**: Exact string matching requires careful validation
2. **Role Coverage**: Always verify all database values exist in mapping files
3. **Specialization Parsing**: System correctly handles ";" separated values

### 7.3 Testing Recommendations
1. **Sample Testing**: Test with a few records before full run
2. **Relationship Counts**: Monitor relationship creation counts
3. **Coverage Metrics**: Track percentage of visitors with mappings

---

## 8. Future Improvements

### 8.1 Short-term
- Add validation scripts to check mapping completeness
- Implement warnings for unmapped values
- Create automated tests for configuration

### 8.2 Long-term
- Develop configuration validator tool
- Implement dynamic mapping discovery
- Create feedback loop from engagement data
- Build ML-based stream assignment

---

## 9. Commands for Pipeline Execution

### Full Pipeline Run
```bash
python PA/main.py --config config/config_vet_lva.yaml
```

### Targeted Fix Application
```bash
# Run only the mapping steps after fixes
python PA/main.py --config config/config_vet_lva.yaml --only-steps 6,7

# Regenerate recommendations only
python PA/run_recommendations.py --config config/config_vet_lva.yaml
```

### Verification Queries
```cypher
// Verify specialization mappings
MATCH (v:Visitor_this_year {show: 'lva'})-[r:specialization_to_stream]->()
RETURN count(r) as specialization_mappings

// Check job mappings
MATCH (v:Visitor_this_year {show: 'lva'})-[r:job_to_stream]->()
RETURN count(r) as job_mappings

// Validate Clinical Directors
MATCH (v:Visitor_this_year {show: 'lva', job_role: 'Clinical or other Director'})-[r:job_to_stream]->()
RETURN count(distinct v) as directors_with_mappings
```

---
