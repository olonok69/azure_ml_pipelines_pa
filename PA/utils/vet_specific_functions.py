"""Veterinary-specific processing helpers for data processing and recommendations."""

import logging
from typing import Any, Dict, List, Tuple

def apply_vet_specific_practice_filling(self, df_reg_demo_this, df_reg_demo_last_bva, df_reg_demo_last_lva, practices):
    """Apply veterinary-specific practice type filling logic."""
    self.logger.info("Applying veterinary-specific practice type filling logic")
    
    # Use practice_type_columns from config for generic behavior
    practice_columns = self.config.get("practice_type_columns", {})
    this_year_col = practice_columns.get("current", "specialization_current")
    past_year_col_bva = practice_columns.get("past_bva", "specialization_past")
    past_year_col_lva = practice_columns.get("past_lva", "specialization_past")

    self.logger.info(f"Using practice type columns - current: {this_year_col}, past BVA: {past_year_col_bva}, past LVA: {past_year_col_lva}")


    # Fill missing practice types using vet-specific columns
    df_reg_demo_this = self.fill_missing_practice_types(
        df_reg_demo_this, practices, column=this_year_col
    )
    df_reg_demo_last_bva = self.fill_missing_practice_types(
        df_reg_demo_last_bva, practices, column=past_year_col_bva
    )
    df_reg_demo_last_lva = self.fill_missing_practice_types(
        df_reg_demo_last_lva, practices, column=past_year_col_lva
    )
    
    return df_reg_demo_this, df_reg_demo_last_bva, df_reg_demo_last_lva


def apply_vet_specific_job_roles(self, df):
    """Apply veterinary-specific job role processing logic - EXACTLY as original."""
    import re
    from difflib import SequenceMatcher
    
    self.logger.info("Applying veterinary-specific job role processing logic")
    
    df_copy = df.copy()

    # VET-SPECIFIC potential roles (exactly as in old processor)
    potential_roles = [
        "Student",
        "Other (please specify)",
        "Other",
        "Receptionist",
        "Head Nurse/Senior Nurse",
        "Vet/Vet Surgeon",
        "Practice Partner/Owner",
        "Academic",
        "Clinical or other Director",
        "Assistant Vet",
        "Vet/Owner",
        "Vet Nurse",
        "Locum Vet",
        "Practice Manager",
        "Locum RVN",
    ]

    mask = df_copy["job_role"] == "NA"

    # Apply VET-SPECIFIC rules (exactly as in old processor)
    for idx in df_copy[mask].index:
        job_title = str(df_copy.loc[idx, "JobTitle"]).lower()

        # VET-SPECIFIC keyword matching
        if "surgeon" in job_title:
            df_copy.loc[idx, "job_role"] = "Vet/Vet Surgeon"
        elif "nurse" in job_title:
            df_copy.loc[idx, "job_role"] = "Vet Nurse"
        elif "rvn" in job_title:
            df_copy.loc[idx, "job_role"] = "Vet Nurse"
        elif "locum" in job_title:
            df_copy.loc[idx, "job_role"] = "Locum Vet"
        elif "student" in job_title:
            df_copy.loc[idx, "job_role"] = "Student"
        elif "assistant" in job_title:
            df_copy.loc[idx, "job_role"] = "Assistant Vet"
        else:
            # Fuzzy matching with vet-specific roles
            cleaned_title = re.sub(r"\b(and|the|of|in|at|for)\b", "", job_title)
            best_match = None
            best_score = 0

            for role in potential_roles:
                role_lower = role.lower()
                role_terms = role_lower.split("/")
                role_terms.extend(role_lower.split())

                max_term_score = 0
                for term in role_terms:
                    if len(term) > 2:
                        if term in cleaned_title:
                            term_score = 0.9
                        else:
                            term_score = SequenceMatcher(None, cleaned_title, term).ratio()
                        max_term_score = max(max_term_score, term_score)

                if max_term_score > best_score:
                    best_score = max_term_score
                    best_match = role

            if best_score > 0.3:
                df_copy.loc[idx, "job_role"] = best_match
            else:
                df_copy.loc[idx, "job_role"] = "Other (please specify)"

    # Final vet-specific rule
    other_mask = df_copy["job_role"].str.contains("Other", case=False, na=False) & ~df_copy["job_role"].eq("Other (please specify)")
    df_copy.loc[other_mask, "job_role"] = "Other (please specify)"

    self.logger.info(f"Processed {mask.sum()} job roles using veterinary-specific logic")
    return df_copy


def add_vet_specific_methods(processor):
    """Add veterinary-specific methods to the processor instance."""
    import types
    
    logger = logging.getLogger(__name__)
    logger.info("=== APPLYING VET-SPECIFIC FUNCTIONS ===")
    
    try:
        # Store original methods
        processor._original_process_job_roles = processor.process_job_roles
        processor._original_fill_event_specific_practice_types = processor.fill_event_specific_practice_types
        
        # Add vet-specific methods
        processor.apply_vet_specific_practice_filling = types.MethodType(
            apply_vet_specific_practice_filling, processor
        )
        
        # Override job role processing with vet-specific logic
        processor.process_job_roles = types.MethodType(
            apply_vet_specific_job_roles, processor
        )
        
        # Override practice filling
        def vet_fill_practice_types(self, df_reg_demo_this, df_reg_demo_last_bva, df_reg_demo_last_lva, practices):
            return self.apply_vet_specific_practice_filling(df_reg_demo_this, df_reg_demo_last_bva, df_reg_demo_last_lva, practices)
        
        processor.fill_event_specific_practice_types = types.MethodType(
            vet_fill_practice_types, processor
        )
        
        # Set flags
        processor._vet_specific_active = True
        processor._event_type = "veterinary"
        
        logger.info("*** VET-SPECIFIC FUNCTIONS SUCCESSFULLY APPLIED ***")
        processor.logger.info("*** VET-SPECIFIC FUNCTIONS ACTIVE ***")
        
    except Exception as e:
        logger.error(f"Error applying vet-specific functions: {e}")
        raise


def verify_vet_functions_applied(processor):
    """Verify that vet-specific functions have been applied correctly."""
    checks = [
        hasattr(processor, '_vet_specific_active') and processor._vet_specific_active,
        hasattr(processor, 'apply_vet_specific_practice_filling'),
        hasattr(processor, '_original_process_job_roles'),
        hasattr(processor, '_event_type') and processor._event_type == "veterinary"
    ]
    return all(checks)


# ---------------------------------------------------------------------------
# Session recommendation custom rules
# ---------------------------------------------------------------------------

def _normalise_keywords(values: List[str]) -> List[str]:
    return [str(value).lower() for value in values if value]


def _coalesce_text_fields(source: Dict[str, Any], fields: List[str]) -> str:
    parts: List[str] = []
    for field in fields:
        value = source.get(field, "") if isinstance(source, dict) else ""
        if value is None:
            continue
        parts.append(str(value))
    return " ".join(parts).lower()


def _contains_any(text: str, keywords: List[str]) -> bool:
    if not text or not keywords:
        return False
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in keywords if keyword)


def vet_equine_session_requires_equine_keywords(
    visitor: Dict[str, Any],
    sessions: List[Dict[str, Any]],
    params: Dict[str, Any],
    context: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Keep equine sessions only when visitor company/job title confirms equine focus."""

    stream_keywords = _normalise_keywords(
        params.get("stream_keywords", ["equine"])
    )
    required_keywords = _normalise_keywords(
        params.get("required_keywords", ["horse", "equine", "mixed"])
    )
    visitor_fields = params.get("visitor_fields", ["Company", "JobTitle"])

    visitor_text = _coalesce_text_fields(visitor, visitor_fields)
    matches_required = _contains_any(visitor_text, required_keywords)

    if matches_required or not stream_keywords:
        return sessions, {"removed_sessions": [], "notes": []}

    kept: List[Dict[str, Any]] = []
    removed: List[Dict[str, Any]] = []

    for session in sessions:
        stream_value = str(session.get("stream", "")).lower()
        if stream_value and _contains_any(stream_value, stream_keywords):
            removed.append(session)
        else:
            kept.append(session)

    notes: List[str] = []
    if removed:
        identifiers = [
            str(entry.get("session_id") or entry.get("title") or "")
            for entry in removed
        ]
        identifiers = [identifier for identifier in identifiers if identifier]
        message = (
            "Custom rule vet_equine_session_requires_equine_keywords removed "
            f"{len(removed)} equine session(s)"
        )
        if identifiers:
            message += f": {', '.join(identifiers)}"
        notes.append(message)

    return kept, {"removed_sessions": removed, "notes": notes}


def vet_equine_visitors_block_feline_sessions(
    visitor: Dict[str, Any],
    sessions: List[Dict[str, Any]],
    params: Dict[str, Any],
    context: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Avoid recommending feline-focused sessions to clearly equine-focused visitors."""

    visitor_fields = params.get("visitor_fields", ["Company", "JobTitle"])
    visitor_keywords = _normalise_keywords(params.get("visitor_keywords", ["equine", "horse"]))
    blocked_keywords = _normalise_keywords(
        params.get("blocked_session_title_keywords", ["cat", "dog", "feline"])
    )

    visitor_text = _coalesce_text_fields(visitor, visitor_fields)
    if not _contains_any(visitor_text, visitor_keywords):
        return sessions, {"removed_sessions": [], "notes": []}

    kept: List[Dict[str, Any]] = []
    removed: List[Dict[str, Any]] = []

    for session in sessions:
        title_value = str(
            session.get("title")
            or session.get("session_title")
            or session.get("name")
            or ""
        ).lower()
        if title_value and _contains_any(title_value, blocked_keywords):
            removed.append(session)
        else:
            kept.append(session)

    notes: List[str] = []
    if removed:
        identifiers = [
            str(entry.get("session_id") or entry.get("title") or "")
            for entry in removed
        ]
        identifiers = [identifier for identifier in identifiers if identifier]
        message = (
            "Custom rule vet_equine_visitors_block_feline_sessions removed "
            f"{len(removed)} session(s)"
        )
        if identifiers:
            message += f": {', '.join(identifiers)}"
        notes.append(message)

    return kept, {"removed_sessions": removed, "notes": notes}


VET_CUSTOM_RECOMMENDATION_RULES = {
    "vet_equine_session_requires_equine_keywords": vet_equine_session_requires_equine_keywords,
    "vet_equine_visitors_block_feline_sessions": vet_equine_visitors_block_feline_sessions,
}


def apply_vet_custom_recommendation_rules(
    visitor: Dict[str, Any],
    sessions: List[Dict[str, Any]],
    processor,
    rules_config: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Apply configured vet-specific custom recommendation rules sequentially."""

    logger = getattr(processor, "logger", logging.getLogger(__name__))
    current_sessions = list(sessions)
    combined_notes: List[str] = []
    rule_results: List[Dict[str, Any]] = []

    for rule in rules_config.get("rules", []):
        if not isinstance(rule, dict):
            continue

        if not rule.get("enabled", True):
            continue

        rule_name = rule.get("name")
        handler = VET_CUSTOM_RECOMMENDATION_RULES.get(rule_name)
        if handler is None:
            logger.warning("Unknown vet custom recommendation rule '%s'", rule_name)
            continue

        params = rule.get("params", {})
        filtered_sessions, outcome = handler(
            visitor,
            current_sessions,
            params,
            {"processor": processor, "logger": logger},
        )

        if outcome is None:
            outcome = {}

        removed_sessions = outcome.get("removed_sessions", [])
        notes = outcome.get("notes", [])

        if notes:
            combined_notes.extend(notes)

        rule_results.append(
            {
                "name": rule_name,
                "enabled": True,
                "removed_count": len(removed_sessions),
                "removed_sessions": [
                    {
                        "session_id": entry.get("session_id"),
                        "title": entry.get("title"),
                        "stream": entry.get("stream"),
                    }
                    for entry in removed_sessions
                ],
            }
        )

        current_sessions = filtered_sessions if filtered_sessions is not None else current_sessions

    return current_sessions, {"rule_results": rule_results, "notes": combined_notes}