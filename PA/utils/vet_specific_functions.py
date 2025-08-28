"""
Veterinary-specific processing functions - Simplified Version
This module contains functions that are specific to veterinary events.
"""

import logging

def apply_vet_specific_practice_filling(self, df_reg_demo_this, df_reg_demo_last_bva, df_reg_demo_last_lva, practices):
    """Apply veterinary-specific practice type filling logic."""
    self.logger.info("Applying veterinary-specific practice type filling logic")
    
    # Veterinary-specific column names (exactly as in old processor)
    this_year_col = "what_type_does_your_practice_specialise_in"
    past_year_col = "what_areas_do_you_specialise_in"

    # Fill missing practice types using vet-specific columns
    df_reg_demo_this = self.fill_missing_practice_types(
        df_reg_demo_this, practices, column=this_year_col
    )
    df_reg_demo_last_bva = self.fill_missing_practice_types(
        df_reg_demo_last_bva, practices, column=past_year_col
    )
    df_reg_demo_last_lva = self.fill_missing_practice_types(
        df_reg_demo_last_lva, practices, column=past_year_col
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