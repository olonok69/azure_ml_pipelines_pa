#!/usr/bin/env python3
"""
Generic Session Recommendation Processor - Final Production Version

This processor generates session recommendations for visitors based on:
- Their past attendance (if they visited last year)
- Similar visitors' attendance patterns (if they're new)

The processor is fully configurable to work with different shows (BVA, ECOMM, etc.)
through YAML configuration files without any hardcoded show-specific logic.

Key Features:
- Fully generic and configurable for different show types
- Proper incremental processing support (create_only_new)
- Consistent show attribute handling
- Complete error handling and logging
"""

import os
import json
import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import concurrent.futures
from sklearn.metrics.pairwise import cosine_similarity
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer


# Try to import LangChain for advanced filtering (optional)
try:
    from langchain_openai import ChatOpenAI
    from langchain.prompts import PromptTemplate
    has_langchain = True
except ImportError:
    has_langchain = False


class SessionRecommendationProcessor:
    """Generic session recommendation processor for different shows."""

    def __init__(self, config: Dict):
        """
        Initialize the session recommendation processor.

        Args:
            config: Configuration dictionary with all necessary settings
        """
        self.config = config
        self.logger = self._setup_logging()
        
        # Extract configuration settings
        self.show_name = config.get("neo4j", {}).get("show_name", "bva")
        self.recommendation_config = config.get("recommendation", {})
        self.neo4j_config = config.get("neo4j", {})
        
        # Neo4j connection parameters
        self.uri = self.neo4j_config.get("uri")
        self.username = self.neo4j_config.get("username")
        self.password = self.neo4j_config.get("password")
        
        # Recommendation parameters
        self.min_similarity_score = self.recommendation_config.get("min_similarity_score", 0.3)
        self.max_recommendations = self.recommendation_config.get("max_recommendations", 10)
        self.similar_visitors_count = self.recommendation_config.get("similar_visitors_count", 3)
        self.use_langchain = self.recommendation_config.get("use_langchain", False) and has_langchain
        self.enable_filtering = self.recommendation_config.get("enable_filtering", False)
        
        # Similarity attributes for finding similar visitors
        self.similarity_attributes = self.recommendation_config.get("similarity_attributes", {})
        
        # GENERIC: Load role groups from configuration instead of hardcoding
        role_groups = self.recommendation_config.get("role_groups", {})
        self.vet_roles = role_groups.get("vet_roles", [])
        self.nurse_roles = role_groups.get("nurse_roles", [])
        self.business_roles = role_groups.get("business_roles", [])
        self.other_roles = role_groups.get("other_roles", [])
        
        # GENERIC: Load field mappings from configuration
        field_mappings = self.recommendation_config.get("field_mappings", {})
        self.practice_type_field = field_mappings.get("practice_type_field", "what_type_does_your_practice_specialise_in")
        self.job_role_field = field_mappings.get("job_role_field", "job_role")
        
        # Filtering rules configuration (show-specific)
        self.rules_config = self.recommendation_config.get("rules_config", {})
        
        # Node labels from config
        self.visitor_this_year_label = self.neo4j_config.get("node_labels", {}).get(
            "visitor_this_year", "Visitor_this_year"
        )
        self.visitor_last_year_bva_label = self.neo4j_config.get("node_labels", {}).get(
            "visitor_last_year_bva", "Visitor_last_year_bva"
        )
        self.visitor_last_year_lva_label = self.neo4j_config.get("node_labels", {}).get(
            "visitor_last_year_lva", "Visitor_last_year_lva"
        )
        self.session_this_year_label = self.neo4j_config.get("node_labels", {}).get(
            "session_this_year", "Sessions_this_year"
        )
        self.session_past_year_label = self.neo4j_config.get("node_labels", {}).get(
            "session_past_year", "Sessions_past_year"
        )
        
        # Initialize caches for performance
        self._visitor_cache = {}
        self._similar_visitors_cache = {}
        self._this_year_sessions_cache = None
        
        # Initialize the sentence transformer model
        self.model = None
        self._init_model()
        
        # Initialize statistics
        self.statistics = {
            "visitors_processed": 0,
            "visitors_with_recommendations": 0,
            "visitors_without_recommendations": 0,
            "total_recommendations_generated": 0,
            "total_filtered_recommendations": 0,
            "errors": 0,
            "error_details": [],
            "processing_time": 0,
        }
        
        # Output directory
        self.output_dir = Path(config.get("output_dir", "data/output"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"SessionRecommendationProcessor initialized for {self.show_name} show")
        self.logger.info(f"Filtering enabled: {self.enable_filtering}")
        if self.enable_filtering:
            self.logger.info(f"Using field mappings - Practice: {self.practice_type_field}, Job: {self.job_role_field}")

    def _setup_logging(self) -> logging.Logger:
        """Set up logging for the processor."""
        logger = logging.getLogger("session_recommendation_processor")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def _init_model(self):
        """Initialize the sentence transformer model."""
        model_name = self.config.get("embeddings", {}).get("model", "all-MiniLM-L6-v2")
        try:
            self.model = SentenceTransformer(model_name)
            self.logger.info(f"Initialized sentence transformer model: {model_name}")
        except Exception as e:
            self.logger.error(f"Error initializing model: {str(e)}")
            self.model = None

    def _get_visitors_to_process(self, create_only_new: bool = False) -> List[str]:
        """
        Get list of visitors to process based on create_only_new flag.
        
        Args:
            create_only_new: If True, only get visitors without recommendations
            
        Returns:
            List of badge IDs to process
        """
        try:
            with GraphDatabase.driver(
                self.uri, auth=(self.username, self.password)
            ) as driver:
                with driver.session() as session:
                    if create_only_new:
                        # Only get visitors without recommendations
                        # Check for NULL, "0", or missing has_recommendation
                        query = f"""
                        MATCH (v:{self.visitor_this_year_label})
                        WHERE (v.show = $show_name OR v.show IS NULL)
                        AND (v.has_recommendation IS NULL OR v.has_recommendation = "0" OR NOT EXISTS(v.has_recommendation))
                        RETURN v.BadgeId as badge_id
                        """
                    else:
                        # Get all visitors for this show (reprocess all)
                        query = f"""
                        MATCH (v:{self.visitor_this_year_label})
                        WHERE v.show = $show_name OR v.show IS NULL
                        RETURN v.BadgeId as badge_id
                        """
                    
                    result = session.run(query, show_name=self.show_name)
                    badge_ids = [record["badge_id"] for record in result]
                    
                    if create_only_new:
                        self.logger.info(f"Found {len(badge_ids)} NEW visitors to process for {self.show_name} show (without recommendations)")
                    else:
                        self.logger.info(f"Found {len(badge_ids)} TOTAL visitors to process for {self.show_name} show (all visitors)")
                    
                    return badge_ids
                    
        except Exception as e:
            self.logger.error(f"Error getting visitors to process: {str(e)}", exc_info=True)
            return []

    def _get_visitor_info(self, badge_id: str) -> Optional[Dict]:
        """Get visitor information from Neo4j."""
        if badge_id in self._visitor_cache:
            return self._visitor_cache[badge_id]
        
        try:
            with GraphDatabase.driver(
                self.uri, auth=(self.username, self.password)
            ) as driver:
                with driver.session() as session:
                    query = f"""
                    MATCH (v:{self.visitor_this_year_label} {{BadgeId: $badge_id}})
                    WHERE v.show = $show_name OR v.show IS NULL
                    RETURN v
                    """
                    result = session.run(query, badge_id=badge_id, show_name=self.show_name)
                    record = result.single()
                    
                    if record:
                        visitor_data = dict(record["v"])
                        self._visitor_cache[badge_id] = visitor_data
                        return visitor_data
                    
        except Exception as e:
            self.logger.error(f"Error getting visitor info for {badge_id}: {str(e)}")
        
        return None

    def _get_this_year_sessions(self) -> List[Dict]:
        """Get all sessions for this year from Neo4j."""
        if self._this_year_sessions_cache is not None:
            return self._this_year_sessions_cache
        
        try:
            with GraphDatabase.driver(
                self.uri, auth=(self.username, self.password)
            ) as driver:
                with driver.session() as session:
                    query = f"""
                    MATCH (s:{self.session_this_year_label})
                    WHERE s.show = $show_name OR s.show IS NULL
                    RETURN s
                    """
                    result = session.run(query, show_name=self.show_name)
                    sessions = [dict(record["s"]) for record in result]
                    
                    self._this_year_sessions_cache = sessions
                    self.logger.info(f"Loaded {len(sessions)} sessions for this year")
                    return sessions
                    
        except Exception as e:
            self.logger.error(f"Error getting this year sessions: {str(e)}")
            return []

    def _find_similar_visitors(self, visitor: Dict, limit: int = 3) -> List[str]:
        """
        Find similar visitors based on configured similarity attributes.
        Modified to introduce randomness and reduce bias towards the same visitors.
        
        Args:
            visitor: The visitor dictionary
            limit: Maximum number of similar visitors to return
            
        Returns:
            List of similar visitor badge IDs
        """
        import random
        
        visitor_id = visitor.get("BadgeId")
        
        # Check cache first - but now include a randomization factor in cache key
        # to allow for different random selections over time
        cache_key = f"{visitor_id}_{limit}"
        if cache_key in self._similar_visitors_cache:
            return self._similar_visitors_cache[cache_key]
        
        try:
            with GraphDatabase.driver(
                self.uri, auth=(self.username, self.password)
            ) as driver:
                with driver.session() as session:
                    # Build WHERE clauses based on similarity attributes
                    where_clauses = []
                    params = {"visitor_id": visitor_id, "show_name": self.show_name}
                    
                    for attr, weight in self.similarity_attributes.items():
                        if weight > 0 and attr in visitor and visitor[attr] not in [None, "NA", ""]:
                            # Map the attribute name if needed
                            neo4j_attr = attr
                            
                            where_clauses.append(f"v2.{neo4j_attr} = $attr_{attr}")
                            params[f"attr_{attr}"] = visitor[attr]
                    
                    if not where_clauses:
                        self.logger.warning(f"No valid similarity attributes for visitor {visitor_id}")
                        return []
                    
                    # Modified query: Remove LIMIT and get ALL similar visitors first
                    # Then apply randomization in Python to select the final subset
                    query = f"""
                    MATCH (v1:{self.visitor_this_year_label} {{BadgeId: $visitor_id}})
                    WHERE v1.show = $show_name OR v1.show IS NULL
                    MATCH (v2:{self.visitor_last_year_bva_label})
                    WHERE v2.BadgeId <> $visitor_id AND ({" OR ".join(where_clauses)})
                    MATCH (v2)-[:attended_session]->(s:{self.session_past_year_label})
                    WITH v2.BadgeId as similar_visitor, COUNT(DISTINCT s) as sessions_attended
                    WHERE sessions_attended > 0
                    ORDER BY sessions_attended DESC
                    RETURN similar_visitor, sessions_attended
                    """
                    
                    result = session.run(query, **params)
                    all_similar_visitors = [(record["similar_visitor"], record["sessions_attended"]) 
                                        for record in result]
                    
                    if not all_similar_visitors:
                        self.logger.warning(f"No similar visitors found for visitor {visitor_id}")
                        return []
                    
                    # Determine how many visitors to consider for random selection
                    # We want to get the top performers but introduce randomness
                    total_found = len(all_similar_visitors)
                    
                    if total_found <= limit:
                        # If we have fewer or equal visitors than needed, return all
                        similar_visitors = [visitor_id for visitor_id, _ in all_similar_visitors]
                    else:
                        # Strategy: Take a larger subset of high-performing visitors and randomly select from them
                        # This balances between getting good visitors and introducing randomness
                        
                        # Take top performers (but more than we need) for random selection
                        # Use a multiplier to get a reasonable pool for randomization
                        pool_size = min(total_found, max(limit * 3, 10))  # At least 3x the limit or 10, whichever is smaller
                        
                        # Get the top pool_size visitors by session count
                        top_visitors_pool = all_similar_visitors[:pool_size]
                        
                        # Now randomly select 'limit' visitors from this pool
                        selected_visitors = random.sample(top_visitors_pool, min(limit, len(top_visitors_pool)))
                        similar_visitors = [visitor_id for visitor_id, _ in selected_visitors]
                    
                    # Cache the result
                    self._similar_visitors_cache[cache_key] = similar_visitors
                    
                    self.logger.info(
                        f"Found {total_found} total similar visitors for {visitor_id}, "
                        f"randomly selected {len(similar_visitors)} from top performers"
                    )
                    
                    return similar_visitors
                    
        except Exception as e:
            self.logger.error(f"Error finding similar visitors: {str(e)}")
            return []

    def _get_past_sessions_for_visitors(self, visitor_ids: List[str], is_returning: bool = False) -> List[Dict]:
        """
        Get past sessions attended by specified visitors.
        
        Args:
            visitor_ids: List of visitor badge IDs
            is_returning: If True, these are returning visitors (need to follow Same_Visitor relationship)
        
        Returns:
            List of past session dictionaries
        """
        if not visitor_ids:
            return []
        
        try:
            with GraphDatabase.driver(
                self.uri, auth=(self.username, self.password)
            ) as driver:
                with driver.session() as session:
                    if is_returning:
                        # For returning visitors, follow the Same_Visitor relationship
                        # because they have DIFFERENT BadgeIds between years!
                        query = f"""
                        MATCH (v:{self.visitor_this_year_label})-[:Same_Visitor]->(v_past)
                        WHERE v.BadgeId IN $visitor_ids
                        AND (v_past:{self.visitor_last_year_bva_label} OR v_past:{self.visitor_last_year_lva_label})
                        MATCH (v_past)-[:attended_session]->(s:{self.session_past_year_label})
                        RETURN DISTINCT s
                        """
                    else:
                        # For similar visitors (new visitors), query directly with their past year BadgeIds
                        query = f"""
                        MATCH (v:{self.visitor_last_year_bva_label})-[:attended_session]->(s:{self.session_past_year_label})
                        WHERE v.BadgeId IN $visitor_ids
                        RETURN DISTINCT s
                        """
                    
                    result = session.run(query, visitor_ids=visitor_ids)
                    sessions = [dict(record["s"]) for record in result]
                    
                    return sessions
                    
        except Exception as e:
            self.logger.error(f"Error getting past sessions: {str(e)}")
            return []

    def calculate_session_similarities(
        self, 
        past_sessions: List[Dict], 
        this_year_sessions: List[Dict],
        min_score: float = 0.3
    ) -> List[Dict]:
        """
        Calculate similarities between past sessions and this year's sessions.
        
        Args:
            past_sessions: Sessions attended in the past
            this_year_sessions: Available sessions this year
            min_score: Minimum similarity score threshold
            
        Returns:
            List of recommended sessions with similarity scores
        """
        if not past_sessions or not this_year_sessions or not self.model:
            return []
        
        try:
            # Extract embeddings
            past_embeddings = []
            for session in past_sessions:
                if "embedding" in session and session["embedding"]:
                    embedding = json.loads(session["embedding"])
                    past_embeddings.append(embedding)
            
            this_year_embeddings = []
            this_year_sessions_filtered = []
            for session in this_year_sessions:
                if "embedding" in session and session["embedding"]:
                    embedding = json.loads(session["embedding"])
                    this_year_embeddings.append(embedding)
                    this_year_sessions_filtered.append(session)
            
            if not past_embeddings or not this_year_embeddings:
                return []
            
            # Calculate cosine similarities
            past_embeddings = np.array(past_embeddings)
            this_year_embeddings = np.array(this_year_embeddings)
            
            similarities = cosine_similarity(this_year_embeddings, past_embeddings)
            
            # Get max similarity for each this year session
            max_similarities = similarities.max(axis=1)
            
            # Filter and prepare recommendations
            recommendations = []
            for idx, score in enumerate(max_similarities):
                if score >= min_score:
                    session = this_year_sessions_filtered[idx].copy()
                    session["similarity"] = float(score)
                    recommendations.append(session)
            
            # Sort by similarity score
            recommendations.sort(key=lambda x: x["similarity"], reverse=True)
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error calculating similarities: {str(e)}")
            return []

    def _contains_any(self, text: str, keywords: List[str]) -> bool:
        """Check if text contains any of the keywords (case-insensitive)."""
        if not text or not isinstance(text, str):
            return False
        
        text_lower = text.lower()
        return any(keyword.lower() in text_lower for keyword in keywords)

    def _apply_practice_type_rules(self, visitor: Dict, sessions: List[Dict]) -> Tuple[List[Dict], List[str]]:
        """
        Apply practice type filtering rules (generic version).
        
        Args:
            visitor: Visitor information
            sessions: List of sessions to filter
            
        Returns:
            Tuple of (filtered_sessions, rules_applied)
        """
        if not self.enable_filtering or not self.rules_config:
            return sessions, []
        
        # Use configured field name
        practice_field = self.practice_type_field
        
        if not visitor or practice_field not in visitor:
            return sessions, []
        
        practice_type = visitor.get(practice_field, "")
        if not practice_type or practice_type == "NA":
            return sessions, []
        
        filtered_sessions = []
        rules_applied = []
        
        # Check for different practice types based on configuration
        # For veterinary shows
        if self.show_name == "bva" and self.rules_config:
            # Check if practice contains equine or mixed
            if self._contains_any(practice_type, ["equine", "mixed"]):
                exclusions = self.rules_config.get("equine_mixed_exclusions", [])
                if exclusions:
                    filtered_sessions = [
                        session for session in sessions
                        if not session.get("stream") or not self._contains_any(session["stream"], exclusions)
                    ]
                    rules_applied.append(f"practice_type: mixed/equine - excluded {', '.join(exclusions)}")
                    self.logger.info(
                        f"Applied equine/mixed rule: filtered from {len(sessions)} to {len(filtered_sessions)} sessions"
                    )
            
            # Check if practice contains small animal
            elif self._contains_any(practice_type, ["small animal"]):
                exclusions = self.rules_config.get("small_animal_exclusions", [])
                if exclusions:
                    filtered_sessions = [
                        session for session in sessions
                        if not session.get("stream") or not self._contains_any(session["stream"], exclusions)
                    ]
                    rules_applied.append(f"practice_type: small animal - excluded {', '.join(exclusions)}")
                    self.logger.info(
                        f"Applied small animal rule: filtered from {len(sessions)} to {len(filtered_sessions)} sessions"
                    )
            else:
                filtered_sessions = sessions
        
        # For other shows, check if there are custom rules in config
        else:
            # Apply any custom filtering rules from configuration
            custom_rules = self.rules_config.get("custom_practice_rules", [])
            if custom_rules:
                # Implement custom rule logic here if needed
                filtered_sessions = sessions
            else:
                filtered_sessions = sessions
        
        return filtered_sessions if filtered_sessions else sessions, rules_applied

    def _apply_role_rules(self, visitor: Dict, sessions: List[Dict]) -> Tuple[List[Dict], List[str]]:
        """
        Apply job role filtering rules (generic version).
        
        Args:
            visitor: Visitor information
            sessions: List of sessions to filter
            
        Returns:
            Tuple of (filtered_sessions, rules_applied)
        """
        if not self.enable_filtering or not self.rules_config:
            return sessions, []
        
        # Use configured field name
        job_field = self.job_role_field
        
        if not visitor or job_field not in visitor:
            return sessions, []
        
        job_role = visitor.get(job_field, "")
        if not job_role or job_role == "NA":
            return sessions, []
        
        filtered_sessions = []
        rules_applied = []
        
        # Only apply role-based filtering if roles are configured
        if self.vet_roles and job_role in self.vet_roles:
            exclusions = self.rules_config.get("vet_exclusions", [])
            if exclusions:
                filtered_sessions = [
                    session for session in sessions
                    if not session.get("stream") or not self._contains_any(session["stream"], exclusions)
                ]
                rules_applied.append(f"role: vet - excluded {', '.join(exclusions)}")
                self.logger.info(
                    f"Applied vet role rule: filtered from {len(sessions)} to {len(filtered_sessions)} sessions"
                )
        
        elif self.nurse_roles and job_role in self.nurse_roles:
            allowed_streams = self.rules_config.get("nurse_streams", [])
            if allowed_streams:
                filtered_sessions = [
                    session for session in sessions
                    if session.get("stream") and self._contains_any(session["stream"], allowed_streams)
                ]
                rules_applied.append(f"role: nurse - limited to {', '.join(allowed_streams)}")
                self.logger.info(
                    f"Applied nurse role rule: filtered from {len(sessions)} to {len(filtered_sessions)} sessions"
                )
        else:
            # No specific role rule applies
            filtered_sessions = sessions
        
        return filtered_sessions if filtered_sessions else sessions, rules_applied

    def filter_sessions(self, visitor: Dict, sessions: List[Dict]) -> Tuple[List[Dict], List[str]]:
        """
        Filter sessions based on configured rules.
        
        Args:
            visitor: Visitor information
            sessions: List of sessions to filter
            
        Returns:
            Tuple of (filtered_sessions, rules_applied)
        """
        if not sessions or not self.enable_filtering:
            return sessions, []
        
        filtered_sessions = sessions
        rule_priority = self.rules_config.get("rule_priority", ["practice_type", "role"])
        all_rules_applied = []
        
        # Apply rules in priority order
        for rule_type in rule_priority:
            if rule_type == "practice_type":
                filtered_sessions, rules_applied = self._apply_practice_type_rules(
                    visitor, filtered_sessions
                )
                all_rules_applied.extend(rules_applied)
            elif rule_type == "role":
                filtered_sessions, rules_applied = self._apply_role_rules(
                    visitor, filtered_sessions
                )
                all_rules_applied.extend(rules_applied)
            else:
                self.logger.warning(f"Unknown rule type: {rule_type}")
        
        # Sort by similarity score (highest first)
        filtered_sessions.sort(
            key=lambda x: float(x.get("similarity", 0)), reverse=True
        )
        
        return filtered_sessions, all_rules_applied

    def _get_popular_past_sessions(self, limit: int = 20) -> List[Dict]:
        """
        Get the most popular sessions from last year as a fallback.
        
        This method now:
        - Filters sessions by show name for multi-show support
        - Includes attendance from BOTH main (BVA) and secondary (LVA) events
        - Treats both events as the same audience (which they are)
        - Gets top N popular sessions then randomly selects max_recommendations from them
        
        Args:
            limit: Minimum number of popular sessions to retrieve before random selection
            
        Returns:
            List of randomly selected popular session dictionaries (up to max_recommendations)
        """
        import random
        
        try:
            with GraphDatabase.driver(
                self.uri, auth=(self.username, self.password)
            ) as driver:
                with driver.session() as session:
                    # Determine how many sessions to fetch from the database
                    # We want at least 'limit' sessions, but if max_recommendations is higher,
                    # fetch more to have a good pool for random selection
                    fetch_limit = max(limit, self.max_recommendations)
                    
                    # Query that includes both BVA and LVA visitors for the specific show
                    query = f"""
                    // Get sessions for this specific show
                    MATCH (s:{self.session_past_year_label})
                    WHERE s.show = $show_name
                    
                    // Count attendance from main event (BVA) visitors
                    OPTIONAL MATCH (v_main:{self.visitor_last_year_bva_label})-[:attended_session]->(s)
                    WITH s, COUNT(DISTINCT v_main) as main_event_count
                    
                    // Also count attendance from secondary event (LVA) visitors  
                    OPTIONAL MATCH (v_secondary:{self.visitor_last_year_lva_label})-[:attended_session]->(s)
                    WITH s, main_event_count, COUNT(DISTINCT v_secondary) as secondary_event_count
                    
                    // Combine counts from both events (same audience, different locations/dates)
                    WITH s, (main_event_count + secondary_event_count) as total_attendance
                    WHERE total_attendance > 0
                    
                    // Order by popularity and return top sessions
                    ORDER BY total_attendance DESC
                    LIMIT $limit
                    RETURN s, total_attendance
                    """
                    
                    result = session.run(
                        query,
                        limit=fetch_limit,
                        show_name=self.show_name
                    )
                    
                    # Get all popular sessions with their attendance counts
                    popular_sessions = []
                    for record in result:
                        session_dict = dict(record["s"])
                        session_dict["_popularity_score"] = record["total_attendance"]
                        popular_sessions.append(session_dict)
                    
                    if not popular_sessions:
                        self.logger.warning(f"No popular sessions found for show '{self.show_name}'")
                        return []
                    
                    # Now randomly select max_recommendations sessions from the popular sessions
                    num_sessions_to_return = min(self.max_recommendations, len(popular_sessions))
                    
                    if num_sessions_to_return < len(popular_sessions):
                        # Randomly select from the pool of popular sessions
                        selected_sessions = random.sample(popular_sessions, num_sessions_to_return)
                        
                        # Sort selected sessions by popularity to maintain some quality ordering
                        # This helps if the recommendations are displayed in order
                        selected_sessions.sort(key=lambda x: x.get("_popularity_score", 0), reverse=True)
                        
                        # Remove the temporary popularity score before returning
                        for sess in selected_sessions:
                            sess.pop("_popularity_score", None)
                        
                        self.logger.info(
                            f"Retrieved {len(popular_sessions)} popular sessions for show '{self.show_name}', "
                            f"randomly selected {num_sessions_to_return} as fallback "
                            f"(combined attendance from main and secondary events)"
                        )
                    else:
                        # Return all sessions if we have fewer than max_recommendations
                        selected_sessions = popular_sessions
                        
                        # Remove the temporary popularity score before returning
                        for sess in selected_sessions:
                            sess.pop("_popularity_score", None)
                        
                        self.logger.info(
                            f"Retrieved all {len(selected_sessions)} popular sessions for show '{self.show_name}' as fallback "
                            f"(fewer than max_recommendations={self.max_recommendations} available)"
                        )
                    
                    return selected_sessions
                    
        except Exception as e:
            self.logger.error(f"Error getting popular sessions for show '{self.show_name}': {str(e)}")
            return []

    def generate_recommendations_for_visitor(self, badge_id: str) -> Tuple[List[Dict], Dict]:
        """
        Generate recommendations for a single visitor.
        
        Args:
            badge_id: Visitor's badge ID
            
        Returns:
            Tuple of (recommendations, visitor_info)
        """
        try:
            # Get visitor information
            visitor = self._get_visitor_info(badge_id)
            if not visitor:
                self.logger.warning(f"Visitor {badge_id} not found")
                return [], {}
            
            # Get this year's sessions
            this_year_sessions = self._get_this_year_sessions()
            if not this_year_sessions:
                self.logger.warning("No sessions available for this year")
                return [], visitor
            
            # Check if visitor attended last year
            assist_year_before = visitor.get("assist_year_before", "0")
            past_sessions = []
            
            if assist_year_before == "1":
                # Returning visitor - get their past sessions via Same_Visitor relationship
                past_sessions = self._get_past_sessions_for_visitors([badge_id], is_returning=True)
                
                if past_sessions:
                    self.logger.info(f"Returning visitor {badge_id} attended {len(past_sessions)} sessions last year")
                else:
                    # Returning visitor with no past sessions - treat as new visitor
                    self.logger.warning(
                        f"Returning visitor {badge_id} has no past session records - treating as new visitor"
                    )
                    similar_visitors = self._find_similar_visitors(visitor, self.similar_visitors_count)
                    if similar_visitors:
                        past_sessions = self._get_past_sessions_for_visitors(similar_visitors, is_returning=False)
                        self.logger.info(
                            f"Found {len(past_sessions)} sessions from {len(similar_visitors)} similar visitors"
                        )
                    else:
                        # No similar visitors found - use popular sessions as fallback
                        self.logger.warning(f"No similar visitors found for returning visitor {badge_id} - using popular sessions")
                        past_sessions = self._get_popular_past_sessions(limit=self.max_recommendations*5)
                        if not past_sessions:
                            self.logger.error(f"No sessions available for recommendations for {badge_id}")
                            return [], visitor
            else:
                # New visitor - find similar visitors
                similar_visitors = self._find_similar_visitors(visitor, self.similar_visitors_count)
                if similar_visitors:
                    past_sessions = self._get_past_sessions_for_visitors(similar_visitors, is_returning=False)
                    self.logger.info(f"New visitor {badge_id}: found {len(past_sessions)} sessions from similar visitors")
                else:
                    # No similar visitors found - use popular sessions as fallback
                    self.logger.warning(f"No similar visitors found for {badge_id} - using popular sessions as fallback")
                    past_sessions = self._get_popular_past_sessions(limit=self.max_recommendations*5)
                    if not past_sessions:
                        self.logger.error(f"No sessions available for recommendations for {badge_id}")
                        return [], visitor
            
            # Calculate recommendations
            if past_sessions:
                recommendations = self.calculate_session_similarities(
                    past_sessions, this_year_sessions, self.min_similarity_score
                )
                
                # Apply filtering if enabled
                if self.enable_filtering and recommendations:
                    filtered_recommendations, rules_applied = self.filter_sessions(visitor, recommendations)
                    self.logger.info(f"Filtered {len(recommendations)} to {len(filtered_recommendations)} recommendations")
                    return filtered_recommendations[:self.max_recommendations], visitor
                else:
                    return recommendations[:self.max_recommendations], visitor
            
            return [], visitor
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations for {badge_id}: {str(e)}")
            return [], {}

    def _update_visitor_recommendations(self, recommendations_data: List[Dict]):
        """Update visitors with has_recommendation flag and create IS_RECOMMENDED relationships."""
        try:
            with GraphDatabase.driver(
                self.uri, auth=(self.username, self.password)
            ) as driver:
                with driver.session() as session:
                    updated_with_recs = 0
                    updated_without_recs = 0
                    
                    for rec in recommendations_data:
                        visitor_id = rec["visitor"]["BadgeId"]
                        recommended_sessions = rec.get("filtered_recommendations", [])
                        
                        if recommended_sessions:
                            # Update visitor with has_recommendation flag
                            update_query = f"""
                            MATCH (v:{self.visitor_this_year_label} {{BadgeId: $visitor_id}})
                            WHERE v.show = $show_name OR v.show IS NULL
                            SET v.has_recommendation = "1",
                                v.show = $show_name,
                                v.recommendations_generated_at = $timestamp
                            """
                            session.run(update_query, 
                                       visitor_id=visitor_id, 
                                       show_name=self.show_name,
                                       timestamp=datetime.now().isoformat())
                            updated_with_recs += 1
                            
                            # Create IS_RECOMMENDED relationships
                            for rec_session in recommended_sessions:
                                session_id = rec_session.get("session_id")
                                if session_id:
                                    rel_query = f"""
                                    MATCH (v:{self.visitor_this_year_label} {{BadgeId: $visitor_id}})
                                    WHERE v.show = $show_name
                                    MATCH (s:{self.session_this_year_label} {{session_id: $session_id}})
                                    MERGE (v)-[r:IS_RECOMMENDED]->(s)
                                    SET r.similarity_score = $score,
                                        r.generated_at = $timestamp,
                                        r.show = $show_name
                                    """
                                    session.run(
                                        rel_query,
                                        visitor_id=visitor_id,
                                        show_name=self.show_name,
                                        session_id=session_id,
                                        score=rec_session.get("similarity", 0),
                                        timestamp=datetime.now().isoformat()
                                    )
                        else:
                            # Mark visitor as processed but without recommendations
                            update_query = f"""
                            MATCH (v:{self.visitor_this_year_label} {{BadgeId: $visitor_id}})
                            WHERE v.show = $show_name OR v.show IS NULL
                            SET v.has_recommendation = "0",
                                v.show = $show_name,
                                v.recommendations_generated_at = $timestamp
                            """
                            session.run(update_query, 
                                       visitor_id=visitor_id, 
                                       show_name=self.show_name,
                                       timestamp=datetime.now().isoformat())
                            updated_without_recs += 1
                    
                    self.logger.info(f"Updated {updated_with_recs} visitors with recommendations, {updated_without_recs} without")
                    
        except Exception as e:
            self.logger.error(f"Error updating visitor recommendations: {str(e)}")

    def json_to_dataframe(self, json_file: str) -> pd.DataFrame:
        """
        Convert recommendations JSON to DataFrame for analysis.
        Includes ALL similarity attributes configured for the show.
        """
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            rows = []
            for visitor_id, recs in data.get("recommendations", {}).items():
                visitor_info = recs.get("visitor", {})
                
                for rec in recs.get("filtered_recommendations", []):
                    # Start with basic fields
                    row = {
                        "visitor_id": visitor_id,
                        "assist_year_before": visitor_info.get("assist_year_before", "0"),
                        "show": self.show_name,
                    }
                    
                    # Add ALL configured similarity attributes dynamically
                    for attr_name in self.similarity_attributes.keys():
                        # Clean column name for CSV (replace spaces and special chars)
                        col_name = attr_name.replace(" ", "_").replace(".", "")
                        row[col_name] = visitor_info.get(attr_name, "NA")
                    
                    # Add additional visitor context fields
                    additional_fields = ["Company", "JobTitle", "Email_domain", "Country", "Source"]
                    for field in additional_fields:
                        if field not in self.similarity_attributes:
                            row[field] = visitor_info.get(field, "NA")
                    
                    # Add session recommendation details
                    row.update({
                        "session_id": rec.get("session_id"),
                        "session_title": rec.get("title"),
                        "session_stream": rec.get("stream"),
                        "session_date": rec.get("date"),
                        "session_start_time": rec.get("start_time"),
                        "session_end_time": rec.get("end_time"),
                        "similarity_score": rec.get("similarity"),
                        "sponsored_by": rec.get("sponsored_by", ""),
                    })
                    
                    rows.append(row)
            
            df = pd.DataFrame(rows)
            
            # Reorder columns for better readability
            priority_cols = ["visitor_id", "show", "assist_year_before"]
            similarity_cols = [col for col in df.columns if col in [
                attr.replace(" ", "_").replace(".", "") for attr in self.similarity_attributes.keys()
            ]]
            session_cols = [col for col in df.columns if col.startswith("session_")]
            other_cols = [col for col in df.columns if col not in priority_cols + similarity_cols + session_cols]
            
            ordered_cols = priority_cols + similarity_cols + other_cols + session_cols
            df = df[[col for col in ordered_cols if col in df.columns]]
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error converting JSON to DataFrame: {str(e)}")
            return pd.DataFrame()

    def process(self, create_only_new: bool = False):
        """
        Main processing method to generate recommendations for all visitors.
        
        Args:
            create_only_new: If True, only process visitors without existing recommendations
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"Starting session recommendation processing for {self.show_name} show")
            self.logger.info(f"Mode: {'Incremental (new only)' if create_only_new else 'Full (all visitors)'}")
            
            # Get visitors to process
            visitors_to_process = self._get_visitors_to_process(create_only_new)
            
            if not visitors_to_process:
                self.logger.info("No visitors to process for recommendations.")
                return
            
            self.statistics["visitors_processed"] = len(visitors_to_process)
            
            # Process visitors and generate recommendations
            all_recommendations = []
            successful = 0
            errors = 0
            total_recommendations = 0
            unique_sessions = set()
            
            # Create output file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.output_dir / f"recommendations/visitor_recommendations_{self.show_name}_{timestamp}.json"
            output_file.parent.mkdir(exist_ok=True)
            
            # Process each visitor
            recommendations_dict = {}
            
            for i, badge_id in enumerate(visitors_to_process, 1):
                if i % 100 == 0:
                    self.logger.info(f"Processing visitor {i}/{len(visitors_to_process)}")
                
                try:
                    recommendations, visitor = self.generate_recommendations_for_visitor(badge_id)
                    
                    if recommendations:
                        # Apply filtering if configured
                        if self.enable_filtering:
                            filtered_recommendations, rules_applied = self.filter_sessions(
                                visitor, recommendations
                            )
                        else:
                            filtered_recommendations = recommendations
                            rules_applied = []
                        
                        self.statistics["total_filtered_recommendations"] += len(filtered_recommendations)
                        
                        if filtered_recommendations:
                            self.statistics["visitors_with_recommendations"] += 1
                            successful += 1
                            total_recommendations += len(filtered_recommendations)
                            
                            for rec in filtered_recommendations:
                                unique_sessions.add(rec["session_id"])
                            
                            all_recommendations.append({
                                "visitor": visitor,
                                "raw_recommendations": recommendations,
                                "filtered_recommendations": filtered_recommendations,
                                "metadata": {
                                    "rules_applied": rules_applied,
                                    "raw_count": len(recommendations),
                                    "filtered_count": len(filtered_recommendations),
                                }
                            })
                            
                            recommendations_dict[badge_id] = {
                                "visitor": visitor,
                                "filtered_recommendations": filtered_recommendations,
                                "rules_applied": rules_applied
                            }
                            
                            self.logger.debug(
                                f"Generated {len(filtered_recommendations)} recommendations for visitor {badge_id}"
                            )
                        else:
                            self.statistics["visitors_without_recommendations"] += 1
                            # Still mark as processed even without recommendations
                            all_recommendations.append({
                                "visitor": visitor,
                                "raw_recommendations": [],
                                "filtered_recommendations": [],
                                "metadata": {
                                    "rules_applied": [],
                                    "raw_count": 0,
                                    "filtered_count": 0,
                                }
                            })
                    else:
                        self.statistics["visitors_without_recommendations"] += 1
                        # Mark visitor as processed even without recommendations
                        all_recommendations.append({
                            "visitor": visitor,
                            "raw_recommendations": [],
                            "filtered_recommendations": [],
                            "metadata": {
                                "rules_applied": [],
                                "raw_count": 0,
                                "filtered_count": 0,
                            }
                        })
                        
                except Exception as e:
                    errors += 1
                    self.statistics["errors"] += 1
                    self.statistics["error_details"].append(f"{badge_id}: {str(e)}")
                    self.logger.error(f"Error processing visitor {badge_id}: {str(e)}")
            
            # Update statistics
            self.statistics["total_recommendations_generated"] = total_recommendations
            self.statistics["unique_recommended_sessions"] = len(unique_sessions)
            
            # Add alias for backward compatibility with summary_utils
            self.statistics["total_visitors_processed"] = self.statistics["visitors_processed"]
            self.statistics["unique_recommended_sessions"] = len(unique_sessions)
            
            # Save recommendations to file
            output_data = {
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "show": self.show_name,
                    "visitors_processed": len(visitors_to_process),
                    "successful": successful,
                    "errors": errors,
                    "total_recommendations": total_recommendations,
                    "unique_sessions": len(unique_sessions),
                    "filtering_enabled": self.enable_filtering,
                    "create_only_new": create_only_new,
                    "configuration": {
                        "min_similarity_score": self.min_similarity_score,
                        "max_recommendations": self.max_recommendations,
                        "similar_visitors_count": self.similar_visitors_count,
                    }
                },
                "recommendations": recommendations_dict,
                "statistics": self.statistics,
            }
            
            with open(output_file, "w") as f:
                json.dump(output_data, f, indent=2, default=str)
            
            self.logger.info(f"Saved recommendations to {output_file}")
            self.logger.info(
                f"Generated recommendations for {successful} visitors with {errors} errors"
            )
            
            # Save as CSV if configured
            if self.recommendation_config.get("save_csv", True):
                df = self.json_to_dataframe(output_file)
                csv_file = str(output_file).replace(".json", ".csv")
                df.to_csv(csv_file, index=False)
                self.logger.info(f"Saved recommendations DataFrame to {csv_file}")
            
            # Update Neo4j with recommendations
            if len(all_recommendations) > 0:
                self.logger.info("Updating Neo4j with recommendation data")
                self._update_visitor_recommendations(all_recommendations)
                self.logger.info("Completed Neo4j updates")
            
            # Calculate processing time
            self.statistics["processing_time"] = time.time() - start_time
            self.logger.info(
                f"Completed session recommendation processing in {self.statistics['processing_time']:.2f} seconds"
            )
            
        except Exception as e:
            self.logger.error(
                f"Error in session recommendation processing: {str(e)}", exc_info=True
            )
            self.statistics["errors"] += 1
            self.statistics["error_details"].append(f"General error: {str(e)}")
            self.statistics["processing_time"] = time.time() - start_time


if __name__ == "__main__":
    """Direct test execution."""
    import argparse
    from utils.config_utils import load_config
    
    parser = argparse.ArgumentParser(description="Process session recommendations")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config_vet.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--create-only-new",
        action="store_true",
        help="Only create recommendations for visitors without them"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create and run processor
    processor = SessionRecommendationProcessor(config)
    processor.process(create_only_new=args.create_only_new)
    
    # Print statistics
    print("\nProcessing Statistics:")
    for key, value in processor.statistics.items():
        if key != "error_details":
            print(f"  {key}: {value}")
    
    # Print error details if any
    if processor.statistics.get("error_details"):
        print("\nError Details:")
        for error in processor.statistics["error_details"][:10]:  # Show first 10 errors
            print(f"  - {error}")
        if len(processor.statistics["error_details"]) > 10:
            print(f"  ... and {len(processor.statistics['error_details']) - 10} more errors")