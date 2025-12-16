import logging
import os
from typing import Any

import pandas as pd
from neo4j import GraphDatabase
from dotenv import load_dotenv
import inspect
import textwrap

from utils.neo4j_utils import resolve_neo4j_credentials



class Neo4jVisitorRelationshipProcessor:
    """
    A generic class to create relationships between visitors from this year and past events.
    This processor creates Same_Visitor relationships and attended_session relationships.
    """

    def __init__(self, config):
        """
        Initialize the Neo4j Visitor Relationship Processor.

        Args:
            config (dict): Configuration dictionary containing paths and settings
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Initializing Neo4j Visitor Relationship Processor"
        )

        self.config = config
        self.output_dir = os.path.join(config["output_dir"], "output")
        self.mode = config.get("mode", "personal_agendas")

        # Extract Neo4j configuration
        neo4j_config = config.get("neo4j", {})
        self.show_name = neo4j_config.get("show_name", "unknown")
        self.node_labels = neo4j_config.get("node_labels", {})
        self.relationships = neo4j_config.get("relationships", {})
        self.unique_identifiers = neo4j_config.get("unique_identifiers", {})
        self.visitor_relationship_config = neo4j_config.get("visitor_relationship", {})

        # Load the environment variables to get Neo4j credentials
        load_dotenv(config["env_file"])
        credentials = resolve_neo4j_credentials(config, logger=self.logger)
        self.uri = credentials["uri"]
        self.username = credentials["username"]
        self.password = credentials["password"]
        self.neo4j_environment = credentials["environment"]
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Using Neo4j environment: {self.neo4j_environment}"
        )

        # Initialize statistics with generic structure
        same_visitor_properties = self.visitor_relationship_config.get("same_visitor_properties", {})
        self.statistics = {
            "relationships_created": {},
            "relationships_skipped": {},
            "relationships_failed": {},
            "nodes_created": {},
            "nodes_skipped": {},
            "nodes_failed": {},
        }
        
        # Initialize statistics for each configured relationship type
        for relationship_type in same_visitor_properties.keys():
            self.statistics["relationships_created"][f"same_visitor_{relationship_type}"] = 0
            self.statistics["relationships_created"][f"attended_session_{relationship_type}"] = 0
            self.statistics["relationships_skipped"][f"same_visitor_{relationship_type}"] = 0
            self.statistics["relationships_skipped"][f"attended_session_{relationship_type}"] = 0
            self.statistics["relationships_failed"][f"same_visitor_{relationship_type}"] = 0
            self.statistics["relationships_failed"][f"attended_session_{relationship_type}"] = 0

        # Post-analysis relationship tracking
        for relationship_key in ["assisted_session_this_year", "registered_to_show"]:
            self.statistics["relationships_created"][relationship_key] = 0
            self.statistics["relationships_skipped"][relationship_key] = 0
            self.statistics["relationships_failed"][relationship_key] = 0

        # Track show node creation (post-analysis entry scans)
        self.statistics["nodes_created"]["show_this_year"] = 0
        self.statistics["nodes_skipped"]["show_this_year"] = 0
        self.statistics["nodes_failed"]["show_this_year"] = 0

        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Neo4j Visitor Relationship Processor initialized successfully for show: {self.show_name}"
        )

    def _test_connection(self):
        """Test the connection to Neo4j database"""
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Testing connection to Neo4j"
        )

        try:
            driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
            with driver.session() as session:
                result = session.run("MATCH (n) RETURN count(n) AS count")
                count = result.single()["count"]
                self.logger.info(
                    f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Successfully connected to Neo4j. Total nodes: {count}"
                )
            driver.close()
            return True
        except Exception as e:
            self.logger.error(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Failed to connect to Neo4j: {str(e)}"
            )
            return False

    def process(self, create_only_new=False):
        """
        Main process method to create visitor relationships.

        Args:
            create_only_new (bool): If True, only create new relationships (skip if already exists)
        """
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Starting Neo4j visitor relationship processing for show: {self.show_name}"
        )

        # Test connection first
        if not self._test_connection():
            raise Exception("Cannot connect to Neo4j database")

        # Get configured relationship types
        same_visitor_properties = self.visitor_relationship_config.get("same_visitor_properties", {})
        
        if not same_visitor_properties:
            self.logger.warning(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - No visitor relationship configuration found, skipping processing"
            )
            return

        # Process each relationship type configured
        for relationship_type, properties in same_visitor_properties.items():
            self.logger.info(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Processing relationships for {relationship_type}"
            )

            # Create Same_Visitor relationships
            try:
                self._create_same_visitor_relationships(relationship_type, properties, create_only_new)
            except Exception as e:
                self.logger.error(
                    f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Error creating Same_Visitor relationships for {relationship_type}: {str(e)}"
                )

            # Create attended_session relationships
            try:
                self._create_attended_session_relationships(relationship_type, properties, create_only_new)
            except Exception as e:
                self.logger.error(
                    f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Error creating attended_session relationships for {relationship_type}: {str(e)}"
                )

        # Post-analysis assisted relationships
        if self.mode == "post_analysis":
            self.logger.info(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Processing assisted_session_this_year relationships (post-analysis mode)"
            )
            try:
                self._create_assisted_session_this_year_relationships(create_only_new)
            except Exception as e:
                self.logger.error(
                    f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Error creating assisted_session_this_year relationships: {str(e)}"
                )

            self.logger.info(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Processing entry scan show registrations (post-analysis mode)"
            )
            try:
                self._create_entry_scan_show_relationships(create_only_new)
            except Exception as e:
                self.logger.error(
                    f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Error creating entry scan show registrations: {str(e)}"
                )

        # Log summary statistics
        self._log_summary()

        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Neo4j visitor relationship processing completed for show: {self.show_name}"
        )

    @staticmethod
    def _clean_text(text: str) -> str:
        if not isinstance(text, str):
            return ""
        return "".join(char for char in text if char.isalnum()).lower()

    @staticmethod
    def _sanitize_property_key(key: str) -> str:
        normalized = "".join(char if char.isalnum() else "_" for char in str(key).strip()).lower()
        normalized = normalized.strip("_")
        return normalized or "field"

    @staticmethod
    def _build_normalized_show_map(entry_scan_config: dict) -> dict:
        file_show_map_config = entry_scan_config.get("file_show_map", {}) if isinstance(entry_scan_config, dict) else {}
        normalized_map = {}
        if isinstance(file_show_map_config, dict):
            for key, value in file_show_map_config.items():
                if isinstance(key, str) and isinstance(value, str):
                    normalized_map[key.lower()] = value
        return normalized_map

    @staticmethod
    def _normalize_badge_id(value: Any) -> str:
        if pd.isna(value):
            return ""

        value_str = str(value).strip()

        if not value_str or value_str.lower() in {"nan", "none", "null"}:
            return ""

        if value_str.endswith(".0") and value_str[:-2].replace("-", "").isdigit():
            value_str = value_str[:-2]

        return value_str

    def _infer_show_from_filename(self, filename: str, normalized_show_map: dict) -> str:
        basename = os.path.basename(filename).lower()
        for needle, mapped_show in normalized_show_map.items():
            if needle in basename:
                return mapped_show

        if "tfm" in basename:
            return "tfm"
        if any(token in basename for token in ["ece", "ecomm"]):
            return self.show_name
        return ""

    def _create_assisted_session_this_year_relationships(self, create_only_new: bool):
        """Create assisted_session_this_year relationships between current-year visitors and sessions."""

        scan_output_files = self.config.get("scan_output_files", {})
        sessions_visited_config = scan_output_files.get("sessions_visited", {})
        sessions_file = sessions_visited_config.get("this_year_post", "sessions_visited_this_year.csv")
        sessions_path = os.path.join(self.output_dir, sessions_file)

        if not os.path.exists(sessions_path):
            self.logger.warning(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Post-analysis sessions file not found: {sessions_path}"
            )
            return

        try:
            sessions_df = pd.read_csv(sessions_path)
        except Exception as read_error:
            self.logger.error(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Unable to read post-analysis sessions file: {read_error}"
            )
            return

        if sessions_df.empty:
            self.logger.info(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Post-analysis sessions file is empty; no relationships to create"
            )
            return

        badge_column = None
        for candidate in ["Badge Id", "BadgeId", "badge_id"]:
            if candidate in sessions_df.columns:
                badge_column = candidate
                break

        if not badge_column:
            self.logger.error(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - No badge column found in {sessions_path}"
            )
            return

        if "key_text" not in sessions_df.columns:
            if "Seminar Name" in sessions_df.columns:
                sessions_df["key_text"] = sessions_df["Seminar Name"].apply(self._clean_text)
            else:
                self.logger.error(
                    f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - No key_text or Seminar Name column available for matching"
                )
                return

        sessions_df["key_text_lower"] = sessions_df["key_text"].astype(str).str.strip().str.lower()
        sessions_df = sessions_df[sessions_df["key_text_lower"] != ""]

        visitor_label = self.node_labels.get("visitor_this_year", "Visitor_this_year")
        session_label = self.node_labels.get("session_this_year", "Sessions_this_year")
        relationship_name = self.relationships.get(
            "assisted_session_this_year", "assisted_session_this_year"
        )
        visitor_id_field = self.unique_identifiers.get("visitor", "BadgeId")

        created = 0
        skipped = 0
        failed = 0

        driver = None
        try:
            driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))

            with driver.session() as session:
                for _, row in sessions_df.iterrows():
                    badge_id = str(row.get(badge_column, "")).strip()
                    if not badge_id:
                        skipped += 1
                        continue

                    session_key_lower = str(row.get("key_text_lower", "")).strip()
                    if not session_key_lower:
                        skipped += 1
                        continue

                    scan_date = row.get("Scan Date", "")
                    file_name = row.get("File", "")
                    seminar_name = row.get("Seminar Name", row.get("Seminar", ""))

                    query = (
                        f"MATCH (v:{visitor_label} {{{visitor_id_field}: $badge_id}}) "
                        f"MATCH (s:{session_label}) "
                        "WHERE toLower(s.key_text) = $session_key_lower AND v.show = $show_name AND s.show = $show_name "
                        f"MERGE (v)-[r:{relationship_name}]->(s) "
                        "SET r.scan_date = coalesce($scan_date, r.scan_date), "
                        "    r.file = coalesce($file_name, r.file), "
                        "    r.seminar_name = coalesce($seminar_name, r.seminar_name) "
                        "RETURN r"
                    )

                    try:
                        result = session.run(
                            query,
                            badge_id=badge_id,
                            session_key_lower=session_key_lower,
                            show_name=self.show_name,
                            scan_date=str(scan_date) if pd.notna(scan_date) else None,
                            file_name=str(file_name) if pd.notna(file_name) else None,
                            seminar_name=str(seminar_name) if pd.notna(seminar_name) else None,
                        )
                        summary = result.consume()
                        if summary.counters.relationships_created > 0:
                            created += 1
                        else:
                            skipped += 1
                    except Exception as rel_error:
                        failed += 1
                        self.logger.error(
                            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Failed to merge assisted relationship for badge {badge_id}: {rel_error}"
                        )

        except Exception as driver_error:
            failed += 1
            self.logger.error(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Neo4j driver error during assisted relationship creation: {driver_error}"
            )
        finally:
            if driver:
                driver.close()

        self.statistics["relationships_created"]["assisted_session_this_year"] = created
        self.statistics["relationships_skipped"]["assisted_session_this_year"] = skipped
        self.statistics["relationships_failed"]["assisted_session_this_year"] = failed

        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - assisted_session_this_year Relationship Summary: created={created}, skipped={skipped}, failed={failed}"
        )

    def _create_entry_scan_show_relationships(self, create_only_new: bool):
        """Create Show nodes and register_to_show relationships from entry scan files."""

        post_analysis_config = self.config.get("post_analysis_mode", {})
        entry_scan_config = post_analysis_config.get("entry_scan_files", {})
        entry_scan_files = entry_scan_config.get("entry_scans_this", [])
        normalized_show_map = self._build_normalized_show_map(entry_scan_config)

        if not entry_scan_files:
            self.logger.info(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - No entry scan files configured; skipping show registration processing"
            )
            return

        if isinstance(entry_scan_files, str):
            entry_scan_files = [entry_scan_files]

        visitor_label = self.node_labels.get("visitor_this_year", "Visitor_this_year")
        show_label = self.node_labels.get("show_this_year", "Show_this_year")
        visitor_id_field = self.unique_identifiers.get("visitor", "BadgeId")
        show_unique_field = self.unique_identifiers.get("show", "show_ref")
        relationship_name = self.relationships.get("registered_to_show", "registered_to_show")

        show_nodes_created = 0
        show_nodes_skipped = 0
        show_nodes_failed = 0

        relationships_created = 0
        relationships_skipped = 0
        relationships_failed = 0
        missing_visitors = 0
        missing_badge_samples = set()

        seen_shows = set()

        def resolve_path(path_candidate: str) -> str:
            """Attempt to resolve a configured path to an existing file."""

            potential_paths = [path_candidate]
            base_output_dir = self.config.get("output_dir")

            if base_output_dir:
                potential_paths.append(os.path.join(base_output_dir, path_candidate))

            # Include processor output directory (typically <output_dir>/output)
            potential_paths.append(os.path.join(self.output_dir, path_candidate))

            for candidate in potential_paths:
                if candidate and os.path.exists(candidate):
                    return candidate

            return ""

        driver = None

        try:
            driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))

            with driver.session() as session:
                for configured_path in entry_scan_files:
                    resolved_path = resolve_path(configured_path)

                    if not resolved_path:
                        self.logger.warning(
                            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Entry scan file not found: {configured_path}"
                        )
                        continue

                    try:
                        scans_df = pd.read_csv(resolved_path)
                    except Exception as read_error:
                        show_nodes_failed += 1
                        self.logger.error(
                            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Failed to read entry scan file {resolved_path}: {read_error}"
                        )
                        continue

                    if scans_df.empty:
                        self.logger.info(
                            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Entry scan file {resolved_path} is empty; skipping"
                        )
                        continue

                    badge_column = None
                    for candidate in ["Badge Id", "BadgeId", "badge_id"]:
                        if candidate in scans_df.columns:
                            badge_column = candidate
                            break

                    if not badge_column:
                        self.logger.error(
                            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - No badge column found in entry scan file: {resolved_path}"
                        )
                        continue

                    show_column = None
                    for candidate in ["Show Ref", "ShowRef", "Show", "Show Code", "EventCode", "Event Code"]:
                        if candidate in scans_df.columns:
                            show_column = candidate
                            break

                    if not show_column:
                        # Fallback: find the first column containing "show"
                        for column in scans_df.columns:
                            if "show" in column.lower():
                                show_column = column
                                break

                    if not show_column:
                        self.logger.error(
                            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - No show reference column found in entry scan file: {resolved_path}"
                        )
                        continue

                    show_name_column = None
                    for candidate in ["Show Name", "Show_Title", "ShowTitle", "EventName", "Event Name"]:
                        if candidate in scans_df.columns:
                            show_name_column = candidate
                            break

                    target_show_code = self._infer_show_from_filename(resolved_path, normalized_show_map)
                    target_show_code_normalized = target_show_code.lower() if target_show_code else ""

                    visitor_show_candidates = []
                    if target_show_code_normalized:
                        visitor_show_candidates.append(target_show_code_normalized)
                    visitor_show_candidates.append(self.show_name.lower())
                    visitor_show_candidates = list(dict.fromkeys(visitor_show_candidates))

                    relationship_columns = [
                        "Show Ref",
                        "Badge Id",
                        "Badge Type",
                        "Attended",
                        "Virtual Attended",
                        "Day 1",
                        "Day 1 Entry",
                        "Day 2",
                        "Day 2 Entry",
                    ]

                    for _, row in scans_df.iterrows():
                        badge_raw = row.get(badge_column, "")
                        badge_id = self._normalize_badge_id(badge_raw)

                        show_raw = row.get(show_column, "")
                        if pd.isna(show_raw):
                            show_ref = ""
                        else:
                            show_ref = str(show_raw).strip()
                            if show_ref.endswith(".0") and show_ref[:-2].replace("-", "").isdigit():
                                show_ref = show_ref[:-2]

                        if not badge_id or badge_id.lower() == "nan":
                            relationships_skipped += 1
                            continue

                        if not show_ref or show_ref.lower() == "nan":
                            relationships_skipped += 1
                            continue

                        show_ref_normalized = show_ref.upper()
                        show_identifier = show_ref_normalized or (target_show_code.upper() if target_show_code else "")

                        show_properties = {
                            show_unique_field: show_identifier,
                            "source_file": os.path.basename(resolved_path),
                            "parent_show": self.show_name,
                            "show": target_show_code or self.show_name,
                        }

                        if show_unique_field != "show_ref":
                            show_properties.setdefault("show_ref", show_ref_normalized)

                        if show_name_column:
                            show_name_value = row.get(show_name_column, "")
                            if pd.notna(show_name_value):
                                show_properties["show_name"] = str(show_name_value).strip()

                        if target_show_code:
                            show_properties.setdefault("show_code", target_show_code)

                        if show_identifier not in seen_shows:
                            try:
                                result = session.run(
                                    f"MERGE (s:{show_label} {{{show_unique_field}: $show_value}}) "
                                    "ON CREATE SET s += $show_properties "
                                    "ON MATCH SET s += $show_properties "
                                    "RETURN s",
                                    show_value=show_identifier,
                                    show_properties=show_properties,
                                )
                                summary = result.consume()
                                if summary.counters.nodes_created > 0:
                                    show_nodes_created += summary.counters.nodes_created
                                else:
                                    show_nodes_skipped += 1
                                seen_shows.add(show_identifier)
                            except Exception as show_error:
                                show_nodes_failed += 1
                                self.logger.error(
                                    f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Failed to merge show node {show_identifier}: {show_error}"
                                )
                                continue
                        else:
                            show_nodes_skipped += 1

                        # Confirm visitor exists for this show
                        visitor_exists_query = (
                            f"MATCH (v:{visitor_label} {{{visitor_id_field}: $badge_id}}) "
                            "WHERE toLower(v.show) IN $visitor_shows "
                            "RETURN count(v) > 0 AS exists"
                        )

                        visitor_exists = session.run(
                            visitor_exists_query,
                            badge_id=badge_id,
                            visitor_shows=visitor_show_candidates,
                        ).single()["exists"]

                        if not visitor_exists:
                            missing_visitors += 1
                            if len(missing_badge_samples) < 20:
                                missing_badge_samples.add(badge_id)
                            relationships_skipped += 1
                            continue

                        relationship_properties = {
                            "source_file": os.path.basename(resolved_path),
                            "show_file_code": target_show_code or "",
                        }

                        for column in relationship_columns:
                            if column in scans_df.columns:
                                value = row.get(column)
                                if pd.isna(value):
                                    relationship_properties[self._sanitize_property_key(column)] = None
                                else:
                                    relationship_properties[self._sanitize_property_key(column)] = str(value).strip()
                            else:
                                relationship_properties[self._sanitize_property_key(column)] = None

                        protected_relationship_keys = {"source_file", "show_file_code"}
                        for column in scans_df.columns:
                            sanitized_key = self._sanitize_property_key(column)
                            if sanitized_key in protected_relationship_keys:
                                continue

                            value = row.get(column)
                            if pd.isna(value):
                                relationship_properties.setdefault(sanitized_key, None)
                            else:
                                relationship_properties[sanitized_key] = str(value).strip()

                        for candidate in ["Scan Date", "scan_date", "Entry Date", "entry_date"]:
                            if candidate in scans_df.columns:
                                value = row.get(candidate)
                                if pd.notna(value):
                                    relationship_properties.setdefault("scan_date", str(value).strip())
                                    break

                        for candidate in ["Scan Time", "scan_time", "Entry Time", "entry_time"]:
                            if candidate in scans_df.columns:
                                value = row.get(candidate)
                                if pd.notna(value):
                                    relationship_properties.setdefault("scan_time", str(value).strip())
                                    break

                        try:
                            merge_query = (
                                f"MATCH (v:{visitor_label} {{{visitor_id_field}: $badge_id}}) "
                                f"MATCH (s:{show_label} {{{show_unique_field}: $show_value}}) "
                                "WHERE toLower(v.show) IN $visitor_shows "
                                f"MERGE (v)-[r:{relationship_name}]->(s) "
                                "SET r += $relationship_properties "
                                "RETURN r"
                            )

                            result = session.run(
                                merge_query,
                                badge_id=badge_id,
                                show_value=show_identifier,
                                visitor_shows=visitor_show_candidates,
                                relationship_properties=relationship_properties,
                            )
                            summary = result.consume()

                            if summary.counters.relationships_created > 0:
                                relationships_created += summary.counters.relationships_created
                            else:
                                relationships_skipped += 1

                        except Exception as rel_error:
                            relationships_failed += 1
                            self.logger.error(
                                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Failed to merge register_to_show relationship for badge {badge_id} -> {show_ref_normalized}: {rel_error}"
                            )

        except Exception as driver_error:
            self.logger.error(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Neo4j driver error during entry scan processing: {driver_error}"
            )
        finally:
            if driver:
                driver.close()

        self.statistics["nodes_created"]["show_this_year"] = show_nodes_created
        self.statistics["nodes_skipped"]["show_this_year"] = show_nodes_skipped
        self.statistics["nodes_failed"]["show_this_year"] = show_nodes_failed

        self.statistics["relationships_created"]["registered_to_show"] = relationships_created
        self.statistics["relationships_skipped"]["registered_to_show"] = relationships_skipped
        self.statistics["relationships_failed"]["registered_to_show"] = relationships_failed

        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Entry scan processing summary: show_nodes_created={show_nodes_created}, show_nodes_skipped={show_nodes_skipped}, show_nodes_failed={show_nodes_failed}, relationships_created={relationships_created}, relationships_skipped={relationships_skipped}, relationships_failed={relationships_failed}"
        )

        if missing_visitors:
            sample_list = ", ".join(sorted(missing_badge_samples))
            self.logger.warning(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - register_to_show skipped {missing_visitors} rows due to missing visitors. Sample badge IDs: {sample_list}"
            )

    def _create_same_visitor_relationships(self, relationship_type, properties, create_only_new):
        """Create Same_Visitor relationships for a specific visitor segment."""

        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Creating Same_Visitor relationships for {relationship_type}"
        )

        visitor_this_year_label = self.node_labels.get("visitor_this_year", "Visitor_this_year")
        visitor_last_year_label = self.node_labels.get(
            f"visitor_last_year_{relationship_type}", f"Visitor_last_year_{relationship_type}"
        )
        same_visitor_relationship = self.relationships.get("same_visitor", "Same_Visitor")
        properties = properties or {}
        relationship_properties = properties
        visitor_id_field = self.unique_identifiers.get("visitor", "BadgeId")

        relationship_count = 0
        skipped_count = 0
        failed_count = 0

        property_filter_clause = textwrap.dedent(
            """
            AND (
                $relationship_properties IS NULL
                OR size(keys($relationship_properties)) = 0
                OR ALL(key IN keys($relationship_properties) WHERE r[key] = $relationship_properties[key])
            )
            """
        )

        try:
            driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))

            with driver.session() as session:
                initial_query = textwrap.dedent(
                    f"""
                    MATCH (this_year:{visitor_this_year_label})-[r]->(last_year:{visitor_last_year_label})
                    WHERE this_year.show = $show_name
                      AND last_year.show = $show_name
                      AND type(r) = $same_visitor_relationship
                      {property_filter_clause.strip()}
                    RETURN COUNT(r) AS count
                    """
                )

                initial_count = session.run(
                    initial_query,
                    show_name=self.show_name,
                    same_visitor_relationship=same_visitor_relationship,
                    relationship_properties=relationship_properties,
                ).single()["count"]

                self.logger.info(
                    f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Existing {relationship_type} Same_Visitor relationships before run: {initial_count}"
                )

                badge_id_field = f"{visitor_id_field}_last_year_{relationship_type}"

                candidate_query = textwrap.dedent(
                    f"""
                    MATCH (this_year:{visitor_this_year_label})
                    WHERE this_year.show = $show_name
                      AND this_year.{badge_id_field} IS NOT NULL
                      AND this_year.{badge_id_field} <> 'NA'
                    WITH this_year, this_year.{visitor_id_field} AS this_year_id, this_year.{badge_id_field} AS last_year_id
                    MATCH (last_year:{visitor_last_year_label} {{{visitor_id_field}: last_year_id}})
                    WHERE last_year.show = $show_name
                    OPTIONAL MATCH (this_year)-[r]->(last_year)
                    WHERE type(r) = $same_visitor_relationship
                      {property_filter_clause.strip()}
                    WITH this_year_id, last_year_id, r
                    WHERE r IS NULL
                    RETURN this_year_id, last_year_id
                    """
                )

                visitors_result = session.run(
                    candidate_query,
                    show_name=self.show_name,
                    same_visitor_relationship=same_visitor_relationship,
                    relationship_properties=relationship_properties,
                )

                candidates = list(visitors_result)

                self.logger.info(
                    f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Candidate current-year visitors missing Same_Visitor ({relationship_type}): {len(candidates)}"
                )

                for record in candidates:
                    this_year_id = record["this_year_id"]
                    last_year_id = record["last_year_id"]

                    try:
                        set_clause = ", ".join([f"r.{k} = ${k}" for k in properties.keys()])
                        set_fragment = f"SET {set_clause}" if set_clause else ""

                        merge_query = textwrap.dedent(
                            f"""
                            MATCH (this_year:{visitor_this_year_label} {{{visitor_id_field}: $this_year_id}})
                            MATCH (last_year:{visitor_last_year_label} {{{visitor_id_field}: $last_year_id}})
                            WHERE this_year.show = $show_name AND last_year.show = $show_name
                            MERGE (this_year)-[r:{same_visitor_relationship}]->(last_year)
                            {set_fragment}
                            """
                        )

                        session.run(
                            merge_query,
                            this_year_id=this_year_id,
                            last_year_id=last_year_id,
                            show_name=self.show_name,
                            **properties,
                        )

                        relationship_count += 1
                    except Exception as e:
                        failed_count += 1
                        self.logger.error(
                            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Error creating Same_Visitor relationship for {this_year_id}->{last_year_id}: {str(e)}"
                        )

        except Exception as e:
            self.logger.error(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Error creating Same_Visitor relationships for {relationship_type}: {str(e)}"
            )
            raise
        finally:
            driver.close()

        self.statistics["relationships_created"][f"same_visitor_{relationship_type}"] = relationship_count
        self.statistics["relationships_skipped"][f"same_visitor_{relationship_type}"] = skipped_count
        self.statistics["relationships_failed"][f"same_visitor_{relationship_type}"] = failed_count

        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - {relationship_type} Same_Visitor Relationship Summary:"
        )
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Relationships created: {relationship_count}"
        )
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Relationships skipped: {skipped_count}"
        )
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Relationships failed: {failed_count}"
        )

    def _create_attended_session_relationships(self, relationship_type, properties, create_only_new):
        """
        Create attended_session relationships for a specific relationship type.

        Args:
            relationship_type (str): Type of relationship (e.g., 'bva', 'lva')
            properties (dict): Properties to attach to the relationship
            create_only_new (bool): If True, only create new relationships
        """
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Creating attended_session relationships for {relationship_type}"
        )

        # Get node labels
        visitor_last_year_label = self.node_labels.get(f"visitor_last_year_{relationship_type}", f"Visitor_last_year_{relationship_type}")
        session_past_year_label = self.node_labels.get("session_past_year", "Sessions_past_year")
        
        # Get relationship name
        attended_session_relationship = self.relationships.get("attended_session", "attended_session")
        
        # Get unique identifier
        visitor_id_field = self.unique_identifiers.get("visitor", "BadgeId")

        relationship_count = 0
        skipped_count = 0
        failed_count = 0

        try:
            driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))

            # Get initial count of existing relationships
            with driver.session() as session:
                initial_count_query = textwrap.dedent(
                    f"""
                    MATCH (visitor:{visitor_last_year_label})-[r]->(session:{session_past_year_label})
                    WHERE visitor.show = $show_name
                      AND session.show = $show_name
                      AND type(r) = $attended_session_relationship
                    RETURN COUNT(r) AS count
                    """
                )

                initial_count_result = session.run(
                    initial_count_query,
                    show_name=self.show_name,
                    attended_session_relationship=attended_session_relationship,
                )
                initial_count = initial_count_result.single()["count"]

                self.logger.info(
                    f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Found {initial_count} existing {relationship_type} attended_session relationships"
                )
                # Note: Do not globally skip in incremental mode. We'll rely on per-row existence checks
                # so that new relationships get added while existing ones are skipped idempotently.

                # Load and process scan data
                scan_files = self.config.get("scan_files", {})
                scan_output_files = self.config.get("scan_output_files", {})
                
                if relationship_type == "bva":
                    scan_file_key = "session_past_main"
                    output_file_key = "last_year_main"
                elif relationship_type == "lva":
                    scan_file_key = "session_past_secondary"  
                    output_file_key = "last_year_secondary"
                else:
                    # For other show types, skip attended_session processing
                    self.logger.info(
                        f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - No scan file configuration for {relationship_type}, skipping attended_session relationships"
                    )
                    return

                # Load and process scan data to match old processor behavior (now configurable via YAML)
                attended_inputs = scan_output_files.get("attended_session_inputs", {})
                if relationship_type == "bva":
                    scan_filename = attended_inputs.get("past_year_main_scan", "scan_bva_past.csv")
                    filter_filename = attended_inputs.get("past_year_main_filter", "df_reg_demo_last_bva.csv")
                elif relationship_type == "lva":
                    scan_filename = attended_inputs.get("past_year_secondary_scan", "scan_lva_past.csv")
                    filter_filename = attended_inputs.get("past_year_secondary_filter", "df_reg_demo_last_lva.csv")
                else:
                    # For other show types, skip attended_session processing
                    self.logger.info(
                        f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - No scan file configuration for {relationship_type}, skipping attended_session relationships"
                    )
                    return

                # Load scan data file (matches old processor)
                output_dir = os.path.join(self.config["output_dir"], "output")
                scan_file_path = os.path.join(output_dir, scan_filename)
                filter_file_path = os.path.join(output_dir, filter_filename)

                if not os.path.exists(scan_file_path):
                    self.logger.warning(
                        f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Scan file not found: {scan_file_path}, skipping attended_session relationships for {relationship_type}"
                    )
                    return
                    
                if not os.path.exists(filter_file_path):
                    self.logger.warning(
                        f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Filter file not found: {filter_file_path}, skipping attended_session relationships for {relationship_type}"
                    )
                    return

                try:
                    # Load data (matches old processor logic)
                    scan_df = pd.read_csv(scan_file_path)
                    filter_df = pd.read_csv(filter_file_path)

                    # Get list of badge IDs to filter by
                    badge_ids = list(filter_df[visitor_id_field].unique())
                    self.logger.info(
                        f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Loaded {len(badge_ids)} badge IDs for filtering scan data"
                    )

                    # Filter scan data to only include visitors in the filter list
                    scan_df = scan_df[scan_df["Badge Id"].isin(badge_ids)]
                    self.logger.info(
                        f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Filtered scan data contains {len(scan_df)} records for {relationship_type}"
                    )

                    if len(scan_df) == 0:
                        self.logger.warning(
                            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - No scan data records found after filtering for {relationship_type}"
                        )
                        return
                        
                except Exception as e:
                    self.logger.error(
                        f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Error loading scan data for {relationship_type}: {str(e)}"
                    )
                    return

                # Process each scan record (matches old processor logic)
                for _, row in scan_df.iterrows():
                    visitor_badge_id = str(row.get("Badge Id", ""))
                    session_key_text = str(row.get("key_text", ""))
                    scan_date = str(row.get("Scan Date", ""))
                    file = str(row.get("File", ""))
                    seminar_name = str(row.get("Seminar Name", ""))

                    if not visitor_badge_id or visitor_badge_id == "nan" or not session_key_text or session_key_text == "nan":
                        continue

                    # Check if the relationship already exists
                    relationship_exists_query = textwrap.dedent(
                        f"""
                        MATCH (v:{visitor_last_year_label} {{{visitor_id_field}: $visitor_badge_id}})
                        MATCH (s:{session_past_year_label} {{key_text: $session_key_text}})
                        MATCH (v)-[r]->(s)
                        WHERE v.show = $show_name
                          AND s.show = $show_name
                          AND type(r) = $attended_session_relationship
                        RETURN count(*) > 0 AS exists
                        """
                    )

                    relationship_exists = session.run(
                        relationship_exists_query,
                        visitor_badge_id=visitor_badge_id,
                        session_key_text=session_key_text,
                        show_name=self.show_name,
                        attended_session_relationship=attended_session_relationship,
                    ).single()["exists"]

                    if not relationship_exists:
                        # Check if both nodes exist before creating relationship
                        nodes_exist = session.run(
                            f"""
                            MATCH (v:{visitor_last_year_label} {{{visitor_id_field}: $visitor_badge_id}})
                            MATCH (s:{session_past_year_label} {{key_text: $session_key_text}})
                            WHERE v.show = $show_name AND s.show = $show_name
                            RETURN count(*) > 0 AS exists
                            """,
                            visitor_badge_id=visitor_badge_id,
                            session_key_text=session_key_text,
                            show_name=self.show_name
                        ).single()["exists"]

                        if nodes_exist:
                            try:
                                session.run(
                                    f"""
                                    MATCH (v:{visitor_last_year_label} {{{visitor_id_field}: $visitor_badge_id}})
                                    MATCH (s:{session_past_year_label} {{key_text: $session_key_text}})
                                    WHERE v.show = $show_name AND s.show = $show_name
                                    CREATE (v)-[:{attended_session_relationship} {{
                                        scan_date: $scan_date, 
                                        file: $file, 
                                        seminar_name: $seminar_name
                                    }}]->(s)
                                    """,
                                    visitor_badge_id=visitor_badge_id,
                                    session_key_text=session_key_text,
                                    scan_date=scan_date,
                                    file=file,
                                    seminar_name=seminar_name,
                                    show_name=self.show_name
                                )
                                relationship_count += 1
                            except Exception as e:
                                failed_count += 1
                                self.logger.error(
                                    f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Error creating attended_session relationship: {str(e)}"
                                )
                        else:
                            failed_count += 1
                    else:
                        skipped_count += 1

        except Exception as e:
            self.logger.error(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Error creating attended_session relationships for {relationship_type}: {str(e)}"
            )
            raise
        finally:
            driver.close()

        # Update statistics
        self.statistics["relationships_created"][f"attended_session_{relationship_type}"] = relationship_count
        self.statistics["relationships_skipped"][f"attended_session_{relationship_type}"] = skipped_count
        self.statistics["relationships_failed"][f"attended_session_{relationship_type}"] = failed_count

        # Log summary statistics
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - {relationship_type} attended_session Relationship Summary:"
        )
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Relationships created: {relationship_count}"
        )
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Relationships skipped: {skipped_count}"
        )
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Relationships failed: {failed_count}"
        )

    def _log_summary(self):
        """Log summary statistics for all relationship types"""
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Visitor Relationship Processing Summary for show: {self.show_name}:"
        )
        
        for key, value in self.statistics["relationships_created"].items():
            skipped = self.statistics["relationships_skipped"].get(key, 0)
            failed = self.statistics["relationships_failed"].get(key, 0)
            self.logger.info(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - {key}: {value} created, {skipped} skipped, {failed} failed"
            )

        if self.statistics.get("nodes_created"):
            for key, value in self.statistics["nodes_created"].items():
                skipped = self.statistics["nodes_skipped"].get(key, 0)
                failed = self.statistics["nodes_failed"].get(key, 0)
                self.logger.info(
                    f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - {key} nodes: {value} created, {skipped} skipped, {failed} failed"
                )