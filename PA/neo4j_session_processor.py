#!/usr/bin/env python3
"""
Fixed Neo4j Session Processor

This processor creates session and stream nodes in Neo4j with proper relationship handling
that matches the old processor's logic exactly.
"""

import os
import sys
import json
import pandas as pd
import inspect
import logging
from neo4j import GraphDatabase
from dotenv import load_dotenv
from typing import Dict, Any, Tuple, List

from utils.neo4j_utils import resolve_neo4j_credentials

class Neo4jSessionProcessor:
    """
    A processor for loading session data into Neo4j and creating relationships with streams.
    Fixed to match old processor's relationship creation logic exactly.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Neo4j Session Processor.

        Args:
            config: Configuration dictionary containing database settings and file paths
        """
        self.config = config
        self.mode = config.get("mode", "personal_agendas")
        self.logger = logging.getLogger(__name__)
        
        # Load environment variables
        load_dotenv(config.get("env_file", "keys/.env"))
        
        # Neo4j connection details
        credentials = resolve_neo4j_credentials(config, logger=self.logger)
        self.uri = credentials["uri"]
        self.username = credentials["username"] 
        self.password = credentials["password"]
        self.neo4j_environment = credentials["environment"]
        self.logger.info("Neo4j session processor using '%s' environment", self.neo4j_environment)
        
        # Configuration
        self.output_dir = config.get("output_dir", "data/bva")
        
        # Event configuration for generic processing
        event_config = config.get("event", {})
        self.main_event_name = event_config.get("main_event_name", "bva")
        self.secondary_event_name = event_config.get("secondary_event_name", "lva")

        self.session_output_config = config.get("session_output_files", {})
        self.session_processed_files = self.session_output_config.get(
            "processed_sessions", {}
        )
        self.streams_catalog_file = self.session_output_config.get(
            "streams_catalog", "streams.json"
        )
        
        # Show name for node properties
        self.show_name = config.get("neo4j", {}).get("show_name", "bva")
        
        # Statistics tracking
        self.statistics = {
            "nodes_created": {
                "sessions_this_year": 0,
                "sessions_past_year_bva": 0,
                "sessions_past_year_lva": 0,
                "streams": 0
            },
            "nodes_skipped": {
                "sessions_this_year": 0,
                "sessions_past_year_bva": 0,
                "sessions_past_year_lva": 0,
                "streams": 0
            },
            "relationships_created": {
                "sessions_this_year_has_stream": 0,
                "sessions_past_year_has_stream": 0
            },
            "relationships_skipped": {
                "sessions_this_year_has_stream": 0,
                "sessions_past_year_has_stream": 0
            }
        }

        self.logger.info(f"Initialized Neo4j Session Processor for {self.main_event_name}")

    def _test_connection(self) -> bool:
        """Test connection to Neo4j database."""
        try:
            self.logger.info("Testing connection to Neo4j")
            driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
            
            with driver.session() as session:
                result = session.run("MATCH (n) RETURN count(n) as total_nodes")
                total_nodes = result.single()["total_nodes"]
                self.logger.info(f"Successfully connected to Neo4j. Database contains {total_nodes} nodes")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to connect to Neo4j: {str(e)}")
            return False
        finally:
            if 'driver' in locals():
                driver.close()

    def load_csv_to_neo4j(self, csv_file_path: str, node_label: str, properties_map: Dict[str, str], 
                         unique_property: str, create_only_new: bool = True) -> Tuple[int, int]:
        """
        Load CSV data into Neo4j nodes.
        
        Args:
            csv_file_path: Path to the CSV file
            node_label: Label for the Neo4j nodes
            properties_map: Mapping of CSV columns to node properties
            unique_property: Property to use for uniqueness checking
            create_only_new: If True, only create nodes that don't exist
            
        Returns:
            Tuple of (nodes_created, nodes_skipped)
        """
        self.logger.info(f"Loading CSV to Neo4j: {csv_file_path}")
        
        nodes_created = 0
        nodes_skipped = 0
        
        try:
            # Load CSV data
            data = pd.read_csv(csv_file_path)
            
            driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
            
            with driver.session() as session:
                # Define core text fields used downstream (embedding, matching)
                required_text_fields = ["title", "synopsis_stripped", "theatre__name", "key_text", "stream"]

                for index, row in data.iterrows():
                    try:
                        # Create properties dictionary
                        properties = {}
                        for csv_col, node_prop in properties_map.items():
                            if csv_col in row:
                                val = row[csv_col]
                                # Normalize NaN to empty string for text fields
                                if pd.isna(val):
                                    val_str = ""
                                else:
                                    val_str = str(val)
                                properties[node_prop] = val_str
                        
                        # Add show attribute
                        properties['show'] = self.show_name

                        # Skip placeholder sessions with no meaningful text
                        if all(properties.get(f, "").strip() == "" for f in required_text_fields):
                            nodes_skipped += 1
                            continue

                        # Check if node already exists (only if create_only_new is True)
                        if create_only_new:
                            check_query = f"""
                            MATCH (n:{node_label} {{{unique_property}: $unique_value}})
                            RETURN count(n) as count
                            """
                            result = session.run(check_query, unique_value=properties[unique_property])
                            exists = result.single()["count"] > 0

                            if exists:
                                nodes_skipped += 1
                                continue

                        # Create or merge node
                        if create_only_new:
                            # Use CREATE when we know it doesn't exist
                            query = f"""
                            CREATE (n:{node_label})
                            SET n += $properties
                            RETURN n
                            """
                        else:
                            # Use MERGE for upsert behavior
                            query = f"""
                            MERGE (n:{node_label} {{{unique_property}: $unique_value}})
                            SET n += $properties
                            RETURN n
                            """

                        if create_only_new:
                            session.run(query, properties=properties)
                        else:
                            session.run(query, 
                                      unique_value=properties[unique_property], 
                                      properties=properties)
                        
                        nodes_created += 1

                    except Exception as e:
                        self.logger.error(f"Error processing row {index}: {str(e)}")
                        continue

        except Exception as e:
            self.logger.error(f"Error loading CSV to Neo4j: {str(e)}")
            raise
        finally:
            if 'driver' in locals():
                driver.close()

        self.logger.info(f"CSV loaded to Neo4j. Created {nodes_created} nodes, skipped {nodes_skipped} nodes")
        return nodes_created, nodes_skipped

    def create_stream_nodes(self, streams_file_path: str, create_only_new: bool = True) -> Tuple[int, int]:
        """
        Create stream nodes from JSON file.
        FIXED: Handles both old format (dict) and new format (array) for compatibility.
        
        Args:
            streams_file_path: Path to the streams JSON file
            create_only_new: If True, only create nodes that don't exist
            
        Returns:
            Tuple of (nodes_created, nodes_skipped)
        """
        self.logger.info(f"Creating stream nodes from: {streams_file_path}")
        
        nodes_created = 0
        nodes_skipped = 0
        
        try:
            # Load streams data
            with open(streams_file_path, 'r', encoding='utf-8') as f:
                streams_data = json.load(f)
            
            # FIXED: Handle both old and new formats
            streams_dict = {}
            
            if isinstance(streams_data, dict):
                # Old format: {"stream_name": "description", ...}
                streams_dict = streams_data
                self.logger.info("Loaded streams in old format (dictionary)")
            elif isinstance(streams_data, list):
                # New format: [{"stream": "name", "description": "desc"}, ...]
                for item in streams_data:
                    if isinstance(item, dict) and 'stream' in item:
                        stream_name = item['stream']
                        description = item.get('description', f"Stream for {stream_name}")
                        streams_dict[stream_name] = description
                self.logger.info("Loaded streams in new format (array) - converted to old format")
            else:
                raise ValueError(f"Unexpected streams data format: {type(streams_data)}")
            
            driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
            
            with driver.session() as session:
                for stream_name, description in streams_dict.items():
                    try:
                        # Check if stream already exists (only if create_only_new is True)
                        if create_only_new:
                            check_query = "MATCH (s:Stream {stream: $stream}) RETURN count(s) as count"
                            result = session.run(check_query, stream=stream_name)
                            if result.single()["count"] > 0:
                                nodes_skipped += 1
                                continue

                        # Create stream node with 'stream', 'description', and 'show' attributes
                        create_query = """
                        CREATE (s:Stream {stream: $stream, description: $description, show: $show})
                        RETURN s
                        """
                        session.run(create_query, stream=stream_name, description=description, show=self.show_name)
                        nodes_created += 1
                        
                    except Exception as e:
                        self.logger.error(f"Error creating stream node '{stream_name}': {str(e)}")
                        continue

        except Exception as e:
            self.logger.error(f"Error creating stream nodes: {str(e)}")
            raise
        finally:
            if 'driver' in locals():
                driver.close()

        self.logger.info(f"Stream nodes created: {nodes_created}, skipped: {nodes_skipped}")
        return nodes_created, nodes_skipped

    def create_stream_relationships(self, session_node_label: str, stream_node_label: str, 
                                create_only_new: bool = True) -> Tuple[int, int]:
        """
        Create relationships between session nodes and stream nodes.
        SIMPLE FIX: Better relationship existence checking to prevent duplicates.
        
        Args:
            session_node_label: Label of session nodes
            stream_node_label: Label of stream nodes  
            create_only_new: If True, only create relationships that don't exist
            
        Returns:
            Tuple of (relationships_created, relationships_skipped)
        """
        self.logger.info(f"Creating relationships between {session_node_label} and {stream_node_label}")
        
        driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
        
        # Count initial relationships for verification
        with driver.session() as session:
            initial_count_result = session.run(
                f"""
                MATCH (s:{session_node_label})-[r:HAS_STREAM]->(st:{stream_node_label})
                RETURN COUNT(r) AS count
                """
            )
            initial_count = initial_count_result.single()["count"]
            self.logger.info(f"Initial relationship count: {initial_count}")

        tracked_created = 0
        tracked_skipped = 0

        try:
            with driver.session() as session:
                # Get all session nodes with their stream data
                session_query = f"""
                MATCH (s:{session_node_label})
                WHERE s.stream IS NOT NULL AND s.stream <> ''
                RETURN s.session_id as session_id, s.stream as stream
                """
                
                session_results = session.run(session_query)
                
                for session_record in session_results:
                    session_id = session_record["session_id"]
                    streams_str = session_record["stream"]
                    
                    if not streams_str:
                        continue
                    
                    # Split stream string by semicolon and normalize case
                    streams = [s.strip() for s in streams_str.split(";") if s.strip()]  # Removed .lower()
                    
                    for stream in streams:
                        # SIMPLE FIX: Check if relationship already exists using a more robust query
                        if create_only_new:
                            rel_exists_query = f"""
                            MATCH (s:{session_node_label} {{session_id: $session_id}})
                            MATCH (st:{stream_node_label})
                            WHERE LOWER(st.stream) = LOWER($stream)
                            WITH s, st
                            RETURN EXISTS((s)-[:HAS_STREAM]->(st)) AS exists
                            """
                            
                            rel_exists_result = session.run(rel_exists_query, session_id=session_id, stream=stream)
                            exists_record = rel_exists_result.single()
                            
                            if exists_record and exists_record["exists"]:
                                tracked_skipped += 1
                                continue

                        # BULLETPROOF: Use MERGE to create relationship only if it doesn't exist
                        create_query = f"""
                        MATCH (s:{session_node_label} {{session_id: $session_id}})
                        MATCH (st:{stream_node_label})
                        WHERE LOWER(st.stream) = LOWER($stream)
                        MERGE (s)-[r:HAS_STREAM]->(st)
                        RETURN COUNT(r) AS created
                        """

                        try:
                            result = session.run(create_query, session_id=session_id, stream=stream).single()
                            created_count = result["created"] if result else 0
                            
                            if created_count > 0:
                                tracked_created += created_count
                            elif created_count == 0:
                                # Relationship already existed or nodes not found
                                tracked_skipped += 1
                                
                        except Exception as e:
                            self.logger.warning(f"Error creating relationship: {str(e)}")

                # Get final relationship count to verify actual creation
                final_count_result = session.run(
                    f"""
                    MATCH (s:{session_node_label})-[r:HAS_STREAM]->(st:{stream_node_label})
                    RETURN COUNT(r) AS count
                    """
                )
                final_count = final_count_result.single()["count"]

                # Verify counts match
                actual_created = final_count - initial_count
                
                self.logger.info(f"Relationships: Initial={initial_count}, Final={final_count}")
                self.logger.info(f"Tracked created={tracked_created}, Actual created={actual_created}")
                
                if tracked_created != actual_created:
                    self.logger.warning(f"Relationship count mismatch! Tracked: {tracked_created}, Actual: {actual_created}")

        except Exception as e:
            self.logger.error(f"Error creating stream relationships: {str(e)}")
            raise
        finally:
            if 'driver' in locals():
                driver.close()

        self.logger.info(f"Relationships created: {tracked_created}, skipped: {tracked_skipped}")
        return tracked_created, tracked_skipped

    def _connect_new_sessions_to_visitors(self, new_sessions: List[Dict[str, Any]]) -> None:
        """Connect newly created Sessions_this_year nodes to visitors based on scan data."""

        scan_output_files = self.config.get("scan_output_files", {})
        sessions_visited_config = scan_output_files.get("sessions_visited", {})
        sessions_file = sessions_visited_config.get("this_year_post", "sessions_visited_this_year.csv")
        sessions_path = os.path.join(self.output_dir, "output", sessions_file)

        if not os.path.exists(sessions_path):
            self.logger.warning(
                "[post_analysis mode] Sessions visited file not found while wiring new sessions: %s",
                sessions_path,
            )
            return

        try:
            sessions_df = pd.read_csv(sessions_path)
        except Exception as read_error:
            self.logger.error(
                "[post_analysis mode] Unable to read sessions visited file %s: %s",
                sessions_path,
                read_error,
            )
            return

        if sessions_df.empty:
            self.logger.info("[post_analysis mode] Sessions visited file is empty; no visitor connections required")
            return

        badge_column = None
        for candidate in ["Badge Id", "BadgeId", "badge_id"]:
            if candidate in sessions_df.columns:
                badge_column = candidate
                break

        if not badge_column:
            self.logger.error(
                "[post_analysis mode] No badge column found in %s while wiring new sessions",
                sessions_path,
            )
            return

        if "key_text" not in sessions_df.columns:
            if "Seminar Name" in sessions_df.columns:
                sessions_df["key_text"] = sessions_df["Seminar Name"].astype(str)
            else:
                self.logger.error(
                    "[post_analysis mode] No key_text or Seminar Name column available in %s",
                    sessions_path,
                )
                return

        sessions_df["key_text_lower"] = (
            sessions_df["key_text"].astype(str).str.strip().str.lower()
        )
        sessions_df = sessions_df[sessions_df["key_text_lower"] != ""]

        if sessions_df.empty:
            self.logger.info("[post_analysis mode] No valid rows in sessions visited file after normalization")
            return

        grouped_sessions = {
            key: group.to_dict("records")
            for key, group in sessions_df.groupby("key_text_lower")
        }

        neo4j_config = self.config.get("neo4j", {})
        node_labels = neo4j_config.get("node_labels", {})
        relationships = neo4j_config.get("relationships", {})
        unique_ids = neo4j_config.get("unique_identifiers", {})

        visitor_label = node_labels.get("visitor_this_year", "Visitor_this_year")
        session_label = node_labels.get("session_this_year", "Sessions_this_year")
        relationship_name = relationships.get(
            "assisted_session_this_year", "assisted_session_this_year"
        )
        visitor_id_field = unique_ids.get("visitor", "BadgeId")

        driver = None
        created = 0
        skipped = 0
        failed = 0

        try:
            driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))

            with driver.session() as session:
                for session_record in new_sessions:
                    key_text = str(session_record.get("key_text", "")).strip().lower()
                    session_id = session_record.get("session_id")

                    if not key_text or not session_id:
                        skipped += 1
                        continue

                    matching_rows = grouped_sessions.get(key_text)
                    if not matching_rows:
                        continue

                    for row in matching_rows:
                        badge_value = str(row.get(badge_column, "")).strip()
                        if not badge_value:
                            skipped += 1
                            continue

                        scan_date = row.get("Scan Date")
                        file_name = row.get("File")
                        seminar_name = row.get("Seminar Name", row.get("Seminar"))

                        query = (
                            f"MATCH (v:{visitor_label} {{{visitor_id_field}: $badge_id}}) "
                            f"MATCH (s:{session_label} {{session_id: $session_id}}) "
                            "WHERE v.show = $show_name AND s.show = $show_name "
                            f"MERGE (v)-[r:{relationship_name}]->(s) "
                            "SET r.scan_date = coalesce($scan_date, r.scan_date), "
                            "    r.file = coalesce($file_name, r.file), "
                            "    r.seminar_name = coalesce($seminar_name, r.seminar_name) "
                            "RETURN r"
                        )

                        try:
                            result = session.run(
                                query,
                                badge_id=badge_value,
                                session_id=session_id,
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
                                "[post_analysis mode] Failed to merge assisted relationship for badge %s and session %s: %s",
                                badge_value,
                                session_id,
                                rel_error,
                            )
        except Exception as driver_error:
            failed += 1
            self.logger.error(
                "[post_analysis mode] Neo4j driver error while wiring new sessions: %s",
                driver_error,
            )
        finally:
            if driver:
                driver.close()

        self.logger.info(
            "[post_analysis mode] Visitor wiring summary for new sessions: %d relationships created, %d skipped, %d failed",
            created,
            skipped,
            failed,
        )

    def process(self, create_only_new: bool = True) -> Dict[str, Any]:
        """
        Process session data and create Neo4j nodes and relationships.
        Replicates the old processor's exact functionality.

        Args:
            create_only_new (bool): If True, only create nodes and relationships that don't exist.

        Returns:
            dict: Statistics about created and skipped nodes and relationships
        """
        self.logger.info("Starting Neo4j session and stream data processing")

        # Test the connection first
        if not self._test_connection():
            self.logger.error("Cannot proceed with Neo4j upload due to connection failure")
            return self.statistics

        try:
            # Detect post_analysis mode and override session file if needed
            mode = self.mode
            post_analysis_mode = mode == "post_analysis"
            if post_analysis_mode:
                scan_files = self.config.get("scan_files", {})
                this_year_csv_path = scan_files.get("session_this")
                if this_year_csv_path and os.path.exists(this_year_csv_path):
                    self.logger.info(f"[post_analysis mode] Using raw session export for this year: {this_year_csv_path}")
                else:
                    fallback_filename = self.session_processed_files.get(
                        "this_year", "session_this_filtered_valid_cols.csv"
                    )
                    this_year_csv_path = os.path.join(self.output_dir, "output", fallback_filename)
                    self.logger.warning(
                        f"[post_analysis mode] Raw session export not found; falling back to processed file {this_year_csv_path}"
                    )
            else:
                this_year_filename = self.session_processed_files.get(
                    "this_year", "session_this_filtered_valid_cols.csv"
                )
                this_year_csv_path = os.path.join(
                    self.output_dir, "output", this_year_filename
                )
            last_year_main_filename = self.session_processed_files.get(
                "last_year_main",
                f"session_last_filtered_valid_cols_{self.main_event_name}.csv",
            )
            last_year_secondary_filename = self.session_processed_files.get(
                "last_year_secondary",
                f"session_last_filtered_valid_cols_{self.secondary_event_name}.csv",
            )

            # Process sessions from this year
            self.logger.info("Processing sessions from this year")
            csv_file_path = this_year_csv_path
            data = pd.read_csv(csv_file_path)
            node_label = "Sessions_this_year"

            session_incremental_mode = create_only_new or post_analysis_mode
            new_sessions = []

            if session_incremental_mode:
                # INCREMENTAL MODE: Merge / update sessions without deleting existing ones
                if post_analysis_mode:
                    self.logger.info("[post_analysis mode] Updating Sessions_this_year nodes in-place without creating new nodes or touching related structures")
                else:
                    self.logger.info("Incremental mode enabled (create_only_new=True): merging Sessions_this_year; preserving IS_RECOMMENDED relationships")

                driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
                nodes_created = 0
                nodes_updated = 0
                has_stream_added = 0
                has_stream_removed = 0

                # Pre-compute column list for property setting (exclude session_id to avoid accidental overwrite of identifier semantics)
                all_columns = list(data.columns)
                for idx, row in data.iterrows():
                    try:
                        session_id = str(row.get("session_id", "")).strip()
                        if not session_id:
                            continue

                        # Build properties dictionary (stringify to keep consistency)
                        props = {c: ("" if pd.isna(row[c]) else str(row[c])) for c in all_columns if c != "session_id"}
                        props["show"] = self.show_name

                        # Skip placeholder sessions with no meaningful text
                        if all(props.get(f, "").strip() == "" for f in ["title", "synopsis_stripped", "theatre__name", "key_text", "stream"]):
                            nodes_updated += 1  # treat as skipped/update
                            continue

                        with driver.session() as session:
                            # MERGE the session node
                            merge_query = f"""
                            MERGE (s:{node_label} {{session_id:$session_id}})
                            ON CREATE SET s.created_at = timestamp(), s.show = $show
                            SET s += $props
                            RETURN (s.created_at IS NOT NULL) AS created_flag  // placeholder to satisfy RETURN
                            """
                            # Track creation vs update via an existence check of a temp flag
                            # We'll run an existence query before merge
                            exists_query = f"""
                            MATCH (s:{node_label} {{session_id:$session_id}})
                            RETURN count(s) AS cnt
                            """
                            pre = session.run(exists_query, session_id=session_id).single()["cnt"]
                            session.run(merge_query, session_id=session_id, show=self.show_name, props=props)
                            if pre == 0:
                                nodes_created += 1
                                if post_analysis_mode:
                                    new_sessions.append({
                                        "session_id": session_id,
                                        "key_text": props.get("key_text", ""),
                                    })
                            else:
                                nodes_updated += 1

                            if not post_analysis_mode:
                                # Stream handling (add/update relationships, create stream nodes if missing)
                                stream_field = row.get("stream", "")
                                # Normalize and split; allow multiple separated by ';'
                                raw_tokens = [t.strip() for t in str(stream_field).split(";") if t and str(t).strip()]
                                # Case-insensitive unique set while preserving first occurrence case
                                seen_lower = set()
                                tokens = []
                                for t in raw_tokens:
                                    tl = t.lower()
                                    if tl not in seen_lower:
                                        seen_lower.add(tl)
                                        tokens.append(t)

                                # Create missing Stream nodes & relationships
                                for token in tokens:
                                    stream_merge_query = """
                                        MERGE (st:Stream {stream:$stream_case, show:$show})
                                        ON CREATE SET st.description = coalesce(st.description, $default_desc)
                                        WITH st
                                        MATCH (s:Sessions_this_year {session_id:$session_id})
                                        SET s.show = coalesce(s.show, $show)
                                        MERGE (s)-[:HAS_STREAM]->(st)
                                        """
                                    session.run(stream_merge_query, stream_case=token, show=self.show_name, session_id=session_id, default_desc=f"Stream for {token}")
                                has_stream_added += len(tokens)

                                # Remove obsolete HAS_STREAM relationships (only those whose stream no longer listed)
                                if tokens:
                                    remove_query = """
                                        MATCH (s:Sessions_this_year {session_id:$session_id})-[r:HAS_STREAM]->(st:Stream {show:$show})
                                        WHERE NOT toLower(st.stream) IN $current_streams_lower
                                        DELETE r
                                        RETURN count(r) AS removed
                                        """
                                    removed = session.run(remove_query, session_id=session_id, show=self.show_name, current_streams_lower=[t.lower() for t in tokens]).single()["removed"]
                                    has_stream_removed += removed
                    except Exception as e:
                        self.logger.error(f"Incremental update failed for session_id={row.get('session_id')}: {e}")
                        continue

                driver.close()

                # Update statistics (treat updates as 'skipped' to preserve existing summary logic)
                self.statistics["nodes_created"]["sessions_this_year"] = nodes_created
                self.statistics["nodes_skipped"]["sessions_this_year"] = nodes_updated
                # Store additional incremental metrics
                self.statistics.setdefault("incremental_metrics", {})["sessions_this_year_updated"] = nodes_updated
                if not post_analysis_mode:
                    self.statistics["relationships_created"]["sessions_this_year_has_stream"] = has_stream_added
                    self.statistics["relationships_skipped"].setdefault("sessions_this_year_has_stream", 0)
                    self.statistics.setdefault("relationships_removed", {})["sessions_this_year_has_stream"] = has_stream_removed

                if post_analysis_mode and new_sessions:
                    self.logger.info(
                        "[post_analysis mode] Created %d new Sessions_this_year nodes from raw export; wiring visitor relationships",
                        len(new_sessions),
                    )
                    self._connect_new_sessions_to_visitors(new_sessions)

                self.logger.info(
                    "Incremental session processing complete: %s created, %s updated%s",
                    nodes_created,
                    nodes_updated,
                    "; stream relationships untouched" if post_analysis_mode else f"; HAS_STREAM added tokens total={has_stream_added}, removed relationships={has_stream_removed}"
                )
            else:
                # FULL REBUILD MODE (create_only_new=False) retains original delete + recreate semantics
                self.logger.info("Full rebuild mode: recreating all Sessions_this_year nodes (will remove existing HAS_STREAM & IS_RECOMMENDED relationships to those sessions)")
                properties_map = {col: col for col in data.columns}

                # Delete existing Sessions_this_year nodes and their relationships
                driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
                try:
                    with driver.session() as session:
                        delete_query = """
                        MATCH (s:Sessions_this_year)
                        WHERE s.show = $show
                        DETACH DELETE s
                        """
                        session.run(delete_query, show=self.show_name)
                        self.logger.info("Deleted all existing Sessions_this_year nodes")
                finally:
                    driver.close()

                nodes_created, nodes_skipped = self.load_csv_to_neo4j(
                    csv_file_path, node_label, properties_map, "session_id", create_only_new=False
                )
                self.statistics["nodes_created"]["sessions_this_year"] = nodes_created
                self.statistics["nodes_skipped"]["sessions_this_year"] = nodes_skipped

            if not post_analysis_mode:
                # Process sessions from last year (main event)
                self.logger.info(f"Processing sessions from last year {self.main_event_name}")
                csv_file_path = os.path.join(
                    self.output_dir, "output", last_year_main_filename
                )
                data = pd.read_csv(csv_file_path)
                properties_map = {col: col for col in data.columns}
                node_label = "Sessions_past_year"

                nodes_created, nodes_skipped = self.load_csv_to_neo4j(
                    csv_file_path, node_label, properties_map, "session_id", create_only_new
                )

                self.statistics["nodes_created"][f"sessions_past_year_{self.main_event_name}"] = nodes_created
                self.statistics["nodes_skipped"][f"sessions_past_year_{self.main_event_name}"] = nodes_skipped
                
                # Update backward compatible keys
                self.statistics["nodes_created"]["sessions_past_year_bva"] = nodes_created
                self.statistics["nodes_skipped"]["sessions_past_year_bva"] = nodes_skipped

                # Process sessions from last year (secondary event)
                self.logger.info(f"Processing sessions from last year {self.secondary_event_name}")
                csv_file_path = os.path.join(
                    self.output_dir, "output", last_year_secondary_filename
                )
                data = pd.read_csv(csv_file_path)
                properties_map = {col: col for col in data.columns}
                node_label = "Sessions_past_year"

                nodes_created, nodes_skipped = self.load_csv_to_neo4j(
                    csv_file_path, node_label, properties_map, "session_id", create_only_new
                )

                self.statistics["nodes_created"][f"sessions_past_year_{self.secondary_event_name}"] = nodes_created
                self.statistics["nodes_skipped"][f"sessions_past_year_{self.secondary_event_name}"] = nodes_skipped
                
                # Update backward compatible keys
                self.statistics["nodes_created"]["sessions_past_year_lva"] = nodes_created
                self.statistics["nodes_skipped"]["sessions_past_year_lva"] = nodes_skipped

                # CRITICAL FIX: Add combined sessions_past_year statistics that summary_utils.py expects
                main_nodes_created = self.statistics["nodes_created"].get(f"sessions_past_year_{self.main_event_name}", 0)
                main_nodes_skipped = self.statistics["nodes_skipped"].get(f"sessions_past_year_{self.main_event_name}", 0)

                secondary_nodes_created = self.statistics["nodes_created"].get(f"sessions_past_year_{self.secondary_event_name}", 0)
                secondary_nodes_skipped = self.statistics["nodes_skipped"].get(f"sessions_past_year_{self.secondary_event_name}", 0)

                # Create the combined sessions_past_year key that summary_utils.py expects
                self.statistics["nodes_created"]["sessions_past_year"] = main_nodes_created + secondary_nodes_created
                self.statistics["nodes_skipped"]["sessions_past_year"] = main_nodes_skipped + secondary_nodes_skipped

                # Create stream nodes
                self.logger.info("Creating stream nodes")
                streams_file_path = os.path.join(
                    self.output_dir, "output", self.streams_catalog_file
                )
                nodes_created, nodes_skipped = self.create_stream_nodes(streams_file_path, create_only_new)

                self.statistics["nodes_created"]["streams"] = nodes_created
                self.statistics["nodes_skipped"]["streams"] = nodes_skipped

                # Create relationships between sessions this year and streams (only in full rebuild mode)
                if not create_only_new:
                    self.logger.info("Creating stream relationships for sessions this year (full rebuild mode)")
                    rels_created, rels_skipped = self.create_stream_relationships(
                        "Sessions_this_year", "Stream", create_only_new
                    )
                    # Only overwrite if not already populated by incremental logic
                    self.statistics["relationships_created"]["sessions_this_year_has_stream"] = rels_created
                    self.statistics["relationships_skipped"]["sessions_this_year_has_stream"] = rels_skipped

                # Create relationships between sessions past year and streams
                self.logger.info("Creating stream relationships for sessions past year")
                rels_created, rels_skipped = self.create_stream_relationships(
                    "Sessions_past_year", "Stream", create_only_new
                )

                self.statistics["relationships_created"]["sessions_past_year_has_stream"] = rels_created
                self.statistics["relationships_skipped"]["sessions_past_year_has_stream"] = rels_skipped
            else:
                self.logger.info("[post_analysis mode] Skipping updates to past-year sessions, stream nodes, and HAS_STREAM relationships")
                self.statistics["nodes_created"].setdefault(f"sessions_past_year_{self.main_event_name}", 0)
                self.statistics["nodes_skipped"].setdefault(f"sessions_past_year_{self.main_event_name}", 0)
                self.statistics["nodes_created"].setdefault(f"sessions_past_year_{self.secondary_event_name}", 0)
                self.statistics["nodes_skipped"].setdefault(f"sessions_past_year_{self.secondary_event_name}", 0)
                self.statistics["nodes_created"]["sessions_past_year"] = 0
                self.statistics["nodes_skipped"]["sessions_past_year"] = 0
                self.statistics["nodes_created"].setdefault("streams", 0)
                self.statistics["nodes_skipped"].setdefault("streams", 0)
                self.statistics["relationships_created"].setdefault("sessions_past_year_has_stream", 0)
                self.statistics["relationships_skipped"].setdefault("sessions_past_year_has_stream", 0)

            # Log summary
            self.logger.info("Neo4j session data processing summary:")
            total_nodes = (self.statistics["nodes_created"]["sessions_this_year"] + 
                          self.statistics["nodes_created"]["sessions_past_year_bva"] + 
                          self.statistics["nodes_created"]["sessions_past_year_lva"] + 
                          self.statistics["nodes_created"]["streams"])
            total_skipped = (self.statistics["nodes_skipped"]["sessions_this_year"] + 
                           self.statistics["nodes_skipped"]["sessions_past_year_bva"] + 
                           self.statistics["nodes_skipped"]["sessions_past_year_lva"] + 
                           self.statistics["nodes_skipped"]["streams"])
            
            self.logger.info(f"Total nodes created: {total_nodes}, skipped: {total_skipped}")
            
            total_relationships = (self.statistics["relationships_created"]["sessions_this_year_has_stream"] + 
                                 self.statistics["relationships_created"]["sessions_past_year_has_stream"])
            total_rel_skipped = (self.statistics["relationships_skipped"]["sessions_this_year_has_stream"] + 
                               self.statistics["relationships_skipped"]["sessions_past_year_has_stream"])
            
            self.logger.info(f"Total relationships created: {total_relationships}, skipped: {total_rel_skipped}")
            self.logger.info("Neo4j session data processing completed")

        except Exception as e:
            self.logger.error(f"Error in Neo4j session processing: {str(e)}")
            raise

        return self.statistics


def main():
    """Main function for testing."""
    import sys
    sys.path.insert(0, os.getcwd())
    
    from utils.config_utils import load_config
    from utils.logging_utils import setup_logging

    # Setup logging
    logger = setup_logging(log_file="logs/neo4j_session_processor.log")
    
    try:
        # Load configuration
        config = load_config("config/config_vet.yaml")
        
        # Create processor and run
        processor = Neo4jSessionProcessor(config)
        stats = processor.process(create_only_new=False)
        
        print("Processing completed successfully!")
        print(f"Statistics: {stats}")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()