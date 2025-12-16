#!/usr/bin/env python3
"""
Enhanced Neo4j Job Stream Processor with Skip Support

A generic class to process job to stream mappings and create relationships in Neo4j.
This class is responsible for creating relationships between visitor nodes and
stream nodes based on job roles.

ENHANCED: Supports skipping processing for specific shows via configuration.
GENERIC: Supports configurable node labels, relationship names, field names, and show filtering.
Compatible with both BVA and ECOMM show configurations.
"""

import logging
import os
import pandas as pd
import json
from neo4j import GraphDatabase
from dotenv import load_dotenv
import inspect

from utils.neo4j_utils import resolve_neo4j_credentials


class Neo4jJobStreamProcessor:
    """
    Enhanced Neo4j Job Stream Processor with configurable skip support.
    
    NEW FEATURES:
    - Can skip processing entirely based on config (for ECOMM/TFM shows)
    - Enhanced configuration validation
    - Better error handling and logging
    """

    def __init__(self, config):
        """
        Initialize the Neo4j Job Stream Processor.

        Args:
            config (dict): Configuration dictionary containing paths and settings
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Initializing Enhanced Neo4j Job Stream Processor"
        )

        self.config = config
        self.input_dir = config["output_dir"]

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

        # ENHANCED: Get configuration values with fallbacks for backward compatibility
        self.neo4j_config = config.get("neo4j", {})
        self.show_name = self.neo4j_config.get("show_name", None)  # None = backward compatibility mode
        self.node_labels = self.neo4j_config.get("node_labels", {})
        self.relationships = self.neo4j_config.get("relationships", {})
        self.job_stream_mapping_config = self.neo4j_config.get("job_stream_mapping", {})
        
        # NEW: Check if job stream processing is enabled for this show
        self.job_stream_enabled = self.job_stream_mapping_config.get("enabled", True)
        
        # Get node labels with fallback to original hardcoded values
        self.visitor_label = self.node_labels.get("visitor_this_year", "Visitor_this_year")
        self.stream_label = self.node_labels.get("stream", "Stream")
        self.relationship_name = self.relationships.get("job_stream", "job_to_stream")
        
        # Get job role field name with fallback
        self.job_role_field = self.job_stream_mapping_config.get("job_role_field", "job_role")
        
        # Get mapping file from config
        self.job_stream_mapping_file = self.job_stream_mapping_config.get("file", "job_to_stream.csv")

        # Initialize statistics
        self.statistics = {
            "relationships_created": 0,
            "relationships_skipped": 0,
            "relationships_not_found": 0,
            "stream_mappings_applied": 0,
            "visitor_nodes_processed": 0,
            "job_roles_processed": 0,
            "stream_matches_found": 0,
            "initial_relationship_count": 0,
            "final_relationship_count": 0,
            "processing_skipped": False,
            "skip_reason": None
        }

        # Log configuration details
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Configuration: show_name='{self.show_name}', enabled={self.job_stream_enabled}"
        )

    def _test_connection(self):
        """Test the Neo4j connection."""
        try:
            driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
            with driver.session() as session:
                result = session.run("MATCH (n) RETURN count(n) AS count LIMIT 1")
                count = result.single()["count"]
                self.logger.info(
                    f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Neo4j connection successful. Database contains {count} nodes"
                )
            driver.close()
            return True
        except Exception as e:
            self.logger.error(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Failed to connect to Neo4j: {str(e)}"
            )
            return False

    def should_skip_processing(self):
        """
        Determine if processing should be skipped for this configuration.
        
        NEW: Checks configuration to determine if job stream processing should be skipped.
        
        Returns:
            tuple: (should_skip: bool, reason: str)
        """
        # Check if job stream processing is explicitly disabled
        if not self.job_stream_enabled:
            return True, f"Job stream processing disabled in config for show '{self.show_name}'"
        
        # Check if mapping file exists - try multiple locations
        possible_paths = [
            os.path.join(self.input_dir, self.job_stream_mapping_file),  # output_dir/file
            self.job_stream_mapping_file,  # current directory
            os.path.join("data", "bva", self.job_stream_mapping_file),  # data/bva/file
        ]
        
        for mapping_file_path in possible_paths:
            if os.path.exists(mapping_file_path):
                return False, None
        
        return True, f"Job stream mapping file not found: {self.job_stream_mapping_file} (searched: {', '.join(possible_paths)})"

    def load_job_stream_mapping(self, mapping_file_path):
        """
        Load the job role to stream mapping from a CSV file.

        Args:
            mapping_file_path (str): Path to the CSV file containing the mapping.

        Returns:
            dict: Dictionary mapping job roles to streams.
        """
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Loading job to stream mapping from: {mapping_file_path}"
        )

        try:
            map_job_stream = pd.read_csv(mapping_file_path)

            # Verify the CSV has the expected structure
            if "Job Role" not in map_job_stream.columns:
                self.logger.error(
                    f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - CSV file does not contain 'Job Role' column"
                )
                raise ValueError("CSV file does not contain 'Job Role' column")

            # Convert the DataFrame to a dictionary using job role as the index
            job_stream_mapping = json.loads(
                map_job_stream.set_index("Job Role").to_json(orient="index")
            )

            # Log some stats about the mapping
            self.logger.info(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Loaded mapping for {len(job_stream_mapping)} job roles"
            )

            return job_stream_mapping
        except Exception as e:
            self.logger.error(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Error loading job to stream mapping: {str(e)}"
            )
            raise

    def _normalize_stream_name(self, stream_name):
        """
        Apply normalization rules to stream names.
        ENHANCED: Ensures consistent lowercase handling.

        Args:
            stream_name (str): Original stream name

        Returns:
            tuple: (normalized_stream_name, mapping_applied)
        """
        if not stream_name:
            return stream_name, False

        # First normalize to lowercase
        normalized_name = stream_name.strip()
        
        # Apply any additional mapping rules here if needed
        mapping_applied = False
        
        # Known synonyms -> canonical names (based on streams.json)
        lower_name = normalized_name.lower()
        synonyms = {
            "emergency and critical care (ecc)": "Emergency Medicine",
            "ecc": "Emergency Medicine",
        }
        if lower_name in synonyms:
            normalized_name = synonyms[lower_name]
            mapping_applied = True

        return normalized_name, mapping_applied

    def _count_relationships(self, session):
        """
        Count the total number of job_to_stream relationships.
        GENERIC: Uses configurable relationship and node labels.
        """
        query = f"""
        MATCH (:{self.visitor_label})-[r:{self.relationship_name}]->(:{self.stream_label})
        RETURN COUNT(r) AS count
        """
        
        # Add show filtering if configured
        if self.show_name:
            query = f"""
            MATCH (v:{self.visitor_label})-[r:{self.relationship_name}]->(s:{self.stream_label})
            WHERE v.show = $show_name AND s.show = $show_name
            RETURN COUNT(r) AS count
            """
            result = session.run(query, show_name=self.show_name)
        else:
            result = session.run(query)
            
        return result.single()["count"]

    def create_job_stream_relationships(self, job_stream_mapping, create_only_new=True):
        """
        Create relationships between visitor nodes and stream nodes based on job roles.
        ENHANCED: Consistent lowercase stream name handling throughout.
        GENERIC: Supports configurable node labels, relationship names, and show filtering.

        Args:
            job_stream_mapping (dict): Dictionary mapping job roles to streams.
            create_only_new (bool): If True, only create relationships that don't exist.

        Returns:
            dict: Statistics about the relationship creation process.
        """
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Creating job to stream relationships with enhanced tracking and consistent lowercase handling"
        )

        driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))

        # First count the initial relationships
        with driver.session() as session:
            initial_count = self._count_relationships(session)
            self.statistics["initial_relationship_count"] = initial_count
            self.logger.info(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Initial relationship count: {initial_count}"
            )

        # Batch processing function with enhanced tracking and consistent lowercase handling
        def batch_create_relationships(tx, job_stream_mapping, create_only_new):
            batch_stats = {
                "relationships_created": 0,
                "relationships_skipped": 0,
                "relationships_not_found": 0,
                "stream_mappings_applied": 0,
                "visitor_nodes_processed": 0,
                "job_roles_processed": 0,
                "stream_matches_found": 0,
            }

            # Get all unique job roles from the database that match our mapping
            # GENERIC: Use configurable field names and labels, with show filtering
            additional_filter = ""
            if create_only_new:
                additional_filter = (
                    ' AND (v.has_recommendation IS NULL OR v.has_recommendation = "0")'
                )

            show_filter = ""
            if self.show_name:
                show_filter = ' AND v.show = $show_name'

            job_roles_query = f"""
            MATCH (v:{self.visitor_label})
            WHERE v.{self.job_role_field} IS NOT NULL 
              AND LOWER(v.{self.job_role_field}) <> 'na'
              AND v.{self.job_role_field} IN $job_roles
              {show_filter}
              {additional_filter}
            RETURN DISTINCT v.{self.job_role_field} as job_role
            """

            query_params = {"job_roles": list(job_stream_mapping.keys())}
            if self.show_name:
                query_params["show_name"] = self.show_name

            job_roles_result = tx.run(job_roles_query, **query_params)
            matched_job_roles = [record["job_role"] for record in job_roles_result]
            batch_stats["job_roles_processed"] = len(matched_job_roles)

            # Process each job role
            for job_role in matched_job_roles:
                # Get applicable streams for this job role with consistent lowercase normalization
                applicable_streams = []
                mapped_streams = []

                for stream_name, applies in job_stream_mapping[job_role].items():
                    if applies == "YES":
                        # ENHANCED: Apply consistent lowercase normalization
                        normalized_stream_name, mapping_applied = self._normalize_stream_name(stream_name)

                        if mapping_applied:
                            batch_stats["stream_mappings_applied"] += 1
                            self.logger.info(
                                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Applied stream mapping: '{stream_name}' → '{normalized_stream_name}'"
                            )

                        mapped_streams.append(normalized_stream_name)

                # Get unique streams after mapping and normalization
                applicable_streams = list(set(mapped_streams))
                batch_stats["stream_matches_found"] += len(applicable_streams)

                # Count visitors with this job role (with show filtering if configured)
                count_filter = ""
                if self.show_name:
                    count_filter = ' AND v.show = $show_name'

                count_query = f"""
                MATCH (v:{self.visitor_label})
                WHERE v.{self.job_role_field} = $job_role{count_filter}
                RETURN count(v) AS visitor_count
                """

                count_params = {"job_role": job_role}
                if self.show_name:
                    count_params["show_name"] = self.show_name

                count_result = tx.run(count_query, **count_params)
                visitor_count = count_result.single()["visitor_count"]
                batch_stats["visitor_nodes_processed"] += visitor_count

                # Process each applicable stream
                for stream_name in applicable_streams:
                    # Find the actual stream name in Neo4j (case-insensitive search)
                    stream_search_filter = ""
                    if self.show_name:
                        stream_search_filter = ' AND s.show = $show_name'

                    stream_query = f"""
                    MATCH (s:{self.stream_label})
                    WHERE LOWER(s.stream) = LOWER($stream_name){stream_search_filter}
                    RETURN s.stream as actual_stream_name
                    LIMIT 1
                    """

                    stream_params = {"stream_name": stream_name}
                    if self.show_name:
                        stream_params["show_name"] = self.show_name

                    stream_result = tx.run(stream_query, **stream_params)
                    stream_record = stream_result.single()

                    if not stream_record:
                        batch_stats["relationships_not_found"] += visitor_count
                        self.logger.warning(
                            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Stream '{stream_name}' not found in Neo4j for job role '{job_role}'"
                        )
                        continue

                    actual_stream_name = stream_record["actual_stream_name"]
          
                    # Log normalization if different
                    if stream_name != actual_stream_name:
                        self.logger.info(
                            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Stream name normalization: '{stream_name}' → '{actual_stream_name}'"
                        )

                    # Create relationships for this job role and stream
                    # GENERIC: Use configurable labels and relationship names with show filtering
                    visitor_show_filter = ""
                    stream_show_filter = ""
                    if self.show_name:
                        visitor_show_filter = ' AND v.show = $show_name'
                        stream_show_filter = ' AND s.show = $show_name'

                    if create_only_new:
                        # Check for existing relationships first
                        create_query = f"""
                        MATCH (v:{self.visitor_label})
                        WHERE v.{self.job_role_field} = $job_role{visitor_show_filter}
                        
                        MATCH (s:{self.stream_label})
                        WHERE LOWER(s.stream) = LOWER($actual_stream_name){stream_show_filter}
                        
                        WITH v, s
                        WHERE NOT exists((v)-[:{self.relationship_name}]->(s))
                        
                        CREATE (v)-[r:{self.relationship_name}]->(s)
                        RETURN count(r) AS created
                        """

                        create_params = {
                            "job_role": job_role, 
                            "actual_stream_name": actual_stream_name
                        }
                        if self.show_name:
                            create_params["show_name"] = self.show_name

                        result = tx.run(create_query, **create_params)
                        created = result.single()["created"]

                        # Calculate skipped relationships
                        potential = visitor_count
                        skipped = potential - created

                        batch_stats["relationships_created"] += created
                        batch_stats["relationships_skipped"] += skipped
                    else:
                        # Create all possible relationships using MERGE
                        create_query = f"""
                        MATCH (v:{self.visitor_label})
                        WHERE v.{self.job_role_field} = $job_role{visitor_show_filter}
                        
                        MATCH (s:{self.stream_label})
                        WHERE LOWER(s.stream) = LOWER($actual_stream_name){stream_show_filter}
                        
                        MERGE (v)-[r:{self.relationship_name}]->(s)
                        RETURN count(r) AS created
                        """

                        create_params = {
                            "job_role": job_role, 
                            "actual_stream_name": actual_stream_name
                        }
                        if self.show_name:
                            create_params["show_name"] = self.show_name

                        result = tx.run(create_query, **create_params)
                        created = result.single()["created"]
                        batch_stats["relationships_created"] += created

            return batch_stats

        try:
            with driver.session() as session:
                batch_stats = session.execute_write(
                    batch_create_relationships, job_stream_mapping, create_only_new
                )

                # Update our statistics
                self.statistics.update(batch_stats)

                # Now count the final relationships
                final_count = self._count_relationships(session)
                self.statistics["final_relationship_count"] = final_count

                # Calculate the actual relationships created based on database counts
                actual_created = final_count - initial_count

                # Check for discrepancies and log them
                if actual_created != self.statistics["relationships_created"]:
                    self.logger.warning(
                        f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Discrepancy detected! "
                        f"Database shows {actual_created} new relationships, but code tracked {self.statistics['relationships_created']}."
                    )
                    self.logger.warning(
                        f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - This may indicate concurrent database access or transaction issues."
                    )

                self.logger.info(
                    f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Initial relationship count: {initial_count}"
                )
                self.logger.info(
                    f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Final relationship count: {final_count}"
                )
                self.logger.info(
                    f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Job to stream relationships - created: {self.statistics['relationships_created']}, skipped: {self.statistics['relationships_skipped']}"
                )
                self.logger.info(
                    f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Processed {self.statistics['visitor_nodes_processed']} visitor nodes with {self.statistics['job_roles_processed']} matching job roles"
                )

                if self.statistics["stream_mappings_applied"] > 0:
                    self.logger.info(
                        f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Stream name mappings applied: {self.statistics['stream_mappings_applied']}"
                    )

        except Exception as e:
            self.logger.error(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Error creating job stream relationships: {str(e)}"
            )
            raise
        finally:
            driver.close()

    def print_reconciliation_report(self):
        """Print a detailed reconciliation report."""
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - === JOB STREAM PROCESSOR RECONCILIATION REPORT ==="
        )
        
        if self.statistics["processing_skipped"]:
            self.logger.info(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Processing skipped: {self.statistics['skip_reason']}"
            )
            return

        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Relationships created: {self.statistics['relationships_created']}"
        )
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Relationships skipped (already exist): {self.statistics['relationships_skipped']}"
        )

        if self.statistics["relationships_not_found"] > 0:
            self.logger.warning(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Relationships not created due to missing nodes: {self.statistics['relationships_not_found']}"
            )

        if self.statistics["stream_mappings_applied"] > 0:
            self.logger.info(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Stream name mappings applied: {self.statistics['stream_mappings_applied']}"
            )

        # Check for discrepancies
        actual_created = (
            self.statistics["final_relationship_count"]
            - self.statistics["initial_relationship_count"]
        )
        if actual_created != self.statistics["relationships_created"]:
            self.logger.warning(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - WARNING: Discrepancy detected! "
                f"Database shows {actual_created} new relationships, but code tracked {self.statistics['relationships_created']}."
            )
            self.logger.warning(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - This may indicate concurrent database access or transaction issues."
            )

    def process(self, create_only_new=True):
        """
        Process job to stream mappings and create relationships in Neo4j.
        ENHANCED: Added skip logic for shows that don't need job stream processing.

        Args:
            create_only_new (bool): If True, only create relationships that don't exist.
        """
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Starting Enhanced Neo4j job to stream relationship processing"
        )

        # NEW: Check if processing should be skipped
        should_skip, skip_reason = self.should_skip_processing()
        
        if should_skip:
            self.logger.info(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Skipping job stream processing: {skip_reason}"
            )
            self.statistics["processing_skipped"] = True
            self.statistics["skip_reason"] = skip_reason
            return

        # Test the connection first
        if not self._test_connection():
            self.logger.error(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Cannot proceed with Neo4j processing due to connection failure"
            )
            return

        # Get the mapping file path - try multiple locations
        possible_paths = [
            os.path.join(self.input_dir, self.job_stream_mapping_file),  # output_dir/file
            self.job_stream_mapping_file,  # current directory
            os.path.join("data", "bva", self.job_stream_mapping_file),  # data/bva/file
        ]
        
        mapping_file_path = None
        for path in possible_paths:
            if os.path.exists(path):
                mapping_file_path = path
                break
        
        if not mapping_file_path:
            self.logger.error(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - "
                f"Job stream mapping file not found. Searched: {', '.join(possible_paths)}"
            )
            return

        try:
            # Load the mapping
            job_stream_mapping = self.load_job_stream_mapping(mapping_file_path)

            # Create the relationships
            self.create_job_stream_relationships(job_stream_mapping, create_only_new)

            # Print detailed reconciliation report
            self.print_reconciliation_report()

            self.logger.info(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Enhanced Neo4j job to stream relationship processing completed"
            )
        except Exception as e:
            self.logger.error(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Error in job to stream relationship processing: {str(e)}"
            )
            raise