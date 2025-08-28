#!/usr/bin/env python3
"""
Enhanced Neo4j Specialization Stream Processor with Skip Support

A generic class to process specialization to stream mappings and create relationships in Neo4j.
This class is responsible for creating relationships between visitor nodes and
stream nodes based on practice specializations.

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


class Neo4jSpecializationStreamProcessor:
    """
    Enhanced Neo4j Specialization Stream Processor with configurable skip support.
    
    NEW FEATURES:
    - Can skip processing entirely based on config (for ECOMM/TFM shows)
    - Enhanced configuration validation
    - Better error handling and logging
    """

    def __init__(self, config):
        """
        Initialize the Neo4j Specialization Stream Processor.

        Args:
            config (dict): Configuration dictionary containing paths and settings
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Initializing Enhanced Neo4j Specialization Stream Processor"
        )

        self.config = config
        self.input_dir = config["output_dir"]

        # Load the environment variables to get Neo4j credentials
        load_dotenv(config["env_file"])
        self.uri = os.getenv("NEO4J_URI")
        self.username = os.getenv("NEO4J_USERNAME")
        self.password = os.getenv("NEO4J_PASSWORD")

        if not self.uri or not self.username or not self.password:
            self.logger.error(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Missing Neo4j credentials in .env file"
            )
            raise ValueError("Missing Neo4j credentials in .env file")

        # ENHANCED: Get configuration values with fallbacks for backward compatibility
        self.neo4j_config = config.get("neo4j", {})
        self.show_name = self.neo4j_config.get("show_name", None)  # None = backward compatibility mode
        self.node_labels = self.neo4j_config.get("node_labels", {})
        self.relationships = self.neo4j_config.get("relationships", {})
        self.specialization_stream_mapping_config = self.neo4j_config.get("specialization_stream_mapping", {})
        
        # NEW: Check if specialization stream processing is enabled for this show
        self.specialization_stream_enabled = self.specialization_stream_mapping_config.get("enabled", True)
        
        # Get configuration values for specialization mapping
        self.specialization_field_this_year = self.specialization_stream_mapping_config.get(
            "specialization_field_this_year", "what_type_does_your_practice_specialise_in"
        )
        self.specialization_field_bva = self.specialization_stream_mapping_config.get(
            "specialization_field_bva", "what_areas_do_you_specialise_in"
        )
        self.specialization_field_lva = self.specialization_stream_mapping_config.get(
            "specialization_field_lva", "what_areas_do_you_specialise_in"
        )
        
        # Get node labels with fallback to original hardcoded values
        self.visitor_this_year_label = self.node_labels.get("visitor_this_year", "Visitor_this_year")
        self.visitor_last_year_bva_label = self.node_labels.get("visitor_last_year_bva", "Visitor_last_year_bva")
        self.visitor_last_year_lva_label = self.node_labels.get("visitor_last_year_lva", "Visitor_last_year_lva")
        self.stream_label = self.node_labels.get("stream", "Stream")
        self.relationship_name = self.relationships.get("specialization_stream", "specialization_to_stream")

        # Get mapping files from config
        self.specialization_mapping_file = self.specialization_stream_mapping_config.get(
            "file", "spezialization_to_stream.csv"
        )
        self.specialization_map = self.specialization_stream_mapping_config.get(
            "specialization_map", {}
        )
        
        # Fallback to default mapping if not configured
        if not self.specialization_map:
            self.specialization_map = {
                "Wildlife": "Other",
                "Small Animal": "Companion Animal",
                "Mixed": "Other",
                "Dairy": "Farm",
                "Poultry": "Farm",
                "Cattle": "Farm",
                "Pigs": "Farm",
                "Sheep": "Farm",
            }

        # Initialize statistics
        self.statistics = {
            "initial_count": 0,
            "final_count": 0,
            "visitor_nodes_processed": {"this_year": 0, "last_year_bva": 0, "last_year_lva": 0},
            "specializations_processed": 0,
            "specializations_mapped": 0,
            "stream_matches_found": 0,
            "relationships_created": 0,
            "relationships_skipped": 0,
            "relationships_not_found": 0,
            "processing_skipped": False,
            "skip_reason": None
        }

        # Log configuration details
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Configuration: show_name='{self.show_name}', enabled={self.specialization_stream_enabled}"
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
        
        NEW: Checks configuration to determine if specialization stream processing should be skipped.
        
        Returns:
            tuple: (should_skip: bool, reason: str)
        """
        # Check if specialization stream processing is explicitly disabled
        if not self.specialization_stream_enabled:
            return True, f"Specialization stream processing disabled in config for show '{self.show_name}'"
        
        # Check if mapping file exists - try multiple locations
        possible_paths = [
            os.path.join(self.input_dir, self.specialization_mapping_file),  # output_dir/file
            self.specialization_mapping_file,  # current directory
            os.path.join("data", "bva", self.specialization_mapping_file),  # data/bva/file
        ]
        
        for mapping_file_path in possible_paths:
            if os.path.exists(mapping_file_path):
                return False, None
        
        return True, f"Specialization mapping file not found: {self.specialization_mapping_file} (searched: {', '.join(possible_paths)})"

    def load_mappings(self):
        """
        Load the specialization mappings from CSV files.
        GENERIC: Uses configurable file paths and mapping configuration.

        Returns:
            tuple: (map_specialization, specialization_stream_mapping)
        """
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Loading specialization mappings"
        )

        try:
            # Use the configured specialization mapping
            map_specialization = self.specialization_map

            # Load specialization to stream mapping from configured file - try multiple locations
            possible_paths = [
                os.path.join(self.input_dir, self.specialization_mapping_file),  # output_dir/file
                self.specialization_mapping_file,  # current directory
                os.path.join("data", "bva", self.specialization_mapping_file),  # data/bva/file
            ]
            
            specialization_stream_file = None
            for path in possible_paths:
                if os.path.exists(path):
                    specialization_stream_file = path
                    break
                
            if not specialization_stream_file:
                raise FileNotFoundError(
                    f"Specialization mapping file not found: {self.specialization_mapping_file}. "
                    f"Searched: {', '.join(possible_paths)}"
                )
                
            map_specialization_stream = pd.read_csv(specialization_stream_file)

            # Convert to dictionary
            specialization_stream_mapping = json.loads(
                map_specialization_stream.set_index("spezialization").to_json(
                    orient="index"
                )
            )

            # Log some stats about the mapping
            self.logger.info(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Loaded mapping for {len(specialization_stream_mapping)} specializations"
            )

            return map_specialization, specialization_stream_mapping
        except Exception as e:
            self.logger.error(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Error loading specialization mappings: {str(e)}"
            )
            raise

    def _count_relationships(self, session):
        """
        Count the total number of specialization_to_stream relationships.
        GENERIC: Uses configurable relationship and node labels with show filtering.
        """
        query = f"""
        MATCH (:{self.visitor_this_year_label})-[r:{self.relationship_name}]->(:{self.stream_label})
        RETURN COUNT(r) AS count
        """
        
        # Add show filtering if configured
        if self.show_name:
            query = f"""
            MATCH (v:{self.visitor_this_year_label})-[r:{self.relationship_name}]->(s:{self.stream_label})
            WHERE v.show = $show_name AND s.show = $show_name
            RETURN COUNT(r) AS count
            """
            result = session.run(query, show_name=self.show_name)
        else:
            result = session.run(query)
            
        return result.single()["count"]

    def process_visitor_specializations(
        self, tx, node_label, specialization_field, map_specialization, specialization_stream_mapping, create_only_new
    ):
        """
        Process specializations for a specific visitor node type.
        GENERIC: Supports configurable node labels, field names, and show filtering.

        Args:
            tx: Neo4j transaction
            node_label (str): Label of the visitor nodes to process
            specialization_field (str): Field name containing specializations
            map_specialization (dict): Mapping for specialization names
            specialization_stream_mapping (dict): Mapping from specializations to streams
            create_only_new (bool): Whether to only create new relationships

        Returns:
            dict: Statistics for this visitor type processing
        """
        stats = {
            "visitor_nodes_processed": {node_label.lower(): 0},
            "specializations_processed": 0,
            "specializations_mapped": 0,
            "stream_matches_found": 0,
            "relationships_created": 0,
            "relationships_skipped": 0,
            "relationships_not_found": 0,
        }

        # GENERIC: Build query with show filtering if configured
        additional_filter = ""
        if create_only_new:
            additional_filter = ' AND (v.has_recommendation IS NULL OR v.has_recommendation = "0")'
        
        show_filter = ""
        if self.show_name:
            show_filter = ' AND v.show = $show_name'

        query = f"""
        MATCH (v:{node_label})
        WHERE v.{specialization_field} IS NOT NULL 
        AND LOWER(v.{specialization_field}) <> 'na'
        {show_filter}
        {additional_filter}
        RETURN v, v.{specialization_field} as specializations, 
            v.BadgeId as badge_id, v.Email as email
        """

        query_params = {}
        if self.show_name:
            query_params["show_name"] = self.show_name

        results = tx.run(query, **query_params)
        visitor_type_key = node_label.lower()

        for record in results:
            visitor = record["v"]
            specializations_text = record["specializations"]
            badge_id = record.get("badge_id")
            email = record.get("email")

            stats["visitor_nodes_processed"][visitor_type_key] += 1

            # Skip if BadgeId or Email is not available (needed for identification)
            if not badge_id or not email:
                continue

            # Skip processing if the specialization text is explicitly 'NA'
            if specializations_text.upper() == "NA":
                continue

            # Split specializations by semicolon
            specializations = specializations_text.split(";")

            # Process each specialization
            for spec in specializations:
                spec = spec.strip()

                # Map specialization if needed
                if spec in map_specialization:
                    mapped_spec = map_specialization[spec]
                    stats["specializations_mapped"] += 1
                else:
                    mapped_spec = spec

                # Check if this specialization exists in our mapping
                if mapped_spec in specialization_stream_mapping:
                    stream_dict = specialization_stream_mapping[mapped_spec]
                    stats["specializations_processed"] += 1

                    # For each stream that applies to this specialization
                    for stream_name, applies in stream_dict.items():
                        if applies == "YES":
                            stats["stream_matches_found"] += 1

                            # First find the actual stream in Neo4j (case-insensitive search)
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
                                stats["relationships_not_found"] += 1
                                continue

                            actual_stream_name = stream_record["actual_stream_name"]

                            # Create relationship between visitor and stream
                            visitor_show_filter = ""
                            stream_show_filter = ""
                            if self.show_name:
                                visitor_show_filter = ' AND v.show = $show_name'
                                stream_show_filter = ' AND s.show = $show_name'

                            if create_only_new:
                                # Only create if relationship doesn't exist
                                relationship_query = f"""
                                MATCH (v:{node_label})
                                WHERE v.BadgeId = $badge_id AND v.Email = $email{visitor_show_filter}
                                
                                MATCH (s:{self.stream_label})
                                WHERE LOWER(s.stream) = LOWER($actual_stream_name){stream_show_filter}
                                
                                WITH v, s
                                WHERE NOT exists((v)-[:{self.relationship_name}]->(s))
                                
                                CREATE (v)-[r:{self.relationship_name} {{created_by: 'specialization_processor'}}]->(s)
                                RETURN count(r) AS created
                                """
                            else:
                                # Create using MERGE
                                relationship_query = f"""
                                MATCH (v:{node_label})
                                WHERE v.BadgeId = $badge_id AND v.Email = $email{visitor_show_filter}
                                
                                MATCH (s:{self.stream_label})
                                WHERE LOWER(s.stream) = LOWER($actual_stream_name){stream_show_filter}
                                
                                MERGE (v)-[r:{self.relationship_name} {{created_by: 'specialization_processor'}}]->(s)
                                RETURN count(r) AS created
                                """

                            rel_params = {
                                "badge_id": badge_id,
                                "email": email,
                                "actual_stream_name": actual_stream_name
                            }
                            if self.show_name:
                                rel_params["show_name"] = self.show_name

                            result = tx.run(relationship_query, **rel_params)
                            created = result.single()["created"]

                            if created > 0:
                                stats["relationships_created"] += created
                            else:
                                stats["relationships_skipped"] += 1

        return stats

    def create_specialization_stream_relationships(
        self, map_specialization, specialization_stream_mapping, create_only_new=True
    ):
        """
        Create relationships between visitor nodes and stream nodes based on specializations.
        GENERIC: Supports configurable node labels, relationship names, and show filtering.

        Args:
            map_specialization (dict): Dictionary mapping old specialization names to new ones.
            specialization_stream_mapping (dict): Dictionary mapping specializations to streams.
            create_only_new (bool): If True, only create relationships that don't exist.

        Returns:
            dict: Statistics about the relationship creation process.
        """
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Creating specialization to stream relationships (generic mode)"
        )

        driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))

        # First count the initial relationships
        with driver.session() as session:
            initial_count = self._count_relationships(session)
            self.statistics["initial_count"] = initial_count
            self.logger.info(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Initial relationship count: {initial_count}"
            )

        # Process different visitor types
        visitor_types = [
            {
                "node_label": self.visitor_this_year_label,
                "specialization_field": self.specialization_field_this_year,
            },
            {
                "node_label": self.visitor_last_year_bva_label,
                "specialization_field": self.specialization_field_bva,
            },
            {
                "node_label": self.visitor_last_year_lva_label,
                "specialization_field": self.specialization_field_lva,
            },
        ]

        try:
            with driver.session() as session:
                # Process each visitor type
                for visitor_type in visitor_types:
                    node_label = visitor_type["node_label"]
                    specialization_field = visitor_type["specialization_field"]

                    self.logger.info(
                        f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Processing {node_label} with field '{specialization_field}'"
                    )

                    # Process in a transaction
                    visitor_stats = session.execute_write(
                        self.process_visitor_specializations,
                        node_label,
                        specialization_field,
                        map_specialization,
                        specialization_stream_mapping,
                        create_only_new,
                    )

                    # Merge statistics
                    for key, value in visitor_stats.items():
                        if key == "visitor_nodes_processed":
                            self.statistics[key].update(value)
                        else:
                            self.statistics[key] += value

                # Count final relationships
                final_count = self._count_relationships(session)
                self.statistics["final_count"] = final_count

                self.logger.info(
                    f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Final relationship count: {final_count}"
                )

        except Exception as e:
            self.logger.error(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Error creating specialization stream relationships: {str(e)}"
            )
            raise
        finally:
            driver.close()

    def print_reconciliation_report(self):
        """Print a detailed reconciliation report."""
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - === SPECIALIZATION STREAM PROCESSOR RECONCILIATION REPORT ==="
        )
        
        if self.statistics["processing_skipped"]:
            self.logger.info(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Processing skipped: {self.statistics['skip_reason']}"
            )
            return

        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Initial relationship count: {self.statistics['initial_count']}"
        )
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Final relationship count: {self.statistics['final_count']}"
        )
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Relationships created: {self.statistics['relationships_created']}"
        )
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Relationships skipped: {self.statistics['relationships_skipped']}"
        )

        # Log visitor processing stats
        for visitor_type, count in self.statistics["visitor_nodes_processed"].items():
            if count > 0:
                self.logger.info(
                    f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - {visitor_type} visitors processed: {count}"
                )

        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Specializations processed: {self.statistics['specializations_processed']}"
        )
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Stream mappings applied: {self.statistics['specializations_mapped']}"
        )
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Stream matches found: {self.statistics['stream_matches_found']}"
        )

        if self.statistics["relationships_not_found"] > 0:
            self.logger.warning(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Relationships not created due to missing nodes: {self.statistics['relationships_not_found']}"
            )

    def process(self, create_only_new=True):
        """
        Process specialization to stream mappings and create relationships in Neo4j.
        ENHANCED: Added skip logic for shows that don't need specialization stream processing.

        Args:
            create_only_new (bool): If True, only create relationships that don't exist.
        """
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Starting Enhanced Neo4j specialization to stream relationship processing"
        )

        # NEW: Check if processing should be skipped
        should_skip, skip_reason = self.should_skip_processing()
        
        if should_skip:
            self.logger.info(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Skipping specialization stream processing: {skip_reason}"
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

        try:
            # Load the mappings
            map_specialization, specialization_stream_mapping = self.load_mappings()

            # Create the relationships
            self.create_specialization_stream_relationships(
                map_specialization, specialization_stream_mapping, create_only_new
            )

            # Print detailed reconciliation report
            self.print_reconciliation_report()

            self.logger.info(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Enhanced Neo4j specialization to stream relationship processing completed"
            )
        except Exception as e:
            self.logger.error(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Error in specialization to stream relationship processing: {str(e)}"
            )
            raise