import logging
import os
import json
import pandas as pd
from neo4j import GraphDatabase
from dotenv import load_dotenv
import inspect

from utils.neo4j_utils import resolve_neo4j_credentials


class Neo4jSpecializationStreamProcessor:
    """
    A class to process specialization to stream relationships in Neo4j.
    ENHANCED: Supports different mappings for different visitor types.
    """

    def __init__(self, config):
        """
        Initialize the Neo4j Specialization Stream Processor.

        Args:
            config (dict): Configuration dictionary containing paths and settings
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Initializing Neo4j Specialization Stream Processor"
        )
        self.config = config
        self.output_dir = config["output_dir"]
        self.input_dir = os.path.join(self.output_dir, "output")

        # Load environment variables
        load_dotenv(config.get("env_file", "keys/.env"))

        # Neo4j connection details
        credentials = resolve_neo4j_credentials(config, logger=self.logger)
        self.uri = credentials["uri"]
        self.username = credentials["username"]
        self.password = credentials["password"]
        self.neo4j_environment = credentials["environment"]
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Using Neo4j environment: {self.neo4j_environment}"
        )

        # Get configuration for specialization stream mapping
        neo4j_config = config.get("neo4j", {})
        self.show_name = neo4j_config.get("show_name", None)
        self.node_labels = neo4j_config.get("node_labels", {})
        self.relationships = neo4j_config.get("relationships", {})
        self.specialization_stream_mapping_config = neo4j_config.get(
            "specialization_stream_mapping", {}
        )

        # Check if specialization stream processing is enabled for this show
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
        
        # ENHANCED: Get visitor-type-specific mappings
        self.specialization_map_this_year = self.specialization_stream_mapping_config.get(
            "specialization_map_this_year", {}
        )
        self.specialization_map_last_year_bva = self.specialization_stream_mapping_config.get(
            "specialization_map_last_year_bva", {}
        )
        self.specialization_map_last_year_lva = self.specialization_stream_mapping_config.get(
            "specialization_map_last_year_lva", {}
        )
        
        # Fallback to old config for backward compatibility
        if not self.specialization_map_this_year and not self.specialization_map_last_year_bva:
            old_map = self.specialization_stream_mapping_config.get("specialization_map", {})
            if old_map:
                self.logger.warning(
                    "Using deprecated 'specialization_map' config. Please update to use visitor-type-specific mappings."
                )
                self.specialization_map_this_year = old_map
                self.specialization_map_last_year_bva = old_map
                self.specialization_map_last_year_lva = old_map

        # Progress logging interval for created relationships
        self.progress_log_interval = self.specialization_stream_mapping_config.get(
            "progress_log_interval", 10000
        )

        # Initialize statistics
        self.statistics = {
            "initial_count": 0,
            "final_count": 0,
            # Track by actual node labels (lowercased) to match aggregation keys
            "visitor_nodes_processed": {
                self.visitor_this_year_label.lower(): 0,
                self.visitor_last_year_bva_label.lower(): 0,
                self.visitor_last_year_lva_label.lower(): 0,
            },
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
        
        Returns:
            tuple: (should_skip: bool, reason: str)
        """
        # Check if specialization stream processing is explicitly disabled
        if not self.specialization_stream_enabled:
            return True, f"Specialization stream processing disabled in config for show '{self.show_name}'"
        
        # Check if mapping file exists - try multiple locations
        possible_paths = [
            os.path.join(self.input_dir, self.specialization_mapping_file),  # output_dir/output/file
            os.path.join(self.output_dir, self.specialization_mapping_file),  # align with job mapping search path
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
        
        Returns:
            dict: Mapping from specializations to streams
        """
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Loading specialization mappings"
        )

        try:
            # Load specialization to stream mapping from configured file - try multiple locations
            possible_paths = [
                os.path.join(self.input_dir, self.specialization_mapping_file),
                os.path.join(self.output_dir, self.specialization_mapping_file),
                self.specialization_mapping_file,
                os.path.join("data", "bva", self.specialization_mapping_file),
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

            return specialization_stream_mapping
        except Exception as e:
            self.logger.error(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Error loading specialization mappings: {str(e)}"
            )
            raise

    def _count_relationships(self, session):
        """
        Count the total number of specialization_to_stream relationships.
        """
        query = f"""
        MATCH (:{self.visitor_this_year_label})-[r:{self.relationship_name}]->(:{self.stream_label})
        RETURN COUNT(r) AS count
        """
        
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

    def _normalize_stream_name(self, stream_name):
        """
        Normalize stream names to match canonical Stream nodes in Neo4j.
        Currently maps known synonyms that aren't present as-is in the DB.

        Args:
            stream_name (str): Stream name from mapping file

        Returns:
            tuple[str, bool]: (normalized_name, was_mapped)
        """
        if not stream_name:
            return stream_name, False

        original = stream_name
        name_l = stream_name.strip().lower()

        # Known synonyms -> canonical names (based on streams.json)
        synonyms = {
            "emergency and critical care (ecc)": "Emergency Medicine",
            "ecc": "Emergency Medicine",
        }

        if name_l in synonyms:
            return synonyms[name_l], True

        # Default: return original (trimmed) without change
        return original.strip(), False

    def process_visitor_specializations(
        self, tx, node_label, specialization_field, map_specialization, specialization_stream_mapping, create_only_new
    ):
        """
        Process specializations for a specific visitor node type.
        ENHANCED: Uses visitor-type-specific mappings.
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

        # Build query with show filtering if configured
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

            # Skip if BadgeId or Email is not available
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

                # Map specialization if needed (using visitor-type-specific mapping)
                if spec in map_specialization:
                    mapped_spec = map_specialization[spec]
                    stats["specializations_mapped"] += 1
                    self.logger.debug(
                        f"Mapping '{spec}' to '{mapped_spec}' for {node_label}"
                    )
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

                            # Normalize known synonyms to canonical stream names
                            normalized_stream_name, mapped = self._normalize_stream_name(stream_name)
                            # if mapped:
                            #     self.logger.info(
                            #         f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Stream name normalization: '{stream_name}' → '{normalized_stream_name}'"
                            #     )

                            stream_query = f"""
                            MATCH (s:{self.stream_label})
                            WHERE LOWER(s.stream) = LOWER($stream_name){stream_search_filter}
                            RETURN s.stream as actual_stream_name
                            LIMIT 1
                            """

                            stream_params = {"stream_name": normalized_stream_name}
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
                                
                                CREATE (v)-[r:{self.relationship_name}]->(s)
                                SET r.created_by = $created_by
                                RETURN r
                                """
                            else:
                                # Always create (might create duplicates)
                                relationship_query = f"""
                                MATCH (v:{node_label})
                                WHERE v.BadgeId = $badge_id AND v.Email = $email{visitor_show_filter}
                                
                                MATCH (s:{self.stream_label})
                                WHERE LOWER(s.stream) = LOWER($actual_stream_name){stream_show_filter}
                                
                                CREATE (v)-[r:{self.relationship_name}]->(s)
                                SET r.created_by = $created_by
                                RETURN r
                                """

                            rel_params = {
                                "badge_id": badge_id,
                                "email": email,
                                "actual_stream_name": actual_stream_name,
                                "created_by": "specialization_processor"
                            }
                            if self.show_name:
                                rel_params["show_name"] = self.show_name

                            result = tx.run(relationship_query, **rel_params)

                            if result.single():
                                stats["relationships_created"] += 1
                                # Periodic progress logging every N created relationships
                                if (
                                    self.progress_log_interval
                                    and stats["relationships_created"] % self.progress_log_interval == 0
                                ):
                                    self.logger.info(
                                        f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Created {stats['relationships_created']:,} relationships so far for {node_label}"
                                    )
                            else:
                                stats["relationships_skipped"] += 1
                else:
                    # Log warning if mapped specialization not found
                    if mapped_spec != spec:
                        self.logger.warning(
                            f"Mapped specialization '{mapped_spec}' (from '{spec}') not found in specialization_stream_mapping"
                        )

        return stats

    def create_specialization_stream_relationships(
        self, specialization_stream_mapping, create_only_new=True
    ):
        """
        Create relationships between visitors and streams based on specializations.
        ENHANCED: Uses visitor-type-specific mappings.
        """
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Creating specialization to stream relationships with visitor-type-specific mappings"
        )

        driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))

        # First count the initial relationships
        with driver.session() as session:
            initial_count = self._count_relationships(session)
            self.statistics["initial_count"] = initial_count
            self.logger.info(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Initial relationship count: {initial_count}"
            )

        # Process different visitor types with their specific mappings
        visitor_types = [
            {
                "node_label": self.visitor_this_year_label,
                "specialization_field": self.specialization_field_this_year,
                "map_specialization": self.specialization_map_this_year,
            },
            {
                "node_label": self.visitor_last_year_bva_label,
                "specialization_field": self.specialization_field_bva,
                "map_specialization": self.specialization_map_last_year_bva,
            },
            {
                "node_label": self.visitor_last_year_lva_label,
                "specialization_field": self.specialization_field_lva,
                "map_specialization": self.specialization_map_last_year_lva,
            },
        ]

        try:
            with driver.session() as session:
                # Process each visitor type with its specific mapping
                for visitor_type in visitor_types:
                    node_label = visitor_type["node_label"]
                    specialization_field = visitor_type["specialization_field"]
                    map_specialization = visitor_type["map_specialization"]

                    self.logger.info(
                        f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Processing {node_label} with field '{specialization_field}' and {len(map_specialization)} mappings"
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

                    # Update overall statistics
                    for key in visitor_stats["visitor_nodes_processed"]:
                        # Ensure key exists (handles custom labels)
                        if key not in self.statistics["visitor_nodes_processed"]:
                            self.statistics["visitor_nodes_processed"][key] = 0
                        self.statistics["visitor_nodes_processed"][key] += visitor_stats["visitor_nodes_processed"].get(key, 0)

                    self.statistics["specializations_processed"] += visitor_stats[
                        "specializations_processed"
                    ]
                    self.statistics["specializations_mapped"] += visitor_stats[
                        "specializations_mapped"
                    ]
                    self.statistics["stream_matches_found"] += visitor_stats[
                        "stream_matches_found"
                    ]
                    self.statistics["relationships_created"] += visitor_stats[
                        "relationships_created"
                    ]
                    self.statistics["relationships_skipped"] += visitor_stats[
                        "relationships_skipped"
                    ]
                    self.statistics["relationships_not_found"] += visitor_stats[
                        "relationships_not_found"
                    ]

                    self.logger.info(
                        f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Processed {node_label}: "
                        f"Created {visitor_stats['relationships_created']}, "
                        f"Skipped {visitor_stats['relationships_skipped']}, "
                        f"Not found {visitor_stats['relationships_not_found']}"
                    )

                # Count the final relationships
                final_count = self._count_relationships(session)
                self.statistics["final_count"] = final_count
                self.statistics["total_relationships_created"] = (
                    final_count - initial_count
                )
                self.statistics["total_relationships_skipped"] = self.statistics[
                    "relationships_skipped"
                ]

                self.logger.info(
                    f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Final relationship count: {final_count}"
                )
                self.logger.info(
                    f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Total new relationships created: {self.statistics['total_relationships_created']}"
                )

        except Exception as e:
            self.logger.error(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Error creating specialization stream relationships: {str(e)}"
            )
            raise
        finally:
            driver.close()

        return self.statistics

    def print_reconciliation_report(self):
        """Print a reconciliation report of the specialization stream processing."""
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
        """
        self.logger.info("Starting specialization stream processing workflow")

        # Check if processing should be skipped
        should_skip, skip_reason = self.should_skip_processing()
        if should_skip:
            self.statistics["processing_skipped"] = True
            self.statistics["skip_reason"] = skip_reason
            self.logger.warning(f"Skipping processing: {skip_reason}")
            self.print_reconciliation_report()
            return self.statistics

        # Test Neo4j connection
        if not self._test_connection():
            self.logger.error("Failed to connect to Neo4j. Aborting.")
            return None

        try:
            # Load mappings
            specialization_stream_mapping = self.load_mappings()
            
            # Create relationships
            self.create_specialization_stream_relationships(
                specialization_stream_mapping, create_only_new
            )

            # Print reconciliation report
            self.print_reconciliation_report()

            return self.statistics

        except Exception as e:
            self.logger.error(
                f"Specialization stream processing failed: {e}", exc_info=True
            )
            return None


def main():
    """Main function for testing the specialization stream processor."""
    import sys
    sys.path.insert(0, os.getcwd())
    
    from utils.config_utils import load_config
    from utils.logging_utils import setup_logging

    # Setup logging
    logger = setup_logging(log_file="logs/neo4j_specialization_stream_processor.log")
    
    try:
        # Load configuration
        config = load_config("config/config_vet_lva.yaml")
        
        # Create processor and run
        processor = Neo4jSpecializationStreamProcessor(config)
        processor.process()
        
        print("Specialization stream processing completed successfully!")
        
    except Exception as e:
        logger.error(f"Specialization stream processing failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()