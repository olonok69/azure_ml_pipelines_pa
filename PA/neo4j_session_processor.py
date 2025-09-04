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
from typing import Dict, Any, Tuple

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
        self.logger = logging.getLogger(__name__)
        
        # Load environment variables
        load_dotenv(config.get("env_file", "keys/.env"))
        
        # Neo4j connection details
        self.uri = os.getenv("NEO4J_URI")
        self.username = os.getenv("NEO4J_USERNAME") 
        self.password = os.getenv("NEO4J_PASSWORD")
        
        # Configuration
        self.output_dir = config.get("output_dir", "data/bva")
        
        # Event configuration for generic processing
        event_config = config.get("event", {})
        self.main_event_name = event_config.get("main_event_name", "bva")
        self.secondary_event_name = event_config.get("secondary_event_name", "lva")
        
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
                for index, row in data.iterrows():
                    try:
                        # Create properties dictionary
                        properties = {}
                        for csv_col, node_prop in properties_map.items():
                            if csv_col in row and pd.notna(row[csv_col]):
                                properties[node_prop] = str(row[csv_col])
                        
                        # Add show attribute
                        properties['show'] = self.show_name

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
            # Process sessions from this year
            self.logger.info("Processing sessions from this year")
            csv_file_path = os.path.join(self.output_dir, "output/session_this_filtered_valid_cols.csv")
            data = pd.read_csv(csv_file_path)
            properties_map = {col: col for col in data.columns}
            node_label = "Sessions_this_year"

            # For Sessions_this_year, always recreate all nodes (delete existing ones first)
            self.logger.info("Recreating all Sessions_this_year nodes")

            # Delete existing Sessions_this_year nodes and their relationships
            driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
            try:
                with driver.session() as session:
                    # Delete all Sessions_this_year nodes and their relationships
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

            # Process sessions from last year (main event)
            self.logger.info(f"Processing sessions from last year {self.main_event_name}")
            csv_file_path = os.path.join(
                self.output_dir, "output/session_last_filtered_valid_cols_bva.csv"
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
                self.output_dir, "output/session_last_filtered_valid_cols_lva.csv"
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
            streams_file_path = os.path.join(self.output_dir, "output/streams.json")
            nodes_created, nodes_skipped = self.create_stream_nodes(streams_file_path, create_only_new)

            self.statistics["nodes_created"]["streams"] = nodes_created
            self.statistics["nodes_skipped"]["streams"] = nodes_skipped

            # Create relationships between sessions this year and streams
            self.logger.info("Creating stream relationships for sessions this year")
            rels_created, rels_skipped = self.create_stream_relationships(
                "Sessions_this_year", "Stream", create_only_new
            )

            self.statistics["relationships_created"]["sessions_this_year_has_stream"] = rels_created
            self.statistics["relationships_skipped"]["sessions_this_year_has_stream"] = rels_skipped

            # Create relationships between sessions past year and streams
            self.logger.info("Creating stream relationships for sessions past year")
            rels_created, rels_skipped = self.create_stream_relationships(
                "Sessions_past_year", "Stream", create_only_new
            )

            self.statistics["relationships_created"]["sessions_past_year_has_stream"] = rels_created
            self.statistics["relationships_skipped"]["sessions_past_year_has_stream"] = rels_skipped

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