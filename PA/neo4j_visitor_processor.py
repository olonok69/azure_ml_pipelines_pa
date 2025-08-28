import logging
import os
import pandas as pd
from neo4j import GraphDatabase
from dotenv import load_dotenv
import csv
import inspect


class Neo4jVisitorProcessor:
    """
    A class to process visitor data and upload it to Neo4j database.
    This class is responsible for loading CSV files and creating visitor nodes in Neo4j.
    
    This is a generic version that works with any event type configuration.
    """

    def __init__(self, config):
        """
        Initialize the Neo4j Visitor Processor.

        Args:
            config (dict): Configuration dictionary containing paths and settings
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Initializing Neo4j Visitor Processor"
        )

        self.config = config
        self.output_dir = os.path.join(config["output_dir"], "output")

        # Get event configuration
        self.event_config = config.get("event", {})
        
        # Get event names
        self.main_event_name = self.event_config.get("main_event_name", "bva")
        self.secondary_event_name = self.event_config.get("secondary_event_name", "lva")
        # Get Neo4j configuration
        neo4j_config = config.get("neo4j", {})
        self.show_name = neo4j_config.get("show_name", "bva")
        
        # Keep event config for other purposes but use show_name for node properties
        self.event_config = config.get("event", {})
        self.main_event_name = self.event_config.get("main_event_name", self.show_name)
        self.secondary_event_name = self.event_config.get("secondary_event_name", "lva")
        
        # Log the configuration being used
        self.logger.info(f"Using show_name: {self.show_name} for node properties")

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

        # Initialize statistics dictionary with generic event names
        self.statistics = {
            "nodes_created": {
                "visitor_this_year": 0,
                f"visitor_last_year_{self.main_event_name}": 0,
                f"visitor_last_year_{self.secondary_event_name}": 0,
            },
            "nodes_skipped": {
                "visitor_this_year": 0,
                f"visitor_last_year_{self.main_event_name}": 0,
                f"visitor_last_year_{self.secondary_event_name}": 0,
            },
        }

        # Backward compatibility - add traditional BVA/LVA keys
        self.statistics["nodes_created"]["visitor_last_year_bva"] = 0
        self.statistics["nodes_created"]["visitor_last_year_lva"] = 0
        self.statistics["nodes_skipped"]["visitor_last_year_bva"] = 0
        self.statistics["nodes_skipped"]["visitor_last_year_lva"] = 0

        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Neo4j Visitor Processor initialized successfully"
        )
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Adding 'show' attribute to all nodes with value: {self.main_event_name}"
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
                    f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Successfully connected to Neo4j. Database contains {count} nodes"
                )
                driver.close()
                return True
        except Exception as e:
            self.logger.error(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Failed to connect to Neo4j: {str(e)}"
            )
            return False

    def load_csv_to_neo4j(self, csv_file_path, node_label, properties_map, create_only_new=False):
        """
        Load a CSV file into Neo4j, creating nodes with the specified label and properties.

        Args:
            csv_file_path (str): Path to the CSV file
            node_label (str): Label to assign to the nodes
            properties_map (dict): Mapping of CSV column names to Neo4j property names
            create_only_new (bool): If True, only create nodes that don't exist in the database

        Returns:
            tuple: (nodes_created, nodes_skipped)
        """
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Loading data from {csv_file_path} to Neo4j with label {node_label}"
        )

        # Get the unique identifier field
        unique_id_field = self.config.get("neo4j", {}).get("unique_identifiers", {}).get("visitor", "BadgeId")
        
        nodes_created = 0
        nodes_skipped = 0

        try:
            # Connect to Neo4j
            driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))

            # Read CSV file
            with open(csv_file_path, "r", encoding="utf-8") as csv_file:
                reader = csv.DictReader(csv_file)
                headers = reader.fieldnames

                # Process each row in the CSV
                for row in reader:
                    # Create a dictionary of properties for this node
                    properties = {}
                    for csv_col, neo4j_prop in properties_map.items():
                        if csv_col in row:
                            properties[neo4j_prop] = row[csv_col]
                    
                    # Add a 'show' attribute to group all nodes belonging to this show
                    properties["show"] = self.show_name

                    # Check if the unique ID is present
                    if unique_id_field not in properties:
                        self.logger.warning(
                            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Unique ID field '{unique_id_field}' not found in row"
                        )
                        continue

                    unique_id = properties[unique_id_field]

                    # Create or merge the node
                    with driver.session() as session:
                        # Check if the node already exists
                        if create_only_new:
                            result = session.run(
                                f"MATCH (n:{node_label} {{{unique_id_field}: $unique_id}}) RETURN n",
                                unique_id=unique_id,
                            )
                            if result.single():
                                self.logger.debug(
                                    f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Node {node_label} with {unique_id_field}={unique_id} already exists, skipping"
                                )
                                nodes_skipped += 1
                                continue

                        # Create the node
                        query = f"CREATE (n:{node_label} $props) RETURN n"
                        session.run(query, props=properties)
                        nodes_created += 1

            driver.close()
            self.logger.info(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Created {nodes_created} nodes and skipped {nodes_skipped} nodes"
            )
            return nodes_created, nodes_skipped

        except Exception as e:
            self.logger.error(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Error loading CSV to Neo4j: {str(e)}"
            )
            return 0, 0

    def process(self, create_only_new=False):
        """
        Process visitor data and upload to Neo4j database.

        Args:
            create_only_new (bool): If True, only create nodes that don't exist in the database.
        """
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Starting Neo4j visitor data processing"
        )

        # Test the connection first
        if not self._test_connection():
            self.logger.error(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Cannot proceed with Neo4j upload due to connection failure"
            )
            return

        # Get node labels from config
        neo4j_config = self.config.get("neo4j", {})
        node_labels = neo4j_config.get("node_labels", {})
        
        # Process visitors from this year
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Processing visitors from this year"
        )
        try:
            # Get output file from config
            combined_output_files = self.config.get("output_files", {}).get("combined_demographic_registration", {})
            csv_file_path = os.path.join(self.output_dir, combined_output_files.get("this_year", "df_reg_demo_this.csv"))
            
            data = pd.read_csv(csv_file_path)
            properties_map = {col: col for col in data.columns}
            node_label = node_labels.get("visitor_this_year", "Visitor_this_year")

            nodes_created, nodes_skipped = self.load_csv_to_neo4j(
                csv_file_path, node_label, properties_map, create_only_new
            )

            self.statistics["nodes_created"]["visitor_this_year"] = nodes_created
            self.statistics["nodes_skipped"]["visitor_this_year"] = nodes_skipped
        except Exception as e:
            self.logger.error(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Error processing visitors from this year: {str(e)}"
            )

        # Process visitors from last year main event
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Processing visitors from last year {self.main_event_name}"
        )
        try:
            # Get output file from config
            combined_output_files = self.config.get("output_files", {}).get("combined_demographic_registration", {})
            csv_file_path = os.path.join(self.output_dir, combined_output_files.get("last_year_main", "df_reg_demo_last_bva.csv"))
            
            data = pd.read_csv(csv_file_path)
            properties_map = {col: col for col in data.columns}
            node_label = node_labels.get("visitor_last_year_bva", "Visitor_last_year_bva")

            nodes_created, nodes_skipped = self.load_csv_to_neo4j(
                csv_file_path, node_label, properties_map, create_only_new
            )

            # Update statistics for both generic and backward compatibility keys
            self.statistics["nodes_created"][f"visitor_last_year_{self.main_event_name}"] = nodes_created
            self.statistics["nodes_skipped"][f"visitor_last_year_{self.main_event_name}"] = nodes_skipped
            
            # Also update the bva key for backward compatibility
            self.statistics["nodes_created"]["visitor_last_year_bva"] = nodes_created
            self.statistics["nodes_skipped"]["visitor_last_year_bva"] = nodes_skipped
        except Exception as e:
            self.logger.error(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Error processing visitors from last year {self.main_event_name}: {str(e)}"
            )

        # Process visitors from last year secondary event
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Processing visitors from last year {self.secondary_event_name}"
        )
        try:
            # Get output file from config
            combined_output_files = self.config.get("output_files", {}).get("combined_demographic_registration", {})
            csv_file_path = os.path.join(self.output_dir, combined_output_files.get("last_year_secondary", "df_reg_demo_last_lva.csv"))
            
            data = pd.read_csv(csv_file_path)
            properties_map = {col: col for col in data.columns}
            node_label = node_labels.get("visitor_last_year_lva", "Visitor_last_year_lva")

            nodes_created, nodes_skipped = self.load_csv_to_neo4j(
                csv_file_path, node_label, properties_map, create_only_new
            )

            # Update statistics for both generic and backward compatibility keys
            self.statistics["nodes_created"][f"visitor_last_year_{self.secondary_event_name}"] = nodes_created
            self.statistics["nodes_skipped"][f"visitor_last_year_{self.secondary_event_name}"] = nodes_skipped
            
            # Also update the lva key for backward compatibility
            self.statistics["nodes_created"]["visitor_last_year_lva"] = nodes_created
            self.statistics["nodes_skipped"]["visitor_last_year_lva"] = nodes_skipped
        except Exception as e:
            self.logger.error(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Error processing visitors from last year {self.secondary_event_name}: {str(e)}"
            )

        # Log summary statistics
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Neo4j visitor upload summary:"
        )
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Visitors this year: {self.statistics['nodes_created']['visitor_this_year']} created, {self.statistics['nodes_skipped']['visitor_this_year']} skipped"
        )
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Visitors last year {self.main_event_name}: {self.statistics['nodes_created'][f'visitor_last_year_{self.main_event_name}']} created, {self.statistics['nodes_skipped'][f'visitor_last_year_{self.main_event_name}']} skipped"
        )
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Visitors last year {self.secondary_event_name}: {self.statistics['nodes_created'][f'visitor_last_year_{self.secondary_event_name}']} created, {self.statistics['nodes_skipped'][f'visitor_last_year_{self.secondary_event_name}']} skipped"
        )

        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Neo4j visitor data processing completed"
        )