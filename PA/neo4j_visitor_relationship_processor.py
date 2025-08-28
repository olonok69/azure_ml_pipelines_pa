import logging
import os
import pandas as pd
from neo4j import GraphDatabase
from dotenv import load_dotenv
import inspect


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

        # Extract Neo4j configuration
        neo4j_config = config.get("neo4j", {})
        self.show_name = neo4j_config.get("show_name", "unknown")
        self.node_labels = neo4j_config.get("node_labels", {})
        self.relationships = neo4j_config.get("relationships", {})
        self.unique_identifiers = neo4j_config.get("unique_identifiers", {})
        self.visitor_relationship_config = neo4j_config.get("visitor_relationship", {})

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

        # Initialize statistics with generic structure
        same_visitor_properties = self.visitor_relationship_config.get("same_visitor_properties", {})
        self.statistics = {
            "relationships_created": {},
            "relationships_skipped": {},
            "relationships_failed": {},
        }
        
        # Initialize statistics for each configured relationship type
        for relationship_type in same_visitor_properties.keys():
            self.statistics["relationships_created"][f"same_visitor_{relationship_type}"] = 0
            self.statistics["relationships_created"][f"attended_session_{relationship_type}"] = 0
            self.statistics["relationships_skipped"][f"same_visitor_{relationship_type}"] = 0
            self.statistics["relationships_skipped"][f"attended_session_{relationship_type}"] = 0
            self.statistics["relationships_failed"][f"same_visitor_{relationship_type}"] = 0
            self.statistics["relationships_failed"][f"attended_session_{relationship_type}"] = 0

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

        # Log summary statistics
        self._log_summary()

        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Neo4j visitor relationship processing completed for show: {self.show_name}"
        )

    def _create_same_visitor_relationships(self, relationship_type, properties, create_only_new):
        """
        Create Same_Visitor relationships for a specific relationship type.

        Args:
            relationship_type (str): Type of relationship (e.g., 'bva', 'lva')
            properties (dict): Properties to attach to the relationship
            create_only_new (bool): If True, only create new relationships
        """
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Creating Same_Visitor relationships for {relationship_type}"
        )

        # Get node labels
        visitor_this_year_label = self.node_labels.get("visitor_this_year", "Visitor_this_year")
        visitor_last_year_label = self.node_labels.get(f"visitor_last_year_{relationship_type}", f"Visitor_last_year_{relationship_type}")
        
        # Get relationship name
        same_visitor_relationship = self.relationships.get("same_visitor", "Same_Visitor")
        
        # Get unique identifier
        visitor_id_field = self.unique_identifiers.get("visitor", "BadgeId")

        relationship_count = 0
        skipped_count = 0
        failed_count = 0

        try:
            driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))

            # Get initial count of existing relationships
            with driver.session() as session:
                initial_count_result = session.run(
                    f"""
                    MATCH (this_year:{visitor_this_year_label})-[r:{same_visitor_relationship}]->(last_year:{visitor_last_year_label})
                    WHERE this_year.show = $show_name AND last_year.show = $show_name
                    AND r.type = $relationship_type
                    RETURN COUNT(r) AS count
                    """,
                    show_name=self.show_name,
                    relationship_type=properties.get('type', relationship_type)
                )
                initial_count = initial_count_result.single()["count"]

                self.logger.info(
                    f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Found {initial_count} existing {relationship_type} Same_Visitor relationships"
                )

                if create_only_new and initial_count > 0:
                    self.logger.info(
                        f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Skipping creation as create_only_new=True."
                    )
                    skipped_count = initial_count
                    self.statistics["relationships_skipped"][f"same_visitor_{relationship_type}"] = skipped_count
                    return

                # Get all visitor nodes with matching last year IDs (including NA for compatibility)
                badge_id_field = f"{visitor_id_field}_last_year_{relationship_type}"
                visitors_result = session.run(
                    f"""
                    MATCH (this_year:{visitor_this_year_label})
                    WHERE this_year.show = $show_name
                    AND this_year.{badge_id_field} IS NOT NULL
                    RETURN this_year.{visitor_id_field} as this_year_id, this_year.{badge_id_field} as last_year_id
                    """,
                    show_name=self.show_name
                )

                for record in visitors_result:
                    this_year_id = record["this_year_id"]
                    last_year_id = record["last_year_id"]

                    # Skip "NA" values without logging warnings (they are placeholders)
                    if last_year_id == "NA":
                        failed_count += 1
                        continue

                    # Check if the relationship already exists
                    relationship_exists = session.run(
                        f"""
                        MATCH (this_year:{visitor_this_year_label} {{{visitor_id_field}: $this_year_id}})-[:{same_visitor_relationship} {{type: $relationship_type}}]->
                              (last_year:{visitor_last_year_label} {{{visitor_id_field}: $last_year_id}})
                        WHERE this_year.show = $show_name AND last_year.show = $show_name
                        RETURN count(*) > 0 AS exists
                        """,
                        this_year_id=this_year_id,
                        last_year_id=last_year_id,
                        show_name=self.show_name,
                        relationship_type=properties.get('type', relationship_type)
                    ).single()["exists"]

                    if not relationship_exists:
                        # Check if both nodes exist before creating relationship
                        nodes_exist = session.run(
                            f"""
                            MATCH (this_year:{visitor_this_year_label} {{{visitor_id_field}: $this_year_id}})
                            MATCH (last_year:{visitor_last_year_label} {{{visitor_id_field}: $last_year_id}})
                            WHERE this_year.show = $show_name AND last_year.show = $show_name
                            RETURN count(*) > 0 AS exists
                            """,
                            this_year_id=this_year_id,
                            last_year_id=last_year_id,
                            show_name=self.show_name
                        ).single()["exists"]

                        if nodes_exist:
                            try:
                                # Create relationship with properties
                                property_params = ", ".join([f"{k}: ${k}" for k in properties.keys()])
                                session.run(
                                    f"""
                                    MATCH (this_year:{visitor_this_year_label} {{{visitor_id_field}: $this_year_id}})
                                    MATCH (last_year:{visitor_last_year_label} {{{visitor_id_field}: $last_year_id}})
                                    WHERE this_year.show = $show_name AND last_year.show = $show_name
                                    CREATE (this_year)-[:{same_visitor_relationship} {{{property_params}}}]->(last_year)
                                    """,
                                    this_year_id=this_year_id,
                                    last_year_id=last_year_id,
                                    show_name=self.show_name,
                                    **properties
                                )
                                relationship_count += 1
                            except Exception as e:
                                failed_count += 1
                                self.logger.error(
                                    f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Error creating Same_Visitor relationship for {this_year_id} -> {last_year_id}: {str(e)}"
                                )
                        else:
                            failed_count += 1
                            # Note: Not logging warning for missing nodes as they may be legitimately absent
                    else:
                        skipped_count += 1

        except Exception as e:
            self.logger.error(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Error creating Same_Visitor relationships for {relationship_type}: {str(e)}"
            )
            raise
        finally:
            driver.close()

        # Update statistics
        self.statistics["relationships_created"][f"same_visitor_{relationship_type}"] = relationship_count
        self.statistics["relationships_skipped"][f"same_visitor_{relationship_type}"] = skipped_count
        self.statistics["relationships_failed"][f"same_visitor_{relationship_type}"] = failed_count

        # Log summary statistics
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
                initial_count_result = session.run(
                    f"""
                    MATCH (visitor:{visitor_last_year_label})-[r:{attended_session_relationship}]->(session:{session_past_year_label})
                    WHERE visitor.show = $show_name AND session.show = $show_name
                    RETURN COUNT(r) AS count
                    """,
                    show_name=self.show_name
                )
                initial_count = initial_count_result.single()["count"]

                self.logger.info(
                    f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Found {initial_count} existing {relationship_type} attended_session relationships"
                )

                if create_only_new and initial_count > 0:
                    self.logger.info(
                        f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Skipping creation as create_only_new=True."
                    )
                    skipped_count = initial_count
                    self.statistics["relationships_skipped"][f"attended_session_{relationship_type}"] = skipped_count
                    return

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

                # Load and process scan data to match old processor behavior
                if relationship_type == "bva":
                    scan_filename = "scan_bva_past.csv"
                    filter_filename = "df_reg_demo_last_bva.csv"
                elif relationship_type == "lva":
                    scan_filename = "scan_lva_past.csv"
                    filter_filename = "df_reg_demo_last_lva.csv"
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
                    relationship_exists = session.run(
                        f"""
                        MATCH (v:{visitor_last_year_label} {{{visitor_id_field}: $visitor_badge_id}})-[:{attended_session_relationship}]->
                              (s:{session_past_year_label} {{key_text: $session_key_text}})
                        WHERE v.show = $show_name AND s.show = $show_name
                        RETURN count(*) > 0 AS exists
                        """,
                        visitor_badge_id=visitor_badge_id,
                        session_key_text=session_key_text,
                        show_name=self.show_name
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