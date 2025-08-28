import logging
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv


class Neo4jConnection:
    """
    Neo4j database connection class.
    """

    def __init__(self, config):
        """
        Initialize Neo4j connection using configuration or environment variables.

        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)

        # Load environment variables if env_file is specified in config
        if "env_file" in config:
            load_dotenv(config["env_file"])

        # Get Neo4j connection details from environment or config
        self.uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = os.getenv("NEO4J_USER", "neo4j")
        self.password = os.getenv("NEO4J_PASSWORD", "password")

        # Create Neo4j driver
        try:
            self.driver = GraphDatabase.driver(
                self.uri, auth=(self.user, self.password)
            )
            self.logger.info(f"Successfully connected to Neo4j database at {self.uri}")
        except Exception as e:
            self.logger.error(f"Error connecting to Neo4j database: {e}", exc_info=True)
            raise

    def close(self):
        """
        Close the Neo4j driver connection.
        """
        if hasattr(self, "driver"):
            self.driver.close()
            self.logger.info("Neo4j driver connection closed")

    def execute_query(self, query, params=None, write=False):
        """
        Execute a Cypher query on the Neo4j database.

        Args:
            query: Cypher query string
            params: Parameters for the query (optional)
            write: Whether this is a write query (default: False)

        Returns:
            Results of the query execution
        """
        if params is None:
            params = {}

        try:
            with self.driver.session() as session:
                if write:
                    result = session.write_transaction(
                        self._execute_transaction, query, params
                    )
                else:
                    result = session.read_transaction(
                        self._execute_transaction, query, params
                    )
                return result
        except Exception as e:
            self.logger.error(f"Error executing Neo4j query: {e}", exc_info=True)
            self.logger.error(f"Query: {query}")
            self.logger.error(f"Params: {params}")
            raise

    @staticmethod
    def _execute_transaction(tx, query, params):
        """
        Execute a transaction in Neo4j.

        Args:
            tx: Neo4j transaction object
            query: Cypher query string
            params: Parameters for the query

        Returns:
            List of records from the query result
        """
        result = tx.run(query, params)
        return [record for record in result]


def create_node_unique(neo4j_conn, label, properties, unique_key):
    """
    Create a node in Neo4j if it doesn't already exist based on a unique key.

    Args:
        neo4j_conn: Neo4j connection object
        label: Node label
        properties: Node properties dictionary
        unique_key: Property key to use for uniqueness constraint

    Returns:
        Dictionary with created (True/False) and node information
    """
    logger = logging.getLogger(__name__)

    # Create MERGE query
    query = f"""
    MERGE (n:{label} {{{unique_key}: $properties['{unique_key}']}})
    ON CREATE SET n = $properties
    RETURN n, exists((n)) as existed_before
    """

    try:
        result = neo4j_conn.execute_query(query, {"properties": properties}, write=True)

        if result and len(result) > 0:
            node_existed = result[0]["existed_before"]
            return {"created": not node_existed, "node": result[0]["n"]}
        else:
            logger.warning(
                f"No result returned when creating {label} node with {unique_key}={properties.get(unique_key)}"
            )
            return {"created": False, "node": None}
    except Exception as e:
        logger.error(f"Error creating Neo4j node: {e}", exc_info=True)
        return {"created": False, "node": None, "error": str(e)}


def create_relationship_unique(
    neo4j_conn,
    source_label,
    source_key,
    source_value,
    target_label,
    target_key,
    target_value,
    relationship_type,
    relationship_props=None,
):
    """
    Create a relationship in Neo4j if it doesn't already exist.

    Args:
        neo4j_conn: Neo4j connection object
        source_label: Label of the source node
        source_key: Property key for identifying the source node
        source_value: Property value for identifying the source node
        target_label: Label of the target node
        target_key: Property key for identifying the target node
        target_value: Property value for identifying the target node
        relationship_type: Type of relationship to create
        relationship_props: Properties to set on the relationship (optional)

    Returns:
        Dictionary with created (True/False) and relationship information
    """
    logger = logging.getLogger(__name__)

    if relationship_props is None:
        relationship_props = {}

    # Create MERGE query for relationship
    query = f"""
    MATCH (source:{source_label} {{{source_key}: $source_value}})
    MATCH (target:{target_label} {{{target_key}: $target_value}})
    MERGE (source)-[r:{relationship_type}]->(target)
    ON CREATE SET r = $relationship_props
    RETURN source, target, r, exists((source)-[:{relationship_type}]->(target)) as existed_before
    """

    params = {
        "source_value": source_value,
        "target_value": target_value,
        "relationship_props": relationship_props,
    }

    try:
        result = neo4j_conn.execute_query(query, params, write=True)

        if result and len(result) > 0:
            relationship_existed = result[0]["existed_before"]
            return {
                "created": not relationship_existed,
                "source": result[0]["source"],
                "target": result[0]["target"],
                "relationship": result[0]["r"],
            }
        else:
            logger.warning(
                f"No result returned when creating {relationship_type} relationship between {source_label}({source_key}={source_value}) and {target_label}({target_key}={target_value})"
            )
            return {
                "created": False,
                "source": None,
                "target": None,
                "relationship": None,
            }
    except Exception as e:
        logger.error(f"Error creating Neo4j relationship: {e}", exc_info=True)
        return {
            "created": False,
            "source": None,
            "target": None,
            "relationship": None,
            "error": str(e),
        }


def check_node_exists(neo4j_conn, label, property_key, property_value):
    """
    Check if a node exists in Neo4j.

    Args:
        neo4j_conn: Neo4j connection object
        label: Node label
        property_key: Property key for identifying the node
        property_value: Property value for identifying the node

    Returns:
        True if node exists, False otherwise
    """
    query = f"""
    MATCH (n:{label} {{{property_key}: $property_value}})
    RETURN n
    """

    result = neo4j_conn.execute_query(query, {"property_value": property_value})
    return len(result) > 0


def get_node_by_property(neo4j_conn, label, property_key, property_value):
    """
    Get a node from Neo4j by property.

    Args:
        neo4j_conn: Neo4j connection object
        label: Node label
        property_key: Property key for identifying the node
        property_value: Property value for identifying the node

    Returns:
        Node if found, None otherwise
    """
    query = f"""
    MATCH (n:{label} {{{property_key}: $property_value}})
    RETURN n
    """

    result = neo4j_conn.execute_query(query, {"property_value": property_value})
    return result[0]["n"] if result and len(result) > 0 else None


def create_constraints(neo4j_conn, constraints):
    """
    Create constraints in Neo4j.

    Args:
        neo4j_conn: Neo4j connection object
        constraints: List of constraint dictionaries with 'label' and 'property' keys
    """
    logger = logging.getLogger(__name__)

    for constraint in constraints:
        label = constraint["label"]
        property_key = constraint["property"]

        # First check if constraint already exists
        check_query = """
        CALL db.constraints() 
        YIELD name, description
        WHERE description CONTAINS $label AND description CONTAINS $property
        RETURN name
        """

        result = neo4j_conn.execute_query(
            check_query, {"label": label, "property": property_key}
        )

        if result and len(result) > 0:
            logger.info(f"Constraint for {label}.{property_key} already exists")
            continue

        # Create constraint
        constraint_query = f"""
        CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.{property_key} IS UNIQUE
        """

        try:
            neo4j_conn.execute_query(constraint_query, {}, write=True)
            logger.info(f"Created constraint for {label}.{property_key}")
        except Exception as e:
            logger.error(
                f"Error creating constraint for {label}.{property_key}: {e}",
                exc_info=True,
            )
