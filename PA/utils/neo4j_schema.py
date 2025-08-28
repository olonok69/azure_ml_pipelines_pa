"""
Neo4j schema generation and validation utilities.
This module helps create the necessary constraints and indexes in Neo4j
and provides schema documentation.
"""

import logging
from .neo4j_utils import Neo4jConnection, create_constraints


def setup_neo4j_schema(config):
    """
    Set up Neo4j schema with constraints and indexes.

    Args:
        config: Configuration dictionary
    """
    logger = logging.getLogger(__name__)
    logger.info("Setting up Neo4j schema")

    # Create Neo4j connection
    neo4j_conn = Neo4jConnection(config)

    try:
        # Define constraints based on configuration
        constraints = [
            {
                "label": config["neo4j"]["node_labels"]["visitor_this_year"],
                "property": config["neo4j"]["unique_identifiers"]["visitor"],
            },
            {
                "label": config["neo4j"]["node_labels"]["visitor_last_year_bva"],
                "property": config["neo4j"]["unique_identifiers"]["visitor"],
            },
            {
                "label": config["neo4j"]["node_labels"]["visitor_last_year_lva"],
                "property": config["neo4j"]["unique_identifiers"]["visitor"],
            },
            {
                "label": config["neo4j"]["node_labels"]["session_this_year"],
                "property": config["neo4j"]["unique_identifiers"]["session"],
            },
            {
                "label": config["neo4j"]["node_labels"]["session_past_year"],
                "property": config["neo4j"]["unique_identifiers"]["session"],
            },
            {
                "label": config["neo4j"]["node_labels"]["stream"],
                "property": config["neo4j"]["unique_identifiers"]["stream"],
            },
        ]

        # Create constraints
        create_constraints(neo4j_conn, constraints)

        # Create indexes for common lookup fields
        create_indexes(neo4j_conn, config)

        logger.info("Neo4j schema setup complete")
    except Exception as e:
        logger.error(f"Error setting up Neo4j schema: {e}", exc_info=True)
        raise
    finally:
        neo4j_conn.close()


def create_indexes(neo4j_conn, config):
    """
    Create indexes for common lookup fields in Neo4j.

    Args:
        neo4j_conn: Neo4j connection object
        config: Configuration dictionary
    """
    logger = logging.getLogger(__name__)

    # Define indexes based on common query patterns
    indexes = [
        {
            "label": config["neo4j"]["node_labels"]["visitor_this_year"],
            "property": "job_role",
        },
        {
            "label": config["neo4j"]["node_labels"]["visitor_this_year"],
            "property": "what_type_does_your_practice_specialise_in",
        },
        {
            "label": config["neo4j"]["node_labels"]["visitor_last_year_bva"],
            "property": "job_role",
        },
        {
            "label": config["neo4j"]["node_labels"]["visitor_last_year_bva"],
            "property": "what_areas_do_you_specialise_in",
        },
        {
            "label": config["neo4j"]["node_labels"]["visitor_last_year_lva"],
            "property": "job_role",
        },
        {
            "label": config["neo4j"]["node_labels"]["visitor_last_year_lva"],
            "property": "what_areas_do_you_specialise_in",
        },
        {
            "label": config["neo4j"]["node_labels"]["session_this_year"],
            "property": "stream",
        },
        {
            "label": config["neo4j"]["node_labels"]["session_past_year"],
            "property": "stream",
        },
    ]

    for index in indexes:
        label = index["label"]
        property_key = index["property"]

        # Check if index already exists
        check_query = """
        CALL db.indexes() 
        YIELD name, labelsOrTypes, properties
        WHERE labelsOrTypes[0] = $label AND properties[0] = $property
        RETURN name
        """

        result = neo4j_conn.execute_query(
            check_query, {"label": label, "property": property_key}
        )

        if result and len(result) > 0:
            logger.info(f"Index for {label}.{property_key} already exists")
            continue

        # Create index
        index_query = f"""
        CREATE INDEX IF NOT EXISTS FOR (n:{label}) ON (n.{property_key})
        """

        try:
            neo4j_conn.execute_query(index_query, {}, write=True)
            logger.info(f"Created index for {label}.{property_key}")
        except Exception as e:
            logger.error(
                f"Error creating index for {label}.{property_key}: {e}", exc_info=True
            )


def get_schema_documentation(config):
    """
    Generate Neo4j schema documentation based on configuration.

    Args:
        config: Configuration dictionary

    Returns:
        String containing schema documentation in Markdown format
    """
    # Get node labels from config
    visitor_this_year = config["neo4j"]["node_labels"]["visitor_this_year"]
    visitor_last_year_bva = config["neo4j"]["node_labels"]["visitor_last_year_bva"]
    visitor_last_year_lva = config["neo4j"]["node_labels"]["visitor_last_year_lva"]
    session_this_year = config["neo4j"]["node_labels"]["session_this_year"]
    session_past_year = config["neo4j"]["node_labels"]["session_past_year"]
    stream = config["neo4j"]["node_labels"]["stream"]

    # Get relationship types from config
    session_stream_rel = config["neo4j"]["relationships"]["session_stream"]
    job_stream_rel = config["neo4j"]["relationships"]["job_stream"]
    specialization_stream_rel = config["neo4j"]["relationships"][
        "specialization_stream"
    ]
    same_visitor_rel = config["neo4j"]["relationships"]["same_visitor"]
    attended_session_rel = config["neo4j"]["relationships"]["attended_session"]

    # Generate documentation
    doc = f"""# Neo4j Database Schema

## Node Labels

### {visitor_this_year}
Represents visitors registered for the current year's event.

**Properties:**
- `BadgeId` (unique identifier)
- `Email`
- `JobTitle`
- `Company`
- `Country`
- `job_role`
- `what_type_does_your_practice_specialise_in`
- `organisation_type`
- `BadgeType`
- `Source`
- `ShowRef`
- `BadgeId_last_year_bva` (if matched to a BVA visitor from last year)
- `BadgeId_last_year_lva` (if matched to an LVA visitor from last year)

### {visitor_last_year_bva}
Represents visitors from last year's BVA event.

**Properties:**
- `BadgeId` (unique identifier)
- `Email`
- `JobTitle`
- `Company`
- `Country`
- `job_role`
- `what_areas_do_you_specialise_in`
- `organisation_type`
- `BadgeType`
- `Source`
- `ShowRef`

### {visitor_last_year_lva}
Represents visitors from last year's LVA event.

**Properties:**
- `BadgeId` (unique identifier)
- `Email`
- `JobTitle`
- `Company`
- `Country`
- `job_role`
- `what_areas_do_you_specialise_in`
- `organisation_type`
- `BadgeType`
- `Source`
- `ShowRef`

### {session_this_year}
Represents sessions scheduled for the current year's event.

**Properties:**
- `session_id` (unique identifier)
- `title`
- `synopsis_stripped`
- `stream`
- `date`
- `start_time`
- `end_time`
- `theatre__name`
- `sponsored_session`
- `sponsored_by`
- `key_text`

### {session_past_year}
Represents sessions from past events.

**Properties:**
- `session_id` (unique identifier)
- `title`
- `synopsis_stripped`
- `stream`
- `date`
- `start_time`
- `end_time`
- `theatre__name`
- `sponsored_session`
- `sponsored_by`
- `key_text`

### {stream}
Represents stream categories that sessions belong to.

**Properties:**
- `stream` (unique identifier)
- `description`

## Relationships

### {session_stream_rel}
Connects sessions to their streams.

**From:** `{session_this_year}` or `{session_past_year}`  
**To:** `{stream}`  
**Properties:** None

### {job_stream_rel}
Connects visitors to recommended streams based on job roles.

**From:** `{visitor_this_year}`, `{visitor_last_year_bva}`, or `{visitor_last_year_lva}`  
**To:** `{stream}`  
**Properties:** None

### {specialization_stream_rel}
Connects visitors to relevant streams based on practice specializations.

**From:** `{visitor_this_year}`, `{visitor_last_year_bva}`, or `{visitor_last_year_lva}`  
**To:** `{stream}`  
**Properties:** None

### {same_visitor_rel}
Connects visitors this year to corresponding visitors from past events.

**From:** `{visitor_this_year}`  
**To:** `{visitor_last_year_bva}` or `{visitor_last_year_lva}`  
**Properties:**
- `type`: "bva" or "lva"

### {attended_session_rel}
Connects past visitors to sessions they attended.

**From:** `{visitor_last_year_bva}` or `{visitor_last_year_lva}`  
**To:** `{session_past_year}`  
**Properties:**
- `scan_time`: Timestamp when the visitor scanned into the session
"""

    return doc
