# This file makes the utils directory a Python package.
# Import commonly used functions for easier access

from .logging_utils import setup_logging, log_function_call
from .config_utils import load_config
from .data_utils import (
    load_json_data,
    load_csv_data,
    save_dataframe,
    clean_column_names,
    merge_dataframes,
    fill_missing_values,
    parse_date,
    remove_duplicate_rows,
    validate_dataframe,
)
from .neo4j_utils import (
    Neo4jConnection,
    create_node_unique,
    create_relationship_unique,
    check_node_exists,
    get_node_by_property,
    create_constraints,
)
from .summary_utils import generate_and_save_summary

# Make vet_specific_functions available but don't import it automatically
# It will be imported when needed by the pipeline
try:
    from . import vet_specific_functions
except ImportError:
    # vet_specific_functions is optional
    pass