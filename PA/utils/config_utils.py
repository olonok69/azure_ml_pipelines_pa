import yaml
import logging


def load_config(config_file: str):
    """
    Load configuration from YAML file.

    Args:
        config_file: Path to the configuration file

    Returns:
        Dict containing the configuration

    Raises:
        Exception: If there's an error loading the configuration
    """
    logger = logging.getLogger(__name__)
    try:
        with open(config_file, "r") as file:
            config = yaml.safe_load(file)
        logger.info(f"Successfully loaded configuration from {config_file}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}", exc_info=True)
        raise
