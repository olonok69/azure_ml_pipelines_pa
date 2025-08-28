import logging
import os
import functools


def setup_logging(log_file="data_processing.log", log_level=logging.INFO):
    """
    Set up logging configuration with file and line number information.

    Args:
        log_file: Path to the log file
        log_level: Logging level
    """
    # Create directory for log file if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s",
        filename=log_file,
        filemode="w",
    )

    # Create console handler for terminal output
    console = logging.StreamHandler()
    console.setLevel(log_level)
    console_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console.setFormatter(console_format)

    # Add console handler to root logger
    logging.getLogger("").addHandler(console)

    return logging.getLogger(__name__)


# Decorator for logging function entry and exit (optional)
def log_function_call(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        logger.debug(f"Entering {func.__name__}")
        result = func(*args, **kwargs)
        logger.debug(f"Exiting {func.__name__}")
        return result

    return wrapper
