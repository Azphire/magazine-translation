import logging
import sys


def setup_logger(name="AgentPipeline"):
    """
    Sets up a global logger with formatted output.
    """
    logger = logging.getLogger(name)

    # Avoid adding multiple handlers if logger is imported multiple times
    if not logger.handlers:
        logger.setLevel(logging.INFO)

        # Create console handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)

        # Create formatting: [TIME] [LEVEL] MESSAGE
        formatter = logging.Formatter(
            fmt='%(asctime)s | [%(levelname)s] | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)

        logger.addHandler(handler)

    return logger


# Export a default global logger instance
logger = setup_logger()