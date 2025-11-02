"""logger."""

import logging
from typing import Optional

def setup_logger(name: str, log_file: Optional[str] = None, level=logging.INFO):
    """Set up a configurable logger used across the project."""
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler = logging.FileHandler(log_file) if log_file else logging.StreamHandler()
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger