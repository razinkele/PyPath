"""Centralized logging configuration for PyPath Shiny app."""

import logging
import sys
from pathlib import Path

# Create logger
logger = logging.getLogger('pypath_app')
logger.setLevel(logging.DEBUG)

# Create console handler with formatting
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
console_handler.setFormatter(formatter)

# Add handler to logger
logger.addHandler(console_handler)

# Optional: File handler for persistent logs
log_dir = Path(__file__).parent.parent / 'logs'
if not log_dir.exists():
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / 'pypath_app.log')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception:
        # If can't create logs directory, just use console
        pass


def get_logger(name: str = None):
    """Get a logger instance.

    Parameters
    ----------
    name : str, optional
        Logger name (typically __name__). If None, returns root app logger.

    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    if name:
        return logging.getLogger(f'pypath_app.{name}')
    return logger
