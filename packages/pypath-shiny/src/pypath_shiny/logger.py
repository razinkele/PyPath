"""Centralized logging configuration for PyPath Shiny app.

Import-time side effects are deliberately avoided: call `configure_logging()`
once from the application entry point. Library code should only ever do
`logger = logging.getLogger(__name__)`.
"""

import logging
import sys
from pathlib import Path

APP_LOGGER_NAME = "pypath_app"

_FORMAT = (
    "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
)
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Module-level handle for convenience. Handlers are attached only by
# configure_logging(), so importing this module has no side effects.
logger = logging.getLogger(APP_LOGGER_NAME)


def configure_logging(
    level: int = logging.INFO,
    log_dir: Path | str | None = None,
    to_file: bool = True,
) -> logging.Logger:
    """Attach console (and optionally file) handlers to the app logger.

    Safe to call more than once: existing handlers are cleared first, so
    repeated calls do not duplicate log lines.

    Parameters
    ----------
    level : int
        Level for the console handler.
    log_dir : Path or str, optional
        Directory for the log file. Defaults to a `logs/` directory beside the
        package.
    to_file : bool
        Whether to also write to a rotating-free plain log file.

    Returns
    -------
    logging.Logger
        The configured application logger.
    """
    logger = logging.getLogger(APP_LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    formatter = logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if to_file:
        directory = Path(log_dir) if log_dir else Path(__file__).parent.parent / "logs"
        try:
            # `exist_ok=True` matters: the handler must be attached on every
            # run, not only the first one that creates the directory.
            directory.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(
                directory / "pypath_app.log", encoding="utf-8"
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except OSError as e:
            logger.warning("Could not create log directory '%s': %s", directory, e)

    return logger


def get_logger(name: str = None) -> logging.Logger:
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
        return logging.getLogger(f"{APP_LOGGER_NAME}.{name}")
    return logging.getLogger(APP_LOGGER_NAME)
