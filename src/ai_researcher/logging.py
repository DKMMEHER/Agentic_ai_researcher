"""Structured logging setup for the AI Researcher application.

Call `setup_logging()` once at application startup to configure all loggers.
"""

import logging
import logging.config
from datetime import datetime
from pathlib import Path


def setup_logging(level: str | None = None) -> None:
    """Configure structured logging for the entire application.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
               If None, reads from settings.
    """
    if level is None:
        from ai_researcher.config import get_settings

        level = get_settings().log_level

    # Determine project root and create logs directory
    project_root = Path(__file__).resolve().parent.parent.parent
    logs_dir = project_root / "logs"
    logs_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = str(logs_dir / f"ai_researcher_{timestamp}.log")

    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "detailed": {
                "format": (
                    "%(asctime)s | %(levelname)-8s | %(name)s | "
                    "%(filename)s:%(lineno)d | %(funcName)s | %(message)s"
                ),
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": level,
                "formatter": "standard",
                "stream": "ext://sys.stdout",
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "DEBUG",
                "formatter": "detailed",
                "filename": log_file_path,
                "maxBytes": 5_242_880,  # 5 MB
                "backupCount": 3,
                "encoding": "utf-8",
            },
        },
        "loggers": {
            "ai_researcher": {
                "level": level,
                "handlers": ["console", "file"],
                "propagate": False,
            },
        },
        "root": {
            "level": "WARNING",
            "handlers": ["console"],
        },
    }

    logging.config.dictConfig(config)


def get_logger(name: str) -> logging.Logger:
    """Get a logger under the ai_researcher namespace.

    Args:
        name: Module name (typically __name__).

    Returns:
        Configured logger instance.
    """
    # Ensure logger is under our namespace
    if not name.startswith("ai_researcher"):
        name = f"ai_researcher.{name}"
    return logging.getLogger(name)
