"""Structured logging setup for the AI Researcher application.

Call `setup_logging()` once at application startup to configure all loggers.
"""

import logging
import logging.config


def setup_logging(level: str | None = None) -> None:
    """Configure structured logging for the entire application.

    In containerized environments (like Cloud Run), we only log to the console (stdout/stderr).
    Google Cloud Logging automatically captures these logs.
    """
    if level is None:
        try:
            from ai_researcher.config import get_settings

            level = get_settings().log_level
        except Exception:
            level = "INFO"

    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
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
        },
        "loggers": {
            "ai_researcher": {
                "level": level,
                "handlers": ["console"],
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
