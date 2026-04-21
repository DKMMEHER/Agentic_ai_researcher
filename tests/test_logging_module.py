"""Tests for the logging module."""

import logging

from ai_researcher.logging import get_logger, setup_logging


class TestGetLogger:
    """Tests for get_logger namespace behavior."""

    def test_adds_namespace_prefix(self):
        """A module name without 'ai_researcher' prefix gets it added."""
        logger = get_logger("my_module")
        assert logger.name == "ai_researcher.my_module"

    def test_preserves_existing_namespace(self):
        """A module name already starting with 'ai_researcher' is kept as is."""
        logger = get_logger("ai_researcher.tools.arxiv")
        assert logger.name == "ai_researcher.tools.arxiv"

    def test_returns_logger_instance(self):
        """get_logger should return a stdlib logging.Logger."""
        logger = get_logger("test")
        assert isinstance(logger, logging.Logger)


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_setup_logging_does_not_crash(self, tmp_path, monkeypatch):
        """setup_logging should complete without error when called explicitly."""
        # Avoid polluting the project logs directory
        setup_logging(level="WARNING")
        # Should not raise — just validates it's callable
