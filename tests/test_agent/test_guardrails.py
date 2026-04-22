"""Tests for the guardrails safety module."""

from unittest.mock import patch

from ai_researcher.agent.guardrails import (
    MAX_RESEARCHER_ITERATIONS,
    MAX_WRITER_ITERATIONS,
    log_iteration_limit_reached,
)


class TestGuardrailConstants:
    """Tests for guardrail iteration limit constants."""

    def test_researcher_limit_is_positive(self):
        assert MAX_RESEARCHER_ITERATIONS > 0

    def test_writer_limit_is_positive(self):
        assert MAX_WRITER_ITERATIONS > 0

    def test_researcher_limit_is_4(self):
        """The default safety limit for the researcher is 4."""
        assert MAX_RESEARCHER_ITERATIONS == 4

    def test_writer_limit_is_4(self):
        """The default safety limit for the writer is 4."""
        assert MAX_WRITER_ITERATIONS == 4


class TestLogIterationLimitReached:
    """Tests for the guardrail logging function."""

    @patch("ai_researcher.agent.guardrails.logger")
    def test_researcher_guardrail_logs_warning(self, mock_logger):
        log_iteration_limit_reached("researcher", 4)
        mock_logger.warning.assert_called_once()
        call_args = mock_logger.warning.call_args[0]
        assert "GUARDRAIL TRIGGERED" in call_args[0]
        assert "Researcher" in str(call_args)

    @patch("ai_researcher.agent.guardrails.logger")
    def test_writer_guardrail_logs_warning(self, mock_logger):
        log_iteration_limit_reached("writer", 4)
        mock_logger.warning.assert_called_once()
        call_args = mock_logger.warning.call_args[0]
        assert "GUARDRAIL TRIGGERED" in call_args[0]
        assert "Writer" in str(call_args)

    @patch("ai_researcher.agent.guardrails.logger")
    def test_logs_correct_iteration_count(self, mock_logger):
        log_iteration_limit_reached("researcher", 3)
        call_args = mock_logger.warning.call_args[0]
        # The format string uses %d for current_iterations
        assert 3 in call_args
