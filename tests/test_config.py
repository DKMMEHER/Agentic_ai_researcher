"""Tests for the Settings configuration module."""

import os
import pytest
from unittest.mock import patch

from ai_researcher.config import Settings, get_settings


class TestSettingsValidation:
    """Tests for field validators in Settings."""

    def test_invalid_log_level_raises(self, monkeypatch):
        """log_level validator should reject an invalid level like 'TRACE'."""
        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        monkeypatch.setenv("LOG_LEVEL", "TRACE")
        # Clear cached settings so the new env vars are picked up
        get_settings.cache_clear()

        with pytest.raises(Exception) as exc_info:
            Settings()  # type: ignore[call-arg]
        assert "log_level" in str(exc_info.value).lower() or "TRACE" in str(exc_info.value)

    def test_empty_groq_key_raises(self, monkeypatch):
        """GROQ_API_KEY validator should reject empty or whitespace-only values."""
        monkeypatch.setenv("GROQ_API_KEY", "   ")
        get_settings.cache_clear()

        with pytest.raises(Exception):
            Settings()  # type: ignore[call-arg]

    def test_valid_log_level_uppercases(self, monkeypatch):
        """log_level should be uppercased by the validator."""
        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        monkeypatch.setenv("LOG_LEVEL", "debug")
        get_settings.cache_clear()

        settings = Settings()  # type: ignore[call-arg]
        assert settings.log_level == "DEBUG"

    def test_groq_key_is_stripped(self, monkeypatch):
        """GROQ_API_KEY should have whitespace stripped."""
        monkeypatch.setenv("GROQ_API_KEY", "  my-key  ")
        get_settings.cache_clear()

        settings = Settings()  # type: ignore[call-arg]
        assert settings.groq_api_key == "my-key"


class TestGetSettings:
    """Tests for the get_settings() cached factory."""

    def test_returns_settings_instance(self):
        """get_settings() should return a Settings instance."""
        get_settings.cache_clear()
        settings = get_settings()
        assert isinstance(settings, Settings)

    def test_pushes_groq_to_environ(self):
        """get_settings() should push GROQ_API_KEY into os.environ."""
        get_settings.cache_clear()
        settings = get_settings()
        assert os.environ.get("GROQ_API_KEY") == settings.groq_api_key

    def test_langsmith_env_push_when_enabled(self, monkeypatch):
        """When LangSmith tracing is 'true', env vars are pushed."""
        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        monkeypatch.setenv("LANGSMITH_TRACING_V2", "true")
        monkeypatch.setenv("LANGSMITH_API_KEY", "ls-test-key")
        monkeypatch.setenv("LANGSMITH_PROJECT", "test-project")
        get_settings.cache_clear()

        settings = get_settings()
        assert os.environ.get("LANGSMITH_TRACING_V2") == "true"
        assert os.environ.get("LANGSMITH_API_KEY") == "ls-test-key"
        assert os.environ.get("LANGSMITH_PROJECT") == "test-project"
