"""Tests for the checkpointer factory (Track 2: Persistent Checkpointing)."""

import os
import tempfile

import pytest

from ai_researcher.agent.checkpointer import get_checkpointer


class TestGetCheckpointer:
    """Unit tests for get_checkpointer factory function."""

    def test_memory_backend_returns_memory_saver(self):
        """memory backend should return a MemorySaver instance."""
        from langgraph.checkpoint.memory import MemorySaver

        checkpointer = get_checkpointer(backend="memory")
        assert isinstance(checkpointer, MemorySaver)

    def test_sqlite_backend_returns_sqlite_saver(self):
        """sqlite backend should return a context manager from SqliteSaver."""
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: F401
        except ImportError:
            pytest.skip("langgraph-checkpoint-sqlite not installed")

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            checkpointer = get_checkpointer(backend="sqlite", db_url=db_path)
            # from_conn_string() returns a context manager, not SqliteSaver directly.
            # That context manager is what LangGraph's compile() expects.
            assert hasattr(checkpointer, "__enter__"), (
                "Expected a context manager, got: " + str(type(checkpointer))
            )
        finally:
            os.unlink(db_path)

    def test_sqlite_is_default_backend(self):
        """Default backend should be sqlite — returns a context manager."""
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: F401
        except ImportError:
            pytest.skip("langgraph-checkpoint-sqlite not installed")

        checkpointer = get_checkpointer()
        # Result is an async context manager (not MemorySaver)
        from langgraph.checkpoint.memory import MemorySaver

        assert not isinstance(checkpointer, MemorySaver), (
            "Default should be SQLite, not MemorySaver"
        )
        assert hasattr(checkpointer, "__enter__"), (
            "Expected a context manager from SQLite backend"
        )

    def test_unknown_backend_falls_back_to_memory(self):
        """Unknown backend should gracefully fall back to MemorySaver."""
        from langgraph.checkpoint.memory import MemorySaver

        checkpointer = get_checkpointer(backend="invalid_backend")  # type: ignore
        assert isinstance(checkpointer, MemorySaver)

    def test_sqlite_creates_db_file_on_use(self, tmp_path):
        """SQLite checkpointer should be configured with the given db path."""
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: F401
        except ImportError:
            pytest.skip("langgraph-checkpoint-sqlite not installed")

        db_file = tmp_path / "test_checkpoints.db"
        checkpointer = get_checkpointer(backend="sqlite", db_url=str(db_file))
        # Should return a context manager (not None or MemorySaver)
        assert checkpointer is not None
        assert hasattr(checkpointer, "__enter__"), (
            "Expected context manager for SQLite backend"
        )

    def test_settings_expose_checkpoint_fields(self):
        """Settings should have checkpoint_backend and checkpoint_db_url fields."""
        from ai_researcher.config import get_settings

        settings = get_settings()
        assert hasattr(settings, "checkpoint_backend")
        assert hasattr(settings, "checkpoint_db_url")
        assert settings.checkpoint_backend in ("memory", "sqlite")

    def test_default_checkpoint_backend_is_sqlite(self, monkeypatch):
        """Default backend from settings should be sqlite."""
        monkeypatch.setenv("CHECKPOINT_BACKEND", "sqlite")
        from ai_researcher.config import get_settings

        get_settings.cache_clear()
        settings = get_settings()
        assert settings.checkpoint_backend == "sqlite"

    def test_default_checkpoint_db_url_is_set(self):
        """Default db_url from settings should be a non-empty string."""
        from ai_researcher.config import get_settings

        settings = get_settings()
        assert settings.checkpoint_db_url
        assert len(settings.checkpoint_db_url) > 0

    def test_build_graph_uses_sqlite_checkpointer(self, monkeypatch):
        """build_graph should produce a graph using the SQLite context manager by default."""
        monkeypatch.setenv("CHECKPOINT_BACKEND", "sqlite")
        from ai_researcher.config import get_settings

        get_settings.cache_clear()
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: F401
        except ImportError:
            pytest.skip("langgraph-checkpoint-sqlite not installed")

        from ai_researcher.agent.graph import build_graph

        graph, _config = build_graph()

        # The checkpointer is a context manager (from_conn_string result)
        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.graph.state import CompiledStateGraph

        # build_graph returns (graph, config)
        # compile() returns CompiledStateGraph
        assert isinstance(graph, CompiledStateGraph)
        assert graph.checkpointer is not None
        assert not isinstance(graph.checkpointer, MemorySaver), (
            "Graph should use SQLite, not MemorySaver, by default"
        )
        assert hasattr(graph.checkpointer, "__enter__"), (
            "Graph checkpointer should be a context manager"
        )

    def test_build_graph_config_has_thread_id(self):
        """build_graph should return a config dict with a thread_id."""
        from ai_researcher.agent.graph import build_graph

        _graph, config = build_graph(thread_id="test-thread-123")
        assert config["configurable"]["thread_id"] == "test-thread-123"

    def test_build_graph_memory_backend_override(self, monkeypatch):
        """build_graph should use MemorySaver when backend is set to 'memory'."""
        from langgraph.checkpoint.memory import MemorySaver

        from ai_researcher.config import get_settings

        # Temporarily override settings
        monkeypatch.setattr(
            "ai_researcher.agent.graph.get_settings",
            lambda: type(
                "S",
                (),
                {
                    **{
                        k: getattr(get_settings(), k)
                        for k in get_settings().model_fields
                    },
                    "checkpoint_backend": "memory",
                    "checkpoint_db_url": ":memory:",
                },
            )(),
        )

        from ai_researcher.agent.graph import build_graph

        graph, _config = build_graph()
        assert isinstance(graph.checkpointer, MemorySaver)
