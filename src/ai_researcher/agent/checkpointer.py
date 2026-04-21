"""Checkpointer factory for durable LangGraph session persistence.

Supports three backends:
  - "memory"   : In-memory only (MemorySaver). Sessions lost on restart.
  - "sqlite"   : File-based SQLite (AsyncSqliteSaver). Zero-config, local dev.
"""

import logging
from typing import Literal
from langgraph.checkpoint.memory import MemorySaver

logger = logging.getLogger(__name__)

CheckpointBackend = Literal["memory", "sqlite"]


def get_checkpointer(backend: CheckpointBackend = "sqlite", db_url: str = "checkpoints.db"):
    """Return a configured LangGraph checkpointer.

    Args:
        backend: One of "memory" or "sqlite".
        db_url: Database path for SQLite (e.g. "checkpoints.db").

    Returns:
        A ready-to-use checkpointer instance.
    """
    if backend == "memory":
        logger.info("Using MemorySaver checkpointer (in-memory, not persistent)")
        return MemorySaver()

    elif backend == "sqlite":
        try:
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
            logger.info("Using AsyncSqliteSaver checkpointer (db=%s)", db_url)
            # from_conn_string() returns an async context manager.
            # LangGraph's graph.compile() accepts this directly.
            return AsyncSqliteSaver.from_conn_string(db_url)
        except ImportError:
            logger.warning(
                "langgraph-checkpoint-sqlite not installed. "
                "Falling back to MemorySaver. Run: uv pip install langgraph-checkpoint-sqlite"
            )
            return MemorySaver()

    else:
        logger.warning("Unknown checkpoint backend '%s', falling back to MemorySaver", backend)
        return MemorySaver()
