# agent/langgraph_memory.py

from __future__ import annotations

from typing import Any, cast

from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool

from core.settings import settings

checkpoint_pool = ConnectionPool(
    conninfo=settings.langgraph_checkpoint_db_url,
    max_size=10,
    kwargs={
        "autocommit": True,
        "prepare_threshold": 0,
    },
)

checkpointer = PostgresSaver(cast(Any, checkpoint_pool))


def setup_langgraph_checkpointer() -> None:
    """
    Setup langgraph checkpointer.

    Purpose:
        Implements setup_langgraph_checkpointer for the RAG agent layer that builds
            prompts, retrieves context, and generates answers.
    Args:
        None.
    Returns:
        None: Performs work through side effects and does not return a value.
    Why Added:
        Provides a documented entry point for this module-level behavior and keeps
            callers
        from needing to know lower-level implementation details.
    """
    checkpointer.setup()
