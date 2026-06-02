# agent/langgraph_memory.py

from __future__ import annotations

from psycopg_pool import ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver

from core.settings import settings

checkpoint_pool = ConnectionPool(
    conninfo=settings.langgraph_checkpoint_db_url,
    max_size=10,
    kwargs={
        "autocommit": True,
        "prepare_threshold": 0,
    },
)

checkpointer = PostgresSaver(checkpoint_pool)


def setup_langgraph_checkpointer() -> None:
    checkpointer.setup()
