"""Shared pytest configuration for isolated unit tests."""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5433/aiagent_test")
os.environ.setdefault("JWT_SECRET", "unit-test-secret")
os.environ.setdefault("JWT_EXPIRES_SECONDS", "3600")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/15")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("RAG_COLLECTION_NAME", "unit_test_rag_chunks")
os.environ.setdefault("EMBEDDING_MODEL", "unit-test-embedding")
os.environ.setdefault("OLLAMA_URL", "http://localhost:11434")
os.environ.setdefault("TEXT_MODEL", "unit-test-text-model")
os.environ.setdefault("LANGSMITH_TRACING", "false")
os.environ.setdefault("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
os.environ.setdefault("LANGSMITH_API_KEY", "")
os.environ.setdefault("LANGSMITH_PROJECT", "unit-tests")
os.environ.setdefault("LANGGRAPH_CHECKPOINT_DB_URL", "postgresql://postgres:postgres@localhost:5433/aiagent_test")

# Avoid opening a psycopg connection pool when importing agent.rag/main in unit tests.
langgraph_memory_stub = types.ModuleType("agent.langgraph_memory")
langgraph_memory_stub.checkpointer = None  # type: ignore[attr-defined]
langgraph_memory_stub.setup_langgraph_checkpointer = lambda: None  # type: ignore[attr-defined]
sys.modules.setdefault("agent.langgraph_memory", langgraph_memory_stub)
