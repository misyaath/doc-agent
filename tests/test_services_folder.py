"""Folder-level tests for service classes using repository and cache fakes."""

from __future__ import annotations

import json
from datetime import datetime
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException

from schemas.agent import AgentChatRequest
from schemas.chat import ChatHistoryMessageResponse
from schemas.user import UserLoginRequest, UserRegisterRequest
from services.agent_service import AgentService, FileSummaryCacheService
from services.auth_service import hash_password
from services.chat_service import ChatService
from services.user_service import UserService


class FakeChatRepository:
    """Async chat repository double used by chat and agent services."""

    def __init__(self, exists: bool = True) -> None:
        """Configure whether repository lookups should return a chat."""
        self.exists = exists
        self.created: list[dict[str, Any]] = []
        self.chats: list[SimpleNamespace] = []

    async def create(self, *, chat_id: str, user_id: int, name: str) -> SimpleNamespace:
        """Record chat creation without touching a database."""
        self.created.append({"chat_id": chat_id, "user_id": user_id, "name": name})
        return SimpleNamespace(id=chat_id, user_id=user_id, name=name)

    async def get_by_id_and_user_id(self, *, chat_id: str, user_id: int) -> SimpleNamespace | None:
        """Return a fake chat only when the test config allows it."""
        if not self.exists:
            return None
        return SimpleNamespace(chat_id=chat_id, user_id=user_id)

    async def get_by_id_and_user_id_with_files(self, *, chat_id: str, user_id: int) -> SimpleNamespace | None:
        """Return a fake eager-loaded chat only when the test config allows it."""
        if not self.exists:
            return None
        return self.chats[0] if self.chats else None

    async def list_by_user_id_with_files(self, user_id: int) -> list[SimpleNamespace]:
        """Return fake chats for list endpoint tests."""
        return self.chats


class FakeUserRepository:
    """Async user repository double used by user service tests."""

    def __init__(self, existing_user: SimpleNamespace | None = None) -> None:
        """Store the current user record returned for email lookups."""
        self.existing_user = existing_user
        self.created_payload: dict[str, Any] | None = None

    async def get_by_email(self, email: str) -> SimpleNamespace | None:
        """Return the configured fake user for duplicate and login tests."""
        return self.existing_user if self.existing_user and self.existing_user.email == email else None

    async def create(
        self,
        *,
        full_name: str,
        email: str,
        password_hash: str,
        is_active: bool,
        is_verified: bool,
    ) -> SimpleNamespace:
        """Record and return a new fake user."""
        self.created_payload = {
            "full_name": full_name,
            "email": email,
            "password_hash": password_hash,
            "is_active": is_active,
            "is_verified": is_verified,
        }
        return SimpleNamespace(id=7, **self.created_payload)


class FakeRedis:
    """In-memory Redis double for file summary cache tests."""

    def __init__(self, cached_value: str | None = None) -> None:
        """Initialize optional cached JSON and capture writes."""
        self.cached_value = cached_value
        self.writes: dict[str, str] = {}

    def get(self, key: str) -> str | None:
        """Return the configured cached value."""
        return self.cached_value

    def set(self, key: str, value: str) -> None:
        """Capture cache writes for assertions."""
        self.writes[key] = value


class FakeFileSummaryRepository:
    """Async file summary repository double for cache miss tests."""

    def __init__(self) -> None:
        """Initialize deterministic summary records."""
        self.calls: list[str] = []

    async def get_title_summaries_by_chat_id(self, chat_id: str) -> list[dict[str, Any]]:
        """Return file summaries for the requested chat."""
        self.calls.append(chat_id)
        return [{"title": "Doc", "summary": [{"heading": "Intro"}]}]


class FakeFileSummaryCache:
    """Async file summary cache double for agent service tests."""

    def __init__(self, summaries: list[dict[str, Any]]) -> None:
        """Store summaries returned by get_or_load."""
        self.summaries = summaries

    async def get_or_load(self, chat_id: str) -> list[dict[str, Any]]:
        """Return configured summaries for the chat."""
        return self.summaries


class FakeRunner:
    """RAG runner double that returns deterministic agent output."""

    def run(self, **_: Any) -> dict[str, Any]:
        """Return an answer without building a LangGraph graph."""
        return {"answer": "Service answer", "context": "Service context"}


@pytest.mark.asyncio
async def test_chat_service_creates_chat_response_and_repository_record() -> None:
    """Verify chat service creates a chat id and persists it via the repository."""
    repository = FakeChatRepository()
    response = await ChatService(chat_repository=repository).create_chat(user_id=42, name="Research")  # type: ignore[arg-type]

    assert response.user_id == 42
    assert response.name == "Research"
    assert repository.created == [{"chat_id": response.chat_id, "user_id": 42, "name": "Research"}]


@pytest.mark.asyncio
async def test_chat_service_lists_chats_with_file_process_data() -> None:
    """Verify chat service returns nested file metadata and aggregate processing status."""
    repository = FakeChatRepository()
    now = datetime(2026, 6, 4, 11, 30, 0)
    repository.chats = [
        SimpleNamespace(
            id="chat-1",
            name="Research",
            created_at=now,
            updated_at=None,
            files=[
                SimpleNamespace(
                    file_id="file-1",
                    file_name="document.pdf",
                    title="Document title",
                    summary=[{"heading": "Intro"}],
                    created_at=now,
                    process_stages=[
                        SimpleNamespace(stage="uploaded", status="done", created_at=now, updated_at=now),
                        SimpleNamespace(stage="done", status="done", created_at=now, updated_at=now),
                    ],
                )
            ],
        )
    ]

    response = await ChatService(chat_repository=repository).list_chats(user_id=42)  # type: ignore[arg-type]

    assert response.user_id == 42
    assert response.chats[0].name == "Research"
    assert response.chats[0].files[0].title == "Document title"
    assert response.chats[0].files[0].summary == [{"heading": "Intro"}]
    assert response.chats[0].files[0].process_status == "done"
    assert response.chats[0].files[0].process_stages[0].stage == "uploaded"


@pytest.mark.asyncio
async def test_chat_service_returns_chat_detail_with_langgraph_history() -> None:
    """Verify chat detail includes files and injected LangGraph history."""
    repository = FakeChatRepository()
    now = datetime(2026, 6, 4, 11, 30, 0)
    repository.chats = [
        SimpleNamespace(
            id="chat-1",
            name="Research",
            created_at=now,
            updated_at=None,
            files=[
                SimpleNamespace(
                    file_id="file-1",
                    file_name="document.pdf",
                    title="Document title",
                    summary=[{"heading": "Intro"}],
                    created_at=now,
                    process_stages=[
                        SimpleNamespace(stage="done", status="done", created_at=now, updated_at=now),
                    ],
                )
            ],
        )
    ]

    service = ChatService(
        chat_repository=repository,  # type: ignore[arg-type]
        chat_history_loader=lambda chat_id: [
            ChatHistoryMessageResponse(role="user", content=f"question for {chat_id}"),
            ChatHistoryMessageResponse(role="assistant", content="answer"),
        ],
    )
    response = await service.get_chat_detail(chat_id="chat-1", user_id=42)

    assert response.chat_id == "chat-1"
    assert response.files[0].process_status == "done"
    assert response.history[0].role == "user"
    assert response.history[0].content == "question for chat-1"


@pytest.mark.asyncio
async def test_user_service_registers_and_logs_in_active_user() -> None:
    """Verify user service registration and login use normalized credentials."""
    repository = FakeUserRepository()
    service = UserService(user_repository=repository)  # type: ignore[arg-type]

    registered = await service.register(
        UserRegisterRequest(full_name=" Jane Doe ", email="JANE@example.com", password="strongpassword123")
    )

    assert registered.email == "jane@example.com"
    assert repository.created_payload is not None
    assert repository.created_payload["full_name"] == "Jane Doe"

    repository.existing_user = SimpleNamespace(
        id=registered.id,
        email=registered.email,
        password_hash=repository.created_payload["password_hash"],
    )
    token = await service.login(UserLoginRequest(email="jane@example.com", password="strongpassword123"))
    assert token.access_token
    assert token.expires_in == 3600


@pytest.mark.asyncio
async def test_user_service_rejects_duplicate_and_invalid_login() -> None:
    """Verify user service raises HTTP errors for duplicate or bad credentials."""
    existing = SimpleNamespace(id=1, email="jane@example.com", password_hash=hash_password("secret123"))
    service = UserService(user_repository=FakeUserRepository(existing_user=existing))  # type: ignore[arg-type]

    with pytest.raises(HTTPException) as duplicate_error:
        await service.register(
            UserRegisterRequest(full_name="Jane", email="jane@example.com", password="strongpassword123")
        )
    assert duplicate_error.value.status_code == 409

    with pytest.raises(HTTPException) as login_error:
        await service.login(UserLoginRequest(email="jane@example.com", password="wrongpassword"))
    assert login_error.value.status_code == 401


@pytest.mark.asyncio
async def test_file_summary_cache_reads_cache_and_populates_miss() -> None:
    """Verify file summary cache prefers Redis and stores repository misses."""
    cached = [{"title": "Cached", "summary": []}]
    redis = FakeRedis(cached_value=json.dumps(cached))
    repository = FakeFileSummaryRepository()
    cache = FileSummaryCacheService(file_repository=repository, redis_client=redis)  # type: ignore[arg-type]

    assert await cache.get_or_load("chat-1") == cached
    assert repository.calls == []

    miss_redis = FakeRedis()
    miss_repository = FakeFileSummaryRepository()
    miss_cache = FileSummaryCacheService(file_repository=miss_repository, redis_client=miss_redis)  # type: ignore[arg-type]
    loaded = await miss_cache.get_or_load("chat-2")

    assert loaded == [{"title": "Doc", "summary": [{"heading": "Intro"}]}]
    assert miss_repository.calls == ["chat-2"]
    assert json.loads(miss_redis.writes[FileSummaryCacheService.cache_key("chat-2")]) == loaded


@pytest.mark.asyncio
async def test_agent_service_prepare_chat_authorizes_and_requires_indexed_files() -> None:
    """Verify agent service validates chat ownership and indexed file summaries."""
    payload = AgentChatRequest(chat_id="chat-1", prompt="Question?")
    service = AgentService(
        chat_repository=FakeChatRepository(),  # type: ignore[arg-type]
        file_summary_cache=FakeFileSummaryCache([{"title": "Doc", "summary": [{"heading": "Intro"}]}]),  # type: ignore[arg-type]
        runner=FakeRunner(),  # type: ignore[arg-type]
    )

    summary = await service._prepare_chat(payload=payload, user_id=1)
    assert summary == [{"title": "Doc", "summary": [{"heading": "Intro"}]}]

    forbidden = AgentService(
        chat_repository=FakeChatRepository(exists=False),  # type: ignore[arg-type]
        file_summary_cache=FakeFileSummaryCache([]),  # type: ignore[arg-type]
        runner=FakeRunner(),  # type: ignore[arg-type]
    )
    with pytest.raises(HTTPException) as forbidden_error:
        await forbidden._prepare_chat(payload=payload, user_id=1)
    assert forbidden_error.value.status_code == 403

    no_files = AgentService(
        chat_repository=FakeChatRepository(),  # type: ignore[arg-type]
        file_summary_cache=FakeFileSummaryCache([]),  # type: ignore[arg-type]
        runner=FakeRunner(),  # type: ignore[arg-type]
    )
    with pytest.raises(HTTPException) as no_files_error:
        await no_files._prepare_chat(payload=payload, user_id=1)
    assert no_files_error.value.status_code == 400
