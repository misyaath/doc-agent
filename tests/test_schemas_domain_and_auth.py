"""Unit tests for domain enums, API schemas, password hashing, and JWT helpers."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from domain.file_process import FileStage, FileStageStatus
from schemas.agent import AgentChatRequest, AgentChatResponse
from schemas.chat import ChatCreateRequest, ChatCreateResponse, ChatDeleteResponse
from schemas.file import FileUploadItemResponse, FileUploadResponse
from schemas.user import TokenResponse, UserLoginRequest, UserRegisterRequest, UserRegisterResponse
from services.auth_service import create_access_token, hash_password, verify_jwt, verify_password


def test_file_stage_enums_have_expected_values() -> None:
    """Verify file stage enums have expected values."""
    assert FileStage.UPLOADED == "uploaded"
    assert FileStage.EXTRACTING == "extracting"
    assert FileStage.ANALYSING == "analysing"
    assert FileStage.ORGANIZING == "organizing"
    assert FileStage.SUMMARIZING == "summarizing"
    assert FileStage.SAVING == "saving"
    assert FileStage.DONE == "done"
    assert FileStageStatus.WAITING == "waiting"
    assert FileStageStatus.STARTED == "started"
    assert FileStageStatus.PROCESSING == "processing"
    assert FileStageStatus.DONE == "done"
    assert FileStageStatus.FAILED == "failed"


def test_user_request_schemas_normalize_and_validate_input() -> None:
    """Verify user request schemas normalize and validate input."""
    register = UserRegisterRequest(full_name=" Jane Doe ", email=" JANE@EXAMPLE.COM ", password="strongpassword123")
    assert register.full_name == "Jane Doe"
    assert register.email == "jane@example.com"

    login = UserLoginRequest(email=" JANE@EXAMPLE.COM ", password="strongpassword123")
    assert login.email == "jane@example.com"

    with pytest.raises(ValidationError):
        UserRegisterRequest(full_name=" ", email="bad", password="short")

    chat = ChatCreateRequest(name=" Research notes ")
    assert chat.name == "Research notes"

    with pytest.raises(ValidationError):
        ChatCreateRequest(name=" ")


def test_response_schemas_serialize_expected_fields() -> None:
    """Verify response schemas serialize expected fields."""
    user = UserRegisterResponse(id=1, full_name="Jane Doe", email="jane@example.com", is_active=True, is_verified=False)
    token = TokenResponse(access_token="token", expires_in=3600)
    chat = ChatCreateResponse(user_id=1, chat_id="chat-1", name="Research notes")
    deleted_chat = ChatDeleteResponse(chat_id="chat-1", deleted=True)
    file_item = FileUploadItemResponse(
        file_id="file-1",
        chat_id="chat-1",
        file_name="document.pdf",
        unique_generated_name="generated.pdf",
        full_path="/tmp/generated.pdf",
    )
    upload = FileUploadResponse(user_id=1, chat_id="chat-1", files=[file_item])
    agent = AgentChatResponse(chat_id="chat-1", prompt="question", answer="answer")

    assert user.model_dump()["email"] == "jane@example.com"
    assert token.token_type == "bearer"
    assert chat.chat_id == "chat-1"
    assert chat.name == "Research notes"
    assert deleted_chat.deleted is True
    assert upload.files[0].file_id == "file-1"
    assert agent.retrieved_chunks == []


def test_agent_chat_request_accepts_query_alias() -> None:
    """Verify agent chat request accepts query alias."""
    payload = AgentChatRequest(**{"chat_id": "chat-1", "query": "What is this document about?"})
    assert payload.prompt == "What is this document about?"


def test_password_hash_and_jwt_helpers_round_trip() -> None:
    """Verify password hash and jwt helpers round trip."""
    password_hash = hash_password("secret123")
    assert verify_password("secret123", password_hash)
    assert not verify_password("wrong", password_hash)

    token, expires_in = create_access_token("42")
    payload = verify_jwt(token)
    assert payload["sub"] == "42"
    assert expires_in == 3600
