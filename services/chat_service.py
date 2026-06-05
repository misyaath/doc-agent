from __future__ import annotations

import uuid
from collections.abc import Callable
from pathlib import Path
from shutil import rmtree
from typing import Any

from anyio import to_thread
from fastapi import Depends, HTTPException, status
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from sqlalchemy.ext.asyncio import AsyncSession

from core.settings import settings
from database import get_db
from domain.file_process import FileStage, FileStageStatus
from repositories.chat_repository import ChatRepository
from schemas.chat import (
    ChatCreateResponse,
    ChatDeleteResponse,
    ChatDetailResponse,
    ChatFileProcessStageResponse,
    ChatFileResponse,
    ChatHistoryMessageResponse,
    ChatListItemResponse,
    ChatListResponse,
)


def _file_process_status(stage_statuses: dict[str, str]) -> str:
    """
    File process status.

    Purpose:
        Computes a single frontend-friendly file processing status from per-stage
            statuses.
    Args:
        stage_statuses (dict[str, str]): Status keyed by file processing stage.
    Returns:
        str: Aggregate file processing status.
    """
    statuses = set(stage_statuses.values())
    if FileStageStatus.FAILED.value in statuses:
        return FileStageStatus.FAILED.value
    if stage_statuses.get(FileStage.DONE.value) == FileStageStatus.DONE.value:
        return FileStageStatus.DONE.value
    if statuses & {FileStageStatus.STARTED.value, FileStageStatus.PROCESSING.value}:
        return FileStageStatus.PROCESSING.value
    if statuses:
        return FileStageStatus.PROCESSING.value
    return FileStageStatus.WAITING.value


def _message_role(raw_role: str) -> str:
    """
    Message role.

    Purpose:
        Normalizes LangChain/LangGraph message type values into frontend roles.
    Args:
        raw_role (str): Message role or type from the checkpoint payload.
    Returns:
        str: Normalized API role.
    """
    role = raw_role.lower()
    if role in {"human", "user"}:
        return "user"
    if role in {"ai", "assistant"}:
        return "assistant"
    return role


def _message_content(content: Any) -> Any:
    """
    Message content.

    Purpose:
        Returns JSON-safe message content for chat history responses.
    Args:
        content (Any): Raw LangChain message content.
    Returns:
        Any: JSON-safe response content.
    """
    if isinstance(content, str | int | float | bool | list | dict) or content is None:
        return content
    return str(content)


def _history_from_messages(messages: Any) -> list[ChatHistoryMessageResponse]:
    """
    History from messages.

    Purpose:
        Extracts chat history from a LangGraph state messages channel when present.
    Args:
        messages (Any): Raw messages channel from the checkpoint payload.
    Returns:
        list[ChatHistoryMessageResponse]: Structured chat history.
    """
    if not isinstance(messages, list):
        return []

    history: list[ChatHistoryMessageResponse] = []
    for message in messages:
        raw_role = getattr(message, "type", None) or getattr(message, "role", None)
        content = getattr(message, "content", None)
        if raw_role is None and isinstance(message, dict):
            raw_role = message.get("type") or message.get("role")
            content = message.get("content")
        if not isinstance(raw_role, str):
            continue

        role = _message_role(raw_role)
        if role not in {"user", "assistant"}:
            continue
        history.append(ChatHistoryMessageResponse(role=role, content=_message_content(content)))
    return history


def _load_langgraph_chat_history(chat_id: str) -> list[ChatHistoryMessageResponse]:
    """
    Load langgraph chat history.

    Purpose:
        Reads persisted LangGraph checkpoints for a chat thread and returns user/assistant
            history for the API.
    Args:
        chat_id (str): Chat/session identifier used as the LangGraph thread_id.
    Returns:
        list[ChatHistoryMessageResponse]: Structured chat history.
    """
    from agent.langgraph_memory import checkpointer

    if checkpointer is None:
        return []

    config = {"configurable": {"thread_id": chat_id}}
    latest = checkpointer.get_tuple(config)
    if latest is not None:
        latest_values = latest.checkpoint.get("channel_values", {})
        messages_history = _history_from_messages(latest_values.get("messages"))
        if messages_history:
            return messages_history

    history: list[ChatHistoryMessageResponse] = []
    seen_turns: set[tuple[str, str]] = set()
    checkpoint_tuples = list(checkpointer.list(config, limit=200))
    for checkpoint_tuple in reversed(checkpoint_tuples):
        values = checkpoint_tuple.checkpoint.get("channel_values", {})
        question = values.get("question")
        answer = values.get("answer")
        if not isinstance(question, str) or not isinstance(answer, str):
            continue

        turn_key = (question, answer)
        if turn_key in seen_turns:
            continue
        seen_turns.add(turn_key)
        history.append(ChatHistoryMessageResponse(role="user", content=question))
        history.append(ChatHistoryMessageResponse(role="assistant", content=answer))

    return history


def _chat_files_response(chat: Any) -> list[ChatFileResponse]:
    """
    Chat files response.

    Purpose:
        Converts ORM file records into API response objects.
    Args:
        chat (Any): Chat ORM object with eager-loaded files.
    Returns:
        list[ChatFileResponse]: File response payloads.
    """
    return [
        ChatFileResponse(
            file_id=file.file_id,
            file_name=file.file_name,
            title=file.title,
            summary=file.summary,
            process_status=_file_process_status({stage.stage: stage.status for stage in file.process_stages}),
            process_stages=[
                ChatFileProcessStageResponse(
                    stage=stage.stage,
                    status=stage.status,
                    updated_at=stage.updated_at,
                )
                for stage in sorted(file.process_stages, key=lambda item: item.created_at)
            ],
        )
        for file in sorted(chat.files, key=lambda item: item.created_at)
    ]


def _delete_qdrant_chat_points(chat_id: str) -> None:
    """
    Delete qdrant chat points.

    Purpose:
        Deletes all vector points associated with a chat before local/database cleanup.
    Args:
        chat_id (str): Chat/session identifier used to scope vector records.
    Returns:
        None: Performs work through side effects and does not return a value.
    """
    client = QdrantClient(url=settings.qdrant_url)
    if not client.collection_exists(settings.rag_collection_name):
        return

    client.delete(
        collection_name=settings.rag_collection_name,
        points_selector=qdrant_models.FilterSelector(
            filter=qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="chat_id",
                        match=qdrant_models.MatchValue(value=chat_id),
                    )
                ]
            )
        ),
        wait=True,
    )


def _delete_chat_file_artifacts(chat: Any) -> None:
    """
    Delete chat file artifacts.

    Purpose:
        Deletes uploaded source files and extracted artifacts for a chat.
    Args:
        chat (Any): Chat ORM object with eager-loaded files.
    Returns:
        None: Performs work through side effects and does not return a value.
    """
    for file in chat.files:
        full_path = Path(file.full_path)
        if full_path.is_file() or full_path.is_symlink():
            full_path.unlink()

    extracted_path = Path("extracted_files") / str(chat.id)
    if extracted_path.exists():
        rmtree(extracted_path)


class ChatService:
    """
    Chat Service.

    Purpose:
        Defines ChatService in the business-service layer that coordinates repositories,
            security helpers, RAG execution, and API workflows.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.
    """

    def __init__(
        self,
        chat_repository: ChatRepository,
        chat_history_loader: Callable[[str], list[ChatHistoryMessageResponse]] = _load_langgraph_chat_history,
        qdrant_cleanup: Callable[[str], None] = _delete_qdrant_chat_points,
        file_artifact_cleanup: Callable[[Any], None] = _delete_chat_file_artifacts,
    ) -> None:
        """
        Initialize the object with its required dependencies.

        Purpose:
            Implements __init__ for the business-service layer that coordinates
                repositories, security helpers, RAG execution, and API workflows.
        Class:
            Belongs to ChatService; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
            chat_repository (ChatRepository): Input value for the chat repository
                parameter.
            chat_history_loader (Callable[[str], list[ChatHistoryMessageResponse]]):
                Function used to load persisted LangGraph chat history.
            qdrant_cleanup (Callable[[str], None]): Function used to delete Qdrant
                points for the chat.
            file_artifact_cleanup (Callable[[Any], None]): Function used to delete
                uploaded and extracted files for the chat.
        Returns:
            None: Performs work through side effects and does not return a value.
        Why Added:
            Centralizes this behavior inside ChatService so related code remains
                cohesive and testable.
        """
        self._chat_repository = chat_repository
        self._chat_history_loader = chat_history_loader
        self._qdrant_cleanup = qdrant_cleanup
        self._file_artifact_cleanup = file_artifact_cleanup

    async def create_chat(self, *, user_id: int, name: str = "New Chat") -> ChatCreateResponse:
        """
        Create chat.

        Purpose:
            Implements create_chat for the business-service layer that coordinates
                repositories, security helpers, RAG execution, and API workflows.
        Class:
            Belongs to ChatService; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
            user_id (int): Authenticated user identifier used to scope the operation.
            name (str): Human-readable chat name supplied by the user.
        Returns:
            ChatCreateResponse: API response model returned to the client.
        Why Added:
            Centralizes this behavior inside ChatService so related code remains
                cohesive and testable.
        """
        chat_id = str(uuid.uuid4())
        chat = await self._chat_repository.create(chat_id=chat_id, user_id=user_id, name=name)
        return ChatCreateResponse(user_id=user_id, chat_id=chat.id, name=chat.name)

    async def list_chats(self, user_id: int) -> ChatListResponse:
        """
        List chats.

        Purpose:
            Implements list_chats for the business-service layer that coordinates
                repositories, security helpers, RAG execution, and API workflows.
        Class:
            Belongs to ChatService; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
            user_id (int): Authenticated user identifier used to scope the operation.
        Returns:
            ChatListResponse: API response model returned to the client.
        Why Added:
            Centralizes this behavior inside ChatService so related code remains
                cohesive and testable.
        """
        chats = await self._chat_repository.list_by_user_id_with_files(user_id)
        return ChatListResponse(
            user_id=user_id,
            chats=[
                ChatListItemResponse(
                    chat_id=chat.id,
                    name=chat.name,
                    created_at=chat.created_at,
                    updated_at=chat.updated_at,
                    files=_chat_files_response(chat),
                )
                for chat in chats
            ],
        )

    async def get_chat_detail(self, *, chat_id: str, user_id: int) -> ChatDetailResponse:
        """
        Get chat detail.

        Purpose:
            Implements get_chat_detail for the business-service layer that coordinates
                repositories, security helpers, RAG execution, and API workflows.
        Class:
            Belongs to ChatService; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
            chat_id (str): Chat/session identifier used to scope retrieval, tasks, or
                responses.
            user_id (int): Authenticated user identifier used to scope the operation.
        Returns:
            ChatDetailResponse: API response model returned to the client.
        Why Added:
            Centralizes this behavior inside ChatService so related code remains
                cohesive and testable.
        """
        chat = await self._chat_repository.get_by_id_and_user_id_with_files(chat_id=chat_id, user_id=user_id)
        if chat is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat not found")

        try:
            history = await to_thread.run_sync(lambda: self._chat_history_loader(chat_id))
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Chat history is unavailable",
            ) from exc

        return ChatDetailResponse(
            chat_id=chat.id,
            name=chat.name,
            created_at=chat.created_at,
            updated_at=chat.updated_at,
            files=_chat_files_response(chat),
            history=history,
        )

    async def delete_chat(self, *, chat_id: str, user_id: int) -> ChatDeleteResponse:
        """
        Delete chat.

        Purpose:
            Implements delete_chat for the business-service layer that coordinates
                external vector cleanup, filesystem cleanup, and database deletion.
        Class:
            Belongs to ChatService; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
            chat_id (str): Chat/session identifier used to scope deletion.
            user_id (int): Authenticated user identifier used to scope the operation.
        Returns:
            ChatDeleteResponse: API response model returned to the client.
        Why Added:
            Centralizes this behavior inside ChatService so related code remains
                cohesive and testable.
        """
        chat = await self._chat_repository.get_by_id_and_user_id_with_files(chat_id=chat_id, user_id=user_id)
        if chat is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat not found")

        try:
            await to_thread.run_sync(lambda: self._qdrant_cleanup(chat_id))
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Failed to delete chat vectors",
            ) from exc

        try:
            await to_thread.run_sync(lambda: self._file_artifact_cleanup(chat))
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete chat files",
            ) from exc

        await self._chat_repository.delete(chat)
        return ChatDeleteResponse(chat_id=chat_id, deleted=True)


def get_chat_service(db: AsyncSession = Depends(get_db)) -> ChatService:
    """
    Get chat service.

    Purpose:
        Implements get_chat_service for the business-service layer that coordinates
            repositories, security helpers, RAG execution, and API workflows.
    Args:
        db (AsyncSession): Database session used to read or persist application records.
    Returns:
        ChatService: Domain or persistence object produced by the operation.
    Why Added:
        Provides a documented entry point for this module-level behavior and keeps
            callers
        from needing to know lower-level implementation details.
    """
    return ChatService(chat_repository=ChatRepository(db))
