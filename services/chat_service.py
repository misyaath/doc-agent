from __future__ import annotations

import uuid

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from domain.file_process import FileStage, FileStageStatus
from repositories.chat_repository import ChatRepository
from schemas.chat import (
    ChatCreateResponse,
    ChatFileProcessStageResponse,
    ChatFileResponse,
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

    def __init__(self, chat_repository: ChatRepository) -> None:
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
        Returns:
            None: Performs work through side effects and does not return a value.
        Why Added:
            Centralizes this behavior inside ChatService so related code remains
                cohesive and testable.
        """
        self._chat_repository = chat_repository

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
                    files=[
                        ChatFileResponse(
                            file_id=file.file_id,
                            file_name=file.file_name,
                            title=file.title,
                            summary=file.summary,
                            process_status=_file_process_status(
                                {stage.stage: stage.status for stage in file.process_stages}
                            ),
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
                    ],
                )
                for chat in chats
            ],
        )


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
