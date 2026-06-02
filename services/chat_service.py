from __future__ import annotations

import uuid

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from repositories.chat_repository import ChatRepository
from schemas.chat import ChatCreateResponse


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

    async def create_chat(self, user_id: int) -> ChatCreateResponse:
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
        Returns:
            ChatCreateResponse: API response model returned to the client.
        Why Added:
            Centralizes this behavior inside ChatService so related code remains
                cohesive and testable.
        """
        chat_id = str(uuid.uuid4())
        await self._chat_repository.create(chat_id=chat_id, user_id=user_id)
        return ChatCreateResponse(user_id=user_id, chat_id=chat_id)


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
