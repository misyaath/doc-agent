from __future__ import annotations

import uuid

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from repositories.chat_repository import ChatRepository
from schemas.chat import ChatCreateResponse


class ChatService:
    def __init__(self, chat_repository: ChatRepository) -> None:
        self._chat_repository = chat_repository

    async def create_chat(self, user_id: int) -> ChatCreateResponse:
        chat_id = str(uuid.uuid4())
        await self._chat_repository.create(chat_id=chat_id, user_id=user_id)
        return ChatCreateResponse(user_id=user_id, chat_id=chat_id)


def get_chat_service(db: AsyncSession = Depends(get_db)) -> ChatService:
    return ChatService(chat_repository=ChatRepository(db))
