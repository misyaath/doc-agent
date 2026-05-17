from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models.chat import Chat


class ChatRepository:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def get_by_id_and_user_id(self, chat_id: str, user_id: int) -> Chat | None:
        return await self._db.scalar(
            select(Chat).where(Chat.id == chat_id, Chat.user_id == user_id)
        )

    async def create(self, *, chat_id: str, user_id: int) -> Chat:
        chat = Chat(id=chat_id, user_id=user_id)
        self._db.add(chat)
        await self._db.commit()
        await self._db.refresh(chat)
        return chat
