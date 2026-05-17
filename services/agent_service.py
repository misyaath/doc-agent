from __future__ import annotations

import json

from fastapi import Depends, HTTPException, status
from redis import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from core.settings import settings
from database import get_db
from repositories.chat_repository import ChatRepository
from repositories.file_repository import FileRepository
from schemas.agent import AgentChatRequest


class AgentService:
    def __init__(
            self,
            chat_repository: ChatRepository,
            file_repository: FileRepository,
            redis_client: Redis,
    ) -> None:
        self._chat_repository = chat_repository
        self._file_repository = file_repository
        self._redis = redis_client

    @staticmethod
    def _cache_key(chat_id: str) -> str:
        return f"agent:chat:{chat_id}:file_summaries"

    async def handle_chat(self, *, payload: AgentChatRequest, user_id: int) -> None:
        chat = await self._chat_repository.get_by_id_and_user_id(chat_id=payload.chat_id, user_id=user_id)
        if not chat:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized")

        cache_key = self._cache_key(payload.chat_id)
        cached_raw = self._redis.get(cache_key)
        if cached_raw:
            return

        files = await self._file_repository.get_title_summaries_by_chat_id(payload.chat_id)
        self._redis.set(cache_key, json.dumps(files, ensure_ascii=False))


def get_agent_service(db: AsyncSession = Depends(get_db)) -> AgentService:
    redis_client = Redis.from_url(settings.redis_url, decode_responses=True)
    return AgentService(
        chat_repository=ChatRepository(db),
        file_repository=FileRepository(db),
        redis_client=redis_client,
    )
