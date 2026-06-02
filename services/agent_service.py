from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any, AsyncGenerator

from anyio import to_thread
from fastapi import Depends, HTTPException, status
from redis import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from agent.qdrant_retrieval import QdrantRagRetriever
from agent.rag import RagGraphFactory
from core.settings import settings
from database import get_db
from repositories.chat_repository import ChatRepository
from repositories.file_repository import FileRepository
from schemas.agent import AgentChatRequest, AgentChatResponse
from utils.sse import sse_event


class FileSummaryCacheService:
    def __init__(self, file_repository: FileRepository, redis_client: Redis) -> None:
        self._file_repository = file_repository
        self._redis = redis_client

    @staticmethod
    def cache_key(chat_id: str) -> str:
        return f"agent:chat:{chat_id}:file_summaries"

    async def get_or_load(self, chat_id: str) -> list[dict[str, Any]]:
        cache_key = self.cache_key(chat_id)
        cached_raw = self._redis.get(cache_key)

        if cached_raw:
            try:
                data = json.loads(cached_raw)
                if isinstance(data, list):
                    return data
            except json.JSONDecodeError:
                pass

        files = await self._file_repository.get_title_summaries_by_chat_id(chat_id)
        self._redis.set(cache_key, json.dumps(files, ensure_ascii=False))
        return files


class RagAgentRunner:
    def __init__(
            self,
            qdrant_url: str,
            collection_name: str,
            embedding_model: str,
            ollama_url: str,
            text_model: str,
    ) -> None:
        self._qdrant_url = qdrant_url
        self._collection_name = collection_name
        self._embedding_model = embedding_model
        self._ollama_url = ollama_url
        self._text_model = text_model

    def _build_graph(
            self,
            *,
            document_summary: list[dict[str, Any]],
            document_title: str,
    ):
        retriever = QdrantRagRetriever(
            qdrant_url=self._qdrant_url,
            collection_name=self._collection_name,
            embedding_model=self._embedding_model,
            ollama_base_url=self._ollama_url,
        )

        return RagGraphFactory(
            document_summary=document_summary,
            document_title=document_title,
            retriever=retriever,
            llm_model=self._text_model,
            ollama_base_url=self._ollama_url,
        ).build()

    def run(
            self,
            *,
            prompt: str,
            chat_id: str,
            document_summary: list[dict[str, Any]],
            document_title: str,
    ) -> dict[str, Any]:
        retriever = QdrantRagRetriever(
            qdrant_url=self._qdrant_url,
            collection_name=self._collection_name,
            embedding_model=self._embedding_model,
            ollama_base_url=self._ollama_url,
        )
        rag_graph = RagGraphFactory(
            document_summary=document_summary,
            document_title=document_title,
            retriever=retriever,
            llm_model=self._text_model,
            ollama_base_url=self._ollama_url,
        ).build()
        return rag_graph.invoke(
            {
                "question": prompt,
                "chat_id": chat_id,
            }, config={
                "configurable": {
                    "thread_id": chat_id
                }
            }
        )

    def stream(
            self,
            *,
            prompt: str,
            chat_id: str,
            document_summary: list[dict[str, Any]],
            document_title: str,
    ):
        """
        Synchronous generator.

        Starlette/FastAPI can stream a normal sync generator using StreamingResponse.
        """
        rag_graph = self._build_graph(
            document_summary=document_summary,
            document_title=document_title,
        )

        inputs = {
            "question": prompt,
            "chat_id": chat_id,
        }

        for chunk, metadata in rag_graph.stream(inputs, stream_mode="messages", config={
            "configurable": {
                "thread_id": chat_id
            }
        }):
            token = getattr(chunk, "content", None)

            if not token:
                continue

            yield {
                "type": "token",
                "text": token,
                "metadata": {
                    "node": metadata.get("langgraph_node") if isinstance(metadata, dict) else None,
                },
            }

        yield {
            "type": "done",
        }


class AgentService:
    def __init__(
            self,
            chat_repository: ChatRepository,
            file_summary_cache: FileSummaryCacheService,
            runner: RagAgentRunner,
    ) -> None:
        self._chat_repository = chat_repository
        self._file_summary_cache = file_summary_cache
        self._runner = runner

    async def _prepare_chat(
            self,
            *,
            payload: AgentChatRequest,
            user_id: int,
    ) -> tuple[str, Any]:
        chat = await self._chat_repository.get_by_id_and_user_id(
            chat_id=payload.chat_id,
            user_id=user_id,
        )

        if not chat:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized",
            )

        file_summaries = await self._file_summary_cache.get_or_load(payload.chat_id)

        if not file_summaries:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No indexed files found for this chat",
            )

        document_title = file_summaries[0].get("title", "")
        document_summary = file_summaries[0].get("summary", "")

        return document_title, document_summary

    async def handle_chat(self, *, payload: AgentChatRequest, user_id: int) -> AgentChatResponse:
        document_title, document_summary = await self._prepare_chat(
            payload=payload,
            user_id=user_id,
        )

        rag_result = await to_thread.run_sync(
            lambda: self._runner.run(
                prompt=payload.prompt,
                chat_id=payload.chat_id,
                document_summary=document_summary,
                document_title=document_title,
            )
        )

        return AgentChatResponse(
            chat_id=payload.chat_id,
            prompt=payload.prompt,
            answer=str(rag_result.get("answer", "")),
            context=rag_result.get("context"),
        )

    async def stream_chat(
            self,
            *,
            payload: AgentChatRequest,
            user_id: int,
    ) -> AsyncGenerator[str, Any]:

        document_summary = {}
        document_title = ""

        yield sse_event("debug", {"message": "stream_chat started"})

        try:
            document_title, document_summary = await self._prepare_chat(
                payload=payload,
                user_id=user_id,
            )
        except Exception as e:
            yield sse_event("debug", {"message": f"Prepare chat failed {str(e)}"})

        yield sse_event(
            "start",
            {
                "chat_id": payload.chat_id,
            },
        )

        try:
            yield sse_event("debug", {"message": "started chat with ollama"})
            for item in self._runner.stream(
                    prompt=payload.prompt,
                    chat_id=payload.chat_id,
                    document_summary=document_summary,
                    document_title=document_title,
            ):
                if item["type"] == "token":
                    yield sse_event(
                        "token",
                        {
                            "text": item["text"],
                        },
                    )

                elif item["type"] == "done":
                    yield sse_event(
                        "done",
                        {
                            "chat_id": payload.chat_id,
                        },
                    )

        except Exception as exc:
            yield sse_event(
                "error",
                {
                    "message": str(exc),
                },
            )


def get_agent_service(db: AsyncSession = Depends(get_db)) -> AgentService:
    redis_client = Redis.from_url(settings.redis_url, decode_responses=True)
    file_repository = FileRepository(db)
    return AgentService(
        chat_repository=ChatRepository(db),
        file_summary_cache=FileSummaryCacheService(
            file_repository=file_repository,
            redis_client=redis_client,
        ),
        runner=RagAgentRunner(
            qdrant_url=settings.qdrant_url,
            collection_name=settings.rag_collection_name,
            embedding_model=settings.embedding_model,
            ollama_url=settings.ollama_url,
            text_model=settings.text_model,
        ),
    )
