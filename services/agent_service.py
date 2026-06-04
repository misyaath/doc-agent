from __future__ import annotations

import json
from collections.abc import AsyncGenerator, Iterator
from typing import Any, cast

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
from utils.stream_helper import node_label


class FileSummaryCacheService:
    """
    File Summary Cache Service.

    Purpose:
        Defines FileSummaryCacheService in the business-service layer that coordinates
            repositories, security helpers, RAG execution, and API workflows.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.
    """

    def __init__(self, file_repository: FileRepository, redis_client: Redis) -> None:
        """
        Initialize the object with its required dependencies.

        Purpose:
            Implements __init__ for the business-service layer that coordinates
                repositories, security helpers, RAG execution, and API workflows.
        Class:
            Belongs to FileSummaryCacheService; uses that class state and dependencies
                when available.
        Args:
            self (Self): Current instance that owns the operation state.
            file_repository (FileRepository): Input value for the file repository
                parameter.
            redis_client (Redis): Input value for the redis client parameter.
        Returns:
            None: Performs work through side effects and does not return a value.
        Why Added:
            Centralizes this behavior inside FileSummaryCacheService so related code
                remains cohesive and testable.
        """
        self._file_repository = file_repository
        self._redis = redis_client

    @staticmethod
    def cache_key(chat_id: str) -> str:
        """
        Cache key.

        Purpose:
            Implements cache_key for the business-service layer that coordinates
                repositories, security helpers, RAG execution, and API workflows.
        Class:
            Belongs to FileSummaryCacheService; uses that class state and dependencies
                when available.
        Args:
            chat_id (str): Chat/session identifier used to scope retrieval, tasks, or
                responses.
        Returns:
            str: Result produced by the operation.
        Why Added:
            Centralizes this behavior inside FileSummaryCacheService so related code
                remains cohesive and testable.
        """
        return f"agent:chat:{chat_id}:file_summaries"

    async def get_or_load(self, chat_id: str) -> list[dict[str, Any]]:
        """
        Get or load.

        Purpose:
            Implements get_or_load for the business-service layer that coordinates
                repositories, security helpers, RAG execution, and API workflows.
        Class:
            Belongs to FileSummaryCacheService; uses that class state and dependencies
                when available.
        Args:
            self (Self): Current instance that owns the operation state.
            chat_id (str): Chat/session identifier used to scope retrieval, tasks, or
                responses.
        Returns:
            list[dict[str, Any]]: Structured data produced by the operation.
        Why Added:
            Centralizes this behavior inside FileSummaryCacheService so related code
                remains cohesive and testable.
        """
        cache_key = self.cache_key(chat_id)
        cached_raw = self._redis.get(cache_key)

        if cached_raw:
            try:
                data = json.loads(cast(str | bytes | bytearray, cached_raw))
                if isinstance(data, list):
                    return data
            except json.JSONDecodeError:
                pass

        files = await self._file_repository.get_title_summaries_by_chat_id(chat_id)
        self._redis.set(cache_key, json.dumps(files, ensure_ascii=False))
        return files


class RagAgentRunner:
    """
    Rag Agent Runner.

    Purpose:
        Defines RagAgentRunner in the business-service layer that coordinates
            repositories, security helpers, RAG execution, and API workflows.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.
    """

    def __init__(
        self,
        qdrant_url: str,
        collection_name: str,
        embedding_model: str,
        ollama_url: str,
        text_model: str,
    ) -> None:
        """
        Initialize the object with its required dependencies.

        Purpose:
            Implements __init__ for the business-service layer that coordinates
                repositories, security helpers, RAG execution, and API workflows.
        Class:
            Belongs to RagAgentRunner; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
            qdrant_url (str): Input value for the qdrant url parameter.
            collection_name (str): Qdrant collection name targeted by the vector
                operation.
            embedding_model (str): Input value for the embedding model parameter.
            ollama_url (str): Input value for the ollama url parameter.
            text_model (str): Input value for the text model parameter.
        Returns:
            None: Performs work through side effects and does not return a value.
        Why Added:
            Centralizes this behavior inside RagAgentRunner so related code remains
                cohesive and testable.
        """
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
    ) -> Any:
        """
        Build graph.

        Purpose:
            Implements _build_graph for the business-service layer that coordinates
                repositories, security helpers, RAG execution, and API workflows.
        Class:
            Belongs to RagAgentRunner; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
            document_summary (list[dict[str, Any]]): Input value for the document
                summary parameter.
            document_title (str): Input value for the document title parameter.
        Returns:
            Any: Result produced by the operation.
        Why Added:
            Centralizes this behavior inside RagAgentRunner so related code remains
                cohesive and testable.
        """
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
        """
        Run.

        Purpose:
            Implements run for the business-service layer that coordinates repositories,
                security helpers, RAG execution, and API workflows.
        Class:
            Belongs to RagAgentRunner; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
            prompt (str): Prompt text sent to the agent or language model.
            chat_id (str): Chat/session identifier used to scope retrieval, tasks, or
                responses.
            document_summary (list[dict[str, Any]]): Input value for the document
                summary parameter.
            document_title (str): Input value for the document title parameter.
        Returns:
            dict[str, Any]: Structured data produced by the operation.
        Why Added:
            Centralizes this behavior inside RagAgentRunner so related code remains
                cohesive and testable.
        """
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
            },
            config={"configurable": {"thread_id": chat_id}},
        )

    def stream(
        self,
        *,
        prompt: str,
        chat_id: str,
        document_summary: list[dict[str, Any]],
        document_title: str,
    ) -> Iterator[dict[str, Any]]:
        """
        Stream.

        Purpose:
            Implements stream for the business-service layer that coordinates
                repositories, security helpers, RAG execution, and API workflows.
        Class:
            Belongs to RagAgentRunner; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
            prompt (str): Prompt text sent to the agent or language model.
            chat_id (str): Chat/session identifier used to scope retrieval, tasks, or
                responses.
            document_summary (list[dict[str, Any]]): Input value for the document
                summary parameter.
            document_title (str): Input value for the document title parameter.
        Returns:
            Iterator[str]: Streaming response or iterator that yields incremental
                output.
        Why Added:
            Centralizes this behavior inside RagAgentRunner so related code remains
                cohesive and testable.
        """
        rag_graph = self._build_graph(
            document_summary=document_summary,
            document_title=document_title,
        )

        inputs = {
            "question": prompt,
            "chat_id": chat_id,
        }

        for chunk, metadata in rag_graph.stream(
            inputs, stream_mode="messages", config={"configurable": {"thread_id": chat_id}}
        ):
            token = getattr(chunk, "content", None)

            yield {
                "type": "step",
                "text": node_label(metadata.get("langgraph_node")),
            }

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
    """
    Agent Service.

    Purpose:
        Defines AgentService in the business-service layer that coordinates
            repositories, security helpers, RAG execution, and API workflows.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.
    """

    def __init__(
        self,
        chat_repository: ChatRepository,
        file_summary_cache: FileSummaryCacheService,
        runner: RagAgentRunner,
    ) -> None:
        """
        Initialize the object with its required dependencies.

        Purpose:
            Implements __init__ for the business-service layer that coordinates
                repositories, security helpers, RAG execution, and API workflows.
        Class:
            Belongs to AgentService; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
            chat_repository (ChatRepository): Input value for the chat repository
                parameter.
            file_summary_cache (FileSummaryCacheService): Input value for the file
                summary cache parameter.
            runner (RagAgentRunner): Input value for the runner parameter.
        Returns:
            None: Performs work through side effects and does not return a value.
        Why Added:
            Centralizes this behavior inside AgentService so related code remains
                cohesive and testable.
        """
        self._chat_repository = chat_repository
        self._file_summary_cache = file_summary_cache
        self._runner = runner

    async def _prepare_chat(
        self,
        *,
        payload: AgentChatRequest,
        user_id: int,
    ) -> tuple[str, Any]:
        """
        Prepare chat.

        Purpose:
            Implements _prepare_chat for the business-service layer that coordinates
                repositories, security helpers, RAG execution, and API workflows.
        Class:
            Belongs to AgentService; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
            payload (AgentChatRequest): Validated request payload supplied by the API
                caller.
            user_id (int): Authenticated user identifier used to scope the operation.
        Returns:
            tuple[str, Any]: Structured data produced by the operation.
        Why Added:
            Centralizes this behavior inside AgentService so related code remains
                cohesive and testable.
        """
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
        """
        Handle chat.

        Purpose:
            Implements handle_chat for the business-service layer that coordinates
                repositories, security helpers, RAG execution, and API workflows.
        Class:
            Belongs to AgentService; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
            payload (AgentChatRequest): Validated request payload supplied by the API
                caller.
            user_id (int): Authenticated user identifier used to scope the operation.
        Returns:
            AgentChatResponse: API response model returned to the client.
        Why Added:
            Centralizes this behavior inside AgentService so related code remains
                cohesive and testable.
        """
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
        """
        Stream chat.

        Purpose:
            Implements stream_chat for the business-service layer that coordinates
                repositories, security helpers, RAG execution, and API workflows.
        Class:
            Belongs to AgentService; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
            payload (AgentChatRequest): Validated request payload supplied by the API
                caller.
            user_id (int): Authenticated user identifier used to scope the operation.
        Returns:
            AsyncGenerator[str, Any]: Streaming response or iterator that yields
                incremental output.
        Why Added:
            Centralizes this behavior inside AgentService so related code remains
                cohesive and testable.
        """
        document_summary: list[dict[str, Any]] = []
        document_title = ""

        yield sse_event("step", {"message": "Chat started"})

        try:
            document_title, document_summary = await self._prepare_chat(
                payload=payload,
                user_id=user_id,
            )
        except Exception as e:
            yield sse_event("error", {"message": f"Prepare chat failed {str(e)}"})

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
                elif item["type"] == "step":
                    yield sse_event(
                        "step",
                        {
                            "message": item["text"],
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
    """
    Get agent service.

    Purpose:
        Implements get_agent_service for the business-service layer that coordinates
            repositories, security helpers, RAG execution, and API workflows.
    Args:
        db (AsyncSession): Database session used to read or persist application records.
    Returns:
        AgentService: Result produced by the operation.
    Why Added:
        Provides a documented entry point for this module-level behavior and keeps
            callers
        from needing to know lower-level implementation details.
    """
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
