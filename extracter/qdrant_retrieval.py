from __future__ import annotations

from typing import Any

import qdrant_client
from qdrant_client import models

from .qdrant_indexer import EmbeddingModelFactory, QdrantIndexSaver, RagIndexingConfig


class QdrantHybridSearchService:
    """Search service for semantic, BM25, and hybrid retrieval over the same collection."""

    DENSE_VECTOR_NAME = QdrantIndexSaver.DENSE_VECTOR_NAME
    SPARSE_VECTOR_NAME = QdrantIndexSaver.SPARSE_VECTOR_NAME
    BM25_MODEL_NAME = QdrantIndexSaver.BM25_MODEL_NAME

    def __init__(self, config: RagIndexingConfig) -> None:
        self._config = config
        self._client = qdrant_client.QdrantClient(url=self._config.qdrant_url)
        self._embed_model = EmbeddingModelFactory(config).create()

    def semantic_search(
            self,
            query: str,
            limit: int = 10,
            doc_id: str | None = None,
            file_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Dense semantic search using Ollama embeddings."""
        dense_query = self._embed_model.get_query_embedding(query)

        response = self._client.query_points(
            collection_name=self._config.collection_name,
            query=dense_query,
            using=self.DENSE_VECTOR_NAME,
            query_filter=self._build_filter(doc_id=doc_id, file_id=file_id),
            limit=limit,
            with_payload=True,
        )
        return self._format_results(response.points, search_type="semantic")

    def bm25_search(
            self,
            query: str,
            limit: int = 10,
            doc_id: str | None = None,
            file_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Lexical BM25 search using Qdrant sparse vectors."""
        response = self._client.query_points(
            collection_name=self._config.collection_name,
            query=models.Document(
                text=query,
                model=self.BM25_MODEL_NAME,
            ),
            using=self.SPARSE_VECTOR_NAME,
            query_filter=self._build_filter(doc_id=doc_id, file_id=file_id),
            limit=limit,
            with_payload=True,
        )
        return self._format_results(response.points, search_type="bm25")

    def hybrid_search(
            self,
            query: str,
            limit: int = 10,
            prefetch_limit: int = 30,
            doc_id: str | None = None,
            file_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Hybrid search using RRF fusion of BM25 + dense semantic results."""
        dense_query = self._embed_model.get_query_embedding(query)
        query_filter = self._build_filter(doc_id=doc_id, file_id=file_id)

        response = self._client.query_points(
            collection_name=self._config.collection_name,
            prefetch=[
                models.Prefetch(
                    query=models.Document(
                        text=query,
                        model=self.BM25_MODEL_NAME,
                    ),
                    using=self.SPARSE_VECTOR_NAME,
                    query_filter=query_filter,
                    limit=prefetch_limit,
                ),
                models.Prefetch(
                    query=dense_query,
                    using=self.DENSE_VECTOR_NAME,
                    query_filter=query_filter,
                    limit=prefetch_limit,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )
        return self._format_results(response.points, search_type="hybrid_rrf")

    def _build_filter(
            self,
            doc_id: str | None = None,
            file_id: str | None = None,
    ) -> models.Filter | None:
        """Build Qdrant payload filter for per-document / per-file retrieval."""
        conditions: list[models.FieldCondition] = []

        if doc_id:
            conditions.append(
                models.FieldCondition(
                    key="doc_id",
                    match=models.MatchValue(value=doc_id),
                )
            )

        if file_id:
            conditions.append(
                models.FieldCondition(
                    key="file_id",
                    match=models.MatchValue(value=file_id),
                )
            )

        if not conditions:
            return None

        return models.Filter(must=conditions)

    def _format_results(
            self,
            points: list[models.ScoredPoint],
            search_type: str,
    ) -> list[dict[str, Any]]:
        """Normalize Qdrant scored points for API responses."""
        results: list[dict[str, Any]] = []

        for point in points:
            payload = point.payload or {}
            results.append(
                {
                    "id": point.id,
                    "score": point.score,
                    "search_type": search_type,
                    "text": payload.get("text"),
                    "doc_id": payload.get("doc_id"),
                    "doc_title": payload.get("doc_title"),
                    "file_id": payload.get("file_id"),
                    "page_no": payload.get("page_no"),
                    "type": payload.get("type"),
                    "chunk_id": payload.get("chunk_id"),
                    "source_ref": payload.get("source_ref"),
                    "heading_path": payload.get("heading_path"),
                    "payload": payload,
                }
            )

        return results
