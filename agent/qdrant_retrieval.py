import re
from typing import Any

from fastembed import LateInteractionTextEmbedding, SparseTextEmbedding
from langchain_ollama import OllamaEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import FieldCondition, Filter, MatchValue

METADATA_FIELDS = (
    "doc_id",
    "doc_title",
    "file_id",
    "chat_id",
    "page_no",
    "order",
    "chunk_id",
    "chunk_index",
    "semantic_node_index",
    "heading_path",
    "source_ref",
)


class ColbertQueryEmbedder:
    def __init__(self, model_name: str = "colbert-ir/colbertv2.0") -> None:
        self.model = LateInteractionTextEmbedding(model_name)

    def embed_query(self, query: str) -> list[list[float]]:
        clean_query = (query or "").strip()

        if not clean_query:
            return []

        if hasattr(self.model, "query_embed"):
            vectors = list(self.model.query_embed([clean_query]))
        else:
            vectors = list(self.model.embed([clean_query]))

        return vectors[0].tolist()


class ChunkPayloadMapper:
    def __init__(self, metadata_fields: tuple[str, ...] = METADATA_FIELDS) -> None:
        self.metadata_fields = metadata_fields

    def to_chunk(
            self,
            point: models.ScoredPoint,
            score_key: str,
    ) -> dict[str, Any]:
        payload = point.payload or {}
        score = float(point.score or 0.0)

        chunk = {"id": str(point.id), "text": payload.get("text", ""), "score": score, "metadata": {
            key: payload.get(key)
            for key in self.metadata_fields
        }, score_key: score}
        return chunk


class QueryNormalizer:
    def deduplicate(self, queries: list[str]) -> list[str]:
        cleaned: list[str] = []

        for query in queries:
            clean_query = (query or "").strip()
            if clean_query:
                cleaned.append(clean_query)

        return list(dict.fromkeys(cleaned))


class ChunkAggregator:
    def merge(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}

        for chunk in chunks:
            key = chunk.get("id")
            score = float(chunk.get("score", 0.0))

            if key not in merged:
                item = chunk.copy()
                item["final_score"] = score
                merged[key] = item

        return list(merged.values())

    def sort(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(
            chunks,
            key=lambda chunk: float(
                chunk.get("final_score", chunk.get("score", 0.0))
            ),
            reverse=True,
        )


class ChunkQualityEvaluator:
    @staticmethod
    def is_good_chunk(text: str) -> bool:
        clean_text = (text or "").strip()

        if not clean_text:
            return False

        if len(clean_text) < 10:
            return False

        if re.fullmatch(r"[\[\]\d\s\.\-–—pP]+", clean_text):
            return False

        alpha_count = sum(ch.isalpha() for ch in clean_text)

        return alpha_count >= 20

    def filter_bad_chunks(
            self,
            chunks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return [
            chunk
            for chunk in chunks
            if self.is_good_chunk(chunk.get("text", ""))
        ]


class QdrantFilterFactory:
    @staticmethod
    def for_chat(chat_id: str) -> Filter:
        return Filter(
            must=[
                FieldCondition(
                    key="chat_id",
                    match=MatchValue(value=chat_id),
                )
            ]
        )


class HybridQdrantQueryService:
    def __init__(
            self,
            client: QdrantClient,
            collection_name: str,
            dense_embeddings: OllamaEmbeddings,
            sparse_embeddings: SparseTextEmbedding,
            chunk_mapper: ChunkPayloadMapper,
    ) -> None:
        self.client = client
        self.collection_name = collection_name
        self.dense_embeddings = dense_embeddings
        self.sparse_embeddings = sparse_embeddings
        self.chunk_mapper = chunk_mapper

    def retrieve(self, query: str, chat_id: str, limit: int) -> list[dict[str, Any]]:
        dense_query_vector = self.dense_embeddings.embed_query(query)
        sparse_embedding = next(self.sparse_embeddings.embed([query]))
        sparse_query_vector = models.SparseVector(
            indices=sparse_embedding.indices.tolist(),
            values=sparse_embedding.values.tolist(),
        )

        qdrant_filter = QdrantFilterFactory.for_chat(chat_id)

        result = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(
                    query=dense_query_vector,
                    using="dense",
                    filter=qdrant_filter,
                    limit=limit * 4,
                ),
                models.Prefetch(
                    query=sparse_query_vector,
                    using="bm25",
                    filter=qdrant_filter,
                    limit=limit * 4,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )

        return [
            self.chunk_mapper.to_chunk(point=point, score_key="final_score")
            for point in result.points
        ]


class QdrantColbertReranker:
    def __init__(
            self,
            client: QdrantClient,
            collection_name: str,
            chunk_mapper: ChunkPayloadMapper,
            colbert_vector_name: str = "colbert",
    ) -> None:
        self.client = client
        self.collection_name = collection_name
        self.chunk_mapper = chunk_mapper
        self.colbert_vector_name = colbert_vector_name

    def rerank(
        self,
        query_colbert_vector: list[list[float]],
        candidate_point_ids: list[str],
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        if not query_colbert_vector or not candidate_point_ids:
            return []

        candidate_filter = models.Filter(
            must=[
                models.HasIdCondition(
                    has_id=candidate_point_ids,
                )
            ]
        )

        result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_colbert_vector,
            using=self.colbert_vector_name,
            query_filter=candidate_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

        return [
            self.chunk_mapper.to_chunk(point=point, score_key="rerank_score")
            for point in result.points
        ]

    @classmethod
    def rerank_with_qdrant_colbert(
        cls,
        client: QdrantClient,
        collection_name: str,
        query_colbert_vector: list[list[float]],
        candidate_point_ids: list[str],
        colbert_vector_name: str = "colbert",
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        reranker = cls(
            client=client,
            collection_name=collection_name,
            chunk_mapper=ChunkPayloadMapper(),
            colbert_vector_name=colbert_vector_name,
        )
        return reranker.rerank(
            query_colbert_vector=query_colbert_vector,
            candidate_point_ids=candidate_point_ids,
            limit=limit,
        )

class QdrantRagRetriever:
    def __init__(
            self,
            qdrant_url: str,
            collection_name: str,
            embedding_model: str = "qwen3-embedding",
            sparse_model_name: str = "Qdrant/bm25",
            ollama_base_url: str = "http://localhost:11434",
    ) -> None:
        self.client = QdrantClient(url=qdrant_url)
        self.collection_name = collection_name

        dense_embeddings = OllamaEmbeddings(
            model=embedding_model,
            base_url=ollama_base_url,
        )
        sparse_embeddings = SparseTextEmbedding(model_name=sparse_model_name)

        self.chunk_mapper = ChunkPayloadMapper()
        self.query_normalizer = QueryNormalizer()
        self.chunk_aggregator = ChunkAggregator()
        self.quality_evaluator = ChunkQualityEvaluator()

        self.hybrid_query_service = HybridQdrantQueryService(
            client=self.client,
            collection_name=self.collection_name,
            dense_embeddings=dense_embeddings,
            sparse_embeddings=sparse_embeddings,
            chunk_mapper=self.chunk_mapper,
        )
        self.colbert_query_embedder = ColbertQueryEmbedder(
            model_name="colbert-ir/colbertv2.0"
        )
        self.colbert_reranker = QdrantColbertReranker(
            client=self.client,
            collection_name=self.collection_name,
            chunk_mapper=self.chunk_mapper,
            colbert_vector_name="colbert",
        )

    def retrieve(
            self,
            question: str,
            chat_id: str,
            file_id: str | None = None,
            limit: int = 10,
    ) -> list[dict[str, Any]]:
        return self._retrieve_one_query(
            query=question,
            chat_id=chat_id,
            limit=limit,
        )

    def retrieve_many(
            self,
            queries: list[str],
            chat_id: str,
            limit_per_query: int = 8,
            final_limit: int = 10,
    ) -> list[dict[str, Any]]:
        clean_queries = self.query_normalizer.deduplicate(queries)

        all_chunks: list[dict[str, Any]] = []

        for query in clean_queries:
            chunks = self._retrieve_one_query(
                query=query,
                chat_id=chat_id,
                limit=limit_per_query,
            )
            all_chunks.extend(chunks)

        merged = self.chunk_aggregator.merge(all_chunks)
        sorted_chunks = self.chunk_aggregator.sort(merged)

        candidate_point_ids = [
            chunk["id"]
            for chunk in sorted_chunks
            if chunk.get("id")
        ]

        query_colbert_vector = self.colbert_query_embedder.embed_query(
            "\n".join(clean_queries)
        )
        colbert_results = self.colbert_reranker.rerank(
            query_colbert_vector=query_colbert_vector,
            candidate_point_ids=candidate_point_ids,
            limit=final_limit,
        )

        if colbert_results:
            return colbert_results

        return sorted_chunks[:final_limit]

    def _retrieve_one_query(
            self,
            query: str,
            chat_id: str,
            limit: int = 10,
    ) -> list[dict[str, Any]]:
        return self.hybrid_query_service.retrieve(
            query=query,
            chat_id=chat_id,
            limit=limit,
        )

    def _is_good_chunk(self, text: str) -> bool:
        return self.quality_evaluator.is_good_chunk(text)

    def _filter_bad_chunks(
            self,
            chunks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return self.quality_evaluator.filter_bad_chunks(chunks)
