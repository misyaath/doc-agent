import hashlib
import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import qdrant_client
from qdrant_client import models
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Document

from fastembed import LateInteractionTextEmbedding
from llama_index.core.node_parser import SentenceSplitter, SemanticDoubleMergingSplitterNodeParser
from core.settings import settings


class RagChunkingComponents:
    def __init__(
            self,
            model_name: str = "nomic-embed-text",
            base_url: str = "http://localhost:11434",
            initial_threshold: float = 0.15,
            merging_threshold: float = 0.15,
            appending_threshold: float = 0.20,
            max_chunk_size: int = 4000,
            fallback_chunk_size: int = 512,
            fallback_chunk_overlap: int = 50,
    ) -> None:
        self.embed_model = OllamaEmbedding(
            model_name=model_name,
            base_url=base_url,
        )
        self.splitter = SemanticDoubleMergingSplitterNodeParser(
            initial_threshold=initial_threshold,
            merging_threshold=merging_threshold,
            appending_threshold=appending_threshold,
            max_chunk_size=max_chunk_size,
            embed_model=self.embed_model,
        )
        self.fallback_splitter = SentenceSplitter(
            chunk_size=fallback_chunk_size,
            chunk_overlap=fallback_chunk_overlap,
        )
        self.max_safe_chars = max_chunk_size


# ============================================================
# Config
# ============================================================

@dataclass(frozen=True)
class RagIndexingConfig:
    doc_id: str | None = None
    doc_title: str | None = None
    source_file_path: str | Path | None = None

    chat_id: str | None = None
    file_id: str | None = None

    qdrant_url: str = settings.qdrant_url
    collection_name: str = settings.rag_collection_name

    embedding_model: str = settings.embedding_model
    ollama_url: str = settings.ollama_url


@dataclass(frozen=True)
class MarkdownRagChunk:
    id: str
    order: int
    heading: str | None
    heading_level: int | None
    text: str
    chunking_strategy: str = "markdown_heading_v1"

    page_start: int | None = None
    page_end: int | None = None
    image_paths: list[str] | None = None
    source_refs: list[str] | None = None


# ============================================================
# JSON Loader
# ============================================================

class RagChunkLoader:
    """
    Loads final chunks produced from processed markdown.

    Expected JSON format:

    [
      {
        "id": "main_section_0001",
        "order": 1,
        "heading": "1 Introduction",
        "heading_level": 2,
        "text": "...",
        "chunking_strategy": "markdown_heading_v1"
      }
    ]
    """

    def load(self, path: str | Path) -> list[dict[str, Any]]:
        file_path = Path(path)

        if not file_path.exists():
            raise FileNotFoundError(f"RAG chunks file not found: {file_path}")

        data = json.loads(file_path.read_text(encoding="utf-8"))

        if not isinstance(data, list):
            raise ValueError("RAG chunks JSON must be a list of chunk dictionaries.")

        return data


# ============================================================
# JSON Safety
# ============================================================

class JsonSafetyCleaner:
    def clean_value(self, value: Any) -> Any:
        try:
            json.dumps(value, ensure_ascii=False)
            return value
        except TypeError:
            return str(value)


# ============================================================
# Document Identity
# ============================================================

class DocumentIdentityBuilder:
    def clean_text(self, value: Any) -> str:
        if not value:
            return ""

        return " ".join(str(value).replace("\n", " ").split()).strip()

    def detect_title(
            self,
            chunks: list[dict[str, Any]],
            fallback_file_path: str | Path | None = None,
    ) -> str:
        sorted_chunks = sorted(chunks, key=lambda x: x.get("order", 0))

        for chunk in sorted_chunks:
            title = self.clean_text(chunk.get("doc_title"))
            if title:
                return title

        for chunk in sorted_chunks:
            heading = self.clean_text(chunk.get("heading"))
            if heading and heading != "Document Start":
                return heading

        if fallback_file_path:
            return Path(fallback_file_path).stem.replace("_", " ").replace("-", " ")

        return "Untitled Document"

    def create_doc_id(
            self,
            doc_title: str,
            source_file_path: str | Path | None = None,
    ) -> str:
        if source_file_path and Path(source_file_path).exists():
            file_hash = self._compute_file_sha256(source_file_path)
            seed = f"pdf-rag:{doc_title}:{file_hash}"
        else:
            seed = f"pdf-rag:{doc_title}"

        return str(uuid.uuid5(uuid.NAMESPACE_URL, seed))

    def resolve(
            self,
            config: RagIndexingConfig,
            chunks: list[dict[str, Any]],
    ) -> RagIndexingConfig:
        doc_title = config.doc_title or self.detect_title(
            chunks=chunks,
            fallback_file_path=config.source_file_path,
        )

        doc_id = config.doc_id or self.create_doc_id(
            doc_title=doc_title,
            source_file_path=config.source_file_path,
        )

        return RagIndexingConfig(
            doc_id=doc_id,
            doc_title=doc_title,
            source_file_path=config.source_file_path,
            chat_id=config.chat_id,
            file_id=config.file_id,
            qdrant_url=config.qdrant_url,
            collection_name=config.collection_name,
            embedding_model=config.embedding_model,
            ollama_url=config.ollama_url,
        )

    def _compute_file_sha256(self, file_path: str | Path) -> str:
        path = Path(file_path)
        digest = hashlib.sha256()

        with path.open("rb") as file:
            for block in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(block)

        return digest.hexdigest()


# ============================================================
# Embedding Model
# ============================================================

class EmbeddingModelFactory:
    def __init__(self, config: RagIndexingConfig) -> None:
        self._config = config

    def create(self) -> OllamaEmbedding:
        return OllamaEmbedding(
            model_name=self._config.embedding_model,
            base_url=self._config.ollama_url,
            embed_batch_size=1,
        )


# ============================================================
# Compact Payload Builder
# ============================================================

class CompactPayloadBuilder:
    """
    Keeps Qdrant metadata small.

    Required:
      - text
      - chat_id
      - file_id
      - doc_id
      - doc_title
      - chunk_id
      - chunk_index
      - heading

    Optional:
      - page_start
      - page_end
      - image_paths
      - source_refs
    """

    def __init__(
            self,
            config: RagIndexingConfig,
            cleaner: JsonSafetyCleaner | None = None,
    ) -> None:
        self._config = config
        self._cleaner = cleaner or JsonSafetyCleaner()

    def build(
            self,
            chunk: dict[str, Any],
            text: str,
            point_id: str,
            chunk_index: int,
    ) -> dict[str, Any]:

        payload = {
            "text": text,

            "node_id": point_id,

            "chat_id": self._config.chat_id,
            "file_id": self._config.file_id,

            "doc_id": self._config.doc_id,
            "doc_title": self._config.doc_title,

            "chunk_id": chunk.get("id") or f"{self._config.doc_id}_chunk_{chunk_index:05d}",
            "chunk_index": chunk_index,
            "order": chunk.get("order", chunk_index),

            "heading": chunk.get("heading"),
            "heading_level": chunk.get("heading_level"),

            "chunking_strategy": chunk.get("chunking_strategy", "markdown_heading_v1"),
            "source_type": "processed_markdown",
        }

        optional_payload = {
            "page_start": chunk.get("page_start"),
            "page_end": chunk.get("page_end"),
            "image_paths": chunk.get("image_paths"),
            "source_refs": chunk.get("source_refs"),
        }

        for key, value in optional_payload.items():
            if value is not None:
                payload[key] = value

        return {
            key: self._cleaner.clean_value(value)
            for key, value in payload.items()
            if value is not None
        }


# ============================================================
# Point ID Builder
# ============================================================

class QdrantPointIdBuilder:
    """
    Builds deterministic UUID point ids.

    Same document + same chunk id = same Qdrant point id.
    This lets you re-index the same file and overwrite same points.
    """

    def build(
            self,
            doc_id: str,
            chunk_id: int,
            chunk_index: int,
    ) -> str:
        seed = f"{doc_id}:{chunk_id}:{chunk_index}"
        return str(uuid.uuid5(uuid.NAMESPACE_URL, seed))


class ColbertModelFactory:
    def __init__(
            self,
            model_name: str = "colbert-ir/colbertv2.0",
    ) -> None:
        self._model_name = model_name

    def create(self) -> LateInteractionTextEmbedding:
        return LateInteractionTextEmbedding(self._model_name)


# ============================================================
# Qdrant Hybrid Saver
# ============================================================

class QdrantHybridIndexSaver:
    DENSE_VECTOR_NAME = "dense"
    SPARSE_VECTOR_NAME = "bm25"
    BM25_MODEL_NAME = "Qdrant/bm25"
    COLBERT_VECTOR_NAME = "colbert"
    COLBERT_VECTOR_SIZE = 128
    COLBERT_MODEL_NAME = "colbert-ir/colbertv2.0"

    def __init__(
            self,
            config: RagIndexingConfig,
            embedding_model_factory: EmbeddingModelFactory,
            payload_builder: CompactPayloadBuilder,
            point_id_builder: QdrantPointIdBuilder | None = None,
            colbert_model_factory: ColbertModelFactory | None = None,
            chunking_components: RagChunkingComponents | None = None,
    ) -> None:
        self._config = config
        self._embedding_model_factory = embedding_model_factory
        self._payload_builder = payload_builder
        self._point_id_builder = point_id_builder or QdrantPointIdBuilder()
        self._client = qdrant_client.QdrantClient(url=self._config.qdrant_url)
        self._colbert_model_factory = colbert_model_factory or ColbertModelFactory(
            self.COLBERT_MODEL_NAME
        )
        self._chunking_components = chunking_components or RagChunkingComponents(
            model_name=self._config.embedding_model,
            base_url=self._config.ollama_url,
        )

    def _upsert_points_in_batches(
            self,
            points: list[models.PointStruct],
            batch_size: int = 8,
    ) -> None:
        for start in range(0, len(points), batch_size):
            batch = points[start:start + batch_size]

            print(
                f"Upserting Qdrant batch: "
                f"{start // batch_size + 1}, "
                f"points={len(batch)}"
            )

            self._client.upsert(
                collection_name=self._config.collection_name,
                points=batch,
                wait=True,
            )

    def save(self, chunks: list[dict[str, Any]]) -> dict[str, Any]:
        valid_chunks = self._filter_valid_chunks(chunks)

        if not valid_chunks:
            return {
                "collection_name": self._config.collection_name,
                "points_count": 0,
            }

        dense_model = self._embedding_model_factory.create()
        colbert_model = self._colbert_model_factory.create()
        dense_size = self._detect_dense_vector_size(dense_model)

        self._ensure_hybrid_collection(dense_size=dense_size)

        avg_len = self._average_document_length(valid_chunks)

        points: list[models.PointStruct] = []

        pre_splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=100)

        for chunk_index, chunk in enumerate(valid_chunks):
            text = (chunk.get("text") or "").strip()

            initial_blocks = pre_splitter.get_nodes_from_documents([Document(text=text)])

            intermediate_documents = [
                Document(text=node.get_content()) for node in initial_blocks
            ]
            print(f"text: {len(text)}")

            nodes = self._chunking_components.splitter.get_nodes_from_documents(intermediate_documents)

            print(f"nodes: {len(nodes)}")

            for splitter_index, node in enumerate(nodes):
                chunk_text = node.get_content()

                print(f"chunk_text: {len(chunk_text)}")

                point_id = self._point_id_builder.build(
                    doc_id=self._config.doc_id or "unknown_doc",
                    chunk_id=splitter_index,
                    chunk_index=chunk_index,
                )

                dense_vector = dense_model.get_text_embedding(chunk_text)
                colbert_vector = next(colbert_model.passage_embed([text]))

                payload = self._payload_builder.build(
                    chunk=chunk,
                    text=chunk_text,
                    point_id=point_id,
                    chunk_index=chunk_index,
                )

                point = models.PointStruct(
                    id=point_id,
                    vector={
                        self.DENSE_VECTOR_NAME: dense_vector,
                        self.SPARSE_VECTOR_NAME: models.Document(
                            text=text,
                            model=self.BM25_MODEL_NAME,
                            options={
                                "avg_len": avg_len,
                            },
                        ),
                        self.COLBERT_VECTOR_NAME: colbert_vector.tolist(),
                    },
                    payload=payload,
                )

                points.append(point)

        self._upsert_points_in_batches(points=points, batch_size=5)

        return {
            "collection_name": self._config.collection_name,
            "points_count": len(points),
            "dense_vector_name": self.DENSE_VECTOR_NAME,
            "sparse_vector_name": self.SPARSE_VECTOR_NAME,
            "bm25_model": self.BM25_MODEL_NAME,
        }

    def delete_existing_file_points(self) -> None:
        """
        Optional but recommended before re-indexing a file.

        Deletes old points for same chat_id + file_id.
        """

        if not self._config.chat_id or not self._config.file_id:
            return

        if not self._client.collection_exists(self._config.collection_name):
            return

        self._client.delete(
            collection_name=self._config.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="chat_id",
                            match=models.MatchValue(value=self._config.chat_id),
                        ),
                        models.FieldCondition(
                            key="file_id",
                            match=models.MatchValue(value=self._config.file_id),
                        ),
                    ]
                )
            ),
            wait=True,
        )

    def _filter_valid_chunks(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        valid_chunks: list[dict[str, Any]] = []

        for chunk in chunks:
            text = (chunk.get("text") or "").strip()

            if not text:
                continue

            valid_chunks.append(chunk)

        return valid_chunks

    def _detect_dense_vector_size(self, dense_model: OllamaEmbedding) -> int:
        sample_vector = dense_model.get_text_embedding("dimension check")
        return len(sample_vector)

    def _ensure_hybrid_collection(self, dense_size: int) -> None:
        if self._client.collection_exists(self._config.collection_name):
            return

        self._client.create_collection(
            collection_name=self._config.collection_name,
            vectors_config={
                self.DENSE_VECTOR_NAME: models.VectorParams(
                    size=dense_size,
                    distance=models.Distance.COSINE,
                ),
                self.COLBERT_VECTOR_NAME: models.VectorParams(
                    size=self.COLBERT_VECTOR_SIZE,
                    distance=models.Distance.COSINE,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM
                    )
                )
            },
            sparse_vectors_config={
                self.SPARSE_VECTOR_NAME: models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                )
            },
        )

    def _average_document_length(self, chunks: list[dict[str, Any]]) -> float:
        lengths = [
            max(1, len((chunk.get("text") or "").split()))
            for chunk in chunks
        ]

        return sum(lengths) / max(1, len(lengths))


# ============================================================
# Main Ingestion Service
# ============================================================

class MarkdownRagQdrantIngestionService:
    """
    Main service for final Markdown RAG chunks.

    Input:
      chunks.json

    Output:
      Qdrant collection with:
        - dense named vector
        - bm25 named sparse vector
        - compact payload
    """

    def __init__(self, config: RagIndexingConfig) -> None:
        self._input_config = config
        self._loader = RagChunkLoader()
        self._identity_builder = DocumentIdentityBuilder()

    def ingest_from_file(
            self,
            chunks_path: str | Path,
            delete_existing_file_points: bool = True,
    ) -> dict[str, Any]:
        chunks = self._loader.load(chunks_path)

        runtime_config = self._identity_builder.resolve(
            config=self._input_config,
            chunks=chunks,
        )

        cleaner = JsonSafetyCleaner()

        embedding_model_factory = EmbeddingModelFactory(
            config=runtime_config,
        )

        payload_builder = CompactPayloadBuilder(
            config=runtime_config,
            cleaner=cleaner,
        )

        saver = QdrantHybridIndexSaver(
            config=runtime_config,
            embedding_model_factory=embedding_model_factory,
            payload_builder=payload_builder,
        )

        if delete_existing_file_points:
            saver.delete_existing_file_points()

        save_result = saver.save(chunks)

        return {
            "chat_id": runtime_config.chat_id,
            "file_id": runtime_config.file_id,
            "doc_id": runtime_config.doc_id,
            "doc_title": runtime_config.doc_title,
            "collection_name": runtime_config.collection_name,
            "chunks_count": len(chunks),
            "save_result": save_result,
        }
