import hashlib
import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import qdrant_client
from fastembed import LateInteractionTextEmbedding
from llama_index.core import Document
from llama_index.core.node_parser import SemanticDoubleMergingSplitterNodeParser, SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from qdrant_client import models

from core.settings import settings


class RagChunkingComponents:
    """
    Rag Chunking Components.

    Purpose:
        Defines RagChunkingComponents in the document extraction pipeline that
            normalizes PDFs, enriches visual content, builds RAG units, and indexes
            data.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.
    """

    def __init__(
        self,
        initial_threshold: float = 0.15,
        merging_threshold: float = 0.15,
        appending_threshold: float = 0.20,
        max_chunk_size: int = 4000,
        fallback_chunk_size: int = 512,
        fallback_chunk_overlap: int = 128,
    ) -> None:
        """
        Initialize the object with its required dependencies.

        Purpose:
            Implements __init__ for the document extraction pipeline that normalizes
                PDFs, enriches visual content, builds RAG units, and indexes data.
        Class:
            Belongs to RagChunkingComponents; uses that class state and dependencies
                when available.
        Args:
            self (Self): Current instance that owns the operation state.
            initial_threshold (float): Input value for the initial threshold parameter.
            merging_threshold (float): Input value for the merging threshold parameter.
            appending_threshold (float): Input value for the appending threshold
                parameter.
            max_chunk_size (int): Input value for the max chunk size parameter.
            fallback_chunk_size (int): Input value for the fallback chunk size
                parameter.
            fallback_chunk_overlap (int): Input value for the fallback chunk overlap
                parameter.
        Returns:
            None: Performs work through side effects and does not return a value.
        Why Added:
            Centralizes this behavior inside RagChunkingComponents so related code
                remains cohesive and testable.
        """
        self.fallback_splitter = SentenceSplitter(
            chunk_size=fallback_chunk_size,
            chunk_overlap=fallback_chunk_overlap,
        )

        chunking_embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.splitter = SemanticDoubleMergingSplitterNodeParser(
            initial_threshold=initial_threshold,
            merging_threshold=merging_threshold,
            appending_threshold=appending_threshold,
            max_chunk_size=max_chunk_size,
            embed_model=chunking_embed_model,
        )

        self.max_safe_chars = max_chunk_size


# ============================================================
# Config
# ============================================================


@dataclass(frozen=True)
class RagIndexingConfig:
    """
    Rag Indexing Config.

    Purpose:
        Defines RagIndexingConfig in the document extraction pipeline that normalizes
            PDFs, enriches visual content, builds RAG units, and indexes data.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.

    Attributes:
        doc_id (str | None): Declared data field for this class.
        doc_title (str | None): Declared data field for this class.
        source_file_path (str | Path | None): Declared data field for this class.
        chat_id (str | None): Declared data field for this class.
        file_id (str | None): Declared data field for this class.
        qdrant_url (str): Declared data field for this class.
        collection_name (str): Declared data field for this class.
        embedding_model (str): Declared data field for this class.
        ollama_url (str): Declared data field for this class.
    """

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
    """
    Markdown Rag Chunk.

    Purpose:
        Defines MarkdownRagChunk in the document extraction pipeline that normalizes
            PDFs, enriches visual content, builds RAG units, and indexes data.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.

    Attributes:
        id (str): Declared data field for this class.
        order (int): Declared data field for this class.
        heading (str | None): Declared data field for this class.
        heading_level (int | None): Declared data field for this class.
        text (str): Declared data field for this class.
        chunking_strategy (str): Declared data field for this class.
        page_start (int | None): Declared data field for this class.
        page_end (int | None): Declared data field for this class.
        image_paths (list[str] | None): Declared data field for this class.
        source_refs (list[str] | None): Declared data field for this class.
    """

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
    Rag Chunk Loader.

    Purpose:
        Defines RagChunkLoader in the document extraction pipeline that normalizes PDFs,
            enriches visual content, builds RAG units, and indexes data.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.
    """

    def load(self, path: str | Path) -> list[dict[str, Any]]:
        """
        Load.

        Purpose:
            Implements load for the document extraction pipeline that normalizes PDFs,
                enriches visual content, builds RAG units, and indexes data.
        Class:
            Belongs to RagChunkLoader; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
            path (str | Path): Filesystem path used as input or output for the
                operation.
        Returns:
            list[dict[str, Any]]: Structured data produced by the operation.
        Why Added:
            Centralizes this behavior inside RagChunkLoader so related code remains
                cohesive and testable.
        """
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
    """
    Json Safety Cleaner.

    Purpose:
        Defines JsonSafetyCleaner in the document extraction pipeline that normalizes
            PDFs, enriches visual content, builds RAG units, and indexes data.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.
    """

    def clean_value(self, value: Any) -> Any:
        """
        Clean value.

        Purpose:
            Implements clean_value for the document extraction pipeline that normalizes
                PDFs, enriches visual content, builds RAG units, and indexes data.
        Class:
            Belongs to JsonSafetyCleaner; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
            value (Any): Raw value being validated, normalized, or transformed.
        Returns:
            Any: Result produced by the operation.
        Why Added:
            Centralizes this behavior inside JsonSafetyCleaner so related code remains
                cohesive and testable.
        """
        try:
            json.dumps(value, ensure_ascii=False)
            return value
        except TypeError:
            return str(value)


# ============================================================
# Document Identity
# ============================================================


class DocumentIdentityBuilder:
    """
    Document Identity Builder.

    Purpose:
        Defines DocumentIdentityBuilder in the document extraction pipeline that
            normalizes PDFs, enriches visual content, builds RAG units, and indexes
            data.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.
    """

    def clean_text(self, value: Any) -> str:
        """
        Clean text.

        Purpose:
            Implements clean_text for the document extraction pipeline that normalizes
                PDFs, enriches visual content, builds RAG units, and indexes data.
        Class:
            Belongs to DocumentIdentityBuilder; uses that class state and dependencies
                when available.
        Args:
            self (Self): Current instance that owns the operation state.
            value (Any): Raw value being validated, normalized, or transformed.
        Returns:
            str: Result produced by the operation.
        Why Added:
            Centralizes this behavior inside DocumentIdentityBuilder so related code
                remains cohesive and testable.
        """
        if not value:
            return ""

        return " ".join(str(value).replace("\n", " ").split()).strip()

    def detect_title(
        self,
        chunks: list[dict[str, Any]],
        fallback_file_path: str | Path | None = None,
    ) -> str:
        """
        Detect title.

        Purpose:
            Implements detect_title for the document extraction pipeline that normalizes
                PDFs, enriches visual content, builds RAG units, and indexes data.
        Class:
            Belongs to DocumentIdentityBuilder; uses that class state and dependencies
                when available.
        Args:
            self (Self): Current instance that owns the operation state.
            chunks (list[dict[str, Any]]): Input value for the chunks parameter.
            fallback_file_path (str | Path | None): Input value for the fallback file
                path parameter.
        Returns:
            str: Result produced by the operation.
        Why Added:
            Centralizes this behavior inside DocumentIdentityBuilder so related code
                remains cohesive and testable.
        """
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
        """
        Create doc id.

        Purpose:
            Implements create_doc_id for the document extraction pipeline that
                normalizes PDFs, enriches visual content, builds RAG units, and indexes
                data.
        Class:
            Belongs to DocumentIdentityBuilder; uses that class state and dependencies
                when available.
        Args:
            self (Self): Current instance that owns the operation state.
            doc_title (str): Input value for the doc title parameter.
            source_file_path (str | Path | None): Input value for the source file path
                parameter.
        Returns:
            str: Result produced by the operation.
        Why Added:
            Centralizes this behavior inside DocumentIdentityBuilder so related code
                remains cohesive and testable.
        """
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
        """
        Resolve.

        Purpose:
            Implements resolve for the document extraction pipeline that normalizes
                PDFs, enriches visual content, builds RAG units, and indexes data.
        Class:
            Belongs to DocumentIdentityBuilder; uses that class state and dependencies
                when available.
        Args:
            self (Self): Current instance that owns the operation state.
            config (RagIndexingConfig): Configuration object controlling this component.
            chunks (list[dict[str, Any]]): Input value for the chunks parameter.
        Returns:
            RagIndexingConfig: Result produced by the operation.
        Why Added:
            Centralizes this behavior inside DocumentIdentityBuilder so related code
                remains cohesive and testable.
        """
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
        """
        Compute file sha256.

        Purpose:
            Implements _compute_file_sha256 for the document extraction pipeline that
                normalizes PDFs, enriches visual content, builds RAG units, and indexes
                data.
        Class:
            Belongs to DocumentIdentityBuilder; uses that class state and dependencies
                when available.
        Args:
            self (Self): Current instance that owns the operation state.
            file_path (str | Path): Filesystem path to the document or artifact being
                processed.
        Returns:
            str: Result produced by the operation.
        Why Added:
            Centralizes this behavior inside DocumentIdentityBuilder so related code
                remains cohesive and testable.
        """
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
    """
    Embedding Model Factory.

    Purpose:
        Defines EmbeddingModelFactory in the document extraction pipeline that
            normalizes PDFs, enriches visual content, builds RAG units, and indexes
            data.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.
    """

    def __init__(self, config: RagIndexingConfig) -> None:
        """
        Initialize the object with its required dependencies.

        Purpose:
            Implements __init__ for the document extraction pipeline that normalizes
                PDFs, enriches visual content, builds RAG units, and indexes data.
        Class:
            Belongs to EmbeddingModelFactory; uses that class state and dependencies
                when available.
        Args:
            self (Self): Current instance that owns the operation state.
            config (RagIndexingConfig): Configuration object controlling this component.
        Returns:
            None: Performs work through side effects and does not return a value.
        Why Added:
            Centralizes this behavior inside EmbeddingModelFactory so related code
                remains cohesive and testable.
        """
        self._config = config

    def create(self) -> OllamaEmbedding:
        """
        Create.

        Purpose:
            Implements create for the document extraction pipeline that normalizes PDFs,
                enriches visual content, builds RAG units, and indexes data.
        Class:
            Belongs to EmbeddingModelFactory; uses that class state and dependencies
                when available.
        Args:
            self (Self): Current instance that owns the operation state.
        Returns:
            OllamaEmbedding: Result produced by the operation.
        Why Added:
            Centralizes this behavior inside EmbeddingModelFactory so related code
                remains cohesive and testable.
        """
        return OllamaEmbedding(
            model_name=self._config.embedding_model,
            base_url=self._config.ollama_url,
            embed_batch_size=1,
            ollama_additional_kwargs={"num_ctx": 8192},
        )


# ============================================================
# Compact Payload Builder
# ============================================================


class CompactPayloadBuilder:
    """
    Compact Payload Builder.

    Purpose:
        Defines CompactPayloadBuilder in the document extraction pipeline that
            normalizes PDFs, enriches visual content, builds RAG units, and indexes
            data.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.
    """

    def __init__(
        self,
        config: RagIndexingConfig,
        cleaner: JsonSafetyCleaner | None = None,
    ) -> None:
        """
        Initialize the object with its required dependencies.

        Purpose:
            Implements __init__ for the document extraction pipeline that normalizes
                PDFs, enriches visual content, builds RAG units, and indexes data.
        Class:
            Belongs to CompactPayloadBuilder; uses that class state and dependencies
                when available.
        Args:
            self (Self): Current instance that owns the operation state.
            config (RagIndexingConfig): Configuration object controlling this component.
            cleaner (JsonSafetyCleaner | None): Input value for the cleaner parameter.
        Returns:
            None: Performs work through side effects and does not return a value.
        Why Added:
            Centralizes this behavior inside CompactPayloadBuilder so related code
                remains cohesive and testable.
        """
        self._config = config
        self._cleaner = cleaner or JsonSafetyCleaner()

    def build(
        self,
        chunk: dict[str, Any],
        text: str,
        point_id: str,
        chunk_index: int,
    ) -> dict[str, Any]:
        """
        Build.

        Purpose:
            Implements build for the document extraction pipeline that normalizes PDFs,
                enriches visual content, builds RAG units, and indexes data.
        Class:
            Belongs to CompactPayloadBuilder; uses that class state and dependencies
                when available.
        Args:
            self (Self): Current instance that owns the operation state.
            chunk (dict[str, Any]): Input value for the chunk parameter.
            text (str): Input value for the text parameter.
            point_id (str): Input value for the point id parameter.
            chunk_index (int): Input value for the chunk index parameter.
        Returns:
            dict[str, Any]: Structured data produced by the operation.
        Why Added:
            Centralizes this behavior inside CompactPayloadBuilder so related code
                remains cohesive and testable.
        """
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

        return {key: self._cleaner.clean_value(value) for key, value in payload.items() if value is not None}


# ============================================================
# Point ID Builder
# ============================================================


class QdrantPointIdBuilder:
    """
    Qdrant Point Id Builder.

    Purpose:
        Defines QdrantPointIdBuilder in the document extraction pipeline that normalizes
            PDFs, enriches visual content, builds RAG units, and indexes data.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.
    """

    def build(
        self,
        doc_id: str,
        chunk_id: int,
        chunk_index: int,
    ) -> str:
        """
        Build.

        Purpose:
            Implements build for the document extraction pipeline that normalizes PDFs,
                enriches visual content, builds RAG units, and indexes data.
        Class:
            Belongs to QdrantPointIdBuilder; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
            doc_id (str): Input value for the doc id parameter.
            chunk_id (int): Input value for the chunk id parameter.
            chunk_index (int): Input value for the chunk index parameter.
        Returns:
            str: Result produced by the operation.
        Why Added:
            Centralizes this behavior inside QdrantPointIdBuilder so related code
                remains cohesive and testable.
        """
        seed = f"{doc_id}:{chunk_id}:{chunk_index}"
        return str(uuid.uuid5(uuid.NAMESPACE_URL, seed))


class ColbertModelFactory:
    """
    Colbert Model Factory.

    Purpose:
        Defines ColbertModelFactory in the document extraction pipeline that normalizes
            PDFs, enriches visual content, builds RAG units, and indexes data.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.
    """

    def __init__(
        self,
        model_name: str = "colbert-ir/colbertv2.0",
    ) -> None:
        """
        Initialize the object with its required dependencies.

        Purpose:
            Implements __init__ for the document extraction pipeline that normalizes
                PDFs, enriches visual content, builds RAG units, and indexes data.
        Class:
            Belongs to ColbertModelFactory; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
            model_name (str): Input value for the model name parameter.
        Returns:
            None: Performs work through side effects and does not return a value.
        Why Added:
            Centralizes this behavior inside ColbertModelFactory so related code remains
                cohesive and testable.
        """
        self._model_name = model_name

    def create(self) -> LateInteractionTextEmbedding:
        """
        Create.

        Purpose:
            Implements create for the document extraction pipeline that normalizes PDFs,
                enriches visual content, builds RAG units, and indexes data.
        Class:
            Belongs to ColbertModelFactory; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
        Returns:
            LateInteractionTextEmbedding: Result produced by the operation.
        Why Added:
            Centralizes this behavior inside ColbertModelFactory so related code remains
                cohesive and testable.
        """
        return LateInteractionTextEmbedding(self._model_name)


# ============================================================
# Qdrant Hybrid Saver
# ============================================================


class QdrantHybridIndexSaver:
    """
    Qdrant Hybrid Index Saver.

    Purpose:
        Defines QdrantHybridIndexSaver in the document extraction pipeline that
            normalizes PDFs, enriches visual content, builds RAG units, and indexes
            data.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.

    Attributes:
        DENSE_VECTOR_NAME (Any): Class-level value used by this class.
        SPARSE_VECTOR_NAME (Any): Class-level value used by this class.
        BM25_MODEL_NAME (Any): Class-level value used by this class.
        COLBERT_VECTOR_NAME (Any): Class-level value used by this class.
        COLBERT_VECTOR_SIZE (Any): Class-level value used by this class.
        COLBERT_MODEL_NAME (Any): Class-level value used by this class.
    """

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
        """
        Initialize the object with its required dependencies.

        Purpose:
            Implements __init__ for the document extraction pipeline that normalizes
                PDFs, enriches visual content, builds RAG units, and indexes data.
        Class:
            Belongs to QdrantHybridIndexSaver; uses that class state and dependencies
                when available.
        Args:
            self (Self): Current instance that owns the operation state.
            config (RagIndexingConfig): Configuration object controlling this component.
            embedding_model_factory (EmbeddingModelFactory): Input value for the
                embedding model factory parameter.
            payload_builder (CompactPayloadBuilder): Input value for the payload builder
                parameter.
            point_id_builder (QdrantPointIdBuilder | None): Input value for the point id
                builder parameter.
            colbert_model_factory (ColbertModelFactory | None): Input value for the
                colbert model factory parameter.
            chunking_components (RagChunkingComponents | None): Input value for the
                chunking components parameter.
        Returns:
            None: Performs work through side effects and does not return a value.
        Why Added:
            Centralizes this behavior inside QdrantHybridIndexSaver so related code
                remains cohesive and testable.
        """
        self._config = config
        self._embedding_model_factory = embedding_model_factory
        self._payload_builder = payload_builder
        self._point_id_builder = point_id_builder or QdrantPointIdBuilder()
        self._client = qdrant_client.QdrantClient(url=self._config.qdrant_url)
        self._colbert_model_factory = colbert_model_factory or ColbertModelFactory(self.COLBERT_MODEL_NAME)
        self._chunking_components = chunking_components or RagChunkingComponents()

    def _upsert_points_in_batches(
        self,
        points: list[models.PointStruct],
        batch_size: int = 8,
    ) -> None:
        """
        Upsert points in batches.

        Purpose:
            Implements _upsert_points_in_batches for the document extraction pipeline
                that normalizes PDFs, enriches visual content, builds RAG units, and
                indexes data.
        Class:
            Belongs to QdrantHybridIndexSaver; uses that class state and dependencies
                when available.
        Args:
            self (Self): Current instance that owns the operation state.
            points (list[models.PointStruct]): Input value for the points parameter.
            batch_size (int): Input value for the batch size parameter.
        Returns:
            None: Performs work through side effects and does not return a value.
        Why Added:
            Centralizes this behavior inside QdrantHybridIndexSaver so related code
                remains cohesive and testable.
        """
        for start in range(0, len(points), batch_size):
            batch = points[start : start + batch_size]

            print(f"Upserting Qdrant batch: {start // batch_size + 1}, points={len(batch)}")

            self._client.upsert(
                collection_name=self._config.collection_name,
                points=batch,
                wait=True,
            )

    def save(self, chunks: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Save.

        Purpose:
            Implements save for the document extraction pipeline that normalizes PDFs,
                enriches visual content, builds RAG units, and indexes data.
        Class:
            Belongs to QdrantHybridIndexSaver; uses that class state and dependencies
                when available.
        Args:
            self (Self): Current instance that owns the operation state.
            chunks (list[dict[str, Any]]): Input value for the chunks parameter.
        Returns:
            dict[str, Any]: Structured data produced by the operation.
        Why Added:
            Centralizes this behavior inside QdrantHybridIndexSaver so related code
                remains cohesive and testable.
        """
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

        for chunk_index, chunk in enumerate(valid_chunks):
            text = (chunk.get("text") or "").strip()

            nodes = self._chunking_components.splitter.get_nodes_from_documents([Document(text=text)])

            for splitter_index, node in enumerate(nodes):
                chunk_text = node.get_content()

                print(f"chunk_text: {len(chunk_text)}")

                point_id = self._point_id_builder.build(
                    doc_id=self._config.doc_id or "unknown_doc",
                    chunk_id=splitter_index,
                    chunk_index=chunk_index,
                )

                dense_vector = dense_model.get_text_embedding(chunk_text)
                colbert_vector = next(iter(colbert_model.passage_embed([text])))

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
        Delete existing file points.

        Purpose:
            Implements delete_existing_file_points for the document extraction pipeline
                that normalizes PDFs, enriches visual content, builds RAG units, and
                indexes data.
        Class:
            Belongs to QdrantHybridIndexSaver; uses that class state and dependencies
                when available.
        Args:
            self (Self): Current instance that owns the operation state.
        Returns:
            None: Performs work through side effects and does not return a value.
        Why Added:
            Centralizes this behavior inside QdrantHybridIndexSaver so related code
                remains cohesive and testable.
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
        """
        Filter valid chunks.

        Purpose:
            Implements _filter_valid_chunks for the document extraction pipeline that
                normalizes PDFs, enriches visual content, builds RAG units, and indexes
                data.
        Class:
            Belongs to QdrantHybridIndexSaver; uses that class state and dependencies
                when available.
        Args:
            self (Self): Current instance that owns the operation state.
            chunks (list[dict[str, Any]]): Input value for the chunks parameter.
        Returns:
            list[dict[str, Any]]: Structured data produced by the operation.
        Why Added:
            Centralizes this behavior inside QdrantHybridIndexSaver so related code
                remains cohesive and testable.
        """
        valid_chunks: list[dict[str, Any]] = []

        for chunk in chunks:
            text = (chunk.get("text") or "").strip()

            if not text:
                continue

            valid_chunks.append(chunk)

        return valid_chunks

    def _detect_dense_vector_size(self, dense_model: OllamaEmbedding) -> int:
        """
        Detect dense vector size.

        Purpose:
            Implements _detect_dense_vector_size for the document extraction pipeline
                that normalizes PDFs, enriches visual content, builds RAG units, and
                indexes data.
        Class:
            Belongs to QdrantHybridIndexSaver; uses that class state and dependencies
                when available.
        Args:
            self (Self): Current instance that owns the operation state.
            dense_model (OllamaEmbedding): Input value for the dense model parameter.
        Returns:
            int: Result produced by the operation.
        Why Added:
            Centralizes this behavior inside QdrantHybridIndexSaver so related code
                remains cohesive and testable.
        """
        sample_vector = dense_model.get_text_embedding("dimension check")
        return len(sample_vector)

    def _ensure_hybrid_collection(self, dense_size: int) -> None:
        """
        Ensure hybrid collection.

        Purpose:
            Implements _ensure_hybrid_collection for the document extraction pipeline
                that normalizes PDFs, enriches visual content, builds RAG units, and
                indexes data.
        Class:
            Belongs to QdrantHybridIndexSaver; uses that class state and dependencies
                when available.
        Args:
            self (Self): Current instance that owns the operation state.
            dense_size (int): Input value for the dense size parameter.
        Returns:
            None: Performs work through side effects and does not return a value.
        Why Added:
            Centralizes this behavior inside QdrantHybridIndexSaver so related code
                remains cohesive and testable.
        """
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
                    multivector_config=models.MultiVectorConfig(comparator=models.MultiVectorComparator.MAX_SIM),
                ),
            },
            sparse_vectors_config={
                self.SPARSE_VECTOR_NAME: models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                )
            },
        )

    def _average_document_length(self, chunks: list[dict[str, Any]]) -> float:
        """
        Average document length.

        Purpose:
            Implements _average_document_length for the document extraction pipeline
                that normalizes PDFs, enriches visual content, builds RAG units, and
                indexes data.
        Class:
            Belongs to QdrantHybridIndexSaver; uses that class state and dependencies
                when available.
        Args:
            self (Self): Current instance that owns the operation state.
            chunks (list[dict[str, Any]]): Input value for the chunks parameter.
        Returns:
            float: Result produced by the operation.
        Why Added:
            Centralizes this behavior inside QdrantHybridIndexSaver so related code
                remains cohesive and testable.
        """
        lengths = [max(1, len((chunk.get("text") or "").split())) for chunk in chunks]

        return sum(lengths) / max(1, len(lengths))


# ============================================================
# Main Ingestion Service
# ============================================================


class MarkdownRagQdrantIngestionService:
    """
    Markdown Rag Qdrant Ingestion Service.

    Purpose:
        Defines MarkdownRagQdrantIngestionService in the document extraction pipeline
            that normalizes PDFs, enriches visual content, builds RAG units, and indexes
            data.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.
    """

    def __init__(self, config: RagIndexingConfig) -> None:
        """
        Initialize the object with its required dependencies.

        Purpose:
            Implements __init__ for the document extraction pipeline that normalizes
                PDFs, enriches visual content, builds RAG units, and indexes data.
        Class:
            Belongs to MarkdownRagQdrantIngestionService; uses that class state and
                dependencies when available.
        Args:
            self (Self): Current instance that owns the operation state.
            config (RagIndexingConfig): Configuration object controlling this component.
        Returns:
            None: Performs work through side effects and does not return a value.
        Why Added:
            Centralizes this behavior inside MarkdownRagQdrantIngestionService so
                related code remains cohesive and testable.
        """
        self._input_config = config
        self._loader = RagChunkLoader()
        self._identity_builder = DocumentIdentityBuilder()

    def ingest_from_file(
        self,
        chunks_path: str | Path,
        delete_existing_file_points: bool = True,
    ) -> dict[str, Any]:
        """
        Ingest from file.

        Purpose:
            Implements ingest_from_file for the document extraction pipeline that
                normalizes PDFs, enriches visual content, builds RAG units, and indexes
                data.
        Class:
            Belongs to MarkdownRagQdrantIngestionService; uses that class state and
                dependencies when available.
        Args:
            self (Self): Current instance that owns the operation state.
            chunks_path (str | Path): Input value for the chunks path parameter.
            delete_existing_file_points (bool): Input value for the delete existing file
                points parameter.
        Returns:
            dict[str, Any]: Structured data produced by the operation.
        Why Added:
            Centralizes this behavior inside MarkdownRagQdrantIngestionService so
                related code remains cohesive and testable.
        """
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
