from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import re
import tiktoken

import qdrant_client
from qdrant_client import models
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore

MAX_EMBED_CHARS = int(os.getenv("MAX_EMBED_CHARS", "512"))
MAX_ATOMIC_CHARS = int(os.getenv("MAX_ATOMIC_CHARS", "700"))
CHUNK_OVERLAP_CHARS = int(os.getenv("CHUNK_OVERLAP_CHARS", "80"))


@dataclass(frozen=True)
class RagIndexingConfig:
    """Configuration required to build and persist RAG embeddings into Qdrant."""
    doc_id: str | None = None
    doc_title: str | None = None
    source_file_path: str | Path | None = None

    chat_id: str | None = None
    file_id: str | None = None

    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    collection_name: str = os.getenv("QDRANT_COLLECTION", "pdf_rag")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434")


class DocumentIdentityBuilder:
    """Builds stable document identity fields (title and doc_id) for indexing."""

    def clean_text(self, value: Any) -> str:
        """Normalize whitespace and line breaks for title-like text values.

        Args:
            value: Raw value that may contain title text.

        Returns:
            Cleaned single-line text, or empty string for falsy input.
        """
        if not value:
            return ""
        return " ".join(str(value).replace("\n", " ").split()).strip()

    def detect_title(
            self,
            rag_units: list[dict[str, Any]],
            fallback_file_path: str | Path | None = None,
    ) -> str:
        """Infer a document title from RAG units, then fallback to filename.

        Args:
            rag_units: Normalized units used for indexing.
            fallback_file_path: Optional source file path used when title is missing.

        Returns:
            Best-effort document title.
        """
        sorted_units = sorted(rag_units, key=lambda x: x.get("order", 0))

        # Prefer first heading / section title.
        for unit in sorted_units:
            if unit.get("type") == "heading" or unit.get("label") == "section_header":
                title = self.clean_text(unit.get("heading") or unit.get("text"))
                if title:
                    if title.startswith("Section:"):
                        title = title.split("\n")[-1].strip()
                    return title

        # Fallback to heading field from any unit.
        for unit in sorted_units:
            title = self.clean_text(unit.get("heading"))
            if title:
                return title

        # Fallback to filename.
        if fallback_file_path:
            return Path(fallback_file_path).stem.replace("_", " ").replace("-", " ")

        return "Untitled Document"

    def create_doc_id(
            self,
            doc_title: str,
            source_file_path: str | Path | None = None,
    ) -> str:
        """Create a stable UUID for a document identity.

        Args:
            doc_title: Final resolved title for the document.
            source_file_path: Optional source file path used for hashing.

        Returns:
            Deterministic UUID string based on title and optional file hash.
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
            rag_units: list[dict[str, Any]],
    ) -> RagIndexingConfig:
        """Resolve missing identity fields in config.

        Args:
            config: Input indexing configuration.
            rag_units: RAG units used to infer title and id.

        Returns:
            New `RagIndexingConfig` with resolved `doc_title` and `doc_id`.
        """
        doc_title = config.doc_title or self.detect_title(
            rag_units=rag_units,
            fallback_file_path=config.source_file_path,
        )

        doc_id = config.doc_id or self.create_doc_id(
            doc_title=doc_title,
            source_file_path=config.source_file_path,
        )

        return RagIndexingConfig(
            doc_id=doc_id,
            doc_title=doc_title,
            chat_id=config.chat_id,
            file_id=config.file_id,
            source_file_path=config.source_file_path,
            qdrant_url=config.qdrant_url,
            collection_name=config.collection_name,
            embedding_model=config.embedding_model,
            ollama_url=config.ollama_url,
        )

    @staticmethod
    def _compute_file_sha256(file_path: str | Path) -> str:
        """Compute SHA-256 hash for a source file.

        Args:
            file_path: Path to file that should be hashed.

        Returns:
            Hex digest of file content.
        """
        path = Path(file_path)
        digest = hashlib.sha256()

        with path.open("rb") as file:
            for block in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(block)

        return digest.hexdigest()


class RagUnitLoader:
    """Loads serialized RAG units from disk."""

    def load(self, path: str | Path) -> list[dict[str, Any]]:
        """Read and parse a JSON file into RAG unit dictionaries.

        Args:
            path: JSON file path containing serialized rag units.

        Returns:
            List of rag unit dictionaries.
        """
        return json.loads(Path(path).read_text(encoding="utf-8"))


class JsonSafetyCleaner:
    """Ensures metadata values are JSON-serializable before persistence."""

    def clean_value(self, value: Any) -> Any:
        """Ensure value can be serialized to JSON.

        Args:
            value: Arbitrary metadata value.

        Returns:
            Original value if JSON-serializable, otherwise string representation.
        """
        try:
            json.dumps(value, ensure_ascii=False)
            return value
        except TypeError:
            return str(value)

    def compact_bbox(self, bbox: Any) -> Any:
        """Reduce bounding box payload to required metadata fields.

        Args:
            bbox: Raw bbox object from rag unit metadata.

        Returns:
            Compact bbox dictionary or unchanged input when not a dict.
        """
        if not isinstance(bbox, dict):
            return bbox

        return {
            "l": bbox.get("l"),
            "t": bbox.get("t"),
            "r": bbox.get("r"),
            "b": bbox.get("b"),
            "coord_origin": bbox.get("coord_origin"),
        }


class RagIndexingPolicy:
    """Contains inclusion rules for whether a unit should be indexed."""

    def should_index(self, unit: dict[str, Any]) -> bool:
        """Evaluate whether a unit should be indexed.

        Args:
            unit: Single rag unit with type-specific metadata flags.

        Returns:
            `True` if unit should be indexed, otherwise `False`.
        """
        unit_type = unit.get("type")

        if unit_type == "picture":
            metadata = unit.get("vision_metadata") or {}
            return metadata.get("should_index_for_rag", True)

        if unit_type == "table":
            table_vision = unit.get("table_vision") or {}
            return table_vision.get("should_index_for_rag", True)

        return True


class MetadataBuilder:
    """Builds normalized metadata attached to each vector node."""

    def __init__(
            self,
            config: RagIndexingConfig,
            cleaner: JsonSafetyCleaner | None = None,
            policy: RagIndexingPolicy | None = None,
    ) -> None:
        """Initialize metadata builder dependencies.

        Args:
            config: Runtime indexing configuration.
            cleaner: Optional metadata value sanitizer.
            policy: Optional indexing policy helper.
        """
        self._config = config
        self._cleaner = cleaner or JsonSafetyCleaner()
        self._policy = policy or RagIndexingPolicy()

    def build(self, unit: dict[str, Any]) -> dict[str, Any]:
        """Create cleaned metadata payload for one rag unit.

        Args:
            unit: Single rag unit to convert into metadata.

        Returns:
            JSON-safe metadata dictionary for vector node storage.
        """
        table_vision = unit.get("table_vision") or {}
        vision_metadata = unit.get("vision_metadata") or {}

        metadata = {
            # Document identity
            "doc_id": self._config.doc_id,
            "doc_title": self._config.doc_title,

            "chat_id": self._config.chat_id,
            "file_id": self._config.file_id,

            # Source identity
            "source_ref": unit.get("source_ref") or unit.get("id"),
            "source_refs": [unit.get("source_ref") or unit.get("id")],
            "original_id": unit.get("id"),
            "order": unit.get("order"),

            # Type
            "type": unit.get("type"),
            "label": unit.get("label"),

            # Location / citation
            "page_no": unit.get("page_no"),
            "page_start": unit.get("page_no"),
            "page_end": unit.get("page_no"),
            "bbox": self._cleaner.compact_bbox(unit.get("bbox")),
            "bbox_list": [self._cleaner.compact_bbox(unit.get("bbox"))] if unit.get("bbox") else [],

            # Structure
            "heading": unit.get("heading"),
            "heading_level": unit.get("heading_level"),
            "heading_ref": unit.get("heading_ref"),
            "heading_path": unit.get("heading_path") or [],

            # Page furniture
            "page_header": unit.get("page_header", ""),
            "page_footer": unit.get("page_footer", ""),

            # Display / reconstruction
            "image_path": unit.get("image_path"),
            "text": unit.get("text"),

            # Table fields
            "table_markdown": unit.get("table_markdown"),
            "table_type": table_vision.get("table_type") or unit.get("table_type"),
            "columns_summary": table_vision.get("columns_summary"),
            "columns": unit.get("columns"),
            "rows": unit.get("rows"),
            "key_findings": table_vision.get("key_findings") or unit.get("key_findings") or [],
            "rag_keywords": table_vision.get("rag_keywords") or unit.get("rag_keywords") or [],
            "visible_text_summary": table_vision.get("visible_text_summary"),
            "visible_text_long_summary": table_vision.get("visible_text_long_summary"),
            "caption_summary": table_vision.get("caption_summary"),
            "rag_search_text": table_vision.get("rag_search_text"),
            "table_vision": table_vision,

            # Picture fields
            "caption": unit.get("caption"),
            "vision_text": unit.get("vision_text"),
            "vision_metadata": vision_metadata,

            # Retrieval flag
            "should_index_for_rag": self._policy.should_index(unit),
        }

        return {
            key: self._cleaner.clean_value(value)
            for key, value in metadata.items()
            if value is not None
        }


class EmbeddingTextBuilder:
    """Builds final text payload used as embedding input for each unit."""

    def __init__(self, config: RagIndexingConfig) -> None:
        """Store runtime configuration for embedding text construction.

        Args:
            config: Runtime indexing configuration.
        """
        self._config = config

    def build(self, unit: dict[str, Any]) -> str:
        """
        The RAG pipeline already created final searchable text.

        For tables, unit["text"] should already include:
        - Section
        - Table caption
        - Table markdown
        - Table description
        - Visible table text
        - Key findings
        - RAG search text

        So we use unit["text"] directly and only add lightweight document/page/type context.

        Args:
            unit: Single rag unit containing prepared searchable text.

        Returns:
            Final text used as embedding input.
        """
        prepared_text = (unit.get("text") or "").strip()

        if not prepared_text:
            return ""

        prefix_parts: list[str] = []

        if self._config.doc_title:
            prefix_parts.append(f"Document: {self._config.doc_title}")

        heading_path = unit.get("heading_path") or []

        # Avoid duplicate section prefix if already included by previous RAG builder.
        if heading_path and "Section:" not in prepared_text[:300]:
            prefix_parts.append("Section: " + " > ".join(heading_path))

        page_no = unit.get("page_no")
        if page_no:
            prefix_parts.append(f"Page: {page_no}")

        unit_type = unit.get("type")
        if unit_type:
            prefix_parts.append(f"Type: {unit_type}")

        if not prefix_parts:
            return prepared_text

        return "\n".join(prefix_parts) + "\n\n" + prepared_text


class EmbeddingModelFactory:
    """Factory for creating the embedding model instance used in indexing."""

    def __init__(self, config: RagIndexingConfig) -> None:
        """Store runtime configuration for embedding model creation.

        Args:
            config: Runtime indexing configuration.
        """
        self._config = config

    def create(self) -> OllamaEmbedding:
        """Instantiate and return the configured Ollama embedding model.

        Returns:
            Configured `OllamaEmbedding` instance.
        """
        return OllamaEmbedding(
            model_name=self._config.embedding_model,
            base_url=self._config.ollama_url,
            embed_batch_size=1
        )


class NodeBuilder:
    """Converts rag units into `TextNode` objects ready for vector indexing."""

    def __init__(
            self,
            config: RagIndexingConfig,
            metadata_builder: MetadataBuilder,
            text_builder: EmbeddingTextBuilder,
            model_factory: EmbeddingModelFactory,
            policy: RagIndexingPolicy | None = None,
    ) -> None:
        """Initialize node builder services.

        Args:
            config: Runtime indexing configuration.
            metadata_builder: Builds node metadata from rag units.
            text_builder: Builds embedding text from rag units.
            model_factory: Creates embedding model for semantic splitting.
            policy: Optional indexing policy helper.
        """
        self._config = config
        self._metadata_builder = metadata_builder
        self._text_builder = text_builder
        self._model_factory = model_factory
        self._policy = policy or RagIndexingPolicy()
        self._length_guard = TokenLengthGuard(
            max_tokens=MAX_EMBED_CHARS,
            overlap_tokens=CHUNK_OVERLAP_CHARS,
        )

        self._atomic_length_guard = TokenLengthGuard(
            max_tokens=MAX_ATOMIC_CHARS,
            overlap_tokens=CHUNK_OVERLAP_CHARS,
        )

    def build_all(self, rag_units: list[dict[str, Any]]) -> list[TextNode]:
        """Build semantic and atomic nodes from all indexable rag units.

        Args:
            rag_units: Source rag units from normalization/enrichment pipeline.

        Returns:
            List of text nodes ready for vector indexing.
        """
        embed_model = self._model_factory.create()

        text_nodes = self._build_semantic_text_nodes(
            rag_units=rag_units,
            embed_model=embed_model,
        )

        atomic_nodes = self._build_atomic_nodes(
            rag_units=rag_units,
        )

        return text_nodes + atomic_nodes

    def _source_relationships(self) -> dict[NodeRelationship, RelatedNodeInfo]:
        """Attach source document identity so LlamaIndex/Qdrant ref_doc_id is not None."""
        if not self._config.doc_id:
            return {}

        return {
            NodeRelationship.SOURCE: RelatedNodeInfo(
                node_id=self._config.doc_id,
                metadata={
                    "doc_id": self._config.doc_id,
                    "doc_title": self._config.doc_title,
                    "chat_id": self._config.chat_id,
                    "file_id": self._config.file_id,
                },
            )
        }

    def _source_document_id(self, source_ref: Any) -> str:
        """Create stable source Document id for semantic pre-split documents."""
        return str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"{self._config.doc_id}:{source_ref}:source-document",
            )
        )

    def _build_semantic_text_nodes(
            self,
            rag_units: list[dict[str, Any]],
            embed_model: OllamaEmbedding,
    ) -> list[TextNode]:
        """Build semantically split nodes for textual/group content.

        Args:
            rag_units: Source rag units.
            embed_model: Embedding model used by semantic splitter.

        Returns:
            List of semantic text nodes.
        """
        text_docs: list[Document] = []

        for unit in rag_units:
            if unit.get("type") not in {"text", "group"}:
                continue

            if not self._policy.should_index(unit):
                continue

            text = self._text_builder.build(unit)

            if not text.strip():
                continue

            metadata = self._metadata_builder.build(unit)

            # 1. TokenLengthGuard BEFORE semantic splitter
            guarded_chunks = self._length_guard.split(text)

            for chunk_index, chunk_text in enumerate(guarded_chunks):
                source_ref = metadata.get("source_ref", unit.get("id", "unknown"))
                source_doc_id = self._source_document_id(source_ref)

                text_docs.append(
                    Document(
                        id_=source_doc_id,
                        text=chunk_text,
                        metadata={
                            **metadata,
                            "document_id": self._config.doc_id,
                            "ref_doc_id": self._config.doc_id,
                            "source_document_id": source_doc_id,
                            "pre_semantic_chunk_index": chunk_index,
                            "pre_semantic_chunk_count": len(guarded_chunks),
                            "pre_semantic_chunking_strategy": "token_length_guard",
                        },
                    )
                )

        if not text_docs:
            return []

        # 2. Semantic splitter only receives safe-size documents
        splitter = SemanticSplitterNodeParser(
            buffer_size=1,
            breakpoint_percentile_threshold=95,
            embed_model=embed_model,
            # Keep metadata on produced TextNode objects.
            # If this is False in your installed LlamaIndex version, the semantic nodes
            # may only keep chunk metadata and drop doc/page/source metadata.
            include_metadata=True,
            include_prev_next_rel=True,
        )

        semantic_nodes = splitter.get_nodes_from_documents(text_docs)

        final_nodes: list[TextNode] = []

        for i, node in enumerate(semantic_nodes):
            source_ref = node.metadata.get("source_ref", "unknown")
            pre_chunk_index = node.metadata.get("pre_semantic_chunk_index", 0)

            # 3. Final TokenLengthGuard safety check
            final_chunks = self._length_guard.split(node.text)

            for final_index, final_text in enumerate(final_chunks):
                chunking_strategy = (
                    "semantic"
                    if len(final_chunks) == 1
                    else "semantic_token_length_guard"
                )

                node_metadata = {
                    **node.metadata,
                    "document_id": self._config.doc_id,
                    "doc_id": self._config.doc_id,
                    "ref_doc_id": self._config.doc_id,
                    "chunk_id": f"{self._config.doc_id}_text_{len(final_nodes):05d}",
                    "chunking_strategy": chunking_strategy,
                    "semantic_node_index": i,
                    "final_chunk_index": final_index,
                    "final_chunk_count": len(final_chunks),
                }

                safe_node = TextNode(
                    id_=str(
                        uuid.uuid5(
                            uuid.NAMESPACE_URL,
                            f"{self._config.doc_id}:{source_ref}:semantic:{pre_chunk_index}:{i}:{final_index}",
                        )
                    ),
                    text=final_text,
                    metadata=node_metadata,
                    relationships=self._source_relationships(),
                )

                final_nodes.append(exclude_metadata_from_embedding(safe_node))

        return final_nodes

    def _build_atomic_nodes(self, rag_units: list[dict[str, Any]]) -> list[TextNode]:
        """Build non-semantic (atomic) nodes for picture/table content.

        Args:
            rag_units: Source rag units.

        Returns:
            List of atomic text nodes.
        """
        nodes: list[TextNode] = []

        for unit in rag_units:
            unit_type = unit.get("type")

            if unit_type not in {"table", "picture"}:
                continue

            if not self._policy.should_index(unit):
                continue

            text = self._text_builder.build(unit)

            if not text.strip():
                continue

            metadata = self._metadata_builder.build(unit)
            source_ref = metadata.get("source_ref", unit.get("id", "unknown"))

            # Tables/pictures do not use SemanticSplitterNodeParser
            guarded_chunks = self._atomic_length_guard.split(text)

            for chunk_index, chunk_text in enumerate(guarded_chunks):
                chunking_strategy = (
                    "atomic"
                    if len(guarded_chunks) == 1
                    else "atomic_token_length_guard"
                )

                node_metadata = {
                    **metadata,
                    "document_id": self._config.doc_id,
                    "doc_id": self._config.doc_id,
                    "ref_doc_id": self._config.doc_id,
                    "chunk_id": f"{self._config.doc_id}_{unit_type}_{unit.get('order')}_{chunk_index}",
                    "chunking_strategy": chunking_strategy,
                    "atomic_chunk_index": chunk_index,
                    "atomic_chunk_count": len(guarded_chunks),
                }

                node = TextNode(
                    id_=str(
                        uuid.uuid5(
                            uuid.NAMESPACE_URL,
                            f"{self._config.doc_id}:{source_ref}:atomic:{chunk_index}",
                        )
                    ),
                    text=chunk_text,
                    metadata=node_metadata,
                    relationships=self._source_relationships(),
                )

                nodes.append(exclude_metadata_from_embedding(node))

        return nodes


class QdrantIndexSaver:
    """Persists prepared nodes into Qdrant with dense + BM25 sparse vectors.

    This replaces the pure LlamaIndex `QdrantVectorStore` save path because hybrid
    search needs two named vectors on every point:
    - `dense`: semantic vector from Ollama / nomic-embed-text
    - `bm25`: sparse lexical vector from Qdrant BM25
    """

    DENSE_VECTOR_NAME = "dense"
    SPARSE_VECTOR_NAME = "bm25"
    BM25_MODEL_NAME = "Qdrant/bm25"

    def __init__(
            self,
            config: RagIndexingConfig,
            model_factory: EmbeddingModelFactory,
    ) -> None:
        """Initialize saver dependencies.

        Args:
            config: Runtime indexing configuration.
            model_factory: Creates dense embed model for vector indexing.
        """
        self._config = config
        self._model_factory = model_factory
        self._client = qdrant_client.QdrantClient(url=self._config.qdrant_url)

    def save(self, nodes: list[TextNode]) -> dict[str, Any]:
        """Create a hybrid Qdrant collection and write nodes into it.

        Args:
            nodes: Final text nodes prepared for indexing.

        Returns:
            Summary of saved points.
        """
        if not nodes:
            return {"collection_name": self._config.collection_name, "points_count": 0}

        dense_model = self._model_factory.create()
        dense_size = self._detect_dense_vector_size(dense_model)
        self._ensure_hybrid_collection(dense_size=dense_size)

        avg_len = self._average_document_length(nodes)
        points: list[models.PointStruct] = []

        for node in nodes:
            dense_vector = dense_model.get_text_embedding(node.text)
            payload = self._build_payload(node=node)

            points.append(
                models.PointStruct(
                    id=node.node_id,
                    vector={
                        self.DENSE_VECTOR_NAME: dense_vector,
                        self.SPARSE_VECTOR_NAME: models.Document(
                            text=node.text,
                            model=self.BM25_MODEL_NAME,
                            options={"avg_len": avg_len},
                        ),
                    },
                    payload=payload,
                )
            )

        self._client.upsert(
            collection_name=self._config.collection_name,
            points=points,
            wait=True,
        )

        return {
            "collection_name": self._config.collection_name,
            "points_count": len(points),
            "dense_vector_name": self.DENSE_VECTOR_NAME,
            "sparse_vector_name": self.SPARSE_VECTOR_NAME,
            "bm25_model": self.BM25_MODEL_NAME,
        }

    def _detect_dense_vector_size(self, dense_model: OllamaEmbedding) -> int:
        """Detect dense vector dimension from the configured embedding model."""
        sample_vector = dense_model.get_text_embedding("dimension check")
        return len(sample_vector)

    def _ensure_hybrid_collection(self, dense_size: int) -> None:
        """Create collection configured for dense semantic + BM25 sparse vectors."""
        if self._client.collection_exists(self._config.collection_name):
            # Existing collection must already have named vectors `dense` and `bm25`.
            # If your old collection was created by LlamaIndex with an unnamed vector,
            # delete/recreate it before re-ingesting.
            return

        self._client.create_collection(
            collection_name=self._config.collection_name,
            vectors_config={
                self.DENSE_VECTOR_NAME: models.VectorParams(
                    size=dense_size,
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                self.SPARSE_VECTOR_NAME: models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                )
            },
        )

    def _average_document_length(self, nodes: list[TextNode]) -> float:
        """Estimate average token/word length for BM25 normalization."""
        lengths = [max(1, len((node.text or "").split())) for node in nodes]
        return sum(lengths) / max(1, len(lengths))

    def _build_payload(self, node: TextNode) -> dict[str, Any]:
        """Build Qdrant payload from TextNode metadata and text."""
        payload = {
            **(node.metadata or {}),
            "text": node.text,
            "node_id": node.node_id,
            "document_id": (node.metadata or {}).get("document_id") or self._config.doc_id,
            "doc_id": (node.metadata or {}).get("doc_id") or self._config.doc_id,
            "ref_doc_id": (node.metadata or {}).get("ref_doc_id") or self._config.doc_id,
        }

        # Make sure payload is JSON-safe.
        cleaner = JsonSafetyCleaner()
        return {key: cleaner.clean_value(value) for key, value in payload.items() if value is not None}


def exclude_metadata_from_embedding(node: TextNode) -> TextNode:
    """Exclude metadata fields from embedding text while retaining LLM metadata.

    Args:
        node: Node whose metadata should be excluded from embedding input.

    Returns:
        Same node instance with exclusion settings updated.
    """
    node.excluded_embed_metadata_keys = list(node.metadata.keys())
    node.excluded_llm_metadata_keys = []
    return node


class TokenLengthGuard:
    """Token-aware text splitter with overlap to stay within model limits."""

    def __init__(
            self,
            max_tokens: int = 512,
            overlap_tokens: int = 80,
            encoding_name: str = "cl100k_base",
    ) -> None:
        """Initialize token-aware splitter settings.

        Args:
            max_tokens: Maximum tokens allowed per chunk.
            overlap_tokens: Token overlap between adjacent chunks.
            encoding_name: Tiktoken encoding name used for token counting.
        """
        self._max_tokens = max_tokens
        self._overlap_tokens = overlap_tokens
        self._encoding = tiktoken.get_encoding(encoding_name)

    def count_tokens(self, text: str) -> int:
        """Count tokens for text using configured tokenizer.

        Args:
            text: Input text.

        Returns:
            Number of tokens.
        """
        return len(self._encoding.encode(text or ""))

    def split(self, text: str) -> list[str]:
        """Split text into overlapping chunks constrained by `max_tokens`.

        Args:
            text: Input text to split.

        Returns:
            List of chunks that satisfy token constraints.
        """
        text = (text or "").strip()

        if not text:
            return []

        if self.count_tokens(text) <= self._max_tokens:
            return [text]

        sentences = self._split_sentences(text)

        chunks: list[str] = []
        current_sentences: list[str] = []
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)

            # If one sentence itself is too large, hard split by tokens.
            if sentence_tokens > self._max_tokens:
                if current_sentences:
                    chunks.append(" ".join(current_sentences).strip())
                    current_sentences = []
                    current_tokens = 0

                chunks.extend(self._split_large_sentence(sentence))
                continue

            if current_tokens + sentence_tokens > self._max_tokens:
                chunk = " ".join(current_sentences).strip()
                if chunk:
                    chunks.append(chunk)

                overlap_text = self._build_overlap(current_sentences)
                current_sentences = [overlap_text, sentence] if overlap_text else [sentence]
                current_tokens = self.count_tokens(" ".join(current_sentences))

            else:
                current_sentences.append(sentence)
                current_tokens += sentence_tokens

        final_chunk = " ".join(current_sentences).strip()
        if final_chunk:
            chunks.append(final_chunk)

        return chunks

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentence-like segments to preserve boundaries.

        Args:
            text: Input text.

        Returns:
            Sentence-like text segments.
        """
        # Keeps sentence boundaries better than plain character slicing.
        parts = re.split(r"(?<=[.!?])\s+|\n{2,}", text)
        return [part.strip() for part in parts if part.strip()]

    def _build_overlap(self, sentences: list[str]) -> str:
        """Build trailing overlap text under overlap token budget.

        Args:
            sentences: Candidate sentences from current chunk.

        Returns:
            Combined overlap text.
        """
        overlap: list[str] = []
        total_tokens = 0

        for sentence in reversed(sentences):
            sentence_tokens = self.count_tokens(sentence)

            if total_tokens + sentence_tokens > self._overlap_tokens:
                break

            overlap.insert(0, sentence)
            total_tokens += sentence_tokens

        return " ".join(overlap).strip()

    def _split_large_sentence(self, sentence: str) -> list[str]:
        """Hard-split a long sentence by token windows with overlap.

        Args:
            sentence: Single sentence exceeding token budget.

        Returns:
            List of token-window chunks.
        """
        tokens = self._encoding.encode(sentence)

        chunks: list[str] = []
        start = 0

        while start < len(tokens):
            end = min(start + self._max_tokens, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self._encoding.decode(chunk_tokens).strip()

            if chunk_text:
                chunks.append(chunk_text)

            if end >= len(tokens):
                break

            start = max(0, end - self._overlap_tokens)

        return chunks


class RagQdrantIngestionService:
    """Orchestrates file-based RAG unit ingestion into Qdrant."""

    def __init__(self, config: RagIndexingConfig) -> None:
        """Initialize ingestion service with input configuration.

        Args:
            config: Input indexing configuration.
        """
        self._input_config = config
        self._loader = RagUnitLoader()
        self._identity_builder = DocumentIdentityBuilder()

    def _build_runtime_services(
            self,
            config: RagIndexingConfig,
    ) -> tuple[NodeBuilder, QdrantIndexSaver]:
        """Wire runtime collaborators needed to transform and persist nodes.

        Args:
            config: Resolved runtime indexing configuration.

        Returns:
            Tuple of `(NodeBuilder, QdrantIndexSaver)`.
        """
        policy = RagIndexingPolicy()
        cleaner = JsonSafetyCleaner()

        metadata_builder = MetadataBuilder(
            config=config,
            cleaner=cleaner,
            policy=policy,
        )

        text_builder = EmbeddingTextBuilder(
            config=config,
        )

        model_factory = EmbeddingModelFactory(
            config=config,
        )

        node_builder = NodeBuilder(
            config=config,
            metadata_builder=metadata_builder,
            text_builder=text_builder,
            model_factory=model_factory,
            policy=policy,
        )

        saver = QdrantIndexSaver(
            config=config,
            model_factory=model_factory,
        )

        return node_builder, saver

    def ingest_from_file(self, rag_units_path: str | Path) -> dict[str, Any]:
        """Load RAG units, build nodes, index in Qdrant, and return summary metadata.

        Args:
            rag_units_path: Path to JSON file containing rag units.

        Returns:
            Summary dictionary with identity, collection, and node count.
        """
        rag_units = self._loader.load(rag_units_path)

        runtime_config = self._identity_builder.resolve(
            config=self._input_config,
            rag_units=rag_units,
        )

        node_builder, saver = self._build_runtime_services(runtime_config)

        nodes = node_builder.build_all(rag_units)
        save_result = saver.save(nodes)

        return {
            "chat_id": runtime_config.chat_id,
            "file_id": runtime_config.file_id,
            "doc_id": runtime_config.doc_id,
            "doc_title": runtime_config.doc_title,
            "collection_name": runtime_config.collection_name,
            "nodes_count": len(nodes),
            "save_result": save_result,
        }
