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
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.schema import TextNode
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore

MAX_EMBED_CHARS = int(os.getenv("MAX_EMBED_CHARS", "512"))
MAX_ATOMIC_CHARS = int(os.getenv("MAX_ATOMIC_CHARS", "2500"))
CHUNK_OVERLAP_CHARS = int(os.getenv("CHUNK_OVERLAP_CHARS", "80"))


@dataclass(frozen=True)
class RagIndexingConfig:
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
    def clean_text(self, value: Any) -> str:
        if not value:
            return ""
        return " ".join(str(value).replace("\n", " ").split()).strip()

    def detect_title(
            self,
            rag_units: list[dict[str, Any]],
            fallback_file_path: str | Path | None = None,
    ) -> str:
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
        """
        Stable UUID.

        If source_file_path exists:
            doc_id = UUID(title + file hash)
        Else:
            doc_id = UUID(title)
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
        path = Path(file_path)
        digest = hashlib.sha256()

        with path.open("rb") as file:
            for block in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(block)

        return digest.hexdigest()


class RagUnitLoader:
    def load(self, path: str | Path) -> list[dict[str, Any]]:
        return json.loads(Path(path).read_text(encoding="utf-8"))


class JsonSafetyCleaner:
    def clean_value(self, value: Any) -> Any:
        try:
            json.dumps(value, ensure_ascii=False)
            return value
        except TypeError:
            return str(value)

    def compact_bbox(self, bbox: Any) -> Any:
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
    def should_index(self, unit: dict[str, Any]) -> bool:
        unit_type = unit.get("type")

        if unit_type == "picture":
            metadata = unit.get("vision_metadata") or {}
            return metadata.get("should_index_for_rag", True)

        if unit_type == "table":
            table_vision = unit.get("table_vision") or {}
            return table_vision.get("should_index_for_rag", True)

        return True


class MetadataBuilder:
    def __init__(
            self,
            config: RagIndexingConfig,
            cleaner: JsonSafetyCleaner | None = None,
            policy: RagIndexingPolicy | None = None,
    ) -> None:
        self._config = config
        self._cleaner = cleaner or JsonSafetyCleaner()
        self._policy = policy or RagIndexingPolicy()

    def build(self, unit: dict[str, Any]) -> dict[str, Any]:
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
    def __init__(self, config: RagIndexingConfig) -> None:
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
    def __init__(self, config: RagIndexingConfig) -> None:
        self._config = config

    def create(self) -> OllamaEmbedding:
        return OllamaEmbedding(
            model_name=self._config.embedding_model,
            base_url=self._config.ollama_url,
        )


class NodeBuilder:
    def __init__(
            self,
            config: RagIndexingConfig,
            metadata_builder: MetadataBuilder,
            text_builder: EmbeddingTextBuilder,
            model_factory: EmbeddingModelFactory,
            policy: RagIndexingPolicy | None = None,
    ) -> None:
        self._config = config
        self._metadata_builder = metadata_builder
        self._text_builder = text_builder
        self._model_factory = model_factory
        self._policy = policy or RagIndexingPolicy()
        self._length_guard = TokenLengthGuard(
            max_tokens=MAX_EMBED_CHARS,
            overlap_tokens=CHUNK_OVERLAP_CHARS,
        )

    def build_all(self, rag_units: list[dict[str, Any]]) -> list[TextNode]:
        embed_model = self._model_factory.create()

        text_nodes = self._build_semantic_text_nodes(
            rag_units=rag_units,
            embed_model=embed_model,
        )

        atomic_nodes = self._build_atomic_nodes(
            rag_units=rag_units,
        )

        return text_nodes + atomic_nodes

    def _build_semantic_text_nodes(
            self,
            rag_units: list[dict[str, Any]],
            embed_model: OllamaEmbedding,
    ) -> list[TextNode]:
        text_docs: list[Document] = []

        for unit in rag_units:
            if unit.get("type") not in {"text", "group"}:
                continue

            if not self._policy.should_index(unit):
                continue

            text = self._text_builder.build(unit)

            if not text.strip():
                continue

            text_docs.append(
                Document(
                    text=text,
                    metadata=self._metadata_builder.build(unit),
                )
            )

        if not text_docs:
            return []

        splitter = SemanticSplitterNodeParser(
            buffer_size=1,
            breakpoint_percentile_threshold=95,
            embed_model=embed_model,
            include_metadata=True,
            include_prev_next_rel=True,
        )

        nodes = splitter.get_nodes_from_documents(text_docs)

        final_nodes: list[TextNode] = []

        for i, node in enumerate(nodes):
            source_ref = node.metadata.get("source_ref", "unknown")

            node.id_ = str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"{self._config.doc_id}:{source_ref}:semantic:{i}",
                )
            )

            node.metadata["chunk_id"] = f"{self._config.doc_id}_text_{i:05d}"
            node.metadata["chunking_strategy"] = "semantic"

            final_nodes.append(node)

        return final_nodes

    def _build_atomic_nodes(self, rag_units: list[dict[str, Any]]) -> list[TextNode]:
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

            node = TextNode(
                id_=str(
                    uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        f"{self._config.doc_id}:{source_ref}:atomic",
                    )
                ),
                text=text,
                metadata={
                    **metadata,
                    "chunk_id": f"{self._config.doc_id}_{unit_type}_{unit.get('order')}",
                    "chunking_strategy": "atomic",
                },
            )

            nodes.append(node)

        return nodes


class QdrantIndexSaver:
    def __init__(
            self,
            config: RagIndexingConfig,
            model_factory: EmbeddingModelFactory,
    ) -> None:
        self._config = config
        self._model_factory = model_factory

    def save(self, nodes: list[TextNode]) -> VectorStoreIndex:
        client = qdrant_client.QdrantClient(
            url=self._config.qdrant_url,
        )

        vector_store = QdrantVectorStore(
            client=client,
            collection_name=self._config.collection_name,
        )

        storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
        )

        return VectorStoreIndex(
            nodes=nodes,
            storage_context=storage_context,
            embed_model=self._model_factory.create(),
        )


import re
import tiktoken


class TokenLengthGuard:
    def __init__(
            self,
            max_tokens: int = 512,
            overlap_tokens: int = 80,
            encoding_name: str = "cl100k_base",
    ) -> None:
        self._max_tokens = max_tokens
        self._overlap_tokens = overlap_tokens
        self._encoding = tiktoken.get_encoding(encoding_name)

    def count_tokens(self, text: str) -> int:
        return len(self._encoding.encode(text or ""))

    def split(self, text: str) -> list[str]:
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
        # Keeps sentence boundaries better than plain character slicing.
        parts = re.split(r"(?<=[.!?])\s+|\n{2,}", text)
        return [part.strip() for part in parts if part.strip()]

    def _build_overlap(self, sentences: list[str]) -> str:
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
    def __init__(self, config: RagIndexingConfig) -> None:
        self._input_config = config
        self._loader = RagUnitLoader()
        self._identity_builder = DocumentIdentityBuilder()

    def _build_runtime_services(
            self,
            config: RagIndexingConfig,
    ) -> tuple[NodeBuilder, QdrantIndexSaver]:
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
        rag_units = self._loader.load(rag_units_path)

        runtime_config = self._identity_builder.resolve(
            config=self._input_config,
            rag_units=rag_units,
        )

        node_builder, saver = self._build_runtime_services(runtime_config)

        nodes = node_builder.build_all(rag_units)
        saver.save(nodes)

        return {
            "chat_id": runtime_config.chat_id,
            "file_id": runtime_config.file_id,
            "doc_id": runtime_config.doc_id,
            "doc_title": runtime_config.doc_title,
            "collection_name": runtime_config.collection_name,
            "nodes_count": len(nodes),
        }
