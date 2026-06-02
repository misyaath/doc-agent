"""Unit tests for reusable agent helper classes and retrieval formatting behavior."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

from agent.qdrant_retrieval import ChunkAggregator, ChunkPayloadMapper, ChunkQualityEvaluator, QueryNormalizer
from agent.rag import LlmJsonExtractor, QueryPlanBuilder, QueryPlanDefaults, RetrievalContextFormatter


def test_query_plan_defaults_and_json_extractor() -> None:
    """Verify query plan defaults and json extractor."""
    defaults = QueryPlanDefaults.build()
    assert defaults["query_type"] == "unknown"
    assert defaults["keywords"] == []

    extractor = LlmJsonExtractor(default_factory=QueryPlanDefaults())
    parsed = extractor.extract('prefix {"query_type":"summary","keywords":["pdf"]} suffix')
    assert parsed["query_type"] == "summary"
    assert parsed["keywords"] == ["pdf"]

    fallback = extractor.extract("not-json")
    assert fallback == defaults


def test_query_plan_builder_deduplicates_and_combines_terms() -> None:
    """Verify query plan builder deduplicates and combines terms."""
    builder = QueryPlanBuilder()
    queries = builder.build_queries(
        "Original question",
        {
            "clean_question": "Clean question",
            "retrieval_queries": ["Clean question", "extra query", ""],
            "keywords": ["alpha", "beta"],
            "entities": ["Entity"],
            "section_hints": ["Intro"],
        },
    )
    assert queries[:3] == ["Original question", "Clean question", "extra query"]
    assert "alpha beta Entity Intro" in queries
    assert len(queries) == len(set(queries))


def test_retrieval_context_formatter_formats_chunks() -> None:
    """Verify retrieval context formatter formats chunks."""
    formatter = RetrievalContextFormatter()
    context = formatter.format(
        [
            {
                "text": "Relevant paragraph",
                "score": 0.9,
                "metadata": {"page_no": 2, "heading_path": ["Intro", "Scope"]},
            }
        ]
    )
    assert "Relevant paragraph" in context
    assert "Page: 2" in context
    assert "[Chunk 1]" in context


def test_query_normalizer_chunk_aggregator_and_quality_evaluator() -> None:
    """Verify query normalizer chunk aggregator and quality evaluator."""
    assert QueryNormalizer().deduplicate([" alpha ", "alpha", "", "beta"]) == ["alpha", "beta"]

    chunks = [
        {"id": "a", "score": 0.2, "text": "first"},
        {"id": "a", "score": 0.8, "text": "duplicate"},
        {"id": "b", "score": 0.5, "text": "second"},
    ]
    merged = ChunkAggregator().merge(chunks)
    assert len(merged) == 2
    assert merged[0]["final_score"] == 0.2
    assert ChunkAggregator().sort(merged)[0]["id"] == "b"

    evaluator = ChunkQualityEvaluator()
    assert evaluator.is_good_chunk("This is a meaningful document chunk with enough letters.")
    assert not evaluator.is_good_chunk("p. 1")
    assert (
        len(evaluator.filter_bad_chunks([{"text": "short"}, {"text": "This chunk has enough alphabetic content."}]))
        == 1
    )


def test_chunk_payload_mapper_maps_qdrant_point_shape() -> None:
    """Verify chunk payload mapper maps qdrant point shape."""
    point = SimpleNamespace(
        id="point-1",
        score=0.75,
        payload={"text": "chunk text", "doc_id": "doc-1", "page_no": 3, "ignored": "value"},
    )
    chunk = ChunkPayloadMapper().to_chunk(cast(Any, point), score_key="dense_score")
    assert chunk["id"] == "point-1"
    assert chunk["text"] == "chunk text"
    assert chunk["dense_score"] == 0.75
    assert chunk["metadata"]["doc_id"] == "doc-1"
    assert chunk["metadata"]["page_no"] == 3
