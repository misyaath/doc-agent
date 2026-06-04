"""Folder-level tests for agent helpers, query parsing, retrieval, and answer generation."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, cast

from langchain_core.messages import HumanMessage

from agent.AgentPrompts import AgentPrompts
from agent.qdrant_retrieval import QdrantFilterFactory
from agent.rag import (
    AnswerGenerationService,
    LlmJsonExtractor,
    QueryParsingService,
    QueryPlanBuilder,
    QueryPlanDefaults,
    RetrievalContextFormatter,
    RetrievalService,
)


class FakeLlm:
    """Small LLM double that records prompt messages and returns fixed content."""

    def __init__(self, content: str | dict[str, Any]) -> None:
        """Store the fake response payload used by agent service tests."""
        self.content = content
        self.messages: list[Any] = []

    def invoke(self, messages: list[Any]) -> SimpleNamespace:
        """Return the configured response while preserving the prompt for assertions."""
        self.messages = messages
        return SimpleNamespace(content=self.content)


class FakeRetriever:
    """Retriever double that records query execution arguments."""

    def __init__(self) -> None:
        """Initialize a deterministic chunk response for retrieval tests."""
        self.calls: list[dict[str, Any]] = []

    def retrieve_many(
        self,
        queries: list[str],
        *,
        chat_id: str,
        limit_per_query: int,
        final_limit: int,
    ) -> list[dict[str, Any]]:
        """Return one chunk and capture the retrieval options."""
        self.calls.append(
            {
                "queries": queries,
                "chat_id": chat_id,
                "limit_per_query": limit_per_query,
                "final_limit": final_limit,
            }
        )
        return [
            {
                "id": "chunk-1",
                "text": "Agent retrieval context.",
                "score": 0.91,
                "metadata": {"page_no": 4, "source": "doc.pdf"},
            }
        ]


def test_query_parsing_service_uses_prompt_and_json_extractor() -> None:
    """Verify query parsing converts LLM JSON text into a retrieval plan."""
    llm = FakeLlm('{"query_type":"lookup","clean_question":"What is AI?","keywords":["AI"]}')
    prompts = AgentPrompts(document_summary=[])
    service = QueryParsingService(
        llm=cast(Any, llm),
        prompts=prompts,
        json_extractor=LlmJsonExtractor(default_factory=QueryPlanDefaults()),
    )

    parsed = service.parse("What is AI?")

    assert parsed["query_type"] == "lookup"
    assert parsed["keywords"] == ["AI"]
    assert len(llm.messages) == 2
    assert "What is AI?" in cast(str, llm.messages[1].content)


def test_agent_prompts_formats_file_summary_records_and_skips_none_summary() -> None:
    """Verify prompt summaries support file records and skip records without summaries."""
    prompts = AgentPrompts(
        document_summary=[
            {"title": "Indexed Doc", "summary": [{"heading": "Intro", "summary": "Overview"}]},
            {"title": "Pending Doc", "summary": None},
        ],
    )

    formatted = prompts._format_document_summaries_for_prompt()

    assert "Indexed Doc" in formatted
    assert "heading" in formatted
    assert "Intro" in formatted
    assert "Overview" in formatted
    assert "Pending Doc" not in formatted


def test_agent_prompts_returns_fallback_when_all_summaries_are_none() -> None:
    """Verify prompt summaries do not build empty document blocks for null summaries."""
    prompts = AgentPrompts(
        document_summary=[{"title": "Pending Doc", "summary": None}],
    )

    assert prompts._format_document_summaries_for_prompt() == "No document summaries available."


def test_retrieval_service_builds_queries_and_formats_context() -> None:
    """Verify retrieval service builds deduplicated queries and formatted context."""
    retriever = FakeRetriever()
    service = RetrievalService(
        retriever=cast(Any, retriever),
        query_plan_builder=QueryPlanBuilder(),
        context_formatter=RetrievalContextFormatter(),
    )

    chunks, context = service.retrieve(
        question="Explain the document",
        parsed_query={"clean_question": "Explain the document", "keywords": ["summary"]},
        chat_id="chat-1",
    )

    assert chunks[0]["id"] == "chunk-1"
    assert "Agent retrieval context." in context
    assert retriever.calls[0]["queries"] == ["Explain the document", "summary"]
    assert retriever.calls[0]["chat_id"] == "chat-1"


def test_answer_generation_service_handles_empty_and_llm_context() -> None:
    """Verify answer generation avoids hallucination when no context is available."""
    prompts = AgentPrompts(document_summary=[{"heading": "Intro"}])
    empty_service = AnswerGenerationService(llm=cast(Any, FakeLlm("unused")), prompts=prompts)
    assert empty_service.generate("Question?", "   ", []) == "I could not find this in the uploaded document."

    llm = FakeLlm({"answer": "Use the retrieved section."})
    service = AnswerGenerationService(llm=cast(Any, llm), prompts=prompts)
    answer = service.generate("Question?", "Relevant context", [HumanMessage(content="previous")])

    assert json.loads(answer) == {"answer": "Use the retrieved section."}
    assert len(llm.messages) == 3
    assert "Relevant context" in cast(str, llm.messages[-1].content)


def test_qdrant_filter_factory_scopes_queries_to_chat_id() -> None:
    """Verify Qdrant filters include the chat identifier isolation condition."""
    qdrant_filter = QdrantFilterFactory.for_chat("chat-123")

    must_conditions = cast(list[Any], qdrant_filter.must)
    condition = must_conditions[0]
    assert condition.key == "chat_id"
    assert condition.match.value == "chat-123"
