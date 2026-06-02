import json
import re
from typing import Any, TypedDict, Annotated

from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph, add_messages

from agent.AgentPrompts import AgentPrompts
from agent.langgraph_memory import checkpointer
from agent.qdrant_retrieval import QdrantRagRetriever


class RetrievedChunk(TypedDict, total=False):
    id: str
    text: str
    score: float
    metadata: dict[str, Any]


class RagState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    question: str
    chat_id: str
    file_id: str | None
    retrieved_chunks: list[RetrievedChunk]
    context: str
    parsed_query: dict[str, Any]
    answer: str
    error: str | None


class QueryPlanDefaults:
    @staticmethod
    def build() -> dict[str, Any]:
        return {
            "query_type": "unknown",
            "domain": "general",
            "clean_question": "",
            "keywords": [],
            "entities": [],
            "section_hints": [],
            "retrieval_queries": [],
        }


class LlmJsonExtractor:
    def __init__(self, default_factory: QueryPlanDefaults) -> None:
        self.default_factory = default_factory

    def extract(self, response_text: str) -> dict[str, Any]:
        clean_text = (response_text or "").strip()
        json_text = self._find_json_object(clean_text)

        if not json_text:
            return self.default_factory.build()

        try:
            return json.loads(json_text)
        except json.JSONDecodeError:
            return self.default_factory.build()

    def _find_json_object(self, response_text: str) -> str | None:
        code_block_match = re.search(
            r"```(?:json)?\s*(\{.*?\})\s*```",
            response_text,
            flags=re.DOTALL,
        )
        if code_block_match:
            return code_block_match.group(1)

        json_match = re.search(r"\{.*\}", response_text, flags=re.DOTALL)
        if not json_match:
            return None

        return json_match.group(0)


class QueryPlanBuilder:
    def build_queries(
            self,
            original_question: str,
            query_plan: dict[str, Any],
    ) -> list[str]:
        queries: list[str] = []
        self._append_if_not_blank(queries, original_question)
        self._append_if_not_blank(queries, query_plan.get("clean_question"))
        self._append_retrieval_queries(queries, query_plan.get("retrieval_queries", []))
        self._append_combined_terms(queries, query_plan)
        return list(dict.fromkeys(queries))

    @staticmethod
    def _append_if_not_blank(target: list[str], value: Any) -> None:
        if isinstance(value, str):
            clean_value = value.strip()
            if clean_value:
                target.append(clean_value)

    def _append_retrieval_queries(self, target: list[str], queries: Any) -> None:
        if not isinstance(queries, list):
            return

        for query in queries:
            self._append_if_not_blank(target, query)

    def _append_combined_terms(
            self,
            target: list[str],
            query_plan: dict[str, Any],
    ) -> None:
        term_keys = ("keywords", "entities", "section_hints")
        combined_terms: list[str] = []

        for key in term_keys:
            values = query_plan.get(key, [])
            if not isinstance(values, list):
                continue
            for value in values:
                if isinstance(value, str) and value.strip():
                    combined_terms.append(value.strip())

        if combined_terms:
            target.append(" ".join(combined_terms))


class RetrievalContextFormatter:
    @staticmethod
    def format(chunks: list[dict[str, Any]]) -> str:
        formatted_chunks: list[str] = []

        for index, chunk in enumerate(chunks, start=1):
            metadata = chunk.get("metadata", {})
            page_no = metadata.get("page_no")
            source = metadata.get("source") or metadata.get("source_ref")

            formatted_chunks.append(
                f"""
[Chunk {index}]
Source: {source}
Page: {page_no}

Text:
{chunk.get("text", "")}
""".strip()
            )

        return "\n\n---\n\n".join(formatted_chunks)


class QueryParsingService:
    def __init__(
            self,
            llm: ChatOllama,
            prompts: AgentPrompts,
            json_extractor: LlmJsonExtractor,
    ) -> None:
        self.llm = llm
        self.prompts = prompts
        self.json_extractor = json_extractor

    def parse(self, question: str) -> dict[str, Any]:
        messages = [
            SystemMessage(content=self.prompts.build_query_generation_system_prompt()),
            HumanMessage(content=self.prompts.build_query_generation_user_prompt(question)),
        ]
        response = self.llm.invoke(messages)
        return self.json_extractor.extract(response.content)


class RetrievalService:
    def __init__(
            self,
            retriever: QdrantRagRetriever,
            query_plan_builder: QueryPlanBuilder,
            context_formatter: RetrievalContextFormatter,
    ) -> None:
        self.retriever = retriever
        self.query_plan_builder = query_plan_builder
        self.context_formatter = context_formatter

    def retrieve(
            self,
            question: str,
            parsed_query: dict[str, Any],
            chat_id: str,
    ) -> tuple[list[dict[str, Any]], str]:
        queries = self.query_plan_builder.build_queries(question, parsed_query)
        chunks = self.retriever.retrieve_many(
            queries,
            chat_id=chat_id,
            limit_per_query=100,
            final_limit=100,
        )
        context = self.context_formatter.format(chunks)
        return chunks, context


class AnswerGenerationService:
    def __init__(self, llm: ChatOllama, prompts: AgentPrompts) -> None:
        self.llm = llm
        self.prompts = prompts

    def generate(self, question: str, context: str, history: list[BaseMessage]) -> str | list[str | Any]:
        if not context.strip():
            return "I could not find this in the uploaded document."

        messages = [
            SystemMessage(content=self.prompts.build_system_prompt()),
            *history[-8:],
            HumanMessage(content=self.prompts.build_answer_prompt(question, context)),
        ]
        response = self.llm.invoke(messages)
        return response.content


class RagGraphFactory:
    def __init__(
            self,
            document_summary: list[dict[str, Any]],
            document_title: str,
            retriever: QdrantRagRetriever,
            llm_model: str = "llama3.1:8b",
            ollama_base_url: str = "http://localhost:11434",
    ) -> None:
        self.prompts = AgentPrompts(document_summary, document_title)
        self.llm = ChatOllama(
            model=llm_model,
            base_url=ollama_base_url,
            temperature=0,
            num_ctx=32768,
            disable_streaming=False,
        )

        self.query_parsing_service = QueryParsingService(
            llm=self.llm,
            prompts=self.prompts,
            json_extractor=LlmJsonExtractor(default_factory=QueryPlanDefaults()),
        )
        self.retrieval_service = RetrievalService(
            retriever=retriever,
            query_plan_builder=QueryPlanBuilder(),
            context_formatter=RetrievalContextFormatter(),
        )
        self.answer_generation_service = AnswerGenerationService(
            llm=self.llm,
            prompts=self.prompts,
        )

    def build(self):
        graph = StateGraph(RagState)
        graph.add_node("query_parsing", self.query_parsing)
        graph.add_node("retrieve", self.retrieve_node)
        graph.add_node("generate_answer", self.generate_answer_node)
        graph.set_entry_point("query_parsing")
        graph.add_edge("query_parsing", "retrieve")
        graph.add_edge("retrieve", "generate_answer")
        graph.add_edge("generate_answer", END)
        return graph.compile(checkpointer=checkpointer)

    def query_parsing(self, state: RagState) -> RagState:
        parsed_query = self.query_parsing_service.parse(state["question"])
        return {**state, "parsed_query": parsed_query}

    def retrieve_node(self, state: RagState) -> RagState:
        parsed_query = state.get("parsed_query") or QueryPlanDefaults.build()
        chunks, context = self.retrieval_service.retrieve(
            question=state["question"],
            parsed_query=parsed_query,
            chat_id=state["chat_id"],
        )
        return {
            **state,
            "retrieved_chunks": chunks,
            "context": context,
        }

    def generate_answer_node(self, state: RagState) -> RagState:
        history = state.get("messages", [])[:-1]
        answer = self.answer_generation_service.generate(
            question=state["question"],
            context=state.get("context", ""),
            history=history,
        )
        return {**state, "answer": answer}

    @staticmethod
    def format_context(chunks: list[dict]) -> str:
        return RetrievalContextFormatter.format(chunks)
