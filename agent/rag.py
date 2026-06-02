import json
import re
from typing import Annotated, Any, TypedDict, cast

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph, add_messages

from agent.AgentPrompts import AgentPrompts
from agent.langgraph_memory import checkpointer
from agent.qdrant_retrieval import QdrantRagRetriever


class RetrievedChunk(TypedDict, total=False):
    """
    Retrieved Chunk.

    Purpose:
        Defines RetrievedChunk in the RAG agent layer that builds prompts, retrieves
            context, and generates answers.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.

    Attributes:
        id (str): Declared data field for this class.
        text (str): Declared data field for this class.
        score (float): Declared data field for this class.
        metadata (dict[str, Any]): Declared data field for this class.
    """

    id: str
    text: str
    score: float
    metadata: dict[str, Any]


class RagState(TypedDict, total=False):
    """
    Rag State.

    Purpose:
        Defines RagState in the RAG agent layer that builds prompts, retrieves context,
            and generates answers.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.

    Attributes:
        messages (Annotated[list[BaseMessage], add_messages]): Declared data field for
            this class.
        question (str): Declared data field for this class.
        chat_id (str): Declared data field for this class.
        file_id (str | None): Declared data field for this class.
        retrieved_chunks (list[RetrievedChunk]): Declared data field for this class.
        context (str): Declared data field for this class.
        parsed_query (dict[str, Any]): Declared data field for this class.
        answer (str): Declared data field for this class.
        error (str | None): Declared data field for this class.
    """

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
    """
    Query Plan Defaults.

    Purpose:
        Defines QueryPlanDefaults in the RAG agent layer that builds prompts, retrieves
            context, and generates answers.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.
    """

    @staticmethod
    def build() -> dict[str, Any]:
        """
        Build.

        Purpose:
            Implements build for the RAG agent layer that builds prompts, retrieves
                context, and generates answers.
        Class:
            Belongs to QueryPlanDefaults; uses that class state and dependencies when
                available.
        Args:
            None.
        Returns:
            dict[str, Any]: Structured data produced by the operation.
        Why Added:
            Centralizes this behavior inside QueryPlanDefaults so related code remains
                cohesive and testable.
        """
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
    """
    Llm Json Extractor.

    Purpose:
        Defines LlmJsonExtractor in the RAG agent layer that builds prompts, retrieves
            context, and generates answers.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.
    """

    def __init__(self, default_factory: QueryPlanDefaults) -> None:
        """
        Initialize the object with its required dependencies.

        Purpose:
            Implements __init__ for the RAG agent layer that builds prompts, retrieves
                context, and generates answers.
        Class:
            Belongs to LlmJsonExtractor; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
            default_factory (QueryPlanDefaults): Input value for the default factory
                parameter.
        Returns:
            None: Performs work through side effects and does not return a value.
        Why Added:
            Centralizes this behavior inside LlmJsonExtractor so related code remains
                cohesive and testable.
        """
        self.default_factory = default_factory

    def extract(self, response_text: str) -> dict[str, Any]:
        """
        Extract.

        Purpose:
            Implements extract for the RAG agent layer that builds prompts, retrieves
                context, and generates answers.
        Class:
            Belongs to LlmJsonExtractor; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
            response_text (str): Input value for the response text parameter.
        Returns:
            dict[str, Any]: Structured data produced by the operation.
        Why Added:
            Centralizes this behavior inside LlmJsonExtractor so related code remains
                cohesive and testable.
        """
        clean_text = (response_text or "").strip()
        json_text = self._find_json_object(clean_text)

        if not json_text:
            return self.default_factory.build()

        try:
            return json.loads(json_text)
        except json.JSONDecodeError:
            return self.default_factory.build()

    def _find_json_object(self, response_text: str) -> str | None:
        """
        Find json object.

        Purpose:
            Implements _find_json_object for the RAG agent layer that builds prompts,
                retrieves context, and generates answers.
        Class:
            Belongs to LlmJsonExtractor; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
            response_text (str): Input value for the response text parameter.
        Returns:
            str | None: Result produced by the operation.
        Why Added:
            Centralizes this behavior inside LlmJsonExtractor so related code remains
                cohesive and testable.
        """
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
    """
    Query Plan Builder.

    Purpose:
        Defines QueryPlanBuilder in the RAG agent layer that builds prompts, retrieves
            context, and generates answers.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.
    """

    def build_queries(
        self,
        original_question: str,
        query_plan: dict[str, Any],
    ) -> list[str]:
        """
        Build queries.

        Purpose:
            Implements build_queries for the RAG agent layer that builds prompts,
                retrieves context, and generates answers.
        Class:
            Belongs to QueryPlanBuilder; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
            original_question (str): Input value for the original question parameter.
            query_plan (dict[str, Any]): Input value for the query plan parameter.
        Returns:
            list[str]: Structured data produced by the operation.
        Why Added:
            Centralizes this behavior inside QueryPlanBuilder so related code remains
                cohesive and testable.
        """
        queries: list[str] = []
        self._append_if_not_blank(queries, original_question)
        self._append_if_not_blank(queries, query_plan.get("clean_question"))
        self._append_retrieval_queries(queries, query_plan.get("retrieval_queries", []))
        self._append_combined_terms(queries, query_plan)
        return list(dict.fromkeys(queries))

    @staticmethod
    def _append_if_not_blank(target: list[str], value: Any) -> None:
        """
        Append if not blank.

        Purpose:
            Implements _append_if_not_blank for the RAG agent layer that builds prompts,
                retrieves context, and generates answers.
        Class:
            Belongs to QueryPlanBuilder; uses that class state and dependencies when
                available.
        Args:
            target (list[str]): Input value for the target parameter.
            value (Any): Raw value being validated, normalized, or transformed.
        Returns:
            None: Performs work through side effects and does not return a value.
        Why Added:
            Centralizes this behavior inside QueryPlanBuilder so related code remains
                cohesive and testable.
        """
        if isinstance(value, str):
            clean_value = value.strip()
            if clean_value:
                target.append(clean_value)

    def _append_retrieval_queries(self, target: list[str], queries: Any) -> None:
        """
        Append retrieval queries.

        Purpose:
            Implements _append_retrieval_queries for the RAG agent layer that builds
                prompts, retrieves context, and generates answers.
        Class:
            Belongs to QueryPlanBuilder; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
            target (list[str]): Input value for the target parameter.
            queries (Any): Input value for the queries parameter.
        Returns:
            None: Performs work through side effects and does not return a value.
        Why Added:
            Centralizes this behavior inside QueryPlanBuilder so related code remains
                cohesive and testable.
        """
        if not isinstance(queries, list):
            return

        for query in queries:
            self._append_if_not_blank(target, query)

    def _append_combined_terms(
        self,
        target: list[str],
        query_plan: dict[str, Any],
    ) -> None:
        """
        Append combined terms.

        Purpose:
            Implements _append_combined_terms for the RAG agent layer that builds
                prompts, retrieves context, and generates answers.
        Class:
            Belongs to QueryPlanBuilder; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
            target (list[str]): Input value for the target parameter.
            query_plan (dict[str, Any]): Input value for the query plan parameter.
        Returns:
            None: Performs work through side effects and does not return a value.
        Why Added:
            Centralizes this behavior inside QueryPlanBuilder so related code remains
                cohesive and testable.
        """
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
    """
    Retrieval Context Formatter.

    Purpose:
        Defines RetrievalContextFormatter in the RAG agent layer that builds prompts,
            retrieves context, and generates answers.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.
    """

    @staticmethod
    def format(chunks: list[dict[str, Any]]) -> str:
        """
        Format.

        Purpose:
            Implements format for the RAG agent layer that builds prompts, retrieves
                context, and generates answers.
        Class:
            Belongs to RetrievalContextFormatter; uses that class state and dependencies
                when available.
        Args:
            chunks (list[dict[str, Any]]): Input value for the chunks parameter.
        Returns:
            str: Result produced by the operation.
        Why Added:
            Centralizes this behavior inside RetrievalContextFormatter so related code
                remains cohesive and testable.
        """
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
    """
    Query Parsing Service.

    Purpose:
        Defines QueryParsingService in the RAG agent layer that builds prompts,
            retrieves context, and generates answers.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.
    """

    def __init__(
        self,
        llm: ChatOllama,
        prompts: AgentPrompts,
        json_extractor: LlmJsonExtractor,
    ) -> None:
        """
        Initialize the object with its required dependencies.

        Purpose:
            Implements __init__ for the RAG agent layer that builds prompts, retrieves
                context, and generates answers.
        Class:
            Belongs to QueryParsingService; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
            llm (ChatOllama): Input value for the llm parameter.
            prompts (AgentPrompts): Input value for the prompts parameter.
            json_extractor (LlmJsonExtractor): Input value for the json extractor
                parameter.
        Returns:
            None: Performs work through side effects and does not return a value.
        Why Added:
            Centralizes this behavior inside QueryParsingService so related code remains
                cohesive and testable.
        """
        self.llm = llm
        self.prompts = prompts
        self.json_extractor = json_extractor

    def parse(self, question: str) -> dict[str, Any]:
        """
        Parse.

        Purpose:
            Implements parse for the RAG agent layer that builds prompts, retrieves
                context, and generates answers.
        Class:
            Belongs to QueryParsingService; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
            question (str): Input value for the question parameter.
        Returns:
            dict[str, Any]: Structured data produced by the operation.
        Why Added:
            Centralizes this behavior inside QueryParsingService so related code remains
                cohesive and testable.
        """
        messages = [
            SystemMessage(content=self.prompts.build_query_generation_system_prompt()),
            HumanMessage(content=self.prompts.build_query_generation_user_prompt(question)),
        ]
        response = self.llm.invoke(messages)
        content = (
            response.content if isinstance(response.content, str) else json.dumps(response.content, ensure_ascii=False)
        )
        return self.json_extractor.extract(content)


class RetrievalService:
    """
    Retrieval Service.

    Purpose:
        Defines RetrievalService in the RAG agent layer that builds prompts, retrieves
            context, and generates answers.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.
    """

    def __init__(
        self,
        retriever: QdrantRagRetriever,
        query_plan_builder: QueryPlanBuilder,
        context_formatter: RetrievalContextFormatter,
    ) -> None:
        """
        Initialize the object with its required dependencies.

        Purpose:
            Implements __init__ for the RAG agent layer that builds prompts, retrieves
                context, and generates answers.
        Class:
            Belongs to RetrievalService; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
            retriever (QdrantRagRetriever): Input value for the retriever parameter.
            query_plan_builder (QueryPlanBuilder): Input value for the query plan
                builder parameter.
            context_formatter (RetrievalContextFormatter): Input value for the context
                formatter parameter.
        Returns:
            None: Performs work through side effects and does not return a value.
        Why Added:
            Centralizes this behavior inside RetrievalService so related code remains
                cohesive and testable.
        """
        self.retriever = retriever
        self.query_plan_builder = query_plan_builder
        self.context_formatter = context_formatter

    def retrieve(
        self,
        question: str,
        parsed_query: dict[str, Any],
        chat_id: str,
    ) -> tuple[list[RetrievedChunk], str]:
        """
        Retrieve.

        Purpose:
            Implements retrieve for the RAG agent layer that builds prompts, retrieves
                context, and generates answers.
        Class:
            Belongs to RetrievalService; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
            question (str): Input value for the question parameter.
            parsed_query (dict[str, Any]): Input value for the parsed query parameter.
            chat_id (str): Chat/session identifier used to scope retrieval, tasks, or
                responses.
        Returns:
            tuple[list[dict[str, Any]], str]: Structured data produced by the operation.
        Why Added:
            Centralizes this behavior inside RetrievalService so related code remains
                cohesive and testable.
        """
        queries = self.query_plan_builder.build_queries(question, parsed_query)
        chunks = self.retriever.retrieve_many(
            queries,
            chat_id=chat_id,
            limit_per_query=100,
            final_limit=100,
        )
        context = self.context_formatter.format(chunks)
        return cast(list[RetrievedChunk], chunks), context


class AnswerGenerationService:
    """
    Answer Generation Service.

    Purpose:
        Defines AnswerGenerationService in the RAG agent layer that builds prompts,
            retrieves context, and generates answers.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.
    """

    def __init__(self, llm: ChatOllama, prompts: AgentPrompts) -> None:
        """
        Initialize the object with its required dependencies.

        Purpose:
            Implements __init__ for the RAG agent layer that builds prompts, retrieves
                context, and generates answers.
        Class:
            Belongs to AnswerGenerationService; uses that class state and dependencies
                when available.
        Args:
            self (Self): Current instance that owns the operation state.
            llm (ChatOllama): Input value for the llm parameter.
            prompts (AgentPrompts): Input value for the prompts parameter.
        Returns:
            None: Performs work through side effects and does not return a value.
        Why Added:
            Centralizes this behavior inside AnswerGenerationService so related code
                remains cohesive and testable.
        """
        self.llm = llm
        self.prompts = prompts

    def generate(self, question: str, context: str, history: list[BaseMessage]) -> str:
        """
        Generate.

        Purpose:
            Implements generate for the RAG agent layer that builds prompts, retrieves
                context, and generates answers.
        Class:
            Belongs to AnswerGenerationService; uses that class state and dependencies
                when available.
        Args:
            self (Self): Current instance that owns the operation state.
            question (str): Input value for the question parameter.
            context (str): Input value for the context parameter.
            history (list[BaseMessage]): Input value for the history parameter.
        Returns:
            str | list[str | Any]: Structured data produced by the operation.
        Why Added:
            Centralizes this behavior inside AnswerGenerationService so related code
                remains cohesive and testable.
        """
        if not context.strip():
            return "I could not find this in the uploaded document."

        messages = [
            SystemMessage(content=self.prompts.build_system_prompt()),
            *history[-8:],
            HumanMessage(content=self.prompts.build_answer_prompt(question, context)),
        ]
        response = self.llm.invoke(messages)
        if isinstance(response.content, str):
            return response.content
        return json.dumps(response.content, ensure_ascii=False)


class RagGraphFactory:
    """
    Rag Graph Factory.

    Purpose:
        Defines RagGraphFactory in the RAG agent layer that builds prompts, retrieves
            context, and generates answers.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.
    """

    def __init__(
        self,
        document_summary: list[dict[str, Any]],
        document_title: str,
        retriever: QdrantRagRetriever,
        llm_model: str = "llama3.1:8b",
        ollama_base_url: str = "http://localhost:11434",
    ) -> None:
        """
        Initialize the object with its required dependencies.

        Purpose:
            Implements __init__ for the RAG agent layer that builds prompts, retrieves
                context, and generates answers.
        Class:
            Belongs to RagGraphFactory; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
            document_summary (list[dict[str, Any]]): Input value for the document
                summary parameter.
            document_title (str): Input value for the document title parameter.
            retriever (QdrantRagRetriever): Input value for the retriever parameter.
            llm_model (str): Input value for the llm model parameter.
            ollama_base_url (str): Input value for the ollama base url parameter.
        Returns:
            None: Performs work through side effects and does not return a value.
        Why Added:
            Centralizes this behavior inside RagGraphFactory so related code remains
                cohesive and testable.
        """
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

    def build(self) -> Any:
        """
        Build.

        Purpose:
            Implements build for the RAG agent layer that builds prompts, retrieves
                context, and generates answers.
        Class:
            Belongs to RagGraphFactory; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
        Returns:
            Any: Result produced by the operation.
        Why Added:
            Centralizes this behavior inside RagGraphFactory so related code remains
                cohesive and testable.
        """
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
        """
        Query parsing.

        Purpose:
            Implements query_parsing for the RAG agent layer that builds prompts,
                retrieves context, and generates answers.
        Class:
            Belongs to RagGraphFactory; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
            state (RagState): Input value for the state parameter.
        Returns:
            RagState: Result produced by the operation.
        Why Added:
            Centralizes this behavior inside RagGraphFactory so related code remains
                cohesive and testable.
        """
        parsed_query = self.query_parsing_service.parse(state["question"])
        return {**state, "parsed_query": parsed_query}

    def retrieve_node(self, state: RagState) -> RagState:
        """
        Retrieve node.

        Purpose:
            Implements retrieve_node for the RAG agent layer that builds prompts,
                retrieves context, and generates answers.
        Class:
            Belongs to RagGraphFactory; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
            state (RagState): Input value for the state parameter.
        Returns:
            RagState: Result produced by the operation.
        Why Added:
            Centralizes this behavior inside RagGraphFactory so related code remains
                cohesive and testable.
        """
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
        """
        Generate answer node.

        Purpose:
            Implements generate_answer_node for the RAG agent layer that builds prompts,
                retrieves context, and generates answers.
        Class:
            Belongs to RagGraphFactory; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
            state (RagState): Input value for the state parameter.
        Returns:
            RagState: Result produced by the operation.
        Why Added:
            Centralizes this behavior inside RagGraphFactory so related code remains
                cohesive and testable.
        """
        history = state.get("messages", [])[:-1]
        answer = self.answer_generation_service.generate(
            question=state["question"],
            context=state.get("context", ""),
            history=history,
        )
        return {**state, "answer": answer}

    @staticmethod
    def format_context(chunks: list[dict]) -> str:
        """
        Format context.

        Purpose:
            Implements format_context for the RAG agent layer that builds prompts,
                retrieves context, and generates answers.
        Class:
            Belongs to RagGraphFactory; uses that class state and dependencies when
                available.
        Args:
            chunks (list[dict]): Input value for the chunks parameter.
        Returns:
            str: Result produced by the operation.
        Why Added:
            Centralizes this behavior inside RagGraphFactory so related code remains
                cohesive and testable.
        """
        return RetrievalContextFormatter.format(chunks)
