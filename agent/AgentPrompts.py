from typing import Any

from agent.prompts import (
    ANSWER_SYSTEM_PROMPT,
    ANSWER_USER_PROMPT,
    QUERY_GENERATION_SYSTEM_PROMPT,
    QUERY_GENERATION_USER_PROMPT,
)


class AgentPrompts:
    """
    Agent Prompts.

    Purpose:
        Defines AgentPrompts in the RAG agent layer that builds prompts, retrieves
            context, and generates answers.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.
    """

    def __init__(self, document_summary: list[dict[str, Any]], document_title: str) -> None:
        """
        Initialize the object with its required dependencies.

        Purpose:
            Implements __init__ for the RAG agent layer that builds prompts, retrieves
                context, and generates answers.
        Class:
            Belongs to AgentPrompts; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
            document_summary (list[dict[str, Any]]): Input value for the document
                summary parameter.
            document_title (str): Input value for the document title parameter.
        Returns:
            None: Performs work through side effects and does not return a value.
        Why Added:
            Centralizes this behavior inside AgentPrompts so related code remains
                cohesive and testable.
        """
        self.document_summary: list[dict[str, Any]] = document_summary
        self.document_title: str = document_title

    def _format_document_summaries_for_prompt(self) -> str:
        """
        Format document summaries for prompt.

        Purpose:
            Implements _format_document_summaries_for_prompt for the RAG agent layer
                that builds prompts, retrieves context, and generates answers.
        Class:
            Belongs to AgentPrompts; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
        Returns:
            str: Result produced by the operation.
        Why Added:
            Centralizes this behavior inside AgentPrompts so related code remains
                cohesive and testable.
        """
        if not self.document_summary:
            return "No document summaries available."

        parts: list[str] = [f"{self.document_title}\n\n"]

        for doc in self.document_summary:
            document_title, sections = self._extract_title_and_sections(doc)
            parts.append(f"\nDocument Title: {document_title}")

            if isinstance(sections, dict):
                for section_title, summary in sections.items():
                    parts.append(
                        f"""Section: {section_title}\nSummary: {summary}
                        """.strip()
                    )
            elif isinstance(sections, list):
                for index, item in enumerate(sections, start=1):
                    parts.append(f"Summary {index}: {item}")
            else:
                parts.append(f"Summary: {sections}")

            parts.append("\n---")

        return "\n\n".join(parts)

    @staticmethod
    def _extract_title_and_sections(doc: dict[str, Any]) -> tuple[str, Any]:
        # Raw DB/cache shape: {"file_id": "...", "title": "...", "summary": ...}
        """
        Extract title and sections.

        Purpose:
            Implements _extract_title_and_sections for the RAG agent layer that builds
                prompts, retrieves context, and generates answers.
        Class:
            Belongs to AgentPrompts; uses that class state and dependencies when
                available.
        Args:
            doc (dict[str, Any]): Docling document object produced by PDF conversion.
        Returns:
            tuple[str, Any]: Structured data produced by the operation.
        Why Added:
            Centralizes this behavior inside AgentPrompts so related code remains
                cohesive and testable.
        """
        if "summary" in doc and ("title" in doc or "file_id" in doc):
            title = str(doc.get("title") or doc.get("file_id") or "Untitled document").strip()
            sections = doc.get("summary") or {}
            return title, sections

        # Legacy shape: {"Document Title": {...}}
        if len(doc) == 1:
            key, value = next(iter(doc.items()))
            return str(key), value

        return "Untitled document", doc

    def build_query_generation_system_prompt(self) -> str:
        """
        Build query generation system prompt.

        Purpose:
            Implements build_query_generation_system_prompt for the RAG agent layer that
                builds prompts, retrieves context, and generates answers.
        Class:
            Belongs to AgentPrompts; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
        Returns:
            str: Result produced by the operation.
        Why Added:
            Centralizes this behavior inside AgentPrompts so related code remains
                cohesive and testable.
        """
        return QUERY_GENERATION_SYSTEM_PROMPT

    def build_query_generation_user_prompt(self, question: str) -> str:
        """
        Build query generation user prompt.

        Purpose:
            Implements build_query_generation_user_prompt for the RAG agent layer that
                builds prompts, retrieves context, and generates answers.
        Class:
            Belongs to AgentPrompts; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
            question (str): Input value for the question parameter.
        Returns:
            str: Result produced by the operation.
        Why Added:
            Centralizes this behavior inside AgentPrompts so related code remains
                cohesive and testable.
        """
        return QUERY_GENERATION_USER_PROMPT.format(
            document_summary=self._format_document_summaries_for_prompt(),
            question=question,
        )

    def build_system_prompt(self) -> str:
        """
        Build system prompt.

        Purpose:
            Implements build_system_prompt for the RAG agent layer that builds prompts,
                retrieves context, and generates answers.
        Class:
            Belongs to AgentPrompts; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
        Returns:
            str: Result produced by the operation.
        Why Added:
            Centralizes this behavior inside AgentPrompts so related code remains
                cohesive and testable.
        """
        return ANSWER_SYSTEM_PROMPT.format(document_summaries=self._format_document_summaries_for_prompt())

    def build_answer_prompt(self, question: str, context: str) -> str:
        """
        Build answer prompt.

        Purpose:
            Implements build_answer_prompt for the RAG agent layer that builds prompts,
                retrieves context, and generates answers.
        Class:
            Belongs to AgentPrompts; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
            question (str): Input value for the question parameter.
            context (str): Input value for the context parameter.
        Returns:
            str: Result produced by the operation.
        Why Added:
            Centralizes this behavior inside AgentPrompts so related code remains
                cohesive and testable.
        """
        return ANSWER_USER_PROMPT.format(question=question, retrieved_context=context)
