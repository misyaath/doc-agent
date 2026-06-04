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

    def __init__(self, document_summary: list[dict[str, Any]]) -> None:
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
        Returns:
            None: Performs work through side effects and does not return a value.
        Why Added:
            Centralizes this behavior inside AgentPrompts so related code remains
                cohesive and testable.
        """
        self.document_summary: list[dict[str, Any]] = document_summary

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

        parts: list[str] = []

        def append_summary_block(title: str, summary: Any, parent_title: str | None = None) -> None:
            if summary is None:
                return

            normalized_title = title.strip()
            normalized_parent = (parent_title or "").strip()
            should_add_title = normalized_title and normalized_title != normalized_parent

            if isinstance(summary, dict):
                if should_add_title:
                    parts.append(normalized_title)
                for section_title, section_summary in summary.items():
                    append_summary_block(str(section_title), section_summary, normalized_title)
                return

            if isinstance(summary, list):
                if should_add_title:
                    parts.append(normalized_title)
                for item in summary:
                    if item is None:
                        continue
                    if isinstance(item, dict):
                        for item_title, item_summary in item.items():
                            append_summary_block(str(item_title), item_summary, normalized_title)
                    else:
                        parts.append(str(item))
                return

            if should_add_title:
                parts.append(normalized_title)
            parts.append(str(summary).strip())

        for doc in self.document_summary:
            document_title, sections = self._extract_title_and_sections(doc)
            if sections is None:
                continue

            append_summary_block(document_title, sections)

        if not parts:
            return "No document summaries available."

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
            sections = doc.get("summary")
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
