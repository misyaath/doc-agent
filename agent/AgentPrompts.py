from typing import Any

from agent.prompts import QUERY_GENERATION_SYSTEM_PROMPT, QUERY_GENERATION_USER_PROMPT, ANSWER_SYSTEM_PROMPT, \
    ANSWER_USER_PROMPT


class AgentPrompts:
    def __init__(self,
                 document_summary: list[dict[str, Any]],
                 document_title: str) -> None:

        self.document_summary: list[dict[str, Any]] = document_summary
        self.document_title: str = document_title

    def _format_document_summaries_for_prompt(self) -> str:

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
        return QUERY_GENERATION_SYSTEM_PROMPT

    def build_query_generation_user_prompt(self, question: str) -> str:
        return QUERY_GENERATION_USER_PROMPT.format(
            document_summary=self._format_document_summaries_for_prompt(),
            question=question,
        )

    def build_system_prompt(self) -> str:
        return ANSWER_SYSTEM_PROMPT.format(document_summaries=self._format_document_summaries_for_prompt())

    def build_answer_prompt(self, question: str, context: str) -> str:
        return ANSWER_USER_PROMPT.format(question=question, retrieved_context=context)
