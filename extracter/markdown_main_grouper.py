from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MarkdownMainSection:
    """
    Markdown Main Section.

    Purpose:
        Defines MarkdownMainSection in the document extraction pipeline that normalizes
            PDFs, enriches visual content, builds RAG units, and indexes data.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.

    Attributes:
        heading (str): Declared data field for this class.
        heading_level (int): Declared data field for this class.
        text (str): Declared data field for this class.
        order (int): Declared data field for this class.
    """

    heading: str
    heading_level: int
    text: str
    order: int


class MarkdownMainHeadingGrouper:
    """
    Markdown Main Heading Grouper.

    Purpose:
        Defines MarkdownMainHeadingGrouper in the document extraction pipeline that
            normalizes PDFs, enriches visual content, builds RAG units, and indexes
            data.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.

    Attributes:
        HEADING_RE (Any): Class-level value used by this class.
        MAIN_NUMBERED_HEADING_RE (Any): Class-level value used by this class.
        SUB_NUMBERED_HEADING_RE (Any): Class-level value used by this class.
    """

    HEADING_RE = re.compile(r"^(?P<hashes>#{1,6})\s+(?P<title>.+?)\s*$")
    MAIN_NUMBERED_HEADING_RE = re.compile(r"^\d+\s+.+")
    SUB_NUMBERED_HEADING_RE = re.compile(r"^\d+\.\d+.*")

    def group(self, markdown_text: str) -> list[MarkdownMainSection]:
        """
        Group.

        Purpose:
            Implements group for the document extraction pipeline that normalizes PDFs,
                enriches visual content, builds RAG units, and indexes data.
        Class:
            Belongs to MarkdownMainHeadingGrouper; uses that class state and
                dependencies when available.
        Args:
            self (Self): Current instance that owns the operation state.
            markdown_text (str): Input value for the markdown text parameter.
        Returns:
            list[MarkdownMainSection]: Structured data produced by the operation.
        Why Added:
            Centralizes this behavior inside MarkdownMainHeadingGrouper so related code
                remains cohesive and testable.
        """
        sections: list[MarkdownMainSection] = []

        current_heading = "Document Start"
        current_level = 0
        current_lines: list[str] = []
        order = 0

        for line in markdown_text.splitlines():
            heading_match = self.HEADING_RE.match(line.strip())

            if not heading_match:
                current_lines.append(line)
                continue

            heading_level = len(heading_match.group("hashes"))
            heading_title = heading_match.group("title").strip()

            is_main_heading = self._is_main_heading(heading_title)

            if is_main_heading:
                # Save previous main section.
                if current_lines:
                    section_text = "\n".join(current_lines).strip()

                    if section_text:
                        sections.append(
                            MarkdownMainSection(
                                heading=current_heading,
                                heading_level=current_level,
                                text=section_text,
                                order=order,
                            )
                        )
                        order += 1

                current_heading = heading_title
                current_level = heading_level
                current_lines = [line]
            else:
                # Keep subheading inside current main section.
                current_lines.append(line)

        # Save last section.
        if current_lines:
            section_text = "\n".join(current_lines).strip()

            if section_text:
                sections.append(
                    MarkdownMainSection(
                        heading=current_heading,
                        heading_level=current_level,
                        text=section_text,
                        order=order,
                    )
                )

        return sections

    def _is_main_heading(self, title: str) -> bool:
        """
        Is main heading.

        Purpose:
            Implements _is_main_heading for the document extraction pipeline that
                normalizes PDFs, enriches visual content, builds RAG units, and indexes
                data.
        Class:
            Belongs to MarkdownMainHeadingGrouper; uses that class state and
                dependencies when available.
        Args:
            self (Self): Current instance that owns the operation state.
            title (str): Input value for the title parameter.
        Returns:
            bool: True when the condition is satisfied; otherwise False.
        Why Added:
            Centralizes this behavior inside MarkdownMainHeadingGrouper so related code
                remains cohesive and testable.
        """
        title = title.strip()

        # 1 Introduction, 2 Method, 3 Experiments
        if self.MAIN_NUMBERED_HEADING_RE.match(title):
            # Exclude 2.1, 3.4, A.1 etc.
            return not self.SUB_NUMBERED_HEADING_RE.match(title)

        # Allow document title as first section if needed.
        return False


class MarkdownMainHeadingProcessor:
    """
    Markdown Main Heading Processor.

    Purpose:
        Defines MarkdownMainHeadingProcessor in the document extraction pipeline that
            normalizes PDFs, enriches visual content, builds RAG units, and indexes
            data.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.
    """

    def __init__(
        self,
        grouper: MarkdownMainHeadingGrouper | None = None,
    ) -> None:
        """
        Initialize the object with its required dependencies.

        Purpose:
            Implements __init__ for the document extraction pipeline that normalizes
                PDFs, enriches visual content, builds RAG units, and indexes data.
        Class:
            Belongs to MarkdownMainHeadingProcessor; uses that class state and
                dependencies when available.
        Args:
            self (Self): Current instance that owns the operation state.
            grouper (MarkdownMainHeadingGrouper | None): Input value for the grouper
                parameter.
        Returns:
            None: Performs work through side effects and does not return a value.
        Why Added:
            Centralizes this behavior inside MarkdownMainHeadingProcessor so related
                code remains cohesive and testable.
        """
        self._grouper = grouper or MarkdownMainHeadingGrouper()

    def process_file(self, markdown_path: str | Path) -> list[dict]:
        """
        Process file.

        Purpose:
            Implements process_file for the document extraction pipeline that normalizes
                PDFs, enriches visual content, builds RAG units, and indexes data.
        Class:
            Belongs to MarkdownMainHeadingProcessor; uses that class state and
                dependencies when available.
        Args:
            self (Self): Current instance that owns the operation state.
            markdown_path (str | Path): Input value for the markdown path parameter.
        Returns:
            list[dict]: Structured data produced by the operation.
        Why Added:
            Centralizes this behavior inside MarkdownMainHeadingProcessor so related
                code remains cohesive and testable.
        """
        markdown_path = Path(markdown_path)

        if not markdown_path.exists():
            raise FileNotFoundError(f"Markdown file not found: {markdown_path}")

        markdown_text = markdown_path.read_text(encoding="utf-8")

        sections = self._grouper.group(markdown_text)

        return [
            {
                "id": f"main_section_{section.order:04d}",
                "order": section.order,
                "heading": section.heading,
                "heading_level": section.heading_level,
                "text": section.text,
                "chunking_strategy": "markdown_main_heading_group_v1",
            }
            for section in sections
        ]
