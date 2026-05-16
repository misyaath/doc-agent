from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MarkdownMainSection:
    heading: str
    heading_level: int
    text: str
    order: int


class MarkdownMainHeadingGrouper:
    """
    Groups markdown by main paper headings.

    Main headings:
        ## 1 Introduction
        ## 2 Method
        ## 3 Experiments

    Subheadings stay inside the current main heading:
        ## 2.1 Preliminary
        ## 2.2 Document Understanding Transformer
    """

    HEADING_RE = re.compile(r"^(?P<hashes>#{1,6})\s+(?P<title>.+?)\s*$")
    MAIN_NUMBERED_HEADING_RE = re.compile(r"^\d+\s+.+")
    SUB_NUMBERED_HEADING_RE = re.compile(r"^\d+\.\d+.*")

    def group(self, markdown_text: str) -> list[MarkdownMainSection]:
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
        title = title.strip()

        # 1 Introduction, 2 Method, 3 Experiments
        if self.MAIN_NUMBERED_HEADING_RE.match(title):
            # Exclude 2.1, 3.4, A.1 etc.
            if self.SUB_NUMBERED_HEADING_RE.match(title):
                return False
            return True

        # Allow document title as first section if needed.
        return False


class MarkdownMainHeadingProcessor:
    def __init__(
            self,
            grouper: MarkdownMainHeadingGrouper | None = None,
    ) -> None:
        self._grouper = grouper or MarkdownMainHeadingGrouper()

    def process_file(self, markdown_path: str | Path) -> list[dict]:
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
