from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


# ============================================================
# Config
# ============================================================

@dataclass(frozen=True)
class SectionSummaryConfig:
    ollama_url: str = "http://localhost:11434"
    model_name: str = "llama3.1:8b"
    target_words: int = 25
    temperature: float = 0.0


# ============================================================
# Loader
# ============================================================

class RagChunkLoader:
    def load(self, path: str | Path) -> list[dict[str, Any]]:
        file_path = Path(path)

        if not file_path.exists():
            raise FileNotFoundError(f"Chunks file not found: {file_path}")

        data = json.loads(file_path.read_text(encoding="utf-8"))

        if not isinstance(data, list):
            raise ValueError("Chunks JSON must be a list.")

        return data


# ============================================================
# Markdown Section Parser
# ============================================================

@dataclass(frozen=True)
class ParsedSubSection:
    heading: str
    text: str


class MarkdownSubSectionParser:
    """
    Splits one main chunk text into subheading sections.

    Example:
        ## 2 Method
        ## 2.1 Preliminary
        text...
        ## 2.2 Document Understanding Transformer
        text...

    Output:
        [
          ParsedSubSection("2.1 Preliminary", "..."),
          ParsedSubSection("2.2 Document Understanding Transformer", "...")
        ]
    """

    HEADING_RE = re.compile(r"^(?P<hashes>#{1,6})\s+(?P<title>.+?)\s*$")

    def parse(
            self,
            main_heading: str,
            markdown_text: str,
    ) -> list[ParsedSubSection]:
        lines = markdown_text.splitlines()

        sections: list[ParsedSubSection] = []
        current_heading: str | None = None
        current_lines: list[str] = []

        for line in lines:
            match = self.HEADING_RE.match(line.strip())

            if match:
                heading_title = match.group("title").strip()

                # Skip the main heading itself.
                if heading_title == main_heading:
                    if current_heading is None:
                        current_heading = main_heading
                    current_lines.append(line)
                    continue

                # Save previous subsection.
                if current_heading and current_lines:
                    section_text = "\n".join(current_lines).strip()
                    if section_text:
                        sections.append(
                            ParsedSubSection(
                                heading=current_heading,
                                text=section_text,
                            )
                        )

                current_heading = heading_title
                current_lines = [line]
            else:
                current_lines.append(line)

        if current_heading and current_lines:
            section_text = "\n".join(current_lines).strip()
            if section_text:
                sections.append(
                    ParsedSubSection(
                        heading=current_heading,
                        text=section_text,
                    )
                )

        # If no subheadings found, summarize whole chunk under main heading.
        if not sections:
            clean_text = markdown_text.strip()
            if clean_text:
                return [
                    ParsedSubSection(
                        heading=main_heading,
                        text=clean_text,
                    )
                ]

        return sections


# ============================================================
# Ollama Client
# ============================================================

class OllamaChatClient:
    def __init__(self, config: SectionSummaryConfig) -> None:
        self._config = config

    def generate(self, prompt: str) -> str:
        url = f"{self._config.ollama_url.rstrip('/')}/api/chat"

        payload = {
            "model": self._config.model_name,
            "stream": False,
            "options": {
                "temperature": self._config.temperature,
            },
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a precise document summarizer for a RAG system. "
                        "Use only the provided section text. Do not invent facts."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        }

        response = requests.post(url, json=payload, timeout=180)
        response.raise_for_status()

        data = response.json()
        message = data.get("message") or {}

        return str(message.get("content") or "").strip()


# ============================================================
# Summarizer
# ============================================================

class SectionSummarizer:
    def __init__(
            self,
            ollama_client: OllamaChatClient,
            config: SectionSummaryConfig,
    ) -> None:
        self._ollama_client = ollama_client
        self._config = config

    def summarize(
            self,
            main_heading: str,
            sub_heading: str,
            section_text: str,
    ) -> str:
        prompt = f"""
Summarize the following PDF/document section for a RAG system.

Main heading:
{main_heading}

Sub heading:
{sub_heading}

Section text:
{section_text}

Requirements:
- Write around {self._config.target_words} words.
- Preserve the most important information from the section.
- Keep important names, entities, dates, numbers, definitions, claims, requirements, rules, results, examples, and conclusions when present.
- If the section contains financial, legal, medical, technical, academic, business, or policy content, preserve the meaning accurately.
- If the section includes image vision analysis, charts, tables, figures, or extracted table text, summarize the important meaning from them.
- Do not add information that is not present in the section.
- Do not give advice, opinions, diagnosis, legal interpretation, or financial recommendation.
- Return only the summary text.
- Do not return JSON.
- Do not use bullet points unless the section itself is mainly a list or requirements.
""".strip()

        return self._ollama_client.generate(prompt)


# ============================================================
# Nested Summary Builder
# ============================================================

class NestedSectionSummaryBuilder:
    """
    Converts final chunks into nested summary JSON:

    [
      {
        "1 Introduction": {
          "1 Introduction": "summary..."
        }
      },
      {
        "2 Method": {
          "2.1 Preliminary: background": "summary...",
          "2.2 Document Understanding Transformer": "summary..."
        }
      }
    ]
    """

    def __init__(
            self,
            parser: MarkdownSubSectionParser,
            summarizer: SectionSummarizer,
    ) -> None:
        self._parser = parser
        self._summarizer = summarizer

    def build(self, chunks: list[dict[str, Any]]) -> list[dict[str, dict[str, str]]]:
        result: list[dict[str, dict[str, str]]] = []

        sorted_chunks = sorted(
            chunks,
            key=lambda item: item.get("order", item.get("chunk_index", 0)),
        )

        for chunk in sorted_chunks:
            main_heading = self._clean_heading(
                chunk.get("heading") or "Untitled Section"
            )

            text = (chunk.get("text") or "").strip()

            if not text:
                continue

            sub_sections = self._parser.parse(
                main_heading=main_heading,
                markdown_text=text,
            )

            nested_section: dict[str, str] = {}

            for sub_section in sub_sections:
                sub_heading = self._clean_heading(sub_section.heading)

                print(f"Summarizing: {main_heading} -> {sub_heading}")

                summary = self._summarizer.summarize(
                    main_heading=main_heading,
                    sub_heading=sub_heading,
                    section_text=sub_section.text,
                )

                nested_section[sub_heading] = summary

            result.append(
                {
                    main_heading: nested_section,
                }
            )

        return result

    def _clean_heading(self, value: Any) -> str:
        text = str(value or "").strip()
        text = re.sub(r"\s+", " ", text)
        return text or "Untitled Section"


# ============================================================
# Pipeline
# ============================================================

class SectionSummaryPipeline:
    def __init__(
            self,
            config: SectionSummaryConfig,
            loader: RagChunkLoader | None = None,
            parser: MarkdownSubSectionParser | None = None,
    ) -> None:
        self._config = config
        self._loader = loader or RagChunkLoader()
        self._parser = parser or MarkdownSubSectionParser()

        ollama_client = OllamaChatClient(config=config)
        summarizer = SectionSummarizer(
            ollama_client=ollama_client,
            config=config,
        )

        self._builder = NestedSectionSummaryBuilder(
            parser=self._parser,
            summarizer=summarizer,
        )

    def run_from_file(
            self,
            chunks_path: str | Path,
            output_path: str | Path | None = None,
    ) -> list[dict[str, dict[str, str]]]:
        chunks = self._loader.load(chunks_path)

        nested_summary = self._builder.build(chunks)

        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(
                json.dumps(nested_summary, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        return nested_summary
