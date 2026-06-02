"""Folder-level tests for extractor utilities and local transformation pipelines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from extracter.markdown_image_vision_processor import (
    CachedImageVisionAnalyzer,
    JsonVisionAnalysisCache,
    MarkdownImageReference,
    MarkdownImageReplacer,
    VisionAnalysisResult,
)
from extracter.markdown_main_grouper import MarkdownMainHeadingProcessor
from extracter.rag_pipeline import OrderedRagUnitBuilder, VisualElementEnricher
from extracter.vision_classifier import BasicJsonParser, TableJsonPostProcessor, VisionAnalysisService


class FakeEncoder:
    """Image encoder double for vision analysis tests."""

    def encode(self, image_path: str | Path) -> str:
        """Return deterministic base64 text without opening the image."""
        return f"encoded:{Path(image_path).name}"


class FakeVisionClient:
    """Vision client double that records prompts and returns JSON output."""

    def __init__(self) -> None:
        """Initialize prompt capture for assertions."""
        self.prompts: list[str] = []

    def classify(self, image_base64: str, prompt: str) -> dict[str, Any]:
        """Return JSON text that the parser and post-processor can consume."""
        self.prompts.append(prompt)
        return {
            "raw_model_output": json.dumps(
                {
                    "short_description": image_base64,
                    "rag_search_text": "table search text",
                    "should_index_for_rag": True,
                }
            )
        }


class FakeImageAnalyzer:
    """Image analyzer double for cache tests."""

    def __init__(self) -> None:
        """Track how many uncached analyses were executed."""
        self.calls = 0

    def analyze(self, image_ref: MarkdownImageReference) -> VisionAnalysisResult:
        """Return deterministic vision text for the supplied image."""
        self.calls += 1
        return VisionAnalysisResult(
            image_path=str(image_ref.resolved_path),
            caption=image_ref.caption,
            vision_text="cached vision text",
            vision_metadata={"short_description": "Cached"},
            raw_model_output="{}",
        )


class FakeHashService:
    """Hash service double that avoids real image hash calculation."""

    def hash_file(self, image_path: Path) -> str:
        """Return a stable hash based on the filename."""
        return f"hash:{image_path.name}"


def test_basic_json_parser_and_table_post_processor_validate_model_output() -> None:
    """Verify vision JSON parsing and table defaults are applied."""
    parsed = BasicJsonParser().parse('noise ```json\n{"rag_keywords":["ai"]}\n``` tail')
    assert parsed == {"rag_keywords": ["ai"]}

    with pytest.raises(ValueError):
        BasicJsonParser().parse("no json here")

    processed = TableJsonPostProcessor().process({"key_findings": "wrong", "should_index_for_rag": "yes"})
    assert processed["table_type"] == "other"
    assert processed["key_findings"] == []
    assert processed["should_index_for_rag"] is True


def test_vision_analysis_service_uses_injected_encoder_and_client() -> None:
    """Verify vision analysis can be tested without calling Ollama."""
    client = FakeVisionClient()
    service = VisionAnalysisService(encoder=FakeEncoder(), client=client)  # type: ignore[arg-type]

    figure = service.analyze_figure("figure.png", caption="Figure caption")
    table = service.analyze_table("table.png", caption="Table caption")

    assert figure["parsed"]["short_description"] == "encoded:figure.png"
    assert table["parsed"]["rag_search_text"] == "table search text"
    assert any("Figure caption" in prompt for prompt in client.prompts)
    assert any("Table caption" in prompt for prompt in client.prompts)


def test_json_vision_cache_and_cached_analyzer_reuse_results(tmp_path: Path) -> None:
    """Verify cached image analysis avoids repeated analyzer calls and persists JSON."""
    cache_path = tmp_path / "vision_cache.json"
    cache = JsonVisionAnalysisCache(cache_path)
    image_ref = MarkdownImageReference(
        alt_text="alt",
        raw_path="image.png",
        resolved_path=tmp_path / "image.png",
        original_markdown="![alt](image.png)",
        caption="Caption",
    )
    analyzer = FakeImageAnalyzer()
    cached_analyzer = CachedImageVisionAnalyzer(analyzer=analyzer, cache=cache, hash_service=FakeHashService())  # type: ignore[arg-type]

    first = cached_analyzer.analyze(image_ref)
    second = cached_analyzer.analyze(image_ref)
    cache.save()

    assert first.vision_text == second.vision_text
    assert analyzer.calls == 1
    assert json.loads(cache_path.read_text(encoding="utf-8"))


def test_ordered_rag_unit_builder_builds_text_table_and_picture_units() -> None:
    """Verify normalized extractor elements become searchable RAG units in order."""
    result = OrderedRagUnitBuilder().build(
        [
            {
                "self_ref": "#/pictures/0",
                "order": 3,
                "type": "picture",
                "caption": "Architecture",
                "vision_text": "A system diagram.",
                "heading_path": ["Intro"],
            },
            {
                "self_ref": "#/texts/0",
                "order": 1,
                "type": "text",
                "text": "Main body",
                "heading_path": ["Intro"],
            },
            {
                "self_ref": "#/tables/0",
                "order": 2,
                "type": "table",
                "caption": "Metrics",
                "table_vision": {"key_findings": ["Higher accuracy"], "rag_search_text": "accuracy metrics"},
            },
        ]
    )

    assert [unit["id"] for unit in result.rag_units] == ["#/texts/0", "#/tables/0", "#/pictures/0"]
    assert "Section: Intro" in result.rag_units[0]["text"]
    assert result.rag_units[1]["key_findings"] == ["Higher accuracy"]
    assert result.rag_units[2]["vision_text"] == "A system diagram."


def test_visual_element_enricher_and_markdown_processor_transform_local_files(tmp_path: Path) -> None:
    """Verify extractor helpers enrich existing image paths and write grouped sections."""
    image_path = tmp_path / "figure.png"
    image_path.write_bytes(b"image")

    class FakeVisionService:
        """Vision service double for visual element enrichment."""

        def analyze_figure(self, image_path: Path, caption: str | None = None) -> dict[str, Any]:
            """Return figure analysis text for enrichment."""
            return {"raw_model_output": f"figure:{image_path.name}:{caption}", "parsed": {"type": "figure"}}

        def analyze_table(self, image_path: Path, caption: str | None = None) -> dict[str, Any]:
            """Return table analysis text for enrichment."""
            return {"parsed": {"rag_search_text": f"table:{image_path.name}:{caption}"}}

    enriched = VisualElementEnricher(vision_service=FakeVisionService()).enrich(  # type: ignore[arg-type]
        [{"type": "picture", "image_path": str(image_path), "caption": "Cap"}]
    )
    assert enriched[0]["vision_text"] == "figure:figure.png:Cap"

    markdown_path = tmp_path / "document.md"
    markdown_path.write_text("# 1 Intro\nBody\n## 1.1 Detail\nMore", encoding="utf-8")
    sections = MarkdownMainHeadingProcessor().process_file(markdown_path)
    assert sections[0]["heading"] == "1 Intro"
    assert "1.1 Detail" in sections[0]["text"]


def test_markdown_image_replacer_swaps_image_markup_for_vision_text(tmp_path: Path) -> None:
    """Verify markdown image replacement injects generated vision descriptions."""
    image_ref = MarkdownImageReference(
        alt_text="alt",
        raw_path="image.png",
        resolved_path=tmp_path / "image.png",
        original_markdown="![alt](image.png)",
        caption="Caption",
    )
    result = VisionAnalysisResult(
        image_path=str(image_ref.resolved_path),
        caption="Caption",
        vision_text="Detailed visual description.",
        vision_metadata={"image_type": "figure"},
        raw_model_output="{}",
    )

    updated = MarkdownImageReplacer().replace(
        "Before\n![alt](image.png)\nAfter", {image_ref.original_markdown: (image_ref, result)}
    )

    assert "![alt](image.png)" not in updated
    assert "Detailed visual description." in updated
