"""Unit tests for extractor configuration, markdown, RAG, and vision helper classes."""

from __future__ import annotations

import json
from pathlib import Path

from extracter.docling_normalizer import DoclingJsonLoader, DoclingRefParser, NormalizerConfig
from extracter.document_title_exctracter import DocumentTitleDetector
from extracter.markdown_image_vision_processor import (
    FileMarkdownReader,
    FileMarkdownWriter,
    ImageHashService,
    MarkdownImageExtractor,
    MarkdownImageReference,
    MarkdownVisionProcessingConfig,
    VisionAnalysisResult,
    VisionMarkdownFormatter,
)
from extracter.markdown_main_grouper import MarkdownMainHeadingGrouper, MarkdownMainSection
from extracter.pdf_extracter import ExtractionConfig, ExtractionResult
from extracter.qdrant_indexer import DocumentIdentityBuilder, JsonSafetyCleaner, QdrantPointIdBuilder
from extracter.rag_pipeline import DataUrlStripper, JsonFileWriter, RagTextContextBuilder, RagUnitBuildResult
from extracter.vision_classifier import BasicJsonParser, FigurePromptBuilder, TableJsonPostProcessor, TablePromptBuilder


def test_pdf_extraction_config_and_result_paths(tmp_path: Path) -> None:
    """Verify pdf extraction config and result paths."""
    config = ExtractionConfig(pdf_path=tmp_path / "doc.pdf", output_dir=tmp_path / "out")
    assert config.pictures_dir == tmp_path / "out" / "pictures"
    assert config.tables_dir == tmp_path / "out" / "tables"

    result = ExtractionResult(
        pictures_count=1,
        tables_count=2,
        texts_count=3,
        json_path=tmp_path / "document.json",
        markdown_path=tmp_path / "document.md",
    )
    assert result.tables_count == 2


def test_docling_loader_ref_parser_and_normalizer_config(tmp_path: Path) -> None:
    """Verify docling loader ref parser and normalizer config."""
    data_path = tmp_path / "document.json"
    data_path.write_text(json.dumps({"item": {"$ref": "#/texts/0"}}), encoding="utf-8")

    config = NormalizerConfig(document_json_path=data_path)
    assert config.document_json_path == data_path
    assert DoclingJsonLoader().load(data_path) == {"item": {"$ref": "#/texts/0"}}

    parser = DoclingRefParser()
    assert parser.get_ref({"self_ref": "#/texts/1"}) is None
    assert parser.get_ref({"$ref": "#/texts/2"}) == "#/texts/2"
    assert parser.get_parent_ref({"parent": {"$ref": "#/groups/0"}}) == "#/groups/0"
    assert parser.ref_to_parts("#/texts/12") == ("texts", 12)


def test_markdown_image_helpers_read_extract_format_and_hash(tmp_path: Path) -> None:
    """Verify markdown image helpers read extract format and hash."""
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"fake-image")
    markdown = "Figure 1: Demo caption\n![alt text](image.png)"
    markdown_path = tmp_path / "input.md"

    FileMarkdownWriter().write(markdown_path, markdown)
    assert FileMarkdownReader().read(markdown_path) == markdown

    refs = MarkdownImageExtractor().extract(markdown, markdown_base_dir=tmp_path)
    assert refs == [
        MarkdownImageReference(
            alt_text="alt text",
            raw_path="image.png",
            resolved_path=image_path,
            original_markdown="![alt text](image.png)",
            caption="Figure 1: Demo caption",
        )
    ]

    result = VisionAnalysisResult(
        image_path=str(image_path),
        caption="Demo caption",
        vision_text="A small demo image.",
        vision_metadata={"image_type": "figure", "short_description": "Demo"},
        raw_model_output="{}",
    )
    formatted = VisionMarkdownFormatter().format(refs[0], result)
    assert "Image vision analysis" in formatted
    assert "Demo" in formatted
    assert ImageHashService().hash_file(image_path)

    config = MarkdownVisionProcessingConfig(input_markdown_path=markdown_path, output_markdown_path=tmp_path / "out.md")
    assert config.cache_path is None


def test_markdown_heading_grouper_groups_main_sections() -> None:
    """Verify markdown heading grouper groups main sections."""
    markdown = "Intro before headings\n# 1 Introduction\nBody\n## 1.1 Detail\nMore\n# 2 Method\nSteps"
    sections = MarkdownMainHeadingGrouper().group(markdown)
    assert all(isinstance(section, MarkdownMainSection) for section in sections)
    assert [section.heading for section in sections] == ["Document Start", "1 Introduction", "2 Method"]
    assert "1.1 Detail" in sections[1].text


def test_rag_pipeline_json_and_text_helpers(tmp_path: Path) -> None:
    """Verify rag pipeline json and text helpers."""
    output_path = tmp_path / "nested" / "data.json"
    JsonFileWriter().write(output_path, {"a": 1})
    assert json.loads(output_path.read_text(encoding="utf-8")) == {"a": 1}

    assert DataUrlStripper.strip_base64_data_url("data:image/png;base64,abc123") == "abc123"
    assert DataUrlStripper.strip_base64_data_url(" abc123 ") == "abc123"

    text = RagTextContextBuilder().prepend_metadata_context({"heading_path": ["A", "B"]}, "body")
    assert text == "Section: A > B\nbody"
    assert RagUnitBuildResult(rag_units=[{"id": 1}]).rag_units == [{"id": 1}]


def test_document_identity_and_qdrant_id_helpers(tmp_path: Path) -> None:
    """Verify document identity and qdrant id helpers."""
    identity = DocumentIdentityBuilder()
    assert identity.clean_text(" a\n b ") == "a b"
    assert identity.detect_title([{"order": 1, "heading": "Document Start"}], tmp_path / "my-file.pdf") == "my file"
    assert identity.create_doc_id("Title") == identity.create_doc_id("Title")
    assert JsonSafetyCleaner().clean_value({"ok": True}) == {"ok": True}
    assert QdrantPointIdBuilder().build("doc", 1, 2) == QdrantPointIdBuilder().build("doc", 1, 2)


def test_document_title_detector_and_vision_json_helpers(tmp_path: Path) -> None:
    """Verify document title detector and vision json helpers."""
    markdown_path = tmp_path / "doc.md"
    markdown_path.write_text("# Real Title\nBody", encoding="utf-8")
    assert DocumentTitleDetector().detect_from_markdown(markdown_path) == "Real Title"
    assert (
        DocumentTitleDetector().detect_from_markdown(tmp_path / "missing.md", fallback_title="Fallback") == "Fallback"
    )

    assert "Caption" in FigurePromptBuilder().build("Caption")
    assert "Caption" in TablePromptBuilder().build("Caption")
    assert BasicJsonParser().parse('```json\n{"a": 1}\n```') == {"a": 1}

    table_data = TableJsonPostProcessor().process(
        {"key_findings": "bad", "rag_keywords": "bad", "should_index_for_rag": "yes"}
    )
    assert table_data["key_findings"] == []
    assert table_data["rag_keywords"] == []
    assert table_data["should_index_for_rag"] is True
