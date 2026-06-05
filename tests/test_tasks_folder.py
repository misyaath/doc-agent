"""Folder-level tests for staged file-processing task classes."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import tasks.file_extracter as file_tasks
from tasks.file_extracter import (
    EmbeddingTask,
    ExtractionTask,
    FileTaskContext,
    HeadingGroupingTask,
    MarkdownVisionTask,
    SectionSummarizationTask,
)


def _task_context(tmp_path: Path) -> FileTaskContext:
    """Build a FileTaskContext rooted in a temporary PDF path."""
    source_pdf = tmp_path / "source.pdf"
    source_pdf.write_bytes(b"%PDF-fake")
    return FileTaskContext(
        file_id="file-1",
        chat_id="chat-1",
        user_id=5,
        file_path=str(source_pdf),
        filename="source.pdf",
    )


def test_extraction_task_builds_docling_config_and_runs_extractor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify extraction task passes the PDF path and output directory to Docling."""
    ctx = _task_context(tmp_path)

    class FakeDoclingPdfExtractor:
        """Docling extractor double for task tests."""

        config: Any | None = None

        def __init__(self, config: Any) -> None:
            """Capture extraction config passed by the task."""
            FakeDoclingPdfExtractor.config = config

        def run(self) -> SimpleNamespace:
            """Return extraction counts expected by task logging."""
            return SimpleNamespace(pictures_count=1, tables_count=2, texts_count=3)

    monkeypatch.setattr(file_tasks, "DoclingPdfExtractor", FakeDoclingPdfExtractor)

    ExtractionTask().run(ctx)

    assert FakeDoclingPdfExtractor.config is not None
    assert FakeDoclingPdfExtractor.config.pdf_path == Path(ctx.file_path)
    assert FakeDoclingPdfExtractor.config.output_dir == ctx.file_base_path


def test_markdown_vision_task_configures_processor_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify markdown vision task reads document.md and writes document_rag.md."""
    ctx = _task_context(tmp_path)

    class FakeMarkdownImageVisionProcessor:
        """Markdown vision processor double for task tests."""

        config: Any | None = None

        def __init__(self, config: Any) -> None:
            """Capture processor config supplied by the task."""
            FakeMarkdownImageVisionProcessor.config = config

        def run(self) -> Path:
            """Return the output markdown path."""
            assert FakeMarkdownImageVisionProcessor.config is not None
            return FakeMarkdownImageVisionProcessor.config.output_markdown_path

    monkeypatch.setattr(file_tasks, "MarkdownImageVisionProcessor", FakeMarkdownImageVisionProcessor)

    MarkdownVisionTask().run(ctx)

    assert FakeMarkdownImageVisionProcessor.config is not None
    assert FakeMarkdownImageVisionProcessor.config.input_markdown_path == ctx.file_base_path / "document.md"
    assert FakeMarkdownImageVisionProcessor.config.output_markdown_path == ctx.file_base_path / "document_rag.md"
    assert FakeMarkdownImageVisionProcessor.config.cache_path == ctx.file_base_path / "vision_cache.json"


def test_heading_grouping_task_writes_rag_json(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify heading grouping task serializes grouped sections to rag.json."""
    ctx = _task_context(tmp_path)
    ctx.file_base_path.mkdir(parents=True, exist_ok=True)

    class FakeMarkdownMainHeadingProcessor:
        """Heading processor double for task tests."""

        requested_path: Path | None = None

        def process_file(self, markdown_path: Path) -> list[dict[str, Any]]:
            """Return grouped markdown sections for serialization."""
            FakeMarkdownMainHeadingProcessor.requested_path = markdown_path
            return [{"heading": "1 Intro", "text": "Body", "order": 0}]

    monkeypatch.setattr(file_tasks, "MarkdownMainHeadingProcessor", FakeMarkdownMainHeadingProcessor)

    HeadingGroupingTask().run(ctx)

    assert FakeMarkdownMainHeadingProcessor.requested_path == ctx.file_base_path / "document_rag.md"
    assert json.loads((ctx.file_base_path / "rag.json").read_text(encoding="utf-8")) == [
        {"heading": "1 Intro", "text": "Body", "order": 0}
    ]


def test_embedding_task_runs_qdrant_ingestion_with_expected_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify embedding task indexes rag.json using the configured chat and file ids."""
    ctx = _task_context(tmp_path)

    class FakeMarkdownRagQdrantIngestionService:
        """Qdrant ingestion service double for task tests."""

        config: Any | None = None
        ingest_args: dict[str, Any] | None = None

        def __init__(self, config: Any) -> None:
            """Capture indexing config supplied by the task."""
            FakeMarkdownRagQdrantIngestionService.config = config

        def ingest_from_file(self, *, chunks_path: Path, delete_existing_file_points: bool) -> dict[str, Any]:
            """Record ingest options and return a fake result."""
            FakeMarkdownRagQdrantIngestionService.ingest_args = {
                "chunks_path": chunks_path,
                "delete_existing_file_points": delete_existing_file_points,
            }
            return {"indexed": 1}

    monkeypatch.setattr(file_tasks, "MarkdownRagQdrantIngestionService", FakeMarkdownRagQdrantIngestionService)

    EmbeddingTask().run(ctx)

    assert FakeMarkdownRagQdrantIngestionService.config is not None
    assert FakeMarkdownRagQdrantIngestionService.config.chat_id == "chat-1"
    assert FakeMarkdownRagQdrantIngestionService.config.file_id == "file-1"
    assert FakeMarkdownRagQdrantIngestionService.config.source_file_path == ctx.file_base_path / "document_rag.md"
    assert FakeMarkdownRagQdrantIngestionService.ingest_args == {
        "chunks_path": ctx.file_base_path / "rag.json",
        "delete_existing_file_points": True,
    }


def test_section_summarization_task_updates_file_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify section summarization task generates summaries and updates metadata."""
    ctx = _task_context(tmp_path)
    loop = asyncio.new_event_loop()

    class FakeSectionSummaryPipeline:
        """Section summary pipeline double for task tests."""

        config: Any | None = None
        run_args: dict[str, Path] | None = None

        def __init__(self, config: Any) -> None:
            """Capture summary config supplied by the task."""
            FakeSectionSummaryPipeline.config = config

        def run_from_file(self, *, chunks_path: Path, output_path: Path) -> list[dict[str, Any]]:
            """Return deterministic summary JSON for repository update."""
            FakeSectionSummaryPipeline.run_args = {"chunks_path": chunks_path, "output_path": output_path}
            return [{"heading": "Intro", "summary": "Short summary"}]

    class FakeDocumentTitleDetector:
        """Document title detector double for task tests."""

        def detect_from_markdown(self, *, markdown_path: Path, fallback_title: str) -> str:
            """Return a deterministic detected document title."""
            assert markdown_path == ctx.file_base_path / "document.md"
            assert fallback_title == ctx.file_base_path.name
            return "Detected Title"

    class FakeFileSummaryRepository:
        """File summary repository double for task tests."""

        updates: list[dict[str, Any]] = []

        def update_file_summary_sync(
            self,
            *,
            file_id: str,
            title: str,
            summary: list[dict[str, Any]],
            loop: Any,
        ) -> None:
            """Capture file summary update arguments."""
            FakeFileSummaryRepository.updates.append(
                {"file_id": file_id, "title": title, "summary": summary, "loop": loop}
            )

    monkeypatch.setattr(file_tasks, "SectionSummaryPipeline", FakeSectionSummaryPipeline)
    monkeypatch.setattr(file_tasks, "DocumentTitleDetector", FakeDocumentTitleDetector)
    monkeypatch.setattr(file_tasks, "FileSummaryRepository", FakeFileSummaryRepository)

    try:
        SectionSummarizationTask(loop).run(ctx)
    finally:
        loop.close()

    assert FakeSectionSummaryPipeline.run_args == {
        "chunks_path": ctx.file_base_path / "rag.json",
        "output_path": ctx.file_base_path / "section_summary.json",
    }
    assert FakeFileSummaryRepository.updates == [
        {
            "file_id": "file-1",
            "title": "Detected Title",
            "summary": [{"heading": "Intro", "summary": "Short summary"}],
            "loop": loop,
        }
    ]


def test_section_summarization_stage_status_updates(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify section summarization writes processing and done stage statuses."""
    ctx = _task_context(tmp_path)
    loop = asyncio.new_event_loop()
    updates: list[dict[str, str]] = []

    class FakeSectionSummarizationTask:
        """Section summarization stage double for status tests."""

        stage = "summarizing"

        def run(self, ctx: FileTaskContext) -> None:
            """Perform a successful no-op summarization stage."""
            assert ctx.file_id == "file-1"

    def fake_upsert_stage_status_sync(*, file_id: str, stage_name: str, status: str, loop: Any) -> None:
        """Capture stage status updates."""
        updates.append({"file_id": file_id, "stage_name": stage_name, "status": status})

    monkeypatch.setattr(file_tasks, "_upsert_stage_status_sync", fake_upsert_stage_status_sync)

    try:
        file_tasks._run_staged_task(FakeSectionSummarizationTask(), ctx, loop)
    finally:
        loop.close()

    assert updates == [
        {"file_id": "file-1", "stage_name": "summarizing", "status": "processing"},
        {"file_id": "file-1", "stage_name": "summarizing", "status": "done"},
    ]


def test_section_summarization_stage_status_fails_on_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify section summarization writes failed status when the stage raises."""
    ctx = _task_context(tmp_path)
    loop = asyncio.new_event_loop()
    updates: list[dict[str, str]] = []

    class FakeSectionSummarizationTask:
        """Failing section summarization stage double for status tests."""

        stage = "summarizing"

        def run(self, ctx: FileTaskContext) -> None:
            """Raise an error to exercise failed status updates."""
            raise RuntimeError("summary failed")

    def fake_upsert_stage_status_sync(*, file_id: str, stage_name: str, status: str, loop: Any) -> None:
        """Capture stage status updates."""
        updates.append({"file_id": file_id, "stage_name": stage_name, "status": status})

    monkeypatch.setattr(file_tasks, "_upsert_stage_status_sync", fake_upsert_stage_status_sync)

    try:
        with pytest.raises(RuntimeError, match="summary failed"):
            file_tasks._run_staged_task(FakeSectionSummarizationTask(), ctx, loop)
    finally:
        loop.close()

    assert updates == [
        {"file_id": "file-1", "stage_name": "summarizing", "status": "processing"},
        {"file_id": "file-1", "stage_name": "summarizing", "status": "failed"},
    ]
