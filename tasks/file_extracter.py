import json
import os
from pathlib import Path
import asyncio
from dataclasses import dataclass

from sqlalchemy import select

from database import SessionLocal
from extracter import ExtractionConfig, DoclingPdfExtractor, RagIndexingConfig
from extracter.document_title_exctracter import DocumentTitleDetector
from extracter.markdown_image_vision_processor import MarkdownImageVisionProcessor, MarkdownVisionProcessingConfig
from extracter.markdown_main_grouper import MarkdownMainHeadingProcessor
from extracter.qdrant_indexer import MarkdownRagQdrantIngestionService
from extracter.summarizer import SectionSummaryPipeline, SectionSummaryConfig
from models.file_process_stage import FileProcessStage
from repositories.file_title_and_sumary_updater import FileSummaryRepository
from worker import celery_app


@dataclass(frozen=True)
class FileTaskContext:
    file_id: str
    chat_id: str
    user_id: int
    file_path: str
    filename: str

    @property
    def file_base_path(self) -> Path:
        return Path(f"extracted_files/{self.chat_id}/{self.file_id}")


class ExtractionTask:
    stage = "extracted"

    def run(self, ctx: FileTaskContext) -> None:
        config = ExtractionConfig(
            pdf_path=Path(ctx.file_path),
            output_dir=ctx.file_base_path,
        )
        extractor = DoclingPdfExtractor(config=config)
        result = extractor.run()

        print("pictures:", result.pictures_count)
        print("tables:", result.tables_count)
        print("texts:", result.texts_count)


class MarkdownVisionTask:
    stage = "normalizer"

    def run(self, ctx: FileTaskContext) -> None:
        print(f"parsing and analyzing images... {ctx.chat_id}-{ctx.file_id}")

        config = MarkdownVisionProcessingConfig(
            input_markdown_path=ctx.file_base_path / "document.md",
            output_markdown_path=ctx.file_base_path / "document_rag.md",
            cache_path=ctx.file_base_path / "vision_cache.json",
        )

        output_path = MarkdownImageVisionProcessor(config=config).run()
        print(f"Created RAG markdown: {output_path}")


class HeadingGroupingTask:
    stage = "enriched"

    def run(self, ctx: FileTaskContext) -> None:
        print(f"Grouping by heading... {ctx.chat_id}-{ctx.file_id}")

        processor = MarkdownMainHeadingProcessor()
        sections = processor.process_file(ctx.file_base_path / "document_rag.md")

        with open(ctx.file_base_path / "rag.json", "w", encoding="utf-8") as file:
            json.dump(sections, file, indent=4)

        print(f"Grouped by heading... {ctx.chat_id}-{ctx.file_id}")


class SectionSummarizationTask:
    def run(self, ctx: FileTaskContext) -> None:
        print(f"Summarizing by heading... {ctx.chat_id}-{ctx.file_id}")
        pipeline = SectionSummaryPipeline(
            config=SectionSummaryConfig(
                ollama_url=os.getenv("OLLAMA_URL", "http://ollama:11434"),
                model_name=os.getenv("TEXT_MODEL", "llama3.1:8b"),
                target_words=25,
                temperature=0.0,
            )
        )

        summary_json = pipeline.run_from_file(
            chunks_path=ctx.file_base_path / "rag.json",
            output_path=ctx.file_base_path / "section_summary.json",
        )

        doc_title = DocumentTitleDetector().detect_from_markdown(
            markdown_path=ctx.file_base_path / "document.md",
            fallback_title=ctx.file_base_path.name,
        )

        repository = FileSummaryRepository()
        repository.update_file_summary_sync(
            file_id=ctx.file_id,
            title=doc_title,
            summary=summary_json,
        )
        print(f"Summarized by heading... {ctx.chat_id}-{ctx.file_id}")


class EmbeddingTask:
    stage = "embedding"

    def run(self, ctx: FileTaskContext) -> None:
        print(f"embedding... {ctx.chat_id}-{ctx.file_id}")

        config = RagIndexingConfig(
            chat_id=ctx.chat_id,
            file_id=ctx.file_id,
            source_file_path=ctx.file_base_path / "document_rag.md",
            qdrant_url=os.getenv("QDRANT_URL", ""),
            collection_name=os.getenv("RAG_COLLECTION_NAME", "pdf_rag"),
            embedding_model=os.getenv("EMBEDDING_MODEL", ""),
            ollama_url=os.getenv("OLLAMA_URL", ""),
        )

        service = MarkdownRagQdrantIngestionService(config=config)
        result = service.ingest_from_file(
            chunks_path=ctx.file_base_path / "rag.json",
            delete_existing_file_points=True,
        )
        print(f"result: {result}")
        print(f"embedded... {ctx.chat_id}-{ctx.file_id}")


def _run_staged_task(task: object, ctx: FileTaskContext) -> None:
    stage_name = getattr(task, "stage")
    _upsert_stage_status_sync(file_id=ctx.file_id, stage_name=stage_name, status="processing")
    try:
        task.run(ctx)
        _upsert_stage_status_sync(file_id=ctx.file_id, stage_name=stage_name, status="done")
    except Exception:
        _upsert_stage_status_sync(file_id=ctx.file_id, stage_name=stage_name, status="failed")
        raise


@celery_app.task(name="process_uploaded_file")
def process_uploaded_file(
        file_id: str,
        chat_id: str,
        user_id: int,
        file_path: str,
        filename: str,
):
    ctx = FileTaskContext(
        file_id=file_id,
        chat_id=chat_id,
        user_id=user_id,
        file_path=file_path,
        filename=filename,
    )

    path = Path(ctx.file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    print(
        f"[file_extract] starting extraction: "
        f"chat_id={chat_id}, user_id={user_id}, filename={filename}"
    )
    print(f"Processing file: {filename}")
    print(f"file_id={file_id}, chat_id={chat_id}, user_id={user_id}")

    _run_staged_task(ExtractionTask(), ctx)
    _run_staged_task(MarkdownVisionTask(), ctx)
    _run_staged_task(HeadingGroupingTask(), ctx)

    try:
        SectionSummarizationTask().run(ctx)
    except Exception:
        print(f"Failed to summarize by heading... {ctx.chat_id}-{ctx.file_id}")
        raise

    _run_staged_task(EmbeddingTask(), ctx)
    _upsert_stage_status_sync(file_id=ctx.file_id, stage_name="done", status="done")


def _upsert_stage_status_sync(file_id: str, stage_name: str, status: str) -> None:
    asyncio.run(_upsert_stage_status(file_id=file_id, stage_name=stage_name, status=status))


async def _upsert_stage_status(file_id: str, stage_name: str, status: str) -> None:
    async with SessionLocal() as session:
        stage = await session.scalar(
            select(FileProcessStage).where(
                FileProcessStage.file_id == file_id,
                FileProcessStage.stage == stage_name,
            )
        )

        if stage is None:
            stage_record = FileProcessStage(
                file_id=file_id,
                stage=stage_name,
                status=status,
            )
            session.add(stage_record)
        else:
            stage.status = status

        await session.commit()
