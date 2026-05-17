import json
from pathlib import Path
import asyncio
from asyncio import AbstractEventLoop
from dataclasses import dataclass
from typing import Protocol

from sqlalchemy import select

from core.logging import get_logger
from core.settings import settings
from domain.file_process import FileStage, FileStageStatus
from database import SessionLocal
from extracter import ExtractionConfig, DoclingPdfExtractor, RagIndexingConfig
from extracter.document_title_exctracter import DocumentTitleDetector
from extracter.markdown_image_vision_processor import MarkdownImageVisionProcessor, MarkdownVisionProcessingConfig
from extracter.markdown_main_grouper import MarkdownMainHeadingProcessor
from extracter.qdrant_indexer import MarkdownRagQdrantIngestionService
from extracter.summarizer import SectionSummaryPipeline, SectionSummaryConfig
from models import Chat, File, FileProcessStage, User  # noqa: F401
from repositories.file_title_and_sumary_updater import FileSummaryRepository
from worker import celery_app

logger = get_logger(__name__)


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
    stage = FileStage.EXTRACTED.value

    def run(self, ctx: FileTaskContext) -> None:
        config = ExtractionConfig(
            pdf_path=Path(ctx.file_path),
            output_dir=ctx.file_base_path,
        )
        extractor = DoclingPdfExtractor(config=config)
        result = extractor.run()

        logger.info("Extraction completed", extra={
            "file_id": ctx.file_id,
            "pictures": result.pictures_count,
            "tables": result.tables_count,
            "texts": result.texts_count,
        })


class MarkdownVisionTask:
    stage = FileStage.NORMALIZER.value

    def run(self, ctx: FileTaskContext) -> None:
        logger.info("Vision markdown processing started", extra={"chat_id": ctx.chat_id, "file_id": ctx.file_id})

        config = MarkdownVisionProcessingConfig(
            input_markdown_path=ctx.file_base_path / "document.md",
            output_markdown_path=ctx.file_base_path / "document_rag.md",
            cache_path=ctx.file_base_path / "vision_cache.json",
        )

        output_path = MarkdownImageVisionProcessor(config=config).run()
        logger.info("Vision markdown processing completed", extra={"file_id": ctx.file_id, "output_path": str(output_path)})


class HeadingGroupingTask:
    stage = FileStage.ENRICHED.value

    def run(self, ctx: FileTaskContext) -> None:
        logger.info("Heading grouping started", extra={"chat_id": ctx.chat_id, "file_id": ctx.file_id})

        processor = MarkdownMainHeadingProcessor()
        sections = processor.process_file(ctx.file_base_path / "document_rag.md")

        with open(ctx.file_base_path / "rag.json", "w", encoding="utf-8") as file:
            json.dump(sections, file, indent=4)

        logger.info("Heading grouping completed", extra={"chat_id": ctx.chat_id, "file_id": ctx.file_id})


class SectionSummarizationTask:
    def run(self, ctx: FileTaskContext, loop: AbstractEventLoop) -> None:
        logger.info("Section summarization started", extra={"chat_id": ctx.chat_id, "file_id": ctx.file_id})
        pipeline = SectionSummaryPipeline(
            config=SectionSummaryConfig(
                ollama_url=settings.ollama_url,
                model_name=settings.text_model,
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
            loop=loop,
        )
        logger.info("Section summarization completed", extra={"chat_id": ctx.chat_id, "file_id": ctx.file_id})


class EmbeddingTask:
    stage = FileStage.EMBEDDING.value

    def run(self, ctx: FileTaskContext) -> None:
        logger.info("Embedding started", extra={"chat_id": ctx.chat_id, "file_id": ctx.file_id})

        config = RagIndexingConfig(
            chat_id=ctx.chat_id,
            file_id=ctx.file_id,
            source_file_path=ctx.file_base_path / "document_rag.md",
            qdrant_url=settings.qdrant_url,
            collection_name=settings.rag_collection_name,
            embedding_model=settings.embedding_model,
            ollama_url=settings.ollama_url,
        )

        service = MarkdownRagQdrantIngestionService(config=config)
        result = service.ingest_from_file(
            chunks_path=ctx.file_base_path / "rag.json",
            delete_existing_file_points=True,
        )
        logger.info("Embedding completed", extra={"chat_id": ctx.chat_id, "file_id": ctx.file_id, "result": result})


class StageTask(Protocol):
    stage: str

    def run(self, ctx: FileTaskContext) -> None:
        ...


def _run_staged_task(task: StageTask, ctx: FileTaskContext, loop: AbstractEventLoop) -> None:
    stage_name = task.stage
    _upsert_stage_status_sync(
        file_id=ctx.file_id,
        stage_name=stage_name,
        status=FileStageStatus.PROCESSING.value,
        loop=loop,
    )
    try:
        task.run(ctx)
        _upsert_stage_status_sync(
            file_id=ctx.file_id,
            stage_name=stage_name,
            status=FileStageStatus.DONE.value,
            loop=loop,
        )
    except Exception:
        _upsert_stage_status_sync(
            file_id=ctx.file_id,
            stage_name=stage_name,
            status=FileStageStatus.FAILED.value,
            loop=loop,
        )
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

    logger.info(
        "File extraction pipeline started",
        extra={"chat_id": chat_id, "user_id": user_id, "source_file_name": filename, "file_id": file_id},
    )

    loop = asyncio.new_event_loop()
    try:
        _run_staged_task(ExtractionTask(), ctx, loop)
        _run_staged_task(MarkdownVisionTask(), ctx, loop)
        _run_staged_task(HeadingGroupingTask(), ctx, loop)

        try:
            SectionSummarizationTask().run(ctx, loop)
        except Exception:
            logger.exception("Section summarization failed", extra={"chat_id": ctx.chat_id, "file_id": ctx.file_id})
            raise

        _run_staged_task(EmbeddingTask(), ctx, loop)
        _upsert_stage_status_sync(
            file_id=ctx.file_id,
            stage_name=FileStage.DONE.value,
            status=FileStageStatus.DONE.value,
            loop=loop,
        )
    finally:
        loop.close()


def _upsert_stage_status_sync(file_id: str, stage_name: str, status: str, loop: AbstractEventLoop) -> None:
    loop.run_until_complete(_upsert_stage_status(file_id=file_id, stage_name=stage_name, status=status))


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
