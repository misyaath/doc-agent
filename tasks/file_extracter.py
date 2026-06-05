import asyncio
import json
from asyncio import AbstractEventLoop
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from sqlalchemy import select

from core.logging import get_logger
from core.settings import settings
from database import SessionLocal
from domain.file_process import FileStage, FileStageStatus
from extracter import DoclingPdfExtractor, ExtractionConfig, RagIndexingConfig
from extracter.document_title_exctracter import DocumentTitleDetector
from extracter.markdown_image_vision_processor import MarkdownImageVisionProcessor, MarkdownVisionProcessingConfig
from extracter.markdown_main_grouper import MarkdownMainHeadingProcessor
from extracter.qdrant_indexer import MarkdownRagQdrantIngestionService
from extracter.summarizer import SectionSummaryConfig, SectionSummaryPipeline
from models import Chat, File, FileProcessStage, User  # noqa: F401
from repositories.file_title_and_sumary_updater import FileSummaryRepository
from worker import celery_app

logger = get_logger(__name__)


@dataclass(frozen=True)
class FileTaskContext:
    """
    File Task Context.

    Purpose:
        Defines FileTaskContext in the Celery background-task layer that runs staged
            file processing work.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.

    Attributes:
        file_id (str): Declared data field for this class.
        chat_id (str): Declared data field for this class.
        user_id (int): Declared data field for this class.
        file_path (str): Declared data field for this class.
        filename (str): Declared data field for this class.
    """

    file_id: str
    chat_id: str
    user_id: int
    file_path: str
    filename: str

    @property
    def file_base_path(self) -> Path:
        """
        File base path.

        Purpose:
            Implements file_base_path for the Celery background-task layer that runs
                staged file processing work.
        Class:
            Belongs to FileTaskContext; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
        Returns:
            Path: Filesystem path resolved or created by the operation.
        Why Added:
            Centralizes this behavior inside FileTaskContext so related code remains
                cohesive and testable.
        """
        return Path(f"extracted_files/{self.chat_id}/{self.file_id}")


class ExtractionTask:
    """
    Extraction Task.

    Purpose:
        Defines ExtractionTask in the Celery background-task layer that runs staged file
            processing work.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.

    Attributes:
        stage (Any): Class-level value used by this class.
    """

    stage = FileStage.EXTRACTING.value

    def run(self, ctx: FileTaskContext) -> None:
        """
        Run.

        Purpose:
            Implements run for the Celery background-task layer that runs staged file
                processing work.
        Class:
            Belongs to ExtractionTask; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
            ctx (FileTaskContext): Input value for the ctx parameter.
        Returns:
            None: Performs work through side effects and does not return a value.
        Why Added:
            Centralizes this behavior inside ExtractionTask so related code remains
                cohesive and testable.
        """
        config = ExtractionConfig(
            pdf_path=Path(ctx.file_path),
            output_dir=ctx.file_base_path,
        )
        extractor = DoclingPdfExtractor(config=config)
        result = extractor.run()

        logger.info(
            "Extraction completed",
            extra={
                "file_id": ctx.file_id,
                "pictures": result.pictures_count,
                "tables": result.tables_count,
                "texts": result.texts_count,
            },
        )


class MarkdownVisionTask:
    """
    Markdown Vision Task.

    Purpose:
        Defines MarkdownVisionTask in the Celery background-task layer that runs staged
            file processing work.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.

    Attributes:
        stage (Any): Class-level value used by this class.
    """

    stage = FileStage.ANALYSING.value

    def run(self, ctx: FileTaskContext) -> None:
        """
        Run.

        Purpose:
            Implements run for the Celery background-task layer that runs staged file
                processing work.
        Class:
            Belongs to MarkdownVisionTask; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
            ctx (FileTaskContext): Input value for the ctx parameter.
        Returns:
            None: Performs work through side effects and does not return a value.
        Why Added:
            Centralizes this behavior inside MarkdownVisionTask so related code remains
                cohesive and testable.
        """
        logger.info("Vision markdown processing started", extra={"chat_id": ctx.chat_id, "file_id": ctx.file_id})

        config = MarkdownVisionProcessingConfig(
            input_markdown_path=ctx.file_base_path / "document.md",
            output_markdown_path=ctx.file_base_path / "document_rag.md",
            cache_path=ctx.file_base_path / "vision_cache.json",
        )

        output_path = MarkdownImageVisionProcessor(config=config).run()
        logger.info(
            "Vision markdown processing completed",
            extra={"file_id": ctx.file_id, "output_path": str(output_path)},
        )


class HeadingGroupingTask:
    """
    Heading Grouping Task.

    Purpose:
        Defines HeadingGroupingTask in the Celery background-task layer that runs staged
            file processing work.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.

    Attributes:
        stage (Any): Class-level value used by this class.
    """

    stage = FileStage.ORGANIZING.value

    def run(self, ctx: FileTaskContext) -> None:
        """
        Run.

        Purpose:
            Implements run for the Celery background-task layer that runs staged file
                processing work.
        Class:
            Belongs to HeadingGroupingTask; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
            ctx (FileTaskContext): Input value for the ctx parameter.
        Returns:
            None: Performs work through side effects and does not return a value.
        Why Added:
            Centralizes this behavior inside HeadingGroupingTask so related code remains
                cohesive and testable.
        """
        logger.info("Heading grouping started", extra={"chat_id": ctx.chat_id, "file_id": ctx.file_id})

        processor = MarkdownMainHeadingProcessor()
        sections = processor.process_file(ctx.file_base_path / "document_rag.md")

        with open(ctx.file_base_path / "rag.json", "w", encoding="utf-8") as file:
            json.dump(sections, file, indent=4)

        logger.info("Heading grouping completed", extra={"chat_id": ctx.chat_id, "file_id": ctx.file_id})


class SectionSummarizationTask:
    """
    Section Summarization Task.

    Purpose:
        Defines SectionSummarizationTask in the Celery background-task layer that runs
            staged file processing work.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.
    """

    stage = FileStage.SUMMARIZING.value

    def __init__(self, loop: AbstractEventLoop) -> None:
        """
        Initialize the object with its required dependencies.

        Purpose:
            Stores the event loop used by synchronous repository update helpers.
        Args:
            loop (AbstractEventLoop): Event loop used for async database calls.
        Returns:
            None: Performs work through side effects and does not return a value.
        """
        self._loop = loop

    def run(self, ctx: FileTaskContext) -> None:
        """
        Run.

        Purpose:
            Implements run for the Celery background-task layer that runs staged file
                processing work.
        Class:
            Belongs to SectionSummarizationTask; uses that class state and dependencies
                when available.
        Args:
            self (Self): Current instance that owns the operation state.
            ctx (FileTaskContext): Input value for the ctx parameter.
        Returns:
            None: Performs work through side effects and does not return a value.
        Why Added:
            Centralizes this behavior inside SectionSummarizationTask so related code
                remains cohesive and testable.
        """
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
            loop=self._loop,
        )
        logger.info("Section summarization completed", extra={"chat_id": ctx.chat_id, "file_id": ctx.file_id})


class EmbeddingTask:
    """
    Embedding Task.

    Purpose:
        Defines EmbeddingTask in the Celery background-task layer that runs staged file
            processing work.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.

    Attributes:
        stage (Any): Class-level value used by this class.
    """

    stage = FileStage.SAVING.value

    def run(self, ctx: FileTaskContext) -> None:
        """
        Run.

        Purpose:
            Implements run for the Celery background-task layer that runs staged file
                processing work.
        Class:
            Belongs to EmbeddingTask; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
            ctx (FileTaskContext): Input value for the ctx parameter.
        Returns:
            None: Performs work through side effects and does not return a value.
        Why Added:
            Centralizes this behavior inside EmbeddingTask so related code remains
                cohesive and testable.
        """
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
    """
    Stage Task.

    Purpose:
        Defines StageTask in the Celery background-task layer that runs staged file
            processing work.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.

    Attributes:
        stage (str): Declared data field for this class.
    """

    stage: str

    def run(self, ctx: FileTaskContext) -> None:
        """
        Run.

        Purpose:
            Implements run for the Celery background-task layer that runs staged file
                processing work.
        Class:
            Belongs to StageTask; uses that class state and dependencies when available.
        Args:
            self (Self): Current instance that owns the operation state.
            ctx (FileTaskContext): Input value for the ctx parameter.
        Returns:
            None: Performs work through side effects and does not return a value.
        Why Added:
            Centralizes this behavior inside StageTask so related code remains cohesive
                and testable.
        """
        ...


def _run_staged_task(task: StageTask, ctx: FileTaskContext, loop: AbstractEventLoop) -> None:
    """
    Run staged task.

    Purpose:
        Implements _run_staged_task for the Celery background-task layer that runs
            staged file processing work.
    Args:
        task (StageTask): Input value for the task parameter.
        ctx (FileTaskContext): Input value for the ctx parameter.
        loop (AbstractEventLoop): Input value for the loop parameter.
    Returns:
        None: Performs work through side effects and does not return a value.
    Why Added:
        Provides a documented entry point for this module-level behavior and keeps
            callers
        from needing to know lower-level implementation details.
    """
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
) -> None:
    """
    Process uploaded file.

    Purpose:
        Implements process_uploaded_file for the Celery background-task layer that runs
            staged file processing work.
    Args:
        file_id (str): File identifier used to locate metadata, processing stages, or
            indexed chunks.
        chat_id (str): Chat/session identifier used to scope retrieval, tasks, or
            responses.
        user_id (int): Authenticated user identifier used to scope the operation.
        file_path (str): Filesystem path to the document or artifact being processed.
        filename (str): Original uploaded filename retained for metadata and logging.
    Returns:
        None: Performs work through side effects and does not return a value.
    Why Added:
        Provides a documented entry point for this module-level behavior and keeps
            callers
        from needing to know lower-level implementation details.
    """
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

        _run_staged_task(SectionSummarizationTask(loop), ctx, loop)
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
    """
    Upsert stage status sync.

    Purpose:
        Implements _upsert_stage_status_sync for the Celery background-task layer that
            runs staged file processing work.
    Args:
        file_id (str): File identifier used to locate metadata, processing stages, or
            indexed chunks.
        stage_name (str): Input value for the stage name parameter.
        status (str): Input value for the status parameter.
        loop (AbstractEventLoop): Input value for the loop parameter.
    Returns:
        None: Performs work through side effects and does not return a value.
    Why Added:
        Provides a documented entry point for this module-level behavior and keeps
            callers
        from needing to know lower-level implementation details.
    """
    loop.run_until_complete(_upsert_stage_status(file_id=file_id, stage_name=stage_name, status=status))


async def _upsert_stage_status(file_id: str, stage_name: str, status: str) -> None:
    """
    Upsert stage status.

    Purpose:
        Implements _upsert_stage_status for the Celery background-task layer that runs
            staged file processing work.
    Args:
        file_id (str): File identifier used to locate metadata, processing stages, or
            indexed chunks.
        stage_name (str): Input value for the stage name parameter.
        status (str): Input value for the status parameter.
    Returns:
        None: Performs work through side effects and does not return a value.
    Why Added:
        Provides a documented entry point for this module-level behavior and keeps
            callers
        from needing to know lower-level implementation details.
    """
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
