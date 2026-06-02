"""Unit tests for model, repository, service, and task class wiring."""

from __future__ import annotations

from unittest.mock import Mock

from controller.task_controller import RetryTaskRequest
from database import Base
from models.chat import Chat
from models.file import File
from models.file_process_stage import FileProcessStage
from models.user import User
from repositories.chat_repository import ChatRepository
from repositories.file_repository import FileRepository
from repositories.file_title_and_sumary_updater import FileSummaryRepository
from repositories.user_repository import UserRepository
from services.agent_service import AgentService, FileSummaryCacheService, RagAgentRunner
from services.chat_service import ChatService
from services.file_service import FileService, _is_pdf_file
from services.user_service import UserService
from tasks.file_extracter import (
    EmbeddingTask,
    ExtractionTask,
    FileTaskContext,
    HeadingGroupingTask,
    MarkdownVisionTask,
    SectionSummarizationTask,
)


def test_sqlalchemy_model_classes_define_expected_tables() -> None:
    """Verify sqlalchemy model classes define expected tables."""
    assert issubclass(User, Base)
    assert issubclass(Chat, Base)
    assert issubclass(File, Base)
    assert issubclass(FileProcessStage, Base)
    assert User.__tablename__ == "users"
    assert Chat.__tablename__ == "chats"
    assert File.__tablename__ == "files"
    assert FileProcessStage.__tablename__ == "file_process_stages"


def test_repository_classes_store_injected_database_dependency() -> None:
    """Verify repository classes store injected database dependency."""
    db = Mock()
    assert UserRepository(db)._db is db
    assert ChatRepository(db)._db is db
    assert FileRepository(db)._db is db
    assert isinstance(FileSummaryRepository(), FileSummaryRepository)


def test_service_classes_store_injected_dependencies() -> None:
    """Verify service classes store injected dependencies."""
    user_repo = Mock()
    chat_repo = Mock()
    file_repo = Mock()
    cache = Mock()
    runner = Mock()

    assert UserService(user_repo)._user_repository is user_repo
    assert ChatService(chat_repo)._chat_repository is chat_repo

    file_service = FileService(chat_repository=chat_repo, file_repository=file_repo)
    assert file_service._chat_repository is chat_repo
    assert file_service._file_repository is file_repo

    agent_service = AgentService(chat_repository=chat_repo, file_summary_cache=cache, runner=runner)
    assert agent_service._chat_repository is chat_repo
    assert agent_service._file_summary_cache is cache
    assert agent_service._runner is runner


def test_file_summary_cache_service_and_runner_configuration() -> None:
    """Verify file summary cache service and runner configuration."""
    redis = Mock()
    file_repo = Mock()
    cache = FileSummaryCacheService(file_repository=file_repo, redis_client=redis)
    assert cache._file_repository is file_repo
    assert cache._redis is redis
    assert cache.cache_key("chat-1") == "agent:chat:chat-1:file_summaries"

    runner = RagAgentRunner(
        qdrant_url="http://qdrant",
        collection_name="collection",
        embedding_model="embedding",
        ollama_url="http://ollama",
        text_model="llm",
    )
    assert runner._qdrant_url == "http://qdrant"
    assert runner._collection_name == "collection"
    assert runner._embedding_model == "embedding"
    assert runner._ollama_url == "http://ollama"
    assert runner._text_model == "llm"


def test_file_service_pdf_detection() -> None:
    """Verify file service pdf detection."""
    assert _is_pdf_file(Mock(filename="document.pdf", content_type="application/pdf"))
    assert _is_pdf_file(Mock(filename="document.pdf", content_type="application/x-pdf"))
    assert not _is_pdf_file(Mock(filename="document.txt", content_type="text/plain"))


def test_celery_task_context_and_stage_classes() -> None:
    """Verify celery task context and stage classes."""
    ctx = FileTaskContext(
        file_id="file-1",
        chat_id="chat-1",
        user_id=1,
        file_path="/tmp/document.pdf",
        filename="document.pdf",
    )
    assert str(ctx.file_base_path) == "extracted_files/chat-1/file-1"

    assert ExtractionTask.stage == "extracted"
    assert MarkdownVisionTask.stage == "normalizer"
    assert HeadingGroupingTask.stage == "enriched"
    assert not hasattr(SectionSummarizationTask, "stage")
    assert EmbeddingTask.stage == "embedding"


def test_retry_task_request_defaults() -> None:
    """Verify retry task request defaults."""
    request = RetryTaskRequest()
    assert request.task_name is None
    assert request.args == []
    assert request.kwargs == {}
    assert request.countdown == 0
