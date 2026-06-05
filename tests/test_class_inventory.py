"""Inventory tests that keep project class coverage explicit and importable."""

from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_DIRS = {".git", ".venv", "venv", "__pycache__", "extracted_files", "uploads", "data", "tests"}

EXPECTED_CLASSES: dict[str, set[str]] = {
    "agent.AgentPrompts": {"AgentPrompts"},
    "agent.qdrant_retrieval": {
        "ChunkAggregator",
        "ChunkPayloadMapper",
        "ChunkQualityEvaluator",
        "ColbertQueryEmbedder",
        "HybridQdrantQueryService",
        "QdrantColbertReranker",
        "QdrantFilterFactory",
        "QdrantRagRetriever",
        "QueryNormalizer",
    },
    "agent.rag": {
        "AnswerGenerationService",
        "LlmJsonExtractor",
        "QueryParsingService",
        "QueryPlanBuilder",
        "QueryPlanDefaults",
        "RagGraphFactory",
        "RagState",
        "RetrievalContextFormatter",
        "RetrievalService",
        "RetrievedChunk",
    },
    "controller.task_controller": {"RetryTaskRequest"},
    "core.settings": {"Settings"},
    "database": {"Base"},
    "domain.file_process": {"FileStage", "FileStageStatus"},
    "extracter.docling_normalizer": {
        "DoclingIndexBuilder",
        "DoclingJsonLoader",
        "DoclingJsonNormalizer",
        "DoclingRefParser",
        "GroupTextCollector",
        "HeadingTracker",
        "NormalizerConfig",
        "PageFurnitureCollector",
        "PictureCaptionCollector",
        "TableMarkdownExtractor",
    },
    "extracter.document_title_exctracter": {"DocumentTitleDetector"},
    "extracter.markdown_image_vision_processor": {
        "CachedImageVisionAnalyzer",
        "FileMarkdownReader",
        "FileMarkdownWriter",
        "ImageHashService",
        "ImageVisionAnalyzer",
        "JsonVisionAnalysisCache",
        "MarkdownImageExtractor",
        "MarkdownImageReference",
        "MarkdownImageReplacer",
        "MarkdownImageVisionProcessor",
        "MarkdownReader",
        "MarkdownVisionProcessingConfig",
        "MarkdownWriter",
        "VisionAnalysisResult",
        "VisionAnalysisServiceImageAnalyzer",
        "VisionCache",
        "VisionMarkdownFormatter",
    },
    "extracter.markdown_main_grouper": {
        "MarkdownMainHeadingGrouper",
        "MarkdownMainHeadingProcessor",
        "MarkdownMainSection",
    },
    "extracter.pdf_extracter": {
        "DocAssetExporter",
        "DoclingConverter",
        "DoclingPdfExtractor",
        "ExtractionConfig",
        "ExtractionResult",
        "IDocConverter",
    },
    "extracter.qdrant_indexer": {
        "ColbertModelFactory",
        "CompactPayloadBuilder",
        "DocumentIdentityBuilder",
        "EmbeddingModelFactory",
        "FilterChunk",
        "JsonSafetyCleaner",
        "MarkdownRagChunk",
        "MarkdownRagQdrantIngestionService",
        "QdrantHybridIndexSaver",
        "QdrantPointIdBuilder",
        "RagChunkLoader",
        "RagChunkingComponents",
        "RagIndexingConfig",
    },
    "extracter.rag_pipeline": {
        "DataUrlStripper",
        "JsonFileWriter",
        "OrderedRagUnitBuilder",
        "RagTextContextBuilder",
        "RagUnitBuildResult",
        "VisualElementEnricher",
    },
    "extracter.summarizer": {
        "NestedSectionSummaryBuilder",
        "OllamaChatClient",
        "ParsedSubSection",
        "RagChunkLoader",
        "SectionSummarizer",
        "SectionSummaryConfig",
        "SectionSummaryPipeline",
        "MarkdownSubSectionParser",
    },
    "extracter.vision_classifier": {
        "BasicJsonParser",
        "FigurePromptBuilder",
        "ImageEncoder",
        "JsonPostProcessor",
        "OllamaVisionClient",
        "PromptBuilder",
        "ResizedJpegBase64Encoder",
        "TableJsonPostProcessor",
        "TablePromptBuilder",
        "VisionAnalysisService",
        "VisionClient",
        "VisionConfig",
    },
    "models.chat": {"Chat"},
    "models.file": {"File"},
    "models.file_process_stage": {"FileProcessStage"},
    "models.user": {"User"},
    "repositories.chat_repository": {"ChatRepository"},
    "repositories.file_repository": {"FileRepository"},
    "repositories.file_title_and_sumary_updater": {"FileSummaryRepository"},
    "repositories.user_repository": {"UserRepository"},
    "schemas.agent": {"AgentChatRequest", "AgentChatResponse"},
    "schemas.chat": {
        "ChatCreateRequest",
        "ChatCreateResponse",
        "ChatDeleteResponse",
        "ChatDetailResponse",
        "ChatFileProcessStageResponse",
        "ChatFileResponse",
        "ChatHistoryMessageResponse",
        "ChatListItemResponse",
        "ChatListResponse",
    },
    "schemas.file": {"FileUploadItemResponse", "FileUploadResponse"},
    "schemas.user": {"TokenResponse", "UserLoginRequest", "UserRegisterRequest", "UserRegisterResponse"},
    "services.agent_service": {"AgentService", "FileSummaryCacheService", "RagAgentRunner"},
    "services.chat_service": {"ChatService"},
    "services.file_service": {"FileService"},
    "services.user_service": {"UserService"},
    "tasks.file_extracter": {
        "EmbeddingTask",
        "ExtractionTask",
        "FileTaskContext",
        "HeadingGroupingTask",
        "MarkdownVisionTask",
        "SectionSummarizationTask",
        "StageTask",
    },
}


def _module_name(path: Path) -> str:
    relative = path.relative_to(PROJECT_ROOT).with_suffix("")
    return ".".join(relative.parts)


def _discover_classes() -> dict[str, set[str]]:
    discovered: dict[str, set[str]] = {}
    for path in PROJECT_ROOT.rglob("*.py"):
        if any(part in EXCLUDED_DIRS for part in path.parts):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        classes = {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
        if classes:
            discovered[_module_name(path)] = classes
    return discovered


def test_every_project_class_is_accounted_for() -> None:
    """Verify every project class is accounted for."""
    assert _discover_classes() == EXPECTED_CLASSES


@pytest.mark.parametrize(
    ("module_name", "class_name"),
    sorted((module, name) for module, names in EXPECTED_CLASSES.items() for name in names),
)
def test_every_accounted_class_is_importable(module_name: str, class_name: str) -> None:
    """Verify every accounted class is importable."""
    module = importlib.import_module(module_name)
    class_obj = getattr(module, class_name)
    assert inspect.isclass(class_obj)
    assert class_obj.__name__ == class_name
