import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from extracter import VisionAnalysisService

# =========================
# Data Models
# =========================


@dataclass(frozen=True)
class MarkdownImageReference:
    """
    Markdown Image Reference.

    Purpose:
        Defines MarkdownImageReference in the document extraction pipeline that
            normalizes PDFs, enriches visual content, builds RAG units, and indexes
            data.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.

    Attributes:
        alt_text (str): Declared data field for this class.
        raw_path (str): Declared data field for this class.
        resolved_path (Path): Declared data field for this class.
        original_markdown (str): Declared data field for this class.
        caption (str | None): Declared data field for this class.
    """

    alt_text: str
    raw_path: str
    resolved_path: Path
    original_markdown: str
    caption: str | None


@dataclass(frozen=True)
class VisionAnalysisResult:
    """
    Vision Analysis Result.

    Purpose:
        Defines VisionAnalysisResult in the document extraction pipeline that normalizes
            PDFs, enriches visual content, builds RAG units, and indexes data.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.

    Attributes:
        image_path (str): Declared data field for this class.
        caption (str | None): Declared data field for this class.
        vision_text (str): Declared data field for this class.
        vision_metadata (dict[str, Any] | None): Declared data field for this class.
        raw_model_output (str | None): Declared data field for this class.
    """

    image_path: str
    caption: str | None
    vision_text: str
    vision_metadata: dict[str, Any] | None
    raw_model_output: str | None


@dataclass(frozen=True)
class MarkdownVisionProcessingConfig:
    """
    Markdown Vision Processing Config.

    Purpose:
        Defines MarkdownVisionProcessingConfig in the document extraction pipeline that
            normalizes PDFs, enriches visual content, builds RAG units, and indexes
            data.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.

    Attributes:
        input_markdown_path (Path): Declared data field for this class.
        output_markdown_path (Path): Declared data field for this class.
        cache_path (Path | None): Declared data field for this class.
    """

    input_markdown_path: Path
    output_markdown_path: Path
    cache_path: Path | None = None


# =========================
# Interfaces
# =========================


class MarkdownReader(Protocol):
    """
    Markdown Reader.

    Purpose:
        Defines MarkdownReader in the document extraction pipeline that normalizes PDFs,
            enriches visual content, builds RAG units, and indexes data.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.
    """

    def read(self, path: Path) -> str:
        """
        Read.

        Purpose:
            Implements read for the document extraction pipeline that normalizes PDFs,
                enriches visual content, builds RAG units, and indexes data.
        Class:
            Belongs to MarkdownReader; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
            path (Path): Filesystem path used as input or output for the operation.
        Returns:
            str: Result produced by the operation.
        Why Added:
            Centralizes this behavior inside MarkdownReader so related code remains
                cohesive and testable.
        """
        ...


class MarkdownWriter(Protocol):
    """
    Markdown Writer.

    Purpose:
        Defines MarkdownWriter in the document extraction pipeline that normalizes PDFs,
            enriches visual content, builds RAG units, and indexes data.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.
    """

    def write(self, path: Path, text: str) -> None:
        """
        Write.

        Purpose:
            Implements write for the document extraction pipeline that normalizes PDFs,
                enriches visual content, builds RAG units, and indexes data.
        Class:
            Belongs to MarkdownWriter; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
            path (Path): Filesystem path used as input or output for the operation.
            text (str): Input value for the text parameter.
        Returns:
            None: Performs work through side effects and does not return a value.
        Why Added:
            Centralizes this behavior inside MarkdownWriter so related code remains
                cohesive and testable.
        """
        ...


class ImageVisionAnalyzer(Protocol):
    """
    Image Vision Analyzer.

    Purpose:
        Defines ImageVisionAnalyzer in the document extraction pipeline that normalizes
            PDFs, enriches visual content, builds RAG units, and indexes data.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.
    """

    def analyze(self, image_ref: MarkdownImageReference) -> VisionAnalysisResult:
        """
        Analyze.

        Purpose:
            Implements analyze for the document extraction pipeline that normalizes
                PDFs, enriches visual content, builds RAG units, and indexes data.
        Class:
            Belongs to ImageVisionAnalyzer; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
            image_ref (MarkdownImageReference): Input value for the image ref parameter.
        Returns:
            VisionAnalysisResult: Result produced by the operation.
        Why Added:
            Centralizes this behavior inside ImageVisionAnalyzer so related code remains
                cohesive and testable.
        """
        ...


class VisionCache(Protocol):
    """
    Vision Cache.

    Purpose:
        Defines VisionCache in the document extraction pipeline that normalizes PDFs,
            enriches visual content, builds RAG units, and indexes data.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.
    """

    def get(self, key: str) -> VisionAnalysisResult | None:
        """
        Get.

        Purpose:
            Implements get for the document extraction pipeline that normalizes PDFs,
                enriches visual content, builds RAG units, and indexes data.
        Class:
            Belongs to VisionCache; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
            key (str): Input value for the key parameter.
        Returns:
            VisionAnalysisResult | None: Result produced by the operation.
        Why Added:
            Centralizes this behavior inside VisionCache so related code remains
                cohesive and testable.
        """
        ...

    def set(self, key: str, value: VisionAnalysisResult) -> None:
        """
        Set.

        Purpose:
            Implements set for the document extraction pipeline that normalizes PDFs,
                enriches visual content, builds RAG units, and indexes data.
        Class:
            Belongs to VisionCache; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
            key (str): Input value for the key parameter.
            value (VisionAnalysisResult): Raw value being validated, normalized, or
                transformed.
        Returns:
            None: Performs work through side effects and does not return a value.
        Why Added:
            Centralizes this behavior inside VisionCache so related code remains
                cohesive and testable.
        """
        ...

    def save(self) -> None:
        """
        Save.

        Purpose:
            Implements save for the document extraction pipeline that normalizes PDFs,
                enriches visual content, builds RAG units, and indexes data.
        Class:
            Belongs to VisionCache; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
        Returns:
            None: Performs work through side effects and does not return a value.
        Why Added:
            Centralizes this behavior inside VisionCache so related code remains
                cohesive and testable.
        """
        ...


# =========================
# File IO
# =========================


class FileMarkdownReader:
    """
    File Markdown Reader.

    Purpose:
        Defines FileMarkdownReader in the document extraction pipeline that normalizes
            PDFs, enriches visual content, builds RAG units, and indexes data.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.
    """

    def read(self, path: Path) -> str:
        """
        Read.

        Purpose:
            Implements read for the document extraction pipeline that normalizes PDFs,
                enriches visual content, builds RAG units, and indexes data.
        Class:
            Belongs to FileMarkdownReader; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
            path (Path): Filesystem path used as input or output for the operation.
        Returns:
            str: Result produced by the operation.
        Why Added:
            Centralizes this behavior inside FileMarkdownReader so related code remains
                cohesive and testable.
        """
        if not path.exists():
            raise FileNotFoundError(f"Markdown file not found: {path}")

        return path.read_text(encoding="utf-8")


class FileMarkdownWriter:
    """
    File Markdown Writer.

    Purpose:
        Defines FileMarkdownWriter in the document extraction pipeline that normalizes
            PDFs, enriches visual content, builds RAG units, and indexes data.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.
    """

    def write(self, path: Path, text: str) -> None:
        """
        Write.

        Purpose:
            Implements write for the document extraction pipeline that normalizes PDFs,
                enriches visual content, builds RAG units, and indexes data.
        Class:
            Belongs to FileMarkdownWriter; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
            path (Path): Filesystem path used as input or output for the operation.
            text (str): Input value for the text parameter.
        Returns:
            None: Performs work through side effects and does not return a value.
        Why Added:
            Centralizes this behavior inside FileMarkdownWriter so related code remains
                cohesive and testable.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


# =========================
# Cache
# =========================


class JsonVisionAnalysisCache:
    """
    Json Vision Analysis Cache.

    Purpose:
        Defines JsonVisionAnalysisCache in the document extraction pipeline that
            normalizes PDFs, enriches visual content, builds RAG units, and indexes
            data.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.
    """

    def __init__(self, cache_path: Path | None) -> None:
        """
        Initialize the object with its required dependencies.

        Purpose:
            Implements __init__ for the document extraction pipeline that normalizes
                PDFs, enriches visual content, builds RAG units, and indexes data.
        Class:
            Belongs to JsonVisionAnalysisCache; uses that class state and dependencies
                when available.
        Args:
            self (Self): Current instance that owns the operation state.
            cache_path (Path | None): Input value for the cache path parameter.
        Returns:
            None: Performs work through side effects and does not return a value.
        Why Added:
            Centralizes this behavior inside JsonVisionAnalysisCache so related code
                remains cohesive and testable.
        """
        self._cache_path = cache_path
        self._cache: dict[str, dict[str, Any]] = self._load()

    def get(self, key: str) -> VisionAnalysisResult | None:
        """
        Get.

        Purpose:
            Implements get for the document extraction pipeline that normalizes PDFs,
                enriches visual content, builds RAG units, and indexes data.
        Class:
            Belongs to JsonVisionAnalysisCache; uses that class state and dependencies
                when available.
        Args:
            self (Self): Current instance that owns the operation state.
            key (str): Input value for the key parameter.
        Returns:
            VisionAnalysisResult | None: Result produced by the operation.
        Why Added:
            Centralizes this behavior inside JsonVisionAnalysisCache so related code
                remains cohesive and testable.
        """
        value = self._cache.get(key)

        if not value:
            return None

        return VisionAnalysisResult(
            image_path=value["image_path"],
            caption=value.get("caption"),
            vision_text=value["vision_text"],
            vision_metadata=value.get("vision_metadata"),
            raw_model_output=value.get("raw_model_output"),
        )

    def set(self, key: str, value: VisionAnalysisResult) -> None:
        """
        Set.

        Purpose:
            Implements set for the document extraction pipeline that normalizes PDFs,
                enriches visual content, builds RAG units, and indexes data.
        Class:
            Belongs to JsonVisionAnalysisCache; uses that class state and dependencies
                when available.
        Args:
            self (Self): Current instance that owns the operation state.
            key (str): Input value for the key parameter.
            value (VisionAnalysisResult): Raw value being validated, normalized, or
                transformed.
        Returns:
            None: Performs work through side effects and does not return a value.
        Why Added:
            Centralizes this behavior inside JsonVisionAnalysisCache so related code
                remains cohesive and testable.
        """
        self._cache[key] = {
            "image_path": value.image_path,
            "caption": value.caption,
            "vision_text": value.vision_text,
            "vision_metadata": value.vision_metadata,
            "raw_model_output": value.raw_model_output,
        }

    def save(self) -> None:
        """
        Save.

        Purpose:
            Implements save for the document extraction pipeline that normalizes PDFs,
                enriches visual content, builds RAG units, and indexes data.
        Class:
            Belongs to JsonVisionAnalysisCache; uses that class state and dependencies
                when available.
        Args:
            self (Self): Current instance that owns the operation state.
        Returns:
            None: Performs work through side effects and does not return a value.
        Why Added:
            Centralizes this behavior inside JsonVisionAnalysisCache so related code
                remains cohesive and testable.
        """
        if not self._cache_path:
            return

        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache_path.write_text(
            json.dumps(self._cache, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _load(self) -> dict[str, dict[str, Any]]:
        """
        Load.

        Purpose:
            Implements _load for the document extraction pipeline that normalizes PDFs,
                enriches visual content, builds RAG units, and indexes data.
        Class:
            Belongs to JsonVisionAnalysisCache; uses that class state and dependencies
                when available.
        Args:
            self (Self): Current instance that owns the operation state.
        Returns:
            dict[str, dict[str, Any]]: Structured data produced by the operation.
        Why Added:
            Centralizes this behavior inside JsonVisionAnalysisCache so related code
                remains cohesive and testable.
        """
        if not self._cache_path:
            return {}

        if not self._cache_path.exists():
            return {}

        return json.loads(self._cache_path.read_text(encoding="utf-8"))


# =========================
# Markdown Image Extraction
# =========================


class MarkdownImageExtractor:
    """
    Markdown Image Extractor.

    Purpose:
        Defines MarkdownImageExtractor in the document extraction pipeline that
            normalizes PDFs, enriches visual content, builds RAG units, and indexes
            data.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.

    Attributes:
        IMAGE_RE (Any): Class-level value used by this class.
    """

    IMAGE_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<path>[^)]+)\)")

    def extract(
        self,
        markdown_text: str,
        markdown_base_dir: Path,
    ) -> list[MarkdownImageReference]:
        """
        Extract.

        Purpose:
            Implements extract for the document extraction pipeline that normalizes
                PDFs, enriches visual content, builds RAG units, and indexes data.
        Class:
            Belongs to MarkdownImageExtractor; uses that class state and dependencies
                when available.
        Args:
            self (Self): Current instance that owns the operation state.
            markdown_text (str): Input value for the markdown text parameter.
            markdown_base_dir (Path): Input value for the markdown base dir parameter.
        Returns:
            list[MarkdownImageReference]: Structured data produced by the operation.
        Why Added:
            Centralizes this behavior inside MarkdownImageExtractor so related code
                remains cohesive and testable.
        """
        lines = markdown_text.splitlines()
        image_refs: list[MarkdownImageReference] = []

        for index, line in enumerate(lines):
            match = self.IMAGE_RE.search(line)

            if not match:
                continue

            raw_path = match.group("path").strip().strip('"').strip("'")
            resolved_path = self._resolve_path(
                raw_path=raw_path,
                markdown_base_dir=markdown_base_dir,
            )

            caption = self._find_previous_caption(
                lines=lines,
                image_line_index=index,
            )

            image_refs.append(
                MarkdownImageReference(
                    alt_text=match.group("alt").strip(),
                    raw_path=raw_path,
                    resolved_path=resolved_path,
                    original_markdown=match.group(0),
                    caption=caption,
                )
            )

        return image_refs

    def _resolve_path(
        self,
        raw_path: str,
        markdown_base_dir: Path,
    ) -> Path:
        """
        Resolve path.

        Purpose:
            Implements _resolve_path for the document extraction pipeline that
                normalizes PDFs, enriches visual content, builds RAG units, and indexes
                data.
        Class:
            Belongs to MarkdownImageExtractor; uses that class state and dependencies
                when available.
        Args:
            self (Self): Current instance that owns the operation state.
            raw_path (str): Input value for the raw path parameter.
            markdown_base_dir (Path): Input value for the markdown base dir parameter.
        Returns:
            Path: Filesystem path resolved or created by the operation.
        Why Added:
            Centralizes this behavior inside MarkdownImageExtractor so related code
                remains cohesive and testable.
        """
        path = Path(raw_path)

        if path.is_absolute():
            return path

        candidate = markdown_base_dir / path
        if candidate.exists():
            return candidate

        # fallback when markdown path is relative to project root
        return Path.cwd() / path

    def _find_previous_caption(
        self,
        lines: list[str],
        image_line_index: int,
        lookback: int = 6,
    ) -> str | None:
        """
        Find previous caption.

        Purpose:
            Implements _find_previous_caption for the document extraction pipeline that
                normalizes PDFs, enriches visual content, builds RAG units, and indexes
                data.
        Class:
            Belongs to MarkdownImageExtractor; uses that class state and dependencies
                when available.
        Args:
            self (Self): Current instance that owns the operation state.
            lines (list[str]): Input value for the lines parameter.
            image_line_index (int): Input value for the image line index parameter.
            lookback (int): Input value for the lookback parameter.
        Returns:
            str | None: Result produced by the operation.
        Why Added:
            Centralizes this behavior inside MarkdownImageExtractor so related code
                remains cohesive and testable.
        """
        start = max(0, image_line_index - lookback)

        for line in reversed(lines[start:image_line_index]):
            cleaned = line.strip()

            if not cleaned:
                continue

            if cleaned.lower().startswith(("fig.", "figure", "caption:", "image:")):
                return cleaned

        return None


# =========================
# Existing Vision Service Adapter
# =========================


class VisionAnalysisServiceImageAnalyzer:
    """
    Vision Analysis Service Image Analyzer.

    Purpose:
        Defines VisionAnalysisServiceImageAnalyzer in the document extraction pipeline
            that normalizes PDFs, enriches visual content, builds RAG units, and indexes
            data.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.
    """

    def __init__(
        self,
        vision_service: VisionAnalysisService | None = None,
    ) -> None:
        """
        Initialize the object with its required dependencies.

        Purpose:
            Implements __init__ for the document extraction pipeline that normalizes
                PDFs, enriches visual content, builds RAG units, and indexes data.
        Class:
            Belongs to VisionAnalysisServiceImageAnalyzer; uses that class state and
                dependencies when available.
        Args:
            self (Self): Current instance that owns the operation state.
            vision_service (VisionAnalysisService | None): Input value for the vision
                service parameter.
        Returns:
            None: Performs work through side effects and does not return a value.
        Why Added:
            Centralizes this behavior inside VisionAnalysisServiceImageAnalyzer so
                related code remains cohesive and testable.
        """
        self._vision_service = vision_service or VisionAnalysisService()

    def analyze(self, image_ref: MarkdownImageReference) -> VisionAnalysisResult:
        """
        Analyze.

        Purpose:
            Implements analyze for the document extraction pipeline that normalizes
                PDFs, enriches visual content, builds RAG units, and indexes data.
        Class:
            Belongs to VisionAnalysisServiceImageAnalyzer; uses that class state and
                dependencies when available.
        Args:
            self (Self): Current instance that owns the operation state.
            image_ref (MarkdownImageReference): Input value for the image ref parameter.
        Returns:
            VisionAnalysisResult: Result produced by the operation.
        Why Added:
            Centralizes this behavior inside VisionAnalysisServiceImageAnalyzer so
                related code remains cohesive and testable.
        """
        result = self._vision_service.analyze_figure(
            image_path=image_ref.resolved_path,
            caption=image_ref.caption,
        )

        parsed = result.get("parsed") or {}
        raw_model_output = result.get("raw_model_output")

        vision_text = self._select_best_rag_text(
            parsed=parsed,
            raw_model_output=raw_model_output,
        )

        return VisionAnalysisResult(
            image_path=image_ref.raw_path,
            caption=image_ref.caption,
            vision_text=vision_text,
            vision_metadata=parsed,
            raw_model_output=raw_model_output,
        )

    def _select_best_rag_text(
        self,
        parsed: dict[str, Any],
        raw_model_output: str | None,
    ) -> str:
        """
        Select best rag text.

        Purpose:
            Implements _select_best_rag_text for the document extraction pipeline that
                normalizes PDFs, enriches visual content, builds RAG units, and indexes
                data.
        Class:
            Belongs to VisionAnalysisServiceImageAnalyzer; uses that class state and
                dependencies when available.
        Args:
            self (Self): Current instance that owns the operation state.
            parsed (dict[str, Any]): Input value for the parsed parameter.
            raw_model_output (str | None): Input value for the raw model output
                parameter.
        Returns:
            str: Result produced by the operation.
        Why Added:
            Centralizes this behavior inside VisionAnalysisServiceImageAnalyzer so
                related code remains cohesive and testable.
        """
        candidates = [
            parsed.get("rag_search_text"),
            parsed.get("detailed_description"),
            parsed.get("short_description"),
            raw_model_output,
        ]

        for candidate in candidates:
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()

        return "No useful visual analysis was generated for this image."


# =========================
# Cached Analyzer Decorator
# =========================


class ImageHashService:
    """
    Image Hash Service.

    Purpose:
        Defines ImageHashService in the document extraction pipeline that normalizes
            PDFs, enriches visual content, builds RAG units, and indexes data.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.
    """

    def hash_file(self, image_path: Path) -> str:
        """
        Hash file.

        Purpose:
            Implements hash_file for the document extraction pipeline that normalizes
                PDFs, enriches visual content, builds RAG units, and indexes data.
        Class:
            Belongs to ImageHashService; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
            image_path (Path): Input value for the image path parameter.
        Returns:
            str: Result produced by the operation.
        Why Added:
            Centralizes this behavior inside ImageHashService so related code remains
                cohesive and testable.
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        return hashlib.sha256(image_path.read_bytes()).hexdigest()


class CachedImageVisionAnalyzer:
    """
    Cached Image Vision Analyzer.

    Purpose:
        Defines CachedImageVisionAnalyzer in the document extraction pipeline that
            normalizes PDFs, enriches visual content, builds RAG units, and indexes
            data.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.
    """

    def __init__(
        self,
        analyzer: ImageVisionAnalyzer,
        cache: VisionCache,
        hash_service: ImageHashService | None = None,
    ) -> None:
        """
        Initialize the object with its required dependencies.

        Purpose:
            Implements __init__ for the document extraction pipeline that normalizes
                PDFs, enriches visual content, builds RAG units, and indexes data.
        Class:
            Belongs to CachedImageVisionAnalyzer; uses that class state and dependencies
                when available.
        Args:
            self (Self): Current instance that owns the operation state.
            analyzer (ImageVisionAnalyzer): Input value for the analyzer parameter.
            cache (VisionCache): Input value for the cache parameter.
            hash_service (ImageHashService | None): Input value for the hash service
                parameter.
        Returns:
            None: Performs work through side effects and does not return a value.
        Why Added:
            Centralizes this behavior inside CachedImageVisionAnalyzer so related code
                remains cohesive and testable.
        """
        self._analyzer = analyzer
        self._cache = cache
        self._hash_service = hash_service or ImageHashService()

    def analyze(self, image_ref: MarkdownImageReference) -> VisionAnalysisResult:
        """
        Analyze.

        Purpose:
            Implements analyze for the document extraction pipeline that normalizes
                PDFs, enriches visual content, builds RAG units, and indexes data.
        Class:
            Belongs to CachedImageVisionAnalyzer; uses that class state and dependencies
                when available.
        Args:
            self (Self): Current instance that owns the operation state.
            image_ref (MarkdownImageReference): Input value for the image ref parameter.
        Returns:
            VisionAnalysisResult: Result produced by the operation.
        Why Added:
            Centralizes this behavior inside CachedImageVisionAnalyzer so related code
                remains cohesive and testable.
        """
        cache_key = self._build_cache_key(image_ref)

        cached = self._cache.get(cache_key)
        if cached:
            return cached

        result = self._analyzer.analyze(image_ref)
        self._cache.set(cache_key, result)

        return result

    def _build_cache_key(self, image_ref: MarkdownImageReference) -> str:
        """
        Build cache key.

        Purpose:
            Implements _build_cache_key for the document extraction pipeline that
                normalizes PDFs, enriches visual content, builds RAG units, and indexes
                data.
        Class:
            Belongs to CachedImageVisionAnalyzer; uses that class state and dependencies
                when available.
        Args:
            self (Self): Current instance that owns the operation state.
            image_ref (MarkdownImageReference): Input value for the image ref parameter.
        Returns:
            str: Result produced by the operation.
        Why Added:
            Centralizes this behavior inside CachedImageVisionAnalyzer so related code
                remains cohesive and testable.
        """
        image_hash = self._hash_service.hash_file(image_ref.resolved_path)
        caption_hash = hashlib.sha256((image_ref.caption or "").encode("utf-8")).hexdigest()

        return f"{image_hash}:{caption_hash}"


# =========================
# Markdown Formatting
# =========================


class VisionMarkdownFormatter:
    """
    Vision Markdown Formatter.

    Purpose:
        Defines VisionMarkdownFormatter in the document extraction pipeline that
            normalizes PDFs, enriches visual content, builds RAG units, and indexes
            data.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.
    """

    def format(
        self,
        image_ref: MarkdownImageReference,
        result: VisionAnalysisResult,
    ) -> str:
        """
        Format.

        Purpose:
            Implements format for the document extraction pipeline that normalizes PDFs,
                enriches visual content, builds RAG units, and indexes data.
        Class:
            Belongs to VisionMarkdownFormatter; uses that class state and dependencies
                when available.
        Args:
            self (Self): Current instance that owns the operation state.
            image_ref (MarkdownImageReference): Input value for the image ref parameter.
            result (VisionAnalysisResult): Input value for the result parameter.
        Returns:
            str: Result produced by the operation.
        Why Added:
            Centralizes this behavior inside VisionMarkdownFormatter so related code
                remains cohesive and testable.
        """
        parts: list[str] = []

        parts.append("[Image vision analysis]")

        if image_ref.caption:
            parts.append(f"Caption: {image_ref.caption}")

        parts.append(f"Original image path: {image_ref.raw_path}")

        metadata = result.vision_metadata or {}

        image_type = metadata.get("image_type") or metadata.get("classification")
        if image_type:
            parts.append(f"Image type: {image_type}")

        short_description = metadata.get("short_description")
        if short_description:
            parts.append(f"Short description: {short_description}")

        parts.append("")
        parts.append("RAG search text:")
        parts.append(result.vision_text)

        return "\n".join(parts).strip()


class MarkdownImageReplacer:
    """
    Markdown Image Replacer.

    Purpose:
        Defines MarkdownImageReplacer in the document extraction pipeline that
            normalizes PDFs, enriches visual content, builds RAG units, and indexes
            data.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.
    """

    def __init__(
        self,
        formatter: VisionMarkdownFormatter | None = None,
    ) -> None:
        """
        Initialize the object with its required dependencies.

        Purpose:
            Implements __init__ for the document extraction pipeline that normalizes
                PDFs, enriches visual content, builds RAG units, and indexes data.
        Class:
            Belongs to MarkdownImageReplacer; uses that class state and dependencies
                when available.
        Args:
            self (Self): Current instance that owns the operation state.
            formatter (VisionMarkdownFormatter | None): Input value for the formatter
                parameter.
        Returns:
            None: Performs work through side effects and does not return a value.
        Why Added:
            Centralizes this behavior inside MarkdownImageReplacer so related code
                remains cohesive and testable.
        """
        self._formatter = formatter or VisionMarkdownFormatter()

    def replace(
        self,
        markdown_text: str,
        replacements: dict[str, tuple[MarkdownImageReference, VisionAnalysisResult]],
    ) -> str:
        """
        Replace.

        Purpose:
            Implements replace for the document extraction pipeline that normalizes
                PDFs, enriches visual content, builds RAG units, and indexes data.
        Class:
            Belongs to MarkdownImageReplacer; uses that class state and dependencies
                when available.
        Args:
            self (Self): Current instance that owns the operation state.
            markdown_text (str): Input value for the markdown text parameter.
            replacements (dict[str, tuple[MarkdownImageReference,
                VisionAnalysisResult]]): Input value for the replacements parameter.
        Returns:
            str: Result produced by the operation.
        Why Added:
            Centralizes this behavior inside MarkdownImageReplacer so related code
                remains cohesive and testable.
        """
        updated_text = markdown_text

        for original_markdown, replacement in replacements.items():
            image_ref, result = replacement

            replacement_text = self._formatter.format(
                image_ref=image_ref,
                result=result,
            )

            updated_text = updated_text.replace(
                original_markdown,
                replacement_text,
            )

        return updated_text


# =========================
# Application Pipeline
# =========================


class MarkdownImageVisionProcessor:
    """
    Markdown Image Vision Processor.

    Purpose:
        Defines MarkdownImageVisionProcessor in the document extraction pipeline that
            normalizes PDFs, enriches visual content, builds RAG units, and indexes
            data.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.
    """

    def __init__(
        self,
        config: MarkdownVisionProcessingConfig,
        reader: MarkdownReader | None = None,
        writer: MarkdownWriter | None = None,
        extractor: MarkdownImageExtractor | None = None,
        analyzer: ImageVisionAnalyzer | None = None,
        cache: VisionCache | None = None,
        replacer: MarkdownImageReplacer | None = None,
    ) -> None:
        """
        Initialize the object with its required dependencies.

        Purpose:
            Implements __init__ for the document extraction pipeline that normalizes
                PDFs, enriches visual content, builds RAG units, and indexes data.
        Class:
            Belongs to MarkdownImageVisionProcessor; uses that class state and
                dependencies when available.
        Args:
            self (Self): Current instance that owns the operation state.
            config (MarkdownVisionProcessingConfig): Configuration object controlling
                this component.
            reader (MarkdownReader | None): Input value for the reader parameter.
            writer (MarkdownWriter | None): Input value for the writer parameter.
            extractor (MarkdownImageExtractor | None): Input value for the extractor
                parameter.
            analyzer (ImageVisionAnalyzer | None): Input value for the analyzer
                parameter.
            cache (VisionCache | None): Input value for the cache parameter.
            replacer (MarkdownImageReplacer | None): Input value for the replacer
                parameter.
        Returns:
            None: Performs work through side effects and does not return a value.
        Why Added:
            Centralizes this behavior inside MarkdownImageVisionProcessor so related
                code remains cohesive and testable.
        """
        self._config = config
        self._reader = reader or FileMarkdownReader()
        self._writer = writer or FileMarkdownWriter()
        self._extractor = extractor or MarkdownImageExtractor()
        self._cache = cache or JsonVisionAnalysisCache(config.cache_path)
        self._replacer = replacer or MarkdownImageReplacer()

        base_analyzer = analyzer or VisionAnalysisServiceImageAnalyzer()

        self._analyzer = CachedImageVisionAnalyzer(
            analyzer=base_analyzer,
            cache=self._cache,
        )

    def run(self) -> Path:
        """
        Run.

        Purpose:
            Implements run for the document extraction pipeline that normalizes PDFs,
                enriches visual content, builds RAG units, and indexes data.
        Class:
            Belongs to MarkdownImageVisionProcessor; uses that class state and
                dependencies when available.
        Args:
            self (Self): Current instance that owns the operation state.
        Returns:
            Path: Filesystem path resolved or created by the operation.
        Why Added:
            Centralizes this behavior inside MarkdownImageVisionProcessor so related
                code remains cohesive and testable.
        """
        markdown_text = self._reader.read(self._config.input_markdown_path)

        image_refs = self._extractor.extract(
            markdown_text=markdown_text,
            markdown_base_dir=self._config.input_markdown_path.parent,
        )

        replacements: dict[str, tuple[MarkdownImageReference, VisionAnalysisResult]] = {}

        for image_ref in image_refs:
            if not image_ref.resolved_path.exists():
                print(f"Image not found, skipping: {image_ref.raw_path}")
                continue

            print(f"Analyzing image: {image_ref.resolved_path}")

            result = self._analyzer.analyze(image_ref)

            replacements[image_ref.original_markdown] = (
                image_ref,
                result,
            )

        updated_markdown = self._replacer.replace(
            markdown_text=markdown_text,
            replacements=replacements,
        )

        self._writer.write(
            path=self._config.output_markdown_path,
            text=updated_markdown,
        )

        self._cache.save()

        return self._config.output_markdown_path
