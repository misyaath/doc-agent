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
    alt_text: str
    raw_path: str
    resolved_path: Path
    original_markdown: str
    caption: str | None


@dataclass(frozen=True)
class VisionAnalysisResult:
    image_path: str
    caption: str | None
    vision_text: str
    vision_metadata: dict[str, Any] | None
    raw_model_output: str | None


@dataclass(frozen=True)
class MarkdownVisionProcessingConfig:
    input_markdown_path: Path
    output_markdown_path: Path
    cache_path: Path | None = None


# =========================
# Interfaces
# =========================

class MarkdownReader(Protocol):
    def read(self, path: Path) -> str:
        ...


class MarkdownWriter(Protocol):
    def write(self, path: Path, text: str) -> None:
        ...


class ImageVisionAnalyzer(Protocol):
    def analyze(self, image_ref: MarkdownImageReference) -> VisionAnalysisResult:
        ...


class VisionCache(Protocol):
    def get(self, key: str) -> VisionAnalysisResult | None:
        ...

    def set(self, key: str, value: VisionAnalysisResult) -> None:
        ...

    def save(self) -> None:
        ...


# =========================
# File IO
# =========================

class FileMarkdownReader:
    def read(self, path: Path) -> str:
        if not path.exists():
            raise FileNotFoundError(f"Markdown file not found: {path}")

        return path.read_text(encoding="utf-8")


class FileMarkdownWriter:
    def write(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


# =========================
# Cache
# =========================

class JsonVisionAnalysisCache:
    def __init__(self, cache_path: Path | None) -> None:
        self._cache_path = cache_path
        self._cache: dict[str, dict[str, Any]] = self._load()

    def get(self, key: str) -> VisionAnalysisResult | None:
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
        self._cache[key] = {
            "image_path": value.image_path,
            "caption": value.caption,
            "vision_text": value.vision_text,
            "vision_metadata": value.vision_metadata,
            "raw_model_output": value.raw_model_output,
        }

    def save(self) -> None:
        if not self._cache_path:
            return

        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache_path.write_text(
            json.dumps(self._cache, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _load(self) -> dict[str, dict[str, Any]]:
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
    Finds markdown image syntax:

    ![Image](path/to/image.png)
    """

    IMAGE_RE = re.compile(
        r"!\[(?P<alt>[^\]]*)\]\((?P<path>[^)]+)\)"
    )

    def extract(
            self,
            markdown_text: str,
            markdown_base_dir: Path,
    ) -> list[MarkdownImageReference]:
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
    Adapter around your existing VisionAnalysisService.

    This reuses your existing code:
        VisionAnalysisService().analyze_figure(...)
    """

    def __init__(
            self,
            vision_service: VisionAnalysisService | None = None,
    ) -> None:
        self._vision_service = vision_service or VisionAnalysisService()

    def analyze(self, image_ref: MarkdownImageReference) -> VisionAnalysisResult:
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
    def hash_file(self, image_path: Path) -> str:
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        return hashlib.sha256(image_path.read_bytes()).hexdigest()


class CachedImageVisionAnalyzer:
    def __init__(
            self,
            analyzer: ImageVisionAnalyzer,
            cache: VisionCache,
            hash_service: ImageHashService | None = None,
    ) -> None:
        self._analyzer = analyzer
        self._cache = cache
        self._hash_service = hash_service or ImageHashService()

    def analyze(self, image_ref: MarkdownImageReference) -> VisionAnalysisResult:
        cache_key = self._build_cache_key(image_ref)

        cached = self._cache.get(cache_key)
        if cached:
            return cached

        result = self._analyzer.analyze(image_ref)
        self._cache.set(cache_key, result)

        return result

    def _build_cache_key(self, image_ref: MarkdownImageReference) -> str:
        image_hash = self._hash_service.hash_file(image_ref.resolved_path)
        caption_hash = hashlib.sha256(
            (image_ref.caption or "").encode("utf-8")
        ).hexdigest()

        return f"{image_hash}:{caption_hash}"


# =========================
# Markdown Formatting
# =========================

class VisionMarkdownFormatter:
    def format(
            self,
            image_ref: MarkdownImageReference,
            result: VisionAnalysisResult,
    ) -> str:
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
    def __init__(
            self,
            formatter: VisionMarkdownFormatter | None = None,
    ) -> None:
        self._formatter = formatter or VisionMarkdownFormatter()

    def replace(
            self,
            markdown_text: str,
            replacements: dict[str, tuple[MarkdownImageReference, VisionAnalysisResult]],
    ) -> str:
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
    Main pipeline:

    input markdown
      -> find images
      -> analyze images with existing VisionAnalysisService
      -> replace image markdown with vision text
      -> save new markdown
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
