from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .vision_classifier import VisionAnalysisService


class JsonFileWriter:
    """
    Json File Writer.

    Purpose:
        Defines JsonFileWriter in the document extraction pipeline that normalizes PDFs,
            enriches visual content, builds RAG units, and indexes data.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.
    """

    def write(self, path: str | Path, data: Any) -> None:
        """
        Write.

        Purpose:
            Implements write for the document extraction pipeline that normalizes PDFs,
                enriches visual content, builds RAG units, and indexes data.
        Class:
            Belongs to JsonFileWriter; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
            path (str | Path): Filesystem path used as input or output for the
                operation.
            data (Any): Input value for the data parameter.
        Returns:
            None: Performs work through side effects and does not return a value.
        Why Added:
            Centralizes this behavior inside JsonFileWriter so related code remains
                cohesive and testable.
        """
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


class DataUrlStripper:
    """
    Data Url Stripper.

    Purpose:
        Defines DataUrlStripper in the document extraction pipeline that normalizes
            PDFs, enriches visual content, builds RAG units, and indexes data.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.
    """

    @staticmethod
    def strip_base64_data_url(value: str) -> str:
        """
        Strip base64 data url.

        Purpose:
            Implements strip_base64_data_url for the document extraction pipeline that
                normalizes PDFs, enriches visual content, builds RAG units, and indexes
                data.
        Class:
            Belongs to DataUrlStripper; uses that class state and dependencies when
                available.
        Args:
            value (str): Raw value being validated, normalized, or transformed.
        Returns:
            str: Result produced by the operation.
        Why Added:
            Centralizes this behavior inside DataUrlStripper so related code remains
                cohesive and testable.
        """
        cleaned = value.strip()
        if cleaned.startswith("data:") and ";base64," in cleaned:
            return cleaned.split(";base64,", 1)[1]
        return cleaned


class VisualElementEnricher:
    """
    Visual Element Enricher.

    Purpose:
        Defines VisualElementEnricher in the document extraction pipeline that
            normalizes PDFs, enriches visual content, builds RAG units, and indexes
            data.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.
    """

    def __init__(
        self,
        vision_service: VisionAnalysisService | None = None,
        data_url_stripper: DataUrlStripper | None = None,
    ) -> None:
        """
        Initialize the object with its required dependencies.

        Purpose:
            Implements __init__ for the document extraction pipeline that normalizes
                PDFs, enriches visual content, builds RAG units, and indexes data.
        Class:
            Belongs to VisualElementEnricher; uses that class state and dependencies
                when available.
        Args:
            self (Self): Current instance that owns the operation state.
            vision_service (VisionAnalysisService | None): Input value for the vision
                service parameter.
            data_url_stripper (DataUrlStripper | None): Input value for the data url
                stripper parameter.
        Returns:
            None: Performs work through side effects and does not return a value.
        Why Added:
            Centralizes this behavior inside VisualElementEnricher so related code
                remains cohesive and testable.
        """
        self._vision_service = vision_service or VisionAnalysisService()
        self._stripper = data_url_stripper or DataUrlStripper()

    def enrich(self, normalized: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Enrich.

        Purpose:
            Implements enrich for the document extraction pipeline that normalizes PDFs,
                enriches visual content, builds RAG units, and indexes data.
        Class:
            Belongs to VisualElementEnricher; uses that class state and dependencies
                when available.
        Args:
            self (Self): Current instance that owns the operation state.
            normalized (list[dict[str, Any]]): Input value for the normalized parameter.
        Returns:
            list[dict[str, Any]]: Structured data produced by the operation.
        Why Added:
            Centralizes this behavior inside VisualElementEnricher so related code
                remains cohesive and testable.
        """
        for element in normalized:
            element_type = element.get("type")
            if element_type not in {"table", "picture"}:
                continue

            if element_type == "picture":
                image_path = self._resolve_picture_path(element)
                if not image_path:
                    continue

                print(f"Analyzing picture...: {image_path} \n")
                result = self._vision_service.analyze_figure(
                    image_path=image_path,
                    caption=element.get("caption"),
                )
                element["vision_text"] = result["raw_model_output"]
                element["vision_metadata"] = result.get("parsed")

            if element_type == "table":
                table_path = self._resolve_table_path(element)
                if not table_path:
                    continue
                print(f"Analyzing table...: {table_path} \n")
                result = self._vision_service.analyze_table(
                    image_path=table_path,
                    caption=element.get("caption"),
                )
                table_json = result.get("parsed") or {}
                existing_markdown = element.get("table_markdown")
                element["table_vision"] = table_json
                element["table_markdown"] = existing_markdown
                element["vision_text"] = table_json.get("rag_search_text")

        return normalized

    def _resolve_picture_path(self, element: dict[str, Any]) -> Path | None:
        """
        Resolve picture path.

        Purpose:
            Implements _resolve_picture_path for the document extraction pipeline that
                normalizes PDFs, enriches visual content, builds RAG units, and indexes
                data.
        Class:
            Belongs to VisualElementEnricher; uses that class state and dependencies
                when available.
        Args:
            self (Self): Current instance that owns the operation state.
            element (dict[str, Any]): Input value for the element parameter.
        Returns:
            Path | None: Filesystem path resolved or created by the operation.
        Why Added:
            Centralizes this behavior inside VisualElementEnricher so related code
                remains cohesive and testable.
        """
        image_value = element.get("image_path") or element.get("image")
        if not image_value:
            return None
        path = Path(str(image_value))
        if path.exists():
            return path
        return None

    def _is_probably_base64(self, value: str) -> bool:
        """
        Is probably base64.

        Purpose:
            Implements _is_probably_base64 for the document extraction pipeline that
                normalizes PDFs, enriches visual content, builds RAG units, and indexes
                data.
        Class:
            Belongs to VisualElementEnricher; uses that class state and dependencies
                when available.
        Args:
            self (Self): Current instance that owns the operation state.
            value (str): Raw value being validated, normalized, or transformed.
        Returns:
            bool: True when the condition is satisfied; otherwise False.
        Why Added:
            Centralizes this behavior inside VisualElementEnricher so related code
                remains cohesive and testable.
        """
        if len(value) < 100:
            return False

        # Base64 usually has no path separators and is very long
        return not ("/" in value or "\\" in value)

    def _save_base64_image(self, base64_value: str, output_dir: Path) -> Path:
        """
        Save base64 image.

        Purpose:
            Implements _save_base64_image for the document extraction pipeline that
                normalizes PDFs, enriches visual content, builds RAG units, and indexes
                data.
        Class:
            Belongs to VisualElementEnricher; uses that class state and dependencies
                when available.
        Args:
            self (Self): Current instance that owns the operation state.
            base64_value (str): Input value for the base64 value parameter.
            output_dir (Path): Input value for the output dir parameter.
        Returns:
            Path: Filesystem path resolved or created by the operation.
        Why Added:
            Centralizes this behavior inside VisualElementEnricher so related code
                remains cohesive and testable.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        image_bytes = base64.b64decode(base64_value)
        image_hash = hashlib.sha256(image_bytes).hexdigest()[:16]

        image_path = output_dir / f"table_{image_hash}.png"
        image_path.write_bytes(image_bytes)

        return image_path

    def _resolve_table_path(self, element: dict[str, Any]) -> Path | None:
        """
        Resolve table path.

        Purpose:
            Implements _resolve_table_path for the document extraction pipeline that
                normalizes PDFs, enriches visual content, builds RAG units, and indexes
                data.
        Class:
            Belongs to VisualElementEnricher; uses that class state and dependencies
                when available.
        Args:
            self (Self): Current instance that owns the operation state.
            element (dict[str, Any]): Input value for the element parameter.
        Returns:
            Path | None: Filesystem path resolved or created by the operation.
        Why Added:
            Centralizes this behavior inside VisualElementEnricher so related code
                remains cohesive and testable.
        """
        image_value = element.get("image_path")

        if image_value:
            path = Path(str(image_value))
            if path.exists():
                return path

        table = element.get("table") or {}
        table_image = table.get("image") or {}
        uri = table_image.get("uri")

        if not uri:
            return None

        uri = str(uri).strip()

        # Case 1: data URL base64 image
        if uri.startswith("data:") and ";base64," in uri:
            base64_value = uri.split(";base64,", 1)[1]
            return self._save_base64_image(
                base64_value=base64_value,
                output_dir=Path("extracted_files/temp_table_images"),
            )

        # Case 2: normal file path
        if len(uri) < 500:
            candidate = Path(uri)
            if candidate.exists():
                return candidate

        # Case 3: raw base64 without data:image prefix
        if self._is_probably_base64(uri):
            return self._save_base64_image(
                base64_value=uri,
                output_dir=Path("extracted_files/temp_table_images"),
            )

        return None


class RagTextContextBuilder:
    """
    Rag Text Context Builder.

    Purpose:
        Defines RagTextContextBuilder in the document extraction pipeline that
            normalizes PDFs, enriches visual content, builds RAG units, and indexes
            data.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.
    """

    def prepend_metadata_context(self, element: dict[str, Any], body_text: str) -> str:
        """
        Prepend metadata context.

        Purpose:
            Implements prepend_metadata_context for the document extraction pipeline
                that normalizes PDFs, enriches visual content, builds RAG units, and
                indexes data.
        Class:
            Belongs to RagTextContextBuilder; uses that class state and dependencies
                when available.
        Args:
            self (Self): Current instance that owns the operation state.
            element (dict[str, Any]): Input value for the element parameter.
            body_text (str): Input value for the body text parameter.
        Returns:
            str: Result produced by the operation.
        Why Added:
            Centralizes this behavior inside RagTextContextBuilder so related code
                remains cohesive and testable.
        """
        parts: list[str] = []
        heading_path = element.get("heading_path") or []
        if heading_path:
            parts.append("Section: " + " > ".join(heading_path))
        parts.append(body_text)
        return "\n".join(parts).strip()


@dataclass(frozen=True)
class RagUnitBuildResult:
    """
    Rag Unit Build Result.

    Purpose:
        Defines RagUnitBuildResult in the document extraction pipeline that normalizes
            PDFs, enriches visual content, builds RAG units, and indexes data.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.

    Attributes:
        rag_units (list[dict[str, Any]]): Declared data field for this class.
    """

    rag_units: list[dict[str, Any]]


class OrderedRagUnitBuilder:
    """
    Ordered Rag Unit Builder.

    Purpose:
        Defines OrderedRagUnitBuilder in the document extraction pipeline that
            normalizes PDFs, enriches visual content, builds RAG units, and indexes
            data.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.
    """

    def __init__(self, context_builder: RagTextContextBuilder | None = None) -> None:
        """
        Initialize the object with its required dependencies.

        Purpose:
            Implements __init__ for the document extraction pipeline that normalizes
                PDFs, enriches visual content, builds RAG units, and indexes data.
        Class:
            Belongs to OrderedRagUnitBuilder; uses that class state and dependencies
                when available.
        Args:
            self (Self): Current instance that owns the operation state.
            context_builder (RagTextContextBuilder | None): Input value for the context
                builder parameter.
        Returns:
            None: Performs work through side effects and does not return a value.
        Why Added:
            Centralizes this behavior inside OrderedRagUnitBuilder so related code
                remains cohesive and testable.
        """
        self._context_builder = context_builder or RagTextContextBuilder()

    def build(self, normalized: list[dict[str, Any]]) -> RagUnitBuildResult:
        """
        Build.

        Purpose:
            Implements build for the document extraction pipeline that normalizes PDFs,
                enriches visual content, builds RAG units, and indexes data.
        Class:
            Belongs to OrderedRagUnitBuilder; uses that class state and dependencies
                when available.
        Args:
            self (Self): Current instance that owns the operation state.
            normalized (list[dict[str, Any]]): Input value for the normalized parameter.
        Returns:
            RagUnitBuildResult: Result produced by the operation.
        Why Added:
            Centralizes this behavior inside OrderedRagUnitBuilder so related code
                remains cohesive and testable.
        """
        rag_units: list[dict[str, Any]] = []

        for element in sorted(normalized, key=lambda x: x["order"]):
            element_type = element.get("type")
            searchable_text = self._build_searchable_text(element, element_type)
            if not searchable_text:
                continue

            searchable_text = self._context_builder.prepend_metadata_context(element, searchable_text)
            rag_unit = self._build_base_rag_unit(element, element_type, searchable_text)

            if element_type == "table":
                self._attach_table_fields(rag_unit, element)
            if element_type == "picture":
                self._attach_picture_fields(rag_unit, element)

            rag_units.append(rag_unit)

        return RagUnitBuildResult(rag_units=rag_units)

    def _build_searchable_text(self, element: dict[str, Any], element_type: str | None) -> str:
        """
        Build searchable text.

        Purpose:
            Implements _build_searchable_text for the document extraction pipeline that
                normalizes PDFs, enriches visual content, builds RAG units, and indexes
                data.
        Class:
            Belongs to OrderedRagUnitBuilder; uses that class state and dependencies
                when available.
        Args:
            self (Self): Current instance that owns the operation state.
            element (dict[str, Any]): Input value for the element parameter.
            element_type (str | None): Input value for the element type parameter.
        Returns:
            str: Result produced by the operation.
        Why Added:
            Centralizes this behavior inside OrderedRagUnitBuilder so related code
                remains cohesive and testable.
        """
        if element_type in {"text", "group", "heading", "page_header", "page_footer"}:
            return (element.get("text") or "").strip()

        if element_type == "picture":
            parts: list[str] = []
            if element.get("caption"):
                parts.append(f"Caption: {element['caption']}")
            if element.get("vision_text"):
                parts.append(f"Vision analysis: {element['vision_text']}")
            return "\n".join(parts).strip()

        if element_type == "table":
            table_parts: list[str] = []
            table_vision = element.get("table_vision") or {}
            if element.get("caption"):
                table_parts.append(f"Table caption: {element['caption']}")
            if element.get("table_markdown"):
                table_parts.append(f"Table markdown:\n{element['table_markdown']}")
            elif table_vision.get("table_markdown"):
                table_parts.append(f"Table markdown:\n{table_vision['table_markdown']}")
            if table_vision.get("short_description"):
                table_parts.append(f"Table description: {table_vision['short_description']}")
            if table_vision.get("visible_text_summary"):
                table_parts.append(f"Visible table text: {table_vision['visible_text_summary']}")
            if table_vision.get("key_findings") and isinstance(table_vision["key_findings"], list):
                findings_text = "\n".join(f"- {finding}" for finding in table_vision["key_findings"])
                table_parts.append(f"Key findings:\n{findings_text}")
            if table_vision.get("rag_search_text"):
                table_parts.append(f"RAG search text: {table_vision['rag_search_text']}")
            elif element.get("vision_text"):
                table_parts.append(f"Vision analysis: {element['vision_text']}")

            text = "\n\n".join(table_parts).strip()
            if not text:
                return "Table extracted from PDF."
            return text

        return ""

    @staticmethod
    def _build_base_rag_unit(element: dict[str, Any], element_type: str | None, searchable_text: str) -> dict[str, Any]:
        """
        Build base rag unit.

        Purpose:
            Implements _build_base_rag_unit for the document extraction pipeline that
                normalizes PDFs, enriches visual content, builds RAG units, and indexes
                data.
        Class:
            Belongs to OrderedRagUnitBuilder; uses that class state and dependencies
                when available.
        Args:
            element (dict[str, Any]): Input value for the element parameter.
            element_type (str | None): Input value for the element type parameter.
            searchable_text (str): Input value for the searchable text parameter.
        Returns:
            dict[str, Any]: Structured data produced by the operation.
        Why Added:
            Centralizes this behavior inside OrderedRagUnitBuilder so related code
                remains cohesive and testable.
        """
        return {
            "id": element["self_ref"],
            "order": element["order"],
            "type": element_type,
            "label": element.get("label"),
            "parent_ref": element.get("parent_ref"),
            "page_no": element.get("page_no"),
            "bbox": element.get("bbox"),
            "text": searchable_text,
            "image_path": element.get("image_path"),
            "source_ref": element["self_ref"],
            "heading": element.get("heading"),
            "heading_level": element.get("heading_level"),
            "heading_ref": element.get("heading_ref"),
            "heading_path": element.get("heading_path", []),
            "page_header": element.get("page_header", ""),
            "page_footer": element.get("page_footer", ""),
        }

    @staticmethod
    def _attach_table_fields(rag_unit: dict[str, Any], element: dict[str, Any]) -> None:
        """
        Attach table fields.

        Purpose:
            Implements _attach_table_fields for the document extraction pipeline that
                normalizes PDFs, enriches visual content, builds RAG units, and indexes
                data.
        Class:
            Belongs to OrderedRagUnitBuilder; uses that class state and dependencies
                when available.
        Args:
            rag_unit (dict[str, Any]): Input value for the rag unit parameter.
            element (dict[str, Any]): Input value for the element parameter.
        Returns:
            None: Performs work through side effects and does not return a value.
        Why Added:
            Centralizes this behavior inside OrderedRagUnitBuilder so related code
                remains cohesive and testable.
        """
        table_vision = element.get("table_vision") or {}
        rag_unit.update(
            {
                "table_markdown": element.get("table_markdown"),
                "table_type": table_vision.get("table_type"),
                "columns": table_vision.get("columns"),
                "rows": table_vision.get("rows"),
                "key_findings": table_vision.get("key_findings"),
                "rag_keywords": table_vision.get("rag_keywords"),
                "table_vision": table_vision,
            }
        )

    @staticmethod
    def _attach_picture_fields(rag_unit: dict[str, Any], element: dict[str, Any]) -> None:
        """
        Attach picture fields.

        Purpose:
            Implements _attach_picture_fields for the document extraction pipeline that
                normalizes PDFs, enriches visual content, builds RAG units, and indexes
                data.
        Class:
            Belongs to OrderedRagUnitBuilder; uses that class state and dependencies
                when available.
        Args:
            rag_unit (dict[str, Any]): Input value for the rag unit parameter.
            element (dict[str, Any]): Input value for the element parameter.
        Returns:
            None: Performs work through side effects and does not return a value.
        Why Added:
            Centralizes this behavior inside OrderedRagUnitBuilder so related code
                remains cohesive and testable.
        """
        rag_unit.update(
            {
                "caption": element.get("caption"),
                "vision_metadata": element.get("vision_metadata"),
                "vision_text": element.get("vision_text"),
            }
        )
