from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .vision_classifier import VisionAnalysisService


class JsonFileWriter:
    def write(self, path: str | Path, data: Any) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


class DataUrlStripper:
    @staticmethod
    def strip_base64_data_url(value: str) -> str:
        cleaned = value.strip()
        if cleaned.startswith("data:") and ";base64," in cleaned:
            return cleaned.split(";base64,", 1)[1]
        return cleaned


class VisualElementEnricher:
    def __init__(
            self,
            vision_service: VisionAnalysisService | None = None,
            data_url_stripper: DataUrlStripper | None = None,
    ) -> None:
        self._vision_service = vision_service or VisionAnalysisService()
        self._stripper = data_url_stripper or DataUrlStripper()

    def enrich(self, normalized: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
        image_value = element.get("image_path") or element.get("image")
        if not image_value:
            return None
        path = Path(str(image_value))
        if path.exists():
            return path
        return None

    def _is_probably_base64(self, value: str) -> bool:
        if len(value) < 100:
            return False

        # Base64 usually has no path separators and is very long
        if "/" in value or "\\" in value:
            return False

        return True

    def _save_base64_image(self, base64_value: str, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)

        image_bytes = base64.b64decode(base64_value)
        image_hash = hashlib.sha256(image_bytes).hexdigest()[:16]

        image_path = output_dir / f"table_{image_hash}.png"
        image_path.write_bytes(image_bytes)

        return image_path

    def _resolve_table_path(self, element: dict[str, Any]) -> Path | None:
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
    def prepend_metadata_context(self, element: dict[str, Any], body_text: str) -> str:
        parts: list[str] = []
        heading_path = element.get("heading_path") or []
        if heading_path:
            parts.append("Section: " + " > ".join(heading_path))
        parts.append(body_text)
        return "\n".join(parts).strip()


@dataclass(frozen=True)
class RagUnitBuildResult:
    rag_units: list[dict[str, Any]]


class OrderedRagUnitBuilder:
    def __init__(self, context_builder: RagTextContextBuilder | None = None) -> None:
        self._context_builder = context_builder or RagTextContextBuilder()

    def build(self, normalized: list[dict[str, Any]]) -> RagUnitBuildResult:
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
            parts: list[str] = []
            table_vision = element.get("table_vision") or {}
            if element.get("caption"):
                parts.append(f"Table caption: {element['caption']}")
            if element.get("table_markdown"):
                parts.append(f"Table markdown:\n{element['table_markdown']}")
            elif table_vision.get("table_markdown"):
                parts.append(f"Table markdown:\n{table_vision['table_markdown']}")
            if table_vision.get("short_description"):
                parts.append(f"Table description: {table_vision['short_description']}")
            if table_vision.get("visible_text_summary"):
                parts.append(f"Visible table text: {table_vision['visible_text_summary']}")
            if table_vision.get("key_findings") and isinstance(table_vision["key_findings"], list):
                findings_text = "\n".join(f"- {finding}" for finding in table_vision["key_findings"])
                parts.append(f"Key findings:\n{findings_text}")
            if table_vision.get("rag_search_text"):
                parts.append(f"RAG search text: {table_vision['rag_search_text']}")
            elif element.get("vision_text"):
                parts.append(f"Vision analysis: {element['vision_text']}")

            text = "\n\n".join(parts).strip()
            if not text:
                return "Table extracted from PDF."
            return text

        return ""

    @staticmethod
    def _build_base_rag_unit(element: dict[str, Any], element_type: str | None, searchable_text: str) -> dict[str, Any]:
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
        rag_unit.update(
            {
                "caption": element.get("caption"),
                "vision_metadata": element.get("vision_metadata"),
                "vision_text": element.get("vision_text"),
            }
        )
