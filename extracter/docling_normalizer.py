from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class NormalizerConfig:
    document_json_path: Path
    picture_dir: Path = Path("docling_output/pictures")
    table_dir: Path = Path("docling_output/tables")
    include_headers_as_elements: bool = False


class DoclingJsonLoader:
    def load(self, path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))


class DoclingRefParser:
    @staticmethod
    def get_ref(obj: dict[str, Any]) -> str | None:
        return obj.get("$ref")

    @staticmethod
    def get_parent_ref(item: dict[str, Any]) -> str | None:
        return (item.get("parent") or {}).get("$ref")

    @staticmethod
    def get_page_no(item: dict[str, Any]) -> int | None:
        prov = item.get("prov") or []
        if not prov:
            return None
        return prov[0].get("page_no")

    @staticmethod
    def get_bbox(item: dict[str, Any]) -> dict[str, Any] | None:
        prov = item.get("prov") or []
        if not prov:
            return None
        return prov[0].get("bbox")

    @staticmethod
    def ref_to_parts(ref: str) -> tuple[str, int]:
        parts = ref.strip("#/").split("/")
        return parts[0], int(parts[1])


class DoclingIndexBuilder:
    def build(self, doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
        index: dict[str, dict[str, Any]] = {}
        for section in ["texts", "pictures", "tables", "groups"]:
            for item in doc.get(section, []):
                self_ref = item.get("self_ref")
                if self_ref:
                    index[self_ref] = item
        return index


class TableMarkdownExtractor:
    def extract(self, table_item: dict[str, Any]) -> str:
        """
        Try to extract markdown from Docling table JSON.

        Works if Docling JSON contains table structure/cells.
        Returns empty string if only table image exists.
        """
        # Case 1: markdown already exists in JSON
        for key in ["table_markdown", "markdown", "md"]:
            value = table_item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        # Case 2: table data exists with cells
        data = table_item.get("data") or {}
        table_cells = data.get("table_cells") or data.get("cells") or []

        if table_cells:
            return self._cells_to_markdown(table_cells)

        # Case 3: grid exists in some Docling exports
        grid = data.get("grid") or table_item.get("grid")
        if isinstance(grid, list) and grid:
            return self._grid_to_markdown(grid)

        return ""

    def _cells_to_markdown(self, cells: list[dict[str, Any]]) -> str:
        rows: dict[int, dict[int, str]] = {}

        for cell in cells:
            row_idx = (
                    cell.get("start_row_offset_idx")
                    or cell.get("row")
                    or cell.get("row_idx")
                    or 0
            )
            col_idx = (
                    cell.get("start_col_offset_idx")
                    or cell.get("col")
                    or cell.get("col_idx")
                    or 0
            )

            text = (
                    cell.get("text")
                    or cell.get("content")
                    or cell.get("value")
                    or ""
            )

            text = str(text).replace("\n", " ").strip()

            rows.setdefault(int(row_idx), {})[int(col_idx)] = text

        if not rows:
            return ""

        max_col = max(
            col_idx
            for row in rows.values()
            for col_idx in row.keys()
        )

        ordered_rows: list[list[str]] = []

        for row_idx in sorted(rows.keys()):
            row = rows[row_idx]
            ordered_rows.append(
                [row.get(col_idx, "") for col_idx in range(max_col + 1)]
            )

        return self._grid_to_markdown(ordered_rows)

    def _grid_to_markdown(self, grid: list[list[Any]]) -> str:
        if not grid:
            return ""

        clean_grid = [
            [str(cell or "").replace("\n", " ").strip() for cell in row]
            for row in grid
        ]

        if not clean_grid:
            return ""

        header = clean_grid[0]
        body = clean_grid[1:]

        lines = []
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")

        for row in body:
            # make row same length as header
            row = row + [""] * (len(header) - len(row))
            row = row[: len(header)]
            lines.append("| " + " | ".join(row) + " |")

        return "\n".join(lines)


class PageFurnitureCollector:
    def __init__(self, ref_parser: DoclingRefParser) -> None:
        self._ref_parser = ref_parser

    def collect(self, doc: dict[str, Any]) -> dict[int, dict[str, str]]:
        page_meta: dict[int, dict[str, list[str]]] = {}

        for item in doc.get("texts", []):
            label = item.get("label")
            content_layer = item.get("content_layer")
            page_no = self._ref_parser.get_page_no(item)
            text = (item.get("text") or "").strip()

            if not page_no or not text or content_layer != "furniture":
                continue
            if label not in {"page_header", "page_footer"}:
                continue

            page_meta.setdefault(page_no, {"page_header": [], "page_footer": []})
            page_meta[page_no][label].append(text)

        result: dict[int, dict[str, str]] = {}
        for page_no, values in page_meta.items():
            result[page_no] = {
                "page_header": " | ".join(values.get("page_header", [])),
                "page_footer": " | ".join(values.get("page_footer", [])),
            }
        return result

    @staticmethod
    def metadata_for(page_no: int | None, page_furniture: dict[int, dict[str, str]]) -> dict[str, str]:
        if not page_no:
            return {"page_header": "", "page_footer": ""}
        return page_furniture.get(page_no, {"page_header": "", "page_footer": ""})


class HeadingTracker:
    def __init__(self) -> None:
        self._stack: list[dict[str, Any]] = []

    def update(self, heading_text: str, level: int | None, self_ref: str) -> None:
        heading_level = level or 1
        self._stack = [h for h in self._stack if h["level"] < heading_level]
        self._stack.append({"text": heading_text, "level": heading_level, "self_ref": self_ref})

    def metadata(self) -> dict[str, Any]:
        if not self._stack:
            return {
                "heading": None,
                "heading_level": None,
                "heading_ref": None,
                "heading_path": [],
            }
        current = self._stack[-1]
        return {
            "heading": current["text"],
            "heading_level": current["level"],
            "heading_ref": current["self_ref"],
            "heading_path": [h["text"] for h in self._stack],
        }


class PictureCaptionCollector:
    def __init__(self, ref_parser: DoclingRefParser) -> None:
        self._ref_parser = ref_parser

    def collect(self, doc: dict[str, Any]) -> dict[str, str]:
        captions_by_picture: dict[str, list[str]] = {}
        for text_item in doc.get("texts", []):
            parent_ref = self._ref_parser.get_parent_ref(text_item)
            label = text_item.get("label")
            text = (text_item.get("text") or "").strip()
            if not text:
                continue
            if parent_ref and parent_ref.startswith("#/pictures/") and label == "caption":
                captions_by_picture.setdefault(parent_ref, []).append(text)
        return {picture_ref: "\n".join(captions) for picture_ref, captions in captions_by_picture.items()}


class GroupTextCollector:
    def __init__(self, ref_parser: DoclingRefParser) -> None:
        self._ref_parser = ref_parser

    def collect(self, group_ref: str, index: dict[str, Any]) -> str:
        group = index[group_ref]
        texts: list[str] = []
        for child in group.get("children", []):
            child_ref = self._ref_parser.get_ref(child)
            if not child_ref:
                continue
            child_item = index.get(child_ref)
            if not child_item:
                continue
            if child_ref.startswith("#/texts/"):
                text = (child_item.get("text") or "").strip()
                if text:
                    texts.append(text)
        return "\n".join(texts).strip()


class DoclingJsonNormalizer:
    def __init__(
            self,
            config: NormalizerConfig,
            loader: DoclingJsonLoader | None = None,
            ref_parser: DoclingRefParser | None = None,
            index_builder: DoclingIndexBuilder | None = None,
            page_furniture_collector: PageFurnitureCollector | None = None,
            picture_caption_collector: PictureCaptionCollector | None = None,
            group_text_collector: GroupTextCollector | None = None,
            table_markdown_extractor: TableMarkdownExtractor | None = None,
    ) -> None:
        self._config = config
        self._loader = loader or DoclingJsonLoader()
        self._ref_parser = ref_parser or DoclingRefParser()
        self._index_builder = index_builder or DoclingIndexBuilder()
        self._page_furniture_collector = page_furniture_collector or PageFurnitureCollector(self._ref_parser)
        self._picture_caption_collector = picture_caption_collector or PictureCaptionCollector(self._ref_parser)
        self._group_text_collector = group_text_collector or GroupTextCollector(self._ref_parser)
        self._table_markdown_extractor = table_markdown_extractor or TableMarkdownExtractor()

    def normalize(self) -> list[dict[str, Any]]:
        doc = self._loader.load(self._config.document_json_path)
        index = self._index_builder.build(doc)
        page_furniture = self._page_furniture_collector.collect(doc)
        captions_by_picture = self._picture_caption_collector.collect(doc)

        normalized: list[dict[str, Any]] = []
        heading_tracker = HeadingTracker()
        order = 0

        for child in doc.get("body", {}).get("children", []):
            ref = self._ref_parser.get_ref(child)
            if not ref:
                continue
            item = index.get(ref)
            if not item:
                continue

            section, idx = self._ref_parser.ref_to_parts(ref)

            if section == "texts":
                order = self._normalize_text_item(
                    normalized, order, item, ref, page_furniture, heading_tracker
                )
            elif section == "groups":
                order = self._normalize_group_item(
                    normalized, order, item, ref, index, page_furniture, heading_tracker
                )
            elif section == "pictures":
                order = self._normalize_picture_item(
                    normalized, order, item, ref, idx, captions_by_picture, page_furniture, heading_tracker
                )
            elif section == "tables":
                order = self._normalize_table_item(
                    normalized, order, item, ref, idx, page_furniture, heading_tracker
                )

        return normalized

    def _normalize_text_item(
            self,
            normalized: list[dict[str, Any]],
            order: int,
            item: dict[str, Any],
            ref: str,
            page_furniture: dict[int, dict[str, str]],
            heading_tracker: HeadingTracker,
    ) -> int:
        label = item.get("label")
        content_layer = item.get("content_layer")
        parent_ref = self._ref_parser.get_parent_ref(item)
        text = (item.get("text") or "").strip()
        page_no = self._ref_parser.get_page_no(item)
        if not text:
            return order

        if content_layer == "furniture":
            if self._config.include_headers_as_elements and label in {"page_header", "page_footer"}:
                normalized.append(
                    {
                        "order": order,
                        "type": label,
                        "label": label,
                        "self_ref": ref,
                        "parent_ref": parent_ref,
                        "page_no": page_no,
                        "bbox": self._ref_parser.get_bbox(item),
                        "text": text,
                        **self._page_furniture_collector.metadata_for(page_no, page_furniture),
                        **heading_tracker.metadata(),
                    }
                )
                return order + 1
            return order

        if parent_ref and parent_ref.startswith("#/pictures/") and label != "caption":
            return order

        if label == "section_header":
            heading_level = item.get("level") or 1
            heading_tracker.update(text, heading_level, ref)
            normalized.append(
                {
                    "order": order,
                    "type": "heading",
                    "label": label,
                    "self_ref": ref,
                    "parent_ref": parent_ref,
                    "page_no": page_no,
                    "bbox": self._ref_parser.get_bbox(item),
                    "text": text,
                    **self._page_furniture_collector.metadata_for(page_no, page_furniture),
                    **heading_tracker.metadata(),
                }
            )
            return order + 1

        normalized.append(
            {
                "order": order,
                "type": "text",
                "label": label,
                "self_ref": ref,
                "parent_ref": parent_ref,
                "page_no": page_no,
                "bbox": self._ref_parser.get_bbox(item),
                "text": text,
                **self._page_furniture_collector.metadata_for(page_no, page_furniture),
                **heading_tracker.metadata(),
            }
        )
        return order + 1

    def _normalize_group_item(
            self,
            normalized: list[dict[str, Any]],
            order: int,
            item: dict[str, Any],
            ref: str,
            index: dict[str, Any],
            page_furniture: dict[int, dict[str, str]],
            heading_tracker: HeadingTracker,
    ) -> int:
        group_text = self._group_text_collector.collect(ref, index)
        page_no = self._ref_parser.get_page_no(item)
        if not group_text:
            return order

        normalized.append(
            {
                "order": order,
                "type": "group",
                "label": item.get("label"),
                "self_ref": ref,
                "page_no": page_no,
                "bbox": self._ref_parser.get_bbox(item),
                "text": group_text,
                **self._page_furniture_collector.metadata_for(page_no, page_furniture),
                **heading_tracker.metadata(),
            }
        )
        return order + 1

    def _normalize_picture_item(
            self,
            normalized: list[dict[str, Any]],
            order: int,
            item: dict[str, Any],
            ref: str,
            idx: int,
            captions_by_picture: dict[str, str],
            page_furniture: dict[int, dict[str, str]],
            heading_tracker: HeadingTracker,
    ) -> int:
        page_no = self._ref_parser.get_page_no(item)
        image_path = str(self._config.picture_dir / f"picture_{idx}.png")
        normalized.append(
            {
                "order": order,
                "type": "picture",
                "label": item.get("label"),
                "self_ref": ref,
                "page_no": page_no,
                "bbox": self._ref_parser.get_bbox(item),
                "image_path": image_path,
                "caption": captions_by_picture.get(ref, ""),
                "vision_text": None,
                "vision_metadata": None,
                **self._page_furniture_collector.metadata_for(page_no, page_furniture),
                **heading_tracker.metadata(),
            }
        )
        return order + 1

    def _normalize_table_item(
            self,
            normalized: list[dict[str, Any]],
            order: int,
            item: dict[str, Any],
            ref: str,
            idx: int,
            page_furniture: dict[int, dict[str, str]],
            heading_tracker: HeadingTracker,
    ) -> int:
        page_no = self._ref_parser.get_page_no(item)
        image_path = str(self._config.table_dir / f"table_{idx}.png")

        table_md_path = self._config.table_dir / f"table_{idx}.md"

        print(table_md_path.exists(), table_md_path)

        if table_md_path.exists():
            table_markdown = table_md_path.read_text(encoding="utf-8").strip()
        else:
            table_markdown = self._table_markdown_extractor.extract(item)

        normalized.append(
            {
                "order": order,
                "type": "table",
                "label": item.get("label"),
                "self_ref": ref,
                "page_no": page_no,
                "bbox": self._ref_parser.get_bbox(item),
                "image_path": image_path,
                "table_markdown_path": str(table_md_path) if table_md_path.exists() else None,
                "table": item,
                "table_markdown": table_markdown,
                "table_vision": None,
                "vision_text": None,
                **self._page_furniture_collector.metadata_for(page_no, page_furniture),
                **heading_tracker.metadata(),
            }
        )

        return order + 1


def normalize_docling_json_with_heading_metadata(
        document_json_path: str,
        picture_dir: str,
        table_dir: str,
        include_headers_as_elements: bool = False,
) -> list[dict[str, Any]]:
    config = NormalizerConfig(
        document_json_path=Path(document_json_path),
        picture_dir=Path(picture_dir),
        table_dir=Path(table_dir),
        include_headers_as_elements=include_headers_as_elements,
    )
    normalizer = DoclingJsonNormalizer(config=config)
    return normalizer.normalize()
