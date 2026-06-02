from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class NormalizerConfig:
    """
    Normalizer Config.

    Purpose:
        Defines NormalizerConfig in the document extraction pipeline that normalizes
            PDFs, enriches visual content, builds RAG units, and indexes data.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.

    Attributes:
        document_json_path (Path): Declared data field for this class.
        picture_dir (Path): Declared data field for this class.
        table_dir (Path): Declared data field for this class.
        include_headers_as_elements (bool): Declared data field for this class.
    """

    document_json_path: Path
    picture_dir: Path = Path("docling_output/pictures")
    table_dir: Path = Path("docling_output/tables")
    include_headers_as_elements: bool = False


class DoclingJsonLoader:
    """
    Docling Json Loader.

    Purpose:
        Defines DoclingJsonLoader in the document extraction pipeline that normalizes
            PDFs, enriches visual content, builds RAG units, and indexes data.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.
    """

    def load(self, path: Path) -> dict[str, Any]:
        """
        Load.

        Purpose:
            Implements load for the document extraction pipeline that normalizes PDFs,
                enriches visual content, builds RAG units, and indexes data.
        Class:
            Belongs to DoclingJsonLoader; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
            path (Path): Filesystem path used as input or output for the operation.
        Returns:
            dict[str, Any]: Structured data produced by the operation.
        Why Added:
            Centralizes this behavior inside DoclingJsonLoader so related code remains
                cohesive and testable.
        """
        return json.loads(path.read_text(encoding="utf-8"))


class DoclingRefParser:
    """
    Docling Ref Parser.

    Purpose:
        Defines DoclingRefParser in the document extraction pipeline that normalizes
            PDFs, enriches visual content, builds RAG units, and indexes data.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.
    """

    @staticmethod
    def get_ref(obj: dict[str, Any]) -> str | None:
        """
        Get ref.

        Purpose:
            Implements get_ref for the document extraction pipeline that normalizes
                PDFs, enriches visual content, builds RAG units, and indexes data.
        Class:
            Belongs to DoclingRefParser; uses that class state and dependencies when
                available.
        Args:
            obj (dict[str, Any]): Input value for the obj parameter.
        Returns:
            str | None: Result produced by the operation.
        Why Added:
            Centralizes this behavior inside DoclingRefParser so related code remains
                cohesive and testable.
        """
        return obj.get("$ref")

    @staticmethod
    def get_parent_ref(item: dict[str, Any]) -> str | None:
        """
        Get parent ref.

        Purpose:
            Implements get_parent_ref for the document extraction pipeline that
                normalizes PDFs, enriches visual content, builds RAG units, and indexes
                data.
        Class:
            Belongs to DoclingRefParser; uses that class state and dependencies when
                available.
        Args:
            item (dict[str, Any]): Input value for the item parameter.
        Returns:
            str | None: Result produced by the operation.
        Why Added:
            Centralizes this behavior inside DoclingRefParser so related code remains
                cohesive and testable.
        """
        return (item.get("parent") or {}).get("$ref")

    @staticmethod
    def get_page_no(item: dict[str, Any]) -> int | None:
        """
        Get page no.

        Purpose:
            Implements get_page_no for the document extraction pipeline that normalizes
                PDFs, enriches visual content, builds RAG units, and indexes data.
        Class:
            Belongs to DoclingRefParser; uses that class state and dependencies when
                available.
        Args:
            item (dict[str, Any]): Input value for the item parameter.
        Returns:
            int | None: Result produced by the operation.
        Why Added:
            Centralizes this behavior inside DoclingRefParser so related code remains
                cohesive and testable.
        """
        prov = item.get("prov") or []
        if not prov:
            return None
        return prov[0].get("page_no")

    @staticmethod
    def get_bbox(item: dict[str, Any]) -> dict[str, Any] | None:
        """
        Get bbox.

        Purpose:
            Implements get_bbox for the document extraction pipeline that normalizes
                PDFs, enriches visual content, builds RAG units, and indexes data.
        Class:
            Belongs to DoclingRefParser; uses that class state and dependencies when
                available.
        Args:
            item (dict[str, Any]): Input value for the item parameter.
        Returns:
            dict[str, Any] | None: Structured data produced by the operation.
        Why Added:
            Centralizes this behavior inside DoclingRefParser so related code remains
                cohesive and testable.
        """
        prov = item.get("prov") or []
        if not prov:
            return None
        return prov[0].get("bbox")

    @staticmethod
    def ref_to_parts(ref: str) -> tuple[str, int]:
        """
        Ref to parts.

        Purpose:
            Implements ref_to_parts for the document extraction pipeline that normalizes
                PDFs, enriches visual content, builds RAG units, and indexes data.
        Class:
            Belongs to DoclingRefParser; uses that class state and dependencies when
                available.
        Args:
            ref (str): Input value for the ref parameter.
        Returns:
            tuple[str, int]: Structured data produced by the operation.
        Why Added:
            Centralizes this behavior inside DoclingRefParser so related code remains
                cohesive and testable.
        """
        parts = ref.strip("#/").split("/")
        return parts[0], int(parts[1])


class DoclingIndexBuilder:
    """
    Docling Index Builder.

    Purpose:
        Defines DoclingIndexBuilder in the document extraction pipeline that normalizes
            PDFs, enriches visual content, builds RAG units, and indexes data.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.
    """

    def build(self, doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
        """
        Build.

        Purpose:
            Implements build for the document extraction pipeline that normalizes PDFs,
                enriches visual content, builds RAG units, and indexes data.
        Class:
            Belongs to DoclingIndexBuilder; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
            doc (dict[str, Any]): Docling document object produced by PDF conversion.
        Returns:
            dict[str, dict[str, Any]]: Structured data produced by the operation.
        Why Added:
            Centralizes this behavior inside DoclingIndexBuilder so related code remains
                cohesive and testable.
        """
        index: dict[str, dict[str, Any]] = {}
        for section in ["texts", "pictures", "tables", "groups"]:
            for item in doc.get(section, []):
                self_ref = item.get("self_ref")
                if self_ref:
                    index[self_ref] = item
        return index


class TableMarkdownExtractor:
    """
    Table Markdown Extractor.

    Purpose:
        Defines TableMarkdownExtractor in the document extraction pipeline that
            normalizes PDFs, enriches visual content, builds RAG units, and indexes
            data.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.
    """

    def extract(self, table_item: dict[str, Any]) -> str:
        """
        Extract.

        Purpose:
            Implements extract for the document extraction pipeline that normalizes
                PDFs, enriches visual content, builds RAG units, and indexes data.
        Class:
            Belongs to TableMarkdownExtractor; uses that class state and dependencies
                when available.
        Args:
            self (Self): Current instance that owns the operation state.
            table_item (dict[str, Any]): Input value for the table item parameter.
        Returns:
            str: Result produced by the operation.
        Why Added:
            Centralizes this behavior inside TableMarkdownExtractor so related code
                remains cohesive and testable.
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
        """
        Cells to markdown.

        Purpose:
            Implements _cells_to_markdown for the document extraction pipeline that
                normalizes PDFs, enriches visual content, builds RAG units, and indexes
                data.
        Class:
            Belongs to TableMarkdownExtractor; uses that class state and dependencies
                when available.
        Args:
            self (Self): Current instance that owns the operation state.
            cells (list[dict[str, Any]]): Input value for the cells parameter.
        Returns:
            str: Result produced by the operation.
        Why Added:
            Centralizes this behavior inside TableMarkdownExtractor so related code
                remains cohesive and testable.
        """
        rows: dict[int, dict[int, str]] = {}

        for cell in cells:
            row_idx = cell.get("start_row_offset_idx") or cell.get("row") or cell.get("row_idx") or 0
            col_idx = cell.get("start_col_offset_idx") or cell.get("col") or cell.get("col_idx") or 0

            text = cell.get("text") or cell.get("content") or cell.get("value") or ""

            text = str(text).replace("\n", " ").strip()

            rows.setdefault(int(row_idx), {})[int(col_idx)] = text

        if not rows:
            return ""

        max_col = max(col_idx for row in rows.values() for col_idx in row)

        ordered_rows: list[list[str]] = []

        for row_idx in sorted(rows.keys()):
            row = rows[row_idx]
            ordered_rows.append([row.get(col_idx, "") for col_idx in range(max_col + 1)])

        return self._grid_to_markdown(ordered_rows)

    def _grid_to_markdown(self, grid: list[list[Any]]) -> str:
        """
        Grid to markdown.

        Purpose:
            Implements _grid_to_markdown for the document extraction pipeline that
                normalizes PDFs, enriches visual content, builds RAG units, and indexes
                data.
        Class:
            Belongs to TableMarkdownExtractor; uses that class state and dependencies
                when available.
        Args:
            self (Self): Current instance that owns the operation state.
            grid (list[list[Any]]): Input value for the grid parameter.
        Returns:
            str: Result produced by the operation.
        Why Added:
            Centralizes this behavior inside TableMarkdownExtractor so related code
                remains cohesive and testable.
        """
        if not grid:
            return ""

        clean_grid = [[str(cell or "").replace("\n", " ").strip() for cell in row] for row in grid]

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
    """
    Page Furniture Collector.

    Purpose:
        Defines PageFurnitureCollector in the document extraction pipeline that
            normalizes PDFs, enriches visual content, builds RAG units, and indexes
            data.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.
    """

    def __init__(self, ref_parser: DoclingRefParser) -> None:
        """
        Initialize the object with its required dependencies.

        Purpose:
            Implements __init__ for the document extraction pipeline that normalizes
                PDFs, enriches visual content, builds RAG units, and indexes data.
        Class:
            Belongs to PageFurnitureCollector; uses that class state and dependencies
                when available.
        Args:
            self (Self): Current instance that owns the operation state.
            ref_parser (DoclingRefParser): Input value for the ref parser parameter.
        Returns:
            None: Performs work through side effects and does not return a value.
        Why Added:
            Centralizes this behavior inside PageFurnitureCollector so related code
                remains cohesive and testable.
        """
        self._ref_parser = ref_parser

    def collect(self, doc: dict[str, Any]) -> dict[int, dict[str, str]]:
        """
        Collect.

        Purpose:
            Implements collect for the document extraction pipeline that normalizes
                PDFs, enriches visual content, builds RAG units, and indexes data.
        Class:
            Belongs to PageFurnitureCollector; uses that class state and dependencies
                when available.
        Args:
            self (Self): Current instance that owns the operation state.
            doc (dict[str, Any]): Docling document object produced by PDF conversion.
        Returns:
            dict[int, dict[str, str]]: Structured data produced by the operation.
        Why Added:
            Centralizes this behavior inside PageFurnitureCollector so related code
                remains cohesive and testable.
        """
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
        """
        Metadata for.

        Purpose:
            Implements metadata_for for the document extraction pipeline that normalizes
                PDFs, enriches visual content, builds RAG units, and indexes data.
        Class:
            Belongs to PageFurnitureCollector; uses that class state and dependencies
                when available.
        Args:
            page_no (int | None): Input value for the page no parameter.
            page_furniture (dict[int, dict[str, str]]): Input value for the page
                furniture parameter.
        Returns:
            dict[str, str]: Structured data produced by the operation.
        Why Added:
            Centralizes this behavior inside PageFurnitureCollector so related code
                remains cohesive and testable.
        """
        if not page_no:
            return {"page_header": "", "page_footer": ""}
        return page_furniture.get(page_no, {"page_header": "", "page_footer": ""})


class HeadingTracker:
    """
    Heading Tracker.

    Purpose:
        Defines HeadingTracker in the document extraction pipeline that normalizes PDFs,
            enriches visual content, builds RAG units, and indexes data.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.
    """

    def __init__(self) -> None:
        """
        Initialize the object with its required dependencies.

        Purpose:
            Implements __init__ for the document extraction pipeline that normalizes
                PDFs, enriches visual content, builds RAG units, and indexes data.
        Class:
            Belongs to HeadingTracker; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
        Returns:
            None: Performs work through side effects and does not return a value.
        Why Added:
            Centralizes this behavior inside HeadingTracker so related code remains
                cohesive and testable.
        """
        self._stack: list[dict[str, Any]] = []

    def update(self, heading_text: str, level: int | None, self_ref: str) -> None:
        """
        Update.

        Purpose:
            Implements update for the document extraction pipeline that normalizes PDFs,
                enriches visual content, builds RAG units, and indexes data.
        Class:
            Belongs to HeadingTracker; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
            heading_text (str): Input value for the heading text parameter.
            level (int | None): Input value for the level parameter.
            self_ref (str): Input value for the self ref parameter.
        Returns:
            None: Performs work through side effects and does not return a value.
        Why Added:
            Centralizes this behavior inside HeadingTracker so related code remains
                cohesive and testable.
        """
        heading_level = level or 1
        self._stack = [h for h in self._stack if h["level"] < heading_level]
        self._stack.append({"text": heading_text, "level": heading_level, "self_ref": self_ref})

    def metadata(self) -> dict[str, Any]:
        """
        Metadata.

        Purpose:
            Implements metadata for the document extraction pipeline that normalizes
                PDFs, enriches visual content, builds RAG units, and indexes data.
        Class:
            Belongs to HeadingTracker; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
        Returns:
            dict[str, Any]: Structured data produced by the operation.
        Why Added:
            Centralizes this behavior inside HeadingTracker so related code remains
                cohesive and testable.
        """
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
    """
    Picture Caption Collector.

    Purpose:
        Defines PictureCaptionCollector in the document extraction pipeline that
            normalizes PDFs, enriches visual content, builds RAG units, and indexes
            data.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.
    """

    def __init__(self, ref_parser: DoclingRefParser) -> None:
        """
        Initialize the object with its required dependencies.

        Purpose:
            Implements __init__ for the document extraction pipeline that normalizes
                PDFs, enriches visual content, builds RAG units, and indexes data.
        Class:
            Belongs to PictureCaptionCollector; uses that class state and dependencies
                when available.
        Args:
            self (Self): Current instance that owns the operation state.
            ref_parser (DoclingRefParser): Input value for the ref parser parameter.
        Returns:
            None: Performs work through side effects and does not return a value.
        Why Added:
            Centralizes this behavior inside PictureCaptionCollector so related code
                remains cohesive and testable.
        """
        self._ref_parser = ref_parser

    def collect(self, doc: dict[str, Any]) -> dict[str, str]:
        """
        Collect.

        Purpose:
            Implements collect for the document extraction pipeline that normalizes
                PDFs, enriches visual content, builds RAG units, and indexes data.
        Class:
            Belongs to PictureCaptionCollector; uses that class state and dependencies
                when available.
        Args:
            self (Self): Current instance that owns the operation state.
            doc (dict[str, Any]): Docling document object produced by PDF conversion.
        Returns:
            dict[str, str]: Structured data produced by the operation.
        Why Added:
            Centralizes this behavior inside PictureCaptionCollector so related code
                remains cohesive and testable.
        """
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
    """
    Group Text Collector.

    Purpose:
        Defines GroupTextCollector in the document extraction pipeline that normalizes
            PDFs, enriches visual content, builds RAG units, and indexes data.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.
    """

    def __init__(self, ref_parser: DoclingRefParser) -> None:
        """
        Initialize the object with its required dependencies.

        Purpose:
            Implements __init__ for the document extraction pipeline that normalizes
                PDFs, enriches visual content, builds RAG units, and indexes data.
        Class:
            Belongs to GroupTextCollector; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
            ref_parser (DoclingRefParser): Input value for the ref parser parameter.
        Returns:
            None: Performs work through side effects and does not return a value.
        Why Added:
            Centralizes this behavior inside GroupTextCollector so related code remains
                cohesive and testable.
        """
        self._ref_parser = ref_parser

    def collect(self, group_ref: str, index: dict[str, Any]) -> str:
        """
        Collect.

        Purpose:
            Implements collect for the document extraction pipeline that normalizes
                PDFs, enriches visual content, builds RAG units, and indexes data.
        Class:
            Belongs to GroupTextCollector; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
            group_ref (str): Input value for the group ref parameter.
            index (dict[str, Any]): Input value for the index parameter.
        Returns:
            str: Result produced by the operation.
        Why Added:
            Centralizes this behavior inside GroupTextCollector so related code remains
                cohesive and testable.
        """
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
    """
    Docling Json Normalizer.

    Purpose:
        Defines DoclingJsonNormalizer in the document extraction pipeline that
            normalizes PDFs, enriches visual content, builds RAG units, and indexes
            data.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.
    """

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
        """
        Initialize the object with its required dependencies.

        Purpose:
            Implements __init__ for the document extraction pipeline that normalizes
                PDFs, enriches visual content, builds RAG units, and indexes data.
        Class:
            Belongs to DoclingJsonNormalizer; uses that class state and dependencies
                when available.
        Args:
            self (Self): Current instance that owns the operation state.
            config (NormalizerConfig): Configuration object controlling this component.
            loader (DoclingJsonLoader | None): Input value for the loader parameter.
            ref_parser (DoclingRefParser | None): Input value for the ref parser
                parameter.
            index_builder (DoclingIndexBuilder | None): Input value for the index
                builder parameter.
            page_furniture_collector (PageFurnitureCollector | None): Input value for
                the page furniture collector parameter.
            picture_caption_collector (PictureCaptionCollector | None): Input value for
                the picture caption collector parameter.
            group_text_collector (GroupTextCollector | None): Input value for the group
                text collector parameter.
            table_markdown_extractor (TableMarkdownExtractor | None): Input value for
                the table markdown extractor parameter.
        Returns:
            None: Performs work through side effects and does not return a value.
        Why Added:
            Centralizes this behavior inside DoclingJsonNormalizer so related code
                remains cohesive and testable.
        """
        self._config = config
        self._loader = loader or DoclingJsonLoader()
        self._ref_parser = ref_parser or DoclingRefParser()
        self._index_builder = index_builder or DoclingIndexBuilder()
        self._page_furniture_collector = page_furniture_collector or PageFurnitureCollector(self._ref_parser)
        self._picture_caption_collector = picture_caption_collector or PictureCaptionCollector(self._ref_parser)
        self._group_text_collector = group_text_collector or GroupTextCollector(self._ref_parser)
        self._table_markdown_extractor = table_markdown_extractor or TableMarkdownExtractor()

    def normalize(self) -> list[dict[str, Any]]:
        """
        Normalize.

        Purpose:
            Implements normalize for the document extraction pipeline that normalizes
                PDFs, enriches visual content, builds RAG units, and indexes data.
        Class:
            Belongs to DoclingJsonNormalizer; uses that class state and dependencies
                when available.
        Args:
            self (Self): Current instance that owns the operation state.
        Returns:
            list[dict[str, Any]]: Structured data produced by the operation.
        Why Added:
            Centralizes this behavior inside DoclingJsonNormalizer so related code
                remains cohesive and testable.
        """
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
                order = self._normalize_text_item(normalized, order, item, ref, page_furniture, heading_tracker)
            elif section == "groups":
                order = self._normalize_group_item(normalized, order, item, ref, index, page_furniture, heading_tracker)
            elif section == "pictures":
                order = self._normalize_picture_item(
                    normalized, order, item, ref, idx, captions_by_picture, page_furniture, heading_tracker
                )
            elif section == "tables":
                order = self._normalize_table_item(normalized, order, item, ref, idx, page_furniture, heading_tracker)

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
        """
        Normalize text item.

        Purpose:
            Implements _normalize_text_item for the document extraction pipeline that
                normalizes PDFs, enriches visual content, builds RAG units, and indexes
                data.
        Class:
            Belongs to DoclingJsonNormalizer; uses that class state and dependencies
                when available.
        Args:
            self (Self): Current instance that owns the operation state.
            normalized (list[dict[str, Any]]): Input value for the normalized parameter.
            order (int): Input value for the order parameter.
            item (dict[str, Any]): Input value for the item parameter.
            ref (str): Input value for the ref parameter.
            page_furniture (dict[int, dict[str, str]]): Input value for the page
                furniture parameter.
            heading_tracker (HeadingTracker): Input value for the heading tracker
                parameter.
        Returns:
            int: Result produced by the operation.
        Why Added:
            Centralizes this behavior inside DoclingJsonNormalizer so related code
                remains cohesive and testable.
        """
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
        """
        Normalize group item.

        Purpose:
            Implements _normalize_group_item for the document extraction pipeline that
                normalizes PDFs, enriches visual content, builds RAG units, and indexes
                data.
        Class:
            Belongs to DoclingJsonNormalizer; uses that class state and dependencies
                when available.
        Args:
            self (Self): Current instance that owns the operation state.
            normalized (list[dict[str, Any]]): Input value for the normalized parameter.
            order (int): Input value for the order parameter.
            item (dict[str, Any]): Input value for the item parameter.
            ref (str): Input value for the ref parameter.
            index (dict[str, Any]): Input value for the index parameter.
            page_furniture (dict[int, dict[str, str]]): Input value for the page
                furniture parameter.
            heading_tracker (HeadingTracker): Input value for the heading tracker
                parameter.
        Returns:
            int: Result produced by the operation.
        Why Added:
            Centralizes this behavior inside DoclingJsonNormalizer so related code
                remains cohesive and testable.
        """
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
        """
        Normalize picture item.

        Purpose:
            Implements _normalize_picture_item for the document extraction pipeline that
                normalizes PDFs, enriches visual content, builds RAG units, and indexes
                data.
        Class:
            Belongs to DoclingJsonNormalizer; uses that class state and dependencies
                when available.
        Args:
            self (Self): Current instance that owns the operation state.
            normalized (list[dict[str, Any]]): Input value for the normalized parameter.
            order (int): Input value for the order parameter.
            item (dict[str, Any]): Input value for the item parameter.
            ref (str): Input value for the ref parameter.
            idx (int): Input value for the idx parameter.
            captions_by_picture (dict[str, str]): Input value for the captions by
                picture parameter.
            page_furniture (dict[int, dict[str, str]]): Input value for the page
                furniture parameter.
            heading_tracker (HeadingTracker): Input value for the heading tracker
                parameter.
        Returns:
            int: Result produced by the operation.
        Why Added:
            Centralizes this behavior inside DoclingJsonNormalizer so related code
                remains cohesive and testable.
        """
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
        """
        Normalize table item.

        Purpose:
            Implements _normalize_table_item for the document extraction pipeline that
                normalizes PDFs, enriches visual content, builds RAG units, and indexes
                data.
        Class:
            Belongs to DoclingJsonNormalizer; uses that class state and dependencies
                when available.
        Args:
            self (Self): Current instance that owns the operation state.
            normalized (list[dict[str, Any]]): Input value for the normalized parameter.
            order (int): Input value for the order parameter.
            item (dict[str, Any]): Input value for the item parameter.
            ref (str): Input value for the ref parameter.
            idx (int): Input value for the idx parameter.
            page_furniture (dict[int, dict[str, str]]): Input value for the page
                furniture parameter.
            heading_tracker (HeadingTracker): Input value for the heading tracker
                parameter.
        Returns:
            int: Result produced by the operation.
        Why Added:
            Centralizes this behavior inside DoclingJsonNormalizer so related code
                remains cohesive and testable.
        """
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
    """
    Normalize docling json with heading metadata.

    Purpose:
        Implements normalize_docling_json_with_heading_metadata for the document
            extraction pipeline that normalizes PDFs, enriches visual content, builds
            RAG units, and indexes data.
    Args:
        document_json_path (str): Input value for the document json path parameter.
        picture_dir (str): Input value for the picture dir parameter.
        table_dir (str): Input value for the table dir parameter.
        include_headers_as_elements (bool): Input value for the include headers as
            elements parameter.
    Returns:
        list[dict[str, Any]]: Structured data produced by the operation.
    Why Added:
        Provides a documented entry point for this module-level behavior and keeps
            callers
        from needing to know lower-level implementation details.
    """
    config = NormalizerConfig(
        document_json_path=Path(document_json_path),
        picture_dir=Path(picture_dir),
        table_dir=Path(table_dir),
        include_headers_as_elements=include_headers_as_elements,
    )
    normalizer = DoclingJsonNormalizer(config=config)
    return normalizer.normalize()
