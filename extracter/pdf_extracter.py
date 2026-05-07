from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import ImageRefMode


@dataclass(frozen=True)
class ExtractionConfig:
    pdf_path: Path
    output_dir: Path
    image_scale: float = 2.0
    generate_picture_images: bool = True
    generate_page_images: bool = True
    generate_table_images: bool = True
    do_table_structure: bool = True

    @property
    def pictures_dir(self) -> Path:
        return self.output_dir / "pictures"

    @property
    def tables_dir(self) -> Path:
        return self.output_dir / "tables"


@dataclass(frozen=True)
class ExtractionResult:
    pictures_count: int
    tables_count: int
    texts_count: int
    json_path: Path
    markdown_path: Path


class IDocConverter(Protocol):
    def convert(self, pdf_path: Path):
        ...


class DoclingConverter(IDocConverter):
    def __init__(self, config: ExtractionConfig) -> None:
        options = PdfPipelineOptions()

        options.accelerator_options = AcceleratorOptions(
            num_threads=8,
            device=AcceleratorDevice.CUDA,
        )
        options.generate_picture_images = config.generate_picture_images
        options.generate_page_images = config.generate_page_images
        options.generate_table_images = config.generate_table_images
        options.images_scale = config.image_scale
        options.do_table_structure = config.do_table_structure

        self._converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=options),
            }
        )

    def convert(self, pdf_path: Path):
        return self._converter.convert(str(pdf_path))


class DocAssetExporter:
    def __init__(self, config: ExtractionConfig) -> None:
        self._config = config

    def prepare_directories(self) -> None:
        self._config.output_dir.mkdir(parents=True, exist_ok=True)
        self._config.pictures_dir.mkdir(parents=True, exist_ok=True)
        self._config.tables_dir.mkdir(parents=True, exist_ok=True)

    def save_images(self, doc) -> None:
        for i, picture in enumerate(doc.pictures):
            if picture.image:
                picture.image.pil_image.save(self._config.pictures_dir / f"picture_{i}.png")

        for i, table in enumerate(doc.tables):
            if table.image:
                table.image.pil_image.save(self._config.tables_dir / f"table_{i}.png")

    def save_documents(self, doc) -> tuple[Path, Path]:
        json_path = self._config.output_dir / "document.json"
        markdown_path = self._config.output_dir / "document.md"
        doc.save_as_json(json_path)
        doc.save_as_markdown(markdown_path, image_mode=ImageRefMode.REFERENCED)
        return json_path, markdown_path


class DoclingPdfExtractor:
    def __init__(
            self,
            config: ExtractionConfig,
            converter: IDocConverter | None = None,
            exporter: DocAssetExporter | None = None,
    ) -> None:
        self._config = config
        self._converter = converter or DoclingConverter(config)
        self._exporter = exporter or DocAssetExporter(config)

    def _table_markdown_extractor(self, doc):
        for i, table in enumerate(doc.tables):
            try:
                df = table.export_to_dataframe()
                markdown = df.to_markdown(index=False)
                (self._config.tables_dir / f"table_{i}.md").write_text(markdown, encoding="utf-8")
            except Exception:
                markdown = ""

    def run(self) -> ExtractionResult:
        if not self._config.pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {self._config.pdf_path}")

        self._exporter.prepare_directories()
        result = self._converter.convert(self._config.pdf_path)
        doc = result.document
        self._exporter.save_images(doc)
        json_path, markdown_path = self._exporter.save_documents(doc)
        self._table_markdown_extractor(doc)

        return ExtractionResult(
            pictures_count=len(doc.pictures),
            tables_count=len(doc.tables),
            texts_count=len(doc.texts),
            json_path=json_path,
            markdown_path=markdown_path,
        )
