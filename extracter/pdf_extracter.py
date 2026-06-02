from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import ImageRefMode


@dataclass(frozen=True)
class ExtractionConfig:
    """
    Extraction Config.

    Purpose:
        Defines ExtractionConfig in the document extraction pipeline that normalizes
            PDFs, enriches visual content, builds RAG units, and indexes data.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.

    Attributes:
        pdf_path (Path): Declared data field for this class.
        output_dir (Path): Declared data field for this class.
        image_scale (float): Declared data field for this class.
        generate_picture_images (bool): Declared data field for this class.
        generate_page_images (bool): Declared data field for this class.
        generate_table_images (bool): Declared data field for this class.
        do_table_structure (bool): Declared data field for this class.
    """

    pdf_path: Path
    output_dir: Path
    image_scale: float = 2.0
    generate_picture_images: bool = True
    generate_page_images: bool = True
    generate_table_images: bool = True
    do_table_structure: bool = True

    @property
    def pictures_dir(self) -> Path:
        """
        Pictures dir.

        Purpose:
            Implements pictures_dir for the document extraction pipeline that normalizes
                PDFs, enriches visual content, builds RAG units, and indexes data.
        Class:
            Belongs to ExtractionConfig; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
        Returns:
            Path: Filesystem path resolved or created by the operation.
        Why Added:
            Centralizes this behavior inside ExtractionConfig so related code remains
                cohesive and testable.
        """
        return self.output_dir / "pictures"

    @property
    def tables_dir(self) -> Path:
        """
        Tables dir.

        Purpose:
            Implements tables_dir for the document extraction pipeline that normalizes
                PDFs, enriches visual content, builds RAG units, and indexes data.
        Class:
            Belongs to ExtractionConfig; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
        Returns:
            Path: Filesystem path resolved or created by the operation.
        Why Added:
            Centralizes this behavior inside ExtractionConfig so related code remains
                cohesive and testable.
        """
        return self.output_dir / "tables"


@dataclass(frozen=True)
class ExtractionResult:
    """
    Extraction Result.

    Purpose:
        Defines ExtractionResult in the document extraction pipeline that normalizes
            PDFs, enriches visual content, builds RAG units, and indexes data.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.

    Attributes:
        pictures_count (int): Declared data field for this class.
        tables_count (int): Declared data field for this class.
        texts_count (int): Declared data field for this class.
        json_path (Path): Declared data field for this class.
        markdown_path (Path): Declared data field for this class.
    """

    pictures_count: int
    tables_count: int
    texts_count: int
    json_path: Path
    markdown_path: Path


class IDocConverter(Protocol):
    """
    IDoc Converter.

    Purpose:
        Defines IDocConverter in the document extraction pipeline that normalizes PDFs,
            enriches visual content, builds RAG units, and indexes data.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.
    """

    def convert(self, pdf_path: Path) -> Any:
        """
        Convert.

        Purpose:
            Implements convert for the document extraction pipeline that normalizes
                PDFs, enriches visual content, builds RAG units, and indexes data.
        Class:
            Belongs to IDocConverter; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
            pdf_path (Path): Filesystem path to the source PDF document.
        Returns:
            Any: Result produced by the operation.
        Why Added:
            Centralizes this behavior inside IDocConverter so related code remains
                cohesive and testable.
        """
        ...


class DoclingConverter(IDocConverter):
    """
    Docling Converter.

    Purpose:
        Defines DoclingConverter in the document extraction pipeline that normalizes
            PDFs, enriches visual content, builds RAG units, and indexes data.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.
    """

    def __init__(self, config: ExtractionConfig) -> None:
        """
        Initialize the object with its required dependencies.

        Purpose:
            Implements __init__ for the document extraction pipeline that normalizes
                PDFs, enriches visual content, builds RAG units, and indexes data.
        Class:
            Belongs to DoclingConverter; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
            config (ExtractionConfig): Configuration object controlling this component.
        Returns:
            None: Performs work through side effects and does not return a value.
        Why Added:
            Centralizes this behavior inside DoclingConverter so related code remains
                cohesive and testable.
        """
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

    def convert(self, pdf_path: Path) -> Any:
        """
        Convert.

        Purpose:
            Implements convert for the document extraction pipeline that normalizes
                PDFs, enriches visual content, builds RAG units, and indexes data.
        Class:
            Belongs to DoclingConverter; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
            pdf_path (Path): Filesystem path to the source PDF document.
        Returns:
            Any: Result produced by the operation.
        Why Added:
            Centralizes this behavior inside DoclingConverter so related code remains
                cohesive and testable.
        """
        return self._converter.convert(str(pdf_path))


class DocAssetExporter:
    """
    Doc Asset Exporter.

    Purpose:
        Defines DocAssetExporter in the document extraction pipeline that normalizes
            PDFs, enriches visual content, builds RAG units, and indexes data.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.
    """

    def __init__(self, config: ExtractionConfig) -> None:
        """
        Initialize the object with its required dependencies.

        Purpose:
            Implements __init__ for the document extraction pipeline that normalizes
                PDFs, enriches visual content, builds RAG units, and indexes data.
        Class:
            Belongs to DocAssetExporter; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
            config (ExtractionConfig): Configuration object controlling this component.
        Returns:
            None: Performs work through side effects and does not return a value.
        Why Added:
            Centralizes this behavior inside DocAssetExporter so related code remains
                cohesive and testable.
        """
        self._config = config

    def prepare_directories(self) -> None:
        """
        Prepare directories.

        Purpose:
            Implements prepare_directories for the document extraction pipeline that
                normalizes PDFs, enriches visual content, builds RAG units, and indexes
                data.
        Class:
            Belongs to DocAssetExporter; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
        Returns:
            None: Performs work through side effects and does not return a value.
        Why Added:
            Centralizes this behavior inside DocAssetExporter so related code remains
                cohesive and testable.
        """
        self._config.output_dir.mkdir(parents=True, exist_ok=True)
        self._config.pictures_dir.mkdir(parents=True, exist_ok=True)
        self._config.tables_dir.mkdir(parents=True, exist_ok=True)

    def save_images(self, doc: Any) -> None:
        """
        Save images.

        Purpose:
            Implements save_images for the document extraction pipeline that normalizes
                PDFs, enriches visual content, builds RAG units, and indexes data.
        Class:
            Belongs to DocAssetExporter; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
            doc (Any): Docling document object produced by PDF conversion.
        Returns:
            None: Performs work through side effects and does not return a value.
        Why Added:
            Centralizes this behavior inside DocAssetExporter so related code remains
                cohesive and testable.
        """
        for i, picture in enumerate(doc.pictures):
            if picture.image:
                picture.image.pil_image.save(self._config.pictures_dir / f"picture_{i}.png")

        for i, table in enumerate(doc.tables):
            if table.image:
                table.image.pil_image.save(self._config.tables_dir / f"table_{i}.png")

    def save_documents(self, doc: Any) -> tuple[Path, Path]:
        """
        Save documents.

        Purpose:
            Implements save_documents for the document extraction pipeline that
                normalizes PDFs, enriches visual content, builds RAG units, and indexes
                data.
        Class:
            Belongs to DocAssetExporter; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
            doc (Any): Docling document object produced by PDF conversion.
        Returns:
            tuple[Path, Path]: Structured data produced by the operation.
        Why Added:
            Centralizes this behavior inside DocAssetExporter so related code remains
                cohesive and testable.
        """
        json_path = self._config.output_dir / "document.json"
        markdown_path = self._config.output_dir / "document.md"
        doc.save_as_json(json_path)
        doc.save_as_markdown(markdown_path, image_mode=ImageRefMode.REFERENCED)
        return json_path, markdown_path


class DoclingPdfExtractor:
    """
    Docling Pdf Extractor.

    Purpose:
        Defines DoclingPdfExtractor in the document extraction pipeline that normalizes
            PDFs, enriches visual content, builds RAG units, and indexes data.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.
    """

    def __init__(
        self,
        config: ExtractionConfig,
        converter: IDocConverter | None = None,
        exporter: DocAssetExporter | None = None,
    ) -> None:
        """
        Initialize the object with its required dependencies.

        Purpose:
            Implements __init__ for the document extraction pipeline that normalizes
                PDFs, enriches visual content, builds RAG units, and indexes data.
        Class:
            Belongs to DoclingPdfExtractor; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
            config (ExtractionConfig): Configuration object controlling this component.
            converter (IDocConverter | None): Input value for the converter parameter.
            exporter (DocAssetExporter | None): Input value for the exporter parameter.
        Returns:
            None: Performs work through side effects and does not return a value.
        Why Added:
            Centralizes this behavior inside DoclingPdfExtractor so related code remains
                cohesive and testable.
        """
        self._config = config
        self._converter = converter or DoclingConverter(config)
        self._exporter = exporter or DocAssetExporter(config)

    def _table_markdown_extractor(self, doc: Any) -> None:
        """
        Table markdown extractor.

        Purpose:
            Implements _table_markdown_extractor for the document extraction pipeline
                that normalizes PDFs, enriches visual content, builds RAG units, and
                indexes data.
        Class:
            Belongs to DoclingPdfExtractor; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
            doc (Any): Docling document object produced by PDF conversion.
        Returns:
            None: Performs work through side effects and does not return a value.
        Why Added:
            Centralizes this behavior inside DoclingPdfExtractor so related code remains
                cohesive and testable.
        """
        for i, table in enumerate(doc.tables):
            try:
                df = table.export_to_dataframe()
                markdown = df.to_markdown(index=False)
                (self._config.tables_dir / f"table_{i}.md").write_text(markdown, encoding="utf-8")
            except Exception:
                markdown = ""

    def run(self) -> ExtractionResult:
        """
        Run.

        Purpose:
            Implements run for the document extraction pipeline that normalizes PDFs,
                enriches visual content, builds RAG units, and indexes data.
        Class:
            Belongs to DoclingPdfExtractor; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
        Returns:
            ExtractionResult: Result produced by the operation.
        Why Added:
            Centralizes this behavior inside DoclingPdfExtractor so related code remains
                cohesive and testable.
        """
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
