import json
from pathlib import Path
import time
import asyncio

from sqlalchemy import select

from database import SessionLocal
from extracter import ExtractionConfig, DoclingPdfExtractor, normalize_docling_json_with_heading_metadata, \
    VisualElementEnricher, OrderedRagUnitBuilder, RagQdrantIngestionService, RagIndexingConfig
from models.file_process_stage import FileProcessStage
from models.file import File
from models.chat import Chat
from models.user import User
from worker import celery_app


@celery_app.task(name="process_uploaded_file")
def process_uploaded_file(
        file_id: str,
        chat_id: str,
        user_id: int,
        file_path: str,
        filename: str,
):
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    print(
        f"[file_extract] starting extraction: "
        f"chat_id={chat_id}, user_id={user_id}, filename={filename}"
    )
    print(f"Processing file: {filename}")
    print(f"file_id={file_id}, chat_id={chat_id}, user_id={user_id}")

    _upsert_stage_status(file_id=file_id, stage_name="extracted", status="processing")

    try:
        config = ExtractionConfig(
            pdf_path=Path(file_path),
            output_dir=Path(f"extracted_files/{chat_id}/{user_id}"),
        )
        extractor = DoclingPdfExtractor(config=config)
        result = extractor.run()

        print("pictures:", result.pictures_count)
        print("tables:", result.tables_count)
        print("texts:", result.texts_count)

        _upsert_stage_status(file_id=file_id, stage_name="extracted", status="done")
    except Exception:
        _upsert_stage_status(file_id=file_id, stage_name="extracted", status="failed")
        raise

    try:
        print(f"normalizing... {chat_id}-{file_id}")
        _upsert_stage_status(file_id=file_id, stage_name="normalizer", status="processing")

        file_base_path = Path(f"extracted_files/{chat_id}/{user_id}/")

        normalized = normalize_docling_json_with_heading_metadata(
            document_json_path=f"{file_base_path}/document.json",
            picture_dir=f"{file_base_path}/pictures",
            table_dir=f"{file_base_path}/tables",
            include_headers_as_elements=False,
        )

        Path(f"{file_base_path}/normalized_elements.json").write_text(
            json.dumps(normalized, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"normalized... {chat_id}-{file_id}")
        _upsert_stage_status(file_id=file_id, stage_name="normalizer", status="done")
    except Exception:
        _upsert_stage_status(file_id=file_id, stage_name="normalizer", status="failed")
        raise

    try:
        print(f"enriching... {chat_id}-{file_id}")
        _upsert_stage_status(file_id=file_id, stage_name="enriched", status="processing")
        enricher = VisualElementEnricher()
        enriched = enricher.enrich(normalized)
        rag_pipeline = OrderedRagUnitBuilder()
        rag = rag_pipeline.build(enriched)

        Path(f"{file_base_path}/rag_units.json").write_text(
            json.dumps(rag.rag_units, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print(f"enriched... {chat_id}-{file_id}")
        _upsert_stage_status(file_id=file_id, stage_name="enriched", status="done")
    except Exception:
        _upsert_stage_status(file_id=file_id, stage_name="enriched", status="failed")
        raise

    try:
        _upsert_stage_status(file_id=file_id, stage_name="embedding", status="processing")
        print(f"embedding... {chat_id}-{file_id}")
        service = RagQdrantIngestionService(
            RagIndexingConfig(
                source_file_path=path,
                collection_name="pdf_rag",
            )
        )
        service.ingest_from_file(f"{file_base_path}/rag_units.json")
        _upsert_stage_status(file_id=file_id, stage_name="embedding", status="processing")
        print(f"embedded... {chat_id}-{file_id}")
    except Exception:
        _upsert_stage_status(file_id=file_id, stage_name="embedding", status="failed")
        raise
    _upsert_stage_status(file_id=file_id, stage_name="done", status="done")


async def _upsert_stage_status(file_id: str, stage_name: str, status: str) -> None:
    async with SessionLocal() as session:
        stage = await session.scalar(
            select(FileProcessStage).where(
                FileProcessStage.file_id == file_id,
                FileProcessStage.stage == stage_name,
            )
        )

        if stage is None:
            stage_record = FileProcessStage(
                file_id=file_id,
                stage=stage_name,
                status=status,
            )
            session.add(stage_record)
        else:
            stage.status = status

        await session.commit()
