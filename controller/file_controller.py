import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from middleware.auth_middleware import get_current_user_id
from models.chat import Chat
from models.file import File as FileModel
from models.file_process_stage import FileProcessStage
from schemas.file import FileUploadItemResponse, FileUploadResponse

router = APIRouter(prefix="/files", tags=["files"])


def _is_pdf_file(upload: UploadFile) -> bool:
    name = (upload.filename or "").lower()
    content_type = (upload.content_type or "").lower()
    return name.endswith(".pdf") and content_type in {"application/pdf", "application/x-pdf"}


@router.post(
    "/upload",
    response_model=FileUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload PDF files",
    description="Uploads multiple PDF files for a chat. Route is protected with Bearer token auth.",
)
async def upload_pdf_files(
    chat_id: str = Form(...),
    files: list[UploadFile] = File(...),
    db: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
) -> FileUploadResponse:
    chat = await db.scalar(select(Chat).where(Chat.id == chat_id, Chat.user_id == user_id))
    if not chat:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat not found")

    if not files:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="No files provided")

    upload_dir = Path("uploads") / str(user_id)
    upload_dir.mkdir(parents=True, exist_ok=True)

    created_records: list[FileModel] = []
    for upload in files:
        if not _is_pdf_file(upload):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Only PDF files are allowed: {upload.filename}",
            )

        first_chunk = await upload.read(5)
        if first_chunk != b"%PDF-":
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid PDF content: {upload.filename}",
            )

        file_id = str(uuid.uuid4())
        generated_name = f"{uuid.uuid4()}.pdf"
        full_path = (upload_dir / generated_name).resolve()

        record = FileModel(
            file_id=file_id,
            chat_id=chat_id,
            file_name=upload.filename or generated_name,
            unique_generated_name=generated_name,
            full_path=str(full_path),
        )
        stage_record = FileProcessStage(file_id=file_id, stage="uploaded", status="started")
        db.add(record)
        db.add(stage_record)

        with full_path.open("wb") as out:
            out.write(first_chunk)
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
        stage_record.status = "Done"
        created_records.append(record)
        await upload.close()

    await db.commit()
    for record in created_records:
        await db.refresh(record)

    return FileUploadResponse(
        user_id=user_id,
        chat_id=chat_id,
        files=[
            FileUploadItemResponse(
                file_id=record.file_id,
                chat_id=record.chat_id,
                file_name=record.file_name,
                unique_generated_name=record.unique_generated_name,
                full_path=record.full_path,
            )
            for record in created_records
        ],
    )
