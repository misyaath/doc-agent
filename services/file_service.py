from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import Depends, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from database import get_db
from domain.file_process import FileStage, FileStageStatus
from repositories.chat_repository import ChatRepository
from repositories.file_repository import FileRepository
from schemas.file import FileUploadItemResponse, FileUploadResponse
from tasks.file_extracter import process_uploaded_file

logger = get_logger(__name__)


def _is_pdf_file(upload: UploadFile) -> bool:
    name = (upload.filename or "").lower()
    content_type = (upload.content_type or "").lower()
    return name.endswith(".pdf") and content_type in {"application/pdf", "application/x-pdf"}


class FileService:
    def __init__(self, chat_repository: ChatRepository, file_repository: FileRepository) -> None:
        self._chat_repository = chat_repository
        self._file_repository = file_repository

    async def upload_pdf_files(
            self,
            *,
            chat_id: str,
            files: list[UploadFile],
            user_id: int,
    ) -> FileUploadResponse:
        chat = await self._chat_repository.get_by_id_and_user_id(chat_id=chat_id, user_id=user_id)
        if not chat:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat not found")

        if not files:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="No files provided")

        upload_dir = Path("uploads") / str(user_id)
        upload_dir.mkdir(parents=True, exist_ok=True)

        created_records = []
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

            with full_path.open("wb") as out:
                out.write(first_chunk)
                while True:
                    chunk = await upload.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)

            record = self._file_repository.add_file_with_stage(
                file_id=file_id,
                chat_id=chat_id,
                file_name=upload.filename or generated_name,
                unique_generated_name=generated_name,
                full_path=str(full_path),
                stage=FileStage.UPLOADED.value,
                status=FileStageStatus.DONE.value,
            )
            created_records.append(record)
            logger.info("File uploaded", extra={"file_id": file_id, "chat_id": chat_id, "user_id": user_id})
            await upload.close()

        await self._file_repository.commit()
        for record in created_records:
            await self._file_repository.refresh(record)
            process_uploaded_file.delay(
                file_id=record.file_id,
                chat_id=record.chat_id,
                user_id=user_id,
                file_path=record.full_path,
                filename=record.file_name,
            )

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


def get_file_service(db: AsyncSession = Depends(get_db)) -> FileService:
    return FileService(
        chat_repository=ChatRepository(db),
        file_repository=FileRepository(db),
    )
