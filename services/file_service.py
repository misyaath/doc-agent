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
    """
    Is pdf file.

    Purpose:
        Implements _is_pdf_file for the business-service layer that coordinates
            repositories, security helpers, RAG execution, and API workflows.
    Args:
        upload (UploadFile): Input value for the upload parameter.
    Returns:
        bool: True when the condition is satisfied; otherwise False.
    Why Added:
        Provides a documented entry point for this module-level behavior and keeps
            callers
        from needing to know lower-level implementation details.
    """
    name = (upload.filename or "").lower()
    content_type = (upload.content_type or "").lower()
    return name.endswith(".pdf") and content_type in {"application/pdf", "application/x-pdf"}


class FileService:
    """
    File Service.

    Purpose:
        Defines FileService in the business-service layer that coordinates repositories,
            security helpers, RAG execution, and API workflows.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.
    """

    def __init__(self, chat_repository: ChatRepository, file_repository: FileRepository) -> None:
        """
        Initialize the object with its required dependencies.

        Purpose:
            Implements __init__ for the business-service layer that coordinates
                repositories, security helpers, RAG execution, and API workflows.
        Class:
            Belongs to FileService; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
            chat_repository (ChatRepository): Input value for the chat repository
                parameter.
            file_repository (FileRepository): Input value for the file repository
                parameter.
        Returns:
            None: Performs work through side effects and does not return a value.
        Why Added:
            Centralizes this behavior inside FileService so related code remains
                cohesive and testable.
        """
        self._chat_repository = chat_repository
        self._file_repository = file_repository

    async def upload_pdf_files(
        self,
        *,
        chat_id: str,
        files: list[UploadFile],
        user_id: int,
    ) -> FileUploadResponse:
        """
        Upload pdf files.

        Purpose:
            Implements upload_pdf_files for the business-service layer that coordinates
                repositories, security helpers, RAG execution, and API workflows.
        Class:
            Belongs to FileService; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
            chat_id (str): Chat/session identifier used to scope retrieval, tasks, or
                responses.
            files (list[UploadFile]): Input value for the files parameter.
            user_id (int): Authenticated user identifier used to scope the operation.
        Returns:
            FileUploadResponse: API response model returned to the client.
        Why Added:
            Centralizes this behavior inside FileService so related code remains
                cohesive and testable.
        """
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
    """
    Get file service.

    Purpose:
        Implements get_file_service for the business-service layer that coordinates
            repositories, security helpers, RAG execution, and API workflows.
    Args:
        db (AsyncSession): Database session used to read or persist application records.
    Returns:
        FileService: Domain or persistence object produced by the operation.
    Why Added:
        Provides a documented entry point for this module-level behavior and keeps
            callers
        from needing to know lower-level implementation details.
    """
    return FileService(
        chat_repository=ChatRepository(db),
        file_repository=FileRepository(db),
    )
