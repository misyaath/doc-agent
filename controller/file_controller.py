from fastapi import APIRouter, Depends, File, Form, UploadFile, status

from middleware.auth_middleware import get_current_user_id
from schemas.file import FileUploadResponse
from services.file_service import FileService, get_file_service

router = APIRouter(prefix="/files", tags=["files"])


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
    service: FileService = Depends(get_file_service),
    user_id: int = Depends(get_current_user_id),
) -> FileUploadResponse:
    """
    Upload pdf files.

    Purpose:
        Implements upload_pdf_files for the HTTP controller layer that validates
            incoming requests, delegates to services, and shapes API responses.
    Args:
        chat_id (str): Chat/session identifier used to scope retrieval, tasks, or
            responses.
        files (list[UploadFile]): Input value for the files parameter.
        service (FileService): Injected service dependency that performs the business
            operation.
        user_id (int): Authenticated user identifier used to scope the operation.
    Returns:
        FileUploadResponse: API response model returned to the client.
    Why Added:
        Provides a documented entry point for this module-level behavior and keeps
            callers
        from needing to know lower-level implementation details.
    """
    return await service.upload_pdf_files(
        chat_id=chat_id,
        files=files,
        user_id=user_id,
    )
