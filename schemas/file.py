from pydantic import BaseModel, ConfigDict


class FileUploadItemResponse(BaseModel):
    """
    File Upload Item Response.

    Purpose:
        Defines FileUploadItemResponse in the Pydantic schema layer that validates API
            request and response payloads.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.

    Attributes:
        file_id (str): Declared data field for this class.
        chat_id (str): Declared data field for this class.
        file_name (str): Declared data field for this class.
        unique_generated_name (str): Declared data field for this class.
        full_path (str): Declared data field for this class.
    """

    file_id: str
    chat_id: str
    file_name: str
    unique_generated_name: str
    full_path: str


class FileUploadResponse(BaseModel):
    """
    File Upload Response.

    Purpose:
        Defines FileUploadResponse in the Pydantic schema layer that validates API
            request and response payloads.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.

    Attributes:
        model_config (Any): Class-level value used by this class.
        user_id (int): Declared data field for this class.
        chat_id (str): Declared data field for this class.
        files (list[FileUploadItemResponse]): Declared data field for this class.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "user_id": 1,
                "chat_id": "8bb6bda8-b84f-4908-8a2c-06d3e016726c",
                "files": [
                    {
                        "file_id": "f4f2919c-77fb-4a23-bab1-cd4b95c7b7ca",
                        "chat_id": "8bb6bda8-b84f-4908-8a2c-06d3e016726c",
                        "file_name": "document.pdf",
                        "unique_generated_name": "19f3f617-4d15-4328-ad1a-8f2aa9d7f450.pdf",
                        "full_path": "/abs/path/uploads/1/19f3f617-4d15-4328-ad1a-8f2aa9d7f450.pdf",
                    }
                ],
            }
        }
    )

    user_id: int
    chat_id: str
    files: list[FileUploadItemResponse]
