from pydantic import BaseModel, ConfigDict


class FileUploadItemResponse(BaseModel):
    file_id: str
    chat_id: str
    file_name: str
    unique_generated_name: str
    full_path: str


class FileUploadResponse(BaseModel):
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
