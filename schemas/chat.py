from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ChatCreateRequest(BaseModel):
    """
    Chat Create Request.

    Purpose:
        Defines ChatCreateRequest in the Pydantic schema layer that validates API
            request and response payloads.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.

    Attributes:
        model_config (Any): Class-level value used by this class.
        name (str): Declared data field for this class.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "Research notes",
            }
        }
    )

    name: str = Field(default="New Chat", min_length=1, max_length=120, description="Human-readable chat name")

    @field_validator("name")
    @classmethod
    def validate_name(cls: type, value: str) -> str:
        """
        Validate name.

        Purpose:
            Implements validate_name for the Pydantic schema layer that validates API
                request and response payloads.
        Class:
            Belongs to ChatCreateRequest; uses that class state and dependencies when
                available.
        Args:
            cls (type): Class object used by validators or class-level helpers.
            value (str): Raw value being validated, normalized, or transformed.
        Returns:
            str: Result produced by the operation.
        Why Added:
            Centralizes this behavior inside ChatCreateRequest so related code remains
                cohesive and testable.
        """
        normalized = value.strip()
        if not normalized:
            raise ValueError("name cannot be empty")
        return normalized


class ChatCreateResponse(BaseModel):
    """
    Chat Create Response.

    Purpose:
        Defines ChatCreateResponse in the Pydantic schema layer that validates API
            request and response payloads.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.

    Attributes:
        model_config (Any): Class-level value used by this class.
        user_id (int): Declared data field for this class.
        chat_id (str): Declared data field for this class.
        name (str): Declared data field for this class.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "user_id": 1,
                "chat_id": "8bb6bda8-b84f-4908-8a2c-06d3e016726c",
                "name": "Research notes",
            }
        }
    )

    user_id: int
    chat_id: str
    name: str


class ChatFileProcessStageResponse(BaseModel):
    """
    Chat File Process Stage Response.

    Purpose:
        Defines ChatFileProcessStageResponse in the Pydantic schema layer that validates
            API request and response payloads.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.

    Attributes:
        stage (str): Declared data field for this class.
        status (str): Declared data field for this class.
        updated_at (datetime): Declared data field for this class.
    """

    stage: str
    status: str
    updated_at: datetime


class ChatFileResponse(BaseModel):
    """
    Chat File Response.

    Purpose:
        Defines ChatFileResponse in the Pydantic schema layer that validates API request
            and response payloads.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.

    Attributes:
        file_id (str): Declared data field for this class.
        file_name (str): Declared data field for this class.
        title (str | None): Declared data field for this class.
        summary (Any | None): Declared data field for this class.
        process_status (str): Declared data field for this class.
        process_stages (list[ChatFileProcessStageResponse]): Declared data field for
            this class.
    """

    file_id: str
    file_name: str
    title: str | None
    summary: Any | None
    process_status: str
    process_stages: list[ChatFileProcessStageResponse]


class ChatListItemResponse(BaseModel):
    """
    Chat List Item Response.

    Purpose:
        Defines ChatListItemResponse in the Pydantic schema layer that validates API
            request and response payloads.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.

    Attributes:
        chat_id (str): Declared data field for this class.
        name (str): Declared data field for this class.
        created_at (datetime): Declared data field for this class.
        updated_at (datetime | None): Declared data field for this class.
        files (list[ChatFileResponse]): Declared data field for this class.
    """

    chat_id: str
    name: str
    created_at: datetime
    updated_at: datetime | None
    files: list[ChatFileResponse]


class ChatListResponse(BaseModel):
    """
    Chat List Response.

    Purpose:
        Defines ChatListResponse in the Pydantic schema layer that validates API request
            and response payloads.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.

    Attributes:
        model_config (Any): Class-level value used by this class.
        user_id (int): Declared data field for this class.
        chats (list[ChatListItemResponse]): Declared data field for this class.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "user_id": 1,
                "chats": [
                    {
                        "chat_id": "8bb6bda8-b84f-4908-8a2c-06d3e016726c",
                        "name": "Research notes",
                        "created_at": "2026-06-04T11:00:00",
                        "updated_at": None,
                        "files": [
                            {
                                "file_id": "f4f2919c-77fb-4a23-bab1-cd4b95c7b7ca",
                                "file_name": "document.pdf",
                                "title": "Document title",
                                "summary": [{"heading": "Intro", "summary": "Overview"}],
                                "process_status": "done",
                                "process_stages": [
                                    {
                                        "stage": "uploaded",
                                        "status": "done",
                                        "updated_at": "2026-06-04T11:01:00",
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
        }
    )

    user_id: int
    chats: list[ChatListItemResponse]


class ChatHistoryMessageResponse(BaseModel):
    """
    Chat History Message Response.

    Purpose:
        Defines ChatHistoryMessageResponse in the Pydantic schema layer that validates
            API request and response payloads.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.

    Attributes:
        role (str): Declared data field for this class.
        content (Any): Declared data field for this class.
    """

    role: str
    content: Any


class ChatDetailResponse(BaseModel):
    """
    Chat Detail Response.

    Purpose:
        Defines ChatDetailResponse in the Pydantic schema layer that validates API
            request and response payloads.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.

    Attributes:
        chat_id (str): Declared data field for this class.
        name (str): Declared data field for this class.
        created_at (datetime): Declared data field for this class.
        updated_at (datetime | None): Declared data field for this class.
        files (list[ChatFileResponse]): Declared data field for this class.
        history (list[ChatHistoryMessageResponse]): Declared data field for this class.
    """

    chat_id: str
    name: str
    created_at: datetime
    updated_at: datetime | None
    files: list[ChatFileResponse]
    history: list[ChatHistoryMessageResponse]


class ChatDeleteResponse(BaseModel):
    """
    Chat Delete Response.

    Purpose:
        Defines ChatDeleteResponse in the Pydantic schema layer that validates API
            request and response payloads.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.

    Attributes:
        chat_id (str): Declared data field for this class.
        deleted (bool): Declared data field for this class.
    """

    chat_id: str
    deleted: bool
