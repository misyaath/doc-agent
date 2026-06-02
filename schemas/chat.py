from pydantic import BaseModel, ConfigDict


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
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "user_id": 1,
                "chat_id": "8bb6bda8-b84f-4908-8a2c-06d3e016726c",
            }
        }
    )

    user_id: int
    chat_id: str
