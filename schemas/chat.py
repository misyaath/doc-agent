from pydantic import BaseModel, ConfigDict


class ChatCreateResponse(BaseModel):
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
