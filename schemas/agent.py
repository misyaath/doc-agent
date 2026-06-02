from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class AgentChatRequest(BaseModel):
    chat_id: str = Field(description="Chat UUID")
    prompt: str = Field(
        min_length=1,
        description="User prompt",
        validation_alias=AliasChoices("prompt", "query"),
    )


class AgentChatResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "chat_id": "8bb6bda8-b84f-4908-8a2c-06d3e016726c",
                "prompt": "How image transform in to json ?",
                "answer": "The document explains the image-to-JSON pipeline in these steps...",
                "parsed_query": {
                    "query_type": "steps_or_process",
                    "domain": "technical",
                },
                "retrieved_chunks": [],
            }
        }
    )

    chat_id: str
    prompt: str
    answer: str
    parsed_query: dict[str, Any] | None = None
    retrieved_chunks: list[dict[str, Any]] = Field(default_factory=list)
    context: str | None = None
