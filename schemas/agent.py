from pydantic import BaseModel, ConfigDict, Field


class AgentChatRequest(BaseModel):
    chat_id: str = Field(description="Chat UUID")
    query: str = Field(min_length=1, description="User query")


class AgentFileSummaryItem(BaseModel):
    file_id: str
    title: str | None
    summary: dict | None


class AgentChatResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "chat_id": "8bb6bda8-b84f-4908-8a2c-06d3e016726c",
                "query": "Summarize files",
                "cached": True,
                "files": [
                    {
                        "file_id": "f4f2919c-77fb-4a23-bab1-cd4b95c7b7ca",
                        "title": "Document title",
                        "summary": {"sections": []},
                    }
                ],
            }
        }
    )

    chat_id: str
    query: str
    cached: bool
    files: list[AgentFileSummaryItem]
