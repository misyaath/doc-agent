from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).resolve().parents[1] / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = Field(validation_alias="DATABASE_URL")
    jwt_secret: str = Field(validation_alias="JWT_SECRET")
    jwt_expires_seconds: int = Field(validation_alias="JWT_EXPIRES_SECONDS")

    redis_url: str = Field(validation_alias="REDIS_URL")
    qdrant_url: str = Field(validation_alias="QDRANT_URL")
    rag_collection_name: str = Field(validation_alias="RAG_COLLECTION_NAME")

    embedding_model: str = Field(validation_alias="EMBEDDING_MODEL")
    ollama_url: str = Field(validation_alias="OLLAMA_URL")
    text_model: str = Field(validation_alias="TEXT_MODEL")

    langsmith_tracing: bool = Field(validation_alias="LANGSMITH_TRACING")
    langsmith_endpoint: str = Field(validation_alias="LANGSMITH_ENDPOINT")
    langsmith_api_key: str | None = Field(validation_alias="LANGSMITH_API_KEY")
    langsmith_project: str = Field(validation_alias="LANGSMITH_PROJECT")
    langgraph_checkpoint_db_url: str = Field(validation_alias="LANGGRAPH_CHECKPOINT_DB_URL")

    @field_validator(
        "database_url",
        "jwt_secret",
        "redis_url",
        "qdrant_url",
        "rag_collection_name",
        "embedding_model",
        "ollama_url",
        "text_model",
        "langgraph_checkpoint_db_url",
        mode="before",
    )
    @classmethod
    def validate_not_empty(cls, value: str) -> str:
        if value is None or not str(value).strip():
            raise ValueError("Environment variable value cannot be empty")
        return str(value).strip()


settings = Settings()
