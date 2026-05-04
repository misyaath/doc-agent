import re

from pydantic import BaseModel, ConfigDict, Field, field_validator


class UserRegisterRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "full_name": "Jane Doe",
                "email": "jane@example.com",
                "password": "strongpassword123",
            }
        }
    )

    full_name: str = Field(min_length=1, max_length=120, description="User full name")
    email: str = Field(min_length=3, max_length=255, description="User email address")
    password: str = Field(min_length=8, max_length=128, description="Raw password (min 8 chars)")

    @field_validator("full_name")
    @classmethod
    def validate_full_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("full_name cannot be empty")
        return normalized

    @field_validator("email")
    @classmethod
    def validate_email(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", normalized):
            raise ValueError("Invalid email format")
        return normalized


class UserRegisterResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": 1,
                "full_name": "Jane Doe",
                "email": "jane@example.com",
                "is_active": True,
                "is_verified": False,
            }
        }
    )

    id: int
    full_name: str
    email: str
    is_active: bool
    is_verified: bool


class UserLoginRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "email": "jane@example.com",
                "password": "strongpassword123",
            }
        }
    )

    email: str = Field(min_length=3, max_length=255, description="User email address")
    password: str = Field(min_length=8, max_length=128, description="Raw password")

    @field_validator("email")
    @classmethod
    def validate_email(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", normalized):
            raise ValueError("Invalid email format")
        return normalized


class TokenResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "bearer",
                "expires_in": 3600,
            }
        }
    )

    access_token: str
    token_type: str = "bearer"
    expires_in: int
