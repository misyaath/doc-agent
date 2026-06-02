import re

from pydantic import BaseModel, ConfigDict, Field, field_validator


class UserRegisterRequest(BaseModel):
    """
    User Register Request.

    Purpose:
        Defines UserRegisterRequest in the Pydantic schema layer that validates API
            request and response payloads.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.

    Attributes:
        model_config (Any): Class-level value used by this class.
        full_name (str): Declared data field for this class.
        email (str): Declared data field for this class.
        password (str): Declared data field for this class.
    """

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
    def validate_full_name(cls: type, value: str) -> str:
        """
        Validate full name.

        Purpose:
            Implements validate_full_name for the Pydantic schema layer that validates
                API request and response payloads.
        Class:
            Belongs to UserRegisterRequest; uses that class state and dependencies when
                available.
        Args:
            cls (type): Class object used by validators or class-level helpers.
            value (str): Raw value being validated, normalized, or transformed.
        Returns:
            str: Result produced by the operation.
        Why Added:
            Centralizes this behavior inside UserRegisterRequest so related code remains
                cohesive and testable.
        """
        normalized = value.strip()
        if not normalized:
            raise ValueError("full_name cannot be empty")
        return normalized

    @field_validator("email")
    @classmethod
    def validate_email(cls: type, value: str) -> str:
        """
        Validate email.

        Purpose:
            Implements validate_email for the Pydantic schema layer that validates API
                request and response payloads.
        Class:
            Belongs to UserRegisterRequest; uses that class state and dependencies when
                available.
        Args:
            cls (type): Class object used by validators or class-level helpers.
            value (str): Raw value being validated, normalized, or transformed.
        Returns:
            str: Result produced by the operation.
        Why Added:
            Centralizes this behavior inside UserRegisterRequest so related code remains
                cohesive and testable.
        """
        normalized = value.strip().lower()
        if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", normalized):
            raise ValueError("Invalid email format")
        return normalized


class UserRegisterResponse(BaseModel):
    """
    User Register Response.

    Purpose:
        Defines UserRegisterResponse in the Pydantic schema layer that validates API
            request and response payloads.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.

    Attributes:
        model_config (Any): Class-level value used by this class.
        id (int): Declared data field for this class.
        full_name (str): Declared data field for this class.
        email (str): Declared data field for this class.
        is_active (bool): Declared data field for this class.
        is_verified (bool): Declared data field for this class.
    """

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
    """
    User Login Request.

    Purpose:
        Defines UserLoginRequest in the Pydantic schema layer that validates API request
            and response payloads.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.

    Attributes:
        model_config (Any): Class-level value used by this class.
        email (str): Declared data field for this class.
        password (str): Declared data field for this class.
    """

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
    def validate_email(cls: type, value: str) -> str:
        """
        Validate email.

        Purpose:
            Implements validate_email for the Pydantic schema layer that validates API
                request and response payloads.
        Class:
            Belongs to UserLoginRequest; uses that class state and dependencies when
                available.
        Args:
            cls (type): Class object used by validators or class-level helpers.
            value (str): Raw value being validated, normalized, or transformed.
        Returns:
            str: Result produced by the operation.
        Why Added:
            Centralizes this behavior inside UserLoginRequest so related code remains
                cohesive and testable.
        """
        normalized = value.strip().lower()
        if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", normalized):
            raise ValueError("Invalid email format")
        return normalized


class TokenResponse(BaseModel):
    """
    Token Response.

    Purpose:
        Defines TokenResponse in the Pydantic schema layer that validates API request
            and response payloads.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.

    Attributes:
        model_config (Any): Class-level value used by this class.
        access_token (str): Declared data field for this class.
        token_type (str): Declared data field for this class.
        expires_in (int): Declared data field for this class.
    """

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
