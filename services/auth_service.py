from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
from base64 import b64decode, b64encode

from core.settings import settings

JWT_ALGORITHM = "HS256"
PBKDF2_ITERATIONS = 200_000


def hash_password(password: str) -> str:
    """
    Hash password.

    Purpose:
        Implements hash_password for the business-service layer that coordinates
            repositories, security helpers, RAG execution, and API workflows.
    Args:
        password (str): Plain-text password supplied for hashing or verification.
    Returns:
        str: Result produced by the operation.
    Why Added:
        Provides a documented entry point for this module-level behavior and keeps
            callers
        from needing to know lower-level implementation details.
    """
    salt = os.urandom(16)
    derived = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return f"pbkdf2_sha256${b64encode(salt).decode()}${b64encode(derived).decode()}"


def verify_password(password: str, password_hash: str) -> bool:
    """
    Verify password.

    Purpose:
        Implements verify_password for the business-service layer that coordinates
            repositories, security helpers, RAG execution, and API workflows.
    Args:
        password (str): Plain-text password supplied for hashing or verification.
        password_hash (str): Stored password hash used for credential verification.
    Returns:
        bool: True when the condition is satisfied; otherwise False.
    Why Added:
        Provides a documented entry point for this module-level behavior and keeps
            callers
        from needing to know lower-level implementation details.
    """
    try:
        algorithm, salt_b64, digest_b64 = password_hash.split("$", 2)
        if algorithm != "pbkdf2_sha256":
            return False
        salt = b64decode(salt_b64.encode())
        expected = b64decode(digest_b64.encode())
    except ValueError:
        return False

    actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return hmac.compare_digest(actual, expected)


def _b64url_encode(raw: bytes) -> str:
    """
    B64url encode.

    Purpose:
        Implements _b64url_encode for the business-service layer that coordinates
            repositories, security helpers, RAG execution, and API workflows.
    Args:
        raw (bytes): Input value for the raw parameter.
    Returns:
        str: Result produced by the operation.
    Why Added:
        Provides a documented entry point for this module-level behavior and keeps
            callers
        from needing to know lower-level implementation details.
    """
    return b64encode(raw).decode().replace("+", "-").replace("/", "_").rstrip("=")


def _b64url_decode(value: str) -> bytes:
    """
    B64url decode.

    Purpose:
        Implements _b64url_decode for the business-service layer that coordinates
            repositories, security helpers, RAG execution, and API workflows.
    Args:
        value (str): Raw value being validated, normalized, or transformed.
    Returns:
        bytes: Result produced by the operation.
    Why Added:
        Provides a documented entry point for this module-level behavior and keeps
            callers
        from needing to know lower-level implementation details.
    """
    padding = "=" * ((4 - len(value) % 4) % 4)
    return b64decode((value + padding).replace("-", "+").replace("_", "/"))


def create_access_token(subject: str) -> tuple[str, int]:
    """
    Create access token.

    Purpose:
        Implements create_access_token for the business-service layer that coordinates
            repositories, security helpers, RAG execution, and API workflows.
    Args:
        subject (str): Input value for the subject parameter.
    Returns:
        tuple[str, int]: Structured data produced by the operation.
    Why Added:
        Provides a documented entry point for this module-level behavior and keeps
            callers
        from needing to know lower-level implementation details.
    """
    now = int(time.time())
    exp = now + settings.jwt_expires_seconds
    header = {"alg": JWT_ALGORITHM, "typ": "JWT"}
    payload = {"sub": subject, "iat": now, "exp": exp}
    header_part = _b64url_encode(json.dumps(header, separators=(",", ":")).encode())
    payload_part = _b64url_encode(json.dumps(payload, separators=(",", ":")).encode())
    signing_input = f"{header_part}.{payload_part}".encode()
    signature = hmac.new(settings.jwt_secret.encode(), signing_input, hashlib.sha256).digest()
    token = f"{header_part}.{payload_part}.{_b64url_encode(signature)}"
    return token, settings.jwt_expires_seconds


def verify_jwt(token: str) -> dict:
    """
    Verify jwt.

    Purpose:
        Implements verify_jwt for the business-service layer that coordinates
            repositories, security helpers, RAG execution, and API workflows.
    Args:
        token (str): JWT access token supplied by the caller.
    Returns:
        dict: Structured data produced by the operation.
    Why Added:
        Provides a documented entry point for this module-level behavior and keeps
            callers
        from needing to know lower-level implementation details.
    """
    header_b64, payload_b64, signature_b64 = token.split(".")
    signing_input = f"{header_b64}.{payload_b64}".encode()
    expected = hmac.new(settings.jwt_secret.encode(), signing_input, hashlib.sha256).digest()
    actual = _b64url_decode(signature_b64)
    if not hmac.compare_digest(expected, actual):
        raise ValueError("Invalid signature")

    payload = json.loads(_b64url_decode(payload_b64))
    exp = int(payload.get("exp", 0))
    if exp <= int(time.time()):
        raise ValueError("Token expired")
    return payload
