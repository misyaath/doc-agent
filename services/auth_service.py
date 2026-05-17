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
    salt = os.urandom(16)
    derived = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return f"pbkdf2_sha256${b64encode(salt).decode()}${b64encode(derived).decode()}"


def verify_password(password: str, password_hash: str) -> bool:
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
    return b64encode(raw).decode().replace("+", "-").replace("/", "_").rstrip("=")


def _b64url_decode(value: str) -> bytes:
    padding = "=" * ((4 - len(value) % 4) % 4)
    return b64decode((value + padding).replace("-", "+").replace("_", "/"))


def create_access_token(subject: str) -> tuple[str, int]:
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
