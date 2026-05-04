import hashlib
import hmac
import json
import os
import time
from base64 import b64decode, b64encode

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from models.user import User
from schemas.user import TokenResponse, UserLoginRequest, UserRegisterRequest, UserRegisterResponse

router = APIRouter(prefix="/users", tags=["users"]) 

JWT_SECRET = os.getenv("JWT_SECRET", "change-this-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRES_SECONDS = int(os.getenv("JWT_EXPIRES_SECONDS", "3600"))


def _hash_password(password: str) -> str:
    salt = os.urandom(16)
    derived = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000)
    return f"pbkdf2_sha256${b64encode(salt).decode()}${b64encode(derived).decode()}"


def _verify_password(password: str, password_hash: str) -> bool:
    try:
        algorithm, salt_b64, digest_b64 = password_hash.split("$", 2)
        if algorithm != "pbkdf2_sha256":
            return False
        salt = b64decode(salt_b64.encode())
        expected = b64decode(digest_b64.encode())
    except ValueError:
        return False

    actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000)
    return hmac.compare_digest(actual, expected)


def _b64url_encode(raw: bytes) -> str:
    return b64encode(raw).decode().replace("+", "-").replace("/", "_").rstrip("=")


def _create_access_token(subject: str) -> tuple[str, int]:
    now = int(time.time())
    exp = now + JWT_EXPIRES_SECONDS
    header = {"alg": JWT_ALGORITHM, "typ": "JWT"}
    payload = {"sub": subject, "iat": now, "exp": exp}
    header_part = _b64url_encode(json.dumps(header, separators=(",", ":")).encode())
    payload_part = _b64url_encode(json.dumps(payload, separators=(",", ":")).encode())
    signing_input = f"{header_part}.{payload_part}".encode()
    signature = hmac.new(JWT_SECRET.encode(), signing_input, hashlib.sha256).digest()
    token = f"{header_part}.{payload_part}.{_b64url_encode(signature)}"
    return token, JWT_EXPIRES_SECONDS


@router.post(
    "/register",
    response_model=UserRegisterResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user",
    description="Creates a user account with a unique email and securely hashed password.",
    responses={
        201: {"description": "User created"},
        409: {"description": "Email already registered"},
        422: {"description": "Validation error"},
    },
)
async def register_user(payload: UserRegisterRequest, db: AsyncSession = Depends(get_db)) -> UserRegisterResponse:
    email = payload.email

    existing = await db.scalar(select(User).where(User.email == email))
    if existing:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already registered")

    password_hash = _hash_password(payload.password)

    user = User(
        full_name=payload.full_name.strip(),
        email=email,
        password_hash=password_hash,
        is_active=True,
        is_verified=False,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)

    return UserRegisterResponse(
        id=user.id,
        full_name=user.full_name,
        email=user.email,
        is_active=user.is_active,
        is_verified=user.is_verified,
    )


@router.post(
    "/login",
    response_model=TokenResponse,
    status_code=status.HTTP_200_OK,
    summary="Login user",
    description="Authenticates user credentials and returns a JWT access token.",
    responses={
        200: {"description": "Login successful"},
        401: {"description": "Invalid credentials"},
        422: {"description": "Validation error"},
    },
)
async def login_user(payload: UserLoginRequest, db: AsyncSession = Depends(get_db)) -> TokenResponse:
    user = await db.scalar(select(User).where(User.email == payload.email))
    if not user or not _verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")

    token, expires_in = _create_access_token(str(user.id))
    return TokenResponse(access_token=token, expires_in=expires_in)
