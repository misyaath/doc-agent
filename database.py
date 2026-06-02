from collections.abc import AsyncGenerator
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from core.settings import settings


class Base(DeclarativeBase):
    """
    Base.

    Purpose:
        Defines Base in the application bootstrap and shared infrastructure layer.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.
    """

    pass


def _normalize_database_url(raw_url: str) -> str:
    """
    Normalize database url.

    Purpose:
        Implements _normalize_database_url for the application bootstrap and shared
            infrastructure layer.
    Args:
        raw_url (str): Input value for the raw url parameter.
    Returns:
        str: Result produced by the operation.
    Why Added:
        Provides a documented entry point for this module-level behavior and keeps
            callers
        from needing to know lower-level implementation details.
    """
    parts = urlsplit(raw_url)
    query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if k != "schema"]
    url = urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return url


raw_database_url = settings.database_url

DATABASE_URL = _normalize_database_url(raw_database_url)

engine = create_async_engine(DATABASE_URL, future=True)
SessionLocal = async_sessionmaker(bind=engine, autoflush=False, autocommit=False, class_=AsyncSession)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Get db.

    Purpose:
        Implements get_db for the application bootstrap and shared infrastructure layer.
    Args:
        None.
    Returns:
        AsyncGenerator[AsyncSession, None]: Streaming response or iterator that yields
            incremental output.
    Why Added:
        Provides a documented entry point for this module-level behavior and keeps
            callers
        from needing to know lower-level implementation details.
    """
    async with SessionLocal() as session:
        yield session
