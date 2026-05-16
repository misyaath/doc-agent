from logging.config import fileConfig
import os
from pathlib import Path
import asyncio
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from sqlalchemy import engine_from_config
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context

from database import Base
from models.user import User
from models.chat import Chat
from models.file import File
from models.file_process_stage import FileProcessStage


# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config


def _load_dotenv(dotenv_path: Path) -> None:
    """Minimal .env loader for Alembic runtime without extra dependencies."""
    if not dotenv_path.exists():
        return

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _configure_sqlalchemy_url_from_env() -> None:
    project_root = Path(__file__).resolve().parents[1]
    _load_dotenv(project_root / ".env")

    database_url = os.getenv("DATABASE_URL")
    if database_url:
        parts = urlsplit(database_url)
        query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if k != "schema"]
        database_url = urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))

        if database_url.startswith("postgresql://"):
            # Prefer asyncpg when no explicit driver is set.
            database_url = database_url.replace(
                "postgresql://", "postgresql+asyncpg://", 1
            )
        config.set_main_option("sqlalchemy.url", database_url)


def _is_async_url(url: str) -> bool:
    return "+asyncpg://" in url

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

_configure_sqlalchemy_url_from_env()

# add your model's MetaData object here
# for 'autogenerate' support
# from myapp import mymodel
# target_metadata = mymodel.Base.metadata
target_metadata = Base.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    url = config.get_main_option("sqlalchemy.url")
    if _is_async_url(url):
        asyncio.run(_run_async_migrations())
        return

    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


async def _run_async_migrations() -> None:
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(_do_run_migrations)

    await connectable.dispose()


def _do_run_migrations(connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
