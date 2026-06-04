from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agent.langgraph_memory import setup_langgraph_checkpointer
from controller.agent_controller import router as agent_router
from controller.chat_controller import router as chat_router
from controller.file_controller import router as file_router
from controller.task_controller import router as task_router
from controller.user_controller import router as user_router
from core.settings import settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Lifespan.

    Purpose:
        Implements lifespan for the application bootstrap and shared infrastructure
            layer.
    Args:
        app (FastAPI): FastAPI application instance whose startup and shutdown are being
            managed.
    Returns:
        AsyncIterator[None]: Streaming response or iterator that yields incremental
            output.
    Why Added:
        Provides a documented entry point for this module-level behavior and keeps
            callers
        from needing to know lower-level implementation details.
    """
    setup_langgraph_checkpointer()
    yield


app = FastAPI(
    title="AI Agent API",
    description="API for user registration and related services.",
    version="1.0.0",
    openapi_tags=[
        {"name": "users", "description": "User registration and user operations"},
        {"name": "chats", "description": "Chat creation and chat operations"},
        {"name": "files", "description": "File upload operations"},
        {"name": "agent", "description": "Agent chat operations"},
        {"name": "tasks", "description": "Celery task monitoring and retries"},
        {"name": "system", "description": "Service health and system endpoints"},
    ],
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(user_router)
app.include_router(chat_router)
app.include_router(file_router)
app.include_router(agent_router)
app.include_router(task_router)


@app.get("/health", tags=["system"], summary="Health check")
async def health() -> dict[str, str]:
    """
    Health.

    Purpose:
        Implements health for the application bootstrap and shared infrastructure layer.
    Args:
        None.
    Returns:
        dict[str, str]: Structured data produced by the operation.
    Why Added:
        Provides a documented entry point for this module-level behavior and keeps
            callers
        from needing to know lower-level implementation details.
    """
    return {"status": "ok"}
