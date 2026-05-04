from controller.chat_controller import router as chat_router
from controller.file_controller import router as file_router
from fastapi import FastAPI

from controller.user_controller import router as user_router
from database import engine
from models import chat, file, file_process_stage  # noqa: F401
from models.user import Base

app = FastAPI(
    title="AI Agent API",
    description="API for user registration and related services.",
    version="1.0.0",
    openapi_tags=[
        {"name": "users", "description": "User registration and user operations"},
        {"name": "chats", "description": "Chat creation and chat operations"},
        {"name": "files", "description": "File upload operations"},
        {"name": "system", "description": "Service health and system endpoints"},
    ],
)
app.include_router(user_router)
app.include_router(chat_router)
app.include_router(file_router)


@app.get("/health", tags=["system"], summary="Health check")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.on_event("startup")
async def startup() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
