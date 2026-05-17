from controller.agent_controller import router as agent_router
from controller.chat_controller import router as chat_router
from controller.file_controller import router as file_router
from controller.task_controller import router as task_router
from fastapi import FastAPI

from controller.user_controller import router as user_router

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
)
app.include_router(user_router)
app.include_router(chat_router)
app.include_router(file_router)
app.include_router(agent_router)
app.include_router(task_router)


@app.get("/health", tags=["system"], summary="Health check")
async def health() -> dict[str, str]:
    return {"status": "ok"}
