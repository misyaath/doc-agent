from celery import Celery

from core.settings import settings

REDIS_URL = settings.redis_url

celery_app = Celery(
    "agent-doc-extracter",
    broker=REDIS_URL,
    backend=REDIS_URL,
)

celery_app.conf.update(
    task_track_started=True,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="Asia/Colombo",
    enable_utc=True,
)

import tasks.file_extracter
