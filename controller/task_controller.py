import json
from typing import Any, cast

from celery.result import AsyncResult
from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from redis import Redis

from core.settings import settings
from worker import celery_app

router = APIRouter(prefix="/tasks", tags=["tasks"])


def get_redis_client() -> Redis:
    """
    Get redis client.

    Purpose:
        Implements get_redis_client for the HTTP controller layer that validates
            incoming requests, delegates to services, and shapes API responses.
    Args:
        None.
    Returns:
        Redis: Result produced by the operation.
    Why Added:
        Provides a documented entry point for this module-level behavior and keeps
            callers
        from needing to know lower-level implementation details.
    """
    return Redis.from_url(settings.redis_url, decode_responses=True)


class RetryTaskRequest(BaseModel):
    """
    Retry Task Request.

    Purpose:
        Defines RetryTaskRequest in the HTTP controller layer that validates incoming
            requests, delegates to services, and shapes API responses.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.

    Attributes:
        task_name (str | None): Declared data field for this class.
        args (list[Any]): Declared data field for this class.
        kwargs (dict[str, Any]): Declared data field for this class.
        countdown (int): Declared data field for this class.
    """

    task_name: str | None = Field(default=None, description="Celery task name override")
    args: list[Any] = Field(default_factory=list, description="Positional args for retry task")
    kwargs: dict[str, Any] = Field(default_factory=dict, description="Keyword args for retry task")
    countdown: int = Field(default=0, ge=0, description="Delay in seconds before retry")


def _collect_result_backend_tasks() -> list[dict[str, Any]]:
    """
    Collect result backend tasks.

    Purpose:
        Implements _collect_result_backend_tasks for the HTTP controller layer that
            validates incoming requests, delegates to services, and shapes API
            responses.
    Args:
        None.
    Returns:
        list[dict[str, Any]]: Structured data produced by the operation.
    Why Added:
        Provides a documented entry point for this module-level behavior and keeps
            callers
        from needing to know lower-level implementation details.
    """
    redis_client = get_redis_client()

    tasks: list[dict[str, Any]] = []

    for key in redis_client.scan_iter("celery-task-meta-*"):
        raw = redis_client.get(key)

        if not raw:
            continue

        try:
            data = json.loads(cast(str | bytes | bytearray, raw))
        except json.JSONDecodeError:
            continue

        task_id = key.replace("celery-task-meta-", "")

        tasks.append(
            {
                "task_id": task_id,
                "task_name": data.get("name"),
                "status": data.get("status"),
                "result": data.get("result"),
                "traceback": data.get("traceback"),
                "date_done": data.get("date_done"),
                "worker": None,
                "args": None,
                "kwargs": None,
                "eta": None,
            }
        )

    return tasks


def _match_filters(task: dict[str, Any], status_filter: set[str], name_filter: str | None) -> bool:
    """
    Match filters.

    Purpose:
        Implements _match_filters for the HTTP controller layer that validates incoming
            requests, delegates to services, and shapes API responses.
    Args:
        task (dict[str, Any]): Input value for the task parameter.
        status_filter (set[str]): Input value for the status filter parameter.
        name_filter (str | None): Input value for the name filter parameter.
    Returns:
        bool: True when the condition is satisfied; otherwise False.
    Why Added:
        Provides a documented entry point for this module-level behavior and keeps
            callers
        from needing to know lower-level implementation details.
    """
    if status_filter and (task.get("status") or "").upper() not in status_filter:
        return False
    return not (name_filter and task.get("task_name") != name_filter)


@router.get("/{task_id}", summary="Get task status by id")
def get_task_status(task_id: str) -> dict[str, Any]:
    """
    Get task status.

    Purpose:
        Implements get_task_status for the HTTP controller layer that validates incoming
            requests, delegates to services, and shapes API responses.
    Args:
        task_id (str): Input value for the task id parameter.
    Returns:
        dict[str, Any]: Structured data produced by the operation.
    Why Added:
        Provides a documented entry point for this module-level behavior and keeps
            callers
        from needing to know lower-level implementation details.
    """
    result: Any = AsyncResult(task_id, app=celery_app)
    response: dict[str, Any] = {"task_id": task_id, "status": result.status}

    if result.successful():
        response["result"] = result.result
    if result.failed():
        response["error"] = str(result.result)
        response["traceback"] = result.traceback

    return response


@router.get("", summary="List Celery tasks with filters")
def list_tasks(
    status_filter: str | None = Query(
        default=None,
        description="Comma-separated statuses: ACTIVE,RESERVED,SCHEDULED,SUCCESS,FAILURE,PENDING,RETRY",
    ),
    task_name: str | None = Query(default=None, description="Filter by exact Celery task name"),
) -> dict[str, Any]:
    """
    List tasks.

    Purpose:
        Implements list_tasks for the HTTP controller layer that validates incoming
            requests, delegates to services, and shapes API responses.
    Args:
        status_filter (str | None): Input value for the status filter parameter.
        task_name (str | None): Input value for the task name parameter.
    Returns:
        dict[str, Any]: Structured data produced by the operation.
    Why Added:
        Provides a documented entry point for this module-level behavior and keeps
            callers
        from needing to know lower-level implementation details.
    """
    allowed = {
        "ACTIVE",
        "RESERVED",
        "SCHEDULED",
        "SUCCESS",
        "FAILURE",
        "PENDING",
        "RETRY",
        "STARTED",
    }

    parsed_statuses = {s.strip().upper() for s in (status_filter or "").split(",") if s.strip()}

    invalid = parsed_statuses - allowed
    if invalid:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid status_filter values: {sorted(invalid)}",
        )

    backend_tasks = _collect_result_backend_tasks()

    all_tasks = backend_tasks

    # Deduplicate by task_id
    dedup: dict[str, dict[str, Any]] = {}
    for task in all_tasks:
        task_id = task.get("task_id")
        if task_id and task_id not in dedup:
            dedup[task_id] = task

    tasks = list(dedup.values())

    filtered = [task for task in tasks if _match_filters(task, parsed_statuses, task_name)]

    return {
        "total": len(filtered),
        "filters": {
            "status_filter": sorted(parsed_statuses),
            "task_name": task_name,
        },
        "tasks": filtered,
    }


@router.post("/{task_id}/retry", summary="Retry Celery task by id")
def retry_task(task_id: str, payload: RetryTaskRequest) -> dict[str, Any]:
    """
    Retry task.

    Purpose:
        Implements retry_task for the HTTP controller layer that validates incoming
            requests, delegates to services, and shapes API responses.
    Args:
        task_id (str): Input value for the task id parameter.
        payload (RetryTaskRequest): Validated request payload supplied by the API
            caller.
    Returns:
        dict[str, Any]: Structured data produced by the operation.
    Why Added:
        Provides a documented entry point for this module-level behavior and keeps
            callers
        from needing to know lower-level implementation details.
    """
    tasks = _collect_result_backend_tasks()
    source_task = next((task for task in tasks if task.get("task_id") == task_id), None)

    task_name = payload.task_name or (source_task or {}).get("task_name")
    args = payload.args or []
    kwargs = payload.kwargs or {}

    if not task_name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Unable to infer task_name from running tasks. "
                "Provide task_name (and args/kwargs if needed) in request body."
            ),
        )

    new_task = celery_app.send_task(
        task_name,
        args=args,
        kwargs=kwargs,
        countdown=payload.countdown,
    )

    return {
        "source_task_id": task_id,
        "retried_task_id": new_task.id,
        "task_name": task_name,
        "args": args,
        "kwargs": kwargs,
        "countdown": payload.countdown,
    }
