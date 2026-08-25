"""A deliberately small, deterministic dependency-aware task runner."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Callable, Iterable


class TaskState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class Task:
    name: str
    action: Callable[[], dict]
    dependencies: tuple[str, ...] = ()


@dataclass
class TaskResult:
    name: str
    state: TaskState = TaskState.PENDING
    details: dict = field(default_factory=dict)


class DagError(ValueError):
    """The task graph is malformed."""


def topological_order(tasks: Iterable[Task]) -> list[str]:
    task_list = list(tasks)
    by_name = {task.name: task for task in task_list}
    if len(by_name) != len(task_list):
        raise DagError("Task names must be unique")
    for task in task_list:
        unknown = set(task.dependencies) - set(by_name)
        if unknown:
            raise DagError(f"Task {task.name} has unknown dependencies: {', '.join(sorted(unknown))}")
    remaining = {task.name: set(task.dependencies) for task in task_list}
    ordered: list[str] = []
    while remaining:
        ready = sorted(name for name, dependencies in remaining.items() if not dependencies)
        if not ready:
            raise DagError("Task graph contains a dependency cycle")
        ordered.extend(ready)
        for name in ready:
            del remaining[name]
        completed = set(ready)
        for dependencies in remaining.values():
            dependencies.difference_update(completed)
    return ordered


def run_dag(tasks: Iterable[Task], *, max_workers: int = 4) -> dict[str, TaskResult]:
    """Run ready tasks concurrently and block descendants of failed tasks."""

    task_list = list(tasks)
    topological_order(task_list)  # validate before any task can execute
    by_name = {task.name: task for task in task_list}
    results = {name: TaskResult(name) for name in by_name}
    pending = set(by_name)

    def execute(task: Task) -> dict:
        started = datetime.now(timezone.utc)
        try:
            details = task.action()
        except BaseException as exc:
            ended = datetime.now(timezone.utc)
            return {"_failure": f"{type(exc).__name__}: {exc}", **getattr(exc, "details", {}), "started_utc": started.isoformat().replace("+00:00", "Z"), "ended_utc": ended.isoformat().replace("+00:00", "Z"), "duration_seconds": (ended - started).total_seconds()}
        ended = datetime.now(timezone.utc)
        return {**details, "started_utc": started.isoformat().replace("+00:00", "Z"), "ended_utc": ended.isoformat().replace("+00:00", "Z"), "duration_seconds": (ended - started).total_seconds()}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        while pending:
            newly_blocked = [
                name for name in pending
                if any(results[dependency].state in {TaskState.FAILED, TaskState.BLOCKED} for dependency in by_name[name].dependencies)
            ]
            for name in newly_blocked:
                failed_dependencies = sorted(dependency for dependency in by_name[name].dependencies if results[dependency].state in {TaskState.FAILED, TaskState.BLOCKED})
                results[name] = TaskResult(name, TaskState.BLOCKED, {"reason": "dependency_failed", "dependencies": failed_dependencies})
                pending.remove(name)
            ready = sorted(name for name in pending if all(results[dependency].state == TaskState.SUCCEEDED for dependency in by_name[name].dependencies))
            if not ready:
                if newly_blocked:
                    # A dependency may have become blocked only in this pass.
                    # Re-evaluate the remaining tasks so blocking propagates to
                    # every descendant before declaring the DAG unrunnable.
                    continue
                if pending:
                    raise DagError("No runnable tasks remain")
                break
            futures: dict[Future[dict], str] = {}
            for name in ready:
                results[name].state = TaskState.RUNNING
                pending.remove(name)
                futures[executor.submit(execute, by_name[name])] = name
            for future in as_completed(futures):
                name = futures[future]
                details = future.result()
                failure = details.pop("_failure", None)
                results[name] = TaskResult(name, TaskState.FAILED if failure else TaskState.SUCCEEDED, {**details, **({"reason": failure} if failure else {})})
    return results
