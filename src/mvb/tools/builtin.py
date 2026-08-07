"""The tools the blueprint ships with.

Four, deliberately. A registry with thirty tools is a demo; a registry with four
that each do one legible thing is something a developer can extend on day one.
Each is a thin wrapper over the evaluation suite so the whole tool path is
exercisable with no external services.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..evalsuite.suite import load_suite
from ..schemas import Example
from .registry import registry

_SUITE_PATH = Path("eval/datasets/suite.jsonl")


@lru_cache(maxsize=1)
def _examples() -> list[Example]:
    if not _SUITE_PATH.exists():
        return []
    return load_suite(_SUITE_PATH)


def reset_cache() -> None:
    _examples.cache_clear()


@registry.register(
    name="search_events",
    description=(
        "Search the indexed frame corpus by free text and optional camera filter. "
        "Returns matching frame records with camera, timestamp, and the associated question."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Free-text search over prompts and answers."},
            "camera": {"type": "string", "description": "Restrict to one camera id, e.g. 'aisle-03'."},
            "limit": {"type": "integer", "description": "Max results, default 5.", "minimum": 1, "maximum": 50},
        },
        "required": ["query"],
    },
    tags=["retrieval"],
)
def search_events(query: str, camera: str = "", limit: int = 5) -> list[dict[str, Any]]:
    terms = [t for t in query.lower().split() if len(t) > 2]
    hits: list[tuple[int, Example]] = []
    for ex in _examples():
        if camera and not any(f.camera == camera for f in ex.frames):
            continue
        blob = f"{ex.prompt} {ex.target}".lower()
        score = sum(1 for t in terms if t in blob)
        if score:
            hits.append((score, ex))
    hits.sort(key=lambda p: (-p[0], p[1].id))
    return [
        {
            "id": ex.id,
            "camera": ex.frames[0].camera if ex.frames else "unknown",
            "timestamp_s": ex.frames[0].timestamp_s if ex.frames else 0.0,
            "question": ex.prompt,
            "tags": ex.tags,
            "score": score,
        }
        for score, ex in hits[: max(1, min(50, limit))]
    ]


@registry.register(
    name="get_frame",
    description="Fetch the frame metadata for a known record id.",
    parameters={
        "type": "object",
        "properties": {"id": {"type": "string", "description": "Record id from search_events."}},
        "required": ["id"],
    },
    tags=["retrieval"],
)
def get_frame(id: str) -> dict[str, Any]:  # noqa: A002 - matches the published schema
    for ex in _examples():
        if ex.id == id:
            return {
                "id": ex.id,
                "task": ex.task.value,
                "frames": [f.model_dump() for f in ex.frames],
                "tags": ex.tags,
                "difficulty": ex.difficulty,
            }
    raise KeyError(f"no record with id {id!r}")


@registry.register(
    name="describe_scene",
    description=(
        "Run the served VLM over a known record and return its description. "
        "Use after get_frame when the metadata alone does not answer the question."
    ),
    parameters={
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "Record id."},
            "adapter": {"type": "string", "description": "Adapter name; defaults to the configured one."},
        },
        "required": ["id"],
    },
    tags=["inference"],
)
def describe_scene(id: str, adapter: str = "") -> dict[str, Any]:  # noqa: A002
    from ..config import get_settings
    from ..nim.client import get_backend

    adapter = adapter or get_settings().train.adapter_name
    for ex in _examples():
        if ex.id == id:
            pred = get_backend().predict(ex, adapter=adapter)
            return {"id": id, "adapter": adapter, "description": pred.raw, "latency_ms": pred.latency_ms}
    raise KeyError(f"no record with id {id!r}")


@registry.register(
    name="count_objects",
    description="Count records matching an object term, optionally grouped by camera.",
    parameters={
        "type": "object",
        "properties": {
            "term": {"type": "string", "description": "Object term, e.g. 'forklift'."},
            "group_by_camera": {"type": "boolean", "description": "Return per-camera counts."},
        },
        "required": ["term"],
    },
    tags=["analytics"],
)
def count_objects(term: str, group_by_camera: bool = False) -> dict[str, Any]:
    term = term.lower()
    counts: dict[str, int] = {}
    total = 0
    for ex in _examples():
        if term not in f"{ex.prompt} {ex.target}".lower():
            continue
        total += 1
        cam = ex.frames[0].camera if ex.frames else "unknown"
        counts[cam] = counts.get(cam, 0) + 1
    return {"term": term, "total": total, "by_camera": counts if group_by_camera else None}


def tools_manifest() -> str:
    """Human-readable dump, used by `mvb tools` and the troubleshooting guide."""
    return json.dumps(registry.to_mcp_tools(), indent=2)
