"""The inference service.

This is the artefact the "45 minutes" claim is about: one process that exposes
inference, the tool registry, the agent loop, evaluation, and traces behind a
single port, with a health endpoint that tells you which backend you actually
got. Most of the setup time it removes was spent discovering that the model was
answering from a mock, or that a tool was registered but not reachable.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from ..config import get_settings
from ..evalsuite.suite import load_suite, run_suite
from ..nim.client import get_backend
from ..observability.trace import get_tracer
from ..schemas import Example
from ..tools import registry
from .agent import run_agent

_STATIC = Path(__file__).parent / "static"
_SUITE = Path("eval/datasets/suite.jsonl")

_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["backend"] = get_backend()
    _state["examples"] = load_suite(_SUITE) if _SUITE.exists() else []
    yield
    get_tracer().flush()


app = FastAPI(
    title="Metropolis VLM Blueprint",
    version="0.3.0",
    description="Post-training, evaluation, and serving reference for vision-language agents on NIM.",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    prompt: str
    frame_uri: str | None = None
    task: str = "scene_qa"
    adapter: str | None = None


class AskRequest(BaseModel):
    question: str
    max_steps: int = Field(default=6, ge=1, le=12)
    adapter: str | None = None


class ToolRequest(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


@app.get("/")
def index() -> Any:
    path = _STATIC / "index.html"
    if path.exists():
        return FileResponse(path)
    return JSONResponse({"service": "mvb", "docs": "/docs"})


@app.get("/api/health")
def health() -> dict[str, Any]:
    """Reports the backend actually in use, not the one that was configured.

    This endpoint exists because the most common support question during
    developer testing was some form of "is it really hitting the model" -- and
    the honest answer was often no.
    """
    cfg = get_settings()
    backend = _state.get("backend") or get_backend()
    return {
        "status": "ok",
        "backend": backend.name,
        "live_nim": cfg.nim.enabled,
        "vlm_model": cfg.nim.vlm_model,
        "adapter": cfg.train.adapter_name,
        "tools": registry.names(),
        "suite_loaded": len(_state.get("examples", [])),
        "tracing": cfg.observability.enabled,
    }


@app.post("/api/predict")
def predict(req: PredictRequest) -> dict[str, Any]:
    from ..schemas import Frame, Provenance, Split, TaskKind

    try:
        task = TaskKind(req.task)
    except ValueError as exc:
        raise HTTPException(400, f"unknown task {req.task!r}; expected one of {[t.value for t in TaskKind]}") from exc

    ex = Example(
        id="adhoc",
        task=task,
        split=Split.TEST,
        provenance=Provenance.HUMAN_LABELLED,
        frames=[Frame(uri=req.frame_uri)] if req.frame_uri else [],
        prompt=req.prompt,
        target="n/a",
    )
    adapter = req.adapter or get_settings().train.adapter_name
    tracer = get_tracer()
    with tracer.span("api.predict", adapter=adapter):
        pred = (_state.get("backend") or get_backend()).predict(ex, adapter=adapter)
    return pred.model_dump()


@app.post("/api/ask")
def ask(req: AskRequest) -> dict[str, Any]:
    run = run_agent(req.question, max_steps=req.max_steps, adapter=req.adapter)
    return {
        "question": run.question,
        "answer": run.answer,
        "truncated": run.truncated,
        "tools_used": run.tool_names(),
        "steps": [
            {
                "kind": s.kind,
                "tool": s.call.name if s.call else None,
                "arguments": s.call.arguments if s.call else None,
                "ok": s.result.ok if s.result else None,
                "error": s.result.error if s.result else None,
                "text": s.text,
            }
            for s in run.steps
        ],
        "trace": run.trace,
    }


@app.get("/api/tools")
def list_tools() -> dict[str, Any]:
    return {"tools": registry.to_mcp_tools(), "openai_format": registry.to_openai_tools()}


@app.post("/api/tools/call")
def call_tool(req: ToolRequest) -> dict[str, Any]:
    result = registry.call(req.name, req.arguments)
    if not result.ok and result.error and result.error.startswith("unknown tool"):
        raise HTTPException(404, result.error)
    return result.model_dump()


@app.get("/api/eval")
def evaluate(adapter: str = "lora-v3", limit: int = 200) -> dict[str, Any]:
    examples = _state.get("examples") or []
    if not examples:
        raise HTTPException(404, "no suite loaded; run `make data` first")
    report, _, _ = run_suite(examples[: max(1, limit)], adapter=adapter)
    return report.model_dump()


@app.get("/api/traces")
def traces() -> dict[str, Any]:
    tracer = get_tracer()
    return {"summary": tracer.summary(), "spans": [s.to_dict() for s in tracer.spans()[-200:]]}
