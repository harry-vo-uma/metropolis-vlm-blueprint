"""Structured tracing.

Deliberately not OpenTelemetry. OTel is the right answer once there is a
collector to ship to; requiring one before a developer can see their first trace
is a setup step that loses people. This writes newline-delimited JSON spans to a
directory, which is greppable, diffable, and costs nothing to enable. The span
shape is OTel-compatible on purpose, so `export_otlp()` is a mapping rather than
a rewrite when the time comes.

What is traced is chosen from the failure analysis, not from what was easy: every
model call carries its adapter and token counts, and every tool call carries its
arguments and whether it succeeded. Those two facts answer most "why was this
slow" and "why did the agent give up" questions without a debugger.
"""

from __future__ import annotations

import json
import os
import random
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import get_settings


@dataclass
class Span:
    name: str
    trace_id: str
    span_id: str
    parent_id: str | None = None
    start_ns: int = field(default_factory=time.time_ns)
    end_ns: int | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"
    error: str | None = None

    def duration_ms(self) -> float:
        if self.end_ns is None:
            return 0.0
        return (self.end_ns - self.start_ns) / 1e6

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "start_ns": self.start_ns,
            "duration_ms": round(self.duration_ms(), 3),
            "status": self.status,
            "error": self.error,
            "attributes": self.attributes,
        }


class Tracer:
    """Thread-local span stack, append-only JSONL sink."""

    def __init__(self, trace_dir: str | Path | None = None) -> None:
        cfg = get_settings().observability
        self.enabled = cfg.enabled
        self.sample_rate = cfg.sample_rate
        self.redact = cfg.redact_prompts
        self.dir = Path(trace_dir or cfg.trace_dir)
        self._local = threading.local()
        self._lock = threading.Lock()
        self._buffer: list[Span] = []

    # -- stack -------------------------------------------------------------

    @property
    def _stack(self) -> list[Span]:
        if not hasattr(self._local, "stack"):
            self._local.stack = []
        return self._local.stack

    def current(self) -> Span | None:
        return self._stack[-1] if self._stack else None

    @contextmanager
    def span(self, name: str, **attributes: Any):
        if not self.enabled or random.random() > self.sample_rate:
            yield _NullSpan()
            return

        parent = self.current()
        span = Span(
            name=name,
            trace_id=parent.trace_id if parent else uuid.uuid4().hex,
            span_id=uuid.uuid4().hex[:16],
            parent_id=parent.span_id if parent else None,
            attributes=self._clean(attributes),
        )
        self._stack.append(span)
        try:
            yield span
        except Exception as exc:
            span.status = "error"
            span.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            span.end_ns = time.time_ns()
            self._stack.pop()
            with self._lock:
                self._buffer.append(span)

    def _clean(self, attrs: dict[str, Any]) -> dict[str, Any]:
        if not self.redact:
            return dict(attrs)
        return {
            k: ("<redacted>" if k in ("prompt", "response", "raw") else v)
            for k, v in attrs.items()
        }

    # -- sink --------------------------------------------------------------

    def spans(self) -> list[Span]:
        with self._lock:
            return list(self._buffer)

    def flush(self, filename: str | None = None) -> Path | None:
        with self._lock:
            batch, self._buffer = self._buffer, []
        if not batch:
            return None
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.dir / (filename or f"trace-{int(time.time())}-{os.getpid()}.jsonl")
        with open(path, "a", encoding="utf-8") as fh:
            for span in batch:
                fh.write(json.dumps(span.to_dict()) + "\n")
        return path

    def summary(self) -> dict[str, Any]:
        """Per-operation latency roll-up. This is the whole point of tracing at
        this scale: which stage is slow, and how often does it fail."""
        by_name: dict[str, list[Span]] = {}
        for span in self.spans():
            by_name.setdefault(span.name, []).append(span)
        out: dict[str, Any] = {}
        for name, spans in sorted(by_name.items()):
            durations = sorted(s.duration_ms() for s in spans)
            out[name] = {
                "count": len(spans),
                "errors": sum(1 for s in spans if s.status == "error"),
                "p50_ms": round(durations[len(durations) // 2], 2),
                "p95_ms": round(durations[min(len(durations) - 1, int(0.95 * (len(durations) - 1)))], 2),
                "total_ms": round(sum(durations), 2),
            }
        return out

    def export_otlp(self) -> list[dict[str, Any]]:
        """Shape spans as OTLP resource spans. Not shipped anywhere -- the point
        is that the migration is a mapping, not a rewrite."""
        return [
            {
                "traceId": s.trace_id,
                "spanId": s.span_id,
                "parentSpanId": s.parent_id or "",
                "name": s.name,
                "startTimeUnixNano": str(s.start_ns),
                "endTimeUnixNano": str(s.end_ns or s.start_ns),
                "status": {"code": 2 if s.status == "error" else 1, "message": s.error or ""},
                "attributes": [{"key": k, "value": {"stringValue": str(v)}} for k, v in s.attributes.items()],
            }
            for s in self.spans()
        ]


class _NullSpan:
    """No-op span so callers never need to branch on whether tracing is on.

    It has to be *structurally* compatible with `Span`, not merely silent. An
    earlier version only swallowed attribute assignment, so `span.attributes[k]
    = v` -- a read followed by a mutation, not an assignment -- raised
    `AttributeError` the moment tracing was turned off. That failure appeared
    only under `MVB_TRACE_ENABLED=0`, which is what CI and anyone shaving
    overhead uses: the path that was supposed to be the cheap one was the only
    one that crashed.
    """

    def __init__(self) -> None:
        # A real dict, because callers index into it. Writes land here and are
        # discarded when the span goes out of scope, which is the point.
        object.__setattr__(self, "attributes", {})

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "attributes":
            object.__setattr__(self, name, value)

    def set(self, **kwargs: Any) -> None:
        self.attributes.update(kwargs)


_TRACER: Tracer | None = None


def get_tracer() -> Tracer:
    global _TRACER
    if _TRACER is None:
        _TRACER = Tracer()
    return _TRACER


def reset_tracer() -> None:
    global _TRACER
    _TRACER = None
