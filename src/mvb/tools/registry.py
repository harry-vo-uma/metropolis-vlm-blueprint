"""Tool registry.

One registration path, three consumers: the local agent loop, the OpenAI-style
function-calling payload sent to NIM, and the MCP server. Registering a tool
three times in three shapes is the single most common source of drift in agent
codebases -- the schema in the prompt stops matching the function that actually
runs, and the model starts getting confidently rejected arguments.

So a tool is declared once, with a JSON Schema, and every surface is *derived*.
`to_openai_tools()` and `to_mcp_tools()` are pure projections of the same
registry; there is nowhere for them to disagree.
"""

from __future__ import annotations

import inspect
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..schemas import ToolResult


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., Any]
    read_only: bool = True
    """Declared so a caller can run in a dry-run mode without an allowlist that
    has to be maintained separately from the tools themselves."""

    tags: list[str] = field(default_factory=list)

    def required(self) -> list[str]:
        return list(self.parameters.get("required", []))


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        *,
        read_only: bool = True,
        tags: list[str] | None = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
            if name in self._tools:
                raise ValueError(f"tool {name!r} already registered")
            self._validate(name, parameters, fn)
            self._tools[name] = ToolSpec(
                name=name,
                description=description,
                parameters=parameters,
                fn=fn,
                read_only=read_only,
                tags=tags or [],
            )
            return fn

        return deco

    @staticmethod
    def _validate(name: str, parameters: dict[str, Any], fn: Callable[..., Any]) -> None:
        """Fail at import time if the schema and the signature disagree.

        This check is the whole reason the registry exists. Without it the
        mismatch surfaces at runtime as a model that "hallucinates" an argument,
        and hours get spent blaming the model for a typo in a dict.
        """
        props = set(parameters.get("properties", {}))
        sig = inspect.signature(fn)
        params = {p for p in sig.parameters if p != "self"}
        missing = props - params
        if missing:
            raise ValueError(f"tool {name!r}: schema declares {sorted(missing)}, function does not accept them")
        for req in parameters.get("required", []):
            if req not in props:
                raise ValueError(f"tool {name!r}: {req!r} is required but not declared in properties")
        for p_name, p in sig.parameters.items():
            if p_name in ("self",) or p_name in props:
                continue
            if p.default is inspect.Parameter.empty and p.kind not in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                raise ValueError(
                    f"tool {name!r}: parameter {p_name!r} has no default and is not in the schema, "
                    "so the model can never supply it"
                )

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def specs(self) -> list[ToolSpec]:
        return [self._tools[n] for n in self.names()]

    def call(self, name: str, arguments: dict[str, Any] | None = None) -> ToolResult:
        spec = self._tools.get(name)
        if spec is None:
            return ToolResult(name=name, ok=False, error=f"unknown tool: {name}")

        args = dict(arguments or {})
        missing = [r for r in spec.required() if r not in args]
        if missing:
            # Returned as a result, not raised. The agent loop can feed a clear
            # error back to the model and let it retry, which recovers far more
            # often than an exception that kills the turn.
            return ToolResult(
                name=name, ok=False, error=f"missing required arguments: {sorted(missing)}"
            )

        unknown = [k for k in args if k not in spec.parameters.get("properties", {})]
        for k in unknown:
            args.pop(k)

        t0 = time.perf_counter()
        try:
            content = spec.fn(**args)
            return ToolResult(
                name=name, ok=True, content=content, latency_ms=(time.perf_counter() - t0) * 1000
            )
        except Exception as exc:  # noqa: BLE001 - surfaced to the model verbatim
            return ToolResult(
                name=name,
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

    def to_openai_tools(self) -> list[dict[str, Any]]:
        """Projection for NIM's OpenAI-compatible function-calling payload."""
        return [
            {
                "type": "function",
                "function": {
                    "name": s.name,
                    "description": s.description,
                    "parameters": s.parameters,
                },
            }
            for s in self.specs()
        ]

    def to_mcp_tools(self) -> list[dict[str, Any]]:
        """Projection for MCP's `tools/list`."""
        return [
            {"name": s.name, "description": s.description, "inputSchema": s.parameters}
            for s in self.specs()
        ]


registry = ToolRegistry()
