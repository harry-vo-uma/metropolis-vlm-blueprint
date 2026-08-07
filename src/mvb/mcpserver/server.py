"""MCP server over the same tool registry.

The point of this file is how little is in it. Because tools are declared once in
`tools/registry.py`, exposing them over MCP is a projection plus a transport --
there is no second copy of the schemas to keep in sync, and a tool added to the
registry is reachable from an MCP client with no edit here.

Two transports:

* `serve_stdio()` uses the official `mcp` package when installed. That is the
  path an MCP client will actually take.
* `handle_message()` is a dependency-free JSON-RPC handler for the same methods.
  It exists so the MCP surface is *testable* without installing an SDK, and so
  the protocol shape is legible to someone reading the repo rather than hidden
  behind a decorator.
"""

from __future__ import annotations

import json
from typing import Any

from ..observability.trace import get_tracer
from ..tools import registry

PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {"name": "metropolis-vlm-blueprint", "version": "0.3.0"}


def _ok(msg_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def _err(msg_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}


def handle_message(message: dict[str, Any]) -> dict[str, Any] | None:
    """Handle one JSON-RPC message. Returns None for notifications."""
    method = message.get("method")
    msg_id = message.get("id")
    params = message.get("params") or {}

    if method == "initialize":
        return _ok(
            msg_id,
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": SERVER_INFO,
            },
        )

    if method in ("notifications/initialized", "initialized"):
        return None

    if method == "ping":
        return _ok(msg_id, {})

    if method == "tools/list":
        return _ok(msg_id, {"tools": registry.to_mcp_tools()})

    if method == "tools/call":
        name = params.get("name", "")
        arguments = params.get("arguments") or {}
        with get_tracer().span("mcp.tools/call", tool=name):
            result = registry.call(name, arguments)
        if not result.ok:
            # MCP models tool errors as a *result* with isError, not as a
            # protocol error. Returning -32603 here would make the client treat
            # a recoverable bad argument as a transport fault.
            return _ok(
                msg_id,
                {"content": [{"type": "text", "text": result.error or "tool failed"}], "isError": True},
            )
        return _ok(
            msg_id,
            {
                "content": [{"type": "text", "text": json.dumps(result.content, default=str)}],
                "isError": False,
            },
        )

    return _err(msg_id, -32601, f"method not found: {method}")


def serve_stdio() -> None:  # pragma: no cover - requires the mcp package
    """Run over stdio using the official SDK when it is available."""
    try:
        import anyio
        import mcp.types as types
        from mcp.server import Server
        from mcp.server.stdio import stdio_server
    except ImportError as exc:
        raise SystemExit(
            "the `mcp` package is not installed. Install with `pip install 'mvb[mcp]'`, "
            "or use handle_message() directly for a dependency-free JSON-RPC loop."
        ) from exc

    server = Server(SERVER_INFO["name"])

    @server.list_tools()
    async def _list() -> list[types.Tool]:
        return [types.Tool(**spec) for spec in registry.to_mcp_tools()]

    @server.call_tool()
    async def _call(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
        result = registry.call(name, arguments)
        payload = result.content if result.ok else {"error": result.error}
        return [types.TextContent(type="text", text=json.dumps(payload, default=str))]

    async def _run() -> None:
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())

    anyio.run(_run)


def client_config(python: str = "python", cwd: str = ".") -> dict[str, Any]:
    """The stanza a developer pastes into an MCP client config.

    Shipped as code rather than as a snippet in the README so it cannot drift
    from the actual entry point.
    """
    return {
        "mcpServers": {
            "metropolis-vlm-blueprint": {
                "command": python,
                "args": ["-m", "mvb.mcpserver"],
                "cwd": cwd,
                "env": {"MVB_FORCE_MOCK": "1"},
            }
        }
    }
