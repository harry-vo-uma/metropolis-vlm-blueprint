"""Tool registry for the agent loop."""
from . import builtin  # noqa: F401,E402  (import registers the built-in tools)
from .registry import ToolRegistry, ToolSpec, registry  # noqa: F401
