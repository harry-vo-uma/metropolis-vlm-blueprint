"""The registry's job is to fail early. These tests check that it does."""

from __future__ import annotations

import pytest

from mvb.tools import registry
from mvb.tools.registry import ToolRegistry


def _schema(**props: dict) -> dict:
    return {"type": "object", "properties": props, "required": list(props)}


def test_schema_declaring_an_unknown_parameter_fails_at_registration() -> None:
    """The whole point: drift is a startup crash, not a TypeError inside a trace."""
    reg = ToolRegistry()
    with pytest.raises(ValueError):

        @reg.register(
            name="bad",
            description="schema and signature disagree",
            parameters=_schema(cam_id={"type": "string"}),
        )
        def tool(camera_id: str) -> dict:
            return {"camera_id": camera_id}


def test_a_valid_declaration_registers_cleanly() -> None:
    reg = ToolRegistry()

    @reg.register(
        name="good",
        description="schema and signature agree",
        parameters=_schema(camera_id={"type": "string"}),
    )
    def tool(camera_id: str) -> dict:
        return {"camera_id": camera_id}

    assert reg.names() == ["good"]
    assert reg.call("good", {"camera_id": "aisle-03"}).ok


def test_duplicate_registration_is_rejected() -> None:
    reg = ToolRegistry()
    params = _schema(camera_id={"type": "string"})

    @reg.register(name="dup", description="first", parameters=params)
    def first(camera_id: str) -> dict:
        return {}

    with pytest.raises(ValueError):

        @reg.register(name="dup", description="second", parameters=params)
        def second(camera_id: str) -> dict:
            return {}


def test_missing_required_argument_is_a_result_not_an_exception() -> None:
    """The agent feeds tool errors back to the model; raising would end the run."""
    result = registry.call("count_objects", {})
    assert result.ok is False
    assert result.error


def test_unknown_tool_is_reported_not_raised() -> None:
    result = registry.call("no_such_tool", {})
    assert result.ok is False
    assert "unknown tool" in (result.error or "")


def test_both_projections_cover_the_same_tools() -> None:
    """One declaration, two views. If these diverge, an MCP client and the model
    disagree about what exists -- which is exactly the bug this prevents."""
    mcp_names = {t["name"] for t in registry.to_mcp_tools()}
    openai_names = {t["function"]["name"] for t in registry.to_openai_tools()}
    assert mcp_names == openai_names == set(registry.names())


def test_builtin_tools_are_registered() -> None:
    assert set(registry.names()) >= {
        "count_objects",
        "describe_scene",
        "get_frame",
        "search_events",
    }
