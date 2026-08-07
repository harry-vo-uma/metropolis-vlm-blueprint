"""MCP protocol shape and agent-loop guardrails.

These are the two surfaces where a wrong shape is worse than a wrong answer: an
MCP client that misreads a tool error as a transport fault gives up, and an
agent whose truncation is invisible returns a partial answer that reads complete.
"""

from __future__ import annotations

from mvb.mcpserver.server import PROTOCOL_VERSION, handle_message
from mvb.serve.agent import run_agent
from mvb.tools import registry


def test_initialize_advertises_the_protocol_version() -> None:
    reply = handle_message({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    assert reply is not None
    assert reply["result"]["protocolVersion"] == PROTOCOL_VERSION
    assert reply["result"]["serverInfo"]["name"]


def test_initialized_notification_gets_no_reply() -> None:
    assert handle_message({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None


def test_tools_list_matches_the_registry() -> None:
    reply = handle_message({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    assert reply is not None
    assert {t["name"] for t in reply["result"]["tools"]} == set(registry.names())


def test_tool_failure_is_an_iserror_result_not_a_protocol_error() -> None:
    """A tool that returned an error is a *successful* RPC carrying a failure.
    Returning -32603 makes clients treat a recoverable bad argument as a
    transport fault and stop retrying."""
    reply = handle_message(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "count_objects", "arguments": {}},
        }
    )
    assert reply is not None
    assert "error" not in reply
    assert reply["result"]["isError"] is True


def test_successful_tool_call_is_not_flagged_as_an_error() -> None:
    reply = handle_message(
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {"name": "count_objects", "arguments": {"term": "forklift"}},
        }
    )
    assert reply is not None
    assert reply["result"]["isError"] is False
    assert reply["result"]["content"][0]["type"] == "text"


def test_agent_uses_a_tool_and_answers() -> None:
    run = run_agent("which cameras saw a blocked keep-clear aisle")
    assert run.answer
    assert run.tool_names()
    assert run.truncated is False


def test_a_one_step_budget_truncates_and_says_so() -> None:
    """The flag is the point. A truncated answer reads exactly like a complete
    one, so the caller cannot tell them apart without it."""
    run = run_agent("which cameras saw a blocked keep-clear aisle", max_steps=1)
    assert run.truncated is True
    assert len(run.steps) <= 2
