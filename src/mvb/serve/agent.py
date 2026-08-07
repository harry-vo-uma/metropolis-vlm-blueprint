"""The tool-calling loop.

Small on purpose. The interesting decisions are not in the orchestration -- they
are in the three guardrails, each of which exists because the naive loop failed
in a specific way during the developer-testing rounds:

* **A step budget.** Without one, a model that cannot answer will call
  `search_events` with slightly different phrasing until the request times out.
  A budget converts an infinite loop into a partial answer, which is far more
  useful to the caller.
* **Repeat-call detection.** Identical (tool, arguments) pairs are answered from
  the previous result and the model is told so. This is what actually breaks the
  loop above; the budget is only the backstop.
* **Errors go back to the model as content.** A failed tool call is a message,
  not an exception. Most failures are a missing argument the model can fix on
  the next step if it is simply shown the error.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ..config import get_settings
from ..nim.client import get_backend
from ..observability.trace import get_tracer
from ..schemas import ToolCall, ToolResult
from ..tools import registry

SYSTEM = """You are a warehouse video analyst. Answer the user's question using the tools available.

Rules:
- Call one tool at a time. Reply with a single JSON object: {"tool": "<name>", "arguments": {...}}
- When you have enough information, reply with {"answer": "<your answer>"} instead.
- Never invent record ids. Get them from search_events first.
"""


@dataclass
class AgentStep:
    kind: str  # "tool" | "answer" | "error"
    call: ToolCall | None = None
    result: ToolResult | None = None
    text: str = ""


@dataclass
class AgentRun:
    question: str
    answer: str = ""
    steps: list[AgentStep] = field(default_factory=list)
    truncated: bool = False
    """True when the step budget ran out. Surfaced to the caller rather than
    hidden, because a truncated answer and a complete one deserve different
    trust."""

    trace: dict[str, Any] = field(default_factory=dict)

    def tool_names(self) -> list[str]:
        return [s.call.name for s in self.steps if s.call]


def _parse_action(raw: str) -> dict[str, Any] | None:
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None


def run_agent(question: str, max_steps: int = 6, adapter: str | None = None) -> AgentRun:
    backend = get_backend()
    tracer = get_tracer()
    adapter = adapter or get_settings().train.adapter_name
    run = AgentRun(question=question)

    transcript: list[str] = [SYSTEM, f"Question: {question}"]
    seen: dict[str, ToolResult] = {}

    with tracer.span("agent.run", question=question, adapter=adapter):
        for step_i in range(max_steps):
            with tracer.span("agent.step", step=step_i) as span:
                prompt = "\n\n".join(transcript) + "\n\nAvailable tools: " + ", ".join(
                    registry.names()
                )
                with tracer.span("nim.chat", adapter=adapter):
                    raw = backend.chat(prompt)

                action = _parse_action(raw)
                if action is None:
                    run.steps.append(AgentStep(kind="error", text=f"unparseable action: {raw[:200]}"))
                    transcript.append(
                        "Your last reply was not valid JSON. Reply with a single JSON object."
                    )
                    continue

                if "answer" in action:
                    run.answer = str(action["answer"])
                    run.steps.append(AgentStep(kind="answer", text=run.answer))
                    span.attributes["terminal"] = True
                    break

                call = ToolCall(name=str(action.get("tool", "")), arguments=action.get("arguments") or {})
                key = f"{call.name}|{json.dumps(call.arguments, sort_keys=True)}"

                if key in seen:
                    result = seen[key]
                    transcript.append(
                        f"You already called {call.name} with those arguments. "
                        f"Result was: {json.dumps(result.content)[:400]}. "
                        "Use it or answer the question."
                    )
                    run.steps.append(AgentStep(kind="tool", call=call, result=result))
                    continue

                with tracer.span("tool.call", tool=call.name, arguments=call.arguments) as tspan:
                    result = registry.call(call.name, call.arguments)
                    tspan.attributes["ok"] = result.ok
                    if not result.ok:
                        tspan.status = "error"
                        tspan.error = result.error

                seen[key] = result
                run.steps.append(AgentStep(kind="tool", call=call, result=result))

                if result.ok:
                    transcript.append(
                        f"Tool {call.name} returned: {json.dumps(result.content)[:800]}"
                    )
                else:
                    transcript.append(f"Tool {call.name} failed: {result.error}. Fix the call or answer.")
        else:
            run.truncated = True
            if not run.answer:
                run.answer = "Unable to answer within the step budget. Partial evidence: " + ", ".join(
                    run.tool_names()
                )

    run.trace = tracer.summary()
    return run
