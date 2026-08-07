# Architecture

The blueprint is five layers, each of which can be replaced without touching the
others. That is the whole point: a reference architecture whose parts are welded
together is a demo, not a starting point.

```
                        ┌──────────────────────────────────────┐
  frames, prompts  ───► │  schemas.py — the cross-stage contract │
                        └───────────────┬──────────────────────┘
                                        │  Example / Prediction / Grade
             ┌──────────────────────────┼──────────────────────────┐
             │                          │                          │
      ┌──────▼──────┐          ┌────────▼────────┐        ┌────────▼────────┐
      │  data/      │          │  nim/           │        │  tools/         │
      │  synth      │          │  client  mock   │        │  registry       │
      │  curate     │          └────────┬────────┘        │  builtin        │
      └──────┬──────┘                   │                 └────────┬────────┘
             │                          │                          │
             │                 ┌────────▼────────┐        ┌────────▼────────┐
             └────────────────►│  evalsuite/     │        │  serve/ agent   │
                               │  graders suite  │        │  serve/ app     │
                               └────────┬────────┘        │  mcpserver/     │
                                        │                 └────────┬────────┘
                               ┌────────▼────────┐                 │
                               │  train/         │        ┌────────▼────────┐
                               │  sft            │        │ observability/  │
                               │  failure_analysis│◄──────┤ trace           │
                               └─────────────────┘        └─────────────────┘
```

## Layer 1 — schemas

`src/mvb/schemas.py` holds every type that crosses a stage boundary: `Example`,
`Prediction`, `Grade`, `EvalReport`, `ToolCall`, `ToolResult`. Nothing else in
the package defines a shape that another module has to know about.

Two decisions here pay for themselves repeatedly.

**Example ids are content hashes.** `Example.make_id` hashes
`task|prompt|target`. Regenerating the pool with a different seed produces
different examples but the *same* example always gets the same id, so a grade
from last week can be joined against a suite built today.

**An empty target is a validation error, not a zero.** A reference string that is
blank scores 0.0 under every grader in the file, silently, forever. Rejecting it
at construction turns a permanently-failing eval row into an immediate,
locatable error.

## Layer 2 — the model backend

`nim/client.py` speaks the OpenAI-compatible chat completions API that NIM
exposes, either against `integrate.api.nvidia.com/v1` or a locally hosted NIM
container. Adapters are addressed as `model:adapter` so switching from `base` to
`lora-v3` is a string change, not a deployment.

Retries are deliberately narrow: 429 and 5xx get exponential backoff, everything
else raises immediately. Retrying a 400 is how a malformed request turns into
four malformed requests and a confusing latency graph.

`nim/mock.py` is the reason this repo runs on a laptop. **It is not a model.** It
is a behavioural simulator: given an adapter, a task, and the tags on an
example, it computes a competence probability and then either emits a correct
answer or corrupts one in a task-appropriate way.

```python
logit = _TASK_LOGIT[adapter][ex.task]
for tag in ex.tags:
    logit -= _TAG_PENALTY[tag] * (1.0 - _TAG_RECOVERY[adapter].get(tag, 0.0))
logit -= 2.6 * (ex.difficulty - 0.5)
return _sigmoid(logit)
```

The corruptions are real text, not labels. A `FORMAT_VIOLATION` is a correct JSON
object wrapped in a code fence and an apology. A `SPATIAL_INVERSION` is the same
sentence with `left` swapped for `right`. This matters because the graders then
have to *actually detect* those failures from the string — which is how two real
grader bugs surfaced during development (see `docs/evaluation.md`).

What the mock legitimately buys you: a deterministic, free, offline CI signal,
and a harness that has been exercised against every failure mode before it ever
sees a real model. What it does not buy you: any evidence about a real model's
capability. Point `NVIDIA_API_KEY` at a live endpoint before you believe a
number.

## Layer 3 — data and evaluation

`data/synth.py` generates the pool; `data/curate.py` turns a pool into a suite.
Curation is four ordered stages — deduplicate, balance, drop-provenance, trim —
and each reports what it removed. `docs/evaluation.md` covers the reasoning.

`evalsuite/graders.py` keeps **scoring** and **failure attribution** separate,
which is the single most consequential structural decision in the repo. The
score answers "did we ship a regression". The attribution answers "what do we do
about it". Collapse them into one number and you get a dashboard that goes down
with no indication of which dataset to go build.

## Layer 4 — tools, agent, and MCP

A tool is declared once, in `tools/builtin.py`, as a Python function plus a JSON
Schema. `tools/registry.py` validates at *import time* that the schema and the
signature agree:

```python
registry.register(ToolSpec(name="count_objects", fn=count_objects, schema={...}))
# ValueError at import if the schema declares a parameter the function
# does not accept, or omits a required one.
```

That check exists because a tool whose advertised schema has drifted from its
implementation fails at model-call time, inside an agent trace, as a confusing
`TypeError` several layers down. Catching it at import turns a production
mystery into a startup crash.

From that one declaration the registry projects two views: `to_openai_tools()`
for function calling and `to_mcp_tools()` for MCP `tools/list`. They are pure
projections, so the agent and an external MCP client are guaranteed to see the
same tools with the same schemas.

`serve/agent.py` runs the loop with three guardrails: a step budget, suppression
of repeated identical calls, and tool errors fed back to the model as content
rather than raised. The last one is the interesting one — a tool that fails
should give the model a chance to try something else, not kill the request.
`AgentRun.truncated` is set when the budget runs out so a partial answer is
never mistaken for a confident one.

`mcpserver/server.py` has a dependency-free `handle_message` implementing the
JSON-RPC surface (`initialize`, `ping`, `tools/list`, `tools/call`) that can be
unit-tested without the SDK, plus `serve_stdio` that uses the official SDK when
it is installed. Tool failures come back as `isError: true` results, not
`-32603` protocol errors — a tool that returned an error is a successful RPC
carrying a failure, and conflating the two breaks clients.

## Layer 5 — observability

`observability/trace.py` is a thread-local span stack with a JSONL sink and an
OTLP-shaped export. Spans nest (`agent.run` → `agent.step` → `nim.chat`,
`tool.call`) and roll up:

```
agent.run           n=1   p50=  10.91ms  errors=0
agent.step          n=2   p50=  10.74ms  errors=0
nim.chat            n=2   p50=   0.12ms  errors=0
tool.call           n=1   p50=  10.68ms  errors=0
```

Tracing is on by default and costs a dictionary append per span. `_NullSpan`
makes `MVB_TRACE_ENABLED=0` genuinely free rather than merely quiet, which is
why the test suite sets it.

## What is deliberately not here

No vector database — the retrieval in the agent tools is a linear scan over a
small in-memory corpus, because introducing a database dependency to demonstrate
tool calling teaches the reader about the database.

No orchestration framework in the core path. `serve/agent.py` is a while loop.
LangGraph is a good fit for real branching workflows and the companion repo
(`agentic-warehouse-vision`) uses it; here it would obscure a loop that fits on
one screen.

No training run in CI. `train/sft.py` builds the dataset, the labels, the LoRA
config, and the run manifest, and lazily imports torch only inside `train()`.
Everything up to the point where a GPU is required is testable without one.
