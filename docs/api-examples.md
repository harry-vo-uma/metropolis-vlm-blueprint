# API examples

Start the service with `make serve`. Interactive docs are at
`http://localhost:8000/docs`.

All examples below run against the mock backend. Nothing changes when you point
at a live NIM except the contents of the responses.

## Health — start here

```bash
curl -s localhost:8000/api/health | python -m json.tool
```

```json
{
  "status": "ok",
  "backend": "mock-vlm",
  "live_nim": false,
  "vlm_model": "nvidia/vila",
  "adapter": "lora-v3",
  "tools": ["count_objects", "describe_scene", "get_frame", "search_events"],
  "suite_loaded": 1500,
  "tracing": true
}
```

`backend` is the backend **actually in use**, not the one that was configured.
The most common support question during developer testing was some form of "is
it really hitting the model", and the honest answer was often no — a key was set
but empty, or `MVB_FORCE_MOCK` was still exported from a previous shell. Read
this field before you believe any other response on this page.

## Single prediction

```bash
curl -s localhost:8000/api/predict \
  -H 'content-type: application/json' \
  -d '{
        "prompt": "How many forklifts are visible in this frame?",
        "frame_uri": "s3://frames/aisle-03/000412.jpg",
        "task": "scene_qa"
      }' | python -m json.tool
```

```json
{
  "example_id": "adhoc",
  "adapter": "lora-v3",
  "raw": "There are 3 forklifts visible in aisle-03.",
  "latency_ms": 134.2,
  "model": "nvidia/vila:lora-v3"
}
```

`task` must be one of `scene_qa`, `attribute_extraction`, `spatial_relation`,
`anomaly_judgement`, `temporal_ordering`. An unknown task returns a 400 listing
the valid values rather than a 500 — the task string selects the system prompt,
so getting it wrong silently would produce a plausible answer built on the wrong
instructions.

Override the adapter per request:

```bash
curl -s localhost:8000/api/predict \
  -H 'content-type: application/json' \
  -d '{"prompt": "...", "task": "scene_qa", "adapter": "base"}'
```

## Structured extraction

```bash
curl -s localhost:8000/api/predict \
  -H 'content-type: application/json' \
  -d '{
        "prompt": "Extract the attributes of the forklift in this frame as JSON with keys colour, state, zone.",
        "task": "attribute_extraction"
      }' | python -m json.tool
```

```json
{
  "example_id": "adhoc",
  "adapter": "lora-v3",
  "raw": "{\"colour\": \"yellow\", \"state\": \"moving\", \"zone\": \"aisle-03\"}",
  "latency_ms": 131.8,
  "model": "nvidia/vila:lora-v3"
}
```

The service returns the raw string. Parsing is the caller's job, and
`mvb.evalsuite.graders.extract_json` is exported for it — it returns
`(value, was_clean)` so you can distinguish a clean object from one you had to
dig out of a code fence. If you strip fences silently, you will conclude your
model has no formatting problem.

## Agent loop

```bash
curl -s localhost:8000/api/ask \
  -H 'content-type: application/json' \
  -d '{"question": "which cameras saw a blocked keep-clear aisle"}' | python -m json.tool
```

```json
{
  "question": "which cameras saw a blocked keep-clear aisle",
  "answer": "Evidence found on 4 camera(s): aisle-03, aisle-07, pack-01, pack-03.",
  "truncated": false,
  "tools_used": ["search_events"],
  "steps": [
    {"kind": "tool", "tool": "search_events",
     "arguments": {"query": "blocked keep-clear aisle"}, "ok": true, "error": null, "text": null},
    {"kind": "answer", "tool": null, "arguments": null, "ok": null, "error": null,
     "text": "Evidence found on 4 camera(s): aisle-03, aisle-07, pack-01, pack-03."}
  ],
  "trace": "..."
}
```

**Always check `truncated`.** It is set when the step budget ran out, which means
the answer is partial. A partial answer is not distinguishable from a complete
one by reading the prose, which is the entire reason the flag exists.

Raise the budget for multi-hop questions:

```bash
-d '{"question": "...", "max_steps": 10}'
```

The budget is capped at 12 by the request schema. An agent that can loop 200
times will eventually loop 200 times, and it will do it against a metered
endpoint.

## Tools

```bash
curl -s localhost:8000/api/tools | python -m json.tool
```

Returns both projections of the registry — MCP `tools/list` format and OpenAI
function-calling format — from the same single declaration. They cannot drift.

Call one directly, without the model in the loop:

```bash
curl -s localhost:8000/api/tools/call \
  -H 'content-type: application/json' \
  -d '{"name": "count_objects", "arguments": {"camera_id": "aisle-03", "object_type": "forklift"}}' \
  | python -m json.tool
```

```json
{"ok": true, "name": "count_objects", "content": {"camera_id": "aisle-03", "object_type": "forklift", "count": 3}, "error": null, "latency_ms": 0.4}
```

This endpoint is the fastest way to debug an agent that is calling a tool and
getting nothing useful back — take the model out of the loop and see what the
tool actually returns.

A missing required argument comes back as `ok: false` with an error string, not
an exception:

```json
{"ok": false, "name": "count_objects", "content": null, "error": "missing required argument: camera_id", "latency_ms": 0.1}
```

That shape is deliberate. The agent feeds tool errors back to the model as
content so it can correct itself; raising would end the request instead. Only an
unknown tool name produces a 404, because that is a client bug rather than
something a model can recover from.

## Evaluation over HTTP

```bash
curl -s 'localhost:8000/api/eval?adapter=lora-v3&limit=200' | python -m json.tool
```

Returns a full `EvalReport`: accuracy, mean score, p50/p95 latency, `by_task`,
`by_tag`, `by_provenance`, and `failure_counts`. `limit` defaults to 200 because
this endpoint runs synchronously; use `eval/run_eval.py` for the full 1,500.

## Traces

```bash
curl -s localhost:8000/api/traces | python -m json.tool
```

```json
{
  "summary": {
    "agent.run":  {"n": 1, "p50_ms": 10.91, "errors": 0},
    "agent.step": {"n": 2, "p50_ms": 10.74, "errors": 0},
    "nim.chat":   {"n": 2, "p50_ms": 0.12,  "errors": 0},
    "tool.call":  {"n": 1, "p50_ms": 10.68, "errors": 0}
  },
  "spans": [ ... last 200 ... ]
}
```

Spans nest, so `agent.run` duration includes its children. The roll-up is what
tells you whether a slow request was slow in the model or slow in a tool, which
is not answerable from a single end-to-end latency number.

Set `MVB_TRACE_DIR` to persist spans as JSONL; `Tracer.export_otlp()` produces an
OTLP-shaped payload for a collector.

## Python client

```python
from mvb.nim.client import get_backend
from mvb.schemas import Example, TaskKind, Split, Provenance

backend = get_backend()
ex = Example(
    id="demo-1",
    task=TaskKind.SPATIAL_RELATION,
    split=Split.TEST,
    provenance=Provenance.HUMAN_LABELLED,
    prompt="Where is the blue pallet relative to the loading dock?",
    target="The blue pallet is left of the loading dock.",
)
pred = backend.predict(ex, adapter="lora-v3")
print(pred.raw)
```

`templates/minimal_predict.py` is this, runnable. `templates/custom_tool.py`,
`templates/custom_grader.py`, and `templates/eval_your_data.py` cover the three
other things people asked how to do most often.
