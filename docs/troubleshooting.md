# Troubleshooting

Ordered by how often each one came up during developer testing. Run
`mvb doctor` first — it checks most of what follows in one command.

## "It runs but I don't think it's using the model"

By far the most common report, and usually correct.

```bash
curl -s localhost:8000/api/health | python -m json.tool
```

If `"backend": "mock-vlm"`, you are on the mock. Three things cause that:

1. **No `NVIDIA_API_KEY`.** `cp .env.example .env` and fill it in.
2. **`MVB_FORCE_MOCK=1` is still exported.** The Makefile sets it, and it
   survives in a shell where you previously ran `make`. `unset MVB_FORCE_MOCK`.
3. **The key is set but empty.** `NIMSettings.enabled` requires a non-empty
   string; `export NVIDIA_API_KEY=` sets it to empty, which is not a key.

The mock is convincing on purpose. It produces plausible text with realistic
latency and a believable accuracy curve. That is what makes it useful for
developing the harness and dangerous for evaluating a model — which is why
health reports the backend in use rather than the one configured.

## `suite not found at eval/datasets/suite.jsonl`

```bash
make data
```

The suite is generated, not committed as a fixture, so that the curation report
is something you watch happen rather than take on faith.

## `mvb ask` says "Unable to answer within the step budget", tools used: none

The model never emitted a parseable tool action. Check, in order:

- **Live NIM, small model.** Some models will not follow the action format from
  the system prompt alone. Inspect the raw steps: `curl` the `/api/ask` endpoint
  and read `steps[].text`.
- **Empty registry.** `mvb tools` should list four tools. If it lists none,
  `mvb.tools.builtin` was not imported — registration happens at import time.
- **Budget genuinely too low** for a multi-hop question. Raise `--max-steps`.

If the answer came back but `truncated` is true, it is partial. Do not treat it
as complete because it reads like a complete sentence.

## `ValueError` at import from `tools/registry.py`

```
ValueError: tool 'count_objects' schema declares 'cam_id' which the function does not accept
```

Working as intended. The registry validates the JSON Schema against the function
signature at import time. Fix whichever one is wrong — they must agree, because
a tool whose advertised schema has drifted from its implementation fails later,
inside an agent trace, as a `TypeError` several frames down.

## `mvb mcp` exits with a message about the MCP SDK

```bash
pip install -e ".[mcp]"
```

The stdio server needs the official SDK. The JSON-RPC handler itself
(`mcpserver.handle_message`) is dependency-free and unit-tested without it, so
the tests pass either way.

## MCP client connects but shows no tools

Print the config rather than hand-writing it:

```bash
mvb mcp --print-config
```

The `cwd` matters. The server loads the suite and tool corpus by relative path;
started from the wrong directory it comes up with an empty corpus and no error.

## `ModuleNotFoundError: No module named 'torch'`

Training extras are not installed by default — they are several gigabytes and
nothing except `train()` needs them.

```bash
pip install -e ".[train]"
```

If you hit this from an import rather than from `train()`, that is a bug: the
torch import is deliberately inside the function so the package stays usable on
a laptop.

## `DeprecationWarning` turns into a test failure

`pyproject.toml` sets `filterwarnings = ["error::DeprecationWarning"]`.
Deliberate. A deprecation warning in a dependency is a scheduled outage, and the
only reliable way to act on one is to be unable to ignore it. Fix the call site;
if it is genuinely third-party and unfixable, add a targeted `ignore` with a
comment naming the dependency and the version that will fix it.

## Accuracy numbers do not match the README

Expected if you changed a seed, a size, or a grader. The suite is regenerated
from `--n 1900 --size 1500 --seed 1337`; any of those changes the composition.
`make data && make report` reprints everything with your parameters.

If you did not change anything and the numbers still differ, check that
`MVB_FORCE_MOCK=1` — with a live key the numbers *should* differ, because then
they mean something.

## Rate limits (429) against live NIM

`NIMBackend._post` retries 429 and 5xx with exponential backoff and raises
immediately on everything else. If you are still hitting the ceiling, run the
eval with a smaller `--limit`, or lower concurrency. Retrying a 400 would just
produce four identical failures and a confusing latency graph, so it does not.

## Port 8000 already in use

```bash
mvb serve --port 8001
```

## Everything is slow on the mock

The mock simulates latency (p50 ≈ 133 ms) so that traces and p95 numbers look
like something. `MVB_TRACE_ENABLED=0` removes the tracing overhead but not the
simulated latency — that is intentional, because a demo where every call returns
in 0.1 ms teaches the wrong thing about where time goes.

## Still stuck

Open an issue with the output of `mvb doctor` and `curl -s
localhost:8000/api/health`. Those two answer most of what anyone would ask first.
