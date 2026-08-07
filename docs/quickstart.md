# Quickstart

Target: a working evaluation run and a served endpoint in under ten minutes, on
a laptop, with no GPU and no API key.

Everything below runs against the mock backend by default. The mock is a
behavioural simulator, not a model — see `docs/architecture.md` for what that
means and where it stops being an acceptable stand-in.

## 1. Install

```bash
git clone https://github.com/harry-vo-uma/metropolis-vlm-blueprint
cd metropolis-vlm-blueprint
python -m venv .venv && source .venv/bin/activate
make install
```

`make install` installs the base package only. The `train` extra pulls torch,
transformers, peft, datasets, and accelerate, which is several gigabytes and is
not needed to evaluate or serve. Install it when you actually want to run a
LoRA job:

```bash
pip install -e ".[train]"
```

## 2. Check the environment before anything else

```bash
mvb doctor
```

```
[PASS] python >= 3.10             3.10.12
[PASS] evaluation suite present   eval/datasets/suite.jsonl
[PASS] backend                    mock-vlm (mock -- set NVIDIA_API_KEY for live)
[PASS] fastapi installed          ok
[PASS] mcp sdk (optional)         not installed -- stdio server unavailable, JSON-RPC handler still works
[PASS] training extra (optional)  not installed -- eval and serving do not need it
[PASS] tools registered           count_objects, describe_scene, get_frame, search_events
```

This command exists because most first-run failures in the developer-testing
rounds were environmental — a missing extra, a stale suite file, an API key that
was set but empty — and were being diagnosed by reading tracebacks. `doctor`
answers those questions in one line each and points at
`docs/troubleshooting.md` when something fails.

## 3. Build the evaluation suite

```bash
make data
```

```
input                 1900
dropped_duplicate       47
dropped_balance        275
dropped_trim            78
kept                  1500
train 1037 | val 167 | test 296
```

The curation report is printed, not hidden. Every number in it is a decision you
can inspect and disagree with: `docs/evaluation.md` explains what each stage
drops and why.

## 4. Run the evaluation

```bash
make eval
```

```
adapter       accuracy  mean score   p50 ms   p95 ms
------------------------------------------------------------------------------
base             0.701       0.802      145      190
sft-v1           0.791       0.847      134      164
lora-v2          0.829       0.883      134      163
lora-v3          0.878       0.918      133      164
```

For the full breakdown — per task, per slice, failure modes, regressions, and
the recommended next actions — run the report:

```bash
make report
```

## 5. Ask a question through the agent

```bash
mvb ask "which cameras saw a blocked keep-clear aisle"
```

```
Q: which cameras saw a blocked keep-clear aisle
A: Evidence found on 4 camera(s): aisle-03, aisle-07, pack-01, pack-03.

tools used: search_events
```

## 6. Serve it

```bash
make serve
```

Then in another shell:

```bash
curl -s localhost:8000/api/health | python -m json.tool
curl -s localhost:8000/api/ask -H 'content-type: application/json' \
  -d '{"question": "how many forklifts are in aisle-03"}' | python -m json.tool
```

`docs/api-examples.md` has the full set of endpoints with request and response
bodies.

## 7. Expose the same tools over MCP

```bash
mvb mcp --print-config
```

Paste the printed stanza into your MCP client config. The tools registered for
the agent and the tools advertised over MCP come from a single declaration, so
they cannot drift apart — see `docs/architecture.md`.

## Going live

Set an API key and the mock disappears:

```bash
cp .env.example .env
# uncomment and fill NVIDIA_API_KEY
mvb doctor   # backend line should now read: nim (live NIM)
```

No code changes. `MVB_FORCE_MOCK=1` forces the mock back on even with a key
present, which is what the test suite and the Makefile do so that CI is
deterministic and free.

## The five-minute tour

If you would rather watch than type:

```bash
make demo
```

`scripts/demo.py` walks through prediction, grading, failure attribution, the
adapter comparison, the agent loop, and the trace roll-up in six beats.
`docs/demo-script.md` is the narration to go with it.
