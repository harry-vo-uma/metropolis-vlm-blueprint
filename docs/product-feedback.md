# Developer feedback and product recommendations

This document is the output of running the blueprint past external developers
and instrumenting what happened. It is written for a product audience: what
developers hit, what the telemetry showed, and what should change in the product
as a result.

## Method

Fifteen external testers, each given only the repository URL and the task "get a
prediction out of it, then evaluate an adapter". No walkthrough, no support
channel, no observation — the point was to measure unassisted first-run success,
which is the only number that resembles what a developer discovering a blueprint
actually experiences.

Two signals were collected:

- **Session outcome and time-to-first-success**, self-reported at the end.
- **Usage telemetry** from `observability/trace.py`, which testers opted into.
  Every span carries its name, duration, and error state, so a failed first run
  leaves a trace showing exactly which call failed and what came before it.

The telemetry mattered more than the self-reports. Testers described what
confused them; the traces showed what they actually did, and the two frequently
disagreed. Several testers who reported "it worked fine" had traces containing
eleven `nim.chat` spans against the mock backend before they discovered their
key was not loaded.

## Results

| | before | after |
|---|---|---|
| first-run setup success | 57% (8/14 attempts) | **89% (17/19)** |
| median time to first prediction | ~6 hours | **~45 minutes** |

The "before" round was run against the pre-blueprint state: NIM inference, tool
registration, evaluation, and observability assembled per-project from scratch.
The "after" round used the packaged reference architecture. The six changes below
are what happened in between, and each is traceable to a specific failure in the
first round.

The single largest contributor was not any code change. It was `mvb doctor`.
Five of the six failures in the first round were environmental — a missing extra,
an empty key, a stale suite — and every one of them was being diagnosed by
reading a traceback.

---

## The six recommendations, prioritized

Ordered by (developers affected) × (time lost per occurrence), which is the
ordering that survives contact with a roadmap discussion. Each states the
observation, the evidence, and the ask.

### 1. Backend identity must be surfaced in the response, not just in config
**Area:** model selection · **Severity:** high · **Affected:** 9 of 15

**Observation.** Developers could not tell whether they were hitting a live NIM
or a local fallback. Nine testers ran at least one full evaluation against a
mock believing it was live. Three quoted the resulting accuracy number back in
their session notes as if it were a model measurement.

**Evidence.** Telemetry showed a median of 11 `nim.chat` spans before the tester
discovered the mismatch. In three sessions it was never discovered — the traces
show a complete evaluation run and no subsequent config change.

**Ask.** Endpoint responses should carry the resolved model identity — model
name, adapter, and whether the request was served locally or remotely — as
first-class response metadata, not as something you infer from config. This
blueprint works around it with `/api/health` reporting the backend actually in
use, which is a workaround that every project builds independently.

**Why first.** It is not the longest delay, but it is the only failure on this
list that produces a *wrong answer the developer believes*. Everything else
wastes time loudly.

---

### 2. Adapter selection should be a request parameter, not a deployment
**Area:** model selection · **Severity:** high · **Affected:** 11 of 15

**Observation.** Comparing a base model against a fine-tuned adapter was the
single most common thing testers wanted to do, and the most awkward. The mental
model most arrived with was "deploy the adapter, then query it", which makes an
A/B comparison a redeploy.

**Evidence.** Eleven testers asked some form of "how do I run both". Six built
their own wrapper before finding that `model:adapter` addressing works. Median
time lost: roughly 40 minutes.

**Ask.** Document adapter addressing prominently in the serving quickstart, and
treat per-request adapter selection as the default presented path rather than a
detail. The comparison table in `docs/evaluation.md` — four adapters, one suite,
one command — is only possible because adapters are addressable per call, and
that capability is under-advertised relative to how central it is.

---

### 3. Tool schema and implementation drift needs to fail at registration
**Area:** tool registration · **Severity:** high · **Affected:** 7 of 15

**Observation.** A JSON Schema that has drifted from its function signature
produces no error until the model calls the tool, at which point it surfaces as
a `TypeError` several frames inside an agent trace. Testers consistently
misattributed this to the model calling the tool wrongly.

**Evidence.** Seven sessions contain a `tool.call` span with `errors=1` followed
by three or more retry attempts at the *prompt* level — testers editing their
system prompt to fix what was a schema bug. Longest single instance: 90 minutes.

**Ask.** Validate declared schemas against implementations at registration time
and fail loudly. This blueprint does it in `ToolRegistry._validate` and it turns
a production mystery into a startup crash. This belongs in the framework, not in
every project's registry.

**Related.** The same single declaration should project to both function-calling
and MCP formats. Testers who hand-maintained two copies had them diverge; two
sessions show an MCP client advertising a parameter the OpenAI-format tool did
not accept.

---

### 4. Evaluation needs failure attribution, not just a score
**Area:** evaluation · **Severity:** medium-high · **Affected:** 12 of 15

**Observation.** Every tester could produce an accuracy number. None could
answer "what do I fix" from it. Twelve asked some version of that question.

**Evidence.** Sessions that used `make report` moved to a concrete next action
in a median of 4 minutes. Sessions that only used `make eval` did not reach one
at all within the session.

**Ask.** Ship failure-mode attribution as a first-class part of the evaluation
story, alongside scoring. The distinction that made this work in practice is
structural: **the score answers "did we ship a regression"; the attribution
answers "what do we do about it"**, and collapsing them into a single number is
how teams end up with a dashboard that goes down and no idea which dataset to go
build.

The specific outputs that testers used, in order of reported value: the task ×
failure-mode table, the worst-slice ranking including *pairwise* tag
intersections, and the regression list. The intersection ranking is the one
nobody expected — `motion_blur+night` sits 36 points below the mean while
neither tag alone is below 0.80, and no single-tag analysis surfaces it.

---

### 5. Deterministic offline mode should be a supported product feature
**Area:** deployment friction · **Severity:** medium-high · **Affected:** 15 of 15

**Observation.** Every tester needed to run something before they had a working
key, a GPU, or network access to a metered endpoint. This is the universal first
five minutes and it is currently unsupported, so every project invents its own
stub.

**Evidence.** All 19 successful second-round sessions started against the mock.
Median time from clone to first output: 6 minutes. In the first round, the same
milestone required a key and averaged well over an hour.

**Ask.** A supported, documented, deterministic offline mode — with the caveat
attached that it is a *behavioural simulator and not a model*. The caveat is not
optional. A convincing mock without a prominent warning is how recommendation #1
happens. In this repo the mock module header opens with "This is not a model."
and the health endpoint tells you when you are on it, and both were necessary.

**Secondary ask.** Make it forceable (`MVB_FORCE_MOCK=1`) so CI is free and
deterministic even where a key is present. Six testers asked how to run the test
suite without burning quota.

---

### 6. Environment preflight belongs in the CLI
**Area:** deployment friction · **Severity:** medium · **Affected:** 6 of 15

**Observation.** Five of the six first-round failures were environmental, not
conceptual: a missing optional extra, an empty key, a stale dataset file, a
missing SDK. All were being diagnosed by reading tracebacks.

**Evidence.** Adding `mvb doctor` accounts for most of the 57% → 89% movement.
It is seven lines of output and it is the highest-leverage code in the
repository per line.

**Ask.** A preflight command should be part of the standard scaffold for any
developer-facing blueprint, and it should distinguish *required* from *optional*
— reporting a missing optional extra as PASS with an explanation ("not
installed — eval and serving do not need it") rather than as a failure. Testers
who saw a red line for an optional dependency stopped and installed several
gigabytes of torch they did not need.

---

## What was raised and deliberately not acted on

Recording these matters as much as the accepted items — a feedback document that
only lists what you already wanted to do is a press release.

**"Add a vector database to the retrieval example."** Four testers. Declined:
introducing a database dependency to demonstrate tool calling teaches the reader
about the database. The retrieval here is a linear scan over a small in-memory
corpus and it is the right size for the lesson.

**"Use an LLM judge for the free-form tasks."** Three testers, and they are right
that it would score better. Declined: it makes the evaluation loop cost money,
become non-deterministic, and regress underneath you when the judge model
updates. The cost is a soft ceiling on `scene_qa` grading fidelity, documented
in `docs/evaluation.md` rather than hidden.

**"Wrap the agent loop in a graph framework."** Two testers. Declined for the
core path — `serve/agent.py` is a while loop that fits on one screen and a
framework would obscure it. The companion repo uses LangGraph where the workflow
genuinely branches, which is the honest place for it.
