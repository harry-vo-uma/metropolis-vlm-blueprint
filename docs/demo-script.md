# Demo script

`make demo` runs six beats in about forty seconds, offline, with no GPU. This is
the narration to go with it.

The structure is deliberate: it opens with the data, not the model. A demo that
opens with an accuracy number has already asked the audience to trust the
measurement before showing them what was measured.

**Before you start:** `make data` (once), and confirm `mvb doctor` is clean.
Nothing in the demo touches the network, so it cannot fail on conference wifi.

---

## [1] The suite — 1,500 examples, five tasks, five slices

> "Everything you're about to see is measured against this. Five tasks, five
> visual conditions, and their intersections. It's generated and curated in
> front of you by `make data` — it isn't a fixture I'm asking you to trust."

Point at the curation counters: 1,900 in, 1,500 out, and each stage says what it
dropped. If someone asks about the 275 dropped for balance — the generator
naturally produces 38% `scene_qa`, which is also the task every adapter is
already good at. An unbalanced suite lets a model coast to a high headline
number without improving on anything hard.

## [2] Base checkpoint versus post-trained adapter

> "70% to 88%. Which is the number on the slide, and it's also the least
> interesting thing on this screen."

Say the number and move past it quickly. The next four beats are the actual
argument. If you linger here the questions you get will be about the number.

## [3] Where the base model actually failed

> "144 of the base model's failures are format violations. It knew the answer
> and wrapped it in a code fence. That's not a data-volume problem — collecting
> another thousand examples wouldn't fix it. It's an SFT problem, and the
> harness says so, in those words, without me interpreting it."

This is the beat that earns the rest. The distinction being demonstrated is that
**scoring and attribution are separate**: the score told us the run was worse,
the attribution told us what to do about it.

If you have time for one aside, make it this: a lot of harnesses strip code
fences before parsing, and then report that their model has no formatting
problem. Here `extract_json` returns `(value, was_clean)` and a correct object
buried in prose is capped below threshold — because it still breaks the caller.

## [4] What to build next, derived from the grades

> "`motion_blur+night` is 36 points below the mean. Neither tag alone is below
> 0.80. If you'd only sliced by single tags you'd have concluded both were fine.
> That intersection is what the next collection round should target — and this
> came out of the run, not out of me staring at it."

The output ends in imperative sentences: *collect paired examples that differ
only in the relation term*, *mine hard negatives here*, *this provenance is being
memorised, consider re-splitting*. That is the loop closing — grades in,
next dataset out.

## [5] The same model behind a tool-calling agent

> "Same backend, same adapter, now with four tools. It picks one, calls it, and
> answers from what came back."

```
Q: which cameras saw a blocked keep-clear aisle
A: Evidence found on 4 camera(s): aisle-03, aisle-07, pack-01, pack-03.
tools used: ['search_events']
```

Worth surfacing: those four tools are declared once and projected into both
OpenAI function-calling format and MCP `tools/list`. They can't drift apart,
because there's only one declaration. And the registry validates each schema
against its function signature at import — a mismatch is a startup crash, not a
`TypeError` buried in an agent trace at 2am.

## [6] Every step of that was traced

```
agent.run           n=1   p50=  10.91ms  errors=0
agent.step          n=2   p50=  10.74ms  errors=0
nim.chat            n=2   p50=   0.12ms  errors=0
tool.call           n=1   p50=  10.68ms  errors=0
```

> "Spans nest, so when a request is slow you can see whether it was slow in the
> model or slow in a tool. A single end-to-end latency number can't answer that."

## Closing

> "One command builds the data, one evaluates four adapters and tells you what to
> fix, one serves it, one exposes the tools over MCP. All of it runs offline on a
> laptop. Set an API key and the mock disappears — no code changes."

---

## Handling the two questions you will get

**"Is that a real model?"** No, and say so immediately and without hedging. The
default backend is a behavioural simulator. What it demonstrates is that the
*pipeline* — curation, grading, attribution, slice analysis, hard-negative
mining, regression detection — behaves correctly and produces actionable output,
against text that genuinely exhibits each failure mode. It says nothing about
any model's capability. Set `NVIDIA_API_KEY` and everything reruns against a
live NIM.

Volunteering this before you are asked is much better than being caught by it.
It is also the reason `/api/health` reports the backend actually in use rather
than the one configured.

**"Why not use an LLM as a judge?"** It would score free-form answers better.
It would also make the evaluation loop cost money, become non-deterministic, and
be able to regress underneath you when the judge model updates. The price of
that choice is a soft ceiling on `scene_qa` grading fidelity, and it is written
down in `docs/evaluation.md` rather than hidden.

## If you have five more minutes

Show the regression list. 108 examples were correct under `base` and wrong under
`lora-v3` — 7.2% of the suite moving backwards inside a run that gained 17.7
points. It is invisible in every aggregate, and it is what a ship review should
actually be looking at.
